/* r2SMT radare2 core plugin -- MIT */

#define R_LOG_ORIGIN "r2smt"

#include <r_core.h>
#include <errno.h>

static RCoreHelpMessage help_msg_r2smt = {
	"Usage:", "r2smt [action] [options]", "Run r2SMT against the current file and seek",
	"r2smt", " [options]", "analyze the conditional at the current seek",
	"r2smt at", " [options]", "alias for the default single-branch analysis",
	"r2smt explain", " [options]", "single-branch verdict with formula and slice evidence",
	"r2smt ctx", " [options]", "single-branch verdict with decompiler context",
	"r2smt solve", " [options]", "full finding for the branch at the current seek",
	"r2smt solve-deep", " [options]", "solve at the current seek after r2 deep analysis",
	"r2smt sweep", " [options]", "solve every branch in the current analyzed function",
	"r2smt annotate", " [options]", "apply r2SMT comments to this r2 session",
	"r2smt patch", " [options]", "write a verified sibling .r2smt.patched file",
	"r2smt patch-dry", " [options]", "show the patch plan for the current seek",
	"r2smt rollback", " [options]", "restore an in-place patch from its manifest",
	"r2smt doctor", "", "show dependency and compatibility status",
	"r2smt version", "", "show the r2SMT CLI version",
	"", "", "Set R2SMT_CLI to override the r2smt executable path.",
	"", "", "CLI output uses file offsets; annotations are rebased to live r2 addresses.",
	NULL
};

static bool strs_eq(RStrs value, const char *expected) {
	return r_strs_equals_str (value, expected);
}

static bool append_arg(RStrBuf *command, const char *arg) {
	char *escaped = r_str_escape_sh (arg);
	if (!escaped) {
		return false;
	}
	const bool ok = r_strbuf_appendf (command, " \"%s\"", escaped);
	free (escaped);
	return ok;
}

static bool append_args(RStrBuf *command, RStrs *args, size_t from, size_t argc) {
	size_t i;
	for (i = from; i < argc; i++) {
		char *arg = r_str_ndup (args[i].a, (int)r_strs_len (args[i]));
		if (!arg || !append_arg (command, arg)) {
			free (arg);
			return false;
		}
		free (arg);
	}
	return true;
}

static char *find_cli(void) {
	char *override = r_sys_getenv ("R2SMT_CLI");
	if (R_STR_ISNOTEMPTY (override)) {
		return override;
	}
	free (override);
	return r_file_path ("r2smt");
}

static const char *current_file(RCore *core) {
	RBinFile *binfile = core && core->bin? r_bin_cur (core->bin): NULL;
	return binfile? binfile->file: NULL;
}

/* The Rust provider pins io.va=false because patch addresses must be stable
 * file offsets. Translate the live r2 address space at the plugin boundary. */
static ut64 file_offset(RCore *core, ut64 address) {
	if (core && core->io && r_config_get_b (core->config, "io.va")) {
		return r_io_v2p (core->io, address);
	}
	return address;
}

static ut64 session_address(RCore *core, ut64 physical) {
	if (!core || !r_config_get_b (core->config, "io.va")) {
		return physical;
	}
	RBinObject *object = core->bin? r_bin_cur_object (core->bin): NULL;
	RBinSection *section = object? r_bin_get_section_at (object, physical, false): NULL;
	if (section && physical >= section->paddr
			&& physical - section->paddr < section->size) {
		const ut64 section_offset = physical - section->paddr;
		if (section->vaddr <= UT64_MAX - section_offset) {
			return r_bin_get_vaddr (
				core->bin, physical, section->vaddr + section_offset);
		}
	}
	if (file_offset (core, core->addr) == physical) {
		return core->addr;
	}
	ut64 address;
	if (core->io && r_io_p2v (core->io, physical, &address)) {
		return address;
	}
	return physical;
}

/* `annotate --r2-script` emits comments plus `CCu ... @ <file-offset>` lines.
 * Validate that narrow contract, discard the informational comments (r2's
 * line runner stops on them), and rebase CCu offsets for the parent session. */
static char *rebase_annotations(RCore *core, const char *script) {
	RStrBuf *rebased = r_strbuf_new ("");
	if (!rebased) {
		return NULL;
	}
	const char *line = script;
	while (*line) {
		const char *line_end = strchr (line, '\n');
		if (!line_end) {
			line_end = line + strlen (line);
		}
		if (line_end == line) {
			line++;
			continue;
		}
		if (*line == '#') {
			line = *line_end? line_end + 1: line_end;
			continue;
		}
		if (line_end - line < 4 || strncmp (line, "CCu ", 4)) {
			r_strbuf_free (rebased);
			return NULL;
		}
		const char *marker = NULL;
		const char *cursor;
		for (cursor = line; cursor + 3 <= line_end; cursor++) {
			if (!strncmp (cursor, " @ ", 3)) {
				marker = cursor;
			}
		}
		if (!marker || marker + 3 == line_end) {
			r_strbuf_free (rebased);
			return NULL;
		}
		char *address_text = r_str_ndup (marker + 3, (int)(line_end - marker - 3));
		if (!address_text) {
			r_strbuf_free (rebased);
			return NULL;
		}
		errno = 0;
		char *tail = NULL;
		const unsigned long long value = strtoull (address_text, &tail, 0);
		const bool valid = !errno && tail != address_text && !*tail;
		free (address_text);
		if (!valid || !r_strbuf_append_n (rebased, line, marker + 3 - line)
				|| !r_strbuf_appendf (rebased, "0x%"PFMT64x"\n",
					session_address (core, (ut64)value))) {
			r_strbuf_free (rebased);
			return NULL;
		}
		line = *line_end? line_end + 1: line_end;
	}
	return r_strbuf_drain (rebased);
}

static RCmdResult command_error(RCmdContext *ctx, const char *message) {
	r_cons_printf (ctx->cons, "r2smt: %s\n", message);
	return (RCmdResult) { .status = 1 };
}

static RCmdResult run_cli(RCmdContext *ctx, RStrBuf *command, bool apply_script, bool reopen) {
	char *output = NULL;
	char *error = NULL;
	int output_len = 0;
	char *command_string = r_strbuf_drain (command);
	if (!command_string) {
		return command_error (ctx, "cannot allocate command line");
	}
	const bool ok = r_sys_cmd_str_full (
		command_string, NULL, 0, &output, &output_len, &error);
	free (command_string);

	if (R_STR_ISNOTEMPTY (error)) {
		r_cons_printf (ctx->cons, "%s%s", error, r_str_endswith (error, "\n")? "": "\n");
	}
	if (ok && apply_script && R_STR_ISNOTEMPTY (output)) {
		RCore *core = ctx->user;
		char *rebased = rebase_annotations (core, output);
		if (!rebased || !r_core_cmd_lines (core, rebased)) {
			free (rebased);
			free (output);
			free (error);
			return command_error (ctx, "invalid or unapplicable generated annotation script");
		}
		free (rebased);
	} else if (output_len > 0) {
		r_cons_write (ctx->cons, output, output_len);
	}
	free (output);
	free (error);

	if (!ok) {
		return command_error (ctx, "CLI execution failed");
	}
	if (reopen) {
		r_core_cmd0 (ctx->user, "oo");
	}
	return (RCmdResult) { 0 };
}

static RStrBuf *command_begin(void) {
	char *cli = find_cli ();
	if (!cli) {
		return NULL;
	}
	RStrBuf *command = r_strbuf_new ("");
	if (command && !append_arg (command, cli)) {
		r_strbuf_free (command);
		command = NULL;
	}
	free (cli);
	return command;
}

static RCmdResult run_without_file(
		RCmdContext *ctx, const char *subcommand, RStrs *args, size_t from, size_t argc) {
	RStrBuf *command = command_begin ();
	if (!command) {
		return command_error (ctx, "cannot find the r2smt CLI (set R2SMT_CLI or install it with r2pm)");
	}
	if (!append_arg (command, subcommand) || !append_args (command, args, from, argc)) {
		r_strbuf_free (command);
		return command_error (ctx, "cannot allocate command line");
	}
	return run_cli (ctx, command, false, false);
}

static RCmdResult run_at(
		RCmdContext *ctx, const char *subcommand, const char *fixed_option,
		RStrs *args, size_t from, size_t argc) {
	RCore *core = ctx->user;
	const char *file = current_file (core);
	if (R_STR_ISEMPTY (file)) {
		return command_error (ctx, "no binary is open");
	}
	RStrBuf *command = command_begin ();
	if (!command) {
		return command_error (ctx, "cannot find the r2smt CLI (set R2SMT_CLI or install it with r2pm)");
	}
	char address[32];
	const ut64 physical = file_offset (core, core->addr);
	if (physical == UT64_MAX) {
		r_strbuf_free (command);
		return command_error (ctx, "current address is not backed by the input file");
	}
	snprintf (address, sizeof (address), "0x%"PFMT64x,
		physical);
	if (!append_arg (command, subcommand) || !append_arg (command, file)
			|| !append_arg (command, address)
			|| (fixed_option && !append_arg (command, fixed_option))
			|| !append_args (command, args, from, argc)) {
		r_strbuf_free (command);
		return command_error (ctx, "cannot allocate command line");
	}
	return run_cli (ctx, command, false, false);
}

static RCmdResult run_selected(
		RCmdContext *ctx, const char *subcommand, const char *selector,
		ut64 address, const char *fixed_option, RStrs *args, size_t from,
		size_t argc, bool apply_script, bool reopen) {
	RCore *core = ctx->user;
	const char *file = current_file (core);
	if (R_STR_ISEMPTY (file)) {
		return command_error (ctx, "no binary is open");
	}
	RStrBuf *command = command_begin ();
	if (!command) {
		return command_error (ctx, "cannot find the r2smt CLI (set R2SMT_CLI or install it with r2pm)");
	}
	char address_string[32];
	const ut64 physical = file_offset (core, address);
	if (physical == UT64_MAX) {
		r_strbuf_free (command);
		return command_error (ctx, "selected address is not backed by the input file");
	}
	snprintf (address_string, sizeof (address_string), "0x%"PFMT64x,
		physical);
	if (!append_arg (command, subcommand) || !append_arg (command, file)
			|| (selector && (!append_arg (command, selector)
				|| !append_arg (command, address_string)))
			|| (fixed_option && !append_arg (command, fixed_option))
			|| !append_args (command, args, from, argc)) {
		r_strbuf_free (command);
		return command_error (ctx, "cannot allocate command line");
	}
	return run_cli (ctx, command, apply_script, reopen);
}

static RCmdResult r2smt_callback(RCmdContext *ctx) {
	const size_t argc = RVecRStrs_length (&ctx->args);
	RStrs *args = R_VEC_START_ITER (&ctx->args);
	if (r_cmd_ctx_help (ctx) || (argc == 1 && strs_eq (args[0], "?"))) {
		r_cons_cmd_help (ctx->cons, help_msg_r2smt);
		return (RCmdResult) { 0 };
	}

	if (!argc || args[0].a[0] == '-') {
		return run_at (ctx, "at", NULL, args, 0, argc);
	}
	if (strs_eq (args[0], "at")) {
		return run_at (ctx, "at", NULL, args, 1, argc);
	}
	if (strs_eq (args[0], "explain")) {
		return run_at (ctx, "at", "--explain", args, 1, argc);
	}
	if (strs_eq (args[0], "ctx")) {
		return run_at (ctx, "at", "--with-decompiler", args, 1, argc);
	}
	if (strs_eq (args[0], "version") || strs_eq (args[0], "doctor")) {
		char *action = r_str_ndup (args[0].a, (int)r_strs_len (args[0]));
		if (!action) {
			return command_error (ctx, "cannot allocate command line");
		}
		RCmdResult result = run_without_file (ctx, action, args, 1, argc);
		free (action);
		return result;
	}

	RCore *core = ctx->user;
	if (strs_eq (args[0], "solve")) {
		return run_selected (ctx, "solve", "--at", core->addr,
			"--include-suspicious", args, 1, argc, false, false);
	}
	if (strs_eq (args[0], "solve-deep")) {
		RStrBuf *command = command_begin ();
		const char *file = current_file (core);
		if (!command) {
			return command_error (ctx, "cannot find the r2smt CLI (set R2SMT_CLI or install it with r2pm)");
		}
		char address[32];
		const ut64 physical = file_offset (core, core->addr);
		if (physical == UT64_MAX) {
			r_strbuf_free (command);
			return command_error (ctx, "current address is not backed by the input file");
		}
		snprintf (address, sizeof (address), "0x%"PFMT64x,
			physical);
		if (R_STR_ISEMPTY (file) || !append_arg (command, "--deep-analysis")
				|| !append_arg (command, "solve") || !append_arg (command, file)
				|| !append_arg (command, "--at") || !append_arg (command, address)
				|| !append_arg (command, "--include-suspicious")
				|| !append_args (command, args, 1, argc)) {
			r_strbuf_free (command);
			return command_error (ctx, R_STR_ISEMPTY (file)? "no binary is open": "cannot allocate command line");
		}
		return run_cli (ctx, command, false, false);
	}
	if (strs_eq (args[0], "sweep")) {
		RAnalFunction *function = r_anal_get_fcn_in (
			core->anal, core->addr, R_ANAL_FCN_TYPE_ANY);
		if (!function) {
			return command_error (ctx, "the current seek is not in an analyzed function (run aaa first)");
		}
		return run_selected (ctx, "solve", "--function", function->addr,
			"--include-suspicious", args, 1, argc, false, false);
	}
	if (strs_eq (args[0], "annotate")) {
		return run_selected (ctx, "annotate", "--at", core->addr,
			"--r2-script", args, 1, argc, true, false);
	}
	if (strs_eq (args[0], "patch")) {
		return run_at (ctx, "at", "--patch", args, 1, argc);
	}
	if (strs_eq (args[0], "patch-dry")) {
		return run_selected (ctx, "patch", "--at", core->addr,
			NULL, args, 1, argc, false, false);
	}
	if (strs_eq (args[0], "rollback")) {
		return run_selected (ctx, "patch", NULL, core->addr,
			"--rollback", args, 1, argc, false, true);
	}

	r_cons_cmd_help (ctx->cons, help_msg_r2smt);
	return (RCmdResult) { .status = 2 };
}

static bool plugin_init(RCorePluginSession *session) {
	RCore *core = session->core;
	if (!core) {
		return true;
	}
	if (!r_cmd_register (core->rcmd, "r2smt", r2smt_callback, NULL)) {
		return false;
	}
	session->data = core->rcmd;
	return true;
}

static bool plugin_fini(RCorePluginSession *session) {
	if (session->data) {
		r_cmd_unregister (session->data, "r2smt");
	}
	return true;
}

RCorePlugin r_core_plugin_r2smt = {
	.meta = {
		.name = "r2smt",
		.desc = "SMT-assisted branch analysis through the r2SMT CLI",
		.author = "r2SMT contributors",
		.license = "MIT",
	},
	.init = plugin_init,
	.fini = plugin_fini,
};

#ifndef R2_PLUGIN_INCORE
R_API RLibStruct radare_plugin = {
	.type = R_LIB_TYPE_CORE,
	.data = &r_core_plugin_r2smt,
	.version = R2_VERSION
};
#endif
