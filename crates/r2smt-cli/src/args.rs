//! CLI argument parsing — clap `Cli`/`Command` definitions and the
//! `--solver` / `--min-confidence` value-enum wrappers.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use r2smt_core::Confidence;

/// Parsed top-level CLI arguments.
#[derive(Debug, Parser)]
#[command(
    name = "r2smt",
    version,
    about = "SMT-assisted deobfuscation for radare2",
    long_about = None,
)]
pub(crate) struct Cli {
    /// Increase log verbosity (`-v` for `debug`, `-vv` for `trace`).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub(crate) verbose: u8,

    /// Run radare2's deep analysis pass (`aaaa`) instead of the
    /// default `aaa`. Slower but catches more functions — useful when
    /// a binary defeats the standard heuristics (heavily obfuscated
    /// CFG, missing relocations, …).
    #[arg(long, global = true)]
    pub(crate) deep_analysis: bool,

    /// IR source feeding the slicer / SMT pipeline. `esil` (default)
    /// uses radare2 ESIL; `pcode` / `auto` additionally attach
    /// r2ghidra SLEIGH P-code (decompiler-grade) and prefer it per
    /// instruction, falling back to ESIL where the P-code lifter
    /// declines. Requires the r2ghidra plugin for any effect.
    #[arg(long, value_enum, default_value_t = IrSourceArg::Esil, global = true)]
    pub(crate) ir: IrSourceArg,

    /// Send every instruction whose ESIL writes flags to this
    /// project's per-mnemonic handler instead of radare2's ESIL.
    ///
    /// The ESIL rung is on by default: since 2026-08-07 on x86, when the
    /// differential harness reached zero disagreements there, and since
    /// 2026-08-10 on both ARM ISAs, when its own corpus verdict diff
    /// added 7 actionable findings and lost none. This is the escape
    /// hatch: it restores the flags-declining lowering so the two can be
    /// A/B'd on one binary without rebuilding. See
    /// `LiftCtx::allows_esil_flags`.
    #[arg(long, global = true)]
    pub(crate) no_esil_flags: bool,

    /// Subcommand to dispatch.
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// IR source selector exposed via `--ir`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub(crate) enum IrSourceArg {
    /// radare2 ESIL only (historical default — no behavior change).
    #[default]
    Esil,
    /// Attach r2ghidra P-code and prefer it; fall back to ESIL.
    Pcode,
    /// Alias of `pcode` (P-code preferred, ESIL fallback).
    Auto,
}

impl IrSourceArg {
    /// Whether the r2ghidra adapter should attach P-code at load.
    pub(crate) fn wants_pcode(self) -> bool {
        matches!(self, Self::Pcode | Self::Auto)
    }
}

/// Top-level subcommand the user invoked.
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Print the build version and exit.
    Version,

    /// Report runtime dependency versions and compatibility gates.
    Doctor,

    /// Open a binary with radare2 and emit the normalized program.
    Analyze {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Dump the full Program model as JSON.
        #[arg(long)]
        dump_program: bool,

        /// Write the JSON output to this path instead of stdout.
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
    },

    /// Collect every conditional branch candidate in a binary.
    Branches {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Restrict the collection to the function starting at this
        /// address (decimal or `0x`-prefixed hex).
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Emit the candidates as JSON to this path instead of stdout
        /// summary.
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
    },

    /// Apply r2SMT findings as live radare2 `CCu` comments through the
    /// same r2 session that ran the analysis (Phase 9). Optionally save
    /// the annotated session as an r2 project.
    Annotate {
        /// Path to the binary to annotate.
        file: PathBuf,

        /// Annotate only the branch at this address.
        #[arg(long, value_name = "ADDR")]
        at: Option<String>,

        /// Restrict to branches inside the function starting at this
        /// address.
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Per-branch solver budget in milliseconds. Default: 500.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u32>,

        /// Per-branch solver budget in Z3 resource units instead of
        /// wall-clock time. Defaults to 2 000 000 since 2026-08-07 —
        /// pass `0` to unset it and get the old wall-clock-only
        /// behaviour. A wall-clock budget makes
        /// the verdict depend on host load, so the same binary over the
        /// same input can classify a branch `real_branch` on an idle
        /// machine and `suspicious_but_unknown` on a busy one; this
        /// bound counts work, not seconds, and so is reproducible.
        /// Prefer it when diffing verdicts across two builds.
        #[arg(long, value_name = "UNITS")]
        rlimit: Option<u32>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat the unresolved roots of a truncated slice as free
        /// symbolic inputs and run the SMT pipeline anyway. Sound
        /// (only widens `AlwaysX` to `BothPossible`, never
        /// fabricates a verdict) but downgrades the resulting
        /// confidence to `medium`.
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// SMT backend to consult. `z3` (default) uses the in-process
        /// Z3 binding; `cvc5` / `bitwuzla` shell out to a system
        /// `cvc5` / `bitwuzla` binary via SMT-LIB2. Useful as an
        /// independent cross-check.
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum number of basic blocks the slicer may traverse
        /// per branch. `1` (default) keeps the walk inside the
        /// branch's own block; raising this enables multi-block
        /// slicing through unique-predecessor chains.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Minimum confidence to act on.
        #[arg(long, value_enum, default_value_t = ConfidenceArg::High)]
        min_confidence: ConfidenceArg,

        /// Compute the annotations but do not write them to the r2
        /// session.
        #[arg(long)]
        dry_run: bool,

        /// Save the annotated r2 session as a project under this name
        /// (`Ps <name>`). Implies the comments were applied.
        #[arg(long, value_name = "NAME")]
        save_project: Option<String>,
    },

    /// Apply or roll back conservative byte-level patches derived from
    /// r2SMT findings (Phase 10). Always takes a full-file backup
    /// before writing and records every change in a JSON manifest.
    Patch {
        /// Path to the binary to patch.
        file: PathBuf,

        /// Solve and plan only at this address.
        #[arg(long, value_name = "ADDR")]
        at: Option<String>,

        /// Restrict to branches inside the function starting at this
        /// address.
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Per-branch solver budget in milliseconds. Default: 500.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u32>,

        /// Per-branch solver budget in Z3 resource units instead of
        /// wall-clock time. Defaults to 2 000 000 since 2026-08-07 —
        /// pass `0` to unset it and get the old wall-clock-only
        /// behaviour. A wall-clock budget makes
        /// the verdict depend on host load, so the same binary over the
        /// same input can classify a branch `real_branch` on an idle
        /// machine and `suspicious_but_unknown` on a busy one; this
        /// bound counts work, not seconds, and so is reproducible.
        /// Prefer it when diffing verdicts across two builds.
        #[arg(long, value_name = "UNITS")]
        rlimit: Option<u32>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat the unresolved roots of a truncated slice as free
        /// symbolic inputs and run the SMT pipeline anyway. Sound
        /// (only widens `AlwaysX` to `BothPossible`, never
        /// fabricates a verdict) but downgrades the resulting
        /// confidence to `medium`.
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// SMT backend to consult. `z3` (default) uses the in-process
        /// Z3 binding; `cvc5` / `bitwuzla` shell out to a system
        /// `cvc5` / `bitwuzla` binary via SMT-LIB2. Useful as an
        /// independent cross-check.
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum number of basic blocks the slicer may traverse
        /// per branch. `1` (default) keeps the walk inside the
        /// branch's own block; raising this enables multi-block
        /// slicing through unique-predecessor chains.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Minimum confidence required to apply a patch.
        #[arg(long, value_enum, default_value_t = ConfidenceArg::High)]
        min_confidence: ConfidenceArg,

        /// Actually write bytes to the file. Without this flag, the
        /// command only prints the plan (full dry-run).
        #[arg(long)]
        apply: bool,

        /// Override the default backup path
        /// (`<binary>.r2smt.bak`).
        #[arg(long, value_name = "PATH")]
        backup: Option<PathBuf>,

        /// Override the default manifest path
        /// (`<binary>.r2smt.manifest.json`).
        #[arg(long, value_name = "PATH")]
        manifest: Option<PathBuf>,

        /// Reverse a previous patch run using the manifest at
        /// `--manifest` (defaults to `<binary>.r2smt.manifest.json`).
        /// Implies write access to the binary.
        #[arg(long)]
        rollback: bool,
    },

    /// Solve every branch with Z3 and emit classified findings
    /// (`opaque_predicate`, `dead_branch`, `constant_condition`, …).
    Solve {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Solve only the branch at this address.
        #[arg(long, value_name = "ADDR")]
        at: Option<String>,

        /// Restrict to branches inside the function starting at this
        /// address.
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Per-branch solver budget in milliseconds. Default: 500.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u32>,

        /// Per-branch solver budget in Z3 resource units instead of
        /// wall-clock time. Defaults to 2 000 000 since 2026-08-07 —
        /// pass `0` to unset it and get the old wall-clock-only
        /// behaviour. A wall-clock budget makes
        /// the verdict depend on host load, so the same binary over the
        /// same input can classify a branch `real_branch` on an idle
        /// machine and `suspicious_but_unknown` on a busy one; this
        /// bound counts work, not seconds, and so is reproducible.
        /// Prefer it when diffing verdicts across two builds.
        #[arg(long, value_name = "UNITS")]
        rlimit: Option<u32>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat the unresolved roots of a truncated slice as free
        /// symbolic inputs and run the SMT pipeline anyway. Sound
        /// (only widens `AlwaysX` to `BothPossible`, never
        /// fabricates a verdict) but downgrades the resulting
        /// confidence to `medium`.
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// SMT backend to consult. `z3` (default) uses the in-process
        /// Z3 binding; `cvc5` / `bitwuzla` shell out to a system
        /// `cvc5` / `bitwuzla` binary via SMT-LIB2. Useful as an
        /// independent cross-check.
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum number of basic blocks the slicer may traverse
        /// per branch. `1` (default) keeps the walk inside the
        /// branch's own block; raising this enables multi-block
        /// slicing through unique-predecessor chains.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Minimum confidence to include in the findings list.
        #[arg(long, value_enum, default_value_t = ConfidenceArg::Medium)]
        min_confidence: ConfidenceArg,

        /// Also include `real_branch` findings in the output.
        #[arg(long)]
        include_real: bool,

        /// Also include `suspicious_but_unknown` findings.
        #[arg(long)]
        include_suspicious: bool,

        /// Treat a CFG join (≥2 predecessors) as a sound free-input
        /// boundary instead of abandoning the slice. Scoped to joins
        /// only; sound (widens `AlwaysX` to `BothPossible`, never
        /// fabricates) but downgrades confidence.
        #[arg(long)]
        allow_join_merge: bool,

        /// Emit the findings as JSON to this path (full Report shape).
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,

        /// Emit a Markdown report to this path.
        #[arg(long, value_name = "PATH")]
        markdown: Option<PathBuf>,

        /// Emit a radare2 annotation script to this path.
        #[arg(long, value_name = "PATH")]
        r2_script: Option<PathBuf>,

        /// Attach decompiler pseudocode (r2ghidra / r2dec) to each
        /// finding as analyst context. Best-effort: silently omitted
        /// when no decompiler plugin is available.
        #[arg(long)]
        with_decompiler: bool,

        /// Cross-check every instruction's P-code / ESIL / per-mnemonic
        /// lowerings for semantic equivalence and report the
        /// lifter-agreement rate. A proven disagreement is an
        /// engine-integrity defect (emitted as a `lifter_disagreement`
        /// finding). Off by default; output is byte-identical without
        /// it.
        #[arg(long)]
        differential_lift: bool,

        /// Comparison budget for `--differential-lift`. The scan stops
        /// once this many pairs have been compared and says so, since a
        /// run that stopped early would otherwise be indistinguishable
        /// from one that covered the program. Default: 500 000, which
        /// covers the largest sample in the ISA corpora; raise it for a
        /// bigger binary rather than reading a truncated scan as a
        /// clean one.
        #[arg(long, value_name = "N")]
        max_difflift_comparisons: Option<usize>,
    },

    /// Sweep every regular file directly inside a directory, solving
    /// each sample in its own isolated radare2 process, and emit one
    /// aggregated report. Per-sample failures are recorded, never
    /// fatal. Non-recursive and deterministic (sorted-path order).
    Batch {
        /// Directory of samples to analyze (non-recursive).
        dir: PathBuf,

        /// Worker threads. Default (or `0`): one per logical CPU.
        #[arg(long, value_name = "N")]
        threads: Option<usize>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Per-branch solver budget in milliseconds. Default: 500.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u32>,

        /// Per-branch solver budget in Z3 resource units instead of
        /// wall-clock time. Defaults to 2 000 000 since 2026-08-07 —
        /// pass `0` to unset it and get the old wall-clock-only
        /// behaviour. A wall-clock budget makes
        /// the verdict depend on host load, so the same binary over the
        /// same input can classify a branch `real_branch` on an idle
        /// machine and `suspicious_but_unknown` on a busy one; this
        /// bound counts work, not seconds, and so is reproducible.
        /// Prefer it when diffing verdicts across two builds.
        #[arg(long, value_name = "UNITS")]
        rlimit: Option<u32>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat unresolved roots of a truncated slice as free
        /// symbolic inputs (sound; downgrades confidence to medium).
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// Treat a CFG join as a sound free-input boundary (scoped to
        /// joins; widens, never fabricates).
        #[arg(long)]
        allow_join_merge: bool,

        /// SMT backend to consult (`z3` default, or `cvc5`).
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum basic blocks the slicer may traverse per branch.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Emit the aggregated report as JSON to this path.
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,

        /// Emit the aggregated report as Markdown to this path.
        #[arg(long, value_name = "PATH")]
        markdown: Option<PathBuf>,

        /// Attach decompiler pseudocode (r2ghidra / r2dec) to each
        /// finding as analyst context. Best-effort.
        #[arg(long)]
        with_decompiler: bool,
    },

    /// Interactive single-branch analysis: solve the conditional at
    /// `addr` and print a one-line verdict. Designed to be driven from
    /// inside a live radare2 session via the `$r2smt-at` macro
    /// (`r2smt at "${R2_FILE}" $$`).
    At {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Address of the conditional instruction (decimal or `0x` hex).
        addr: String,

        /// After solving, apply a conservative byte patch when the
        /// verdict is actionable at `high` confidence (backup +
        /// manifest written next to the binary).
        #[arg(long)]
        patch: bool,

        /// Per-branch solver budget in milliseconds. Default: 500.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u32>,

        /// Per-branch solver budget in Z3 resource units instead of
        /// wall-clock time. Defaults to 2 000 000 since 2026-08-07 —
        /// pass `0` to unset it and get the old wall-clock-only
        /// behaviour. A wall-clock budget makes
        /// the verdict depend on host load, so the same binary over the
        /// same input can classify a branch `real_branch` on an idle
        /// machine and `suspicious_but_unknown` on a busy one; this
        /// bound counts work, not seconds, and so is reproducible.
        /// Prefer it when diffing verdicts across two builds.
        #[arg(long, value_name = "UNITS")]
        rlimit: Option<u32>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// SMT backend to consult (`z3` default, or `cvc5`).
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Print decompiler pseudocode (r2ghidra / r2dec) for the
        /// owning function after the verdict. Best-effort.
        #[arg(long)]
        with_decompiler: bool,

        /// Only the one-line verdict — suppress the solver-simplified
        /// form, evidence, and decompiled context. Ideal for sweeps.
        #[arg(long)]
        quiet: bool,

        /// Also print the solver-simplified formula and slice
        /// evidence (free inputs, IR-statement / unknown counts).
        /// (Named `--explain` to avoid clashing with the global
        /// `-v/--verbose` log-verbosity counter.)
        #[arg(long)]
        explain: bool,

        /// Treat a CFG join as a sound free-input boundary (scoped to
        /// joins; widens, never fabricates).
        #[arg(long)]
        allow_join_merge: bool,
    },

    /// Rename each lifted slice into Static Single Assignment form.
    Ssa {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Convert only the branch at this address.
        #[arg(long, value_name = "ADDR")]
        at: Option<String>,

        /// Restrict to branches inside the function starting at this
        /// address.
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat the unresolved roots of a truncated slice as free
        /// symbolic inputs and run the SMT pipeline anyway. Sound
        /// (only widens `AlwaysX` to `BothPossible`, never
        /// fabricates a verdict) but downgrades the resulting
        /// confidence to `medium`.
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// SMT backend to consult. `z3` (default) uses the in-process
        /// Z3 binding; `cvc5` / `bitwuzla` shell out to a system
        /// `cvc5` / `bitwuzla` binary via SMT-LIB2. Useful as an
        /// independent cross-check.
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum number of basic blocks the slicer may traverse
        /// per branch. `1` (default) keeps the walk inside the
        /// branch's own block; raising this enables multi-block
        /// slicing through unique-predecessor chains.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Emit the SSA-renamed slices as JSON to this path instead of
        /// the stdout summary.
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
    },

    /// Lift each branch's data-flow slice into the r2SMT IR.
    Lift {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Lift only the branch at this address.
        #[arg(long, value_name = "ADDR")]
        at: Option<String>,

        /// Restrict to branches inside the function starting at this
        /// address.
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat the unresolved roots of a truncated slice as free
        /// symbolic inputs and run the SMT pipeline anyway. Sound
        /// (only widens `AlwaysX` to `BothPossible`, never
        /// fabricates a verdict) but downgrades the resulting
        /// confidence to `medium`.
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// SMT backend to consult. `z3` (default) uses the in-process
        /// Z3 binding; `cvc5` / `bitwuzla` shell out to a system
        /// `cvc5` / `bitwuzla` binary via SMT-LIB2. Useful as an
        /// independent cross-check.
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum number of basic blocks the slicer may traverse
        /// per branch. `1` (default) keeps the walk inside the
        /// branch's own block; raising this enables multi-block
        /// slicing through unique-predecessor chains.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Emit the lifted slices as JSON to this path instead of the
        /// stdout summary.
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
    },

    /// Build a backward data-flow slice for every conditional branch
    /// (or just one, with `--at`).
    Slice {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Slice only the branch at this address.
        #[arg(long, value_name = "ADDR")]
        at: Option<String>,

        /// Restrict slicing to branches inside the function starting
        /// at this address.
        #[arg(long, value_name = "ADDR")]
        function: Option<String>,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Treat the unresolved roots of a truncated slice as free
        /// symbolic inputs and run the SMT pipeline anyway. Sound
        /// (only widens `AlwaysX` to `BothPossible`, never
        /// fabricates a verdict) but downgrades the resulting
        /// confidence to `medium`.
        #[arg(long)]
        unknowns_on_truncation: bool,

        /// SMT backend to consult. `z3` (default) uses the in-process
        /// Z3 binding; `cvc5` / `bitwuzla` shell out to a system
        /// `cvc5` / `bitwuzla` binary via SMT-LIB2. Useful as an
        /// independent cross-check.
        #[arg(long, value_enum, default_value_t = SolverArg::Z3)]
        solver: SolverArg,

        /// Maximum number of basic blocks the slicer may traverse
        /// per branch. `1` (default) keeps the walk inside the
        /// branch's own block; raising this enables multi-block
        /// slicing through unique-predecessor chains.
        #[arg(long, value_name = "N")]
        max_blocks: Option<u32>,

        /// Emit the slices as JSON to this path instead of stdout
        /// summary.
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
    },

    /// Sound may-taint analysis: report which values at `addr` derive
    /// from a seeded source. With `--concretise`, an opaque outcome
    /// (flow hidden behind an unmodelled node) falls through to the
    /// fenced UNSOUND exploration engine to search for a witness.
    Taint {
        /// Path to the binary to analyze.
        file: PathBuf,

        /// Address of the branch / instruction to analyze.
        addr: String,

        /// Source register to seed (repeatable; each a distinct taint
        /// source). If omitted, every free input is seeded as one
        /// source.
        #[arg(long = "source", value_name = "REG")]
        sources: Vec<String>,

        /// On an opaque outcome, search for a concrete witness via the
        /// exploration engine (UNSOUND, banner-stamped).
        #[arg(long)]
        concretise: bool,

        /// Maximum number of instructions per slice. Default: 32.
        #[arg(long, value_name = "N")]
        max_instructions: Option<usize>,

        /// Allow memory load / store instructions in slices.
        #[arg(long)]
        allow_memory: bool,

        /// Allow `call` instructions in slices.
        #[arg(long)]
        allow_calls: bool,

        /// Host-side wall-clock budget in ms for concretisation. Default: 30000.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u64>,

        /// Path-explosion budget for concretisation. Default: 10000.
        #[arg(long, value_name = "N")]
        max_paths: Option<u64>,
    },

    /// Search for a concrete input that reaches an address (UNSOUND
    /// exploration). Drives the fenced radius2 engine to synthesise a
    /// witness — stdin / argv / registers — that would steer execution
    /// to `addr`. A best-effort search, never a verdict: the output is
    /// stamped with a non-suppressible UNSOUND banner and must not be
    /// used for verify / patch decisions. Without the `oracle-radius2`
    /// build feature the engine is not compiled in and the command
    /// reports that it is inconclusive.
    Why {
        /// Path to the binary to explore.
        file: PathBuf,

        /// Target address to reach (decimal or `0x` hex).
        addr: String,

        /// Host-side wall-clock budget in milliseconds. Default: 30000.
        #[arg(long, value_name = "MS")]
        timeout_ms: Option<u64>,

        /// Path-explosion budget: maximum symbolic states to fork
        /// before giving up. Default: 10000.
        #[arg(long, value_name = "N")]
        max_paths: Option<u64>,
    },
}

/// Minimum-confidence threshold exposed via `--min-confidence`.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub(crate) enum ConfidenceArg {
    /// Only act on `High`-confidence findings.
    High,
    /// `Medium` and above.
    Medium,
    /// `Low` and above.
    Low,
    /// Include `Unknown` (every finding the engine emitted).
    Unknown,
}

/// SMT backend selector exposed via `--solver`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub(crate) enum SolverArg {
    /// Z3 via the in-process `z3` crate (default).
    #[default]
    Z3,
    /// CVC5 via the `cvc5` subprocess. Requires `cvc5` to be
    /// available on `$PATH`.
    Cvc5,
    /// Bitwuzla via the `bitwuzla` subprocess. Requires `bitwuzla`
    /// to be available on `$PATH`. A third independent `QF_BV`
    /// cross-check.
    Bitwuzla,
}

impl ConfidenceArg {
    /// Project the user-facing enum onto the domain-level
    /// [`Confidence`] used by `r2smt-core`.
    pub(crate) fn to_confidence(self) -> Confidence {
        match self {
            Self::High => Confidence::High,
            Self::Medium => Confidence::Medium,
            Self::Low => Confidence::Low,
            Self::Unknown => Confidence::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn test_at_parses_quiet_verbose_and_decompiler_flags() {
        let cli = Cli::try_parse_from([
            "r2smt",
            "at",
            "bin",
            "0x401000",
            "--quiet",
            "--with-decompiler",
        ])
        .expect("at args parse");
        match cli.command {
            Command::At {
                quiet,
                explain,
                with_decompiler,
                ..
            } => {
                assert!(quiet);
                assert!(!explain);
                assert!(with_decompiler);
            }
            other => panic!("expected At, got {other:?}"),
        }
    }

    #[test]
    fn test_batch_parses_threads_and_decompiler_flags() {
        let cli = Cli::try_parse_from([
            "r2smt",
            "batch",
            "samples",
            "--threads",
            "4",
            "--with-decompiler",
        ])
        .expect("batch args parse");
        match cli.command {
            Command::Batch {
                threads,
                with_decompiler,
                ..
            } => {
                assert_eq!(threads, Some(4));
                assert!(with_decompiler);
            }
            other => panic!("expected Batch, got {other:?}"),
        }
    }

    #[test]
    fn test_solve_parses_the_difflift_comparison_budget() {
        let cli = Cli::try_parse_from([
            "r2smt",
            "solve",
            "sample",
            "--differential-lift",
            "--max-difflift-comparisons",
            "1000",
        ])
        .expect("solve args parse");
        match cli.command {
            Command::Solve {
                differential_lift,
                max_difflift_comparisons,
                ..
            } => {
                assert!(differential_lift);
                assert_eq!(max_difflift_comparisons, Some(1000));
            }
            other => panic!("expected Solve, got {other:?}"),
        }
    }

    #[test]
    fn test_solve_leaves_the_difflift_budget_unset_by_default() {
        // Unset rather than pre-filled: the default lives at the one
        // call site that reads it, so a `None` here cannot silently
        // disagree with `DEFAULT_MAX_DIFFLIFT_COMPARISONS`.
        let cli = Cli::try_parse_from(["r2smt", "solve", "sample"]).expect("solve args parse");
        match cli.command {
            Command::Solve {
                max_difflift_comparisons,
                ..
            } => assert_eq!(max_difflift_comparisons, None),
            other => panic!("expected Solve, got {other:?}"),
        }
    }
}
