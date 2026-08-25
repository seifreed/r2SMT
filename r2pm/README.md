# r2SMT — r2pm package

This directory ships an [r2pm](https://github.com/radareorg/radare2-pm)
manifest for the `r2smt` CLI and its native radare2 core plugin.

## Install (local clone)

```bash
# from this checkout (use its package database entry)
R2PM_DBDIR="$PWD/r2pm" r2pm -ci r2smt
```

`r2pm -ci` installs the matching prebuilt CLI for supported Linux and macOS
hosts. If that download is unavailable, it falls back to a source build. The
small core plugin is always compiled locally so its ABI matches the installed
radare2. The manifest:

1. Reads the package version from the workspace `Cargo.toml`.
2. Downloads the matching release archive, or runs
   `cargo build --release -p r2smt-cli` as a fallback.
3. Installs `r2smt` into r2pm's `BINDIR`.
4. Builds and installs `core_r2smt` into r2pm's `PLUGDIR`, where r2 discovers
   it automatically.
5. Installs `r2smt.r2` as an optional compatibility layer for the old
   `$r2smt-*` aliases.

## Use inside radare2

Open a binary normally. The native plugin is auto-loaded:

```text
$ r2 sample.exe
[0x00401000]> aaa
[0x00401000]> r2smt explain
```

Position the cursor on a conditional branch and use one of these actions:

| Command | Behaviour |
|---|---|
| `r2smt` / `r2smt at` | One-line verdict for the branch at the current seek. |
| `r2smt explain` | Verdict plus solver-simplified formula and slice evidence. |
| `r2smt ctx` | Verdict plus best-effort r2ghidra/r2dec context. |
| `r2smt solve` | Full finding for the branch at the current seek. |
| `r2smt solve-deep` | Same after radare2's deeper `aaaa` analysis. |
| `r2smt sweep` | Solve every branch in the current analyzed function. |
| `r2smt annotate` | Generate `CCu` commands and apply them in the parent r2 session. |
| `r2smt patch` | Apply a high-confidence patch to a verified sibling `.r2smt.patched` file. |
| `r2smt patch-dry` | Show the patch plan and required input SHA-256 without writing. |
| `r2smt rollback` | Restore an in-place patch from its manifest, then reopen the file in r2. |
| `r2smt doctor` | Report dependency versions and compatibility gates. |

Additional CLI options may follow each action, for example
`r2smt explain --solver portfolio`. Set `R2SMT_CLI=/path/to/r2smt` to override
the companion executable. Analysis output uses stable file offsets because the
CLI pins `io.va=false`; live annotations are rebased to the parent r2 session's
virtual addresses by the plugin.

The old aliases remain available when explicitly requested:

```bash
r2 -i "$(r2pm -H R2PM_PLUGDIR)/r2smt.r2" sample.exe
```

They now delegate to the native command, avoiding shell expansion of r2 state.

## Uninstall

```bash
r2pm -u r2smt
```

## Notes

- Installation requires `make`, `pkg-config`, and a C compiler for the native
  bridge. Prebuilt CLI installs also require `curl` and `tar`; source fallback
  additionally requires Rust, Cargo, CMake, and a C++ toolchain.
- Analysis remains out of process and uses r2SMT's worker isolation. Only the
  small session bridge and generated annotation commands run inside r2.
