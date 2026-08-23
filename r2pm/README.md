# r2SMT — r2pm package

This directory ships an [r2pm](https://github.com/radareorg/radare2-pm)
manifest so radare2 users can install the `r2smt` CLI without
manually invoking `cargo`.

## Install (local clone)

```bash
# from this checkout (use its package database entry)
R2PM_DBDIR="$PWD/r2pm" r2pm -ci r2smt
```

`r2pm -ci` installs the matching prebuilt release for supported Linux
and macOS hosts. If that download is unavailable, it falls back to a
source build. The manifest:

1. Reads the package version from the workspace `Cargo.toml`.
2. Downloads the matching release archive, or runs
   `cargo build --release -p r2smt-cli` as a fallback.
3. Installs `r2smt` into r2pm's `BINDIR` (the binary the
   `!r2smt ...` shell-out relies on).
4. Copies `r2smt.r2` into r2pm's `PLUGDIR` so the cursor-aware macros
   are discoverable.

## Use inside radare2

Load the macros and your binary in one shot:

```bash
r2 -i "$(r2pm -H R2PM_PLUGDIR)/r2smt.r2" sample.exe
```

From the r2 prompt, position the cursor on a conditional branch and
run one of the helpers:

| Macro                 | Behaviour                                                                                            |
|-----------------------|------------------------------------------------------------------------------------------------------|
| `$r2smt-at`           | One-line verdict for the branch under the cursor (terse; quick check).                               |
| `$r2smt-at-v`         | Same, plus the solver-simplified formula and slice evidence (`--explain`).                           |
| `$r2smt-at-ctx`       | Verdict plus the owning function's decompiled pseudocode (`--with-decompiler`).                      |
| `$r2smt-at-patch`     | Verdict, apply the high-confidence patch, then `oo; pd 8` so the patched bytes show in the view.     |
| `$r2smt-solve`        | Classify the branch under the cursor (`opaque_predicate` / `dead_branch` / …) and print the report.  |
| `$r2smt-solve-deep`   | Same, but with `--deep-analysis` (runs `aaaa` for harder samples).                                   |
| `$r2smt-ctx`          | `$r2smt-solve` plus r2ghidra / r2dec pseudocode context for the finding.                             |
| `$r2smt-sweep`        | One-line verdict for **every** branch in the current function (`--function $FB`).                    |
| `$r2smt-annotate`     | Generate `CCu` commands in r2SMT and interpret them in the current r2 session.                       |
| `$r2smt-patch`        | Full-file backup, apply the byte patch, persist a manifest, then refresh the disasm view.            |
| `$r2smt-patch-dry`    | Show the planned patch without writing anything.                                                     |
| `$r2smt-rollback`     | Reverse the most recent patch using the sibling manifest, then refresh the disasm view.              |

After a patch, the `-patch` / `-rollback` macros run `oo` (reopen the
now-modified file) and `pd 8` so the live disassembly reflects the
change immediately — the closest CLI/macro analog to IDA's
`plan_and_wait` post-patch reanalysis. The decompiler macros are
best-effort: with no r2ghidra / r2dec plugin present they simply omit
the context block.

## Uninstall

```bash
r2pm -u r2smt
```

## Notes

- Prebuilt installs require `curl` and `tar`. Source fallback additionally
  requires Rust, Cargo, CMake, and a C++ toolchain; Z3 is vendored.
- The macros shell out to the installed `r2smt` binary; they do not
  run analysis inside the r2 process. `$r2smt-annotate` is the exception
  for write-back: radare2 interprets the generated `CCu` commands in the
  parent session, so `CC` shows the comments immediately.
