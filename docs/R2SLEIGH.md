# r2sleigh adapter

r2SMT's `r2il` command is an optional process-boundary adapter. It invokes
the separately installed `r2sleigh` executable with:

```text
r2sleigh run --action lift --format r2cmd
```

The adapter validates each JSON R2IL sidecar and its paired ESIL line, then
uses r2SMT's strict ESIL lifter. Unsupported or malformed operations fail
closed; no best-effort semantics are inferred.

## Contract

- Supported architectures: `x86`, `x86-64`, and 32-bit `arm`.
- Input: non-empty, even-length hexadecimal instruction bytes.
- Output: validated R2IL operation count plus r2SMT IR statements.
- Limits: 30-second subprocess deadline and 4 MiB combined output cap.
- Installation: `r2sleigh` must be on `PATH`; it is not linked or bundled.
- Licensing: r2SMT remains MIT; the separately distributed executable is
  LGPL-3.0-only.

Run the adapter with:

```bash
r2smt r2il --arch x86-64 --bytes 31c0
```

The crate also exposes `parse_r2cmd` for captured exports, fixtures, and
fuzzing without spawning a process. The checked-in fixture and malformed-input
tests exercise the sidecar contract offline. A real `r2sleigh` installation is
required for an end-to-end invocation; CI does not install the optional tool.

## Compatibility ceiling

The adapter currently covers only the architectures above and only the strict
ESIL subset accepted by r2SMT. AArch64, upstream R2IL format changes, and
operations whose ESIL requires unsupported memory or flag semantics are
explicitly outside the contract. Update the fixture and compatibility table
before widening that boundary.
