# Architecture

r2SMT is a layered Rust workspace. Domain crates depend on shared types and
ports; adapters depend on those ports; `r2smt-cli` is the composition root that
wires radare2, lifters, solvers, reports, and patching together.

## Runtime pipeline

```text
binary
  -> r2smt-r2pipe: radare2 JSON, ESIL, optional r2ghidra P-code
  -> r2smt-ir: normalized Program and symbolic expressions
  -> r2smt-slicer: branch collection and bounded backward slices
  -> r2smt-esil / r2smt-pcode / mnemonic handlers: typed IR
  -> r2smt-ssa: SSA conversion and simplification
  -> r2smt-smt: Z3, CVC5, or Bitwuzla query
  -> r2smt-core: verdict, confidence, and finding classification
  -> r2smt-report / r2smt-patch: analyst output or explicit write-back
```

The normal CLI path currently owns orchestration in `r2smt-cli::support`.
`r2smt-core::Analyzer` is still a configuration holder, not the public
end-to-end application API. This is intentional current-state documentation,
not a claim that the library API has already reached CLI parity.

## Crate responsibilities

| Layer | Crates | Responsibility |
|---|---|---|
| Shared domain | `r2smt-common`, `r2smt-ir` | Errors, architecture data, program model, expressions, and adapter ports |
| Ingestion | `r2smt-r2pipe` | Own the radare2 session and normalize `ij`/`aflj`/`agfj`/`aoj`/`pdgsd` data |
| Semantics | `r2smt-esil`, `r2smt-pcode`, `r2smt-slicer`, `r2smt-ssa` | Lift instructions, slice dependencies, and build solver-ready SSA |
| Proof | `r2smt-solver-port`, `r2smt-smt`, `r2smt-z3fp` | Solver contract and concrete SMT backends |
| Decision | `r2smt-core`, `r2smt-difflift`, `r2smt-taint` | Classification, confidence, differential checks, and taint |
| Effects | `r2smt-report`, `r2smt-patch`, `r2smt-explore` | Reports, explicit patching, and fenced unsound witness exploration |
| Composition | `r2smt-cli` | CLI parsing, dependency wiring, orchestration, and user-facing output |

## Dependency rules

- Domain crates do not depend on radare2, CLI, or concrete solver adapters.
- `r2smt-core` consumes domain types and ports; it does not open processes or
  files.
- `r2smt-r2pipe` is the only normal-analysis adapter that owns a live radare2
  session.
- `r2smt-cli` is allowed to know every concrete adapter and is the only layer
  that writes normal command output.
- `r2smt-explore` is optional and cannot upgrade an authoritative verdict.

See [SOUNDNESS.md](SOUNDNESS.md) for the proof boundary and
[COMPATIBILITY.md](COMPATIBILITY.md) for runtime capability gates.
