# r2SMT — Software Concept Charter

> Source of truth for the design and 10-phase roadmap. Code references in
> this document are normative for the *intent*, not for verbatim syntax — in
> particular, the `z3` crate's API has changed substantially between the
> versions referenced here and the current pin (`0.20`), so the snippets
> below describe shape, not exact calls. The Phase 6 implementation will
> reconcile against the current API.

## 1. Vision

r2SMT is a Rust 2024 tool for SMT-assisted deobfuscation on top of radare2,
starting with detection and safe patching of opaque predicates.

Pipeline:

```
binary
  → radare2 / r2pipe
  → functions + basic blocks + instructions
  → branch collector
  → backward slicing
  → IR
  → SSA
  → Z3
  → decision engine
  → JSON / Markdown / r2 script / patch
```

Goal: detect conditional branches that are constant. Example:

```
mov  eax, ecx
imul eax, eax
and  eax, 1
cmp  eax, 2
jne  real_code
jmp  junk_code
```

r2SMT concludes that `(eax * eax) & 1` can never equal `2`, the `cmp` is
always false, and the `jne` is always taken.

## 2. Positioning

```
MicroSMT  →  IDA + Hex-Rays microcode + Z3
r2SMT     →  radare2 + r2pipe + Rust + SSA + Z3
```

Not a general symbolic-execution engine. A practical tool for opaque
predicate detection, dead-branch cleanup, reports, radare2 annotations, and
conservative patching for high-confidence cases.

## 3. Principles

### 3.1 CLI first

```
v0.1 CLI standalone
v0.2 r2pipe integration
v0.3 reports
v0.4 patching
v0.5 r2 script / r2pm
v1.0 plugin (optional)
```

### 3.2 Independent core

The core must not depend on radare2 directly.

```
r2pipe adapter → Program model → r2smt-core
```

### 3.3 Solve micro-slices

Yes: `cmp` / `test` / arithmetic → flags → `jcc` / `setcc` / `cmovcc`.

No: full heap / OS / external APIs / global symbolic execution.

## 4. Repository layout

```
r2SMT/
├── Cargo.toml
├── crates/
│   ├── r2smt-common/
│   ├── r2smt-ir/
│   ├── r2smt-slicer/      # Phase 3+
│   ├── r2smt-ssa/         # Phase 5
│   ├── r2smt-smt/         # Phase 6
│   ├── r2smt-r2pipe/
│   ├── r2smt-patch/       # Phase 10
│   ├── r2smt-report/      # Phase 8
│   └── r2smt-cli/
├── samples/
├── tests/
├── fuzz/
├── benches/
├── docs/
├── SPEC.md
├── README.md
└── .github/workflows/ci.yml
```

## 5. Crate responsibilities (target)

### 5.1 r2smt-core

Orchestrates analysis; coordinates collector, slicer, lifter, SSA, solver,
and decision engine; produces `Finding`s.

### 5.2 r2smt-ir

Owns `Program / Function / BasicBlock / Instruction` and the IR expression
model. Supported architectures v0: `x86`, `x86_64`.

### 5.3 r2smt-r2pipe

Opens binaries with radare2, runs analysis, extracts functions / blocks /
instructions / bytes, writes comments, applies patches. r2 commands used
initially: `aaa`, `ij`, `aflj`, `agfj`, `pdfj`, `aoj`, `p8`, `CCu`, `wa`,
`wx`.

### 5.4 r2smt-slicer

Identifies branch candidates and performs local backward slicing.
Candidates v0: `je/jz`, `jne/jnz`, `ja/jae`, `jb/jbe`, `jg/jge`, `jl/jle`,
`js/jns`, `jo/jno`, `setcc`, `cmovcc`. Limits v0:

```
max_slice_instructions = 32
max_basic_blocks       = 1
allow_memory           = false
allow_calls            = false
```

### 5.5 r2smt-ssa

Converts IR to SSA; versions registers, flags, temporaries; handles x86
register aliasing (`al/ax/eax/rax`, etc.) and the x86_64 rule that writing
`eax` zeroes the upper 32 bits of `rax`.

### 5.6 r2smt-smt

Translates SSA expressions to SMT-LIB bit-vectors via `z3`. Checks
`cond == true` and `cond == false`. Returns one of:

```
AlwaysTrue | AlwaysFalse | BothPossible | Unsound | Timeout | Unknown
```

### 5.7 r2smt-patch

Generates patch plans, validates safety, writes backups, applies only with
`--apply`. Strategies: `replace_jcc_with_jmp`, `nop_jcc`,
`replace_setcc_with_mov_const`, `replace_cmovcc_with_mov_or_nop`,
`comment_only`.

### 5.8 r2smt-report

JSON, Markdown, r2 script. (SARIF is future work.)

### 5.9 r2smt-cli

Commands target:

```
r2smt analyze sample.exe [--json out.json] [--markdown out.md] [--function 0x401000]
r2smt solve   sample.exe --at 0x401050
r2smt patch   sample.exe --apply --backup
r2smt batch   ./samples --threads 8
r2smt r2-script sample.exe --out annotations.r2
```

## 6. Phase roadmap

| Phase | Status | Deliverable |
|---|---|---|
| 0 | done | Workspace, CI green, `r2smt version`. |
| 1 | done | `r2smt analyze --dump-program`: r2pipe extraction → `Program` JSON. |
| 2 | done | `r2smt branches`: `jcc / setcc / cmovcc` collection with symbolic condition + taken / fallthrough targets. |
| 3 | done | `r2smt slice`: bounded backward data-flow slicing inside a single basic block (no memory, no calls by default; togglable via flags). |
| 4 | done | `r2smt lift`: translate slice instructions into the r2SMT IR (`Expr` / `IrStmt`) and produce the branch's symbolic condition over the flags. |
| 5 | done | `r2smt ssa`: rename every defined `Var` (`rax#0`, `rax#1`, …) and report free symbolic inputs separately. |
| 6 | done | `r2smt solve`: Z3 backend that classifies each branch as `AlwaysTrue` / `AlwaysFalse` / `BothPossible` / `Unsound` / `Timeout` / `Unknown`. |
| 7 | done | Decision engine: `Finding` with `FindingKind` (OpaquePredicate / DeadBranch / ConstantCondition / RealBranch / SuspiciousButUnknown) and `Confidence` levels. CLI filters on `--min-confidence` / `--include-real` / `--include-suspicious`. |
| 8 | done | `r2smt-report` crate: `Report::render_json / render_markdown / render_r2_script`. CLI `solve` accepts `--json`, `--markdown`, `--r2-script` outputs (combinable). Per-finding patch suggestions (`nop_jcc`, `replace_jcc_with_jmp`, `replace_setcc_with_mov_const`, `replace_cmovcc_with_mov_or_nop`, `comment_only`). |
| 9 | done | `r2smt annotate`: apply `CCu` comments live via r2pipe, with optional `Ps <name>` project save (`Annotator` port + `R2PipeProvider` adapter). |
| 10 | done | `r2smt-patch` crate + `r2smt patch` subcommand: build plan from findings, full-file backup, write via `BytePatcher` port, persist JSON manifest, optional `--rollback`. v0 strategies: `nop_jcc` and `replace_jcc_with_jmp`. |
| 11 | done | `setcc` + `cmovcc` real byte-level patching: `replace_setcc_with_mov_const` (`[REX] C6 ModR/M imm8` preserving the original ModR/M and SIB/disp); `replace_cmovcc_with_mov_or_nop` (`8B` + NOP padding for AlwaysTrue, full-instruction NOP for AlwaysFalse). |
| 12 | done | IR pretty-printer (`pretty_condition`) that substitutes SSA defs back into the branch condition, surfaced as `Finding::formula_pretty`. Z3 `.simplify()` pass before SAT/UNSAT queries. |
| 13 | done | Minimal stack memory model: `[rbp ± K]` / `[rsp ± K]` constant offsets are tracked as virtual `stk_…` variables in the slicer and lifter so store/load chains substitute precisely. Dynamic indexing still truncates. |
| 14 | done | Phase E flag-helper hard refusal: branches whose flag predicate depends on `OF` / `PF` (signed `jg/jl/…`, `jo/jno`, `jp/jpo`) downgrade verdict confidence to `Low` so users do not act on findings the lifter cannot soundly model. |
| 15 | done | Sub-register correctness fix: `xor ah, al` (and analogous `cmp` / `test` / `and` / `or` / `add` / `sub` over distinct sub-registers of the same parent) no longer collapses to a `rax OP rax` shape — flagged as `Unknown` so the verdict propagates honestly. Eliminated >7800 false-positive findings on the APT10 ANELLOADER sample. |
| 16 | done | Shellcode block finder: `BinaryProvider::load_block_at` falls back to `af @ addr` then to a `pdj -1` / `aoj` / `axtj` heuristic walk, so `--at addr` works even outside r2's analysed functions. CLI wires the fallback into `slice` / `lift` / `ssa` / `solve` / `annotate` / `patch` via `resolve_targets` + `AnalysisContext`. |
| 17 | done | Deep-analysis flag: global `--deep-analysis` switches r2 from `aaa` to `aaaa` at session spawn (`R2PipeProvider::open_with_analysis(AnalysisLevel::Deep)`). |
| 18 | done | Readable names: `NameHints` (stack slots → r2 locals via `afvj`; globals via `axtj`/`fdj`). Surfaced through `classify_finding_with_hints` and `pretty_condition_with_hints` so the report shows `var_4h[stk_rbp_-4]` instead of just the canonical slot. |
| 19 | done | r2pm wrapper: `r2pm/r2smt.mk` Makefile + `r2pm/r2smt.r2` macros (`$r2smt-solve`, `$r2smt-annotate`, `$r2smt-patch`, …) + `r2pm/README.md`. `r2pm -ci r2smt` builds release and drops macros next to r2's plugin dir. |

Subsequent: native r_core plugin (FFI to libr_core), multi-block slicing, batch mode.

## 7. Configuration (target `r2smt.toml`)

```toml
[analysis]
max_slice_instructions = 32
max_basic_blocks       = 1
allow_memory           = false
allow_calls            = false
solver_timeout_ms      = 500

[patching]
dry_run         = true
require_backup  = true
min_confidence  = "high"

[radare2]
analysis_command = "aaa"
write_comments   = true

[output]
json      = true
markdown  = false
r2_script = false
```

## 8. Workspace dependencies

Versions current at 2026-05. **The `z3` crate has had significant API
churn between 0.12 and 0.20** (thread-local context default since 0.16,
method renames in 0.17, `Context::new` private in 0.19, `bundled` →
`vendored` feature in 0.20). The Phase 6 implementation reconciles against
this API.

```
anyhow             = "1"
thiserror          = "2"
clap (derive)      = "4"
serde (derive)     = "1"
serde_json         = "1"
tracing            = "0.1"
tracing-subscriber = "0.3"
r2pipe             = "0.8"
z3                 = "0.20"
```

Deferred (declared when their phase lands):

| crate | when |
|---|---|
| `rayon` | batch mode |
| `indexmap`, `petgraph` | CFG analysis (Phase 7) |
| `toml` | config loader; **0.8 → 1.x is breaking** |
| `sha2`, `hex` | patch manifests (Phase 10) |
| `tempfile` | when a test needs it |
| `goblin` | probably never — radare2 already parses PE/ELF |

## 9. Definition of Done v1.0

- Rust edition 2024.
- Stable CLI.
- Reliable read-only analysis.
- Working Z3 backend.
- Documented JSON schema.
- Patching with backup and rollback.
- Unit + integration tests.
- Minimal sample corpus.
- Clear documentation.
- r2 script generation.
- Optional r2pm packaging.

## 10. Explicit limits

r2SMT v0 does **not** promise:

- VMProtect / Themida devirtualization.
- Full symbolic execution.
- API resolution.
- Symbolic heap.
- Complex alias analysis.
- Full control-flow-flattening recovery.
- Hex-Rays microcode as an analysis source (IDA-only, not pursued).

A **decompiler-grade IR is now implemented** (overturns the earlier
spike — see `.planning/phase4-pcode-spike.md` resolution note): the
`r2smt-pcode` crate lifts r2ghidra SLEIGH P-code (`pdgsd`, a
structured documented grammar) into the IR. Opt-in via global
`--ir {esil|pcode|auto}` (default `esil` — byte-identical). It is a
sound *strict subset* (lifts the integer/logic/compare/extend/copy/
subpiece/load-store + Z-flag set; declines anything whose lowering is
not provably sound — notably ARM `NZCV` C/V/N polarity — falling back
to ESIL, never a wrong verdict). r2ghidra / r2dec pseudocode is
*also* surfaced as analyst context (`--with-decompiler`).

r2SMT v0 **does** promise:

- Local opaque predicate detection.
- Branch simplification.
- Reproducible reports.
- Conservative patches.
