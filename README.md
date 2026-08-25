<p align="center">
  <img src="https://img.shields.io/badge/r2SMT-SMT--assisted%20deobfuscation-blue?style=for-the-badge" alt="r2SMT">
</p>

<h1 align="center">r2SMT</h1>

<p align="center">
  <strong>SMT-assisted opaque-predicate deobfuscator and symbolic-analysis toolkit for radare2</strong>
</p>

<p align="center">
  <a href="https://github.com/seifreed/r2SMT/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/seifreed/r2SMT/ci.yml?style=flat-square&logo=github&label=CI" alt="CI Status"></a>
  <a href="https://github.com/seifreed/r2SMT/releases"><img src="https://img.shields.io/github/v/release/seifreed/r2SMT?style=flat-square&logo=github&label=release" alt="Latest Release"></a>
  <img src="https://img.shields.io/badge/rust-1.95%2B-orange?style=flat-square&logo=rust&logoColor=white" alt="Rust Version">
  <img src="https://img.shields.io/badge/edition-2024-orange?style=flat-square&logo=rust&logoColor=white" alt="Rust Edition">
  <a href="https://github.com/seifreed/r2SMT/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/seifreed/r2SMT/stargazers"><img src="https://img.shields.io/github/stars/seifreed/r2SMT?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/seifreed/r2SMT/issues"><img src="https://img.shields.io/github/issues/seifreed/r2SMT?style=flat-square" alt="GitHub Issues"></a>
  <a href="https://buymeacoffee.com/seifreed"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow?style=flat-square&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee"></a>
</p>

---

## Overview

**r2SMT** is a Rust toolkit that combines **radare2** with an **SMT solver** (Z3 / CVC5 / Bitwuzla) to reason about the control flow of a binary. It asks the solver whether each conditional branch can *actually* go both ways: branches that cannot are **opaque predicates**, **dead branches**, or **constant conditions** — classic obfuscation — which r2SMT then lets you annotate or patch away.

The pipeline is `radare2 disasm → typed IR → backward slice → SSA → SMT → verdict`. It is **sound under explicit assumptions and fail-closed**: it emits a verdict only when it can prove one, and says *unknown* rather than guessing. It is also **sample-agnostic** by design — no hardcoded malware-family IOCs, so it applies to arbitrary binaries. The exact boundary is documented in [SOUNDNESS.md](SOUNDNESS.md).

### Key Features

| Feature | Description |
|---------|-------------|
| **Sound SMT verdicts** | Proves each branch's determinism with Z3; never fabricates a deobfuscation |
| **Opaque-predicate detection** | Classifies opaque / dead / constant conditions vs. genuine branches |
| **Multi-arch lifters** | x86 / x86_64, AArch64, and AArch32 (including Thumb) |
| **Differential lifting** | Cross-checks P-code ≡ ESIL ≡ per-mnemonic by SMT equivalence, so lifter bugs surface |
| **Multi-solver portfolio** | Z3 (default), CVC5, and Bitwuzla backends behind one contract |
| **Annotate & patch** | Write-back r2 comments and reversible byte patches with a rollback manifest |
| **Batch analysis** | Whole-binary or whole-directory sweeps, one isolated r2 process per sample |
| **Sample-agnostic** | A general symbolic-analysis runtime, not a per-family detector |

### Supported Outputs

```text
Findings      JSON report, human-readable Markdown, one-line CLI verdicts
Annotations   radare2 script (CCu comments) or a live r2 session write-back
Patches       Transactional output + verified rollback manifest (pre/post SHA-256)
Batch         Aggregated JSON / Markdown histogram over a directory of samples
Stages        Slice, lifted IR, and SSA dumps — exactly what the solver sees
```

---

## Why not radius2?

[radius2](https://crates.io/crates/radius2) is an excellent symbolic-execution engine for radare2 — and r2SMT can even drive it as an optional corroborating oracle — but the two answer **dual questions**:

- **radius2 finds a *witness*.** It runs execution forward and produces *one* input that reaches a target address. Existential: *"does some input get here?"*
- **r2SMT proves *determinism*.** It slices a single branch backward and asks the solver whether it can go both ways for *every* input. Universal: *"is this branch fixed no matter the input?"*

An opaque predicate **is** the second question, and forward symbolic execution isn't the natural tool for it: failing to *find* an input for the other side doesn't *prove* none exists once exploration is bounded. So r2SMT is sound and fail-closed, runs in batch over the whole binary with no setup, cross-checks three independent lifters so a lifter bug surfaces instead of silently corrupting a verdict, and patches the neutralised branches with a rollback manifest. Different jobs: r2SMT is the *prover*, radius2 the *witness-finder* it can consult.

---

## Installation

### Prebuilt Binaries

Tagged releases ship prebuilt binaries for six targets (Linux / macOS / Windows, x86_64 and ARM64) on the [Releases page](https://github.com/seifreed/r2SMT/releases). The Windows archive bundles the Z3 runtime.

### From Source

Z3 is vendored and built from source, so you need a C++ toolchain and CMake (no system `libz3` required):

```bash
# macOS:  xcode-select --install && brew install cmake
# Debian: apt-get install build-essential cmake
git clone https://github.com/seifreed/r2SMT.git
cd r2SMT
cargo build --release
./target/release/r2smt version
```

### radare2 Plugin

The r2pm package installs both the CLI and a native core plugin compiled
against the active radare2 ABI:

```bash
# From this checkout
R2PM_DBDIR="$PWD/r2pm" r2pm -ci r2smt

# Or install/symlink a development checkout directly
make user-install
# make symstall
```

The plugin is auto-loaded by r2; no `-i` script is needed:

```text
$ r2 sample
[0x00401000]> aaa
[0x00401000]> r2smt?
[0x00401000]> r2smt explain
[0x00401000]> r2smt sweep
[0x00401000]> r2smt annotate
```

It is a thin in-process bridge: file, seek, and function state come directly
from the current r2 session, while the isolated Rust CLI performs the analysis.
`r2smt patch` writes a verified sibling file; `r2smt rollback` reopens the
current file after a manifest-backed rollback. See [r2pm/README.md](r2pm/README.md)
for every action and the legacy `$r2smt-*` aliases.

### Optional Features

```bash
# Compile in the fenced radius2 witness-finder oracle (off by default)
cargo build --release -p r2smt-explore --features oracle-radius2
```

> **Runtime dependency:** `radare2` ≥ 6.2.0 must be on your `PATH` for the validated analysis path. Run `r2smt doctor` and see [COMPATIBILITY.md](COMPATIBILITY.md). The CVC5 and Bitwuzla portfolio backends are invoked as external binaries when selected with `--solver`.

---

## Quick Start

```bash
# 1. What conditional branches exist?
r2smt branches ./sample

# 2. Solve them all and print classified findings
r2smt solve ./sample

# 3. Drill into one suspicious address — one-line verdict
r2smt at ./sample 0x401234
```

`solve` prints one classified line per branch:

```text
0x00401234  opaque_predicate    AlwaysFalse   high     je   → never taken
0x004012a0  dead_branch         AlwaysTrue    high     jne  → always taken
0x00401310  real_branch         BothPossible  high     jg   (genuine)
```

---

## Usage

### Command Line Interface

```bash
# Solve and export reports for triage in one pass
r2smt solve ./sample --json findings.json --markdown findings.md --r2-script annotate.r2

# See exactly what the solver sees, stage by stage
r2smt slice ./sample --at 0x401234   # bounded backward data-flow slice
r2smt lift  ./sample --at 0x401234   # that slice lifted to IR
r2smt ssa   ./sample --at 0x401234   # IR after SSA renaming

# Annotate a live radare2 session (preview, then apply + save)
r2smt annotate ./sample --dry-run
r2smt annotate ./sample --min-confidence high --save-project triage

# Patch — always backed up, always reversible
r2smt patch ./sample                                      # plan + required input SHA-256
r2smt patch ./sample --apply --expect-sha256 <sha256>     # verified .r2smt.patched output
r2smt patch ./sample --apply --expect-sha256 <sha256> --in-place  # explicit backup + patch
r2smt patch ./sample.r2smt.patched --verify-only          # reopen and verify bytes, CFG, analysis
r2smt patch ./sample --rollback                           # restore originals from manifest

# Sweep a directory (aggregated report)
r2smt batch ./corpus --threads 8 --json corpus.json --markdown corpus.md
```

### Commands

| Command | Description |
|---------|-------------|
| `r2smt branches` | List the conditional branches in a binary |
| `r2smt doctor` | Report dependency versions and active compatibility gates |
| `r2smt solve` | Classify every branch (`json` / `markdown` / `r2-script` output) |
| `r2smt at` | One-line verdict for a single branch (r2-driven entrypoint) |
| `r2smt analyze` | Dump the normalized program model |
| `r2smt slice` / `lift` / `ssa` | Inspect the per-branch slice, IR, and SSA stages |
| `r2smt annotate` | Write findings back as r2 comments (dry-run or apply) |
| `r2smt patch` | Plan, transactionally apply, verify, or roll back byte patches |
| `r2smt batch` | Analyze every sample in a directory and aggregate the results |
| `r2smt taint` | Sound may-taint analysis: which values at an address derive from a seeded source |
| `r2smt why` | **Unsound** best-effort witness search (radius2 oracle) |

### Common Flags

| Flag | Description |
|------|-------------|
| `--at <addr>` / `--function <addr>` | Restrict analysis to one branch or one function |
| `--timeout-ms <ms>` / `--rlimit <n>` | Per-branch solver budgets (clock and resource) |
| `--min-confidence <high\|medium\|low>` | Confidence floor for annotate / patch actions |
| `--solver <z3\|cvc5\|bitwuzla\|portfolio>` | Select one backend or require three-way consensus |
| `--allow-memory` / `--allow-calls` / `--max-blocks <n>` | Widen slicing scope (stays sound) |
| `--allow-join-merge` / `--unknowns-on-truncation` | Recover diamonds / free-input boundaries (stays sound) |
| `--ir <esil\|pcode\|auto>` / `--deep-analysis` | Prefer r2ghidra P-code per instruction / run r2's deeper `aaaa` analysis |

Experimental r2sleigh interop is available without linking its LGPL code:

```bash
r2smt r2il --arch x86-64 --bytes 31c0
```

The command validates `r2sleigh`'s R2IL sidecars, adapts their paired ESIL to
r2SMT IR, and reports operation count, statement count, and elapsed time.

### Verdicts & Findings

| Verdict | Meaning |
|---------|---------|
| `AlwaysTrue` / `AlwaysFalse` | Branch can only go one way → obfuscation |
| `BothPossible` | Genuine branch (real control flow) |
| `Unsound` / `Timeout` | Slice truncated or solver gave up — not actionable |

| Finding kind | Notes |
|--------------|-------|
| `opaque_predicate`, `dead_branch`, `constant_condition` | Actionable; confidence `high` (clean slice) → `medium` (some unmodeled inputs) → `unknown` |
| `real_branch`, `suspicious_but_unknown` | Informational — opt in with `--include-real` / `--include-suspicious` |

Patching only acts at `--min-confidence high` by default; lower it explicitly and at your own risk.

---

## Library

r2SMT is a Cargo workspace of layered crates, so its analysis stages can be embedded independently of the CLI. Domain crates depend only on ports; adapters (radare2, Z3) are wired at the composition root.

```text
r2smt-common       errors, primitives, register-layout tables
r2smt-ir           program model + symbolic IR (Expr / IrStmt) + ports
r2smt-r2pipe       radare2 adapter (BinaryProvider / Annotator / BytePatcher)
r2smt-esil         ESIL stack-machine lifter (radare2 ESIL → IR)
r2smt-pcode        Ghidra/SLEIGH P-code lifter (`pdgsd` → IR)
r2smt-slicer       branch collection, backward slicing, and the per-mnemonic lifter
r2smt-ssa          SSA rename pass
r2smt-difflift     differential harness: P-code ≡ ESIL ≡ per-mnemonic by SMT equivalence
r2smt-taint        sound may-taint lattice over the SSA data flow
r2smt-solver-port  the narrow contract every SMT backend implements
r2smt-smt          Z3 / CVC5 / Bitwuzla backends
r2smt-z3fp         thin audited FFI shim for Z3 FP constructors missing from the safe API
r2smt-core         orchestration, findings, and the decision engine
r2smt-report       Markdown / JSON / r2-script renderers
r2smt-patch        plan / apply / rollback with a durable manifest
r2smt-explore      fenced UNSOUND exploration engine (optional radius2 oracle)
r2smt-cli          the `r2smt` binary (composition root)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for dependency rules and the runtime flow.

---

## CI & Quality Gates

Every change must pass the same gates CI enforces (see `scripts/quality-gates.sh`):

Parser and manifest fuzz targets are documented in [FUZZING.md](FUZZING.md)
and run on the scheduled `Fuzz` workflow.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --exclude r2smt-explore --all-targets --all-features -- -D warnings
cargo clippy -p r2smt-explore --all-targets -- -D warnings
cargo test --all
cargo doc --workspace --exclude r2smt-explore --no-deps --all-features   # under RUSTDOCFLAGS="-D warnings"
cargo doc -p r2smt-explore --no-deps                                     # idem
./scripts/quality-gates.sh supply-chain               # cargo audit
./scripts/quality-gates.sh solver-contracts           # Z3 / CVC5 / Bitwuzla verdict parity
./scripts/quality-gates.sh determinism                # pinned-seed verdict stability
```

The codebase is safe Rust throughout — no `unsafe`, no `unwrap`/`expect` in production paths, and lint bypasses (`#[allow(...)]`) are forbidden without a reviewed justification.

---

## Requirements

- **Rust 1.95+** (edition 2024) to build from source
- A **C++ toolchain + CMake** to build the vendored Z3
- **radare2 ≥ 6.2.0** on `PATH`
- **pkg-config + a C compiler** to build the native radare2 plugin
- Optional: **CVC5** / **Bitwuzla** binaries for the portfolio backends
- Optional: **r2sleigh** for the experimental process-boundary R2IL adapter

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Ensure the quality gates above pass
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## Support the Project

If this project is useful in your workflows, you can support development:

<a href="https://buymeacoffee.com/seifreed" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50">
</a>

---

## License

Licensed under the **MIT** license. See [LICENSE](LICENSE).

**Attribution**
- Author: **Marc Rivero López** | [@seifreed](https://github.com/seifreed)
- Repository: [github.com/seifreed/r2SMT](https://github.com/seifreed/r2SMT)

---

<p align="center">
  <sub>Built for practical malware deobfuscation and symbolic-analysis research</sub>
</p>
