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
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange?style=flat-square&logo=rust&logoColor=white" alt="Rust Version">
  <img src="https://img.shields.io/badge/edition-2024-orange?style=flat-square&logo=rust&logoColor=white" alt="Rust Edition">
  <a href="https://github.com/seifreed/r2SMT/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <a href="https://github.com/seifreed/r2SMT/stargazers"><img src="https://img.shields.io/github/stars/seifreed/r2SMT?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/seifreed/r2SMT/issues"><img src="https://img.shields.io/github/issues/seifreed/r2SMT?style=flat-square" alt="GitHub Issues"></a>
  <a href="https://buymeacoffee.com/seifreed"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow?style=flat-square&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee"></a>
</p>

---

## Overview

**r2SMT** is a Rust 2024 toolchain that combines radare2 with an SMT solver
(Z3) to perform symbolic analysis of obfuscated binaries. It lifts radare2
disassembly into a typed IR, renames it into SSA form, and asks the solver
whether each conditional branch is feasible — detecting **opaque predicates**,
**dead branches**, and **constant conditions**, then optionally annotating or
patching the binary to neutralise them.

It is sample-agnostic by policy: every analysis seam (r2 ingestion, IR lifting,
SMT encoding, solver dispatch, projection, CLI) is general-purpose and never
hardcodes values from a single malware family.

> Conceptual sibling of the Python+IDA original *MicroSMT*. r2SMT is a clean
> Rust reimplementation on top of radare2 with multi-architecture support,
> a durable patch manifest, and rollback.

### Key Features

| Feature | Description |
|---------|-------------|
| **Typed IR + SSA** | Bit-vector `Expr` / `IrStmt` model with single-pass SSA renaming |
| **SMT backend** | Z3-backed verdicts: `AlwaysTrue` / `AlwaysFalse` / `BothPossible` / `Unsound` / `Timeout` |
| **Multi-arch** | x86, x86_64, AArch64, and AArch32 (incl. Thumb) lifters |
| **Classified findings** | `opaque_predicate`, `dead_branch`, `constant_condition`, `real_branch`, `suspicious_but_unknown` with confidence ladder |
| **Backward slicing** | Bounded data-flow slicer with explicit truncation reasons |
| **Safe patching** | Byte-level rewrites with full-file backup, SHA-256 manifest, and reverse rollback |
| **Live annotations** | Write-back `CCu` comments through a live r2 session + project save |
| **CLI + crates** | Use as a command-line tool or as a Rust workspace of ports/adapters |
| **Robust on real corpora** | No-CFG skip, transient-spawn retry, and a data-as-code block filter |

### Supported Outputs

```text
AST / IR        Normalized Program model (JSON)
Findings        Stable JSON Report, human Markdown summary
Annotations     radare2 script (CCu comments + commented-out wa suggestions)
Patching        Applied bytes + JSON manifest (pre/post SHA-256, rollback)
```

---

## Installation

### From Releases (Recommended)

Prebuilt, self-contained binaries (vendored Z3, no runtime dependency) are
published for Linux and macOS on x86_64 and aarch64:

```bash
curl -fL https://github.com/seifreed/r2SMT/releases/latest/download/r2smt-$(uname -s)-$(uname -m).tar.gz \
  | tar -xz
./r2smt version
```

### From Source

```bash
git clone https://github.com/seifreed/r2SMT.git
cd r2SMT
cargo build --release
./target/release/r2smt version
```

Requires `radare2` on `PATH` (tested with r2 ≥ 6.1). Building from source
links Z3; install `libz3` (macOS: `brew install z3`, Debian/Ubuntu:
`apt-get install libz3-dev`) or build with the vendored feature.

### Via r2pm

```bash
r2pm -ci r2smt                 # prebuilt tarball, falls back to source build
USE_PREBUILT=0 r2pm -ci r2smt  # always build from source (dev path)
```

---

## Quick Start

```bash
# Inspect the normalized program model
r2smt analyze sample.bin --dump-program | head

# Collect every conditional-branch candidate
r2smt branches sample.bin

# Run the full pipeline and emit classified findings
r2smt solve sample.bin --json findings.json --markdown report.md
```

---

## Usage

### Command Line Interface

```bash
# Solve and export a JSON report plus an r2 annotation script
r2smt solve sample.bin --json findings.json --r2-script annotate.r2

# Apply only high-confidence findings as live r2 comments, then save a project
r2smt annotate sample.bin --min-confidence high --save-project deob

# Plan a conservative byte patch (dry-run), then apply with backup + manifest
r2smt patch sample.bin --min-confidence high
r2smt patch sample.bin --min-confidence high --apply \
  --backup sample.bak --manifest sample.manifest.json

# Roll the patch back from the manifest
r2smt patch sample.bin --rollback --manifest sample.manifest.json
```

### Commands

| Command | Description |
|---------|-------------|
| `r2smt version` | Print the build version |
| `r2smt analyze` | Open with r2, run `aaa`, emit the normalized Program model |
| `r2smt branches` | Collect `jcc` / `setcc` / `cmovcc` / ARM conditional candidates |
| `r2smt slice` | Backward data-flow slice of each branch (complete / truncated) |
| `r2smt lift` | Lift each slice into the r2SMT IR + symbolic flag condition |
| `r2smt ssa` | Full pipeline through SSA renaming with free-input reporting |
| `r2smt solve` | Full pipeline + Z3 → classified findings and reports |
| `r2smt at` | Interactive single-branch verdict at an address (drive from r2 via `$r2smt-at`) |
| `r2smt annotate` | Write findings back as live `CCu` comments / save project |
| `r2smt patch` | Derive, apply, and roll back conservative byte patches |

### Common Flags

| Option | Description |
|--------|-------------|
| `--at <addr>` / `--function <addr>` | Restrict analysis to one branch / function |
| `--max-instructions N` | Slicer instruction budget |
| `--allow-memory` / `--allow-calls` | Widen slicing past memory ops / calls |
| `--timeout-ms MS` | Per-branch Z3 timeout |
| `--min-confidence <conf>` | Gate findings by confidence (`high`/`medium`/`low`) |
| `--json` / `--markdown` / `--r2-script <file>` | Machine + human + r2 outputs |
| `--apply` / `--backup` / `--manifest` / `--rollback` | Patch lifecycle controls |

---

## Architecture

r2SMT is a Clean-Architecture workspace: inner layers never import outer
layers, ports define contracts, and adapters are wired only at the CLI.

```text
crates/
  r2smt-common   Foundation: Error, Result, Address, Arch
  r2smt-ir       Program model + BinaryProvider/Annotator/BytePatcher ports + IR
  r2smt-r2pipe   radare2 adapter (live session, pure JSON parsers)
  r2smt-esil     ESIL-first lifting helpers
  r2smt-slicer   Branch collector + bounded backward slicer + per-ISA lifter
  r2smt-ssa      Single-pass SSA rename over lifted slices
  r2smt-smt      Z3 backend + textual SMT-LIB renderer
  r2smt-core     Orchestration + decision engine (Finding / Confidence)
  r2smt-report   Renderers: JSON / Markdown / r2 script + patch suggestions
  r2smt-patch    PatchPlan / PatchManifest, apply, rollback
  r2smt-cli      The `r2smt` binary
```

Pipeline: **radare2 → IR → backward slice → lift → SSA → Z3 → classify →
report / annotate / patch**.

---

## Quality Gates

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo deny check
cargo audit
./scripts/quality-gates.sh check
```

Engineering invariants are enforced repo-wide: zero `unwrap`/`expect`/`panic`
in production code, zero `unsafe`, `#![deny(missing_docs)]`, and a strict
sample-agnostic policy. See [`CLAUDE.md`](CLAUDE.md).

---

## Requirements

- Rust 1.85+ (edition 2024)
- `radare2` ≥ 6.1 on `PATH`
- Z3 (system `libz3` or the vendored build feature)

---

## Documentation

- [`SPEC.md`](SPEC.md) — full design spec and phase roadmap
- [`CLAUDE.md`](CLAUDE.md) — enforced engineering rules and architecture map

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

All changes must pass the quality gates above without `#[allow(...)]`
bypasses and must carry a regression artifact for any behavioural change.

---

## Support the Project

If this project is useful in your workflows, you can support development:

<a href="https://buymeacoffee.com/seifreed" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50">
</a>

---

## License

This project is dual-licensed under **MIT OR Apache-2.0** at your option.
See [LICENSE](LICENSE).

**Attribution**
- Author: **Marc Rivero López** | [@seifreed](https://github.com/seifreed)
- Repository: [github.com/seifreed/r2SMT](https://github.com/seifreed/r2SMT)

---

<p align="center">
  <sub>Built for practical malware deobfuscation and symbolic-analysis automation</sub>
</p>
