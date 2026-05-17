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

## What it does

`r2SMT` asks an SMT solver whether each conditional branch in a binary can
*actually* go both ways. Branches that can't are **opaque predicates**, **dead
branches**, or **constant conditions** — classic obfuscation. It then lets you
annotate or patch the binary to neutralise them.

Pipeline: `radare2 disasm → typed IR → backward slice → SSA → Z3 → verdict`.
Sample-agnostic by design (no hardcoded family IOCs). Lifters: x86 / x86_64 /
AArch64 / AArch32 (incl. Thumb).

---

## Install

```bash
# From source — links Z3, so install libz3 first
#   macOS:  brew install z3
#   Debian: apt-get install libz3-dev
git clone https://github.com/seifreed/r2SMT.git
cd r2SMT && cargo build --release
./target/release/r2smt version

# Prebuilt (vendored Z3, no runtime deps) — Linux/macOS, x86_64/aarch64
curl -fL https://github.com/seifreed/r2SMT/releases/latest/download/r2smt-$(uname -s)-$(uname -m).tar.gz | tar -xz

# Or via r2pm
r2pm -ci r2smt
```

Needs `radare2` ≥ 6.1 on `PATH`.

---

## 60-second start

```bash
# 1. What conditional branches exist?
r2smt branches ./sample

# 2. Solve them all and print classified findings
r2smt solve ./sample

# 3. Drill into one suspicious address — one-line verdict
r2smt at ./sample 0x401234
```

`solve` prints one classified line per branch, e.g.:

```text
0x00401234  opaque_predicate    AlwaysFalse   high     je   → never taken
0x004012a0  dead_branch         AlwaysTrue    high     jne  → always taken
0x00401310  real_branch         BothPossible  high     jg   (genuine)
```

---

## Recipes

Every analysis command takes `--at <addr>` (one branch), `--function <addr>`
(one function), and `--timeout-ms <ms>` (per-branch solver budget).

**Dump the normalized program model:**

```bash
r2smt analyze ./sample --dump-program --json program.json
```

**Solve and export reports for triage:**

```bash
# JSON Report + human Markdown + an r2 annotation script, one pass
r2smt solve ./sample --json findings.json --markdown findings.md --r2-script annotate.r2

# Also surface genuine + unknown branches, give the solver more time
r2smt solve ./sample --include-real --include-suspicious --timeout-ms 5000
```

**See exactly what the solver sees (stage by stage):**

```bash
r2smt slice ./sample --at 0x401234   # bounded backward data-flow slice
r2smt lift  ./sample --at 0x401234   # that slice lifted to IR
r2smt ssa   ./sample --at 0x401234   # IR after SSA renaming
```

**Annotate a live radare2 session:**

```bash
r2smt annotate ./sample --dry-run                                  # preview
r2smt annotate ./sample --min-confidence high --save-project triage # apply + save
```

…or from inside r2, on the branch under the cursor:

```text
[0x00401234]> #!pipe r2smt at "${R2_FILE}" $$
```

**Patch — always backed up, always reversible:**

```bash
r2smt patch ./sample                                  # plan only, writes nothing
r2smt patch ./sample --apply --min-confidence high    # backup + manifest + patch
r2smt patch ./sample --rollback                       # restore originals from manifest
```

`--apply` writes `<sample>.r2smt.bak` and `<sample>.r2smt.manifest.json`
(pre/post SHA-256); `--rollback` replays the manifest in reverse.

**Sweep a directory (one isolated r2 process per sample, aggregated):**

```bash
r2smt batch ./corpus --threads 8 --json corpus.json --markdown corpus.md
```

**Reach deeper — all of these are sound** (they can only widen a verdict to
`BothPossible`, never fabricate one):

```bash
r2smt solve ./sample --allow-memory --allow-calls --max-blocks 4 \
  --unknowns-on-truncation --allow-join-merge --solver cvc5
```

---

## Verdicts & findings

| Verdict | Meaning |
|---|---|
| `AlwaysTrue` / `AlwaysFalse` | Branch can only go one way → obfuscation |
| `BothPossible` | Genuine branch (real control flow) |
| `Unsound` / `Timeout` | Slice truncated or solver gave up — not actionable |

| Finding kind | Notes |
|---|---|
| `opaque_predicate`, `dead_branch`, `constant_condition` | actionable; confidence `high` (clean slice) → `medium` (some unmodeled inputs) → `unknown` |
| `real_branch`, `suspicious_but_unknown` | informational — opt in with `--include-real` / `--include-suspicious` |

Patching only acts at `--min-confidence high` by default; lower it explicitly
and at your own risk.

---

## Requirements

- Rust 1.85+ (edition 2024) to build from source
- `radare2` ≥ 6.1 on `PATH`
- Z3 (system `libz3`, or the vendored build feature)

---

## License

Dual-licensed under **MIT OR Apache-2.0** at your option — see [LICENSE](LICENSE).

- Author: **Marc Rivero López** — [@seifreed](https://github.com/seifreed)
- Support: <a href="https://buymeacoffee.com/seifreed">Buy Me a Coffee</a>
