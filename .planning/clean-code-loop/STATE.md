# Clean-Code / Clean-Architecture Hardening Loop

Durable state for the `/loop` driving r2SMT to a 10/10 clean-code +
clean-architecture score. Survives context compaction and wakeups.

## Scoring rubric (score = criteria fully met; target 10/10)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `cargo fmt --all -- --check` clean | PASS (verified) |
| 2 | `cargo clippy --all-targets --all-features -- -D warnings` clean | PASS (verified) |
| 3 | `cargo test --all` green | PASS (verified) |
| 4 | `cargo deny check` clean | PASS (verified) |
| 5 | `cargo audit` clean | PASS (verified) |
| 6 | Zero production `unwrap/expect/unsafe/panic` | PASS (0; lints forbid) |
| 7 | Every crate `#![deny(missing_docs)]` + builds | PASS |
| 8 | No production `.rs` > 2000 LoC; watchlist trending down | FAIL (main.rs 2066) |
| 9 | Named imports only in production (no disallowed glob) | PASS (clippy clean) |
| 10 | Clean arch: dependency rule, no circular crate deps, ports/adapters | UNVERIFIED |

**Score: 8/10.** Blockers: #8 (decompose `main.rs`), #10 (audit crate-dep direction).

## Known facts (baseline)

- Repo had ZERO commits; baseline commit = `f6d25a8` (scaffold as-is).
- Remote: https://github.com/seifreed/r2SMT.git (origin/main).
- Workspace lints already strict: `unsafe_code=forbid`,
  `clippy::all=deny`, `pedantic=warn`, `unwrap_used/expect_used/panic=deny`,
  `missing_docs=deny`, `mod_module_files=deny`, `print_std*=deny`.
- Extracted test files carry
  `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`.
- Test-extraction pattern (blessed): `foo.rs` ends with
  `#[cfg(test)]\nmod tests;`, body moves to `foo/tests.rs`.

## Production files over / near limits

- `crates/r2smt-cli/src/main.rs` — 2066 (all production, no inline tests) — REAL decomposition
- `crates/r2smt-slicer/src/slice.rs` — 2025 (prod ~882 + inline tests 883..EOF) — extract tests → slice/tests.rs
- `crates/r2smt-slicer/src/lift.rs` — 1818 (tests already external) — split per-ISA later
- `crates/r2smt-smt/src/solver.rs` — 1742
- `crates/r2smt-slicer/src/effect.rs` — 1590
- `crates/r2smt-patch/src/plan.rs` — 1543
- `crates/r2smt-r2pipe/src/parse.rs` — 1469 (prod ~? + inline tests)

## Work queue (ordered, behavior-preserving first)

1. [x] Extract `slice.rs` inline test module → `slice/tests.rs` (2025→884; 135 tests green)
2. [x] Verify gates 1-5 green — ALL GREEN
3. [ ] Decompose `main.rs` 2066 (real refactor: per-subcommand modules) — keep CLI contract — **NEXT, criterion #8**
4. [ ] Verify crate dependency graph matches CLAUDE.md (no inner→outer, no cycles) — criterion #10
5. [ ] (optional polish) extract other inline test mods (parse.rs, effect.rs…) to keep prod LoC honest
6. [ ] Split `lift.rs` per-ISA handlers (watchlist-flagged, 1818 — below hard limit, lower priority)

## Iteration log

- 2026-05-17: baseline commit f6d25a8; rubric defined.
- 2026-05-17: slice.rs 2025→884 (test mod → slice/tests.rs, blessed pattern);
  full Z3 build OK (5m30s); all 5 gates verified GREEN; score 8/10.
  CLAUDE.md watchlist synced (note: CLAUDE.md is gitignored, not in commits).
