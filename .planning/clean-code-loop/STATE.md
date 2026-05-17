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
| 8 | No production `.rs` > 2000 LoC; watchlist trending down | PASS (none >2000) |
| 9 | Named imports only in production (no disallowed glob) | PASS (clippy clean) |
| 10 | Clean arch: dependency rule, no circular crate deps, ports/adapters | PASS (verified) |

**Score: 10/10.** All rubric criteria met against the project's own
clean-code/clean-architecture contract (CLAUDE.md) + all 5 mandatory gates.

Residual *documented* debt (NOT a 10/10 blocker — the project's own
watchlist accepts files in the 800–1800 band with ~1800 promote lines):
lift.rs 1818, solver.rs 1742, effect.rs ~1114 prod, plan.rs ~817 prod
could be split further toward the <800 *target*. Tracked in the
CLAUDE.md decomposition watchlist, not mandated by the hard rule
(>2000 requires decomposition — now satisfied everywhere).

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
- 2026-05-17 iter2: #10 verified PASS (crate-dep graph matches documented
  layering exactly; core adapter-free; no cycles) — no code change.
  #8 resolved: main.rs 2066→1590 by extracting the 15-fn presentation
  cluster to render.rs (pure move, compiler-driven import trim). All 5
  gates re-verified GREEN; no prod file >2000. **Score 10/10.**
- 2026-05-17 iter3 (user-requested deeper <800 pass): test-module
  relocation pass — extracted inline `mod tests` to sibling `<stem>/tests.rs`
  for effect/registers/solver/plan/finding/machine (blessed pattern).
  Prod LoC: effect 1590→1116, registers 1354→924, solver 1742→157,
  plan 1543→819, finding 1052→451, machine 884→592. Removed 3 redundant
  subset `#![allow]` lines (clippy `duplicated_attributes`, surfaced by
  extraction — DRY cleanup, not a bypass). parse.rs SKIPPED (raw-string
  JSON fixture at col 0 — needs bespoke handling). 495 tests pass / 0 fail;
  all 5 gates GREEN. Remaining >800 prod: lift.rs 1818, main.rs 1590,
  parse.rs 1469, effect.rs 1116, registers.rs 924, slice.rs 884, plan.rs 819.
- 2026-05-17 iter4: bespoke parse.rs test extraction (1469→844). Verbatim
  body move (NO de-indent — the pdgsd `"\`-continuation fixture has
  semantically-meaningful leading spaces in string content), then rustfmt
  re-indents code while leaving string literals byte-exact. Preserved the
  single `#![allow(clippy::unwrap_used)]` header (no canonical-allow
  duplication). Fixture verified byte-identical post-fmt; the two
  fixture-dependent tests (split_pdgsd_*) pass. 495 tests / 0 fail; all
  5 gates GREEN. Remaining >800 prod: lift.rs 1818, main.rs 1590,
  effect.rs 1116, registers.rs 924, slice.rs 884, parse.rs 844,
  plan.rs 819 — all need *genuine* code splits now (per-ISA / per-cmd /
  per-subcommand seams; watchlist prescribes the seams). slice/parse/plan
  only modestly over — apply DRY-with-a-brake; split only on a real seam.
