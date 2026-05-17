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
- 2026-05-17 POLICY (user decision): force-split EVERY prod file <800,
  **safest-first**. For risky/interleaved files (registers.rs has
  `const fn` + per-ISA name tables interleaved with handlers; load-bearing
  for SMT) use a 2-phase per file: (1) reorder items contiguous by
  ISA/domain → verified commit; (2) extract contiguous blocks to
  submodules → verified commit. Clean-seam files split directly.
  Order: main.rs (commands/*.rs) → effect.rs (per-ISA) → lift.rs
  (merge-trio+per-ISA) → registers.rs (2-phase) → parse.rs/plan.rs/slice.rs.
- 2026-05-17 iter5: main.rs decomposition step 1/N — created shared
  `support.rs` (AnalysisContext+impl, analysis_level, open_provider/_writable,
  resolve_targets; all pub(crate)) + `commands.rs` aggregator +
  `commands/inspect.rs` (analyze/emit_program_json/branches/slice/lift/ssa).
  main.rs 1591→1296. Compiler-driven import trim (cargo fix, 6).
  495 tests/0 fail; all 5 gates GREEN. Remaining main.rs command groups
  to extract → commands/{solve,annotate,patch,batch,at}.rs importing
  crate::support (+ move compute_findings/dispatch_solver/
  resolve_folded_branch/analyze_one/attach_pseudocode/keep_finding to
  support.rs). main.rs still >800 — continue next iterations.
- 2026-05-17 iter6: main.rs step 2/N — moved the 4 cross-command shared
  helpers (attach_pseudocode+MAX_PSEUDOCODE_BYTES, compute_findings,
  dispatch_solver, resolve_folded_branch) to support.rs (pub(crate)).
  main.rs 1296→1131; support.rs 124→297. Compiler-driven imports
  (cargo fix). 495 tests/0 fail (verified via per-binary breakdown — a
  transient "430" was parallel-stdout grep interleaving, NOT lost tests);
  all 5 gates GREEN. Remaining single-caller helpers stay with their
  command group. Next: extract command groups to commands/{patch,batch,
  solve,annotate,at}.rs (+ their owned structs/single-caller helpers)
  importing crate::support, until main.rs<800.
- 2026-05-17 iter7: extracted batch group (analyze_one + batch) →
  commands/batch.rs (pub(crate) fn batch; analyze_one module-private,
  single-caller). main.rs 1131→971; commands/batch.rs 180. Compiler-
  driven imports (+ manual rayon trim cargo fix missed). 495 tests/0
  fail (per-binary); all 5 gates GREEN. Remaining main.rs >800: groups
  patch(+PatchCli/default_*), solve(+SolveFilters/SolveOutputs/
  keep_finding), annotate(+AnnotatePlan), at(+AtVerbosity/AtOptions).
  1-2 more iterations → main.rs<800.

- 2026-05-17 iter8: extracted patch group (PatchCli+DEFAULT_*_SUFFIX+
  default_backup_path/default_manifest_path+patch+patch_dry_run_plan+
  patch_rollback) -> commands/patch.rs (PatchCli pub(crate)+fields,
  patch pub(crate); dry_run/rollback/default_* module-private). run +
  at_command import via use commands::patch::{PatchCli, patch}.
  **main.rs 971->796 — UNDER 800, decomposition COMPLETE** (1591->796
  over iters5-8: support.rs+commands/{inspect,batch,patch}.rs). 495
  tests/0 fail; all 5 gates GREEN. Remaining prod >800: lift.rs 1818,
  effect.rs 1116, registers.rs 924, slice.rs 884, parse.rs 844,
  plan.rs 819. Next: effect.rs per-ISA, then lift.rs, registers.rs
  (2-phase), parse/plan/slice.

- 2026-05-17 iter9: effect.rs per-ISA split (clean seam, NO interleaved
  consts — all fn; zero cross-ISA calls). effect.rs 1116→321 (header +
  InstructionKind/InstructionEffect + 9 shared helpers + analyze
  dispatcher + other_effect); effect/x86.rs 365, effect/aarch64.rs 255,
  effect/aarch32.rs 207. Key: child modules see parent-private items, so
  shared helpers needed NO pub change; only analyze_{x86,aarch64,aarch32}
  → pub(super). Compiler-driven (cargo fix trimmed per-ISA-unused super
  imports) + doc_markdown backtick fix (`AArch32`/`AArch64`). 495 tests/
  0 fail; all 5 gates GREEN. Remaining prod >800: lift.rs 1818,
  registers.rs 924, slice.rs 884, parse.rs 844, plan.rs 819.

- 2026-05-17 iter10: lift.rs step 1/2 (highest-stakes file, safest-first).
  Extracted Φ-merge trio (lower_merge/fold_arm/subst_expr free fns) →
  lift/merge.rs (142; lower_merge pub(super), called by root lift_slice)
  + x86 handler cluster (lift_instruction_x86 pub(super) + lift_mov/
  mov_extending/lea/xor/bitwise/add_sub/imul/cmp/test/shift) →
  lift/x86.rs (399, second `impl LiftCtx` block). x86 has zero cross-ISA
  deps; shared infra stays in root impl (child sees ancestor-private —
  no pub change). lift.rs 1818→1305. Compiler-driven (super:: imports
  for root free fns nonzero_width/lift_branch_condition; cargo fix).
  495 tests/0 fail; all 5 gates GREEN — lifter behavior preserved.
  Remaining prod >800: lift.rs 1305 (NEXT: extract aarch64+aarch32
  clusters; aarch32→aarch64 cross-calls need those aarch64 entrypoints
  pub(crate)), registers.rs 924, slice.rs 884, parse.rs 844, plan.rs 819.

- 2026-05-17 iter11: lift.rs step 2/2 — extracted AArch64 cluster
  (lift_instruction_aarch64 + mov/arith3/aarch64_set_arith_flags/cmp/
  csel/cs_arith/cset/tst) -> lift/aarch64.rs (399) and AArch32 cluster
  (lift_instruction_aarch32 + rsb/bic/cmn/teq/predicated/mvn) ->
  lift/aarch32.rs (253). Key: pub(super) == pub(in lift) is visible to
  ALL lift descendants, so the 5 cross-module aarch64 entrypoints
  (instruction + arith3/cmp/mov/tst, called by root dispatcher AND the
  aarch32 sibling) + lift_instruction_aarch32 just need pub(super) — no
  pub(crate). Built FIRST TRY. **lift.rs 1818->681 — UNDER 800,
  decomposition COMPLETE** (lift/{merge,x86,aarch64,aarch32}.rs).
  Compiler-driven super:: imports for root free fns; cargo fix.
  495 tests/0 fail; all 5 gates GREEN — lifter behavior preserved.
  Remaining prod >800 (4 files, all modestly over): registers.rs 924,
  slice.rs 884, parse.rs 844, plan.rs 819. Next: registers.rs 2-phase
  (reorder-contiguous commit, then extract commit).

- 2026-05-17 iter12: registers.rs 2-phase PHASE 1/2 (reorder). Accurate
  const-fn-aware item map (47 items) + full call graph: clean per-ISA
  EXCEPT aarch64_vector->dword (x86 const builder). Design: ALL tiny
  RegisterLayout const-fn builders stay in shared root (children call
  via super::, ancestor-private); extended_alias is x86-only. Reordered
  registers.rs into contiguous blocks [shared: RegisterLayout+impl+
  register_layout/alias_for dispatchers+10 const builders][x86: layout/
  alias/extended_alias][AArch64 ...][AArch32 ...] + banner comments.
  Coverage assertion (every non-blank line in exactly one unit/header/
  tail) PASSED; item set verified byte-identical (pure permutation).
  registers.rs 924->933 (banners; Phase 1 reorders, Phase 2 reduces).
  495 tests/0 fail; all 5 gates GREEN. PHASE 2 next: extract contiguous
  x86/aarch64/aarch32 blocks -> registers/{x86,aarch64,aarch32}.rs
  (pub(super) on the 6 dispatched entrypoints: x86/aarch64/arm32 _layout
  + _alias; rest private; child sees ancestor-private builders).

- 2026-05-17 iter13: registers.rs PHASE 2/2 (extract). The contiguous
  banner-delimited blocks → registers/{x86.rs 197, aarch64.rs 300,
  aarch32.rs 209}; registers.rs 933→241 (header + RegisterLayout+impl +
  register_layout/alias_for dispatchers + 10 const builders + mod decls
  + use of the 6 pub(super) entrypoints + tests). pub(super)=pub(in
  registers) on x86/aarch64/arm32 _layout+_alias; const builders stay
  root-private (children use super::). Built FIRST TRY. 495 tests/0 fail;
  all 5 gates GREEN — register layout behavior preserved.
  **registers.rs COMPLETE 924→241.** Remaining prod >800 (3, all
  modestly over): slice.rs 884, parse.rs 844, plan.rs 819 — per-domain
  seam, DRY-with-a-brake (split only on a real cohesive seam).
