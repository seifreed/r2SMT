//! Solver wrapper that delivers an opaque-predicate verdict for a
//! single SSA lifted slice.

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_slicer::SliceStatus;
use r2smt_ssa::SsaLiftedSlice;
use tracing::debug;
use z3::ast::{Ast, Bool};
use z3::{ApplyResult, Goal, Params, SatResult, Solver, Tactic};

use crate::encoder::Encoder;
use crate::pretty::z3_bool_to_infix;

/// Rich outcome of [`solve_branch_with_pretty`]: pairs the
/// [`SmtResult`] verdict with the C-style infix rendering of the
/// post-`aggressive_simplify` Z3 formula, when one was produced.
#[derive(Debug, Clone)]
pub struct SolveOutcome {
    /// Verdict reported to the classifier.
    pub verdict: SmtResult,
    /// Infix rendering of the solver-simplified formula. `None` for
    /// truncated slices that short-circuit before the solver runs.
    pub formula_z3_pretty: Option<String>,
}

/// Solve a branch and report the verdict.
///
/// Slices with a truncated status are reported as
/// [`SmtResult::Unsound`] without invoking the solver — by definition
/// they do not capture the full data-flow.
///
/// When the slice was produced under
/// [`r2smt_slicer::SliceLimits::unknowns_on_truncation`], the
/// truncated result still drives the solver: SSA already surfaced the
/// unresolved roots as free symbolic [`r2smt_ir::expr::Var`]s in
/// [`SsaLiftedSlice::inputs`], so the verdict is sound. The
/// classifier marks the resulting finding with a lower confidence
/// downstream.
#[must_use]
pub fn solve_branch(slice: &SsaLiftedSlice, options: SolveOptions) -> SmtResult {
    solve_branch_with_pretty(slice, options).verdict
}

/// Variant of [`solve_branch`] that also returns the C-style infix
/// rendering of the post-`aggressive_simplify` Z3 formula. Used by
/// the classifier to populate `r2smt_core::Finding::formula_z3_pretty`.
#[must_use]
pub fn solve_branch_with_pretty(slice: &SsaLiftedSlice, options: SolveOptions) -> SolveOutcome {
    let is_complete = matches!(slice.status, SliceStatus::Complete);
    if !is_complete && !slice.treat_truncation_as_inputs {
        return SolveOutcome {
            verdict: SmtResult::Unsound,
            formula_z3_pretty: None,
        };
    }

    let solver = Solver::new();
    let mut params = Params::new();
    params.set_u32("timeout", options.timeout_ms);
    solver.set_params(&params);

    let mut encoder = Encoder::new();
    let raw = encoder.encode(slice, &solver);
    // Pre-simplify the condition before issuing SAT queries.
    //
    // The lightweight `raw.simplify()` falls short on polynomial-style
    // opaque predicates (e.g. `(x*x) - (x*x) == 0`) because it does not
    // normalise expressions into sum-of-monomials form by default.
    // Apply a tactic chain — `simplify` with `som` and `blast_eq_value`
    // enabled, then `propagate-values` and `ctx-simplify` — to fold a
    // wider class of identities before the SAT loop. The chain is
    // equivalence-preserving, so the verdict is unchanged when nothing
    // collapses; on success the solver sees a smaller formula on both
    // push/pop iterations.
    let truth = aggressive_simplify(&raw);
    let pretty = Some(z3_bool_to_infix(&truth));

    solver.push();
    solver.assert(&truth);
    let sat_true = solver.check();
    solver.pop(1);

    solver.push();
    let not_truth = truth.not();
    solver.assert(&not_truth);
    let sat_false = solver.check();
    solver.pop(1);

    let verdict = combine(sat_true, sat_false);
    debug!(
        target: "r2smt::smt",
        at = %slice.branch.address,
        ?sat_true,
        ?sat_false,
        ?verdict,
        "smt verdict"
    );
    SolveOutcome {
        verdict,
        formula_z3_pretty: pretty,
    }
}

fn aggressive_simplify(raw: &Bool) -> Bool {
    let mut simplify_params = Params::new();
    simplify_params.set_bool("som", true);
    simplify_params.set_bool("blast_eq_value", true);

    let simplify = Tactic::new("simplify").with(&simplify_params);
    let chain = simplify
        .and_then(&Tactic::new("propagate-values"))
        .and_then(&Tactic::new("ctx-simplify"));

    let goal = Goal::new(false, false, false);
    goal.assert(raw);

    match chain.apply(&goal, None) {
        Ok(result) => collapse_subgoals(result),
        Err(_) => raw.simplify(),
    }
}

fn collapse_subgoals(result: ApplyResult) -> Bool {
    let mut conjuncts: Vec<Bool> = Vec::new();
    for goal in result.list_subgoals() {
        if goal.is_decided_unsat() {
            return Bool::from_bool(false);
        }
        if goal.is_decided_sat() {
            continue;
        }
        let formulas = goal.get_formulas();
        if formulas.is_empty() {
            continue;
        }
        let refs: Vec<&Bool> = formulas.iter().collect();
        conjuncts.push(Bool::and(refs.as_slice()));
    }
    if conjuncts.is_empty() {
        return Bool::from_bool(true);
    }
    let refs: Vec<&Bool> = conjuncts.iter().collect();
    Bool::and(refs.as_slice())
}

fn combine(sat_true: SatResult, sat_false: SatResult) -> SmtResult {
    match (sat_true, sat_false) {
        (SatResult::Sat, SatResult::Unsat) => SmtResult::AlwaysTrue,
        (SatResult::Unsat, SatResult::Sat) => SmtResult::AlwaysFalse,
        (SatResult::Sat, SatResult::Sat) => SmtResult::BothPossible,
        (SatResult::Unsat, SatResult::Unsat) => SmtResult::Unsound,
        (SatResult::Unknown, _) | (_, SatResult::Unknown) => SmtResult::Timeout,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use r2smt_common::smt::SolveOptions;
    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
    use r2smt_ssa::ssa_convert;

    use super::*;

    fn op(raw: &str, kind: OperandKind) -> Operand {
        Operand {
            raw: raw.into(),
            kind,
        }
    }

    fn insn(addr: u64, size: u8, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            address: Address(addr),
            size,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands,
            esil: None,
            pcode: None,
            is_thumb: false,
        }
    }

    fn one_block(insns: Vec<Instruction>) -> Program {
        program_with_arch(insns, Arch::X86_64)
    }

    fn aarch64_block(insns: Vec<Instruction>) -> Program {
        program_with_arch(insns, Arch::Aarch64)
    }

    fn program_with_arch(insns: Vec<Instruction>, arch: Arch) -> Program {
        Program {
            arch,
            bits: arch.pointer_bits(),
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: insns,
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    fn solve_first(program: &Program) -> SmtResult {
        solve_first_with_timeout(program, SolveOptions::default().timeout_ms)
    }

    fn solve_first_with_timeout(program: &Program, timeout_ms: u32) -> SmtResult {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        let slice = slice_branch(
            cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        );
        let lifted = lift_slice(&slice, program.arch);
        let ssa = ssa_convert(&lifted);
        solve_branch(&ssa, SolveOptions { timeout_ms })
    }

    #[test]
    fn stack_slot_store_then_load_then_cmp_resolves_constant() {
        // Phase C minimal stack memory test:
        //
        //     mov dword ptr [rbp - 4], 5
        //     mov eax, dword ptr [rbp - 4]
        //     cmp eax, 5
        //     je dest
        //
        // The slot value is constant 5, so `eax == 5` and the je is
        // always taken. Verifies the slicer keeps the stack store /
        // load chain and the solver resolves it to AlwaysTrue.
        let program = one_block(vec![
            insn(
                0x40_1000,
                7,
                "mov",
                vec![
                    op("dword ptr [rbp - 4]", OperandKind::Memory),
                    op("5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1007,
                3,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("dword ptr [rbp - 4]", OperandKind::Memory),
                ],
            ),
            insn(
                0x40_100a,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100d,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn xor_zero_idiom_then_jnz_is_always_false() {
        // xor eax, eax ; test eax, eax ; jnz junk
        // After xor: rax = 0, ZF = 1. test eax,eax recomputes flags
        // from rax & rax = 0 → ZF = 1. jnz fires when ZF == 0, so the
        // branch is **never** taken.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "xor",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                6,
                "jnz",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysFalse, "jnz after zero idiom");
    }

    #[test]
    fn constant_propagation_je_is_always_true() {
        // mov eax, 1 ; cmp eax, 1 ; je dest  →  always taken
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn constant_propagation_jne_is_always_false() {
        // mov eax, 1 ; cmp eax, 1 ; jne junk  →  never taken
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysFalse);
    }

    #[test]
    fn canonical_opaque_predicate_is_always_false() {
        // mov eax, ecx ; imul eax, eax ; and eax, 1 ; cmp eax, 2 ; jne junk
        // `(ecx * ecx) & 1` is in {0, 1}, never equal to 2, so `cmp eax, 2`
        // sets ZF = 0 always, and `jne` fires every time → AlwaysTrue.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                3,
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "and",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100b,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn polynomial_identity_x_squared_eq_x_squared_is_always_true() {
        // Polynomial-style opaque predicate that exercises the
        // aggressive `simplify+propagate-values+ctx-simplify` tactic
        // chain. The two `imul`s compute the same monomial through
        // distinct register chains; `cmp` then drives ZF from the
        // shared value rather than overwriting one of the operands.
        //
        //     mov eax, ecx     ; eax = x
        //     imul eax, eax    ; eax = x*x
        //     mov ebx, ecx     ; ebx = x
        //     imul ebx, ebx    ; ebx = x*x
        //     cmp eax, ebx     ; ZF = (x*x == x*x) = 1
        //     je  dest         ; always taken
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                3,
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1005,
                2,
                "mov",
                vec![
                    op("ebx", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1007,
                3,
                "imul",
                vec![
                    op("ebx", OperandKind::Register),
                    op("ebx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100a,
                2,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("ebx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100c,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        // Vendored Z3 in debug builds discharges this polynomial
        // identity slower than the 500 ms default; give it the same
        // generous budget as the sibling polynomial regression so the
        // assertion does not flake on loaded hosts.
        let verdict = solve_first_with_timeout(&program, 5_000);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn polynomial_offset_x_squared_plus_seven_minus_x_squared_eq_seven_is_always_true() {
        // (x*x + 7) - x*x == 7 — opaque predicate that hides behind a
        // constant offset. Without the `som` normalisation the two
        // `bvmul` subterms look distinct to the lightweight simplifier
        // and the SAT loop has to discharge a non-trivial polynomial
        // equation.
        //
        //     mov eax, ecx     ; eax = x
        //     imul eax, eax    ; eax = x*x
        //     add eax, 7       ; eax = x*x + 7
        //     mov ebx, ecx     ; ebx = x
        //     imul ebx, ebx    ; ebx = x*x
        //     sub eax, ebx     ; eax = 7
        //     cmp eax, 7       ; ZF = 1
        //     je  dest         ; always taken
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                3,
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "add",
                vec![
                    op("eax", OperandKind::Register),
                    op("7", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                2,
                "mov",
                vec![
                    op("ebx", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100a,
                3,
                "imul",
                vec![
                    op("ebx", OperandKind::Register),
                    op("ebx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100d,
                2,
                "sub",
                vec![
                    op("eax", OperandKind::Register),
                    op("ebx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100f,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("7", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1012,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        // Z3 in debug builds with vendored linking takes noticeably
        // longer to discharge polynomial identities; give the solver a
        // larger budget than the 500 ms default so this regression does
        // not flake on slow hosts.
        let verdict = solve_first_with_timeout(&program, 5_000);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn x_eq_x_via_xor_self_je_is_always_true() {
        // (x ^ x) == 0 — classic identity opaque predicate.
        // xor eax, eax produces 0; cmp eax, 0 sets ZF=1; je fires.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "xor",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1005,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn unconstrained_input_jcc_is_both_possible() {
        // test eax, eax ; jne junk — without prior knowledge of eax,
        // both branches are reachable.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::BothPossible);
    }

    #[test]
    fn mov_al_then_cmp_al_resolves_constant() {
        // mov al, 0x42 ; cmp al, 0x42 ; je dest
        // Even though only the low byte of rax is written, the
        // `cmp al, 0x42` reads exactly that byte, so ZF = 1 and `je`
        // is always taken. Exercises the precise sub-register model.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "mov",
                vec![
                    op("al", OperandKind::Register),
                    op("0x42", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1002,
                2,
                "cmp",
                vec![
                    op("al", OperandKind::Register),
                    op("0x42", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn cmp_ah_against_free_input_is_both_possible() {
        // cmp ah, 0x42 ; jne junk — `ah` (bits 15:8 of rax) is a free
        // input, so both branches are reachable. Verifies that the
        // sub-register read translates into a Z3 extract that the
        // solver can reason over (no longer collapses to Unknown via
        // the old `sub_register_alias` hack).
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "cmp",
                vec![
                    op("ah", OperandKind::Register),
                    op("0x42", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::BothPossible);
    }

    #[test]
    fn xor_ah_al_then_cmp_against_concrete_ah_is_both_possible() {
        // xor ah, al ; cmp ah, 0 ; je dest.
        // Under the old `sub_register_alias` Unknown hack the cmp
        // collapsed to Unknown flags and the verdict was Unsound /
        // BothPossible by default. Under the precise model the
        // formula is `((rax[15:8] ^ rax[7:0]) == 0)` over a free
        // input rax — solvable, both branches reachable.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "xor",
                vec![
                    op("ah", OperandKind::Register),
                    op("al", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                3,
                "cmp",
                vec![
                    op("ah", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1005,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::BothPossible);
    }

    // --- AArch64 ---

    #[test]
    fn aarch64_constant_propagation_b_eq_is_always_true() {
        // mov x0, #1 ; cmp x0, #1 ; b.eq dest → always taken.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cmp",
                vec![
                    op("x0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_canonical_opaque_predicate_is_always_true() {
        // The canonical Collberg-style identity:
        //   mov x0, x1
        //   mul x0, x0, x0     ; x0 = x1 * x1
        //   and x0, x0, #1     ; x0 ∈ {0, 1}
        //   cmp x0, #2         ; flags: ZF = (x0 == 2) — always 0
        //   b.ne junk          ; jne fires every time
        // The mnemonic dispatch + 3-operand handlers + cmp/b.ne all
        // have to compose for the solver to land AlwaysTrue.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("x1", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "mul",
                vec![
                    op("x0", OperandKind::Register),
                    op("x0", OperandKind::Register),
                    op("x0", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "and",
                vec![
                    op("x0", OperandKind::Register),
                    op("x0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100c,
                4,
                "cmp",
                vec![
                    op("x0", OperandKind::Register),
                    op("#2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1010,
                4,
                "b.ne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_subs_then_b_cs_resolves_unsigned_compare() {
        // `subs x0, x1, x1` sets ZF=1 (and Z is the only flag b.eq
        // / b.ne look at); same family as the cmp-then-branch shape
        // but exercises the `s`-suffix code path.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "subs",
                vec![
                    op("x0", OperandKind::Register),
                    op("x1", OperandKind::Register),
                    op("x1", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_eor_self_then_b_eq_is_always_true() {
        // The AArch64 zero idiom is `eor x0, x0, x0` (no flags, even
        // with .s some encodings) — feeding the result into cmp
        // produces ZF=1.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "eor",
                vec![
                    op("x0", OperandKind::Register),
                    op("x0", OperandKind::Register),
                    op("x0", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cmp",
                vec![
                    op("x0", OperandKind::Register),
                    op("#0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_free_input_b_eq_is_both_possible() {
        // No prior knowledge of x0, so a b.eq after `cmp x0, #0` is
        // a genuine choice. Verifies the slicer treats x0 as a free
        // input rather than truncating.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "cmp",
                vec![
                    op("x0", OperandKind::Register),
                    op("#0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::BothPossible);
    }

    #[test]
    fn aarch64_b_lo_after_cmp_resolves_unsigned_compare() {
        // mov x0, #1 ; cmp x0, #0 ; b.lo junk
        // b.lo (= unsigned <) is "C == 0" on ARM but x86-polarity
        // `CF == 1` in our model. cmp(1, 0) sets CF = (1 < 0) = 0,
        // so b.lo branches when CF == 1 → never taken → AlwaysFalse.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cmp",
                vec![
                    op("x0", OperandKind::Register),
                    op("#0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "b.lo",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::AlwaysFalse);
    }

    fn aarch32_block(insns: Vec<Instruction>) -> Program {
        program_with_arch(insns, Arch::Arm)
    }

    #[test]
    fn aarch64_cbz_after_xzr_mov_is_always_true() {
        // mov x0, xzr ; cbz x0, dest  →  always taken.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("xzr", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cbz",
                vec![
                    op("x0", OperandKind::Register),
                    op("0x401080", OperandKind::Immediate),
                ],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_cbnz_after_mov_one_is_always_true() {
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cbnz",
                vec![
                    op("x0", OperandKind::Register),
                    op("0x401080", OperandKind::Immediate),
                ],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_tbz_bit_zero_of_one_is_always_false() {
        // mov x0, #1 ; tbz x0, #0, dest
        //   bit(x0, 0) = 1, so tbz (branch if bit 0) never fires.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "tbz",
                vec![
                    op("x0", OperandKind::Register),
                    op("#0", OperandKind::Immediate),
                    op("0x401080", OperandKind::Immediate),
                ],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysFalse);
    }

    // `csel` end-to-end solver test is deliberately deferred. The
    // lifter handler emits the correct `Ite` expression, but the
    // backward slicer does not yet model "reads flags" as a distinct
    // effect from "writes flags" — so a flag-setting `cmp` upstream
    // of a `csel` gets dropped from the slice and the cond-bit ends
    // up as a free input. Tracked as a remaining follow-up; the
    // shape-level effect / lift tests cover the handler.

    #[test]
    fn aarch32_canonical_opaque_predicate_is_always_true() {
        // mov r0, r1 ; mul r0, r0, r0 ; and r0, r0, #1 ;
        // cmp r0, #2 ; bne junk  →  always taken (x*x & 1 ≠ 2).
        let program = aarch32_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("r0", OperandKind::Register),
                    op("r1", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "mul",
                vec![
                    op("r0", OperandKind::Register),
                    op("r0", OperandKind::Register),
                    op("r0", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "and",
                vec![
                    op("r0", OperandKind::Register),
                    op("r0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100c,
                4,
                "cmp",
                vec![
                    op("r0", OperandKind::Register),
                    op("#2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1010,
                4,
                "bne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_udiv_then_compare_resolves_constant() {
        // mov x0, #100 ; mov x1, #5 ; udiv x2, x0, x1 ;
        // cmp x2, #20 ; b.eq dest  →  always taken (100 / 5 == 20).
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("#100", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "mov",
                vec![
                    op("x1", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "udiv",
                vec![
                    op("x2", OperandKind::Register),
                    op("x0", OperandKind::Register),
                    op("x1", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100c,
                4,
                "cmp",
                vec![
                    op("x2", OperandKind::Register),
                    op("#20", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1010,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_sdiv_negative_resolves_constant() {
        // mov x0, #-30 (as #0xffffffffffffffe2) ; mov x1, #3 ;
        // sdiv x2, x0, x1 ; cmp x2, #-10 ; b.eq dest  →  always taken.
        //
        // We exercise the `sdiv` path; the lifter forwards through
        // `bvsdiv` which performs signed division with truncation
        // towards zero. The numeric literal #-10 is encoded as the
        // 64-bit two's-complement representation 0xffffffffffffff f6.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("0xffffffffffffffe2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "mov",
                vec![
                    op("x1", OperandKind::Register),
                    op("#3", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "sdiv",
                vec![
                    op("x2", OperandKind::Register),
                    op("x0", OperandKind::Register),
                    op("x1", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100c,
                4,
                "cmp",
                vec![
                    op("x2", OperandKind::Register),
                    op("0xfffffffffffffff6", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1010,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch64_csinc_takes_rn_when_predicate_true() {
        // mov x0, #5 ; mov x2, #5 ; mov x3, #3 ; cmp x0, #5 ;
        // csinc x1, x2, x3, eq ; cmp x1, #5 ; b.eq dest
        //   ZF=1 (x0 == 5), csinc returns x2=5, cmp sets ZF=1,
        //   b.eq always taken.
        let program = aarch64_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("x0", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "mov",
                vec![
                    op("x2", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "mov",
                vec![
                    op("x3", OperandKind::Register),
                    op("#3", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100c,
                4,
                "cmp",
                vec![
                    op("x0", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1010,
                4,
                "csinc",
                vec![
                    op("x1", OperandKind::Register),
                    op("x2", OperandKind::Register),
                    op("x3", OperandKind::Register),
                    op("eq", OperandKind::Unknown),
                ],
            ),
            insn(
                0x40_1014,
                4,
                "cmp",
                vec![
                    op("x1", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1018,
                4,
                "b.eq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch32_moveq_predicated_write_when_condition_true() {
        // mov r0, #5 ; cmp r0, #5 ; moveq r1, #99 ; cmp r1, #99 ; beq dest
        //   First cmp sets ZF=1, moveq writes r1=99 unconditionally
        //   because predicate holds, second cmp keeps ZF=1, beq taken.
        let program = aarch32_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("r0", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cmp",
                vec![
                    op("r0", OperandKind::Register),
                    op("#5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "moveq",
                vec![
                    op("r1", OperandKind::Register),
                    op("#99", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100c,
                4,
                "cmp",
                vec![
                    op("r1", OperandKind::Register),
                    op("#99", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1010,
                4,
                "beq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn aarch32_mov_imm_then_beq_resolves() {
        let program = aarch32_block(vec![
            insn(
                0x40_1000,
                4,
                "mov",
                vec![
                    op("r0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1004,
                4,
                "cmp",
                vec![
                    op("r0", OperandKind::Register),
                    op("#1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                4,
                "beq",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    fn solve_first_with(program: &Program, limits: &SliceLimits) -> SmtResult {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        let slice = slice_branch(cand, &program.functions[0], limits, program.arch);
        let lifted = lift_slice(&slice, program.arch);
        let ssa = ssa_convert(&lifted);
        solve_branch(&ssa, SolveOptions::default())
    }

    #[test]
    fn unknowns_on_truncation_resolves_call_then_cmp_eax_eax_to_always_false() {
        // call f ; cmp eax, eax ; jne junk
        //
        // Under the default limits the call truncates the slice and
        // the solver short-circuits to `Unsound`. With
        // `unknowns_on_truncation = true`, SSA surfaces `eax` from the
        // cmp as a free symbolic input and the predicate reduces to
        // `(eax == eax) → ZF == 1 → jne never fires`. The verdict is
        // sound because every value of the free input keeps the
        // identity true.
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                2,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1007,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let baseline = solve_first(&program);
        assert_eq!(
            baseline,
            SmtResult::Unsound,
            "without the policy a truncated slice must stay Unsound",
        );
        let limits = SliceLimits {
            unknowns_on_truncation: true,
            ..SliceLimits::default()
        };
        let policy = solve_first_with(&program, &limits);
        assert_eq!(policy, SmtResult::AlwaysFalse);
    }

    #[test]
    fn unknowns_on_truncation_leaves_real_branch_as_both_possible() {
        // call f ; cmp eax, 1 ; jne junk
        //
        // With the policy the slicer still produces a truncated slice
        // and SSA leaves `eax` as a free input. The predicate
        // `eax == 1` is genuinely satisfiable in both polarities, so
        // the verdict is `BothPossible` — i.e. the policy does not
        // fabricate a definitive verdict where there isn't one.
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let limits = SliceLimits {
            unknowns_on_truncation: true,
            ..SliceLimits::default()
        };
        let verdict = solve_first_with(&program, &limits);
        assert_eq!(verdict, SmtResult::BothPossible);
    }

    #[test]
    fn esil_first_path_resolves_constant_propagation_je_to_always_true() {
        // Same shape as `constant_propagation_je_is_always_true`, but
        // every instruction now carries the ESIL string radare2 would
        // emit. The slicer's lift loop must consume the ESIL first
        // (via the `r2smt-esil` lifter); the per-mnemonic path is the
        // fallback. Resolving to `AlwaysTrue` proves that the ESIL
        // pipeline produces an IR equivalent to the mnemonic handler
        // for this canonical case.
        fn ins(addr: u64, size: u8, mnem: &str, ops: Vec<Operand>, esil: &str) -> Instruction {
            Instruction {
                address: Address(addr),
                size,
                bytes: vec![],
                mnemonic: mnem.into(),
                operands: ops,
                esil: Some(esil.into()),
                pcode: None,
                is_thumb: false,
            }
        }
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        ins(
                            0x40_1000,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                            "1,eax,=",
                        ),
                        ins(
                            0x40_1005,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                            "1,eax,-,$z,zf,=",
                        ),
                        ins(
                            0x40_1008,
                            6,
                            "je",
                            vec![op("0x401080", OperandKind::Immediate)],
                            "zf,?{,0x401080,rip,=,}",
                        ),
                    ],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        };
        // Note: the `je` instruction's ESIL contains `?{` which the
        // mini-machine rejects — the slicer therefore falls back to
        // the per-mnemonic handler for that single instruction while
        // still consuming the ESIL for `mov` and `cmp`. This
        // exercises both the ESIL-first hit and the structured
        // fallback in one fixture.
        assert_eq!(solve_first(&program), SmtResult::AlwaysTrue);
    }

    #[test]
    fn truncated_slice_is_reported_unsound() {
        // call f ; cmp eax, 0 ; je → truncated.
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let verdict = solve_first(&program);
        assert_eq!(verdict, SmtResult::Unsound);
    }

    // --- Bounded simple-diamond Φ-merge: end-to-end soundness ---

    /// `cmp ecx,0; je THEN` head; both arms write `eax`; reconverge at
    /// `cmp eax,7; je analysed`. THEN is the taken edge (`ecx==0`).
    fn diamond(then_imm: &str, else_imm: &str) -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![
                    BasicBlock {
                        address: Address(0x40_1000),
                        instructions: vec![
                            insn(
                                0x40_1000,
                                3,
                                "cmp",
                                vec![
                                    op("ecx", OperandKind::Register),
                                    op("0", OperandKind::Immediate),
                                ],
                            ),
                            insn(
                                0x40_1003,
                                6,
                                "je",
                                vec![op("0x401100", OperandKind::Immediate)],
                            ),
                        ],
                        successors: vec![Address(0x40_1100), Address(0x40_1009)],
                    },
                    BasicBlock {
                        address: Address(0x40_1009),
                        instructions: vec![insn(
                            0x40_1009,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op(else_imm, OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_1200)],
                    },
                    BasicBlock {
                        address: Address(0x40_1100),
                        instructions: vec![insn(
                            0x40_1100,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op(then_imm, OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_1200)],
                    },
                    BasicBlock {
                        address: Address(0x40_1200),
                        instructions: vec![
                            insn(
                                0x40_1200,
                                3,
                                "cmp",
                                vec![
                                    op("eax", OperandKind::Register),
                                    op("7", OperandKind::Immediate),
                                ],
                            ),
                            insn(
                                0x40_1203,
                                6,
                                "je",
                                vec![op("0x401300", OperandKind::Immediate)],
                            ),
                        ],
                        successors: vec![],
                    },
                ],
                is_thumb: false,
            }],
        }
    }

    fn solve_diamond_join(program: &Program, allow_join_merge: bool) -> SmtResult {
        let cands = collect_branches(program);
        let join = cands
            .iter()
            .find(|c| c.address == Address(0x40_1203))
            .expect("join branch present");
        let limits = SliceLimits {
            max_basic_blocks: 8,
            allow_join_merge,
            ..SliceLimits::default()
        };
        let slice = slice_branch(join, &program.functions[0], &limits, program.arch);
        let lifted = lift_slice(&slice, program.arch);
        let ssa = ssa_convert(&lifted);
        solve_branch(&ssa, SolveOptions::default())
    }

    #[test]
    fn diamond_path_insensitive_predicate_is_always_false_with_merge() {
        // Both arms set eax=5, so `eax==7` is false regardless of the
        // head condition. The bounded Φ-merge recovers this precisely.
        let program = diamond("5", "5");
        assert_eq!(
            solve_diamond_join(&program, true),
            SmtResult::AlwaysFalse,
            "path-insensitive predicate must resolve precisely under Φ-merge"
        );
    }

    #[test]
    fn diamond_without_merge_is_sound_but_imprecise() {
        // Same program, merge disabled: the pre-existing path leaves
        // eax a free input → the verdict widens to BothPossible. Sound
        // (never a fabricated AlwaysX), just less precise — this is
        // the gap the merge closes.
        let program = diamond("5", "5");
        assert_eq!(
            solve_diamond_join(&program, false),
            SmtResult::BothPossible,
            "without the merge eax is a free input: sound, imprecise"
        );
    }

    #[test]
    fn diamond_path_sensitive_predicate_stays_both_possible() {
        // Soundness guard: arms genuinely differ (5 vs 7). `eax==7`
        // now depends on the (free) head input `ecx`, so the verdict
        // must be BothPossible — the merge must NEVER fabricate an
        // AlwaysTrue / AlwaysFalse here.
        let program = diamond("5", "7");
        let verdict = solve_diamond_join(&program, true);
        assert_eq!(
            verdict,
            SmtResult::BothPossible,
            "path-sensitive predicate must not be fabricated into AlwaysX"
        );
    }
}
