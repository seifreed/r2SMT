//! The `AArch64` mnemonics whose second source is complemented, and the
//! two-operand spellings of a three-operand instruction.
//!
//! `bic` / `bics` / `orn` / `eon` invert `Operand2` and then combine;
//! `mvn Rd, Op` is `orn Rd, xzr, Op`, `neg` / `negs Rd, Op` is `sub` /
//! `subs Rd, xzr, Op`, and `cmn Rn, Op` is `adds xzr, Rn, Op`. All of
//! them used to fall to the effect table's catch-all, which truncates
//! the slice — sound, but every branch downstream became unknown.
//!
//! Each contract binds concrete sources and checks the destination
//! against a hand-computed A64 value, because the failure mode of a
//! complement in the wrong place is a wrong number and not a decline.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceStatus, lift_branch_condition,
    lift_per_mnemonic,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
const X: u16 = 64;

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn branch(condition: BranchCondition) -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "b.cond".to_string(),
        condition,
        formula: "b.cond".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with('x') || raw.starts_with('w') {
            OperandKind::Register
        } else {
            OperandKind::Immediate
        },
    }
}

fn insn(mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|raw| operand(raw)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

fn lift(mnemonic: &str, operands: &[&str], bindings: &[(&str, u128, u16)]) -> Vec<IrStmt> {
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    let lifted = lift_per_mnemonic(&insn(mnemonic, operands), Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} declined: {lifted:?}"
    );
    statements.extend(lifted);
    statements
}

fn finish(statements: Vec<IrStmt>, condition: Expr) -> SmtResult {
    let slice = LiftedSlice {
        branch: branch(BranchCondition::NotEqual),
        statements,
        condition,
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Aarch64,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

/// Lift one instruction with its sources bound and ask whether the
/// named register is necessarily `expected`.
fn solve_value(
    mnemonic: &str,
    operands: &[&str],
    bindings: &[(&str, u128, u16)],
    observed: &str,
    expected: u128,
) -> SmtResult {
    finish(
        lift(mnemonic, operands, bindings),
        Expr::eq(Expr::var(observed, X), Expr::konst(expected, X)),
    )
}

/// The same, but asking the branch that reads the flags instead of a
/// register — the only way to see a flag-setting form's real answer.
fn solve_branch_after(
    mnemonic: &str,
    operands: &[&str],
    bindings: &[(&str, u128, u16)],
    condition: BranchCondition,
) -> SmtResult {
    let candidate = branch(condition);
    let statements = lift(mnemonic, operands, bindings);
    let cond = lift_branch_condition(&candidate, Arch::Aarch64);
    finish(statements, cond)
}

#[test]
fn bic_clears_the_bits_its_second_source_names() {
    // `bic` is `Rn AND NOT Rm`, not `NOT (Rn AND Rm)`. On 0xff & ~0x0f
    // the right answer is 0xf0; the negated-conjunction reading gives
    // 0xffff_ffff_ffff_fff0.
    assert_eq!(
        solve_value(
            "bic",
            &["x0", "x1", "x2"],
            &[("x1", 0xff, X), ("x2", 0x0f, X)],
            "x0",
            0xf0,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn bic_complements_its_second_source_after_shifting_it() {
    // `bic x0, x1, x2, lsl 4` clears the *shifted* bits. With x2 = 1 the
    // shift makes 0x10, so 0xff loses bit 4 and keeps the rest: 0xef.
    // Complementing before the shift would clear a different set.
    assert_eq!(
        solve_value(
            "bic",
            &["x0", "x1", "x2", "lsl 4"],
            &[("x1", 0xff, X), ("x2", 1, X)],
            "x0",
            0xef,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn orn_sets_every_bit_its_second_source_leaves_clear() {
    // 0x0f | ~0xff = 0x0f | 0xffff_ffff_ffff_ff00.
    assert_eq!(
        solve_value(
            "orn",
            &["x0", "x1", "x2"],
            &[("x1", 0x0f, X), ("x2", 0xff, X)],
            "x0",
            0xffff_ffff_ffff_ff0f,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn eon_is_the_complement_of_exclusive_or() {
    // 0xf0 ^ ~0x0f = 0xf0 ^ 0xffff_ffff_ffff_fff0 = 0xffff_ffff_ffff_ff00.
    assert_eq!(
        solve_value(
            "eon",
            &["x0", "x1", "x2"],
            &[("x1", 0xf0, X), ("x2", 0x0f, X)],
            "x0",
            0xffff_ffff_ffff_ff00,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn mvn_complements_every_bit_of_its_source() {
    assert_eq!(
        solve_value(
            "mvn",
            &["x0", "x1"],
            &[("x1", 0x0f, X)],
            "x0",
            u128::from(!0x0f_u64)
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn neg_subtracts_its_source_from_zero() {
    assert_eq!(
        solve_value(
            "neg",
            &["x0", "x1"],
            &[("x1", 5, X)],
            "x0",
            u128::from(5_u64.wrapping_neg())
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_w_form_complement_leaves_the_upper_half_of_the_parent_clear() {
    // `mvn w0, w1` writes a 32-bit result and the ABI zero-extends it,
    // so the complement must not reach bit 32 and above. Computing the
    // complement at the parent's width instead would set all of them.
    assert_eq!(
        solve_value("mvn", &["w0", "w1"], &[("x1", 0x0f, X)], "x0", 0xffff_fff0),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn cmn_compares_against_the_negation_of_its_second_source() {
    // `cmn x1, x2` sets the flags from `x1 + x2`, so `5 + (-5)` is zero
    // and `b.eq` is taken. Reading it as a subtraction would not be.
    assert_eq!(
        solve_branch_after(
            "cmn",
            &["x1", "x2"],
            &[("x1", 5, X), ("x2", u128::from(5_u64.wrapping_neg()), X)],
            BranchCondition::Equal,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn negs_leaves_the_precise_carry_a_subtract_leaves() {
    // `negs x0, x1` is `subs x0, xzr, x1`, so `0 - 1` borrows and ARM
    // clears C — which makes `b.lo` (C == 0) taken. What the contract is
    // really for is that the answer is *definite*: routing `neg` through
    // the flag helper's catch-all arm instead of its `Sub` one would
    // leave the carry Unknown and answer `BothPossible` here.
    assert_eq!(
        solve_branch_after(
            "negs",
            &["x0", "x1"],
            &[("x1", 1, X)],
            BranchCondition::Below,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn bics_leaves_the_carry_an_ands_leaves() {
    // `bics` is a logical flag-setter, so it clears the architectural C
    // exactly as `ands` does — and therefore stores the inverted bit as
    // one, making `b.hs` (C == 1) unreachable. Sharing `emit_arith3` is
    // what keeps the two on one answer.
    assert_eq!(
        solve_branch_after(
            "bics",
            &["x0", "x1", "x2"],
            &[("x1", 0xff, X), ("x2", 0x0f, X)],
            BranchCondition::AboveOrEqual,
        ),
        SmtResult::AlwaysFalse,
    );
}

#[test]
fn bics_sets_the_zero_flag_from_the_masked_result() {
    // Every bit of x1 is cleared by x2, so the result is zero and
    // `b.eq` is taken. Pins that the flags come from the complemented
    // combination and not from the raw sources.
    assert_eq!(
        solve_branch_after(
            "bics",
            &["x0", "x1", "x2"],
            &[("x1", 0x0f, X), ("x2", 0xff, X)],
            BranchCondition::Equal,
        ),
        SmtResult::AlwaysTrue,
    );
}
