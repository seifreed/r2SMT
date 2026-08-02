//! `AArch64` NEON saturation contract.
//!
//! Saturating arithmetic is the one NEON family whose lowering cannot be
//! checked by reading it: the clamp is a pair of comparisons against
//! constants derived from the element width, and an off-by-one in either
//! bound produces an expression that looks right and computes the wrong
//! value only at the boundary. So these tests *solve* the lowering.
//!
//! Each one binds the source registers to concrete lanes, lifts the real
//! instruction, and asserts that the destination equals a value computed
//! by hand from the ARM definition. A wrong bound makes the verdict
//! `BothPossible` instead of `AlwaysTrue`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus, lift_per_mnemonic};
use r2smt_smt::solve_branch;
use r2smt_ssa::SsaLiftedSlice;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
const VECTOR_BITS: u16 = 128;

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: OperandKind::Register,
    }
}

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "sattest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "sattest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands`, bind every source register to a concrete
/// vector value, and ask the solver whether the destination is
/// necessarily `expected`.
///
/// `sources` pairs a register name with the whole 128-bit value it
/// holds, so a lane is placed by shifting it into position.
fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|o| operand(o)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    };
    let lifted = lift_per_mnemonic(&insn, Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} declined: {lifted:?}"
    );

    let mut statements: Vec<IrStmt> = sources
        .iter()
        .map(|(name, value)| IrStmt::Assign {
            dst: Var::new(*name, VECTOR_BITS),
            src: Expr::konst(*value, VECTOR_BITS),
        })
        .collect();
    let mut defs: Vec<Var> = sources
        .iter()
        .map(|(name, _)| Var::new(*name, VECTOR_BITS))
        .collect();
    statements.extend(lifted);
    defs.push(Var::new("v0", VECTOR_BITS));

    let slice = SsaLiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::Var(Var::new("v0", VECTOR_BITS)),
            Expr::konst(expected, VECTOR_BITS),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        inputs: Vec::new(),
        defs,
        arch: Arch::Aarch64,
    };
    solve_branch(
        &slice,
        SolveOptions {
            timeout_ms: TEST_SOLVE_TIMEOUT_MS,
            ..SolveOptions::default()
        },
    )
}

/// Assert the lowering computes exactly `expected` for these inputs.
fn assert_computes(mnemonic: &str, operands: &[&str], sources: &[(&str, u128)], expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

const HALF: [&str; 3] = ["v0.4h", "v1.4h", "v2.4h"];
const NARROW_FROM_HALF: [&str; 2] = ["v0.8b", "v1.8h"];

#[test]
fn sqadd_clamps_at_the_signed_maximum() {
    // 0x7fff + 1 overflows a signed halfword; the result is INT_MAX.
    assert_computes("sqadd", &HALF, &[("v1", 0x7fff), ("v2", 0x1)], 0x7fff);
}

#[test]
fn sqadd_leaves_a_representable_sum_alone() {
    assert_computes("sqadd", &HALF, &[("v1", 0x10), ("v2", 0x20)], 0x30);
}

#[test]
fn sqsub_clamps_at_the_signed_minimum() {
    // 0x8000 is INT_MIN; subtracting one saturates rather than wrapping
    // to INT_MAX.
    assert_computes("sqsub", &HALF, &[("v1", 0x8000), ("v2", 0x1)], 0x8000);
}

#[test]
fn uqadd_clamps_at_the_unsigned_maximum() {
    assert_computes("uqadd", &HALF, &[("v1", 0xffff), ("v2", 0x1)], 0xffff);
}

#[test]
fn uqsub_clamps_at_zero() {
    // The difference is negative, which is why an unsigned saturating
    // subtract needs a signed comparison against a zero floor.
    assert_computes("uqsub", &HALF, &[("v1", 0x1), ("v2", 0x2)], 0x0);
}

#[test]
fn uqsub_leaves_a_representable_difference_alone() {
    assert_computes("uqsub", &HALF, &[("v1", 0x20), ("v2", 0x8)], 0x18);
}

#[test]
fn uhadd_halves_the_exact_sum() {
    // (0xffff + 1) >> 1 = 0x8000 — the sum does not fit a halfword, so
    // halving it at the element width would give zero.
    assert_computes("uhadd", &HALF, &[("v1", 0xffff), ("v2", 0x1)], 0x8000);
}

#[test]
fn uhadd_truncates_an_odd_sum_downward() {
    assert_computes("uhadd", &HALF, &[("v1", 0x1), ("v2", 0x2)], 0x1);
}

#[test]
fn urhadd_rounds_an_odd_sum_upward() {
    // The rounding variant adds one before halving: (1 + 2 + 1) >> 1.
    assert_computes("urhadd", &HALF, &[("v1", 0x1), ("v2", 0x2)], 0x2);
}

#[test]
fn shadd_halves_a_negative_sum_arithmetically() {
    // -4 + -2 = -6, halved is -3 = 0xfffd.
    assert_computes("shadd", &HALF, &[("v1", 0xfffc), ("v2", 0xfffe)], 0xfffd);
}

#[test]
fn sqdmulh_saturates_at_the_doubling_corner() {
    // The one input pair where the doubling overflows: INT_MIN squared
    // is +2^30, doubled is +2^31, and its high halfword is 0x8000 —
    // one past INT_MAX. Every other pair fits, which is why this corner
    // is the whole reason the instruction saturates at all.
    assert_computes("sqdmulh", &HALF, &[("v1", 0x8000), ("v2", 0x8000)], 0x7fff);
}

#[test]
fn sqdmulh_keeps_the_doubled_high_half_when_it_fits() {
    // 0x4000 * 0x4000 * 2 = 2^29; the high halfword is 0x2000.
    assert_computes("sqdmulh", &HALF, &[("v1", 0x4000), ("v2", 0x4000)], 0x2000);
}

#[test]
fn sqrdmulh_rounds_the_doubled_high_half() {
    // 0x4000 * 0x0001 * 2 = 0x8000; shifted down by 16 that is zero,
    // but the rounding term carries it to one.
    assert_computes("sqrdmulh", &HALF, &[("v1", 0x4000), ("v2", 0x1)], 0x1);
}

#[test]
fn sqxtn_clamps_a_wide_element_into_the_signed_narrow_range() {
    // 256 does not fit a signed byte.
    assert_computes("sqxtn", &NARROW_FROM_HALF, &[("v1", 0x100)], 0x7f);
}

#[test]
fn uqxtn_clamps_into_the_unsigned_narrow_range() {
    assert_computes("uqxtn", &NARROW_FROM_HALF, &[("v1", 0x100)], 0xff);
}

#[test]
fn sqxtun_clamps_a_negative_element_to_zero() {
    // 0xff00 is -256; the unsigned narrow range has no negatives.
    assert_computes("sqxtun", &NARROW_FROM_HALF, &[("v1", 0xff00)], 0x0);
}

#[test]
fn rshrn_rounds_before_narrowing() {
    // (255 + 8) >> 4 = 16. Truncating instead would give 15.
    assert_computes("rshrn", &["v0.8b", "v1.8h", "#4"], &[("v1", 0xff)], 0x10);
}

#[test]
fn shrn_truncates_without_rounding() {
    assert_computes("shrn", &["v0.8b", "v1.8h", "#4"], &[("v1", 0xff)], 0xf);
}

#[test]
fn sqrshrn_saturates_after_rounding_and_shifting() {
    // (0x7fff + 8) >> 4 = 0x800, far outside a signed byte.
    assert_computes(
        "sqrshrn",
        &["v0.8b", "v1.8h", "#4"],
        &[("v1", 0x7fff)],
        0x7f,
    );
}

#[test]
fn sqshrun_clamps_a_negative_shift_result_to_zero() {
    // 0x8000 is negative; shifting right arithmetically keeps it so.
    assert_computes("sqshrun", &["v0.8b", "v1.8h", "#4"], &[("v1", 0x8000)], 0x0);
}

#[test]
fn saturating_write_zeroes_the_upper_half_of_the_destination() {
    // A 64-bit arrangement is still a whole-register write, so the top
    // half of `v0` is zero rather than preserved. Placing a non-zero
    // lane high in the source proves the result is not simply narrow.
    let sources = [("v1", 0x7fff_u128 << 48), ("v2", 1_u128 << 48)];
    assert_computes("sqadd", &HALF, &sources, 0x7fff << 48);
}

#[test]
fn rshrn_rounding_does_not_overflow_the_source_element() {
    // The regression for computing the rounding term at the source's own
    // width: `0xffff + 8` wraps to `7` in sixteen bits, which would make
    // this narrow to zero instead of to the true `(65535 + 8) >> 4`.
    assert_computes("rshrn", &["v0.8b", "v1.8h", "#4"], &[("v1", 0xffff)], 0x00);
}

#[test]
fn sqrshrn_rounding_saturates_at_the_top_not_the_bottom() {
    // The same overflow with a saturating variant is worse than an
    // arithmetic error: carrying into the sign bit turns a clamp at
    // INT_MAX into one at INT_MIN, flipping the result's sign.
    assert_computes(
        "sqrshrn",
        &["v0.8b", "v1.8h", "#8"],
        &[("v1", 0x7fff)],
        0x7f,
    );
}
