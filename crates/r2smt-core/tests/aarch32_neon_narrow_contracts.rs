//! `AArch32` NEON narrowing-shift contracts, solved rather than
//! asserted structurally.
//!
//! These lowerings fail by producing a *wrong value*, not by declining,
//! so the only evidence that means anything is a solver agreeing the
//! destination equals a hand-computed ARM result. The saturating
//! rounding narrows are the sharpest case: the rounding term is added
//! before the shift, and at the source's own width that addition can
//! carry into the sign bit — `0x7fff + 8` in sixteen bits is negative,
//! which turns a saturation at the top of the range into one at the
//! bottom. Computing one bit wider is what makes them right.
//!
//! Until now `AArch32` NEON had only structural coverage
//! (`aarch32_neon_contracts.rs` asserts on declines), so this is the
//! first solver-backed file on that ISA.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceStatus, lift_per_mnemonic,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
const VECTOR_BITS: u16 = 128;

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with('#') {
            OperandKind::Immediate
        } else {
            OperandKind::Register
        },
    }
}

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "narrowtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "narrowtest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands` on `Arch::Arm`, bind every named register to
/// a concrete vector value, and ask the solver whether the destination
/// is necessarily `expected`.
///
/// `AArch32` names a `d` register as half of the synthetic 128-bit
/// parent, so the bindings and the expected value are both stated
/// against the parent (`q0` is `v0`, `d0` its low half).
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
    let lifted = lift_per_mnemonic(&insn, Arch::Arm);
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
    statements.extend(lifted);

    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::extract(Expr::Var(Var::new("v0", VECTOR_BITS)), 63, 0),
            Expr::konst(expected, 64),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Arm,
    };
    solve_branch(
        &ssa_convert(&slice),
        SolveOptions {
            timeout_ms: TEST_SOLVE_TIMEOUT_MS,
            ..SolveOptions::default()
        },
    )
}

/// Assert the lowering computes exactly `expected` in `d0`.
fn assert_computes(mnemonic: &str, operands: &[&str], sources: &[(&str, u128)], expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

/// Pack `values` into consecutive `bits`-wide lanes, little-endian.
fn packed(bits: u16, values: &[u128]) -> u128 {
    values
        .iter()
        .enumerate()
        .fold(0, |acc, (i, v)| acc | (v << (usize::from(bits) * i)))
}

const NARROW: [&str; 2] = ["d0", "q1"];

#[test]
fn vrshrn_rounds_before_narrowing() {
    // (255 + 8) >> 4 = 16. Truncating instead would give 15.
    assert_computes(
        "vrshrn.i16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0xff)],
        0x10,
    );
}

#[test]
fn vshrn_still_truncates() {
    // The companion direction: without rounding the same input gives 15,
    // so the two forms are genuinely distinguished.
    assert_computes(
        "vshrn.i16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0xff)],
        0x0f,
    );
}

#[test]
fn vqrshrn_saturates_after_rounding_and_shifting() {
    // (0x7fff + 8) >> 4 = 0x800, far outside a signed byte, so the lane
    // clamps at 0x7f. Adding the rounding term at the source's own
    // width would carry into the sign bit and clamp at 0x80 instead.
    assert_computes(
        "vqrshrn.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0x7fff)],
        0x7f,
    );
}

#[test]
fn vqshrn_saturates_without_rounding() {
    // 0x7fff >> 4 = 0x7ff, still outside a signed byte.
    assert_computes(
        "vqshrn.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0x7fff)],
        0x7f,
    );
}

#[test]
fn vqshrn_signed_source_saturates_at_the_negative_end() {
    // -32768 >> 4 = -2048, below a signed byte's -128.
    assert_computes(
        "vqshrn.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0x8000)],
        0x80,
    );
}

#[test]
fn vqshrun_clamps_a_negative_source_to_zero() {
    // `vqshrun` reads a signed source and saturates into the *unsigned*
    // destination range, so anything negative lands at zero rather than
    // wrapping to 0xff.
    assert_computes(
        "vqshrun.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0x8000)],
        0x00,
    );
}

#[test]
fn vqrshrun_rounds_then_clamps_into_the_unsigned_range() {
    // (0x7fff + 8) >> 4 = 0x800, above an unsigned byte's 0xff.
    assert_computes(
        "vqrshrun.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0x7fff)],
        0xff,
    );
}

#[test]
fn vqshrn_unsigned_source_does_not_drag_a_sign_bit_down() {
    // 0xff00 read unsigned is 65280; >> 4 is 4080, clamped to 0xff.
    // A signed read would make it negative and clamp to zero.
    assert_computes(
        "vqshrn.u16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0xff00)],
        0xff,
    );
}

#[test]
fn vqrshrn_leaves_an_in_range_lane_alone() {
    // (0x0100 + 8) >> 4 = 16, comfortably inside a signed byte, so the
    // clamp must not fire.
    assert_computes(
        "vqrshrn.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", 0x0100)],
        0x10,
    );
}

#[test]
fn vqrshrn_narrows_every_lane_not_just_the_first() {
    // Four halfword lanes into four bytes: 0x7fff saturates, the rest
    // round and shift normally.
    assert_computes(
        "vqrshrn.s16",
        &[NARROW[0], NARROW[1], "#4"],
        &[("v1", packed(16, &[0x7fff, 0x0100, 0x00ff, 0x0000]))],
        packed(8, &[0x7f, 0x10, 0x10, 0x00]),
    );
}
