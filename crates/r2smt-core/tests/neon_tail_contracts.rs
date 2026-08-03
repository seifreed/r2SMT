//! `AArch64` NEON contracts for the families that resolve their
//! geometry from something other than operand 0.
//!
//! These are the lowerings whose failure mode is a *wrong value* rather
//! than a decline: an across-lane reduction that reads the destination's
//! width as the lane width, a by-element form that picks the wrong lane,
//! a NaN-aware select that reuses x86's `MAXPS` ordering. None of that
//! shows up in a structural assertion on the emitted IR, so each test
//! here binds concrete lanes, lifts the real instruction, and solves the
//! destination against a value computed by hand from the ARM definition.
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
        mnemonic: "neontail".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "neontail".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands`, bind every named register to a concrete
/// vector value, and ask the solver whether the destination is
/// necessarily `expected`.
///
/// The binding assignments run through the real SSA pass ahead of the
/// lifted statements, so an instruction that *reads* its destination
/// sees the bound value rather than contradicting it.
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
    statements.extend(lifted);

    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::Var(Var::new("v0", VECTOR_BITS)),
            Expr::konst(expected, VECTOR_BITS),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Aarch64,
    };
    solve_branch(
        &ssa_convert(&slice),
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

// ===================== across-lane reductions =====================

/// Pack `lanes` little-endian into one vector value.
fn packed(lane_bits: u32, lanes: &[u128]) -> u128 {
    let mut value = 0u128;
    let mut offset = 0u32;
    for lane in lanes {
        value |= lane << offset;
        offset += lane_bits;
    }
    value
}

#[test]
fn addv_sums_every_lane_of_the_source_arrangement() {
    // 1 + 2 + 3 + 4. The destination is `s0`, so a resolver that took
    // the lane width from operand 0 would read one 32-bit lane and stop.
    assert_computes(
        "addv",
        &["s0", "v1.4s"],
        &[("v1", packed(32, &[1, 2, 3, 4]))],
        10,
    );
}

#[test]
fn addv_keeps_the_low_bits_of_an_overflowing_sum() {
    // 0xff + 0x02 is 0x101 in eight bits, and ARM truncates the
    // unbounded sum into the element — a widened result would be wrong.
    assert_computes(
        "addv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0xff, 0x02]))],
        0x01,
    );
}

#[test]
fn uaddlv_widens_before_summing() {
    // Eight lanes of 0xff sum to 2 040, which does not fit the source
    // element. The `l` in the mnemonic is exactly that: the destination
    // is twice as wide, and the sum is exact rather than truncated.
    assert_computes(
        "uaddlv",
        &["h0", "v1.8b"],
        &[("v1", packed(8, &[0xff; 8]))],
        0x7f8,
    );
}

#[test]
fn saddlv_sign_extends_each_lane_before_summing() {
    // One lane of 0xff is -1 signed, so the 16-bit sum is 0xffff. Read
    // unsigned it would be 0xff — the same bits, a different number.
    assert_computes(
        "saddlv",
        &["h0", "v1.8b"],
        &[("v1", packed(8, &[0xff]))],
        0xffff,
    );
}

#[test]
fn smaxv_compares_lanes_signed() {
    // 0x80 is -128, so the signed maximum of the two non-zero lanes is
    // 0x7f — and every other lane is zero, which does not beat it.
    assert_computes(
        "smaxv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x7f,
    );
}

#[test]
fn umaxv_compares_the_same_lanes_unsigned() {
    assert_computes(
        "umaxv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x80,
    );
}

#[test]
fn sminv_compares_lanes_signed() {
    assert_computes(
        "sminv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x80,
    );
}

#[test]
fn uminv_compares_the_same_lanes_unsigned() {
    // The six zero lanes are the unsigned minimum, which is what makes
    // this the mirror of `sminv` rather than a restatement of it.
    assert_computes(
        "uminv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x00,
    );
}
