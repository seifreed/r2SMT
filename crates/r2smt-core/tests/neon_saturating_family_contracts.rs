//! `AArch64` NEON contracts for the saturating tail of the family.
//!
//! Every lowering here fails as a *wrong value* rather than as a
//! decline, so a structural assertion on the emitted IR would pass over
//! all of it. Each test binds concrete lanes, lifts the real
//! instruction, and solves the destination against a value computed by
//! hand from the ARM definition.
//!
//! The hazards the file exists for. `sqabs` / `sqneg` clamp where `abs`
//! and `neg` wrap, so `sqabs` of the most negative element is `INT_MAX`
//! and not itself. The shift inserts read their destination, and which
//! of its bits survive is decided by the shift alone — an inverted mask
//! keeps exactly the wrong half. `suqadd` and `usqadd` add operands of
//! *different* signedness and clamp into the destination's range, which
//! is neither operand's. A saturating left shift overflows whenever
//! shifting the result back does not restore the source, which is the
//! only formulation that also covers an amount past the element width.
//! And the rounding shifts add half an ulp one bit wider than the
//! source, because at the source's own width that carry reaches the
//! sign bit and turns a saturation at `INT_MAX` into one at `INT_MIN`.
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
        mnemonic: "neonsatfamily".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "neonsatfamily".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn instruction(mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|o| operand(o)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Lift `mnemonic operands`, bind every named register to a concrete
/// vector value, and ask the solver whether `v0` is necessarily
/// `expected`.
///
/// The bindings run through the real SSA pass ahead of the lifted
/// statements, so an instruction that reads its destination sees the
/// bound value rather than contradicting it.
fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = instruction(mnemonic, operands);
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

/// Assert the lifter still refuses this instruction — the boundary the
/// families here stop at.
fn assert_declines(mnemonic: &str, operands: &[&str]) {
    let lifted = lift_per_mnemonic(&instruction(mnemonic, operands), Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} should still decline, got {lifted:?}"
    );
}

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

// ===================== the unsigned estimates =====================

#[test]
fn urecpe_leaves_the_destination_free_rather_than_computing_a_reciprocal() {
    // The anti-fabrication half. `urecpe` of `0x8000_0000` is
    // `0xffff_ffff` by the architecture's table, and a lowering that
    // produced it would make this `AlwaysTrue`. The estimate is a free
    // input instead, so that answer stays *possible* and is not
    // asserted.
    assert_eq!(
        solve_lowering(
            "urecpe",
            &["v0.4s", "v1.4s"],
            &[("v1", packed(32, &[0x8000_0000; 4]))],
            packed(32, &[0xffff_ffff; 4]),
        ),
        SmtResult::BothPossible,
    );
}

#[test]
fn ursqrte_keeps_the_slice_complete_rather_than_truncating() {
    // The other half: a decline would have left the destination
    // undefined and the verdict `Unsound`. A verdict ranging over every
    // value the estimate could take is worth more than no verdict.
    assert_ne!(
        solve_lowering(
            "ursqrte",
            &["v0.2s", "v1.2s"],
            &[("v1", packed(32, &[0x4000_0000, 0]))],
            packed(32, &[0xffff_ffff, 0]),
        ),
        SmtResult::Unsound,
    );
}

#[test]
fn the_unsigned_estimates_decline_a_lane_the_architecture_does_not_encode() {
    // `urecpe` / `ursqrte` are encoded for a 32-bit element only, where
    // their float cousins reach 16 and 64 as well.
    assert_declines("urecpe", &["v0.8h", "v1.8h"]);
    assert_declines("ursqrte", &["v0.2d", "v1.2d"]);
}

// ===================== the saturating unary forms =====================

#[test]
fn sqabs_of_the_most_negative_element_saturates_at_the_signed_maximum() {
    // The whole reason `sqabs` is not resolved through `abs`: negating
    // `-128` at eight bits wraps back onto `-128`, so the wrapping
    // family would answer `0x80` where the architecture answers `0x7f`.
    assert_computes(
        "sqabs",
        &["v0.8b", "v1.8b"],
        &[("v1", packed(8, &[0x80, 0x7f, 0xff]))],
        packed(8, &[0x7f, 0x7f, 0x01]),
    );
}

#[test]
fn sqneg_of_the_most_negative_element_saturates_at_the_signed_maximum() {
    assert_computes(
        "sqneg",
        &["v0.4h", "v1.4h"],
        &[("v1", packed(16, &[0x8000, 0x0001, 0xffff]))],
        packed(16, &[0x7fff, 0xffff, 0x0001]),
    );
}

#[test]
fn sqabs_leaves_a_representable_magnitude_alone() {
    // The negative arm has to negate and the positive arm must not, so
    // a lowering that negated unconditionally would fail here and pass
    // the saturation test above.
    assert_computes(
        "sqabs",
        &["v0.4s", "v1.4s"],
        &[("v1", packed(32, &[0x0000_0010, 0xffff_fff0, 0, 0]))],
        packed(32, &[0x0000_0010, 0x0000_0010, 0, 0]),
    );
}

// ===================== the shift inserts =====================

#[test]
fn sli_keeps_the_destination_bits_the_shift_vacated() {
    // `sli #4` moves the source's low nibble up and leaves the
    // destination's low nibble in place. A mask over the *high* nibble
    // instead would keep `0xa0` and drop `0x0a`.
    assert_computes(
        "sli",
        &["v0.8b", "v1.8b", "#4"],
        &[
            ("v0", packed(8, &[0xaa, 0xaa])),
            ("v1", packed(8, &[0x35, 0xff])),
        ],
        packed(8, &[0x5a, 0xfa]),
    );
}

#[test]
fn sri_keeps_the_destination_bits_above_the_shifted_source() {
    // `sri #4` moves the source's high nibble down and leaves the
    // destination's high nibble in place.
    assert_computes(
        "sri",
        &["v0.8b", "v1.8b", "#4"],
        &[
            ("v0", packed(8, &[0xaa, 0xaa])),
            ("v1", packed(8, &[0x35, 0xff])),
        ],
        packed(8, &[0xa3, 0xaf]),
    );
}

#[test]
fn sli_by_zero_is_the_source_and_keeps_no_destination_bit() {
    // The low end of `sli`'s encodable range. A mask helper that
    // declined a zero width would make this a decline instead of the
    // copy the architecture defines.
    assert_computes(
        "sli",
        &["v0.4h", "v1.4h", "#0"],
        &[("v0", packed(16, &[0xffff])), ("v1", packed(16, &[0x1234]))],
        packed(16, &[0x1234]),
    );
}

#[test]
fn sri_by_the_whole_element_keeps_the_destination_and_no_source_bit() {
    // The high end of `sri`'s encodable range, which is the element
    // width itself and not one below it.
    assert_computes(
        "sri",
        &["v0.4h", "v1.4h", "#16"],
        &[("v0", packed(16, &[0xabcd])), ("v1", packed(16, &[0x1234]))],
        packed(16, &[0xabcd]),
    );
}

#[test]
fn the_shift_inserts_decline_an_amount_outside_the_encoding() {
    // `sli` stops one below the element width and `sri` starts at one,
    // so each rejects the amount the other accepts at its own end.
    assert_declines("sli", &["v0.8b", "v1.8b", "#8"]);
    assert_declines("sri", &["v0.8b", "v1.8b", "#0"]);
}
