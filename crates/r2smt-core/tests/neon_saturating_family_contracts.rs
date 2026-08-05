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

// ===================== the saturating left shifts =====================

#[test]
fn sqshl_by_an_immediate_clamps_towards_the_sign_it_is_heading_for() {
    // `0x40 << 1` is 128, one past the signed maximum, so it clamps up;
    // `0xc0 << 1` is exactly `-128` and does not clamp at all; `0x80`
    // is already the minimum and clamps down. A lowering that clamped
    // in one direction only would get two of these three right.
    assert_computes(
        "sqshl",
        &["v0.8b", "v1.8b", "#1"],
        &[("v1", packed(8, &[0x40, 0xc0, 0x80, 0x10]))],
        packed(8, &[0x7f, 0x80, 0x80, 0x20]),
    );
}

#[test]
fn uqshl_reads_the_element_unsigned_where_sqshl_reads_it_signed() {
    // The same `0x40 << 1` that overflows the signed range is an
    // ordinary 128 here, so a lowering that shared one signedness
    // between the two mnemonics would answer `0x7f` for the middle
    // lane.
    assert_computes(
        "uqshl",
        &["v0.8b", "v1.8b", "#1"],
        &[("v1", packed(8, &[0xff, 0x40, 0x80]))],
        packed(8, &[0xff, 0x80, 0xff]),
    );
}

#[test]
fn sqshl_clamps_at_a_wider_element_too() {
    assert_computes(
        "sqshl",
        &["v0.4h", "v1.4h", "#3"],
        &[("v1", packed(16, &[0x1000, 0x8000, 0x0100]))],
        packed(16, &[0x7fff, 0x8000, 0x0800]),
    );
}

#[test]
fn sqshl_by_a_register_shifts_right_when_the_amount_is_negative() {
    // The per-lane amount is *signed*, so one lane can shift left while
    // its neighbour shifts right — and the right direction is
    // arithmetic here, not logical, so `-16 >> 1` is `-8` and not 120.
    assert_computes(
        "sqshl",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[
            ("v1", packed(8, &[0xf0, 0x40])),
            ("v2", packed(8, &[0xff, 0x02])),
        ],
        packed(8, &[0xf8, 0x7f]),
    );
}

#[test]
fn sqshl_by_an_amount_past_the_element_width_saturates_a_nonzero_source() {
    // The case the shift-it-back overflow test exists for: at the
    // element's own width `1 << 8` is zero, which a magnitude
    // comparison would read as an in-range result. Shifting back
    // catches it — and still lets a zero source through, which is what
    // the architecture does.
    assert_computes(
        "sqshl",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[
            ("v1", packed(8, &[0x01, 0x00])),
            ("v2", packed(8, &[0x08, 0x08])),
        ],
        packed(8, &[0x7f, 0x00]),
    );
}

#[test]
fn uqrshl_saturates_the_left_direction_at_the_unsigned_maximum() {
    assert_computes(
        "uqrshl",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x01]))],
        packed(8, &[0xff]),
    );
}

// ===================== the rounding right shifts =====================
//
// The half-ulp trap. ARM defines the rounding on the unbounded integer,
// so the term has to be added one bit wider than the source: at the
// source's own width the carry reaches the sign bit and a saturation at
// the top of the range becomes one at the bottom.

#[test]
fn sqrshl_adds_the_half_ulp_wider_than_the_element() {
    // `(127 + 1) >> 1` is 64. Added at eight bits, `127 + 1` is `0x80`
    // — negative — and an arithmetic shift of it gives `0xc0`, which is
    // `-64` rather than `64`.
    assert_computes(
        "sqrshl",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x7f])), ("v2", packed(8, &[0xff]))],
        packed(8, &[0x40]),
    );
}

#[test]
fn urshl_adds_the_half_ulp_wider_than_the_element() {
    // The unsigned twin, where the same carry leaves the element
    // entirely: `255 + 1` is zero at eight bits, so the wrong width
    // answers `0x00` where the architecture answers `0x80`.
    assert_computes(
        "urshl",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0xff])), ("v2", packed(8, &[0xff]))],
        packed(8, &[0x80]),
    );
}

// ===================== the mixed-signedness accumulates =====================

#[test]
fn suqadd_reads_its_source_unsigned() {
    // `-128 + 255` is `127`. Reading the source signed instead makes it
    // `-128 + -1`, which saturates at the *bottom* of the range — the
    // opposite end from the right answer.
    assert_computes(
        "suqadd",
        &["v0.8b", "v1.8b"],
        &[("v0", packed(8, &[0x80])), ("v1", packed(8, &[0xff]))],
        packed(8, &[0x7f]),
    );
}

#[test]
fn suqadd_clamps_a_sum_that_needs_two_extra_bits() {
    // `127 + 255` is 382, past the `n+1`-bit signed range the
    // same-signedness adds compute in. Computed there it wraps onto
    // `-130` and clamps at `0x80`, the wrong end again.
    assert_computes(
        "suqadd",
        &["v0.8b", "v1.8b"],
        &[("v0", packed(8, &[0x7f])), ("v1", packed(8, &[0xff]))],
        packed(8, &[0x7f]),
    );
}

#[test]
fn suqadd_leaves_a_representable_sum_alone() {
    assert_computes(
        "suqadd",
        &["v0.8b", "v1.8b"],
        &[("v0", packed(8, &[0xf0])), ("v1", packed(8, &[0x08]))],
        packed(8, &[0xf8]),
    );
}

#[test]
fn usqadd_reads_its_source_signed_and_clamps_a_negative_sum_at_zero() {
    // The destination is unsigned here and the source signed, so the
    // sum can go below zero — which is why the clamp is the
    // signed-into-unsigned one and not a plain upper bound.
    assert_computes(
        "usqadd",
        &["v0.8b", "v1.8b"],
        &[
            ("v0", packed(8, &[0x00, 0x80, 0x40])),
            ("v1", packed(8, &[0xff, 0x80, 0x10])),
        ],
        packed(8, &[0x00, 0x00, 0x50]),
    );
}

#[test]
fn usqadd_clamps_at_the_unsigned_maximum() {
    // The second lane is the two-extra-bits case: `255 + 127` is 382,
    // which at `n+1` bits wraps negative and would clamp to zero.
    assert_computes(
        "usqadd",
        &["v0.8b", "v1.8b"],
        &[
            ("v0", packed(8, &[0xff, 0xff])),
            ("v1", packed(8, &[0x01, 0x7f])),
        ],
        packed(8, &[0xff, 0xff]),
    );
}

// ===================== the doubling long multiplies =====================

#[test]
fn sqdmull_saturates_only_at_the_two_most_negative_elements() {
    // `2 * -32768 * -32768` is `2^31`, one past the doubled element's
    // signed maximum and the only input pair where this saturates. The
    // other two lanes are ordinary doubled products, one of them
    // negative, so a lowering that clamped everything fails here.
    assert_computes(
        "sqdmull",
        &["v0.4s", "v1.4h", "v2.4h"],
        &[
            ("v1", packed(16, &[0x8000, 0x0002, 0xffff])),
            ("v2", packed(16, &[0x8000, 0x0003, 0x0002])),
        ],
        packed(32, &[0x7fff_ffff, 0x0000_000c, 0xffff_fffc]),
    );
}

#[test]
fn sqdmull2_reads_the_sources_upper_half() {
    // The lower half of both sources is zero, so a lowering that
    // ignored the `2` suffix would answer zero everywhere.
    assert_computes(
        "sqdmull2",
        &["v0.4s", "v1.8h", "v2.8h"],
        &[
            ("v1", packed(16, &[0, 0, 0, 0, 0x0002])),
            ("v2", packed(16, &[0, 0, 0, 0, 0x0003])),
        ],
        packed(32, &[0x0000_000c]),
    );
}

#[test]
fn sqdmlal_saturates_the_accumulation_as_well_as_the_product() {
    // The second clamp: the product is representable and the *sum* is
    // not.
    assert_computes(
        "sqdmlal",
        &["v0.4s", "v1.4h", "v2.4h"],
        &[
            ("v0", packed(32, &[0x0000_0064, 0x7fff_ffff])),
            ("v1", packed(16, &[0x0002, 0x0001])),
            ("v2", packed(16, &[0x0003, 0x0001])),
        ],
        packed(32, &[0x0000_0070, 0x7fff_ffff]),
    );
}

#[test]
fn sqdmlsl_subtracts_the_product_after_it_has_already_saturated() {
    // The first clamp, isolated. `2 * -32768 * -32768` saturates to
    // `2^31 - 1`, and subtracting *that* from zero gives
    // `0x8000_0001`. Without the inner clamp the true `2^31` would be
    // subtracted instead and the answer would be `0x8000_0000` — a
    // difference of one, visible only because the saturation happens
    // before the accumulate and not after.
    assert_computes(
        "sqdmlsl",
        &["v0.4s", "v1.4h", "v2.4h"],
        &[
            ("v0", packed(32, &[0x0000_0000, 0x0000_0064])),
            ("v1", packed(16, &[0x8000, 0x0002])),
            ("v2", packed(16, &[0x8000, 0x0003])),
        ],
        packed(32, &[0x8000_0001, 0x0000_0058]),
    );
}

#[test]
fn the_doubling_multiplies_decline_the_by_element_spelling() {
    // `v2.h[0]` names one element rather than a vector, which is a
    // different instruction with a different source for every lane.
    assert_declines("sqdmull", &["v0.4s", "v1.4h", "v2.h[0]"]);
    assert_declines("sqdmlal", &["v0.4s", "v1.4h", "v2.h[0]"]);
}

#[test]
fn the_saturating_shift_left_unsigned_form_still_declines() {
    // `sqshlu` reads its element signed and clamps into the *unsigned*
    // range, which is neither of the two clamps this family carries.
    assert_declines("sqshlu", &["v0.8b", "v1.8b", "#1"]);
}
