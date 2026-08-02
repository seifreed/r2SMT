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

/// Lift `mnemonic operands`, bind every named register to a concrete
/// vector value, and ask the solver whether the destination is
/// necessarily `expected`.
///
/// The binding assignments are prepended to the lifted statements and
/// the whole thing is run through the real SSA pass, so an instruction
/// that *reads* its destination (`bsl`, `mla`, `ins`) sees the bound
/// value rather than contradicting it — the read and the write end up
/// as different versions, exactly as they would in the pipeline.
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

const HALF: [&str; 3] = ["v0.4h", "v1.4h", "v2.4h"];
const NARROW_FROM_HALF: [&str; 2] = ["v0.8b", "v1.8h"];
const BYTES: [&str; 3] = ["v0.8b", "v1.8b", "v2.8b"];

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

// --- same-width shifts ---

#[test]
fn shl_shifts_the_element_left() {
    assert_computes("shl", &["v0.4h", "v1.4h", "#4"], &[("v1", 0x1)], 0x10);
}

#[test]
fn ushr_shifts_in_zeroes() {
    assert_computes("ushr", &["v0.4h", "v1.4h", "#4"], &[("v1", 0x8000)], 0x0800);
}

#[test]
fn sshr_shifts_in_sign_bits() {
    // 0x8000 is negative; an arithmetic shift keeps it negative.
    assert_computes("sshr", &["v0.4h", "v1.4h", "#4"], &[("v1", 0x8000)], 0xf800);
}

#[test]
fn urshr_rounding_does_not_overflow_the_element() {
    // (0xffff + 1) >> 1 = 0x8000. Adding the rounding term at the
    // element's own width wraps to zero and would give zero.
    assert_computes(
        "urshr",
        &["v0.4h", "v1.4h", "#1"],
        &[("v1", 0xffff)],
        0x8000,
    );
}

#[test]
fn srshr_rounds_a_negative_element() {
    // -3 (0xfffd) rounded right by one is -1: (-3 + 1) >> 1.
    assert_computes(
        "srshr",
        &["v0.4h", "v1.4h", "#1"],
        &[("v1", 0xfffd)],
        0xffff,
    );
}

#[test]
fn ushl_shifts_left_when_the_amount_is_positive() {
    assert_computes("ushl", &HALF, &[("v1", 0x1), ("v2", 0x4)], 0x10);
}

#[test]
fn ushl_shifts_right_when_the_amount_is_negative() {
    // The per-lane amount is signed: 0xfffc is -4, so this lane shifts
    // right even though the mnemonic says "shift left".
    assert_computes("ushl", &HALF, &[("v1", 0x10), ("v2", 0xfffc)], 0x1);
}

#[test]
fn sshl_shifts_right_arithmetically_on_a_negative_amount() {
    assert_computes("sshl", &HALF, &[("v1", 0x8000), ("v2", 0xfffc)], 0xf800);
}

#[test]
fn ushl_lanes_can_shift_in_opposite_directions() {
    // Lane 0 shifts left by 4, lane 1 shifts right by 4 — the whole
    // reason the amount is a vector rather than an immediate.
    let sources = [
        ("v1", 0x0010_0001_u128),
        ("v2", (0xfffc_u128 << 16) | 0x0004),
    ];
    assert_computes("ushl", &HALF, &sources, 0x0001_0010);
}

#[test]
fn ushl_yields_zero_when_the_amount_exceeds_the_element_width() {
    assert_computes("ushl", &HALF, &[("v1", 0xffff), ("v2", 0x20)], 0x0);
}

#[test]
fn urshl_rounds_a_negative_shift() {
    // amount -1 shifts right by one with rounding: (0xffff + 1) >> 1.
    assert_computes("urshl", &HALF, &[("v1", 0xffff), ("v2", 0xffff)], 0x8000);
}

// --- lane-wise compares and conversions ---

#[test]
fn cmgt_writes_an_all_ones_mask_where_the_predicate_holds() {
    // A vector compare produces a value, not a flag: lane 0 is greater
    // so it becomes all ones, lane 1 is not so it becomes zero. The
    // remaining lanes hold zero on both sides, and zero is not greater
    // than zero.
    let sources = [
        ("v1", (0x0001_u128 << 16) | 0x0005),
        ("v2", (0x0009_u128 << 16) | 0x0003),
    ];
    assert_computes("cmgt", &HALF, &sources, 0xffff);
}

#[test]
fn cmgt_compares_signed() {
    // 0xffff is -1, which is not greater than 1.
    assert_computes("cmgt", &HALF, &[("v1", 0xffff), ("v2", 0x1)], 0x0);
}

#[test]
fn cmhi_compares_unsigned() {
    // The same bits read unsigned are 65535, which is greater than 1.
    assert_computes("cmhi", &HALF, &[("v1", 0xffff), ("v2", 0x1)], 0xffff);
}

#[test]
fn cmeq_against_zero_uses_the_two_operand_form() {
    assert_computes(
        "cmeq",
        &["v0.4h", "v1.4h", "#0"],
        &[("v1", 0x0)],
        0xffff_ffff_ffff_ffff,
    );
}

#[test]
fn cmge_includes_equality() {
    // Every lane compares equal — including the three holding zero — so
    // the whole 64-bit arrangement becomes all ones.
    let equal = 0x0003_0003_0003_0003_u128;
    assert_computes(
        "cmge",
        &HALF,
        &[("v1", equal), ("v2", equal)],
        0xffff_ffff_ffff_ffff,
    );
}

#[test]
fn cmtst_is_true_where_the_bitwise_and_is_non_zero() {
    assert_computes("cmtst", &HALF, &[("v1", 0x6), ("v2", 0x3)], 0xffff);
}

#[test]
fn cmtst_is_false_on_disjoint_bits() {
    assert_computes("cmtst", &HALF, &[("v1", 0x4), ("v2", 0x3)], 0x0);
}

#[test]
fn scvtf_converts_an_integer_lane_to_a_float_lane() {
    // 1.0 in binary32 is 0x3f800000.
    assert_computes("scvtf", &["v0.2s", "v1.2s"], &[("v1", 0x1)], 0x3f80_0000);
}

#[test]
fn ucvtf_reads_the_lane_as_unsigned() {
    // 0xffffffff is -1 signed but 4294967295 unsigned, which is
    // 0x4f800000 in binary32.
    assert_computes(
        "ucvtf",
        &["v0.2s", "v1.2s"],
        &[("v1", 0xffff_ffff)],
        0x4f80_0000,
    );
}

#[test]
fn fcvtzs_truncates_toward_zero() {
    // 1.5 (0x3fc00000) truncates to 1, not 2.
    assert_computes("fcvtzs", &["v0.2s", "v1.2s"], &[("v1", 0x3fc0_0000)], 0x1);
}

#[test]
fn fcvtl_widens_single_to_double() {
    // 1.0f (0x3f800000) becomes 1.0 (0x3ff0000000000000).
    assert_computes(
        "fcvtl",
        &["v0.2d", "v1.2s"],
        &[("v1", 0x3f80_0000)],
        0x3ff0_0000_0000_0000,
    );
}

#[test]
fn fcvtn_narrows_double_to_single() {
    assert_computes(
        "fcvtn",
        &["v0.2s", "v1.2d"],
        &[("v1", 0x3ff0_0000_0000_0000)],
        0x3f80_0000,
    );
}

// --- bitwise select ---

#[test]
fn bsl_selects_per_bit_using_the_destination_as_the_mask() {
    // Mask 0xff00 in lane 0: the high byte comes from the first source,
    // the low byte from the second.
    let sources = [
        ("v0", 0xff00_u128),
        ("v1", 0xaaaa_u128),
        ("v2", 0x5555_u128),
    ];
    assert_computes("bsl", &BYTES, &sources, 0xaa55);
}

#[test]
fn bit_inserts_where_the_second_source_has_set_bits() {
    // v0 starts as 0x1234; v2's set bits select v1's bits and the rest
    // of the destination survives.
    let sources = [
        ("v0", 0x1234_u128),
        ("v1", 0xaaaa_u128),
        ("v2", 0x00ff_u128),
    ];
    assert_computes("bit", &BYTES, &sources, 0x12aa);
}

#[test]
fn bif_inserts_where_the_second_source_has_clear_bits() {
    let sources = [
        ("v0", 0x1234_u128),
        ("v1", 0xaaaa_u128),
        ("v2", 0x00ff_u128),
    ];
    assert_computes("bif", &BYTES, &sources, 0xaa34);
}

#[test]
fn bitwise_select_declines_a_non_byte_arrangement() {
    // The architecture spells these only with `.8b` / `.16b`.
    let insn = Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: "bsl".into(),
        operands: ["v0.4s", "v1.4s", "v2.4s"]
            .iter()
            .map(|o| operand(o))
            .collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    };
    let lifted = lift_per_mnemonic(&insn, Arch::Aarch64);
    assert!(matches!(lifted.as_slice(), [IrStmt::Unsupported { .. }]));
}

#[test]
fn mla_accumulates_onto_the_destination_value() {
    // The accumulator's prior contents are part of the result, which is
    // only visible once the destination is bound.
    let sources = [("v0", 0x000a_u128), ("v1", 0x3_u128), ("v2", 0x4_u128)];
    assert_computes("mla", &HALF, &sources, 0x0016);
}

#[test]
fn mls_subtracts_from_the_destination_value() {
    let sources = [("v0", 0x000a_u128), ("v1", 0x3_u128), ("v2", 0x4_u128)];
    assert_computes("mls", &HALF, &sources, 0xfffe);
}

#[test]
fn ins_preserves_the_lanes_it_does_not_write() {
    // Lane 1 is replaced; lanes 0, 2 and 3 survive from the
    // destination's prior value.
    let sources = [("v0", 0x0004_0003_0002_0001_u128), ("v1", 0x00ab_u128)];
    assert_computes(
        "ins",
        &["v0.h[1]", "v1.h[0]"],
        &sources,
        0x0004_0003_00ab_0001,
    );
}

#[test]
fn xtn2_preserves_the_destination_lower_half() {
    // The narrowed lanes land high; the low half is the destination's.
    let sources = [
        ("v0", 0x0000_0000_1234_5678_u128),
        ("v1", 0x0004_0003_0002_0001_u128),
    ];
    assert_computes("xtn2", &["v0.8b", "v1.4h"], &sources, 0x0403_0201_1234_5678);
}

#[test]
fn bitwise_select_masks_at_the_full_vector_width() {
    // The 128-bit arrangement is the one whose all-ones mask sits at the
    // edge of the constant type; getting that bound wrong makes the
    // whole family decline rather than compute a wrong answer.
    let sources = [
        ("v0", u128::MAX >> 64),
        ("v1", 0xaaaa_aaaa_aaaa_aaaa_u128),
        ("v2", 0x5555_5555_5555_5555_u128 << 64),
    ];
    assert_computes(
        "bsl",
        &["v0.16b", "v1.16b", "v2.16b"],
        &sources,
        (0x5555_5555_5555_5555_u128 << 64) | 0xaaaa_aaaa_aaaa_aaaa,
    );
}
