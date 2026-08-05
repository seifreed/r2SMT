//! `AArch64` NEON contracts for the unary, select, pairwise,
//! absolute-difference and high-narrowing families.
//!
//! Every lowering here fails as a *wrong value* rather than as a
//! decline, so a structural assertion on the emitted IR would pass over
//! all of it. Each test binds concrete lanes, lifts the real
//! instruction, and solves the destination against a value computed by
//! hand from the ARM definition.
//!
//! Four hazards are the reason the file exists. `abs` and `neg` do
//! **not** saturate, so `abs` of the most negative element is itself.
//! `fmax` / `fmin` propagate a NaN and combine the signs of a zero tie
//! where Intel's `MAXPS` returns its second operand, while `fmaxnm` /
//! `fminnm` let a number beat a quiet NaN — three different answers to
//! the same question. The pairwise family folds *adjacent* lanes of one
//! source rather than the lanes at a shared index. And the absolute
//! differences read their operands with the signedness the mnemonic
//! spells, which is what separates `sabd`'s 255 from `uabd`'s 1 on the
//! same two bytes.
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

/// binary32 bit patterns the float tests bind.
const F32_ONE: u128 = 0x3f80_0000;
const F32_TWO: u128 = 0x4000_0000;
const F32_THREE: u128 = 0x4040_0000;
const F32_FOUR: u128 = 0x4080_0000;
const F32_EIGHT: u128 = 0x4100_0000;
const F32_QUIET_NAN: u128 = 0x7fc0_0000;
const F32_POSITIVE_ZERO: u128 = 0x0000_0000;
const F32_NEGATIVE_ZERO: u128 = 0x8000_0000;

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
        mnemonic: "neonunary".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "neonunary".to_string(),
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
/// statements, so an accumulating instruction sees the bound
/// destination rather than contradicting it.
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

// ===================== unary lane functions =====================

#[test]
fn abs_of_the_most_negative_element_is_itself() {
    // `abs` does not saturate: negating `-128` at eight bits wraps back
    // to `-128`. A lowering that clamped at `0x7f` would be the
    // saturating `sqabs`, which is a different instruction.
    assert_computes(
        "abs",
        &["v0.8b", "v1.8b"],
        &[("v1", packed(8, &[0x80, 0x7f, 0xff]))],
        packed(8, &[0x80, 0x7f, 0x01]),
    );
}

#[test]
fn neg_wraps_rather_than_saturating() {
    assert_computes(
        "neg",
        &["v0.4h", "v1.4h"],
        &[("v1", packed(16, &[0x0001, 0x8000]))],
        packed(16, &[0xffff, 0x8000]),
    );
}

#[test]
fn fabs_clears_only_the_sign_bit() {
    // A bit mask rather than an arithmetic negation, which is what
    // makes it exact on a NaN: the payload survives untouched.
    assert_computes(
        "fabs",
        &["v0.2s", "v1.2s"],
        &[("v1", packed(32, &[F32_NEGATIVE_ZERO, 0xffc0_0000]))],
        packed(32, &[F32_POSITIVE_ZERO, F32_QUIET_NAN]),
    );
}

#[test]
fn fneg_flips_the_sign_bit_of_every_lane() {
    assert_computes(
        "fneg",
        &["v0.2s", "v1.2s"],
        &[("v1", packed(32, &[F32_ONE, F32_NEGATIVE_ZERO]))],
        packed(32, &[0xbf80_0000, F32_POSITIVE_ZERO]),
    );
}

#[test]
fn cnt_counts_the_set_bits_of_each_byte() {
    assert_computes(
        "cnt",
        &["v0.8b", "v1.8b"],
        &[("v1", packed(8, &[0xff, 0x00, 0x81, 0x0f]))],
        packed(8, &[8, 0, 2, 4]),
    );
}

#[test]
fn clz_counts_from_the_top_of_the_element() {
    // The all-zero lane answers the element width, and the lane with
    // only its top bit set answers zero — the two ends of the ladder.
    assert_computes(
        "clz",
        &["v0.4h", "v1.4h"],
        &[("v1", packed(16, &[0x0001, 0x8000, 0x0000, 0x00ff]))],
        packed(16, &[15, 0, 16, 8]),
    );
}

#[test]
fn cls_counts_the_bits_repeating_the_sign_and_never_the_sign_itself() {
    // `0x00` and `0xff` both answer 7, not 8: the sign bit is excluded,
    // so the count cannot reach the element width. That is also why the
    // four untouched lanes answer 7 rather than 0.
    assert_computes(
        "cls",
        &["v0.8b", "v1.8b"],
        &[("v1", packed(8, &[0x00, 0xff, 0x0f, 0xf0]))],
        packed(8, &[7, 7, 3, 3, 7, 7, 7, 7]),
    );
}

#[test]
fn rbit_reverses_the_bits_inside_each_byte() {
    assert_computes(
        "rbit",
        &["v0.8b", "v1.8b"],
        &[("v1", packed(8, &[0x01, 0x0f, 0xb1]))],
        packed(8, &[0x80, 0xf0, 0x8d]),
    );
}

// ===================== integer and float selects =====================

#[test]
fn smax_compares_its_lanes_signed() {
    assert_computes(
        "smax",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x7f]))],
        packed(8, &[0x7f]),
    );
}

#[test]
fn umax_compares_the_same_lanes_unsigned() {
    assert_computes(
        "umax",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x7f]))],
        packed(8, &[0x80]),
    );
}

#[test]
fn smin_and_umin_disagree_on_the_same_bytes() {
    assert_computes(
        "smin",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x7f]))],
        packed(8, &[0x80]),
    );
    assert_computes(
        "umin",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x7f]))],
        packed(8, &[0x7f]),
    );
}

#[test]
fn fmax_propagates_a_quiet_nan_where_maxps_would_return_the_number() {
    // Intel's `MAXPS` compares `SRC1 > SRC2`, which is false on
    // unordered, so it yields the *second* operand — an ordinary 1.0.
    // ARM's `FPMax` propagates the NaN, and the difference is a wrong
    // value rather than a wider one.
    assert_computes(
        "fmax",
        &["v0.2s", "v1.2s", "v2.2s"],
        &[
            ("v1", packed(32, &[F32_QUIET_NAN])),
            ("v2", packed(32, &[F32_ONE])),
        ],
        packed(32, &[F32_QUIET_NAN]),
    );
}

#[test]
fn fmaxnm_lets_the_number_beat_the_quiet_nan() {
    // The whole difference between `FPMax` and `FPMaxNum`, on the same
    // two operands the previous test binds.
    assert_computes(
        "fmaxnm",
        &["v0.2s", "v1.2s", "v2.2s"],
        &[
            ("v1", packed(32, &[F32_QUIET_NAN])),
            ("v2", packed(32, &[F32_ONE])),
        ],
        packed(32, &[F32_ONE]),
    );
}

#[test]
fn fmax_combines_the_signs_of_a_zero_tie() {
    // `+0.0` and `-0.0` compare equal, so no comparison can pick
    // between them. ARM ANDs the two patterns for a max, giving `+0.0`;
    // `MAXPS` would take its second operand and give `-0.0`.
    assert_computes(
        "fmax",
        &["v0.2s", "v1.2s", "v2.2s"],
        &[
            ("v1", packed(32, &[F32_POSITIVE_ZERO])),
            ("v2", packed(32, &[F32_NEGATIVE_ZERO])),
        ],
        packed(32, &[F32_POSITIVE_ZERO]),
    );
}

#[test]
fn fmin_combines_the_signs_of_a_zero_tie_the_other_way() {
    assert_computes(
        "fmin",
        &["v0.2s", "v1.2s", "v2.2s"],
        &[
            ("v1", packed(32, &[F32_POSITIVE_ZERO])),
            ("v2", packed(32, &[F32_NEGATIVE_ZERO])),
        ],
        packed(32, &[F32_NEGATIVE_ZERO]),
    );
}

// ===================== pairwise folds =====================

#[test]
fn addp_folds_adjacent_lanes_and_splits_the_halves_between_its_sources() {
    // The destination's low half comes from `v1`'s neighbours and its
    // high half from `v2`'s. A lane-wise lowering would give
    // `[11, 22, 33, 44]`; one that read both sources per pair would mix
    // them.
    assert_computes(
        "addp",
        &["v0.4s", "v1.4s", "v2.4s"],
        &[
            ("v1", packed(32, &[1, 2, 3, 4])),
            ("v2", packed(32, &[10, 20, 30, 40])),
        ],
        packed(32, &[3, 7, 30, 70]),
    );
}

#[test]
fn addp_wraps_inside_the_element() {
    assert_computes(
        "addp",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0xff, 0x02])), ("v2", 0)],
        packed(8, &[0x01]),
    );
}

#[test]
fn smaxp_and_umaxp_disagree_on_the_same_neighbours() {
    assert_computes(
        "smaxp",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80, 0x7f])), ("v2", 0)],
        packed(8, &[0x7f]),
    );
    assert_computes(
        "umaxp",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80, 0x7f])), ("v2", 0)],
        packed(8, &[0x80]),
    );
}

#[test]
fn faddp_sums_the_neighbouring_float_lanes() {
    assert_computes(
        "faddp",
        &["v0.2s", "v1.2s", "v2.2s"],
        &[
            ("v1", packed(32, &[F32_ONE, F32_TWO])),
            ("v2", packed(32, &[F32_FOUR, F32_FOUR])),
        ],
        packed(32, &[F32_THREE, F32_EIGHT]),
    );
}

#[test]
fn fmaxp_propagates_a_nan_and_fmaxnmp_does_not() {
    let sources = [
        ("v1", packed(32, &[F32_QUIET_NAN, F32_ONE])),
        ("v2", packed(32, &[F32_ONE, F32_TWO])),
    ];
    assert_computes(
        "fmaxp",
        &["v0.2s", "v1.2s", "v2.2s"],
        &sources,
        packed(32, &[F32_QUIET_NAN, F32_TWO]),
    );
    assert_computes(
        "fmaxnmp",
        &["v0.2s", "v1.2s", "v2.2s"],
        &sources,
        packed(32, &[F32_ONE, F32_TWO]),
    );
}

// ===================== absolute differences =====================

#[test]
fn sabd_reads_its_bytes_signed_and_the_magnitude_still_fits() {
    // `-128` and `127` differ by 255, which is `0xff` unsigned — the
    // value ARM defines, and exactly what the wrapping subtraction of
    // the smaller from the larger already gives at eight bits.
    assert_computes(
        "sabd",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x7f]))],
        packed(8, &[0xff]),
    );
}

#[test]
fn uabd_reads_the_same_bytes_unsigned() {
    assert_computes(
        "uabd",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[0x80])), ("v2", packed(8, &[0x7f]))],
        packed(8, &[0x01]),
    );
}

#[test]
fn saba_adds_the_difference_onto_the_destination() {
    // The accumulation wraps inside the element like every other NEON
    // add: `0xff + 2` is `0x01`.
    assert_computes(
        "saba",
        &["v0.8b", "v1.8b", "v2.8b"],
        &[
            ("v0", packed(8, &[0x02])),
            ("v1", packed(8, &[0x80])),
            ("v2", packed(8, &[0x7f])),
        ],
        packed(8, &[0x01]),
    );
}

#[test]
fn fabd_takes_the_magnitude_of_the_difference() {
    // Without the sign-bit clear this is `-3.0`, a perfectly plausible
    // wrong answer.
    assert_computes(
        "fabd",
        &["v0.2s", "v1.2s", "v2.2s"],
        &[
            ("v1", packed(32, &[F32_ONE])),
            ("v2", packed(32, &[F32_FOUR])),
        ],
        packed(32, &[F32_THREE]),
    );
}

// ===================== long pairwise addition =====================

#[test]
fn uaddlp_sums_neighbours_at_twice_their_width() {
    // `0xff + 0xff` is `0x1fe`, which does not fit the source element —
    // the `l` in the mnemonic is exactly that.
    assert_computes(
        "uaddlp",
        &["v0.4h", "v1.8b"],
        &[("v1", packed(8, &[0xff, 0xff]))],
        packed(16, &[0x01fe]),
    );
}

#[test]
fn saddlp_sign_extends_each_neighbour_first() {
    // The same two bytes read signed are `-1` each, so the pair is
    // `-2` — the same bits, a different number.
    assert_computes(
        "saddlp",
        &["v0.4h", "v1.8b"],
        &[("v1", packed(8, &[0xff, 0xff]))],
        packed(16, &[0xfffe]),
    );
}

#[test]
fn uadalp_accumulates_onto_the_destination() {
    assert_computes(
        "uadalp",
        &["v0.4h", "v1.8b"],
        &[
            ("v0", packed(16, &[0x0002])),
            ("v1", packed(8, &[0xff, 0xff])),
        ],
        packed(16, &[0x0200]),
    );
}

// ===================== high-half narrowing =====================

#[test]
fn addhn_keeps_the_high_half_of_the_sum() {
    assert_computes(
        "addhn",
        &["v0.8b", "v1.8h", "v2.8h"],
        &[("v1", packed(16, &[0x1234])), ("v2", packed(16, &[0x1000]))],
        packed(8, &[0x22]),
    );
}

#[test]
fn raddhn_adds_half_an_ulp_before_discarding_the_low_half() {
    // Without the rounding term the high byte of `0x0080` is `0x00`.
    assert_computes(
        "raddhn",
        &["v0.8b", "v1.8h", "v2.8h"],
        &[("v1", packed(16, &[0x0080])), ("v2", 0)],
        packed(8, &[0x01]),
    );
}

#[test]
fn raddhn_lets_the_rounding_carry_leave_the_source_width() {
    // `0xffff + 0xffff + 0x80` is `0x2007e`; the window ARM keeps is
    // bits 15..8 of that, which is `0x00`. The carry out of bit 16 is
    // outside the window, which is why wrapping at the source width
    // loses nothing here — unlike in the saturating narrows, where the
    // same carry would reach the sign bit the clamp compares.
    assert_computes(
        "raddhn",
        &["v0.8b", "v1.8h", "v2.8h"],
        &[("v1", packed(16, &[0xffff])), ("v2", packed(16, &[0xffff]))],
        packed(8, &[0x00]),
    );
}

#[test]
fn subhn_keeps_the_high_half_of_a_negative_difference() {
    assert_computes(
        "subhn",
        &["v0.4h", "v1.4s", "v2.4s"],
        &[("v1", 0), ("v2", packed(32, &[0x0001_0000]))],
        packed(16, &[0xffff]),
    );
}

#[test]
fn addhn2_writes_the_upper_half_and_preserves_the_lower_one() {
    assert_computes(
        "addhn2",
        &["v0.16b", "v1.8h", "v2.8h"],
        &[
            ("v0", packed(8, &[0xaa; 8])),
            ("v1", packed(16, &[0x1234])),
            ("v2", packed(16, &[0x1000])),
        ],
        packed(8, &[0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0x22]),
    );
}

// ===================== the boundary =====================

#[test]
fn the_wrapping_unary_forms_have_no_scalar_spelling_of_their_own() {
    // The boundary this file owns, restated now that `sqabs b0, b1`
    // lifts. `abs` and `neg` have scalar `D`-form encodings, but they
    // are not in the scalar-family allowlist and nothing resolves them,
    // so they must still decline rather than be swept in by the
    // saturating family's new arm.
    assert_declines("abs", &["d0", "d1"]);
    assert_declines("neg", &["d0", "d1"]);
}

#[test]
fn the_scalar_saturating_forms_reject_a_width_the_encoding_lacks() {
    // With no arranged operand there is nothing to check the width
    // against, so the encoding's own element list is the only check.
    // `q` is not in it, and `scalar_vector_width` would otherwise report
    // 128 as happily as it reports 8.
    assert_declines("sqabs", &["q0", "q1"]);
    // Mismatched views are not an encoding either.
    assert_declines("sqneg", &["b0", "h1"]);
    // And a general register is not a SIMD view at all.
    assert_declines("sqabs", &["w0", "w1"]);
}
