//! `AArch32` NEON packed contracts.
//!
//! The corpus contained no 32-bit ARM sample until recently, so this
//! ISA's only coverage is synthetic — and a classification test is not
//! enough for it. `vmax.s32` and `vmax.u32` are the same lowering with
//! one boolean flipped, `vabs` and `vneg` differ only in which
//! expression the `Ite` selects, and every one of those mistakes yields
//! an expression that reads correctly and computes the wrong number.
//!
//! So these tests *solve* the lowering. Each binds the source vector
//! registers to concrete values, lifts the real instruction, and asserts
//! the destination is necessarily a value computed by hand from the ARM
//! definition. A wrong lowering makes the verdict `BothPossible` rather
//! than `AlwaysTrue`.
//!
//! Two `AArch32` facts shape the fixtures. The register file is one
//! synthetic 128-bit parent per `q`: `q1` is `v1`, `d2` is the low half
//! of `v1`, `d1` the *high* half of `v0`. And a NEON write **merges** —
//! it preserves the parent bits outside the destination's view, unlike
//! `AArch64`, which zeroes them.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, InstructionKind, LiftedSlice, SliceStatus,
    analyze, lift_per_mnemonic,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
/// Width of the synthetic `AArch32` vector parent every `q` / `d` / `s`
/// view is a slice of.
const VECTOR_BITS: u16 = 128;

/// binary32 `+1.0` and `-1.0`, as bit patterns.
const F32_ONE: u128 = 0x3f80_0000;
const F32_MINUS_ONE: u128 = 0xbf80_0000;

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
        mnemonic: "neontest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "neontest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands` under `Arch::Arm`, bind every named vector
/// parent to a concrete value, and ask the solver whether `v0` is
/// necessarily `expected`.
///
/// The bindings are prepended and the whole thing runs through the real
/// SSA pass, so a merging write sees the bound parent as its prior
/// version rather than contradicting it.
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
            Expr::Var(Var::new("v0", VECTOR_BITS)),
            Expr::konst(expected, VECTOR_BITS),
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

fn assert_computes(mnemonic: &str, operands: &[&str], sources: &[(&str, u128)], expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

/// The full-register three-operand shape: `q0, q1, q2` are `v0`, `v1`
/// and `v2` in their entirety, so nothing merges and the expectation is
/// the whole result.
const QUADS: [&str; 3] = ["q0", "q1", "q2"];
/// The one-source shape.
const QUAD_PAIR: [&str; 2] = ["q0", "q1"];

// ---------------------------------------------------------------
// `vmin` / `vmax` — signedness is the operation
// ---------------------------------------------------------------

#[test]
fn vmax_signed_reads_the_lane_as_two_s_complement() {
    // `0xffffffff` is `-1` signed, so the larger lane is `1`.
    assert_computes("vmax.s32", &QUADS, &[("v1", 0xffff_ffff), ("v2", 1)], 1);
}

#[test]
fn vmax_unsigned_reads_the_lane_as_a_magnitude() {
    // The teeth for the test above: the same bytes under the unsigned
    // encoding make `0xffffffff` the larger lane. One boolean in the
    // lowering separates these two.
    assert_computes(
        "vmax.u32",
        &QUADS,
        &[("v1", 0xffff_ffff), ("v2", 1)],
        0xffff_ffff,
    );
}

#[test]
fn vmin_signed_selects_the_smaller_lane() {
    // The direction, checked against the same inputs `vmax` used: a
    // lowering with its comparison reversed passes one and fails this.
    assert_computes(
        "vmin.s32",
        &QUADS,
        &[("v1", 0xffff_ffff), ("v2", 1)],
        0xffff_ffff,
    );
}

#[test]
fn vmax_selects_per_lane_and_not_across_the_view() {
    // The inputs are chosen so a single comparison over the whole
    // 128-bit view gives a different answer from the lane-wise one:
    // `v1` is the larger register but `v2` holds the larger low lane.
    assert_computes(
        "vmax.u32",
        &QUADS,
        &[("v1", 0x0000_0002_0000_0000), ("v2", 3)],
        0x0000_0002_0000_0003,
    );
}

#[test]
fn vmax_keeps_the_first_operand_when_the_lanes_are_equal() {
    // Equal lanes make the select's condition false, so the result is
    // whichever operand the `else` branch names — and with both equal
    // that is observably the same value either way. Pinned because the
    // boundary is where an off-by-one in the predicate would show.
    assert_computes("vmax.s32", &QUADS, &[("v1", 7), ("v2", 7)], 7);
}

// ---------------------------------------------------------------
// `vabs` / `vneg` — integer lanes
// ---------------------------------------------------------------

#[test]
fn vabs_signed_negates_a_negative_lane() {
    // `0xfffb` is `-5` as a signed halfword.
    assert_computes("vabs.s16", &QUAD_PAIR, &[("v1", 0xfffb)], 5);
}

#[test]
fn vabs_signed_leaves_a_positive_lane_alone() {
    assert_computes("vabs.s16", &QUAD_PAIR, &[("v1", 5)], 5);
}

#[test]
fn vabs_signed_wraps_at_the_most_negative_lane() {
    // `VABS` does not saturate — that is `VQABS` — so the absolute
    // value of `INT_MIN` is `INT_MIN`. The wrapping negation gives
    // that with no special case, which is worth pinning precisely
    // because it looks like a bug.
    assert_computes("vabs.s16", &QUAD_PAIR, &[("v1", 0x8000)], 0x8000);
}

#[test]
fn vneg_signed_negates_every_lane() {
    // Two lanes, so a lowering that computed only the low one leaves
    // the high lane at its bound value and fails here.
    assert_computes(
        "vneg.s32",
        &QUAD_PAIR,
        &[("v1", 0x0000_0002_0000_0003)],
        0xffff_fffe_ffff_fffd,
    );
}

#[test]
fn vneg_signed_wraps_at_the_most_negative_lane() {
    assert_computes("vneg.s8", &QUAD_PAIR, &[("v1", 0x80)], 0x80);
}

// ---------------------------------------------------------------
// `vabs` / `vneg` — floating-point lanes
// ---------------------------------------------------------------

#[test]
fn vabs_float_clears_the_sign_bit() {
    assert_computes("vabs.f32", &QUAD_PAIR, &[("v1", F32_MINUS_ONE)], F32_ONE);
}

#[test]
fn vneg_float_flips_the_sign_bit_of_every_lane() {
    // The regression this family exists for. `vneg.f32 q0, q1` used to
    // reach the *scalar* VFP handler, which computes lane 0 and leaves
    // the rest of the register standing — so three of these four lanes
    // held whatever `v0` did before.
    let ones = F32_ONE | (F32_ONE << 32) | (F32_ONE << 64) | (F32_ONE << 96);
    let negated =
        F32_MINUS_ONE | (F32_MINUS_ONE << 32) | (F32_MINUS_ONE << 64) | (F32_MINUS_ONE << 96);
    assert_computes("vneg.f32", &QUAD_PAIR, &[("v1", ones)], negated);
}

#[test]
fn vmax_float_selects_a_whole_lane_not_a_scalar() {
    // Same shape as the `vneg` regression: the packed `vmax.f32` was
    // reaching the scalar handler, which would leave lane 1 alone.
    let lhs = F32_ONE | (F32_ONE << 32);
    let rhs = F32_MINUS_ONE | (F32_MINUS_ONE << 32);
    assert_computes("vmax.f32", &QUADS, &[("v1", lhs), ("v2", rhs)], lhs);
}

// ---------------------------------------------------------------
// `vmla` / `vmls` — the destination is an input
// ---------------------------------------------------------------

#[test]
fn vmla_accumulates_into_the_destination() {
    // `3 + 5 * 7` is `38`. Overwriting the destination instead of
    // accumulating into it gives `35`, which is the mistake the
    // constants are chosen to separate.
    assert_computes(
        "vmla.i32",
        &QUADS,
        &[("v0", 3), ("v1", 5), ("v2", 7)],
        3 + 5 * 7,
    );
}

#[test]
fn vmls_subtracts_the_product_from_the_destination() {
    // `3 - 5 * 7` is `-32`, wrapping to `0xffffffe0` in the low lane.
    // The subtraction's direction is what this pins: `5 * 7 - 3` would
    // be `32`.
    assert_computes(
        "vmls.i32",
        &QUADS,
        &[("v0", 3), ("v1", 5), ("v2", 7)],
        0xffff_ffe0,
    );
}

#[test]
fn vmla_accumulates_lane_by_lane() {
    // Two lanes with different accumulators, so a lowering that read
    // the destination once for the whole view fails here.
    assert_computes(
        "vmla.i32",
        &QUADS,
        &[
            ("v0", 0x0000_000a_0000_0003),
            ("v1", 0x0000_0002_0000_0005),
            ("v2", 0x0000_0003_0000_0007),
        ],
        0x0000_0010_0000_0026,
    );
}

#[test]
fn vmla_float_rounds_at_the_product_and_again_at_the_sum() {
    // `VMLA.F32` is not fused — that is `VFMA` — so this is exactly two
    // roundings. `1.0 + 2.0 * 3.0` is `7.0`, `0x40e00000`.
    assert_computes(
        "vmla.f32",
        &QUADS,
        &[("v0", F32_ONE), ("v1", 0x4000_0000), ("v2", 0x4040_0000)],
        0x40e0_0000,
    );
}

#[test]
fn vmls_float_subtracts_the_product() {
    // `1.0 - 2.0 * 3.0` is `-5.0`, `0xc0a00000`.
    assert_computes(
        "vmls.f32",
        &QUADS,
        &[("v0", F32_ONE), ("v1", 0x4000_0000), ("v2", 0x4040_0000)],
        0xc0a0_0000,
    );
}

// ---------------------------------------------------------------
// `vqadd` / `vqsub` — saturation
// ---------------------------------------------------------------

#[test]
fn vqadd_signed_clamps_at_the_element_maximum() {
    // `0x7f + 1` overflows a signed byte; the result is `INT_MAX`, not
    // the wrapped `0x80`.
    assert_computes("vqadd.s8", &QUADS, &[("v1", 0x7f), ("v2", 1)], 0x7f);
}

#[test]
fn vqadd_signed_clamps_at_the_element_minimum() {
    // `-128 + -1` underflows; the result is `INT_MIN`. The low clamp is
    // a separate comparison from the high one, so it needs its own
    // test.
    assert_computes("vqadd.s8", &QUADS, &[("v1", 0x80), ("v2", 0xff)], 0x80);
}

#[test]
fn vqadd_signed_leaves_a_representable_sum_alone() {
    assert_computes("vqadd.s8", &QUADS, &[("v1", 0x10), ("v2", 0x20)], 0x30);
}

#[test]
fn vqadd_unsigned_clamps_at_the_unsigned_maximum() {
    // `0xff + 1` as unsigned bytes saturates at `0xff`. Under the
    // signed reading the same bytes are `-1 + 1 = 0`, so this and the
    // signed tests above cannot both pass by accident.
    assert_computes("vqadd.u8", &QUADS, &[("v1", 0xff), ("v2", 1)], 0xff);
}

#[test]
fn vqsub_unsigned_clamps_at_zero() {
    // The case a naive widening gets wrong: `1 - 2` on zero-extended
    // bytes wraps to a large *unsigned* value, and clamping that
    // against the unsigned maximum would give `0xff` instead of `0`.
    assert_computes("vqsub.u8", &QUADS, &[("v1", 1), ("v2", 2)], 0);
}

#[test]
fn vqsub_signed_clamps_at_the_element_minimum() {
    // `-128 - 1` underflows to `INT_MIN`.
    assert_computes("vqsub.s8", &QUADS, &[("v1", 0x80), ("v2", 1)], 0x80);
}

#[test]
fn vqadd_saturates_per_lane() {
    // The high lane saturates and the low one does not, so a lowering
    // that clamped the whole view once fails here.
    assert_computes(
        "vqadd.s8",
        &QUADS,
        &[("v1", 0x7f01), ("v2", 0x0102)],
        0x7f03,
    );
}

// ---------------------------------------------------------------
// `vhadd` / `vhsub` / `vrhadd` — halving
// ---------------------------------------------------------------

#[test]
fn vhadd_unsigned_halves_the_exact_sum() {
    // `(0xff + 0xff) / 2` is `0xff`. Adding at the element width first
    // would wrap to `0xfe` and halve to `0x7f`.
    assert_computes("vhadd.u8", &QUADS, &[("v1", 0xff), ("v2", 0xff)], 0xff);
}

#[test]
fn vhadd_unsigned_truncates_an_odd_sum() {
    // `(3 + 4) / 2` is `3`: the halving forms truncate, which is what
    // separates them from `vrhadd`.
    assert_computes("vhadd.u8", &QUADS, &[("v1", 3), ("v2", 4)], 3);
}

#[test]
fn vrhadd_rounds_an_odd_sum_up() {
    // The teeth for the test above: the same operands under the
    // rounding form give `4`.
    assert_computes("vrhadd.u8", &QUADS, &[("v1", 3), ("v2", 4)], 4);
}

#[test]
fn vhadd_signed_floors_a_negative_sum() {
    // `(-3 + 0) / 2` is `-2` under the architecture's bit-slice
    // definition, not the `-1` a truncation toward zero would give.
    assert_computes("vhadd.s8", &QUADS, &[("v1", 0xfd), ("v2", 0)], 0xfe);
}

#[test]
fn vhsub_unsigned_halves_a_negative_difference() {
    // `(1 - 2) / 2` floors to `-1`, which is `0xff` in the byte. The
    // unsigned element type describes the *operands*, not the exact
    // difference, which can still be negative.
    assert_computes("vhsub.u8", &QUADS, &[("v1", 1), ("v2", 2)], 0xff);
}

// ---------------------------------------------------------------
// Element types the encodings do not have
// ---------------------------------------------------------------

/// Whether the effect table and the lifter both refuse `mnemonic`.
///
/// Checked together on purpose: a mnemonic the slicer retains but the
/// lifter drops leaves the destination silently free.
fn declines(mnemonic: &str, operands: &[&str]) -> bool {
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
    analyze(&insn, Arch::Arm).kind == InstructionKind::Other
        && lift_per_mnemonic(&insn, Arch::Arm)
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
}

#[test]
fn vmax_rejects_the_untyped_element_spelling() {
    // The `i` spelling exists because add, subtract and multiply give
    // the same bits signed or unsigned. A comparison does not, so
    // `VMAX` has no `.i` encoding and claiming one would model an
    // instruction no assembler produces.
    assert!(declines("vmax.i32", &QUADS));
    assert!(declines("vmin.i16", &QUADS));
}

#[test]
fn vabs_rejects_an_unsigned_element_type() {
    // Absolute value and negation are meaningless on an unsigned
    // element, and the encodings are `.s8` / `.s16` / `.s32` / `.f32`
    // accordingly.
    assert!(declines("vabs.u16", &QUAD_PAIR));
    assert!(declines("vneg.u32", &QUAD_PAIR));
}

// ---------------------------------------------------------------
// The merging write
// ---------------------------------------------------------------

#[test]
fn a_half_register_destination_preserves_the_other_half() {
    // `d0` is the low half of `v0` and `d2` the low half of `v1`, so
    // the two lanes of `v1[63:0]` — `3` and `0` — negate into
    // `v0[63:0]` and `v0[127:64]` survives. An `AArch64`-style zeroing
    // write would clear it.
    assert_computes(
        "vneg.s32",
        &["d0", "d2"],
        &[("v0", 0xdead_beef_0000_0000_0000_0000_0000_0000), ("v1", 3)],
        0xdead_beef_0000_0000_0000_0000_ffff_fffd,
    );
}
