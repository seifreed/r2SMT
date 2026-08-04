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

/// Classify a fixture operand the way the radare2 adapter's parser
/// does: a bracketed operand addresses memory, anything starting with a
/// digit is an immediate, everything else names a register.
///
/// `AArch32` NEON shift counts matter here — radare2 prints them bare
/// (`vshl.i32 q0, q1, 3`), and in hex once they reach `0x10`, rather
/// than with the manual's `#`.
///
/// The memory case is load-bearing for the structured accesses rather
/// than cosmetic: their resolver refuses an operand that is not
/// [`OperandKind::Memory`], so a fixture spelling `[r0]` as a register
/// would make every one of their decline assertions pass without
/// reaching the check it names.
fn operand(raw: &str) -> Operand {
    // The leading bracket is what separates an address from an indexed
    // register: `[r0]` addresses memory, `d4[2]` names one lane of a
    // register and must stay `Register` for the by-element resolver.
    let kind = if raw.trim_start().starts_with('[') {
        OperandKind::Memory
    } else if raw.starts_with(|c: char| c.is_ascii_digit()) {
        OperandKind::Immediate
    } else {
        OperandKind::Register
    };
    Operand {
        raw: raw.into(),
        kind,
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
    let bindings: Vec<(&str, u128, u16)> = sources
        .iter()
        .map(|(name, value)| (*name, *value, VECTOR_BITS))
        .collect();
    solve_lowering_at_widths(mnemonic, operands, &bindings, "v0", expected)
}

/// [`solve_lowering`], but each binding names the width its variable
/// holds and the assertion names the register to inspect.
///
/// `vdup` is why the widths are explicit: its source is a
/// general-purpose register, which the lifter reads at 32 bits, so
/// binding it at the vector parent's width would make the slice
/// contradict itself. `vzip` is why the destination is explicit: it
/// writes two registers, and checking only the first would let a wrong
/// second result pass.
fn solve_lowering_at_widths(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128, u16)],
    destination: &str,
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
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    statements.extend(lifted);

    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::Var(Var::new(destination, VECTOR_BITS)),
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

fn assert_computes_at_widths(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128, u16)],
    expected: u128,
) {
    assert_eq!(
        solve_lowering_at_widths(mnemonic, operands, sources, "v0", expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

/// Assert what `mnemonic` leaves in a named register, for the forms
/// whose result is not all in `v0`.
fn assert_computes_into(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    destination: &str,
    expected: u128,
) {
    let bindings: Vec<(&str, u128, u16)> = sources
        .iter()
        .map(|(name, value)| (*name, *value, VECTOR_BITS))
        .collect();
    assert_eq!(
        solve_lowering_at_widths(mnemonic, operands, &bindings, destination, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should leave {expected:#x} in {destination}"
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
// `vshl` / `vshr` / `vsra` — shifts
// ---------------------------------------------------------------

#[test]
fn vshl_immediate_shifts_every_lane_left() {
    // Two lanes with different values, so a lowering that shifted the
    // whole view as one number would carry across the lane boundary
    // and fail here.
    assert_computes(
        "vshl.i32",
        &["q0", "q1", "4"],
        &[("v1", 0x0000_0003_8000_0001)],
        0x0000_0030_0000_0010,
    );
}

#[test]
fn vshr_signed_replicates_the_sign_bit() {
    // `0xfffffff0 >> 2` arithmetic is `0xfffffffc`.
    assert_computes(
        "vshr.s32",
        &["q0", "q1", "2"],
        &[("v1", 0xffff_fff0)],
        0xffff_fffc,
    );
}

#[test]
fn vshr_unsigned_shifts_zeroes_in() {
    // The teeth: the same bytes under the unsigned encoding give
    // `0x3ffffffc`. Arithmetic against logical is the whole difference
    // between these two mnemonics.
    assert_computes(
        "vshr.u32",
        &["q0", "q1", "2"],
        &[("v1", 0xffff_fff0)],
        0x3fff_fffc,
    );
}

#[test]
fn vshr_unsigned_accepts_a_shift_of_the_whole_element() {
    // `VSHR`'s immediate range is `1..=esize`, so a shift by 32 of a
    // 32-bit element is a real encoding — and radare2 prints it in hex
    // (`0x20`), not decimal. The result is zero.
    assert_computes("vshr.u32", &["q0", "q1", "0x20"], &[("v1", 0xffff_ffff)], 0);
}

#[test]
fn vsra_adds_the_shifted_value_into_the_destination() {
    // `vsra` is `vshr` plus an accumulate: `0x10 + (0x40 >> 2)` is
    // `0x20`. A lowering that overwrote instead would give `0x10`.
    assert_computes(
        "vsra.u32",
        &["q0", "q1", "2"],
        &[("v0", 0x10), ("v1", 0x40)],
        0x20,
    );
}

#[test]
fn vshl_register_shifts_left_on_a_positive_amount() {
    assert_computes("vshl.u32", &QUADS, &[("v1", 1), ("v2", 4)], 0x10);
}

#[test]
fn vshl_register_shifts_right_on_a_negative_amount() {
    // The register form reads a *signed* amount whose sign chooses the
    // direction — the one thing that separates it from the immediate
    // encoding beyond where the count comes from. `0xff` is `-1`.
    assert_computes("vshl.u32", &QUADS, &[("v1", 0x10), ("v2", 0xff)], 8);
}

#[test]
fn vshl_register_reads_only_the_low_byte_of_the_amount() {
    // The architecture takes the amount from the element's low byte.
    // Reading the whole element would make `0x100` a shift by 256 —
    // zero — instead of by nothing.
    assert_computes("vshl.u32", &QUADS, &[("v1", 7), ("v2", 0x100)], 7);
}

#[test]
fn vshl_register_signedness_selects_the_right_shift_kind() {
    // The same negative amount, the same value, and the two element
    // classes disagree — which is what the signed flag is for.
    assert_computes(
        "vshl.s32",
        &QUADS,
        &[("v1", 0xffff_fff0), ("v2", 0xff)],
        0xffff_fff8,
    );
}

#[test]
fn vshl_rejects_a_register_amount_under_the_immediate_encoding() {
    // The element class alone tells `VSHL`'s two encodings apart, so
    // an untyped mnemonic beside a register amount is not an
    // instruction. Lifting it would read a vector register as a count.
    assert!(declines("vshl.i32", &QUADS));
}

#[test]
fn vshl_rejects_an_immediate_amount_under_the_register_encoding() {
    assert!(declines("vshl.s32", &["q0", "q1", "4"]));
}

#[test]
fn vshr_rejects_the_untyped_element_spelling() {
    // Arithmetic against logical is the whole distinction, so there is
    // nothing for an untyped right shift to mean and no encoding for
    // one.
    assert!(declines("vshr.i32", &["q0", "q1", "2"]));
    assert!(declines("vsra.i32", &["q0", "q1", "2"]));
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

#[test]
fn vrshr_rounds_the_discarded_bits_rather_than_truncating() {
    // 6 >> 2 is 1 truncated, but vrshr adds half an ulp first:
    // (6 + 2) >> 2 is 2. The whole contract is that difference from a
    // plain vshr.
    assert_computes("vrshr.u32", &["q0", "q1", "2"], &[("v1", 6)], 2);
}

#[test]
fn vrshr_carries_the_round_past_the_signed_maximum() {
    // round(INT_MAX / 2) is 0x4000_0000. The half added before the
    // shift overflows a same-width signed value, so the lane is computed
    // one bit wider — the teeth for that widening.
    assert_computes(
        "vrshr.s32",
        &["q0", "q1", "1"],
        &[("v1", 0x7fff_ffff)],
        0x4000_0000,
    );
}

#[test]
fn vrsra_adds_the_rounded_shift_into_the_destination() {
    // vrshr plus an accumulate: 0x10 + round(6 / 4) is 0x12, where a
    // truncating accumulate would give 0x11.
    assert_computes(
        "vrsra.u32",
        &["q0", "q1", "2"],
        &[("v0", 0x10), ("v1", 6)],
        0x12,
    );
}

#[test]
fn vqshl_clamps_an_unsigned_overflow_to_the_element_maximum() {
    // 0x4000_0000 << 4 is 0x4_0000_0000, past a 32-bit lane; a plain
    // shift wraps it to zero, but vqshl saturates to the unsigned max.
    assert_computes(
        "vqshl.u32",
        &["q0", "q1", "4"],
        &[("v1", 0x4000_0000)],
        0xffff_ffff,
    );
}

#[test]
fn vqshl_clamps_a_signed_positive_overflow_to_the_signed_maximum() {
    assert_computes(
        "vqshl.s32",
        &["q0", "q1", "4"],
        &[("v1", 0x4000_0000)],
        0x7fff_ffff,
    );
}

#[test]
fn vqshl_clamps_a_signed_negative_overflow_to_the_signed_minimum() {
    // 0xc000_0000 is -2^30; shifted left by 4 it underflows, and the
    // lane saturates to the signed minimum rather than wrapping.
    assert_computes(
        "vqshl.s32",
        &["q0", "q1", "4"],
        &[("v1", 0xc000_0000)],
        0x8000_0000,
    );
}

#[test]
fn vqshl_leaves_an_in_range_shift_alone() {
    // No overflow, no clamp: 3 << 2 is 0xc.
    assert_computes("vqshl.u32", &["q0", "q1", "2"], &[("v1", 3)], 0xc);
}

// ---------------------------------------------------------------
// The bit-moving families, whose element is a bare width
//
// `vext` / `vrev` / `vdup` spell their element as a number alone
// (`vext.8`, `vrev64.16`) because they move bits rather than computing
// on them, so there is no arithmetic for a sign to change. These solve
// against hand-computed byte orders: a lane-index error reads correctly
// and produces the wrong permutation, which no structural assertion
// catches.
// ---------------------------------------------------------------

/// Byte `i` holds `i`, so a permuted result can be read straight off
/// the hex digits.
const BYTE_RAMP: u128 = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100;
/// The same ramp continued: byte `i` holds `0x10 + i`.
const BYTE_RAMP_HIGH: u128 = 0x1f1e_1d1c_1b1a_1918_1716_1514_1312_1110;

#[test]
fn vext_takes_the_low_bytes_from_the_first_source() {
    // Destination byte `i` is source-one byte `i + 3` while that is
    // still inside the register, so bytes 0..12 walk 0x03..0x0f; the
    // top three spill into source two at 0x10, 0x11, 0x12.
    assert_computes(
        "vext.8",
        &["q0", "q1", "q2", "3"],
        &[("v1", BYTE_RAMP), ("v2", BYTE_RAMP_HIGH)],
        0x1211_100f_0e0d_0c0b_0a09_0807_0605_0403,
    );
}

#[test]
fn vext_runs_off_the_first_source_into_the_second() {
    // The teeth for the test above, at the far end of the window: at
    // offset 15 only one byte comes from source one and the other
    // fifteen from source two. Concatenating the sources the other way
    // round makes this the mirror image.
    assert_computes(
        "vext.8",
        &["q0", "q1", "q2", "15"],
        &[("v1", BYTE_RAMP), ("v2", BYTE_RAMP_HIGH)],
        0x1e1d_1c1b_1a19_1817_1615_1413_1211_100f,
    );
}

#[test]
fn vext_at_offset_zero_is_the_first_source_entire() {
    assert_computes(
        "vext.8",
        &["q0", "q1", "q2", "0"],
        &[("v1", BYTE_RAMP), ("v2", BYTE_RAMP_HIGH)],
        BYTE_RAMP,
    );
}

#[test]
fn vext_on_half_registers_wraps_at_the_doubleword() {
    // `d2` is the low half of `v1` and `d4` the low half of `v2`, so
    // the window is 8 bytes wide, not 16: bytes 0..4 come from `d2`'s
    // bytes 3..7 and the top three from `d4`'s bytes 0..2.
    assert_computes(
        "vext.8",
        &["d0", "d2", "d4", "3"],
        &[
            ("v0", 0),
            ("v1", 0x0706_0504_0302_0100),
            ("v2", 0x0f0e_0d0c_0b0a_0908),
        ],
        0x0a09_0807_0605_0403,
    );
}

#[test]
fn vrev64_reverses_the_bytes_inside_each_doubleword() {
    // Each 64-bit container reverses independently: the low half's
    // bytes 0..7 come back as 7..0, and the high half's separately.
    // Reversing the whole register instead would swap the two halves.
    assert_computes(
        "vrev64.8",
        &QUAD_PAIR,
        &[("v1", BYTE_RAMP)],
        0x0809_0a0b_0c0d_0e0f_0001_0203_0405_0607,
    );
}

#[test]
fn vrev32_reverses_the_halfwords_inside_each_word() {
    // Same source, a different container and element: each 32-bit word
    // swaps its two halfwords and nothing crosses a word boundary.
    assert_computes(
        "vrev32.16",
        &QUAD_PAIR,
        &[("v1", BYTE_RAMP)],
        0x0d0c_0f0e_0908_0b0a_0504_0706_0100_0302,
    );
}

#[test]
fn vrev16_reverses_the_bytes_inside_each_halfword() {
    assert_computes(
        "vrev16.8",
        &QUAD_PAIR,
        &[("v1", BYTE_RAMP)],
        0x0e0f_0c0d_0a0b_0809_0607_0405_0203_0001,
    );
}

#[test]
fn vrev_on_a_half_register_preserves_the_other_half() {
    // `d0` is the low half of `v0`, so the reversal writes 64 bits and
    // `v0[127:64]` survives. An AArch64-style zeroing write would clear
    // it.
    assert_computes(
        "vrev64.8",
        &["d0", "d2"],
        &[
            ("v0", 0xdead_beef_0000_0000_0000_0000_0000_0000),
            ("v1", 0x0706_0504_0302_0100),
        ],
        0xdead_beef_0000_0000_0001_0203_0405_0607,
    );
}

#[test]
fn vdup_replicates_a_general_register_byte_to_every_lane() {
    // Only the low 8 bits of `r1` are read, so the 0xff above them must
    // not reach the result.
    assert_computes_at_widths(
        "vdup.8",
        &["q0", "r1"],
        &[("r1", 0xffff_ffab, 32)],
        0xabab_abab_abab_abab_abab_abab_abab_abab,
    );
}

#[test]
fn vdup_replicates_a_general_register_halfword_to_every_lane() {
    // The teeth for the test above: the same source at a wider element
    // keeps one more byte, so a lowering that hardcoded the element
    // width fails exactly one of the pair.
    assert_computes_at_widths(
        "vdup.16",
        &["q0", "r1"],
        &[("r1", 0xffff_beef, 32)],
        0xbeef_beef_beef_beef_beef_beef_beef_beef,
    );
}

#[test]
fn vdup_on_a_half_register_fills_only_that_half() {
    assert_computes_at_widths(
        "vdup.32",
        &["d0", "r1"],
        &[
            ("v0", 0xdead_beef_0000_0000_0000_0000_0000_0000, 128),
            ("r1", 0x1234_5678, 32),
        ],
        0xdead_beef_0000_0000_1234_5678_1234_5678,
    );
}

// ---------------------------------------------------------------
// The boundary of what commit-one models
// ---------------------------------------------------------------

#[test]
fn vext_declines_a_window_starting_past_the_view() {
    // A 16-byte view has offsets 0..15; at 16 the window would be the
    // second source entire, which the encoding cannot spell.
    assert!(declines("vext.8", &["q0", "q1", "q2", "16"]));
}

#[test]
fn vext_declines_a_mismatched_register_class() {
    // Every operand names one view width; a `d` beside a `q` is not an
    // instruction.
    assert!(declines("vext.8", &["q0", "d2", "q2", "3"]));
}

#[test]
fn vrev_declines_a_container_no_wider_than_its_element() {
    // There is nothing to reverse inside a container holding one
    // element, and the architecture has no such encoding.
    assert!(declines("vrev32.32", &["q0", "q1"]));
    assert!(declines("vrev16.32", &["q0", "q1"]));
}

#[test]
fn vdup_declines_the_element_source_spelling() {
    // `vdup.32 q0, d2[1]` broadcasts a vector lane rather than a
    // general register. Its operand carries vector shape, which is the
    // by-element seam, not this one.
    assert!(declines("vdup.32", &["q0", "d2[1]"]));
}

// ---------------------------------------------------------------
// The permutations that rewrite both named registers
//
// One AArch32 `vzip` does the work AArch64 splits across `zip1` and
// `zip2`, writing both of its operands. Every test below therefore
// comes in a pair, one per destination: checking only the first would
// let a wrong second result through, and the second is exactly the half
// a single-destination port would forget to write.
// ---------------------------------------------------------------

#[test]
fn vzip_interleaves_the_low_halves_into_the_first_register() {
    // Laid end to end the two registers make one alternating sequence;
    // the first destination takes its low half, so byte `2e` comes from
    // the first source and `2e+1` from the second.
    assert_computes(
        "vzip.8",
        &QUAD_PAIR,
        &[("v0", BYTE_RAMP), ("v1", BYTE_RAMP_HIGH)],
        0x1707_1606_1505_1404_1303_1202_1101_1000,
    );
}

#[test]
fn vzip_interleaves_the_high_halves_into_the_second_register() {
    // The other half of the same instruction: the second destination
    // takes the sequence's high half, starting at each source's element
    // 8. A port that wrote only the first register leaves this one
    // holding its input.
    assert_computes_into(
        "vzip.8",
        &QUAD_PAIR,
        &[("v0", BYTE_RAMP), ("v1", BYTE_RAMP_HIGH)],
        "v1",
        0x1f0f_1e0e_1d0d_1c0c_1b0b_1a0a_1909_1808,
    );
}

#[test]
fn vuzp_collects_the_even_elements_into_the_first_register() {
    // The inverse of `vzip`: the first destination takes every even
    // element of the concatenation — the first source's, then the
    // second's.
    assert_computes(
        "vuzp.8",
        &QUAD_PAIR,
        &[("v0", BYTE_RAMP), ("v1", BYTE_RAMP_HIGH)],
        0x1e1c_1a18_1614_1210_0e0c_0a08_0604_0200,
    );
}

#[test]
fn vuzp_collects_the_odd_elements_into_the_second_register() {
    assert_computes_into(
        "vuzp.8",
        &QUAD_PAIR,
        &[("v0", BYTE_RAMP), ("v1", BYTE_RAMP_HIGH)],
        "v1",
        0x1f1d_1b19_1715_1311_0f0d_0b09_0705_0301,
    );
}

#[test]
fn vtrn_swaps_each_odd_element_with_the_even_one_beside_it() {
    // A 2x2 transpose: even elements of the first register stay put and
    // the odd ones take the second register's even elements. Distinct
    // from `vzip` on the same inputs, which is the point of testing
    // both against one byte ramp.
    assert_computes(
        "vtrn.8",
        &QUAD_PAIR,
        &[("v0", BYTE_RAMP), ("v1", BYTE_RAMP_HIGH)],
        0x1e0e_1c0c_1a0a_1808_1606_1404_1202_1000,
    );
}

#[test]
fn vtrn_moves_the_first_registers_odd_elements_into_the_second() {
    assert_computes_into(
        "vtrn.8",
        &QUAD_PAIR,
        &[("v0", BYTE_RAMP), ("v1", BYTE_RAMP_HIGH)],
        "v1",
        0x1f0f_1d0d_1b0b_1909_1707_1505_1303_1101,
    );
}

#[test]
fn a_paired_permutation_finishes_reading_before_it_starts_writing() {
    // `d0` and `d1` are the low and high halves of the *same* parent,
    // so this is the case that catches a lowering which materialises
    // its second result after performing the first write: it would read
    // the freshly zipped low half back as if it were the input. Both
    // results here are functions of the original `v0` alone.
    assert_computes(
        "vzip.8",
        &["d0", "d1"],
        &[("v0", BYTE_RAMP)],
        0x0f07_0e06_0d05_0c04_0b03_0a02_0901_0800,
    );
}

#[test]
fn vtrn_on_two_element_registers_is_a_single_swap() {
    // `vtrn.32 d0, d2` holds two elements per register, so the
    // transpose reduces to exchanging one element each way. The high
    // half of `v0` is untouched.
    assert_computes(
        "vtrn.32",
        &["d0", "d2"],
        &[("v0", 0xaaaa_aaaa_1111_1111), ("v1", 0xbbbb_bbbb_2222_2222)],
        0x2222_2222_1111_1111,
    );
}

#[test]
fn vtrn_on_two_element_registers_puts_the_other_pair_in_the_second() {
    assert_computes_into(
        "vtrn.32",
        &["d0", "d2"],
        &[("v0", 0xaaaa_aaaa_1111_1111), ("v1", 0xbbbb_bbbb_2222_2222)],
        "v1",
        0xbbbb_bbbb_aaaa_aaaa,
    );
}

#[test]
fn vzip_declines_where_a_register_holds_only_two_elements() {
    // `VZIP.32 Dd, Dm` is UNDEFINED: with one element per half the
    // operation degenerates to the swap `vtrn.32` spells, and an
    // assembler accepting the spelling emits `VTRN`. Modelling it as a
    // zip would compute a different permutation from the one the
    // hardware runs.
    assert!(declines("vzip.32", &["d0", "d2"]));
    assert!(declines("vuzp.32", &["d0", "d2"]));
}

#[test]
fn the_paired_permutations_decline_a_doubleword_element() {
    // UNDEFINED at that size for all three, whatever the register
    // class.
    assert!(declines("vzip.64", &QUAD_PAIR));
    assert!(declines("vuzp.64", &QUAD_PAIR));
    assert!(declines("vtrn.64", &QUAD_PAIR));
}

// ---------------------------------------------------------------
// Widening and narrowing
//
// The families whose destination element is twice its sources' width,
// or half. Two things need solving rather than asserting on shape: that
// the arithmetic really happens at the *wide* width — a sum that
// overflows the narrow element must not wrap — and that the extension
// follows the mnemonic's signedness, which is one boolean and changes
// every lane.
// ---------------------------------------------------------------

/// Bytes chosen so half are negative as two's complement: `0x80`,
/// `0xff` and `0x7f` sit at the boundaries the extension has to get
/// right.
const SIGNED_BYTES: u128 = 0x8001_7fff_0102_ff80;

#[test]
fn vmovl_signed_replicates_the_sign_bit_into_the_wide_element() {
    assert_computes(
        "vmovl.s8",
        &["q0", "d2"],
        &[("v1", SIGNED_BYTES)],
        0xff80_0001_007f_ffff_0001_0002_ffff_ff80,
    );
}

#[test]
fn vmovl_unsigned_fills_the_wide_element_with_zeroes() {
    // The teeth for the test above: the same bytes, one boolean apart,
    // and every lane whose top bit is set differs.
    assert_computes(
        "vmovl.u8",
        &["q0", "d2"],
        &[("v1", SIGNED_BYTES)],
        0x0080_0001_007f_00ff_0001_0002_00ff_0080,
    );
}

#[test]
fn vaddl_computes_at_the_destination_width_so_the_sum_cannot_wrap() {
    // Lane 0 is `0x7fff + 1`, which overflows a halfword and would come
    // back `0x8000` — a negative — if the addition happened before the
    // extension. At the destination's width it is a plain `0x00008000`.
    assert_computes(
        "vaddl.s16",
        &["q0", "d2", "d4"],
        &[("v1", 0xffff_0001_8000_7fff), ("v2", 0x0001_0001_8000_0001)],
        0x0000_0000_0000_0002_ffff_0000_0000_8000,
    );
}

#[test]
fn vaddl_unsigned_extends_the_same_lanes_differently() {
    assert_computes(
        "vaddl.u16",
        &["q0", "d2", "d4"],
        &[("v1", 0xffff_0001_8000_7fff), ("v2", 0x0001_0001_8000_0001)],
        0x0001_0000_0000_0002_0001_0000_0000_8000,
    );
}

#[test]
fn vmull_signed_multiplies_two_negative_bytes_into_a_positive_halfword() {
    // `0xff * 0xff` is `1` read as signed and `0xfe01` read as
    // unsigned, so this pair pins the signedness of the widening on the
    // multiply as well as on the extension.
    assert_computes(
        "vmull.s8",
        &["q0", "d2", "d4"],
        &[("v1", 0xff01_02ff_7f80_01ff), ("v2", 0xff02_03ff_0201_02ff)],
        0x0001_0002_0006_0001_00fe_ff80_0002_0001,
    );
}

#[test]
fn vmull_unsigned_multiplies_the_same_bytes_as_magnitudes() {
    assert_computes(
        "vmull.u8",
        &["q0", "d2", "d4"],
        &[("v1", 0xff01_02ff_7f80_01ff), ("v2", 0xff02_03ff_0201_02ff)],
        0xfe01_0002_0006_fe01_00fe_0080_0002_fe01,
    );
}

#[test]
fn vaddw_reads_its_first_source_at_the_destination_width() {
    // `vaddw` is the `w`-suffixed form: only the second source is
    // narrow. Extending the first as well would read four halfwords
    // where the instruction names four words.
    assert_computes(
        "vaddw.s16",
        &["q0", "q1", "d4"],
        &[
            ("v1", 0x0000_0000_0000_0002_ffff_0000_0000_8000),
            ("v2", 0xffff_0001_8000_7fff),
        ],
        0xffff_ffff_0000_0003_fffe_8000_0000_ffff,
    );
}

#[test]
fn vmovn_keeps_the_low_half_of_every_element() {
    assert_computes(
        "vmovn.i16",
        &["d0", "q1"],
        &[("v0", 0), ("v1", 0x1234_5678_9abc_def0_1122_3344_5566_7788)],
        0x3478_bcf0_2244_6688,
    );
}

#[test]
fn vmovn_on_a_half_register_preserves_the_other_half() {
    assert_computes(
        "vmovn.i16",
        &["d0", "q1"],
        &[
            ("v0", 0xdead_beef_0000_0000_0000_0000_0000_0000),
            ("v1", 0x1234_5678_9abc_def0_1122_3344_5566_7788),
        ],
        0xdead_beef_0000_0000_3478_bcf0_2244_6688,
    );
}

#[test]
fn vshrn_shifts_before_it_truncates() {
    // Shifting after the truncation would discard the bits this is
    // meant to bring down: lane 4 is `0xdef0`, whose shifted low byte
    // is `0xef`, where truncate-then-shift gives `0x0f`.
    assert_computes(
        "vshrn.i16",
        &["d0", "q1", "4"],
        &[("v0", 0), ("v1", 0x1234_5678_9abc_def0_1122_3344_5566_7788)],
        0x2367_abef_1234_5678,
    );
}

// --- the boundary of the widening seam ---

#[test]
fn the_long_forms_decline_the_sign_agnostic_element_spelling() {
    // The extension *is* the operation, so `i` has nothing to mean and
    // the architecture has no encoding for it.
    assert!(declines("vmovl.i8", &["q0", "d2"]));
    assert!(declines("vaddl.i16", &["q0", "d2", "d4"]));
    assert!(declines("vmull.i8", &["q0", "d2", "d4"]));
}

#[test]
fn the_narrowing_forms_decline_a_signed_element_spelling() {
    // Truncation keeps the low half whatever the sign, which is why
    // these are spelled `I` and have no signed encoding to accept.
    assert!(declines("vmovn.s16", &["d0", "q1"]));
    assert!(declines("vshrn.u16", &["d0", "q1", "4"]));
}

#[test]
fn the_long_forms_decline_a_source_that_is_not_narrow() {
    // A `q` source beside a `q` destination is the same-width family,
    // not this one; reading it as narrow would take four halfwords from
    // a register holding four words.
    assert!(declines("vmovl.s8", &["q0", "q1"]));
    assert!(declines("vaddl.s16", &["q0", "q1", "d4"]));
}

#[test]
fn the_narrowing_forms_decline_a_destination_that_is_not_narrow() {
    assert!(declines("vmovn.i16", &["q0", "q1"]));
}

#[test]
fn vmovl_declines_a_doubleword_source() {
    // Doubling it would need a 128-bit destination element, which no
    // arrangement spells.
    assert!(declines("vmovl.s64", &["q0", "d2"]));
}

#[test]
fn vshrn_declines_a_shift_past_the_destination_element() {
    // The encoding bounds the amount by the destination element's
    // width. It is also what makes the logical shift in the lowering
    // exact — past it, the bits shifted in from the top would reach the
    // half that is kept, and signedness would start to matter.
    assert!(declines("vshrn.i16", &["d0", "q1", "9"]));
    assert!(declines("vshrn.i16", &["d0", "q1", "0"]));
}

#[test]
fn vmull_declines_the_polynomial_element_type() {
    // `vmull.p8` is a carry-less multiply — a different lowering, not a
    // wider one.
    assert!(declines("vmull.p8", &["q0", "d2", "d4"]));
}

// --- the saturating narrows ---
//
// `vqmovn` / `vqmovun` clamp each double-width source element into the
// destination range rather than truncating it, so a value outside the
// range lands on the nearest endpoint instead of wrapping. The three
// spellings differ in what "the range" is: `vqmovn.s` keeps the signed
// destination range, `vqmovn.u` the unsigned one, and `vqmovun.s` reads
// a signed source but clamps into the unsigned range, so a negative
// source becomes zero.

#[test]
fn vqmovn_signed_clamps_each_element_to_the_signed_byte_range() {
    // Lanes 2, 3 exceed `+127` and lanes 6, 7 fall below `-128`; each
    // saturates to the nearest endpoint (`0x7f` / `0x80`) rather than
    // truncating, which would keep the wrong low byte.
    assert_computes(
        "vqmovn.s16",
        &["d0", "q1"],
        &[("v0", 0), ("v1", 0x8000_ff00_ff80_ffff_7fff_0080_007f_0000)],
        0x8080_80ff_7f7f_7f00,
    );
}

#[test]
fn vqmovn_unsigned_clamps_each_element_to_the_unsigned_byte_range() {
    assert_computes(
        "vqmovn.u16",
        &["d0", "q1"],
        &[("v0", 0), ("v1", 0x1000_00ab_0080_0001_ffff_0100_00ff_0000)],
        0xffab_8001_ffff_ff00,
    );
}

#[test]
fn vqmovun_sends_a_negative_source_to_zero() {
    // The signed source's negative lanes (4, 5) clamp to `0`, not to the
    // unsigned wrap `0xff`; the positive lanes above `255` clamp to
    // `0xff`.
    assert_computes(
        "vqmovun.s16",
        &["d0", "q1"],
        &[("v0", 0), ("v1", 0x7fff_00ab_8000_ffff_0100_00ff_007f_0000)],
        0xffab_0000_ffff_7f00,
    );
}

#[test]
fn vqmovn_on_a_half_register_preserves_the_other_half() {
    // The write to `d0` merges into `v0`, so the upper doubleword
    // survives.
    assert_computes(
        "vqmovn.s16",
        &["d0", "q1"],
        &[
            ("v0", 0xdead_beef_0000_0000_0000_0000_0000_0000),
            ("v1", 0x8000_ff00_ff80_ffff_7fff_0080_007f_0000),
        ],
        0xdead_beef_0000_0000_8080_80ff_7f7f_7f00,
    );
}

#[test]
fn vqmovun_declines_an_unsigned_source() {
    // `vqmovun` reads a signed source and clamps into the unsigned
    // range; there is no unsigned-source encoding.
    assert!(declines("vqmovun.u16", &["d0", "q1"]));
}

#[test]
fn vqmovn_declines_a_destination_that_is_not_narrow() {
    assert!(declines("vqmovn.s16", &["q0", "q1"]));
}

// ---------------------------------------------------------------
// The by-element forms
//
// The second source contributes one element to every destination lane
// rather than pairing lane with lane. These are blocked differently
// from the rest: `d4[2]` carries vector shape, so the whole instruction
// was declined before any mnemonic was looked at.
//
// Recognising them matters beyond coverage. `vmul.i16 q0, q1, d4[2]`
// has exactly the operand count and kinds the *lane-wise* multiply
// accepts, so a form the resolver misses is not declined — it is
// lowered as a lane-wise multiply reading all of `d4`, which is a wrong
// value. Every test below therefore differs from the lane-wise answer.
// ---------------------------------------------------------------

#[test]
fn vmul_by_element_broadcasts_one_lane_of_the_second_source() {
    // `d4` is the low half of `v2` and its lane 2 holds `0x000c`, so
    // every destination lane is its own source lane times twelve. A
    // lane-wise multiply would pair lane 4 with `d4`'s lane 0 instead.
    assert_computes(
        "vmul.i16",
        &["q0", "q1", "d4[2]"],
        &[
            ("v1", 0x0008_0007_0006_0005_0004_0003_0002_0001),
            ("v2", 0x000d_000c_000b_000a),
        ],
        0x0060_0054_0048_003c_0030_0024_0018_000c,
    );
}

#[test]
fn vmul_by_element_reads_the_index_the_operand_names() {
    // The teeth for the test above: the same registers, a different
    // index, and every lane changes. An index the lowering ignored
    // would give the same answer twice.
    assert_computes(
        "vmul.i16",
        &["q0", "q1", "d4[0]"],
        &[
            ("v1", 0x0008_0007_0006_0005_0004_0003_0002_0001),
            ("v2", 0x000d_000c_000b_000a),
        ],
        0x0050_0046_003c_0032_0028_001e_0014_000a,
    );
}

#[test]
fn vmla_by_element_accumulates_into_the_destination() {
    assert_computes(
        "vmla.i32",
        &["q0", "q1", "d4[1]"],
        &[
            ("v0", 0x0000_0064_0000_0063_0000_0062_0000_0061),
            ("v1", 0x0000_0004_0000_0003_0000_0002_0000_0001),
            ("v2", 0x0000_0007_0000_0006),
        ],
        0x0000_0080_0000_0078_0000_0070_0000_0068,
    );
}

#[test]
fn vmls_by_element_subtracts_the_product_from_the_destination() {
    assert_computes(
        "vmls.i32",
        &["q0", "q1", "d4[1]"],
        &[
            ("v0", 0x0000_0064_0000_0063_0000_0062_0000_0061),
            ("v1", 0x0000_0004_0000_0003_0000_0002_0000_0001),
            ("v2", 0x0000_0007_0000_0006),
        ],
        0x0000_0048_0000_004e_0000_0054_0000_005a,
    );
}

#[test]
fn vmull_by_element_widens_both_the_lane_and_the_element() {
    // The element is `0xffff`, which is `-1` signed, so each product is
    // the negated source lane at twice the width. Extending only the
    // lane-wise side would multiply by `65535` instead.
    assert_computes(
        "vmull.s16",
        &["q0", "d2", "d4[0]"],
        &[("v1", 0xffff_0002_8000_7fff), ("v2", 0x0000_0000_0000_ffff)],
        0x0000_0001_ffff_fffe_0000_8000_ffff_8001,
    );
}

#[test]
fn vmull_by_element_unsigned_reads_the_same_element_as_a_magnitude() {
    assert_computes(
        "vmull.u16",
        &["q0", "d2", "d4[0]"],
        &[("v1", 0xffff_0002_8000_7fff), ("v2", 0x0000_0000_0000_ffff)],
        0xfffe_0001_0001_fffe_7fff_8000_7ffe_8001,
    );
}

// --- the boundary of the by-element seam ---

#[test]
fn a_by_element_index_past_the_register_declines() {
    // `d4` holds four halfwords, so index 4 addresses nothing. The
    // register table maps `d4[4]` onto the whole of `d4` regardless,
    // which is exactly why this has to fail closed here.
    assert!(declines("vmul.i16", &["q0", "q1", "d4[4]"]));
    assert!(declines("vmla.i32", &["q0", "q1", "d4[2]"]));
}

#[test]
fn the_by_element_forms_decline_a_byte_element() {
    // No by-element encoding exists at byte size.
    assert!(declines("vmul.i8", &["q0", "q1", "d4[2]"]));
    assert!(declines("vmull.s8", &["q0", "d2", "d4[2]"]));
}

#[test]
fn the_long_by_element_forms_decline_the_sign_agnostic_spelling() {
    // Widening needs the extension named.
    assert!(declines("vmull.i16", &["q0", "d2", "d4[0]"]));
    assert!(declines("vmlal.i16", &["q0", "d2", "d4[0]"]));
}

// --- the float by-element forms ---
//
// Same width as their sources, not long, and rounded per step: `vmla` /
// `vmls` are the NEON (non-`vfma`) spelling, so the product rounds and
// then the destination accumulate rounds again.

// The registers are chosen so nothing aliases: `d0` is the low half of
// `v0`, `d2` the low half of `v1`, `d4` the low half of `v2`. Using
// `d1` would be the *upper* half of `v0` (q0 = d0:d1), which the
// destination write would then also touch.

#[test]
fn vmul_float_by_element_scales_every_lane_by_one_element() {
    // `d4[0]` is `3.0`, so each lane of `d2` = [2.0, 5.0] becomes
    // [6.0, 15.0].
    assert_computes(
        "vmul.f32",
        &["d0", "d2", "d4[0]"],
        &[
            // `d0` writes only the low half of `v0`; bind the whole
            // register so the preserved upper half is not a free input.
            ("v0", 0),
            ("v1", 0x0000_0000_0000_0000_40a0_0000_4000_0000),
            ("v2", 0x0000_0000_0000_0000_0000_0000_4040_0000),
        ],
        0x0000_0000_0000_0000_4170_0000_40c0_0000,
    );
}

#[test]
fn vmla_float_by_element_adds_into_the_destination_lane() {
    // `d0` = [1.0, 2.0] plus [2.0, 5.0] * 3.0 = [7.0, 17.0].
    assert_computes(
        "vmla.f32",
        &["d0", "d2", "d4[0]"],
        &[
            ("v0", 0x0000_0000_0000_0000_4000_0000_3f80_0000),
            ("v1", 0x0000_0000_0000_0000_40a0_0000_4000_0000),
            ("v2", 0x0000_0000_0000_0000_0000_0000_4040_0000),
        ],
        0x0000_0000_0000_0000_4188_0000_40e0_0000,
    );
}

#[test]
fn vmls_float_by_element_subtracts_the_product_from_the_destination() {
    // `d0` = [20.0, 40.0] minus [2.0, 5.0] * 3.0 = [14.0, 25.0].
    assert_computes(
        "vmls.f32",
        &["d0", "d2", "d4[0]"],
        &[
            ("v0", 0x0000_0000_0000_0000_4220_0000_41a0_0000),
            ("v1", 0x0000_0000_0000_0000_40a0_0000_4000_0000),
            ("v2", 0x0000_0000_0000_0000_0000_0000_4040_0000),
        ],
        0x0000_0000_0000_0000_41c8_0000_4160_0000,
    );
}

#[test]
fn vmul_float_by_element_reads_the_indexed_lane_across_a_quad() {
    // `d4[1]` is `4.0`; the quad destination has four f32 lanes
    // [1,2,3,4] → [4,8,12,16].
    assert_computes(
        "vmul.f32",
        &["q0", "q1", "d4[1]"],
        &[
            ("v1", 0x4080_0000_4040_0000_4000_0000_3f80_0000),
            ("v2", 0x0000_0000_0000_0000_4080_0000_0000_0000),
        ],
        0x4180_0000_4140_0000_4100_0000_4080_0000,
    );
}

#[test]
fn the_long_float_by_element_form_declines() {
    // There is no widening float by-element encoding.
    assert!(declines("vmull.f32", &["q0", "d2", "d4[0]"]));
}

#[test]
fn an_indexed_operand_outside_this_family_still_declines() {
    // The seam opened here is narrow: `vmov r0, d0[1]` moves a lane to
    // a general register and is not modelled, so it must keep failing
    // closed rather than reaching an integer handler that would read
    // all of `d0`.
    assert!(declines("vmov", &["r0", "d0[1]"]));
}

// ---------------------------------------------------------------
// `vtbl` / `vtbx` — table lookup
//
// Each destination byte selects a table byte by index. `vtbl` writes
// zero for an out-of-range index; `vtbx` preserves the destination byte
// there. The table `{d2, d3}` is `v1` in full, the index register `d4`
// is the low half of `v2`, and the destination `d0` the low half of
// `v0`, so nothing aliases.
// ---------------------------------------------------------------

#[test]
fn vtbl_selects_table_bytes_and_zeroes_out_of_range_indices() {
    // Table byte `i` holds `i`. Indices 16 and 255 name no byte and
    // become zero; the rest select their own value.
    assert_computes_into(
        "vtbl.8",
        &["d0", "{d2, d3}", "d4"],
        &[
            ("v0", 0),
            ("v1", 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100),
            ("v2", 0x0000_0000_0000_0000_0aff_0308_100f_0500),
        ],
        "v0",
        0x0000_0000_0000_0000_0a00_0308_000f_0500,
    );
}

#[test]
fn vtbx_preserves_the_destination_byte_for_out_of_range_indices() {
    // The same indices, but the out-of-range lanes 3 and 6 keep `d0`'s
    // prior bytes (`0xf3`, `0xf6`) instead of becoming zero.
    assert_computes_into(
        "vtbx.8",
        &["d0", "{d2, d3}", "d4"],
        &[
            ("v0", 0x0000_0000_0000_0000_f7f6_f5f4_f3f2_f1f0),
            ("v1", 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100),
            ("v2", 0x0000_0000_0000_0000_0aff_0308_100f_0500),
        ],
        "v0",
        0x0000_0000_0000_0000_0af6_0308_f30f_0500,
    );
}

#[test]
fn vtbl_reads_a_single_register_table() {
    // A one-register table has eight bytes, so index 8 already falls off
    // the end.
    assert_computes_into(
        "vtbl.8",
        &["d0", "{d2}", "d4"],
        &[
            ("v0", 0),
            ("v1", 0x0000_0000_0000_0000_0706_0504_0302_0100),
            ("v2", 0x0000_0000_0000_0000_0807_0605_0403_0201),
        ],
        "v0",
        0x0000_0000_0000_0000_0007_0605_0403_0201,
    );
}

#[test]
fn vtbl_declines_a_quad_destination() {
    // The destination is a `d` register; a `q` is not a table-lookup
    // shape.
    assert!(declines("vtbl.8", &["q0", "{d2, d3}", "d4"]));
}

#[test]
fn vtbl_declines_a_table_of_five_registers() {
    // The architecture lists at most four.
    assert!(declines("vtbl.8", &["d0", "{d2, d3, d4, d5, d6}", "d7"]));
}

// ---------------------------------------------------------------
// Structured loads and stores
//
// `vld1`–`vld4` / `vst1`–`vst4` move bytes rather than computing them,
// so the thing worth solving is the *layout*: which memory byte lands
// in which lane of which register. That cannot be observed from one
// instruction, so these store a known pattern and read it back — which
// also exercises the byte-granular memory model end to end.
// ---------------------------------------------------------------

/// Lift a sequence of instructions and ask what a named register holds
/// afterwards.
///
/// Each step gets its own address: the lifter names its temporaries
/// after the instruction address, so two instructions lifted at the
/// same one would collide.
fn solve_sequence(
    steps: &[(&str, &[&str])],
    sources: &[(&str, u128, u16)],
    destination: (&str, u16),
    expected: u128,
) -> SmtResult {
    let mut statements: Vec<IrStmt> = sources
        .iter()
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    for (step, (mnemonic, operands)) in steps.iter().enumerate() {
        let address = Address::new(0x1000 + 4 * step as u64);
        let insn = Instruction {
            address,
            size: 4,
            bytes: vec![],
            mnemonic: (*mnemonic).into(),
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
        statements.extend(lifted);
    }
    let (name, bits) = destination;
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(Expr::Var(Var::new(name, bits)), Expr::konst(expected, bits)),
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

fn assert_sequence(
    steps: &[(&str, &[&str])],
    sources: &[(&str, u128, u16)],
    destination: (&str, u16),
    expected: u128,
) {
    assert_eq!(
        solve_sequence(steps, sources, destination, expected),
        SmtResult::AlwaysTrue,
        "{steps:?} should leave {expected:#x} in {destination:?}"
    );
}

/// A base address and the two source registers holding a 32-byte ramp:
/// memory byte `k` ends up holding `k`.
const RAMP_SOURCES: [(&str, u128, u16); 3] = [
    ("r0", 0x1000, 32),
    ("v0", BYTE_RAMP, 128),
    ("v1", BYTE_RAMP_HIGH, 128),
];

#[test]
fn vst1_then_vld1_round_trips_the_bytes_unchanged() {
    // `vld1` is contiguous: the list is one block, so what went in
    // comes back in the same lanes. This is also the baseline the
    // de-interleaving tests below are read against.
    assert_sequence(
        &[
            ("vst1.8", &["{d0, d1}", "[r0]"]),
            ("vld1.8", &["{d2, d3}", "[r0]"]),
        ],
        &RAMP_SOURCES,
        ("v1", 128),
        BYTE_RAMP,
    );
}

#[test]
fn vld2_sends_alternate_bytes_to_alternate_registers() {
    // Two structures interleaved: the first register collects the even
    // memory bytes and the second the odd ones. A contiguous read would
    // give back the ramp instead.
    assert_sequence(
        &[
            ("vst1.8", &["{d0, d1}", "[r0]"]),
            ("vld2.8", &["{d2, d3}", "[r0]"]),
        ],
        &RAMP_SOURCES,
        ("v1", 128),
        0x0f0d_0b09_0705_0301_0e0c_0a08_0604_0200,
    );
}

#[test]
fn vld3_takes_every_third_byte_into_each_register() {
    assert_sequence(
        &[
            ("vst1.8", &["{d0, d1, d2, d3}", "[r0]"]),
            ("vld3.8", &["{d8, d9, d10}", "[r0]"]),
        ],
        &RAMP_SOURCES,
        ("v4", 128),
        0x1613_100d_0a07_0401_1512_0f0c_0906_0300,
    );
}

#[test]
fn vld4_takes_every_fourth_byte_into_each_register() {
    assert_sequence(
        &[
            ("vst1.8", &["{d0, d1, d2, d3}", "[r0]"]),
            ("vld4.8", &["{d8, d9, d10, d11}", "[r0]"]),
        ],
        &RAMP_SOURCES,
        ("v4", 128),
        0x1d19_1511_0d09_0501_1c18_1410_0c08_0400,
    );
}

#[test]
fn vld4_fills_its_third_and_fourth_registers_too() {
    // The tail of the same instruction: a lowering that stopped after
    // two registers would leave these holding their prior value.
    assert_sequence(
        &[
            ("vst1.8", &["{d0, d1, d2, d3}", "[r0]"]),
            ("vld4.8", &["{d8, d9, d10, d11}", "[r0]"]),
        ],
        &RAMP_SOURCES,
        ("v5", 128),
        0x1f1b_1713_0f0b_0703_1e1a_1612_0e0a_0602,
    );
}

#[test]
fn vst2_interleaves_on_the_way_out() {
    // The store side of the same layout: written interleaved and read
    // back contiguously, the bytes come out in the order `vld2` would
    // have undone.
    assert_sequence(
        &[
            ("vst2.8", &["{d0, d1}", "[r0]"]),
            ("vld1.8", &["{d2, d3}", "[r0]"]),
        ],
        &RAMP_SOURCES,
        ("v1", 128),
        0x0f07_0e06_0d05_0c04_0b03_0a02_0901_0800,
    );
}

#[test]
fn a_structured_access_advances_its_base_by_what_it_transferred() {
    // `[r0]!` is the immediate post-index: two registers is sixteen
    // bytes, whatever the element width.
    assert_sequence(
        &[("vld1.8", &["{d2, d3}", "[r0]!"])],
        &RAMP_SOURCES,
        ("r0", 32),
        0x1010,
    );
}

#[test]
fn a_four_register_access_advances_the_base_by_thirty_two() {
    assert_sequence(
        &[("vld1.32", &["{d2, d3, d4, d5}", "[r0]!"])],
        &RAMP_SOURCES,
        ("r0", 32),
        0x1020,
    );
}

// --- the boundary of the structured seam ---

#[test]
fn a_structured_access_declines_an_alignment_specifier() {
    // Disassemblers spell alignment more than one way (`[r0:64]`,
    // `[r0@64]`), and it constrains the address rather than describing
    // the transfer, so anything but a bare register inside the brackets
    // fails closed rather than being guessed at.
    assert!(declines("vld1.8", &["{d0, d1}", "[r0:64]"]));
    assert!(declines("vld1.8", &["{d0, d1}", "[r0@128]"]));
}

#[test]
fn a_structured_access_declines_the_single_element_shape() {
    // `{d0[3]}` transfers one element rather than whole registers.
    assert!(declines("vld1.8", &["{d0[3]}", "[r0]"]));
    assert!(declines("vld2.8", &["{d0[1], d1[1]}", "[r0]"]));
}

#[test]
fn a_structured_access_declines_a_non_consecutive_list() {
    // The stride-two spellings are a different encoding; reading them
    // as consecutive would send the bytes to the wrong registers.
    assert!(declines("vld2.8", &["{d0, d2}", "[r0]"]));
}

#[test]
fn an_interleaved_access_declines_a_list_that_is_not_its_width() {
    // A `vld3` names exactly three registers. The double-width `vld2`
    // spelling with four is a separate encoding.
    assert!(declines("vld3.8", &["{d0, d1}", "[r0]"]));
    assert!(declines("vld2.8", &["{d0, d1, d2, d3}", "[r0]"]));
}

#[test]
fn an_interleaved_access_declines_a_doubleword_element() {
    // One structure would fill the whole register, so the architecture
    // gives the interleaved family no such encoding.
    assert!(declines("vld2.64", &["{d0, d1}", "[r0]"]));
    assert!(declines("vld4.64", &["{d0, d1, d2, d3}", "[r0]"]));
}

#[test]
fn a_structured_access_declines_a_register_post_index() {
    // `[r0], r1` advances the base by a run-time value, which the
    // constant writeback cannot spell.
    assert!(declines("vld1.8", &["{d0, d1}", "[r0]", "r1"]));
}

#[test]
fn the_structured_decline_assertions_have_teeth() {
    // Positive control for the block above. Every one of those
    // assertions names a memory operand, and the resolver refuses an
    // operand that is not `OperandKind::Memory` — so if the fixture
    // classified `[r0]` as a register they would all pass without
    // reaching the check they name. This is the same shape that does
    // not decline.
    assert!(!declines("vld1.8", &["{d0, d1}", "[r0]"]));
    assert!(!declines("vld1.8", &["{d0, d1}", "[r0]!"]));
    assert!(!declines("vld4.8", &["{d0, d1, d2, d3}", "[r0]"]));
}
