//! The arch-neutral SIMD / packed-vector core of the lifter.
//!
//! Everything here is expressed in four concepts and nothing else: a
//! vector **parent** register, an operand's **view** of it, the **lanes**
//! that tile that view, and the **operation** applied to a lane. None of
//! those is ISA-specific, which is why `addps xmm0, xmm1` and
//! `fadd v0.4s, v1.4s, v2.4s` are the same computation over the same
//! model here — the per-ISA modules differ only in how they derive the
//! lane width (from the mnemonic on x86, from the arrangement on ARM)
//! and in the upper-bits rule they pass to [`LiftCtx::write_simd_dst`].
//!
//! Two consequences are worth stating because they are easy to break.
//! A lane offset is measured from the *view's* offset in the parent, not
//! from bit zero: `AArch32` puts `d1` at bits `[127:64]` of `v0` and `s3`
//! at `[127:96]`, so indexing from zero would silently read the wrong
//! half of the register. And an operand is materialised **once**, with
//! the lanes extracted from that value, so a memory operand costs one
//! load rather than one per lane.
//!
//! The one question this module cannot answer for itself is whether a
//! memory operand's address is modellable — that belongs to a per-ISA
//! memory model, and it arrives through the
//! [`LiftCtx::is_modellable_simd_memory`] hook next to the lifter's other
//! arch dispatch. This module therefore names no [`r2smt_common::Arch`]
//! variant at all.

use r2smt_ir::expr::{Expr, RoundingMode, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::effect::memory_operand_width;
use crate::registers::{RegisterLayout, is_simd_parent, register_layout, simd_parent_bits};

use super::{BinOp, LiftCtx};

/// Integer operation a packed ARM vector instruction applies to each
/// lane.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PackedIntOp {
    /// A lane-wise binary operation over two sources.
    Bin(BinOp),
    /// `bic` — `a & ~b`.
    BitClear,
    /// `mvn` / `not` — `~a`, one source.
    Not,
    /// `mov` — copy, one source.
    Copy,
    /// `vmin` / `vmax` — the lane-wise ordered select. The comparison
    /// is the whole operation, so its signedness is carried here rather
    /// than inferred: `max` of `0xff` and `0x01` is the first byte
    /// unsigned and the second signed.
    MinMax { max: bool, signed: bool },
    /// `vabs` on integer lanes. Non-saturating, so the most negative
    /// value maps to itself — which the two's-complement negation
    /// already gives at a fixed width, with no special case.
    Abs,
    /// `vneg` on integer lanes — `0 - a`, wrapping for the same reason.
    Neg,
    /// `vabs` / `vneg` on floating-point lanes. Both are sign-bit
    /// manipulations, so they are exact at every value including the
    /// NaNs and the infinities and need no float sort at all.
    SignBit { negate: bool },
    /// `vqadd` / `vqsub` — saturating, clamped to the element's range
    /// rather than wrapped.
    Saturating { subtract: bool, signed: bool },
    /// `vhadd` / `vhsub` / `vrhadd` — halving, the exact sum or
    /// difference shifted down one bit. `rounding` adds a half first,
    /// which only the add form (`vrhadd`) has an encoding for.
    Halving {
        subtract: bool,
        signed: bool,
        rounding: bool,
    },
}

/// What a packed ARM vector data-processing instruction computes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PackedOp {
    /// Integer lanes.
    Int(PackedIntOp),
    /// IEEE floating-point lanes.
    Fp(FpArithOp),
    /// Multiply-accumulate: `dst := dst ± a * b`, lane-wise.
    ///
    /// A variant of its own rather than a [`PackedIntOp`] or an
    /// [`FpArithOp`], because both of those describe one lane operation
    /// over at most two sources and this one reads its *destination* as
    /// a third.
    Accumulate {
        /// Whether the lanes are floating-point.
        float: bool,
        /// `vmls` subtracts the product where `vmla` adds it.
        subtract: bool,
    },
    /// A shift whose amount is the same for every lane, taken from the
    /// instruction's last operand rather than from a vector register.
    ///
    /// `left` is `vshl`; the right-shifting forms are `vshr` and, when
    /// they add the shifted value into the destination, `vsra`.
    ShiftImmediate {
        /// Direction. A left shift needs no signedness — the bits it
        /// discards are the same either way.
        left: bool,
        /// Whether a right shift replicates the sign bit.
        signed: bool,
        /// `vsra` adds the shifted lane into the destination.
        accumulate: bool,
        /// `vrshr` / `vrsra` add half an ulp of the shift before
        /// discarding the low bits — the rounding right shifts. Only the
        /// right-shifting forms carry it.
        rounding: bool,
    },
    /// `vshl` with a per-lane amount read from a vector register, whose
    /// *sign* chooses the direction.
    ShiftRegister {
        /// Whether a right shift replicates the sign bit.
        signed: bool,
    },
    /// `vqshl` with an immediate amount — shift left and clamp to the
    /// element's range rather than letting the overflow wrap.
    SaturatingShiftLeftImmediate {
        /// Whether the element is signed, which selects both the
        /// extension and the saturation bounds.
        signed: bool,
    },
}

impl PackedOp {
    /// Number of operands the instruction's packed form carries,
    /// destination included.
    pub(super) const fn operand_count(self) -> usize {
        match self {
            Self::Int(
                PackedIntOp::Not
                | PackedIntOp::Copy
                | PackedIntOp::Abs
                | PackedIntOp::Neg
                | PackedIntOp::SignBit { .. },
            ) => 2,
            _ => 3,
        }
    }

    /// Whether applying the operation to the whole vector view at once
    /// yields the same bits as applying it to each lane separately.
    ///
    /// True for the bitwise family, where no carry crosses a lane
    /// boundary — lowering those lane by lane would multiply the formula
    /// the solver sees by the lane count (sixteen extracts and fifteen
    /// concatenations for a `.16b` operand) for an identical result.
    /// False for the arithmetic family, where the lane boundary is
    /// exactly what stops the carry.
    const fn is_lane_independent(self) -> bool {
        matches!(
            self,
            Self::Int(
                PackedIntOp::Bin(BinOp::And | BinOp::Or | BinOp::Xor)
                    | PackedIntOp::BitClear
                    | PackedIntOp::Not
                    | PackedIntOp::Copy
            )
        )
    }
}

/// Floating-point operation applied to one lane, shared by the scalar
/// (`addss`) and packed (`addps`) handlers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FpArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// IEEE interchange formats the pipeline can name and render.
const FP_HALF_BITS: u16 = 16;
/// IEEE binary32.
const FP_SINGLE_BITS: u16 = 32;
/// IEEE binary64.
pub(crate) const FP_DOUBLE_BITS: u16 = 64;

/// IEEE `(ebits, sbits)` sort for an FP lane width: 16→half `(5, 11)`,
/// 32→single `(8, 24)`, 64→double `(11, 53)`.
///
/// `None` for every other width, which is the only honest answer: the
/// IR carries the sort verbatim into the solver, so resolving an
/// unrecognised width to *some* sort would silently reinterpret the
/// value. x87's 80-bit double-extended format is exactly that case —
/// it is a real format this pipeline cannot render, and it has to
/// decline rather than be read as a double.
pub(crate) fn fp_sort_bits_checked(lane_bits: u16) -> Option<(u16, u16)> {
    Some(match lane_bits {
        FP_HALF_BITS => (5, 11),
        FP_SINGLE_BITS => (8, 24),
        FP_DOUBLE_BITS => (11, 53),
        _ => return None,
    })
}

/// Apply `op` to one lane, given both operands as raw IEEE bit-vectors
/// of `lane_bits`, and return the lane result in the same form.
///
/// The result is a bit-vector rather than a float because `max`/`min`
/// *select* an operand instead of computing one, and the IR's `Ite` is
/// bit-vector-typed — an `Ite` over float-sorted branches has no
/// rendering.
///
/// The rounding mode is the architectural MXCSR default; a function
/// that reprograms MXCSR is rejected by the slicer's guard rather than
/// silently rounded here.
///
/// `None` for a lane width with no IEEE sort. Callers derive the width
/// from an operand's vector view, which can name a width no float
/// format has, and reading such a lane as a double would be a definite
/// wrong value rather than a decline.
pub(crate) fn fp_lane_result(
    op: FpArithOp,
    a_bits: Expr,
    b_bits: Expr,
    lane_bits: u16,
) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
    let a = Expr::bv_to_fp(a_bits.clone(), ebits, sbits);
    let b = Expr::bv_to_fp(b_bits.clone(), ebits, sbits);
    let rm = RoundingMode::NearestTiesEven;
    let arith = match op {
        FpArithOp::Add => Expr::fadd(a, b, rm),
        FpArithOp::Sub => Expr::fsub(a, b, rm),
        FpArithOp::Mul => Expr::fmul(a, b, rm),
        FpArithOp::Div => Expr::fdiv(a, b, rm),
        // `MAXPS: IF SRC1 > SRC2 THEN DEST := SRC1 ELSE DEST := SRC2`
        // (and the mirror for MIN). The comparison is false when either
        // operand is NaN, so the second operand wins on unordered *and*
        // on equality — which is why this is an explicit select rather
        // than `fp.max`/`fp.min`, whose NaN and signed-zero behaviour
        // differs from x86's.
        FpArithOp::Max | FpArithOp::Min => {
            let cond = if matches!(op, FpArithOp::Max) {
                Expr::flt(b, a)
            } else {
                Expr::flt(a, b)
            };
            return Some(Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(a_bits),
                else_expr: Box::new(b_bits),
            });
        }
    };
    Some(Expr::fp_to_ieee_bv(arith))
}

/// An `AArch64` fused multiply step: the product and the following
/// combine round together, once, where a separate `fmul` then `fadd`
/// would round twice.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FusedStep {
    /// `fmla` — accumulator + a·b.
    MulAdd,
    /// `fmls` — accumulator − a·b.
    MulSub,
    /// `frecps` — `2.0 − a·b`, the Newton–Raphson reciprocal step.
    RecipStep,
    /// `frsqrts` — `(3.0 − a·b) / 2.0`, the reciprocal-square-root step.
    RsqrtStep,
}

impl FusedStep {
    /// Whether the destination lane is an input (the accumulator).
    pub(crate) const fn reads_accumulator(self) -> bool {
        matches!(self, Self::MulAdd | Self::MulSub)
    }
}

/// IEEE binary128, the format a binary64 lane's fused step is computed
/// in.
///
/// Deliberately absent from [`fp_sort_bits_checked`]: that table answers
/// "what float is a lane of this width", and no ISA here has a 128-bit
/// float lane. This sort exists only as an intermediate, which is also
/// why it costs nothing on the text backends — `is_renderable_fp_sort`
/// already refuses it, measured below.
const FP_QUAD_SORT: (u16, u16) = (15, 113);

/// binary64 bit patterns of the step constants — the intermediate
/// format for a binary32 lane, and the lane format for a binary64 one,
/// which is why `F64_TWO` serves both roles.
const F64_TWO: u128 = 0x4000_0000_0000_0000;
const F64_THREE: u128 = 0x4008_0000_0000_0000;
/// The same two in binary128, the intermediate for a binary64 lane.
const F128_TWO: u128 = 0x4000_0000_0000_0000_0000_0000_0000_0000;
const F128_THREE: u128 = 0x4000_8000_0000_0000_0000_0000_0000_0000;
/// binary32 bit patterns of the `(inf, 0)` special-case results.
const F32_TWO: u128 = 0x4000_0000;
const F32_ONE_POINT_FIVE: u128 = 0x3fc0_0000;
/// The binary64 spelling of the second (the first is `F64_TWO`).
const F64_ONE_POINT_FIVE: u128 = 0x3ff8_0000_0000_0000;
/// binary32 magnitude mask and the infinity pattern with the sign
/// cleared.
const F32_ABS_MASK: u128 = 0x7fff_ffff;
const F32_INFINITY: u128 = 0x7f80_0000;
/// The binary64 spelling of the same two.
const F64_ABS_MASK: u128 = 0x7fff_ffff_ffff_ffff;
const F64_INFINITY: u128 = 0x7ff0_0000_0000_0000;

/// The two formats one fused step is computed in, plus the constants
/// each of them names.
#[derive(Debug, Clone, Copy)]
struct FusedFormats {
    /// Width of the architectural lane, in bits.
    lane_bits: u16,
    /// IEEE sort of the lane.
    lane: (u16, u16),
    /// IEEE sort the step is computed in before rounding once back.
    wide: (u16, u16),
    /// `2.0` and `3.0` in the wide format — the reciprocal steps' terms.
    wide_two: u128,
    wide_three: u128,
    /// `2.0` and `1.5` in the lane format — the `inf · 0` answers.
    lane_two: u128,
    lane_one_and_a_half: u128,
    /// Magnitude mask and infinity pattern in the lane format.
    lane_abs_mask: u128,
    lane_infinity: u128,
}

/// The formats a fused step at `lane_bits` is exactly emulable in, or
/// `None` when no wide-enough format is available.
///
/// Two conditions make the emulation exact, and both are why this is a
/// closed table rather than "the next format up". The product of two
/// lane values must fit the wide significand (`2·p − 1 ≤ q`), and the
/// wide format's single rounding of the combine must agree with
/// rounding the real result straight to the lane, which needs
/// `q ≥ 2·p + 2`. binary64 over a binary32 lane gives `53 ≥ 50`;
/// binary128 over a binary64 one gives `113 ≥ 108`. There is no third
/// entry: IEEE names no interchange format wide enough to carry a
/// binary128 lane the same way.
fn fused_formats(lane_bits: u16) -> Option<FusedFormats> {
    let lane = fp_sort_bits_checked(lane_bits)?;
    Some(match lane_bits {
        FP_SINGLE_BITS => FusedFormats {
            lane_bits,
            lane,
            wide: fp_sort_bits_checked(FP_DOUBLE_BITS)?,
            wide_two: F64_TWO,
            wide_three: F64_THREE,
            lane_two: F32_TWO,
            lane_one_and_a_half: F32_ONE_POINT_FIVE,
            lane_abs_mask: F32_ABS_MASK,
            lane_infinity: F32_INFINITY,
        },
        FP_DOUBLE_BITS => FusedFormats {
            lane_bits,
            lane,
            wide: FP_QUAD_SORT,
            wide_two: F128_TWO,
            wide_three: F128_THREE,
            lane_two: F64_TWO,
            lane_one_and_a_half: F64_ONE_POINT_FIVE,
            lane_abs_mask: F64_ABS_MASK,
            lane_infinity: F64_INFINITY,
        },
        _ => return None,
    })
}

/// Whether a fused multiply step over `lane_bits`-wide lanes is exactly
/// emulable, and so whether a resolver may claim that geometry.
///
/// Delegates to [`fused_formats`] rather than restating the widths,
/// because the per-ISA resolvers used to restate them and drifted: the
/// binary128 intermediate that opened binary64 lanes landed here while
/// two `== 32` comparisons kept the resolvers on binary32 only. A
/// resolver that answers this question by asking the lowering cannot
/// drift from it again.
pub(crate) fn fused_step_is_emulable(lane_bits: u16) -> bool {
    fused_formats(lane_bits).is_some()
}

/// One lane of a fused multiply step, computed once in the wider format
/// [`fused_formats`] names and rounded once back to the lane.
///
/// The intermediate is exact — see [`fused_formats`] for the two width
/// conditions — so this reproduces the architecture's single rounding
/// rather than the double rounding a separate `fmul` + `fadd` would
/// give. `acc_bits` is the accumulator lane for `fmla` / `fmls`; the
/// step forms ignore it.
///
/// `frecps` / `frsqrts` return `2.0` / `1.5` when one operand is zero and
/// the other infinite, where the fused arithmetic alone would give a NaN
/// — the one input on which the naive lowering is a wrong *value* rather
/// than a wider one, so it is guarded explicitly.
///
/// `None` for a lane width with no wide-enough intermediate, so the
/// caller declines rather than round twice.
pub(crate) fn fused_multiply_lane(
    step: FusedStep,
    a_bits: &Expr,
    b_bits: &Expr,
    acc_bits: Option<Expr>,
    lane_bits: u16,
) -> Option<Expr> {
    let formats = fused_formats(lane_bits)?;
    let (lane_e, lane_s) = formats.lane;
    let (wide_e, wide_s) = formats.wide;
    let rm = RoundingMode::NearestTiesEven;
    let widen =
        |bits: Expr| Expr::fp_to_fp(Expr::bv_to_fp(bits, lane_e, lane_s), rm, wide_e, wide_s);
    let wide_const = |bits: u128| Expr::FpConst {
        bits,
        ebits: wide_e,
        sbits: wide_s,
    };
    let product = Expr::fmul(widen(a_bits.clone()), widen(b_bits.clone()), rm);
    let value = match step {
        FusedStep::MulAdd => Expr::fadd(widen(acc_bits?), product, rm),
        FusedStep::MulSub => Expr::fsub(widen(acc_bits?), product, rm),
        FusedStep::RecipStep => Expr::fsub(wide_const(formats.wide_two), product, rm),
        FusedStep::RsqrtStep => Expr::fdiv(
            Expr::fsub(wide_const(formats.wide_three), product, rm),
            wide_const(formats.wide_two),
            rm,
        ),
    };
    let narrowed = Expr::fp_to_ieee_bv(Expr::fp_to_fp(value, rm, lane_e, lane_s));
    Some(match step {
        FusedStep::MulAdd | FusedStep::MulSub => narrowed,
        FusedStep::RecipStep => {
            zero_times_infinity_result(a_bits, b_bits, formats.lane_two, narrowed, &formats)
        }
        FusedStep::RsqrtStep => zero_times_infinity_result(
            a_bits,
            b_bits,
            formats.lane_one_and_a_half,
            narrowed,
            &formats,
        ),
    })
}

/// Select `result_bits` when one lane operand is zero and the other
/// infinite, otherwise `otherwise` — the `frecps` / `frsqrts` special
/// case, tested on the raw bit patterns since the IR has no is-infinite
/// or is-zero predicate.
fn zero_times_infinity_result(
    a_bits: &Expr,
    b_bits: &Expr,
    result_bits: u128,
    otherwise: Expr,
    formats: &FusedFormats,
) -> Expr {
    let width = formats.lane_bits;
    let magnitude =
        |bits: &Expr| Expr::bv_and(bits.clone(), Expr::konst(formats.lane_abs_mask, width));
    let is_zero = |bits: &Expr| Expr::eq(magnitude(bits), Expr::konst(0, width));
    let is_infinite =
        |bits: &Expr| Expr::eq(magnitude(bits), Expr::konst(formats.lane_infinity, width));
    let special = Expr::bool_or(
        Expr::bool_and(is_infinite(a_bits), is_zero(b_bits)),
        Expr::bool_and(is_zero(a_bits), is_infinite(b_bits)),
    );
    Expr::Ite {
        cond: Box::new(special),
        then_expr: Box::new(Expr::konst(result_bits, width)),
        else_expr: Box::new(otherwise),
    }
}

/// Lane result of a floating-point max / min that **propagates** NaN
/// and resolves a signed-zero tie by combining the two signs, rather
/// than selecting an operand on a bare comparison.
///
/// This is ARM's `FPMax` / `FPMin` (ARM ARM J1.3), and it is emphatically
/// not [`fp_lane_result`]'s `Max` / `Min`, which implement Intel's
/// `MAXPS`. The two differ on the inputs that matter most:
///
/// - **NaN.** `MAXPS` returns the *second operand* when either is NaN,
///   which for a NaN in the first operand means a perfectly ordinary
///   number comes out. `FPMax` returns a NaN. Reusing the x86 helper
///   here would therefore not be an approximation but a wrong value.
/// - **Signed zero.** `MAXPS(+0, -0)` is `-0`, because the comparison
///   is false and the second operand wins. `FPMax(+0, -0)` is `+0`: the
///   architecture combines the two signs, `AND` for max and `OR` for
///   min, which is what makes it order-independent.
///
/// The tie is expressed as a bitwise `AND` / `OR` of the two patterns
/// rather than as an explicit sign-bit rebuild. It comes to the same
/// thing: an IEEE interchange format has no redundant encodings, so two
/// operands that compare equal are bit-identical *except* for the
/// zeros, whose non-sign bits are all clear.
///
/// NaN is propagated in the architecture's priority order — a
/// signalling operand first, quieted by setting the leading significand
/// bit, then a quiet one, first operand before second in each case.
/// Quieting assumes `FPCR.DN` holds its reset value of zero; a function
/// that writes FPCR is already rejected by the slicer's
/// rounding-control guard, which is the same register.
///
/// `number_wins` selects the `FPMaxNum` / `FPMinNum` behaviour instead:
/// a *quiet* NaN in exactly one operand yields the other, numeric one,
/// where `FPMax` would propagate the NaN. The signalling-NaN priority
/// and the both-NaN case are unchanged — a signalling operand is still
/// quieted and propagated, and two NaNs still yield the first — so only
/// the two innermost quiet-NaN arms flip.
///
/// `None` for a lane width with no IEEE sort.
pub(crate) fn fp_propagating_max_min(
    a_bits: Expr,
    b_bits: Expr,
    lane_bits: u16,
    max: bool,
    number_wins: bool,
) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
    let a = Expr::bv_to_fp(a_bits.clone(), ebits, sbits);
    let b = Expr::bv_to_fp(b_bits.clone(), ebits, sbits);
    // The quiet bit is the most significant bit of the stored
    // significand, which `sbits` counts with the implicit bit included.
    let quiet_bit = Expr::konst(1u128 << u32::from(sbits.checked_sub(2)?), lane_bits);
    let select = |cond: Expr, then_expr: Expr, else_expr: Expr| Expr::Ite {
        cond: Box::new(cond),
        then_expr: Box::new(then_expr),
        else_expr: Box::new(else_expr),
    };
    let signalling = |value: &Expr, bits: &Expr| {
        Expr::bool_and(
            Expr::fisnan(value.clone()),
            Expr::eq(
                Expr::bv_and(bits.clone(), quiet_bit.clone()),
                Expr::konst(0, lane_bits),
            ),
        )
    };
    let (first_wins, second_wins) = if max {
        (
            Expr::flt(b.clone(), a.clone()),
            Expr::flt(a.clone(), b.clone()),
        )
    } else {
        (
            Expr::flt(a.clone(), b.clone()),
            Expr::flt(b.clone(), a.clone()),
        )
    };
    let tied = if max {
        Expr::bv_and(a_bits.clone(), b_bits.clone())
    } else {
        Expr::bv_or(a_bits.clone(), b_bits.clone())
    };
    let numeric = select(
        first_wins,
        a_bits.clone(),
        select(second_wins, b_bits.clone(), tied),
    );
    let a_signalling = signalling(&a, &a_bits);
    let b_signalling = signalling(&b, &b_bits);
    let a_quieted = Expr::bv_or(a_bits.clone(), quiet_bit.clone());
    let b_quieted = Expr::bv_or(b_bits.clone(), quiet_bit);
    // Neither operand is signalling here. `FPMax` propagates a quiet NaN;
    // `FPMaxNum` returns the numeric operand when exactly one is a quiet
    // NaN, and still yields the first when both are.
    let quiet = if number_wins {
        select(
            Expr::fisnan(a),
            select(Expr::fisnan(b.clone()), a_bits.clone(), b_bits.clone()),
            select(Expr::fisnan(b), a_bits, numeric),
        )
    } else {
        select(
            Expr::fisnan(a),
            a_bits,
            select(Expr::fisnan(b), b_bits, numeric),
        )
    };
    Some(select(
        a_signalling,
        a_quieted,
        select(b_signalling, b_quieted, quiet),
    ))
}

/// Apply an integer packed operation to one lane (or, for the
/// lane-independent operations, to a whole view) of `bits` width.
///
/// `None` when a two-source operation was handed no second source — an
/// operand-count mismatch the caller declines on rather than inventing a
/// value for.
/// An all-ones bit-vector of `bits`.
///
/// A `Const` carries a `u128` payload, so a mask wider than 128 bits
/// cannot be one literal and is built by concatenating full chunks.
/// Getting that wrong is silent rather than loud: `konst(u128::MAX,
/// 256)` is a *zero-extended* `u128::MAX`, which denotes a perfectly
/// valid — and wrong — mask.
pub(super) fn all_ones(bits: u16) -> Expr {
    if bits > 128 {
        return Expr::concat(all_ones(bits - 128), Expr::konst(u128::MAX, 128));
    }
    let mask = if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    Expr::konst(mask, bits)
}

/// The lane-wise compares.
///
/// Each writes a mask, not a condition flag: the whole point is that the
/// result feeds another vector operation.
///
/// Arch-neutral because both ISAs need the same axes: `AArch64` spells
/// them `cmeq` / `cmgt` / `cmhi` / `cmtst`, x86 `pcmpeqb` / `pcmpgtb`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CompareKind {
    /// `cmeq` / `fcmeq` / `pcmpeq*` — equality.
    Equal { float: bool },
    /// `cmgt` / `cmge` / `cmhi` / `cmhs` / `pcmpgt*` and the float
    /// `fcmgt` / `fcmge` — ordered comparison. `or_equal` picks `ge`
    /// over `gt`.
    Ordered {
        float: bool,
        signed: bool,
        or_equal: bool,
    },
    /// `cmtst` — true where the bitwise AND of the lanes is non-zero.
    TestBits,
}

/// One destination lane of a compare: an all-ones mask where the
/// predicate holds and all-zeros where it does not.
///
/// A vector compare produces a *value*, not a flag, because its result
/// feeds another vector operation — which is why this cannot reuse the
/// scalar compare path that writes NZCV.
pub(crate) fn compare_lane(kind: CompareKind, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    let predicate = match kind {
        CompareKind::Equal { float: false } => Expr::eq(a, b),
        CompareKind::Equal { float: true } => {
            let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
            Expr::feq(
                Expr::bv_to_fp(a, ebits, sbits),
                Expr::bv_to_fp(b, ebits, sbits),
            )
        }
        CompareKind::Ordered {
            float: true,
            or_equal,
            ..
        } => {
            let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
            let (x, y) = (
                Expr::bv_to_fp(a, ebits, sbits),
                Expr::bv_to_fp(b, ebits, sbits),
            );
            // `fcmgt`/`fcmge` are *ordered*: unordered compares false,
            // which `fp.lt` / `fp.leq` already give.
            if or_equal {
                Expr::fle(y, x)
            } else {
                Expr::flt(y, x)
            }
        }
        CompareKind::Ordered {
            float: false,
            signed,
            or_equal,
        } => match (signed, or_equal) {
            (true, false) => Expr::slt(b, a),
            (true, true) => Expr::sle(b, a),
            (false, false) => Expr::ult(b, a),
            (false, true) => Expr::ule(b, a),
        },
        CompareKind::TestBits => Expr::ne(Expr::bv_and(a, b), Expr::konst(0, lane_bits)),
    };
    Some(Expr::Ite {
        cond: Box::new(predicate),
        then_expr: Box::new(all_ones(lane_bits)),
        else_expr: Box::new(Expr::konst(0, lane_bits)),
    })
}

/// A `bits`-wide value with only the most significant bit set.
///
/// Built by concatenation rather than as a literal so it stays correct
/// above the 128 bits a `Const` payload can carry.
fn sign_bit(bits: u16) -> Option<Expr> {
    Some(Expr::concat(
        Expr::konst(1, 1),
        Expr::konst(0, bits.checked_sub(1)?),
    ))
}

/// A `bits`-wide value with every bit but the most significant set —
/// the mask a floating-point `abs` applies.
fn magnitude_mask(bits: u16) -> Option<Expr> {
    Some(Expr::concat(
        Expr::konst(0, 1),
        all_ones(bits.checked_sub(1)?),
    ))
}

/// Extra bits the saturating and halving lanes compute in.
///
/// **Two**, not one, and the difference is load-bearing. One extra bit
/// holds the exact sum for either signedness, but not as a value the
/// *signed* comparisons below can read: an unsigned `0xff + 0xff` fills
/// nine bits, whose top bit a signed compare would take for a sign. Two
/// bits let one set of signed comparisons serve both signednesses, which
/// is what keeps the clamp from needing a second, mirrored copy.
const WIDE_HEADROOM: u16 = 2;

/// `value` as a `wide`-bit two's-complement constant.
fn wide_const(value: i128, wide: u16) -> Option<Expr> {
    let modulus = 1u128.checked_shl(u32::from(wide))?;
    let pattern = u128::from_le_bytes(value.to_le_bytes());
    Some(Expr::konst(pattern & (modulus - 1), wide))
}

/// Widen a lane to the headroom width, preserving its value under the
/// element's own signedness.
fn extend_lane(value: Expr, wide: u16, signed: bool) -> Expr {
    if signed {
        Expr::sign_ext(value, wide)
    } else {
        Expr::zero_ext(value, wide)
    }
}

/// One destination lane of `vqshl` with an immediate amount.
///
/// The shift happens where it cannot overflow — a lane shifted left by
/// at most its own width needs twice the bits, plus one so an unsigned
/// maximum stays inside the signed range the clamp compares in — and the
/// element bounds bring it back. Clamping at the element width would be
/// too late, since the overflow this instruction saturates on would
/// already have wrapped.
fn saturating_shift_left_lane(signed: bool, value: Expr, amount: Expr, bits: u16) -> Option<Expr> {
    let wide = bits.checked_mul(2)?.checked_add(1)?;
    let shifted = Expr::shl(
        extend_lane(value, wide, signed),
        Expr::zero_ext(amount, wide),
    );
    let (min, max) = element_bounds(signed, bits, wide)?;
    let below_max = Expr::Ite {
        cond: Box::new(Expr::sle(shifted.clone(), max.clone())),
        then_expr: Box::new(shifted),
        else_expr: Box::new(max),
    };
    let clamped = Expr::Ite {
        cond: Box::new(Expr::sle(min.clone(), below_max.clone())),
        then_expr: Box::new(below_max),
        else_expr: Box::new(min),
    };
    Some(Expr::extract(clamped, bits - 1, 0))
}

/// The exact sum or difference of two lanes, at the headroom width.
fn wide_combination(subtract: bool, signed: bool, a: Expr, b: Expr, wide: u16) -> Expr {
    let (x, y) = (extend_lane(a, wide, signed), extend_lane(b, wide, signed));
    if subtract {
        Expr::sub(x, y)
    } else {
        Expr::add(x, y)
    }
}

/// The range an element of `bits` can represent, as headroom-width
/// constants.
pub(super) fn element_bounds(signed: bool, bits: u16, wide: u16) -> Option<(Expr, Expr)> {
    let span = 1i128.checked_shl(u32::from(bits))?;
    let (min, max) = if signed {
        (-(span / 2), span / 2 - 1)
    } else {
        (0, span - 1)
    };
    Some((wide_const(min, wide)?, wide_const(max, wide)?))
}

/// Clamp a `wide`-bit **signed** value into the range of a `narrow`-bit
/// element and truncate it to that width.
///
/// The comparisons are signed whatever `signed` says, because that flag
/// describes the *destination* range and not how the incoming value
/// reads. x86's `packuswb` is exactly that combination — it interprets
/// its sources as signed words and saturates them into the unsigned byte
/// range, so a negative source has to land on `0` rather than on `0xff`.
pub(super) fn clamp_to_element(value: Expr, wide: u16, narrow: u16, signed: bool) -> Option<Expr> {
    let (min, max) = element_bounds(signed, narrow, wide)?;
    let below_max = Expr::Ite {
        cond: Box::new(Expr::sle(value.clone(), max.clone())),
        then_expr: Box::new(value),
        else_expr: Box::new(max),
    };
    let clamped = Expr::Ite {
        cond: Box::new(Expr::sle(min.clone(), below_max.clone())),
        then_expr: Box::new(below_max),
        else_expr: Box::new(min),
    };
    Some(Expr::extract(clamped, narrow.checked_sub(1)?, 0))
}

/// One destination lane of `vqadd` / `vqsub`.
///
/// The arithmetic happens where it cannot overflow and the clamp brings
/// it back; doing it at the element width and clamping afterwards would
/// be too late, since the overflow the instruction exists to detect
/// would already have wrapped.
fn saturating_lane(subtract: bool, signed: bool, a: Expr, b: Expr, bits: u16) -> Option<Expr> {
    let wide = bits.checked_add(WIDE_HEADROOM)?;
    let raw = wide_combination(subtract, signed, a, b, wide);
    let (min, max) = element_bounds(signed, bits, wide)?;
    let below_max = Expr::Ite {
        cond: Box::new(Expr::sle(raw.clone(), max.clone())),
        then_expr: Box::new(raw),
        else_expr: Box::new(max),
    };
    let clamped = Expr::Ite {
        cond: Box::new(Expr::sle(min.clone(), below_max.clone())),
        then_expr: Box::new(below_max),
        else_expr: Box::new(min),
    };
    Some(Expr::extract(clamped, bits - 1, 0))
}

/// One destination lane of `vhadd` / `vhsub` / `vrhadd`.
///
/// The shift is arithmetic at the headroom width because the value
/// there is the exact integer in two's complement, and the architecture
/// defines the result as its bit slice `[esize:1]` — which is the floor
/// of the halved value, not a truncation toward zero.
fn halving_lane(
    subtract: bool,
    signed: bool,
    rounding: bool,
    a: Expr,
    b: Expr,
    bits: u16,
) -> Option<Expr> {
    let wide = bits.checked_add(WIDE_HEADROOM)?;
    let raw = wide_combination(subtract, signed, a, b, wide);
    let exact = if rounding {
        Expr::add(raw, wide_const(1, wide)?)
    } else {
        raw
    };
    let halved = Expr::ashr(exact, wide_const(1, wide)?);
    Some(Expr::extract(halved, bits - 1, 0))
}

/// Width of the amount field a register-form vector shift reads.
const SHIFT_AMOUNT_BITS: u16 = 8;

/// A right shift of one lane, replicating the sign bit or not.
fn shift_right_lane(signed: bool, value: Expr, amount: Expr) -> Expr {
    if signed {
        Expr::ashr(value, amount)
    } else {
        Expr::lshr(value, amount)
    }
}

/// A rounding right shift of one lane — `vrshr` / `vrsra`.
///
/// ARM rounds on the unbounded integer, adding half an ulp of the shift
/// (`1 << (amount - 1)`) before discarding the low bits. The addition is
/// done one bit wider than the lane so the half can carry past the top
/// without overflowing — the same widening that keeps the `AArch64`
/// rounding shifts correct at their signed maximum.
fn rounding_shift_right_lane(signed: bool, value: Expr, amount: Expr, bits: u16) -> Option<Expr> {
    let wide = bits.checked_add(1)?;
    let value_wide = if signed {
        Expr::sign_ext(value, wide)
    } else {
        Expr::zero_ext(value, wide)
    };
    let amount_wide = Expr::zero_ext(amount, wide);
    let half = Expr::shl(
        wide_const(1, wide)?,
        Expr::sub(amount_wide.clone(), wide_const(1, wide)?),
    );
    let rounded = Expr::add(value_wide, half);
    let shifted = shift_right_lane(signed, rounded, amount_wide);
    Some(Expr::extract(shifted, bits - 1, 0))
}

/// One destination lane of a register-form vector shift.
///
/// Only the low byte of the amount element is read, as a signed value
/// whose sign chooses the direction — so both directions are built and
/// an `Ite` picks between them. An out-of-range amount needs no special
/// case: a bit-vector shift wider than the element already yields zero,
/// or all sign bits for the arithmetic right shift, which is what the
/// architecture specifies.
fn shift_register_lane(signed: bool, value: Expr, raw_amount: Expr, bits: u16) -> Expr {
    let amount = if bits > SHIFT_AMOUNT_BITS {
        Expr::sign_ext(Expr::extract(raw_amount, SHIFT_AMOUNT_BITS - 1, 0), bits)
    } else {
        raw_amount
    };
    let left = Expr::shl(value.clone(), amount.clone());
    let opposite = Expr::sub(Expr::konst(0, bits), amount.clone());
    let right = shift_right_lane(signed, value, opposite);
    Expr::Ite {
        cond: Box::new(Expr::slt(amount, Expr::konst(0, bits))),
        then_expr: Box::new(right),
        else_expr: Box::new(left),
    }
}

/// One destination lane of a multiply-accumulate.
fn accumulate_lane(
    float: bool,
    subtract: bool,
    acc: Expr,
    a: Expr,
    b: Expr,
    bits: u16,
) -> Option<Expr> {
    if !float {
        let product = Expr::mul(a, b);
        return Some(if subtract {
            Expr::sub(acc, product)
        } else {
            Expr::add(acc, product)
        });
    }
    let product = fp_lane_result(FpArithOp::Mul, a, b, bits)?;
    let combine = if subtract {
        FpArithOp::Sub
    } else {
        FpArithOp::Add
    };
    fp_lane_result(combine, acc, product, bits)
}

fn packed_int_lane(op: PackedIntOp, a: Expr, b: Option<Expr>, bits: u16) -> Option<Expr> {
    Some(match op {
        PackedIntOp::Bin(bin) => bin.apply(a, b?),
        // The IR has no bitwise NOT, so `~x` is `x XOR all-ones`.
        PackedIntOp::BitClear => Expr::bv_and(a, Expr::bv_xor(b?, all_ones(bits))),
        PackedIntOp::Not => Expr::bv_xor(a, all_ones(bits)),
        PackedIntOp::Copy => a,
        PackedIntOp::MinMax { max, signed } => {
            let other = b?;
            // `max` keeps the first operand when the second is smaller;
            // `min` keeps it when the second is larger. Written as one
            // comparison with the operands swapped rather than four
            // predicates, so the two directions cannot drift.
            let (lo, hi) = if max {
                (other.clone(), a.clone())
            } else {
                (a.clone(), other.clone())
            };
            let cond = if signed {
                Expr::slt(lo, hi)
            } else {
                Expr::ult(lo, hi)
            };
            Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(a),
                else_expr: Box::new(other),
            }
        }
        PackedIntOp::Abs => Expr::Ite {
            cond: Box::new(Expr::slt(a.clone(), Expr::konst(0, bits))),
            then_expr: Box::new(Expr::sub(Expr::konst(0, bits), a.clone())),
            else_expr: Box::new(a),
        },
        PackedIntOp::Neg => Expr::sub(Expr::konst(0, bits), a),
        PackedIntOp::SignBit { negate: true } => Expr::bv_xor(a, sign_bit(bits)?),
        PackedIntOp::SignBit { negate: false } => Expr::bv_and(a, magnitude_mask(bits)?),
        PackedIntOp::Saturating { subtract, signed } => {
            saturating_lane(subtract, signed, a, b?, bits)?
        }
        PackedIntOp::Halving {
            subtract,
            signed,
            rounding,
        } => halving_lane(subtract, signed, rounding, a, b?, bits)?,
    })
}

/// `Extract(Extract(x, inner_hi, 0), hi, lo)` denotes the same bits as
/// `Extract(x, hi, lo)` whenever the outer slice fits inside the inner
/// one, which it always does when the inner slice is an operand's
/// vector view and the outer one is a lane of that view.
///
/// Collapsing matters beyond tidiness: a lane read used to go straight
/// to the vector parent, and materialising the operand's view first
/// would otherwise nest one extract inside every lane of every packed
/// operation, growing the formula the solver sees for no gain.
fn extract_collapsing(value: Expr, hi: u16, lo: u16) -> Expr {
    if let Expr::Extract {
        src,
        hi: inner_hi,
        lo: 0,
    } = &value
        && hi <= *inner_hi
    {
        return Expr::extract((**src).clone(), hi, lo);
    }
    Expr::extract(value, hi, lo)
}

impl LiftCtx {
    /// The vector-register layout `op` names, if it names one.
    pub(super) fn simd_layout(&self, op: &Operand) -> Option<RegisterLayout> {
        if op.kind != OperandKind::Register {
            return None;
        }
        let layout = register_layout(&op.raw, self.arch)?;
        is_simd_parent(layout.parent, self.arch).then_some(layout)
    }

    /// Whether `op` names a view of a SIMD vector register.
    pub(super) fn is_simd_register(&self, op: &Operand) -> bool {
        self.simd_layout(op).is_some()
    }

    /// Architectural width of this ISA's vector parent.
    fn simd_parent_width(&self) -> Option<u16> {
        simd_parent_bits(self.arch)
    }

    /// The full value of a SIMD operand at `view_bits`, as a single
    /// expression.
    ///
    /// A register view is the slice `[layout.hi:layout.lo]` of its
    /// canonical vector parent. The offset is load-bearing on `AArch32`,
    /// where `d1` sits at bits `[127:64]` of `v0` and `s3` at `[127:96]`
    /// — indexing every view from bit 0, as x86 and `AArch64` allow, would
    /// silently read the wrong half of the register there.
    ///
    /// A memory operand becomes **one** `LoadMem` of `view_bits` through
    /// [`LiftCtx::read_operand_lowered`], which also supplies the
    /// `stk_<base>_<off>` naming so an analyst alias still resolves.
    /// Reading the whole view once — rather than once per lane — is what
    /// keeps a packed operand to a single load.
    ///
    /// `None` for anything else. The caller marks the instruction
    /// unsupported rather than fabricating a value.
    pub(super) fn simd_operand_value(&mut self, op: &Operand, view_bits: u16) -> Option<Expr> {
        if self.is_modellable_simd_memory(op) {
            return Some(self.read_operand_lowered(op, view_bits));
        }
        let layout = self.simd_layout(op)?;
        let parent_bits = self.simd_parent_width()?;
        let parent = Expr::var(layout.parent, parent_bits);
        Some(if layout.width() == parent_bits {
            parent
        } else {
            Expr::extract(parent, layout.hi, layout.lo)
        })
    }

    /// The value of a SIMD operand at its own view width.
    pub(super) fn read_simd_operand(&mut self, op: &Operand) -> Option<Expr> {
        let view = self.simd_view_bits(op)?;
        self.simd_operand_value(op, view)
    }

    /// Read lane `index` of a SIMD operand and reinterpret it as an IEEE
    /// float of the matching sort (32→single, 64→double). Used by the
    /// scalar SSE FP handlers, which always read lane 0.
    pub(super) fn read_simd_lane_fp(
        &mut self,
        op: &Operand,
        lane_bits: u16,
        index: u16,
    ) -> Option<Expr> {
        let raw = self.read_simd_lane_bits(op, lane_bits, index)?;
        let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
        Some(Expr::bv_to_fp(raw, ebits, sbits))
    }

    /// The raw bit-vector of lane `index`, before any float
    /// reinterpretation. Kept separate so the compare handlers can build
    /// a mask without a pointless round trip through the float sort.
    ///
    /// A scalar memory operand is loaded at the *lane* width, not the
    /// vector view: `movss xmm0, dword [rbp - 8]` reads four bytes. Only
    /// lane 0 can be read that way — a lane index addresses a register's
    /// element, and no ISA spells an indexed element of a memory operand.
    pub(super) fn read_simd_lane_bits(
        &mut self,
        op: &Operand,
        lane_bits: u16,
        index: u16,
    ) -> Option<Expr> {
        if self.is_modellable_simd_memory(op) {
            if index != 0 {
                return None;
            }
            return Some(self.read_operand_lowered(op, lane_bits));
        }
        let view = self.simd_view_bits(op)?;
        let value = self.simd_operand_value(op, view)?;
        Self::extract_lane(value, lane_bits, index)
    }

    /// Extract lane `index` (counting from the least-significant) out of
    /// an already-materialised operand value.
    pub(super) fn extract_lane(value: Expr, lane_bits: u16, index: u16) -> Option<Expr> {
        let lo = lane_bits.checked_mul(index)?;
        let hi = lo.checked_add(lane_bits)?.checked_sub(1)?;
        Some(extract_collapsing(value, hi, lo))
    }

    /// Write `lane_value` (a bit-vector of `lane_bits`) to lane `index`
    /// of a SIMD destination, preserving every other bit of the parent.
    ///
    /// A register destination keeps the parent bits around the lane
    /// (legacy SSE scalar semantics, and every ARM element insert). A
    /// memory destination stores exactly `lane_bits` — there is nothing
    /// around the lane to preserve, which is why `movss [rbp - 8], xmm0`
    /// writes four bytes and leaves the rest of the slot alone; only lane
    /// 0 is addressable that way.
    ///
    /// The lane sits at `index` elements above the view's own offset in
    /// the parent, so on `AArch32` a write to `s3` preserves the bits
    /// *below* it as well as above.
    pub(super) fn write_simd_lane(
        &mut self,
        op: &Operand,
        lane_value: Expr,
        lane_bits: u16,
        index: u16,
    ) -> bool {
        if self.is_modellable_simd_memory(op) {
            return index == 0 && self.write_dst(op, lane_value, lane_bits);
        }
        let (Some(layout), Some(parent_bits)) = (self.simd_layout(op), self.simd_parent_width())
        else {
            return false;
        };
        let Some(lane_lo) = lane_bits
            .checked_mul(index)
            .and_then(|offset| layout.lo.checked_add(offset))
        else {
            return false;
        };
        let Some(lane_top) = lane_lo.checked_add(lane_bits) else {
            return false;
        };
        if lane_top > parent_bits {
            return false;
        }
        let parent = Expr::var(layout.parent, parent_bits);
        let value = Self::splice_into_parent(&parent, lane_value, lane_lo, lane_top, parent_bits);
        self.assign(Var::new(layout.parent, parent_bits), value);
        true
    }

    /// Rebuild a parent register with `value` occupying
    /// `[top-1 : lo]` and every other bit taken from its prior contents.
    fn splice_into_parent(parent: &Expr, value: Expr, lo: u16, top: u16, parent_bits: u16) -> Expr {
        let with_high = if top < parent_bits {
            Expr::concat(Expr::extract(parent.clone(), parent_bits - 1, top), value)
        } else {
            value
        };
        if lo > 0 {
            Expr::concat(with_high, Expr::extract(parent.clone(), lo - 1, 0))
        } else {
            with_high
        }
    }

    /// Number of `lane_bits`-wide lanes in a `view_bits`-wide vector
    /// view, or `None` when the view is not a whole multiple of the lane.
    pub(super) fn packed_lane_count(view_bits: u16, lane_bits: u16) -> Option<u16> {
        if lane_bits == 0 || view_bits % lane_bits != 0 {
            return None;
        }
        Some(view_bits / lane_bits)
    }

    /// Assemble per-lane bit-vectors, least-significant lane first, into
    /// one value of the full view width.
    pub(super) fn concat_lanes(lanes: Vec<Expr>) -> Option<Expr> {
        let mut iter = lanes.into_iter();
        let mut acc = iter.next()?;
        for lane in iter {
            acc = Expr::concat(lane, acc);
        }
        Some(acc)
    }

    /// Build the full-view result of a packed floating-point lane
    /// operation, or `None` when any operand is unmodellable.
    ///
    /// Each operand is materialised **once** and the lanes are extracted
    /// from that value, so a memory operand costs one load rather than
    /// one per lane.
    ///
    /// Arch-neutral in shape: every step below is expressed in views,
    /// lanes and operand values. `addps xmm0, xmm1` and `fadd v0.4s,
    /// v1.4s, v2.4s` are the same computation over the same model,
    /// differing only in how the caller derived the lane width — from
    /// the mnemonic on x86, from the arrangement on ARM.
    ///
    /// **Max and min are the exception, and it is not cosmetic.**
    /// [`fp_lane_result`] implements Intel's `MAXPS`, where the second
    /// operand wins on NaN and on a signed-zero tie; ARM's `FPMax`
    /// propagates NaN and combines the two signs. Both are right for
    /// their own architecture and each is a wrong *value* on the other,
    /// so the lane helper is chosen by [`LiftCtx::fp_max_min_propagates`]
    /// rather than shared.
    pub(super) fn packed_fp_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: &Operand,
        op: FpArithOp,
        lane_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[dst, a_op, b_op])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = self.simd_operand_value(b_op, view)?;
        let propagating =
            self.fp_max_min_propagates() && matches!(op, FpArithOp::Max | FpArithOp::Min);
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let b = Self::extract_lane(b_val.clone(), lane_bits, index)?;
            let lane = if propagating {
                fp_propagating_max_min(a, b, lane_bits, matches!(op, FpArithOp::Max), false)?
            } else {
                fp_lane_result(op, a, b, lane_bits)?
            };
            lanes.push(lane);
        }
        Self::concat_lanes(lanes)
    }

    /// The integer twin of [`Self::packed_fp_result`]. `b_op` is absent
    /// for the one-source forms (`mvn`, `mov`).
    ///
    /// A lane-independent operation is emitted once over the whole view
    /// instead of once per lane — see [`PackedOp::is_lane_independent`].
    pub(super) fn packed_int_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: Option<&Operand>,
        op: PackedIntOp,
        lane_bits: u16,
    ) -> Option<Expr> {
        let mut refs = vec![dst, a_op];
        refs.extend(b_op);
        let view = self.simd_instruction_view_bits(&refs)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = match b_op {
            Some(b) => Some(self.simd_operand_value(b, view)?),
            None => None,
        };
        if PackedOp::Int(op).is_lane_independent() {
            return packed_int_lane(op, a_val, b_val, view);
        }
        let count = Self::packed_lane_count(view, lane_bits)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let b = match b_val.as_ref() {
                Some(value) => Some(Self::extract_lane(value.clone(), lane_bits, index)?),
                None => None,
            };
            lanes.push(packed_int_lane(op, a, b, lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `dst := dst ± a * b`, lane-wise.
    ///
    /// The destination is materialised as a source like any other, so
    /// the SSA pass turns the read into the previous version and the
    /// write into a new one — nothing here has to know that the two
    /// name the same register.
    ///
    /// The float form rounds **twice**: once at the product and once at
    /// the sum. That is the architectural behaviour — `VMLA` is not a
    /// fused multiply-add, which is `VFMA` and a different mnemonic —
    /// and collapsing the two roundings into one would be a definite
    /// wrong value at every operand where they differ.
    fn packed_accumulate_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: &Operand,
        float: bool,
        subtract: bool,
        lane_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[dst, a_op, b_op])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let acc_val = self.simd_operand_value(dst, view)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = self.simd_operand_value(b_op, view)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let acc = Self::extract_lane(acc_val.clone(), lane_bits, index)?;
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let b = Self::extract_lane(b_val.clone(), lane_bits, index)?;
            lanes.push(accumulate_lane(float, subtract, acc, a, b, lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// A whole-vector shift by an amount every lane shares.
    ///
    /// The amount operand is deliberately kept out of the view
    /// resolution: it is an immediate, not a vector, and reading it at
    /// the *lane* width is what lets it flow straight into the shift
    /// node the solver then constant-folds.
    fn packed_shift_immediate_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        amount_op: &Operand,
        shape: PackedOp,
        lane_bits: u16,
    ) -> Option<Expr> {
        let PackedOp::ShiftImmediate {
            left,
            signed,
            accumulate,
            rounding,
        } = shape
        else {
            return None;
        };
        let view = self.simd_instruction_view_bits(&[dst, a_op])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let acc_val = if accumulate {
            Some(self.simd_operand_value(dst, view)?)
        } else {
            None
        };
        let amount = self.read_operand_at(amount_op, lane_bits);
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let shifted = if left {
                Expr::shl(a, amount.clone())
            } else if rounding {
                rounding_shift_right_lane(signed, a, amount.clone(), lane_bits)?
            } else {
                shift_right_lane(signed, a, amount.clone())
            };
            lanes.push(match acc_val.as_ref() {
                Some(value) => Expr::add(
                    Self::extract_lane(value.clone(), lane_bits, index)?,
                    shifted,
                ),
                None => shifted,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// `vqshl` with an immediate amount — each lane shifted left and
    /// clamped to the element's range.
    fn packed_saturating_shift_left_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        amount_op: &Operand,
        signed: bool,
        lane_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[dst, a_op])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let amount = self.read_operand_at(amount_op, lane_bits);
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            lanes.push(saturating_shift_left_lane(
                signed,
                a,
                amount.clone(),
                lane_bits,
            )?);
        }
        Self::concat_lanes(lanes)
    }

    /// A whole-vector shift whose amount is itself a vector, one
    /// element per lane.
    fn packed_shift_register_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        amount_op: &Operand,
        signed: bool,
        lane_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[dst, a_op, amount_op])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let amount_val = self.simd_operand_value(amount_op, view)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let amount = Self::extract_lane(amount_val.clone(), lane_bits, index)?;
            lanes.push(shift_register_lane(signed, a, amount, lane_bits));
        }
        Self::concat_lanes(lanes)
    }

    /// Lower one packed ARM vector data-processing instruction: the same
    /// lane operation applied independently to every lane of the
    /// destination's view.
    ///
    /// `zero_upper` is the ISA's rule for the vector-register bits above
    /// the view. `AArch64` zeroes them on every SIMD write; `AArch32`
    /// preserves them, because a `Dd` operand is one half of a `Qd` and
    /// the other half survives the instruction.
    ///
    /// A decline here is sound by the free-input boundary: the slicer
    /// consumed the destination as a definition and therefore stopped
    /// tracking its upstream definitions, so emitting no assignment
    /// leaves the register a free SSA input rather than bound to a stale
    /// value.
    pub(super) fn lift_packed_vector(
        &mut self,
        insn: &Instruction,
        op: PackedOp,
        lane_bits: u16,
        zero_upper: bool,
    ) {
        let ops = &insn.operands;
        let (Some(dst), Some(a_op)) = (ops.first(), ops.get(1)) else {
            self.push_packed_unsupported(insn);
            return;
        };
        let b_op = ops.get(2);
        let result = match op {
            PackedOp::Fp(fp) => match b_op {
                Some(b) => self.packed_fp_result(dst, a_op, b, fp, lane_bits),
                None => None,
            },
            PackedOp::Int(int) => self.packed_int_result(dst, a_op, b_op, int, lane_bits),
            PackedOp::Accumulate { float, subtract } => match b_op {
                Some(b) => self.packed_accumulate_result(dst, a_op, b, float, subtract, lane_bits),
                None => None,
            },
            PackedOp::ShiftImmediate { .. } => match b_op {
                Some(amount) => {
                    self.packed_shift_immediate_result(dst, a_op, amount, op, lane_bits)
                }
                None => None,
            },
            PackedOp::ShiftRegister { signed } => match b_op {
                Some(amount) => {
                    self.packed_shift_register_result(dst, a_op, amount, signed, lane_bits)
                }
                None => None,
            },
            PackedOp::SaturatingShiftLeftImmediate { signed } => match b_op {
                Some(amount) => {
                    self.packed_saturating_shift_left_result(dst, a_op, amount, signed, lane_bits)
                }
                None => None,
            },
        };
        let Some(value) = result else {
            self.push_packed_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, value, zero_upper) {
            self.push_packed_unsupported(insn);
        }
    }

    fn push_packed_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable packed operand at {addr}", addr = insn.address),
        });
    }

    /// Width of a SIMD operand's view: 128 / 256 / 512 for a register,
    /// and for memory whatever size prefix radare2 attached
    /// (`xmmword [rsi]` → 128). `None` if `op` is neither, or is a
    /// memory operand with no size prefix — guessing a width there would
    /// silently load the wrong number of bytes.
    pub(super) fn simd_view_bits(&self, op: &Operand) -> Option<u16> {
        if self.is_modellable_simd_memory(op) {
            return memory_operand_width(&op.raw);
        }
        Some(self.simd_layout(op)?.width())
    }

    /// The view width an instruction operates at, given its operands.
    ///
    /// A memory operand carries its own size prefix, but a register one
    /// is the more reliable source (r2 always spells the register), so
    /// the first operand that resolves wins.
    pub(super) fn simd_instruction_view_bits(&self, ops: &[&Operand]) -> Option<u16> {
        ops.iter()
            .find(|op| self.is_simd_register(op))
            .or_else(|| ops.first())
            .and_then(|op| self.simd_view_bits(op))
    }

    /// Write a `value` (already sized to the operand's view width) to a
    /// SIMD **register** destination, reconstructing the full-width
    /// parent per the write's upper-bits rule: `zero_upper` (VEX-encoded
    /// `v*` forms, and every `AArch64` SIMD write) zeroes bits above the
    /// view; otherwise (legacy SSE, `AArch32` VFP) the bits above the
    /// view are preserved from the parent's prior value. A memory
    /// destination stores the view width verbatim — there are no parent
    /// bits to reconstruct.
    pub(super) fn write_simd_dst(&mut self, op: &Operand, value: Expr, zero_upper: bool) -> bool {
        if self.is_modellable_simd_memory(op) {
            let Some(width) = self.simd_view_bits(op) else {
                return false;
            };
            return self.write_dst(op, value, width);
        }
        let (Some(layout), Some(parent_bits)) = (self.simd_layout(op), self.simd_parent_width())
        else {
            return false;
        };
        let full = if layout.width() == parent_bits {
            value
        } else if zero_upper {
            Expr::zero_ext(value, parent_bits)
        } else {
            let parent = Expr::var(layout.parent, parent_bits);
            Self::splice_into_parent(&parent, value, layout.lo, layout.hi + 1, parent_bits)
        };
        self.assign(Var::new(layout.parent, parent_bits), full);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FP_DOUBLE_BITS, FP_HALF_BITS, FP_SINGLE_BITS, FusedStep, fused_formats, fused_multiply_lane,
    };
    use r2smt_ir::expr::Expr;

    fn lane(bits: u16) -> Expr {
        Expr::konst(0, bits)
    }

    #[test]
    fn fused_multiply_lane_accepts_a_binary32_lane_over_a_binary64_intermediate() {
        assert!(
            fused_multiply_lane(
                FusedStep::MulAdd,
                &lane(FP_SINGLE_BITS),
                &lane(FP_SINGLE_BITS),
                Some(lane(FP_SINGLE_BITS)),
                FP_SINGLE_BITS,
            )
            .is_some()
        );
    }

    #[test]
    fn fused_multiply_lane_accepts_a_binary64_lane_over_a_binary128_intermediate() {
        assert!(
            fused_multiply_lane(
                FusedStep::MulAdd,
                &lane(FP_DOUBLE_BITS),
                &lane(FP_DOUBLE_BITS),
                Some(lane(FP_DOUBLE_BITS)),
                FP_DOUBLE_BITS,
            )
            .is_some()
        );
    }

    #[test]
    fn a_binary64_lane_widens_to_binary128_and_not_to_something_narrower() {
        // Accepting the lane is not the property that matters — *which*
        // intermediate it widens to is. A binary64 intermediate would
        // still lift, still be accepted by the test above, and round
        // twice: a wrong value, silently. The exactness argument needs
        // `q >= 2p + 2`, and only binary128 gives it for `p = 53`.
        assert_eq!(
            fused_formats(FP_DOUBLE_BITS).map(|formats| formats.wide),
            Some((15, 113)),
            "a binary64 fused step must widen to binary128"
        );
        assert_eq!(
            fused_formats(FP_DOUBLE_BITS)
                .map(|formats| u32::from(formats.wide.1) >= 2 * u32::from(formats.lane.1) + 2),
            Some(true),
            "the intermediate must be wide enough for a single rounding"
        );
    }

    #[test]
    fn fused_multiply_lane_declines_a_lane_with_no_wide_enough_intermediate() {
        // A binary16 lane has an IEEE sort but no entry in the fused
        // table, so it declines rather than rounding twice.
        assert!(
            fused_multiply_lane(
                FusedStep::MulAdd,
                &lane(FP_HALF_BITS),
                &lane(FP_HALF_BITS),
                Some(lane(FP_HALF_BITS)),
                FP_HALF_BITS,
            )
            .is_none()
        );
    }
}
