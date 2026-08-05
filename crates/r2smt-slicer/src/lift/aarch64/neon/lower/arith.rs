//! Lowering for the `AArch64` NEON families whose result is a function
//! of whole source lanes.
//!
//! The same taxonomy the resolvers use: the lane compares, the same-width
//! shifts, the saturating / halving / rounding forms, the lane-wise and
//! pairwise folds, the absolute differences, the per-bit unary counts,
//! and the reciprocal estimates.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use super::{extend, integer_min_max, unsigned_max};
use crate::lift::aarch64::neon::NeonShape;
use crate::lift::aarch64::neon::arith::{
    AbsDiffKind, BitwiseUnary, PairOp, SaturateTo, SaturatingKind, ShiftKind,
};
use crate::lift::aarch64::neon::geometry::{BITS_PER_BYTE, operand_arrangement};
use crate::lift::simd::{CompareKind, compare_lane};
use crate::lift::{FpArithOp, LiftCtx, fp_lane_result, fp_propagating_max_min};

impl LiftCtx {
    /// `frecpe` / `frsqrte` — a fresh value that is never assigned.
    ///
    /// The architecture guarantees only a relative error bound for
    /// these and leaves the value itself implementation-defined, so
    /// `FDiv(1.0, x)` would not be an approximation of the result: it
    /// would be a *definite* number the machine is not required to
    /// produce, which is the fabrication this pipeline exists to avoid.
    ///
    /// Emitting nothing is not the alternative either. The slicer has
    /// already consumed the destination as a definition, so a decline
    /// truncates the slice and every downstream verdict becomes
    /// `Unsound`. Assigning a temp that is never defined keeps the slice
    /// `Complete` instead: `ssa_convert` surfaces a variable read before
    /// it is written as a free input, so the solver considers every
    /// value the estimate could take and an `AlwaysTrue` verdict is
    /// still one that holds for all of them.
    ///
    /// The temp is fresh per instruction, which loses one true fact —
    /// two `frecpe`s over the same input agree, since the estimate is a
    /// function. That is a widening, not an unsoundness, and recovering
    /// it would mean naming the value by its operand rather than by its
    /// address.
    pub(super) fn estimate_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        Some(Expr::Var(self.new_temp(insn.address, view)))
    }

    /// The lane-wise compares, each writing an all-ones or all-zeros
    /// mask per lane.
    pub(super) fn compare_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: CompareKind,
        zero: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = if zero {
            None
        } else {
            Some(self.widen_source(insn, 2)?)
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = match second.as_ref() {
                Some(value) => LiftCtx::extract_lane(value.clone(), shape.lane_bits, index)?,
                None => Expr::konst(0, shape.lane_bits),
            };
            lanes.push(compare_lane(kind, a, b, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The same-width shift family.
    pub(super) fn shift_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ShiftKind,
        signed: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let amounts = match kind {
            ShiftKind::Register { .. } => Some(self.widen_source(insn, 2)?),
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let value = LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?;
            let amount = match amounts.as_ref() {
                Some(vector) => Some(LiftCtx::extract_lane(
                    vector.clone(),
                    shape.lane_bits,
                    index,
                )?),
                None => None,
            };
            lanes.push(shift_lane(kind, signed, value, amount, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The saturating, halving and rounding-narrow family.
    ///
    /// Every member computes its lane at a width where the result cannot
    /// overflow, then brings it back down — by clamping, by halving, or
    /// by truncating. Doing the arithmetic at the destination's width
    /// and clamping afterwards would be too late: the overflow the
    /// instruction exists to detect would already have wrapped.
    pub(super) fn saturating_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: SaturatingKind,
        signed_sources: bool,
        upper: bool,
    ) -> Option<Expr> {
        let narrowing = matches!(
            kind,
            SaturatingKind::Narrow { .. } | SaturatingKind::ShiftNarrow { .. }
        );
        let source_bits = if narrowing {
            shape.lane_bits.checked_mul(2)?
        } else {
            shape.lane_bits
        };
        let first = self.widen_source(insn, 1)?;
        let second = match insn.operands.get(2) {
            Some(operand) if operand_arrangement(operand).is_some() => {
                Some(self.widen_source(insn, 2)?)
            }
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), source_bits, index)?;
            let b = match second.as_ref() {
                Some(value) => Some(LiftCtx::extract_lane(value.clone(), source_bits, index)?),
                None => None,
            };
            lanes.push(saturating_lane(
                kind,
                a,
                b,
                shape.lane_bits,
                signed_sources,
            )?);
        }
        let narrowed = Self::concat_lanes(lanes)?;
        if !upper {
            return Some(narrowed);
        }
        // A `2` form writes the destination's upper half and preserves
        // the lower one.
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            narrowed,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// `cnt` / `clz` / `cls` / `rbit` — one lane at a time, since every
    /// member is a function of the lane's own bits alone.
    pub(super) fn bitwise_unary_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: BitwiseUnary,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let element = LiftCtx::extract_lane(source.clone(), shape.lane_bits, index)?;
            lanes.push(bitwise_unary_lane(kind, &element, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The lane-wise selects: each destination lane folds the two source
    /// lanes at its own index.
    pub(super) fn lane_combine_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        op: PairOp,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = LiftCtx::extract_lane(second.clone(), shape.lane_bits, index)?;
            lanes.push(pair_lane(op, a, b, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The pairwise family: the destination's low half folds the first
    /// source's neighbouring lanes and its high half the second's.
    ///
    /// That split is the whole shape of the family — reading both
    /// sources at the destination lane's own index, as the lane-wise
    /// fold next door does, would pair lanes the instruction never
    /// brings together.
    pub(super) fn pairwise_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        op: PairOp,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let half = shape.lanes / 2;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let (source, pair) = if index < half {
                (&first, index)
            } else {
                (&second, index.checked_sub(half)?)
            };
            let low = pair.checked_mul(2)?;
            let a = LiftCtx::extract_lane(source.clone(), shape.lane_bits, low)?;
            let b = LiftCtx::extract_lane(source.clone(), shape.lane_bits, low.checked_add(1)?)?;
            lanes.push(pair_lane(op, a, b, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `sabd` / `uabd` / `saba` / `uaba` / `fabd`.
    pub(super) fn absolute_difference_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: AbsDiffKind,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let accumulator = match kind {
            AbsDiffKind::Integer {
                accumulate: true, ..
            } => Some(self.destination_value(insn, shape)?),
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = LiftCtx::extract_lane(second.clone(), shape.lane_bits, index)?;
            let difference = absolute_difference_lane(kind, a, b, shape.lane_bits)?;
            lanes.push(match accumulator.as_ref() {
                Some(previous) => Expr::add(
                    LiftCtx::extract_lane(previous.clone(), shape.lane_bits, index)?,
                    difference,
                ),
                None => difference,
            });
        }
        Self::concat_lanes(lanes)
    }
}

/// One destination lane of a two-lane fold, shared by the lane-wise
/// selects and the pairwise family.
///
/// The float selects go through [`fp_propagating_max_min`] and never
/// through [`fp_lane_result`]'s `Max` / `Min`: the latter is Intel's
/// `MAXPS`, which returns its second operand on unordered where ARM
/// propagates the NaN, and takes the second operand on a signed-zero tie
/// where ARM combines the two signs. Both are wrong *values*.
fn pair_lane(op: PairOp, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    match op {
        PairOp::Add => Some(Expr::add(a, b)),
        PairOp::FloatAdd => fp_lane_result(FpArithOp::Add, a, b, lane_bits),
        PairOp::MinMax { signed, max } => Some(integer_min_max(a, b, signed, max)),
        PairOp::FloatMinMax { max, number_wins } => {
            fp_propagating_max_min(a, b, lane_bits, max, number_wins)
        }
    }
}

/// One destination lane of an absolute difference.
///
/// The integer forms subtract in whichever direction the comparison
/// chooses, at the element's own width and with no widening. That is
/// exact rather than lucky: the magnitude of the difference of two
/// `n`-bit values always fits `n` unsigned bits, and the wrapping
/// subtraction of the smaller from the larger *is* that magnitude —
/// `sabd` of `-128` and `127` gives `0xff`, which is the 255 ARM
/// defines and not an overflow.
///
/// `fabd` is `FPAbs(FPSub(a, b))`, and clearing the sign bit of the
/// difference is exact at every value, NaNs and infinities included.
fn absolute_difference_lane(kind: AbsDiffKind, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    match kind {
        AbsDiffKind::Integer { signed, .. } => {
            let cond = if signed {
                Expr::slt(a.clone(), b.clone())
            } else {
                Expr::ult(a.clone(), b.clone())
            };
            Some(Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(Expr::sub(b.clone(), a.clone())),
                else_expr: Box::new(Expr::sub(a, b)),
            })
        }
        AbsDiffKind::Float => {
            let difference = fp_lane_result(FpArithOp::Sub, a, b, lane_bits)?;
            Some(Expr::bv_and(difference, magnitude_mask(lane_bits)?))
        }
    }
}

/// A `bits`-wide mask with every bit but the sign one set.
fn magnitude_mask(bits: u16) -> Option<Expr> {
    Some(Expr::konst(unsigned_max(bits.checked_sub(1)?)?, bits))
}

/// One destination lane of `cnt` / `clz` / `cls` / `rbit`.
///
/// None of the four has an `Expr` node, so each is built from
/// single-bit slices: a running sum for the population count, an `Ite`
/// ladder for the leading-zero counts, and a reversed concatenation for
/// the bit reversal.
fn bitwise_unary_lane(kind: BitwiseUnary, value: &Expr, lane_bits: u16) -> Option<Expr> {
    match kind {
        BitwiseUnary::PopulationCount => {
            let mut count = Expr::konst(0, lane_bits);
            for bit in 0..lane_bits {
                count = Expr::add(
                    count,
                    Expr::zero_ext(Expr::extract(value.clone(), bit, bit), lane_bits),
                );
            }
            Some(count)
        }
        BitwiseUnary::LeadingZeros => Some(leading_zeros(value, lane_bits)),
        BitwiseUnary::LeadingSignBits => {
            // The bits that repeat their neighbour are the zeros of the
            // lane exclusive-ORed with itself shifted one place, so the
            // leading sign bits are the leading zeros of that fold. It
            // is one bit narrower, which is also why `cls` can never
            // reach the element width: the sign bit itself is not
            // counted.
            let width = lane_bits.checked_sub(1)?;
            let folded = Expr::bv_xor(
                Expr::extract(value.clone(), lane_bits - 1, 1),
                Expr::extract(value.clone(), width - 1, 0),
            );
            Some(Expr::zero_ext(leading_zeros(&folded, width), lane_bits))
        }
        BitwiseUnary::ReverseBits => {
            let mut bits = Vec::with_capacity(usize::from(lane_bits));
            // `concat_lanes` puts the first element at the low end, so
            // pushing from the top down is the reversal.
            for bit in (0..lane_bits).rev() {
                bits.push(Expr::extract(value.clone(), bit, bit));
            }
            LiftCtx::concat_lanes(bits)
        }
    }
}

/// The number of leading zero bits of a `bits`-wide value, as a
/// `bits`-wide result.
///
/// The ladder is built from the least significant bit upwards, so the
/// layer testing the *most* significant one ends up outermost and its
/// answer wins — which is what makes the count that of the highest set
/// bit rather than of the lowest.
fn leading_zeros(value: &Expr, bits: u16) -> Expr {
    let mut count = Expr::konst(u128::from(bits), bits);
    for bit in 0..bits {
        count = Expr::Ite {
            cond: Box::new(Expr::eq(
                Expr::extract(value.clone(), bit, bit),
                Expr::konst(1, 1),
            )),
            then_expr: Box::new(Expr::konst(u128::from(bits - 1 - bit), bits)),
            else_expr: Box::new(count),
        };
    }
    count
}

/// Clamp `value`, computed at `wide` bits, into the `narrow`-bit range
/// `to` names, returning a `narrow`-bit result.
///
/// The comparison's signedness follows the *value*, not the target: an
/// unsigned sum is compared unsigned, while `uqsub`'s difference can be
/// negative and so is compared signed even though it clamps into the
/// unsigned range.
fn clamp(value: Expr, wide: u16, narrow: u16, to: SaturateTo) -> Option<Expr> {
    let konst = |v: u128| Expr::konst(v, wide);
    let clamped = match to {
        SaturateTo::Signed => {
            let magnitude = 1u128.checked_shl(u32::from(narrow.checked_sub(1)?))?;
            let high = konst(magnitude - 1);
            // The lower bound is `-2^(narrow-1)` in `wide`-bit two's
            // complement.
            let low = konst(unsigned_max(wide)?.checked_add(1)?.checked_sub(magnitude)?);
            let above = Expr::Ite {
                cond: Box::new(Expr::slt(high.clone(), value.clone())),
                then_expr: Box::new(high),
                else_expr: Box::new(value.clone()),
            };
            Expr::Ite {
                cond: Box::new(Expr::slt(value, low.clone())),
                then_expr: Box::new(low),
                else_expr: Box::new(above),
            }
        }
        SaturateTo::Unsigned => {
            let high = konst(unsigned_max(narrow)?);
            Expr::Ite {
                cond: Box::new(Expr::ult(high.clone(), value.clone())),
                then_expr: Box::new(high),
                else_expr: Box::new(value),
            }
        }
        SaturateTo::SignedToUnsigned => {
            let high = konst(unsigned_max(narrow)?);
            let above = Expr::Ite {
                cond: Box::new(Expr::slt(high.clone(), value.clone())),
                then_expr: Box::new(high),
                else_expr: Box::new(value.clone()),
            };
            Expr::Ite {
                cond: Box::new(Expr::slt(value, konst(0))),
                then_expr: Box::new(konst(0)),
                else_expr: Box::new(above),
            }
        }
    };
    Some(Expr::extract(clamped, narrow - 1, 0))
}

/// One destination lane of a saturating, halving or rounding-narrow
/// operation.
///
/// `a` and `b` arrive at the *source* width, which for the narrowing
/// members is twice `lane_bits`.
fn saturating_lane(
    kind: SaturatingKind,
    a: Expr,
    b: Option<Expr>,
    lane_bits: u16,
    signed_sources: bool,
) -> Option<Expr> {
    match kind {
        SaturatingKind::AddSub { op, to } => {
            // One extra bit makes the sum or difference exact, so the
            // clamp sees the true value rather than a wrapped one.
            let wide = lane_bits.checked_add(1)?;
            let value = op.apply(
                extend(a, wide, signed_sources),
                extend(b?, wide, signed_sources),
            );
            clamp(value, wide, lane_bits, to)
        }
        SaturatingKind::Halving { rounding } => {
            let wide = lane_bits.checked_add(1)?;
            let mut sum = Expr::add(
                extend(a, wide, signed_sources),
                extend(b?, wide, signed_sources),
            );
            if rounding {
                sum = Expr::add(sum, Expr::konst(1, wide));
            }
            // The exact sum needs no clamp — halving it always fits.
            let halved = if signed_sources {
                Expr::ashr(sum, Expr::konst(1, wide))
            } else {
                Expr::lshr(sum, Expr::konst(1, wide))
            };
            Some(Expr::extract(halved, lane_bits - 1, 0))
        }
        SaturatingKind::Narrow { to } => clamp(a, lane_bits.checked_mul(2)?, lane_bits, to),
        SaturatingKind::ShiftNarrow {
            shift,
            rounding,
            to,
        } => {
            // One bit above the source width. The rounding term is added
            // *before* the shift, and at the source's own width that
            // addition can carry into the sign bit — `0x7fff + 8` in
            // sixteen bits is negative, which would turn a saturation at
            // the top of the range into one at the bottom. ARM defines
            // the rounding on the unbounded integer, and this is the
            // narrowest width that reproduces it.
            let wide = lane_bits.checked_mul(2)?.checked_add(1)?;
            let mut value = extend(a, wide, signed_sources);
            if rounding {
                // Half an ulp of the shift, added before the low bits
                // are discarded.
                let half = 1u128.checked_shl(u32::from(shift - 1))?;
                value = Expr::add(value, Expr::konst(half, wide));
            }
            let shifted = if signed_sources {
                Expr::ashr(value, Expr::konst(u128::from(shift), wide))
            } else {
                Expr::lshr(value, Expr::konst(u128::from(shift), wide))
            };
            match to {
                Some(target) => clamp(shifted, wide, lane_bits, target),
                // `shrn` / `rshrn` truncate: with the shift bounded by
                // the destination's element width, every surviving bit
                // is a real bit of the source rather than fill.
                None => Some(Expr::extract(shifted, lane_bits - 1, 0)),
            }
        }
        SaturatingKind::DoublingMultiplyHigh { rounding } => {
            // The product needs `2 * lane_bits`, and doubling it needs
            // one more — which is exactly the `INT_MIN * INT_MIN` corner
            // where this instruction saturates.
            let wide = lane_bits.checked_mul(2)?.checked_add(1)?;
            let product = Expr::mul(extend(a, wide, true), extend(b?, wide, true));
            let mut doubled = Expr::shl(product, Expr::konst(1, wide));
            if rounding {
                let half = 1u128.checked_shl(u32::from(lane_bits - 1))?;
                doubled = Expr::add(doubled, Expr::konst(half, wide));
            }
            let high = Expr::ashr(doubled, Expr::konst(u128::from(lane_bits), wide));
            clamp(high, wide, lane_bits, SaturateTo::Signed)
        }
    }
}

/// Shift `value` right by `amount`, rounding when asked.
///
/// Rounding adds half an ulp of the shift *before* the low bits are
/// discarded, and at the element's own width that addition can carry out
/// — `0xff + 1` in eight bits is zero. The sum is therefore taken one
/// bit wider, which is the narrowest width that reproduces ARM's
/// unbounded-integer definition.
fn shift_right(
    value: Expr,
    amount: &Expr,
    lane_bits: u16,
    signed: bool,
    rounding: bool,
) -> Option<Expr> {
    let shift_by = |v: Expr, by: Expr| {
        if signed {
            Expr::ashr(v, by)
        } else {
            Expr::lshr(v, by)
        }
    };
    if !rounding {
        return Some(shift_by(value, amount.clone()));
    }
    let wide = lane_bits.checked_add(1)?;
    let wide_amount = Expr::zero_ext(amount.clone(), wide);
    let half = Expr::shl(
        Expr::konst(1, wide),
        Expr::sub(wide_amount.clone(), Expr::konst(1, wide)),
    );
    let sum = Expr::add(extend(value, wide, signed), half);
    Some(Expr::extract(shift_by(sum, wide_amount), lane_bits - 1, 0))
}

/// One destination lane of a same-width shift.
///
/// The register forms read a *signed* per-lane amount whose sign chooses
/// the direction, so both directions are built and an `Ite` selects
/// between them. An out-of-range amount needs no special case: a
/// bit-vector shift by more than the width already yields zero, or all
/// sign bits for an arithmetic right shift, which is what the
/// architecture specifies.
fn shift_lane(
    kind: ShiftKind,
    signed: bool,
    value: Expr,
    amount: Option<Expr>,
    lane_bits: u16,
) -> Option<Expr> {
    match kind {
        ShiftKind::LeftImmediate { shift } => {
            Some(Expr::shl(value, Expr::konst(u128::from(shift), lane_bits)))
        }
        ShiftKind::RightImmediate { shift, rounding } => shift_right(
            value,
            &Expr::konst(u128::from(shift), lane_bits),
            lane_bits,
            signed,
            rounding,
        ),
        ShiftKind::Register { rounding } => {
            // Only the low byte of the amount element is read, as a
            // signed value.
            let raw = amount?;
            let signed_amount = if lane_bits > BITS_PER_BYTE {
                Expr::sign_ext(Expr::extract(raw, BITS_PER_BYTE - 1, 0), lane_bits)
            } else {
                raw
            };
            let left = Expr::shl(value.clone(), signed_amount.clone());
            let negated = Expr::sub(Expr::konst(0, lane_bits), signed_amount.clone());
            let right = shift_right(value, &negated, lane_bits, signed, rounding)?;
            Some(Expr::Ite {
                cond: Box::new(Expr::slt(signed_amount, Expr::konst(0, lane_bits))),
                then_expr: Box::new(right),
                else_expr: Box::new(left),
            })
        }
    }
}
