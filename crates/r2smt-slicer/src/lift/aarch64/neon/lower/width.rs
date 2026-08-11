//! Lowering for the `AArch64` NEON families that relate two different
//! element geometries.
//!
//! Widening and narrowing arithmetic, the conversions, the across-lane
//! reductions, the pairwise-long accumulations, and the high-half
//! narrows — so each source is sized against the destination rather than
//! assumed equal to it.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use super::{extend, integer_min_max};
use crate::lift::aarch64::neon::NeonShape;
use crate::lift::aarch64::neon::geometry::operand_arrangement;
use crate::lift::aarch64::neon::width::{ConvertKind, ReduceKind, WidenKind};
use crate::lift::{LiftCtx, fp_propagating_max_min};

impl LiftCtx {
    /// The across-lane reductions — every source lane folded into the
    /// one element the scalar destination holds.
    ///
    /// The fold is left-associative, which the architecture permits to
    /// matter for none of these: integer addition and the min / max
    /// selects are associative, so no ordering is observable.
    pub(super) fn reduce_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ReduceKind,
        source_lanes: u16,
        source_lane_bits: u16,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let mut accumulator: Option<Expr> = None;
        for index in 0..source_lanes {
            let raw = LiftCtx::extract_lane(source.clone(), source_lane_bits, index)?;
            // The widening forms extend before the fold; the same-width
            // ones already sit at the destination's element width.
            let element = match kind {
                ReduceKind::AddLong { signed } => extend(raw, shape.lane_bits, signed),
                ReduceKind::Add | ReduceKind::MinMax { .. } | ReduceKind::Float { .. } => raw,
            };
            accumulator = Some(match accumulator {
                None => element,
                Some(previous) => reduce_step(kind, previous, element, shape.lane_bits)?,
            });
        }
        accumulator
    }

    /// `frint<mode>` — each lane rounded to an integral value, still a
    /// float. `saturate` carries the `frint32*` / `frint64*` clamp.
    pub(super) fn round_to_integral_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        rounding: r2smt_ir::expr::RoundingMode,
        saturate: Option<u16>,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let (ebits, sbits) = fp_sort(shape.lane_bits)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let element = LiftCtx::extract_lane(source.clone(), shape.lane_bits, index)?;
            lanes.push(round_to_integral_lane(
                Expr::bv_to_fp(element, ebits, sbits),
                shape.lane_bits,
                rounding,
                saturate,
            )?);
        }
        Self::concat_lanes(lanes)
    }

    /// The lane-wise conversions.
    /// `fcvtas w0, s1` — one float element converted into a general
    /// register, at the register's own width.
    ///
    /// The two widths are independent here, which is exactly the shape
    /// [`convert_lane`] already takes: it has carried a separate
    /// `source_bits` and `lane_bits` since the widening conversions
    /// needed them. So this reads the element and hands both widths
    /// over, and the unsigned spellings keep the extra bit of range that
    /// makes them cover the whole unsigned interval rather than
    /// saturating at the signed maximum.
    pub(super) fn float_to_gpr(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
        rounding: r2smt_ir::expr::RoundingMode,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let element = LiftCtx::extract_lane(source, shape.lane_bits, 0)?;
        let destination_bits = self.operand_width(insn.operands.first()?);
        convert_lane(
            ConvertKind::FloatToInt { signed },
            element,
            shape.lane_bits,
            destination_bits,
            0,
            rounding,
        )
    }

    pub(super) fn convert_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ConvertKind,
        upper: bool,
        fbits: u16,
        rounding: r2smt_ir::expr::RoundingMode,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let source_bits = match kind {
            ConvertKind::FloatToFloat { widening: true } => shape.lane_bits / 2,
            ConvertKind::FloatToFloat { widening: false } => shape.lane_bits.checked_mul(2)?,
            _ => shape.lane_bits,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            // `fcvtl2` reads the source's upper half; `fcvtn2` reads the
            // same lanes `fcvtn` does and moves only the destination.
            let source_lane =
                if upper && matches!(kind, ConvertKind::FloatToFloat { widening: true }) {
                    index.checked_add(shape.lanes)?
                } else {
                    index
                };
            let element = LiftCtx::extract_lane(source.clone(), source_bits, source_lane)?;
            // The mode comes from the mnemonic on every one of these —
            // `fcvtz*` truncates, `fcvta*` / `fcvtn*` / `fcvtp*` /
            // `fcvtm*` name the other four — so no control register is
            // assumed and none of them pins one.
            lanes.push(convert_lane(
                kind,
                element,
                source_bits,
                shape.lane_bits,
                fbits,
                rounding,
            )?);
        }
        let converted = Self::concat_lanes(lanes)?;
        if !upper || !matches!(kind, ConvertKind::FloatToFloat { widening: false }) {
            return Some(converted);
        }
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            converted,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// The widening and narrowing family: read each source element,
    /// extend or truncate it to the destination's width, and operate
    /// there.
    ///
    /// Extending *before* operating is the whole point of the family —
    /// it is what stops the result overflowing the element, which is
    /// exactly what the same-width lane-wise form would do.
    pub(super) fn widen_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: WidenKind,
        signed: bool,
        upper: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = match insn.operands.get(2) {
            Some(operand) if operand_arrangement(operand).is_some() => {
                Some(self.widen_source(insn, 2)?)
            }
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            lanes.push(Self::widen_lane(
                &first,
                second.as_ref(),
                shape,
                kind,
                signed,
                index,
                upper,
            )?);
        }
        let narrowed = Self::concat_lanes(lanes)?;
        if !matches!(kind, WidenKind::Narrow) || !upper {
            return Some(narrowed);
        }
        // `xtn2` writes the destination's upper half and preserves the
        // lower one.
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            narrowed,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// One destination lane of a widening or narrowing operation.
    ///
    /// `upper` shifts the lane read out of a *narrow* source by half a
    /// register, which is what the `2` suffix means. A wide source is
    /// never halved — `uaddw2 v0.4s, v1.4s, v2.8h` takes `v1`'s lane `i`
    /// and `v2`'s lane `i + 4`.
    fn widen_lane(
        first: &Expr,
        second: Option<&Expr>,
        shape: NeonShape,
        kind: WidenKind,
        signed: bool,
        index: u16,
        upper: bool,
    ) -> Option<Expr> {
        let extend = |value: Expr| {
            if signed {
                Expr::sign_ext(value, shape.lane_bits)
            } else {
                Expr::zero_ext(value, shape.lane_bits)
            }
        };
        let narrow_lane = if upper {
            index.checked_add(shape.lanes)?
        } else {
            index
        };
        let narrow = shape.lane_bits / 2;
        match kind {
            WidenKind::Narrow => {
                // Truncation needs no signedness: the low half of a
                // two's-complement value is the same bits either way.
                // The source is wide, so it is never halved — `xtn2`
                // narrows the same lanes `xtn` does and only the
                // *destination* half differs.
                let wide = shape.lane_bits.checked_mul(2)?;
                let element = LiftCtx::extract_lane(first.clone(), wide, index)?;
                Some(Expr::extract(element, shape.lane_bits - 1, 0))
            }
            WidenKind::ShiftLong { shift } => {
                let element = LiftCtx::extract_lane(first.clone(), narrow, narrow_lane)?;
                let widened = extend(element);
                if shift == 0 {
                    return Some(widened);
                }
                Some(Expr::shl(
                    widened,
                    Expr::konst(u128::from(shift), shape.lane_bits),
                ))
            }
            WidenKind::Arith { op, wide_first } => {
                // A `w`-form's first source is already at the
                // destination's width and is read at its own lane index.
                let lhs = if wide_first {
                    LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?
                } else {
                    extend(LiftCtx::extract_lane(first.clone(), narrow, narrow_lane)?)
                };
                let rhs = extend(LiftCtx::extract_lane(second?.clone(), narrow, narrow_lane)?);
                Some(op.apply(lhs, rhs))
            }
        }
    }

    /// `saddlp` / `uaddlp` / `sadalp` / `uadalp` — adjacent source lanes
    /// extended to the destination's element width and summed there.
    ///
    /// The extension happens before the addition, so the sum is exact:
    /// two `n`-bit values always fit `n + 1` bits, and the destination
    /// holds `2n`.
    pub(super) fn pairwise_long_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
        accumulate: bool,
    ) -> Option<Expr> {
        let narrow = shape.lane_bits / 2;
        let source = self.widen_source(insn, 1)?;
        let accumulator = if accumulate {
            Some(self.destination_value(insn, shape)?)
        } else {
            None
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let low = index.checked_mul(2)?;
            let a = LiftCtx::extract_lane(source.clone(), narrow, low)?;
            let b = LiftCtx::extract_lane(source.clone(), narrow, low.checked_add(1)?)?;
            let sum = Expr::add(
                extend(a, shape.lane_bits, signed),
                extend(b, shape.lane_bits, signed),
            );
            lanes.push(match accumulator.as_ref() {
                Some(previous) => Expr::add(
                    LiftCtx::extract_lane(previous.clone(), shape.lane_bits, index)?,
                    sum,
                ),
                None => sum,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// `addhn` / `subhn` / `raddhn` / `rsubhn`.
    pub(super) fn high_narrow_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        subtract: bool,
        rounding: bool,
        upper: bool,
    ) -> Option<Expr> {
        let source_bits = shape.lane_bits.checked_mul(2)?;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), source_bits, index)?;
            let b = LiftCtx::extract_lane(second.clone(), source_bits, index)?;
            lanes.push(high_narrow_lane(subtract, rounding, a, b, shape.lane_bits)?);
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
}

/// Fold one more lane into an across-lane reduction's accumulator.
///
/// `addv` sums at the element width and keeps the low bits, which is
/// what the architecture's truncation of the unbounded sum comes to; a
/// wrapping add reproduces it exactly. `uaddlv` / `saddlv` arrive here
/// already extended, and no encodable arrangement can overflow the
/// doubled width (sixteen bytes reach 4 080, eight halfwords 524 280),
/// so their sum is exact rather than truncated too.
/// `fmaxv` / `fminv` fold with [`fp_propagating_max_min`] and *not*
/// with the integer comparison beside it, nor with
/// [`super::super::super::fp_lane_result`]'s `Max` / `Min`, which is
/// Intel's `MAXPS`: ARM's `FPMax` propagates NaN where `MAXPS` returns
/// its second operand, and combines the signs of a zero tie where
/// `MAXPS` again takes the second. Both differences are wrong values,
/// not wider ones.
fn reduce_step(kind: ReduceKind, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    Some(match kind {
        ReduceKind::Add | ReduceKind::AddLong { .. } => Expr::add(a, b),
        ReduceKind::Float { max, number_wins } => {
            return fp_propagating_max_min(a, b, lane_bits, max, number_wins);
        }
        ReduceKind::MinMax { signed, max } => integer_min_max(a, b, signed, max),
    })
}

/// One destination lane of `addhn` / `subhn` and their rounding forms:
/// the high half of the double-width sum or difference.
///
/// The arithmetic wraps at the source width and that loses nothing. ARM
/// defines the result on the unbounded integer, but the window kept
/// (`sum<2n-1:n>`) sits entirely inside the low `2n` bits, which
/// wrapping preserves exactly — including for the rounding forms, whose
/// added half ulp can carry out of the top without touching a bit the
/// window keeps. This is where the family differs from the saturating
/// narrows next door: there the same carry reaches the *sign* bit the
/// clamp compares against, turning a saturation at `INT_MAX` into one at
/// `INT_MIN`, and the lowering has to compute a bit wider to avoid it.
pub(in crate::lift) fn high_narrow_lane(
    subtract: bool,
    rounding: bool,
    a: Expr,
    b: Expr,
    lane_bits: u16,
) -> Option<Expr> {
    let source_bits = lane_bits.checked_mul(2)?;
    let mut value = if subtract {
        Expr::sub(a, b)
    } else {
        Expr::add(a, b)
    };
    if rounding {
        // Half an ulp of the discarded low half.
        let half = 1u128.checked_shl(u32::from(lane_bits.checked_sub(1)?))?;
        value = Expr::add(value, Expr::konst(half, source_bits));
    }
    Some(Expr::extract(value, source_bits - 1, lane_bits))
}

/// The IEEE `(ebits, sbits)` pair for a float lane width.
fn fp_sort(lane_bits: u16) -> Option<(u16, u16)> {
    match lane_bits {
        16 => Some((5, 11)),
        32 => Some((8, 24)),
        64 => Some((11, 53)),
        _ => None,
    }
}

/// The IEEE bit pattern of `2^exponent` in the `(ebits, sbits)` sort,
/// or `None` when that power is not representable there.
///
/// A subnormal result is built rather than declined: the fixed-point
/// conversions scale by `2^-fbits`, and for a 16-bit element with
/// `fbits = 16` that lands one exponent below the normal range — where
/// the value is still exact, since a power of two is a single set bit
/// of the stored significand.
fn power_of_two(exponent: i32, ebits: u16, sbits: u16) -> Option<u128> {
    let bias = (1i32 << (i32::from(ebits) - 1)) - 1;
    let stored = i32::from(sbits) - 1;
    let field = exponent.checked_add(bias)?;
    if field >= 1 {
        // Normal: the significand is implicit, so the pattern is the
        // biased exponent alone. The all-ones field is infinity / NaN.
        if field >= (1i32 << i32::from(ebits)) - 1 {
            return None;
        }
        return Some(u128::try_from(field).ok()? << u32::try_from(stored).ok()?);
    }
    // Subnormal: the value is `significand * 2^(1 - bias - stored)`, so
    // the set bit sits `exponent - (1 - bias - stored)` places up.
    let bit = exponent.checked_sub(1 - bias - stored)?;
    if bit < 0 || bit >= stored {
        return None;
    }
    Some(1u128 << u32::try_from(bit).ok()?)
}

/// Scale `value` by `2^-fbits`, the fixed-point conversions' fraction
/// width, in whichever direction `divide` names.
///
/// The constant is always built as `2^-fbits` and never as `2^fbits`,
/// which is the only spelling representable at every encodable width:
/// `2^16` is infinity in binary16, while `2^-16` is a perfectly good
/// subnormal. Scaling by a power of two only shifts the exponent, so it
/// introduces no rounding of its own — the result is the same one the
/// architecture's single correctly-rounded conversion produces, except
/// in a subnormal corner where the integer is small enough to be exact
/// anyway.
fn scale_by_fraction(
    value: Expr,
    fbits: u16,
    ebits: u16,
    sbits: u16,
    divide: bool,
) -> Option<Expr> {
    if fbits == 0 {
        return Some(value);
    }
    let scale = Expr::FpConst {
        bits: power_of_two(-i32::from(fbits), ebits, sbits)?,
        ebits,
        sbits,
    };
    let round = r2smt_ir::expr::RoundingMode::NearestTiesEven;
    Some(if divide {
        Expr::fdiv(value, scale, round)
    } else {
        Expr::fmul(value, scale, round)
    })
}

/// One destination lane of a conversion.
///
/// The IR carries no unsigned integer conversion, so the unsigned forms
/// go through the signed node with one extra bit of range — which covers
/// the unsigned range exactly rather than approximately, the same trick
/// the scalar `ucvtf` / `fcvtzu` already use.
///
/// `fbits` is the fixed-point fraction width, zero for the plain
/// register forms. It reads the integer side as `Int(lane) / 2^fbits`,
/// so the conversion into float multiplies by that factor and the
/// conversion out of float divides by it.
/// `to_int_rounding` is the mode a float-to-integer conversion rounds
/// with. `AArch64` always passes round-toward-zero, the `z` its only
/// packed spelling carries in the opcode; `AArch32` also reaches here
/// with the directed forms `vcvta` / `vcvtn` / `vcvtp` / `vcvtm` and
/// with `vcvtr`, whose mode is the FPSCR default. The other two
/// directions round to the control word's default on both ISAs and so
/// take no parameter.
pub(in crate::lift) fn convert_lane(
    kind: ConvertKind,
    element: Expr,
    source_bits: u16,
    lane_bits: u16,
    fbits: u16,
    to_int_rounding: r2smt_ir::expr::RoundingMode,
) -> Option<Expr> {
    let round = r2smt_ir::expr::RoundingMode::NearestTiesEven;
    match kind {
        ConvertKind::IntToFloat { signed } => {
            let (ebits, sbits) = fp_sort(lane_bits)?;
            let source = if signed {
                element
            } else {
                Expr::zero_ext(element, source_bits.checked_add(1)?)
            };
            let converted = Expr::sbv_to_fp(source, round, ebits, sbits);
            Some(Expr::fp_to_ieee_bv(scale_by_fraction(
                converted, fbits, ebits, sbits, false,
            )?))
        }
        ConvertKind::FloatToInt { signed } => {
            let (ebits, sbits) = fp_sort(source_bits)?;
            let value = scale_by_fraction(
                Expr::bv_to_fp(element, ebits, sbits),
                fbits,
                ebits,
                sbits,
                true,
            )?;
            Some(if signed {
                Expr::fp_to_sbv(value, to_int_rounding, lane_bits)
            } else {
                let in_range = Expr::extract(
                    Expr::fp_to_sbv(value.clone(), to_int_rounding, lane_bits.checked_add(1)?),
                    lane_bits - 1,
                    0,
                );
                crate::lift::simd::clamp_unsigned_float_to_int(
                    value, ebits, sbits, lane_bits, in_range,
                )
            })
        }
        ConvertKind::FloatToFloat { .. } => {
            // No fixed-point form: `convert_shape` rejects a third
            // operand here, so `fbits` is zero and nothing scales.
            let (from_e, from_s) = fp_sort(source_bits)?;
            let (to_e, to_s) = fp_sort(lane_bits)?;
            Some(Expr::fp_to_ieee_bv(Expr::fp_to_fp(
                Expr::bv_to_fp(element, from_e, from_s),
                round,
                to_e,
                to_s,
            )))
        }
    }
}

/// The IEEE bit pattern of `2^exponent` in the `(ebits, sbits)` sort,
/// negated when `negative`.
///
/// Computed rather than tabulated because the `frint32*` / `frint64*`
/// clamp needs four `(lane, integer width)` combinations and all four
/// are legal: the saturation width comes from the mnemonic and the float
/// sort from the register, so they vary independently.
fn power_of_two_bits(exponent: u16, ebits: u16, sbits: u16, negative: bool) -> Option<u128> {
    let bias = (1u128 << ebits.checked_sub(1)?) - 1;
    let field = bias.checked_add(u128::from(exponent))?;
    if field >= (1u128 << ebits) {
        return None;
    }
    let sign = u128::from(negative) << ebits.checked_add(sbits)?.checked_sub(1)?;
    Some(sign | (field << sbits.checked_sub(1)?))
}

/// One lane of `frint<mode>`, with the `frint32*` / `frint64*`
/// saturation when `saturate` names an integer width.
///
/// `value` is FP-sorted; the result is the lane's bit pattern.
///
/// The clamp is what separates `FEAT_FRINTTS` from the plain family:
/// the result stays a float, but one that the named signed integer width
/// could hold, and everything else — a NaN, an infinity, an out-of-range
/// magnitude — becomes the *most negative* such value rather than
/// anything nearer.
///
/// The upper bound is written `2^(N-1) <= rounded` and not
/// `rounded > 2^(N-1) - 1`, and that is not a stylistic choice:
/// `2^31 - 1` is **not representable in binary32**, so comparing against
/// it would compare against `2^31` after rounding and admit one value
/// the integer cannot hold. `2^(N-1)` is a power of two and exact in
/// both sorts. The NaN test is separate because every float comparison
/// is false on a NaN, so neither bound would catch it.
pub(in crate::lift) fn round_to_integral_lane(
    value: Expr,
    lane_bits: u16,
    rounding: r2smt_ir::expr::RoundingMode,
    saturate: Option<u16>,
) -> Option<Expr> {
    let rounded = Expr::fround_to_integral(value.clone(), rounding);
    let Some(int_bits) = saturate else {
        return Some(Expr::fp_to_ieee_bv(rounded));
    };
    let (ebits, sbits) = fp_sort(lane_bits)?;
    let exponent = int_bits.checked_sub(1)?;
    let limit = |negative| {
        power_of_two_bits(exponent, ebits, sbits, negative).map(|bits| Expr::FpConst {
            bits,
            ebits,
            sbits,
        })
    };
    let most_negative = limit(true)?;
    let out_of_range = Expr::bool_or(
        Expr::fisnan(value),
        Expr::bool_or(
            Expr::flt(rounded.clone(), most_negative.clone()),
            Expr::fle(limit(false)?, rounded.clone()),
        ),
    );
    Some(Expr::Ite {
        cond: Box::new(out_of_range),
        then_expr: Box::new(Expr::fp_to_ieee_bv(most_negative)),
        else_expr: Box::new(Expr::fp_to_ieee_bv(rounded)),
    })
}
