//! The `AArch32` NEON families that come in a plain form and an
//! accumulating twin: `vabd` / `vaba` and `vpaddl` / `vpadal`.
//!
//! The twin is what puts them here rather than beside the lane-wise
//! families. `vaba` and `vpadal` read their destination as a third
//! input, so the lowering has to materialise it as a source and the
//! effect table has to record it as a use — which on `AArch32` it
//! already does, since a vector write merges and the destination's prior
//! value is live whatever the operation.
//!
//! Both also compute at a width their operands do not have. An absolute
//! difference of two `n`-bit elements needs `n + 1` bits to be
//! representable before the magnitude is taken — `|(-128) - 127|` is
//! `255`, which no signed byte holds — and a pairwise-long sum is
//! defined at twice its source element outright. Computing either at the
//! source width would wrap, which is a wrong value rather than a
//! decline.
//!
//! [`super::super::neon_packed_op`] reaches neither: its integer table
//! lists no `vabd` and no `vpaddl`, and a form it does not know does not
//! decline — it falls through to the arms below it. That is why these
//! resolve here, ahead of both the packed and the scalar arms of the
//! dispatcher.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use crate::lift::LiftCtx;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;

use super::lower::extend;
use super::{NeonOp, NeonShape, uniform_vector_view};

/// Widest element either family encodes. `VABD` has no doubleword form,
/// and a doubleword *source* would give `VPADDL` a 128-bit destination
/// element the register file has no room for.
const MAX_SOURCE_ELEMENT_BITS: u16 = 32;

/// The signedness an integer element type names, or `None` for the
/// classes these families have no encoding for.
///
/// Both reject the untyped `i` spelling, and for the same reason `vmax`
/// does: the comparison is the operation. `|a - b|` and the extension a
/// pairwise-long sum performs both change with the sign, so there is
/// nothing for a sign-agnostic form to mean. The float classes are
/// rejected here too — `VABD.F32` exists in the architecture but is a
/// different lane path, and declining it is sound.
fn integer_signedness(element: ElementKind) -> Option<bool> {
    match element {
        ElementKind::Signed => Some(true),
        ElementKind::Unsigned => Some(false),
        ElementKind::Untyped | ElementKind::Float => None,
    }
}

/// `vabd` / `vaba` — the lane-wise absolute difference, optionally
/// accumulated onto the destination.
pub(super) fn absolute_difference_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let accumulate = match base {
        "vabd" => false,
        "vaba" => true,
        _ => return None,
    };
    let (element, lane_bits) = neon_element_type(ty)?;
    let signed = integer_signedness(element)?;
    if insn.operands.len() != 3 || lane_bits > MAX_SOURCE_ELEMENT_BITS {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    if view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::AbsoluteDifference { signed, accumulate },
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}

/// `vpaddl` / `vpadal` — adjacent source elements summed into one
/// destination element twice as wide.
///
/// The mnemonic names the **source** element, as every `AArch32`
/// widening form does, and both operands name the same view: the element
/// doubles, so half as many of them fill the same register.
pub(super) fn pairwise_long_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let accumulate = match base {
        "vpaddl" => false,
        "vpadal" => true,
        _ => return None,
    };
    let (element, source_bits) = neon_element_type(ty)?;
    let signed = integer_signedness(element)?;
    if insn.operands.len() != 2 || source_bits > MAX_SOURCE_ELEMENT_BITS {
        return None;
    }
    let lane_bits = source_bits.checked_mul(2)?;
    let view = uniform_vector_view(insn, 2)?;
    if lane_bits > view || view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::PairwiseLong { signed, accumulate },
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}

/// One destination element of `vabd` — the magnitude of the difference,
/// as an unsigned value of the element's own width.
///
/// Computed one bit wider than the source, which is the narrowest width
/// that holds every difference: at the signed extreme `(-2^(n-1)) -
/// (2^(n-1) - 1)` is `-(2^n - 1)`, and its magnitude `2^n - 1` is
/// exactly the largest positive value `n + 1` signed bits carry. The
/// result then fills the element unsigned, which is why `|(-128) - 127|`
/// is the byte `0xff` and not a saturation.
fn absolute_difference_lane(signed: bool, a: Expr, b: Expr, bits: u16) -> Option<Expr> {
    let wide = bits.checked_add(1)?;
    let difference = Expr::sub(extend(a, wide, signed), extend(b, wide, signed));
    let magnitude = Expr::Ite {
        cond: Box::new(Expr::slt(difference.clone(), Expr::konst(0, wide))),
        then_expr: Box::new(Expr::sub(Expr::konst(0, wide), difference.clone())),
        else_expr: Box::new(difference),
    };
    Some(Expr::extract(magnitude, bits.checked_sub(1)?, 0))
}

impl LiftCtx {
    /// `vabd` / `vaba`.
    pub(super) fn aarch32_absolute_difference_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
        accumulate: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let previous = if accumulate {
            Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?)
        } else {
            None
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = Self::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = Self::extract_lane(second.clone(), shape.lane_bits, index)?;
            let magnitude = absolute_difference_lane(signed, a, b, shape.lane_bits)?;
            lanes.push(match previous.as_ref() {
                // The accumulation wraps at the element width, which is
                // what `VABA` defines — there is a separate saturating
                // family and this is not in it.
                Some(accumulator) => Expr::add(
                    Self::extract_lane(accumulator.clone(), shape.lane_bits, index)?,
                    magnitude,
                ),
                None => magnitude,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// `vpaddl` / `vpadal`.
    ///
    /// The sum is computed at the destination's element width with both
    /// sources extended into it, so it cannot overflow: two `n`-bit
    /// values sum in `n + 1` bits and the destination has `2n`.
    pub(super) fn aarch32_pairwise_long_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
        accumulate: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let previous = if accumulate {
            Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?)
        } else {
            None
        };
        let narrow = shape.lane_bits.checked_div(2)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let low_index = index.checked_mul(2)?;
            let low = Self::extract_lane(source.clone(), narrow, low_index)?;
            let high = Self::extract_lane(source.clone(), narrow, low_index.checked_add(1)?)?;
            let sum = Expr::add(
                extend(low, shape.lane_bits, signed),
                extend(high, shape.lane_bits, signed),
            );
            lanes.push(match previous.as_ref() {
                Some(accumulator) => Expr::add(
                    Self::extract_lane(accumulator.clone(), shape.lane_bits, index)?,
                    sum,
                ),
                None => sum,
            });
        }
        Self::concat_lanes(lanes)
    }
}
