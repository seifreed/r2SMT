//! The `AArch32` NEON families whose result is clamped into the
//! destination element instead of being allowed to wrap.
//!
//! What puts them here rather than in [`super::super::neon_packed_op`]'s
//! integer table is that every one computes at a width its operands do
//! not have. `|INT_MIN|` and `-INT_MIN` are both one past the top of a
//! signed element, and a doubled product is one bit past the double
//! width a plain `vmull` produces — so a lowering that computed at the
//! element's own width would wrap, and the clamp that follows would then
//! be clamping a value that already lost the overflow it exists to
//! detect. Each resolver therefore states the headroom width and the
//! lowering clamps from there.
//!
//! None of these mnemonics is spelled by the scalar VFP family, so a
//! form a resolver here misses declines rather than being lowered as a
//! lane-zero operation. That is *not* a licence to be loose about the
//! operand checks: the packed arm below this one in the dispatcher
//! knows `vqadd` / `vqsub` / `vqshl`, and the register-class checks
//! [`uniform_vector_view`] performs are what keeps the two families
//! from overlapping.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use crate::lift::LiftCtx;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;
use crate::lift::simd::clamp_to_element;

use super::{NeonOp, NeonShape, uniform_vector_view, vector_parent_bits, vector_view_bits};

/// Widest element the saturating unary forms encode. The architecture
/// gives `VQABS` / `VQNEG` byte, halfword and word elements and no
/// doubleword one.
const MAX_UNARY_ELEMENT_BITS: u16 = 32;

/// The elements a doubling multiply encodes.
///
/// Both the same-width `VQDMULH` and the long `VQDMULL` are halfword and
/// word only — a doubled byte product would be a sixteen-bit destination
/// element the same-width form has no room for, and a doubled word
/// product a 64-bit one the long form's destination could hold but the
/// architecture does not encode.
const DOUBLING_ELEMENT_BITS: [u16; 2] = [16, 32];

/// The signedness an element type names, for the families that have a
/// signed encoding alone.
///
/// Every member here is signed-only in the architecture, and for the
/// same reason: a magnitude, a negation and a doubled product that
/// saturates at `INT_MAX` are all statements about a two's-complement
/// sign. There is nothing for an unsigned or an untyped spelling to
/// mean, so one is a decline rather than a signedness to carry.
fn is_signed_element(element: ElementKind) -> bool {
    element == ElementKind::Signed
}

/// `vqabs` / `vqneg` — the lane-wise magnitude and negation, clamped
/// into the element instead of wrapping.
///
/// The single value that saturates is `INT_MIN`, whose magnitude and
/// whose negation are both `INT_MAX + 1`; every other lane is exact.
pub(super) fn saturating_unary_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let negate = match base {
        "vqabs" => false,
        "vqneg" => true,
        _ => return None,
    };
    let (element, lane_bits) = neon_element_type(ty)?;
    if !is_signed_element(element) || lane_bits > MAX_UNARY_ELEMENT_BITS {
        return None;
    }
    if insn.operands.len() != 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    if view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::SaturatingUnary { negate },
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}

/// `vqdmulh` / `vqrdmulh` — the doubled product's high half, clamped.
///
/// The by-element spelling (`vqdmulh.s16 q0, q1, d2[0]`) is refused
/// here: [`uniform_vector_view`] declines an operand carrying vector
/// shape, and no other resolver claims the mnemonic, so it falls through
/// to a decline rather than being read as a whole-vector multiply. That
/// is the sound answer — reading `d2` entire where the instruction names
/// one of its lanes would be a wrong value.
pub(super) fn doubling_multiply_high_shape(
    insn: &Instruction,
    mnemonic: &str,
) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let rounding = match base {
        "vqdmulh" => false,
        "vqrdmulh" => true,
        _ => return None,
    };
    let (element, lane_bits) = neon_element_type(ty)?;
    if !is_signed_element(element) || !DOUBLING_ELEMENT_BITS.contains(&lane_bits) {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    if view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::DoublingMultiplyHigh { rounding },
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}

/// `vqdmull` / `vqdmlal` / `vqdmlsl` — the doubled product kept whole in
/// a destination element twice the source's.
///
/// The mnemonic names the **source** element, as every `AArch32`
/// widening form does, so the destination is the full parent view and
/// each source half of it.
pub(super) fn doubling_multiply_long_shape(
    insn: &Instruction,
    mnemonic: &str,
) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let accumulate = match base {
        "vqdmull" => None,
        "vqdmlal" => Some(false),
        "vqdmlsl" => Some(true),
        _ => return None,
    };
    let (element, source_bits) = neon_element_type(ty)?;
    if !is_signed_element(element) || !DOUBLING_ELEMENT_BITS.contains(&source_bits) {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let lane_bits = source_bits.checked_mul(2)?;
    let destination_view = vector_parent_bits()?;
    if vector_view_bits(insn.operands.first()?)? != destination_view {
        return None;
    }
    let lanes = destination_view.checked_div(lane_bits)?;
    let narrow_view = source_bits.checked_mul(lanes)?;
    for operand in insn.operands.iter().skip(1) {
        if vector_view_bits(operand)? != narrow_view {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::DoublingMultiplyLong { accumulate },
        lane_bits,
        lanes,
    })
}

impl LiftCtx {
    /// `vqabs` / `vqneg`.
    ///
    /// One extra bit is the whole model: at the element's own width
    /// `-INT_MIN` is `INT_MIN` again, so the clamp would see a negative
    /// value and leave it alone instead of raising it to `INT_MAX`.
    pub(super) fn aarch32_saturating_unary_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        negate: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let wide = shape.lane_bits.checked_add(1)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let element = Expr::sign_ext(
                Self::extract_lane(source.clone(), shape.lane_bits, index)?,
                wide,
            );
            let negated = Expr::sub(Expr::konst(0, wide), element.clone());
            let value = if negate {
                negated
            } else {
                Expr::Ite {
                    cond: Box::new(Expr::slt(element.clone(), Expr::konst(0, wide))),
                    then_expr: Box::new(negated),
                    else_expr: Box::new(element),
                }
            };
            lanes.push(clamp_to_element(value, wide, shape.lane_bits, true)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `vqdmulh` / `vqrdmulh`.
    ///
    /// The product needs `2n` bits and doubling it needs one more, which
    /// is exactly the `INT_MIN * INT_MIN` corner this instruction
    /// saturates at: `2 * (-2^(n-1))^2 >> (n-1)` is `2^(n-1)`, one past
    /// the element, and the clamp turns it into `INT_MAX`.
    ///
    /// The rounding term is half an ulp of the discarded low half, added
    /// at the same headroom width for the reason the narrowing family
    /// documents — added at `2n` it could carry into the sign bit and
    /// turn a saturation at the top of the range into one at the bottom.
    pub(super) fn aarch32_doubling_multiply_high_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        rounding: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let wide = shape.lane_bits.checked_mul(2)?.checked_add(1)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = Self::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = Self::extract_lane(second.clone(), shape.lane_bits, index)?;
            let mut doubled = Expr::shl(
                Expr::mul(Expr::sign_ext(a, wide), Expr::sign_ext(b, wide)),
                Expr::konst(1, wide),
            );
            if rounding {
                let half = 1u128.checked_shl(u32::from(shape.lane_bits.checked_sub(1)?))?;
                doubled = Expr::add(doubled, Expr::konst(half, wide));
            }
            let high = Expr::ashr(doubled, Expr::konst(u128::from(shape.lane_bits), wide));
            lanes.push(clamp_to_element(high, wide, shape.lane_bits, true)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `vqdmull` / `vqdmlal` / `vqdmlsl`.
    ///
    /// Two saturations, not one, and the order is what the architecture
    /// defines: the doubled product is clamped into the destination
    /// element first, and only then combined with the accumulator and
    /// clamped again. Clamping once at the end would let `INT_MIN *
    /// INT_MIN` cancel against a negative accumulator and land inside
    /// the range, which is a different number from the one the machine
    /// produces.
    pub(super) fn aarch32_doubling_multiply_long_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        accumulate: Option<bool>,
    ) -> Option<Expr> {
        let wide_view = shape.view_bits()?;
        let narrow_bits = shape.lane_bits.checked_div(2)?;
        let narrow_view = narrow_bits.checked_mul(shape.lanes)?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), narrow_view)?;
        let second = self.simd_operand_value(&insn.operands.get(2)?.clone(), narrow_view)?;
        let previous = match accumulate {
            Some(_) => Some(self.simd_operand_value(&insn.operands.first()?.clone(), wide_view)?),
            None => None,
        };
        // One bit above the destination element: enough for the doubled
        // product before it is clamped, and enough for the sum of two
        // in-range elements after it is.
        let wide = shape.lane_bits.checked_add(1)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = Self::extract_lane(first.clone(), narrow_bits, index)?;
            let b = Self::extract_lane(second.clone(), narrow_bits, index)?;
            let doubled = Expr::shl(
                Expr::mul(Expr::sign_ext(a, wide), Expr::sign_ext(b, wide)),
                Expr::konst(1, wide),
            );
            let product = clamp_to_element(doubled, wide, shape.lane_bits, true)?;
            lanes.push(match (accumulate, previous.as_ref()) {
                (None, _) => product,
                (Some(subtract), Some(accumulator)) => {
                    let acc = Expr::sign_ext(
                        Self::extract_lane(accumulator.clone(), shape.lane_bits, index)?,
                        wide,
                    );
                    let product = Expr::sign_ext(product, wide);
                    let combined = if subtract {
                        Expr::sub(acc, product)
                    } else {
                        Expr::add(acc, product)
                    };
                    clamp_to_element(combined, wide, shape.lane_bits, true)?
                }
                (Some(_), None) => return None,
            });
        }
        Self::concat_lanes(lanes)
    }
}
