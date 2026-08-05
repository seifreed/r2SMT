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
use r2smt_ir::program::{Instruction, Operand};

use crate::lift::LiftCtx;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;
use crate::lift::simd::clamp_to_element;

use super::element::indexed_element;
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

/// How a doubling multiply addresses its second source.
#[derive(Debug, Clone, Copy)]
pub(in crate::lift::aarch32) enum SecondSource {
    /// The whole vector, paired lane with lane.
    Lanewise,
    /// One lane of it, contributing to every destination lane.
    Element(u16),
}

/// Which of the two spellings `op` is, or `None` for a decline.
///
/// The by-element check has to come first and has to be exact. An
/// indexed operand has the register kind and the operand count the plain
/// spelling accepts, so a resolver that only asked
/// [`vector_view_bits`] would either refuse the form outright — which is
/// what the family did before this — or, worse, read the whole of `d2`
/// where the instruction names one of its lanes.
fn second_source(op: &Operand, source_bits: u16, expected: u16) -> Option<SecondSource> {
    if let Some((view, index)) = indexed_element(op) {
        // The index has to address an element the register holds.
        return (index < view.checked_div(source_bits)?).then_some(SecondSource::Element(index));
    }
    (vector_view_bits(op)? == expected).then_some(SecondSource::Lanewise)
}

/// A doubling multiply's second source, materialised once.
///
/// The by-element spelling contributes the same element to every
/// destination lane, so it is read outside the per-lane loop rather than
/// re-extracted: identical value, and on a memory operand the difference
/// would be one load against one per lane.
enum Multiplier {
    Lanewise(Expr),
    Element(Expr),
}

impl Multiplier {
    fn lane(&self, source_bits: u16, index: u16) -> Option<Expr> {
        match self {
            Self::Element(value) => Some(value.clone()),
            Self::Lanewise(value) => LiftCtx::extract_lane(value.clone(), source_bits, index),
        }
    }
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

/// `vqdmulh` / `vqrdmulh` — the doubled product's high half, clamped,
/// in both the lane-wise and the by-element spelling.
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
    same_width_doubling_shape(insn, ty, |source| NeonOp::DoublingMultiplyHigh {
        rounding,
        source,
    })
}

/// `vqrdmlah` / `vqrdmlsh` — the rounded doubled product combined with
/// the destination, in both spellings.
///
/// Separate from [`doubling_multiply_high_shape`] only because the
/// lowering saturates once over the whole expression where that one
/// saturates the product alone; the geometry is identical, which is why
/// both go through the same resolver body.
pub(super) fn doubling_multiply_accumulate_shape(
    insn: &Instruction,
    mnemonic: &str,
) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let subtract = match base {
        "vqrdmlah" => false,
        "vqrdmlsh" => true,
        _ => return None,
    };
    same_width_doubling_shape(insn, ty, |source| NeonOp::DoublingMultiplyAccumulate {
        subtract,
        source,
    })
}

/// The geometry the same-width doubling multiplies share: three
/// operands, a signed halfword or word element, and a second source that
/// is either the whole vector or one of its lanes.
fn same_width_doubling_shape(
    insn: &Instruction,
    ty: &str,
    op: impl FnOnce(SecondSource) -> NeonOp,
) -> Option<NeonShape> {
    let (element, lane_bits) = neon_element_type(ty)?;
    if !is_signed_element(element) || !DOUBLING_ELEMENT_BITS.contains(&lane_bits) {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let view = vector_view_bits(insn.operands.first()?)?;
    if vector_view_bits(insn.operands.get(1)?)? != view || view % lane_bits != 0 {
        return None;
    }
    let source = second_source(insn.operands.get(2)?, lane_bits, view)?;
    Some(NeonShape {
        op: op(source),
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
    if vector_view_bits(insn.operands.get(1)?)? != narrow_view {
        return None;
    }
    let source = second_source(insn.operands.get(2)?, source_bits, narrow_view)?;
    Some(NeonShape {
        op: NeonOp::DoublingMultiplyLong { accumulate, source },
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
        source: SecondSource,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.doubling_multiplier(insn, source, shape.lane_bits, view)?;
        let wide = shape.lane_bits.checked_mul(2)?.checked_add(1)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for lane in 0..shape.lanes {
            let a = Self::extract_lane(first.clone(), shape.lane_bits, lane)?;
            let b = second.lane(shape.lane_bits, lane)?;
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

    /// `vqrdmlah` / `vqrdmlsh`.
    ///
    /// **One** saturation, and that is the whole difference from
    /// `vqrdmulh` followed by a saturating add. ARM scales the
    /// accumulator up by the element width, adds the doubled product and
    /// the rounding term to it, and clamps the shifted-down total once —
    /// so a product that would have saturated on its own can still land
    /// inside the range once a negative accumulator is applied. Clamping
    /// twice would pin it at `INT_MAX` first and give a different number.
    ///
    /// Two bits of headroom above the double width: the scaled
    /// accumulator and the doubled product each reach `2^(2n-1)`, so
    /// their sum needs `2n + 1` bits and the rounding term one more.
    pub(super) fn aarch32_doubling_multiply_accumulate_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        subtract: bool,
        source: SecondSource,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.doubling_multiplier(insn, source, shape.lane_bits, view)?;
        let previous = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let wide = shape.lane_bits.checked_mul(2)?.checked_add(2)?;
        let shift = Expr::konst(u128::from(shape.lane_bits), wide);
        let half = Expr::konst(
            1u128.checked_shl(u32::from(shape.lane_bits.checked_sub(1)?))?,
            wide,
        );
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for lane in 0..shape.lanes {
            let a = Self::extract_lane(first.clone(), shape.lane_bits, lane)?;
            let b = second.lane(shape.lane_bits, lane)?;
            let accumulator = Expr::sign_ext(
                Self::extract_lane(previous.clone(), shape.lane_bits, lane)?,
                wide,
            );
            let doubled = Expr::shl(
                Expr::mul(Expr::sign_ext(a, wide), Expr::sign_ext(b, wide)),
                Expr::konst(1, wide),
            );
            let scaled = Expr::shl(accumulator, shift.clone());
            let combined = if subtract {
                Expr::sub(scaled, doubled)
            } else {
                Expr::add(scaled, doubled)
            };
            let total = Expr::ashr(Expr::add(combined, half.clone()), shift.clone());
            lanes.push(clamp_to_element(total, wide, shape.lane_bits, true)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The second source of a doubling multiply, read once.
    fn doubling_multiplier(
        &mut self,
        insn: &Instruction,
        source: SecondSource,
        source_bits: u16,
        view: u16,
    ) -> Option<Multiplier> {
        let operand = insn.operands.get(2)?.clone();
        Some(match source {
            SecondSource::Element(position) => {
                Multiplier::Element(self.read_simd_lane_bits(&operand, source_bits, position)?)
            }
            SecondSource::Lanewise => {
                Multiplier::Lanewise(self.simd_operand_value(&operand, view)?)
            }
        })
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
        source: SecondSource,
    ) -> Option<Expr> {
        let wide_view = shape.view_bits()?;
        let narrow_bits = shape.lane_bits.checked_div(2)?;
        let narrow_view = narrow_bits.checked_mul(shape.lanes)?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), narrow_view)?;
        let second = self.doubling_multiplier(insn, source, narrow_bits, narrow_view)?;
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
            let b = second.lane(narrow_bits, index)?;
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
