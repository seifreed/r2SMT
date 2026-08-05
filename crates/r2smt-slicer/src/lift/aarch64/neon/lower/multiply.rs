//! Lowering for the `AArch64` NEON families built on a lane product.
//!
//! Multiply-accumulate, the by-element forms, the dot products, the
//! polynomial multiply, and the fused multiply steps. The plain lane-wise
//! `mul` is not here; it is one member of the lane-wise family the
//! neutral packed core lowers.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use super::extend;
use crate::lift::aarch64::neon::NeonShape;
use crate::lift::aarch64::neon::geometry::{BITS_PER_BYTE, dot_product_element};
use crate::lift::aarch64::neon::multiply::{
    AccumulateKind, AccumulateSources, ByElementKind, DOT_PRODUCT_TERMS,
};
use crate::lift::{FpArithOp, FusedStep, LiftCtx, fp_lane_result, fused_multiply_lane};

impl LiftCtx {
    /// `fmla` / `fmls` / `frecps` / `frsqrts` — each binary32 lane
    /// computed as a single fused multiply step. `fmla` / `fmls` read the
    /// destination lane as the accumulator; the reciprocal steps read
    /// only their two sources.
    pub(super) fn fused_step_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        step: FusedStep,
    ) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let a = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let b = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let accumulator = if step.reads_accumulator() {
            Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?)
        } else {
            None
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a_lane = Self::extract_lane(a.clone(), shape.lane_bits, index)?;
            let b_lane = Self::extract_lane(b.clone(), shape.lane_bits, index)?;
            let acc_lane = match &accumulator {
                Some(acc) => Some(Self::extract_lane(acc.clone(), shape.lane_bits, index)?),
                None => None,
            };
            lanes.push(fused_multiply_lane(
                step,
                &a_lane,
                &b_lane,
                acc_lane,
                shape.lane_bits,
            )?);
        }
        Self::concat_lanes(lanes)
    }

    /// `pmull` / `pmull2` — the carry-less product of two narrow
    /// elements, which is the `XOR` of the multiplicand shifted to
    /// every set bit of the multiplier.
    ///
    /// Carry-less is the whole point: an ordinary multiply is the same
    /// sum of shifted copies but with carries propagating between them,
    /// so `Expr::mul` is not an approximation of this, it is a
    /// different function. The product of two `n`-bit elements needs
    /// `2n - 1` bits, so the destination element always holds it.
    pub(super) fn polynomial_multiply_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        upper: bool,
    ) -> Option<Expr> {
        let narrow = shape.lane_bits / 2;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let multiplicand = Expr::zero_ext(
                LiftCtx::extract_lane(first.clone(), narrow, source_lane)?,
                shape.lane_bits,
            );
            let multiplier = LiftCtx::extract_lane(second.clone(), narrow, source_lane)?;
            let mut product = Expr::konst(0, shape.lane_bits);
            for bit in 0..narrow {
                let shifted = Expr::shl(
                    multiplicand.clone(),
                    Expr::konst(u128::from(bit), shape.lane_bits),
                );
                product = Expr::bv_xor(
                    product,
                    Expr::Ite {
                        cond: Box::new(Expr::eq(
                            Expr::extract(multiplier.clone(), bit, bit),
                            Expr::konst(1, 1),
                        )),
                        then_expr: Box::new(shifted),
                        else_expr: Box::new(Expr::konst(0, shape.lane_bits)),
                    },
                );
            }
            lanes.push(product);
        }
        Self::concat_lanes(lanes)
    }

    /// The by-element multiplies — one source element multiplied into
    /// every destination lane.
    ///
    /// The element is read once, outside the loop. Reading it per lane
    /// would emit the same extract `lanes` times over, and for a memory
    /// operand it would be a load per lane.
    pub(super) fn by_element_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ByElementKind,
        upper: bool,
    ) -> Option<Expr> {
        let source_bits = if kind.widens() {
            shape.lane_bits / 2
        } else {
            shape.lane_bits
        };
        let first = self.widen_source(insn, 1)?;
        let element = self.read_simd_lane_bits(
            &insn.operands.get(2)?.clone(),
            source_bits,
            shape.source_index,
        )?;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = if kind.combines() {
            Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?)
        } else {
            None
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            // A `2` form reads the first source's upper half; the
            // element operand is indexed absolutely and is never halved.
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let a = LiftCtx::extract_lane(first.clone(), source_bits, source_lane)?;
            let product = match kind {
                ByElementKind::Long { signed, .. } => Expr::mul(
                    extend(a, shape.lane_bits, signed),
                    extend(element.clone(), shape.lane_bits, signed),
                ),
                ByElementKind::Integer { .. } => Expr::mul(a, element.clone()),
                ByElementKind::Float => {
                    fp_lane_result(FpArithOp::Mul, a, element.clone(), shape.lane_bits)?
                }
                // The accumulate happens inside the single rounding, so
                // this arm consumes the destination lane itself rather
                // than leaving it to the combine below.
                ByElementKind::Fused { step } => {
                    let previous = LiftCtx::extract_lane(
                        accumulator.as_ref()?.clone(),
                        shape.lane_bits,
                        index,
                    )?;
                    crate::lift::simd::fused_multiply_lane(
                        step,
                        &a,
                        &element,
                        Some(previous),
                        shape.lane_bits,
                    )?
                }
            };
            lanes.push(match (kind.combine(), accumulator.as_ref()) {
                (Some(combine), Some(previous)) => combine.apply(
                    LiftCtx::extract_lane(previous.clone(), shape.lane_bits, index)?,
                    product,
                ),
                _ => product,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// `sdot` / `udot` — four byte products summed into each 32-bit
    /// destination lane, on top of that lane's prior value.
    ///
    /// The products are formed at the destination's width rather than
    /// the byte's, which is what the architecture's extension of each
    /// element before the multiply comes to; the accumulation then wraps
    /// at 32 bits exactly as ARM defines it.
    pub(super) fn dot_product_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
        by_element: bool,
    ) -> Option<Expr> {
        // Each source register is a full 128-bit view; a by-element
        // source re-spells its group selector to the bare register.
        const REGISTER_BITS: u16 = 128;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.widen_source(insn, 1)?;
        let second = if by_element {
            let (register, _) = dot_product_element(insn.operands.get(2)?)?;
            self.simd_operand_value(&register, REGISTER_BITS)?
        } else {
            self.widen_source(insn, 2)?
        };
        let read = |value: &Expr, byte: u16| -> Option<Expr> {
            let raw = LiftCtx::extract_lane(value.clone(), BITS_PER_BYTE, byte)?;
            Some(extend(raw, shape.lane_bits, signed))
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let mut sum = LiftCtx::extract_lane(accumulator.clone(), shape.lane_bits, index)?;
            for term in 0..DOT_PRODUCT_TERMS {
                let first_byte = index.checked_mul(DOT_PRODUCT_TERMS)?.checked_add(term)?;
                // By-element broadcasts one group to every lane; the
                // whole-vector form pairs each lane with its own.
                let second_group = if by_element {
                    shape.source_index
                } else {
                    index
                };
                let second_byte = second_group
                    .checked_mul(DOT_PRODUCT_TERMS)?
                    .checked_add(term)?;
                sum = Expr::add(
                    sum,
                    Expr::mul(read(&first, first_byte)?, read(&second, second_byte)?),
                );
            }
            lanes.push(sum);
        }
        Self::concat_lanes(lanes)
    }

    /// The multiply-accumulate family: each destination lane is its own
    /// prior value plus or minus the product of the two source lanes.
    pub(super) fn multiply_accumulate_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: AccumulateKind,
    ) -> Option<Expr> {
        let AccumulateKind {
            combine,
            sources,
            upper,
        } = kind;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let source_bits = match sources {
            AccumulateSources::SameWidth => shape.lane_bits,
            AccumulateSources::Long { .. } => shape.lane_bits / 2,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let read = |value: &Expr| -> Option<Expr> {
                let element = LiftCtx::extract_lane(value.clone(), source_bits, source_lane)?;
                Some(match sources {
                    AccumulateSources::SameWidth => element,
                    AccumulateSources::Long { signed: true } => {
                        Expr::sign_ext(element, shape.lane_bits)
                    }
                    AccumulateSources::Long { signed: false } => {
                        Expr::zero_ext(element, shape.lane_bits)
                    }
                })
            };
            let product = Expr::mul(read(&first)?, read(&second)?);
            let previous = LiftCtx::extract_lane(accumulator.clone(), shape.lane_bits, index)?;
            lanes.push(combine.apply(previous, product));
        }
        Self::concat_lanes(lanes)
    }
}
