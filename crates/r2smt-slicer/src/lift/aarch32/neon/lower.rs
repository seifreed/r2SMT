//! The expression each resolved `AArch32` NEON shape builds.
//!
//! Every lowering here is written over primitives the IR already had —
//! lane extraction, concatenation and a bit-slice — so none of these
//! families needed a new `Expr` variant. A shape that cannot be
//! lowered pushes [`IrStmt::Unsupported`] rather than a guess, which
//! truncates the slice at the sound free-input boundary.

use std::cmp::Ordering;

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;
use r2smt_ir::stmt::IrStmt;

use crate::lift::LiftCtx;

use super::element::ByElementKind;
use super::permute::{PairKind, PairSource, paired_source};
use super::width::WidenKind;
use super::{BITS_PER_BYTE, NeonOp, NeonShape};
use crate::lift::BinOp;
use crate::lift::{FpArithOp, fp_lane_result};

/// Widen `value` to `target` bits, replicating the sign bit or not.
fn extend(value: Expr, target: u16, signed: bool) -> Expr {
    if signed {
        Expr::sign_ext(value, target)
    } else {
        Expr::zero_ext(value, target)
    }
}

impl LiftCtx {
    /// Lower a resolved `AArch32` NEON instruction.
    pub(in crate::lift::aarch32) fn lift_aarch32_neon(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
    ) {
        if let NeonOp::PermutePair(kind) = shape.op {
            self.lift_aarch32_permute_pair(insn, shape, kind);
            return;
        }
        let Some(destination) = insn.operands.first().cloned() else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        let Some(value) = self.aarch32_neon_value(insn, shape) else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        // An `AArch32` NEON write merges: the parent bits outside the
        // destination's view survive, so `d1` lives through a write to
        // `d0`. `AArch64` zeroes them instead.
        if !self.write_simd_dst(&destination, value, false) {
            self.push_aarch32_neon_unsupported(insn);
        }
    }

    fn aarch32_neon_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        match shape.op {
            NeonOp::Extract { byte_offset } => {
                self.aarch32_extract_window(insn, shape, byte_offset)
            }
            NeonOp::Reverse { container_bits } => {
                self.aarch32_reverse_lanes(insn, shape, container_bits)
            }
            NeonOp::Duplicate => self.aarch32_duplicate_lanes(insn, shape),
            NeonOp::Widen { kind, signed } => match kind {
                WidenKind::Narrow => self.aarch32_narrow_lanes(insn, shape, 0),
                WidenKind::ShiftNarrow { shift } => self.aarch32_narrow_lanes(insn, shape, shift),
                WidenKind::SaturatingNarrow {
                    source_signed,
                    to_unsigned,
                } => self.aarch32_saturating_narrow_lanes(insn, shape, source_signed, to_unsigned),
                WidenKind::Long => self.aarch32_long_lanes(insn, shape, None, false, signed),
                WidenKind::LongArith { op, wide_first } => {
                    self.aarch32_long_lanes(insn, shape, Some(op), wide_first, signed)
                }
            },
            NeonOp::ByElement { kind, index } => {
                self.aarch32_by_element_lanes(insn, shape, kind, index)
            }
            NeonOp::TableLookup { keep, table_lanes } => {
                self.aarch32_table_lookup_lanes(insn, shape, keep, table_lanes)
            }
            NeonOp::Compare { kind, zero } => self.aarch32_compare_lanes(insn, shape, kind, zero),
            // Handled by the caller; it writes two destinations and so
            // does not fit the one-value contract here.
            NeonOp::PermutePair(_) => None,
        }
    }

    /// The by-element forms — one element of the second source
    /// multiplied into every destination lane.
    ///
    /// The element is read **once**, outside the loop, rather than
    /// re-extracted per lane: it is the same value every time, and on a
    /// memory operand the difference would be one load against one per
    /// lane.
    fn aarch32_by_element_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ByElementKind,
        index: u16,
    ) -> Option<Expr> {
        let (combine, signed, source_bits) = match kind {
            ByElementKind::Same { combine } => (combine, false, shape.lane_bits),
            ByElementKind::Long { combine, signed } => {
                (combine, signed, shape.lane_bits.checked_div(2)?)
            }
            ByElementKind::Float { accumulate } => {
                return self.aarch32_by_element_float_lanes(insn, shape, accumulate, index);
            }
        };
        let widens = source_bits != shape.lane_bits;
        let lanewise_view = source_bits.checked_mul(shape.lanes)?;
        let lanewise = self.simd_operand_value(&insn.operands.get(1)?.clone(), lanewise_view)?;
        let indexed = insn.operands.get(2)?.clone();
        let element = self.read_simd_lane_bits(&indexed, source_bits, index)?;
        let element = if widens {
            extend(element, shape.lane_bits, signed)
        } else {
            element
        };
        let accumulator = match combine {
            Some(_) => {
                Some(self.simd_operand_value(&insn.operands.first()?.clone(), shape.view_bits()?)?)
            }
            None => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for lane in 0..shape.lanes {
            let a = Self::extract_lane(lanewise.clone(), source_bits, lane)?;
            let a = if widens {
                extend(a, shape.lane_bits, signed)
            } else {
                a
            };
            let product = Expr::mul(a, element.clone());
            lanes.push(match (combine, accumulator.as_ref()) {
                (None, _) => product,
                (Some(op), Some(previous)) => op.apply(
                    Self::extract_lane(previous.clone(), shape.lane_bits, lane)?,
                    product,
                ),
                (Some(_), None) => return None,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// The float by-element forms — one lane of the second source
    /// multiplied into every destination lane, rounded once.
    ///
    /// `vmla` / `vmls` are not fused: the product rounds, then the
    /// destination-lane accumulate rounds again, which is what the
    /// architecture defines for the NEON (non-`vfma`) spelling. So there
    /// is nothing to widen — every step is at the lane width.
    fn aarch32_by_element_float_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        accumulate: Option<FpArithOp>,
        index: u16,
    ) -> Option<Expr> {
        let lane_bits = shape.lane_bits;
        let lanewise_view = lane_bits.checked_mul(shape.lanes)?;
        let lanewise = self.simd_operand_value(&insn.operands.get(1)?.clone(), lanewise_view)?;
        let indexed = insn.operands.get(2)?.clone();
        let element = self.read_simd_lane_bits(&indexed, lane_bits, index)?;
        let accumulator = match accumulate {
            Some(_) => {
                Some(self.simd_operand_value(&insn.operands.first()?.clone(), shape.view_bits()?)?)
            }
            None => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for lane in 0..shape.lanes {
            let a = Self::extract_lane(lanewise.clone(), lane_bits, lane)?;
            let product = fp_lane_result(FpArithOp::Mul, a, element.clone(), lane_bits)?;
            lanes.push(match (accumulate, accumulator.as_ref()) {
                (None, _) => product,
                (Some(op), Some(previous)) => {
                    let acc = Self::extract_lane(previous.clone(), lane_bits, lane)?;
                    fp_lane_result(op, acc, product, lane_bits)?
                }
                (Some(_), None) => return None,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// The long forms — each destination element is computed at twice
    /// the width of the source elements feeding it, which is exactly
    /// what stops a product or a sum overflowing.
    ///
    /// `wide_first` marks `vaddw` / `vsubw`, whose first source is
    /// already at the destination's width and so is read rather than
    /// extended.
    fn aarch32_long_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        op: Option<BinOp>,
        wide_first: bool,
        signed: bool,
    ) -> Option<Expr> {
        let wide_view = shape.view_bits()?;
        let narrow_bits = shape.lane_bits.checked_div(2)?;
        let narrow_view = narrow_bits.checked_mul(shape.lanes)?;
        let first_view = if wide_first { wide_view } else { narrow_view };
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), first_view)?;
        let second = match insn.operands.get(2) {
            Some(operand) => Some(self.simd_operand_value(&operand.clone(), narrow_view)?),
            None => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = if wide_first {
                Self::extract_lane(first.clone(), shape.lane_bits, index)?
            } else {
                extend(
                    Self::extract_lane(first.clone(), narrow_bits, index)?,
                    shape.lane_bits,
                    signed,
                )
            };
            lanes.push(match (op, second.as_ref()) {
                (None, _) => a,
                (Some(op), Some(second)) => {
                    let b = extend(
                        Self::extract_lane(second.clone(), narrow_bits, index)?,
                        shape.lane_bits,
                        signed,
                    );
                    op.apply(a, b)
                }
                (Some(_), None) => return None,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// The narrowing forms — each source element is shifted, then cut
    /// down to the destination's width.
    ///
    /// The shift is logical rather than arithmetic, and that is exact
    /// rather than approximate: the encoding bounds the amount by the
    /// destination element's width, so no bit shifted in from the top
    /// ever reaches the half that is kept.
    fn aarch32_narrow_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        shift: u16,
    ) -> Option<Expr> {
        let source_bits = shape.lane_bits.checked_mul(2)?;
        let source_view = source_bits.checked_mul(shape.lanes)?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), source_view)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let mut element = Self::extract_lane(source.clone(), source_bits, index)?;
            if shift > 0 {
                element = Expr::lshr(element, Expr::konst(u128::from(shift), source_bits));
            }
            lanes.push(Expr::extract(element, shape.lane_bits.checked_sub(1)?, 0));
        }
        Self::concat_lanes(lanes)
    }

    /// `vqmovn` / `vqmovun` — each source element clamped into the
    /// destination range instead of truncated.
    ///
    /// The clamp compares at the source width, where the source value is
    /// already exact, so no headroom bit is needed the way an addition
    /// would need one. The comparison signedness follows the *source*
    /// (`vqmovun` reads a signed source); the range follows the
    /// *destination* (`vqmovun` clamps into `[0, 2^n - 1]`, so a negative
    /// source lands at zero).
    fn aarch32_saturating_narrow_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        source_signed: bool,
        to_unsigned: bool,
    ) -> Option<Expr> {
        let source_bits = shape.lane_bits.checked_mul(2)?;
        let source_view = source_bits.checked_mul(shape.lanes)?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), source_view)?;
        // The destination range is signed only for `vqmovn.s`: `vqmovn.u`
        // and `vqmovun` both clamp into the unsigned range.
        let dest_signed = source_signed && !to_unsigned;
        let (min, max) =
            crate::lift::simd::element_bounds(dest_signed, shape.lane_bits, source_bits)?;
        let le = |left: Expr, right: Expr| {
            if source_signed {
                Expr::sle(left, right)
            } else {
                Expr::ule(left, right)
            }
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let element = Self::extract_lane(source.clone(), source_bits, index)?;
            let below_max = Expr::Ite {
                cond: Box::new(le(element.clone(), max.clone())),
                then_expr: Box::new(element),
                else_expr: Box::new(max.clone()),
            };
            let clamped = Expr::Ite {
                cond: Box::new(le(min.clone(), below_max.clone())),
                then_expr: Box::new(below_max),
                else_expr: Box::new(min.clone()),
            };
            lanes.push(Expr::extract(clamped, shape.lane_bits.checked_sub(1)?, 0));
        }
        Self::concat_lanes(lanes)
    }

    /// `vzip` / `vuzp` / `vtrn` — one permutation, two destinations.
    ///
    /// The second result is stashed in a temp **before** the first
    /// destination is written, and that is load-bearing rather than
    /// tidy. Both results read both operands, and SSA renames by
    /// statement position, not by when the expression was built: an
    /// assignment to the second register placed after the first would
    /// have its reference to the first register rewritten to the
    /// freshly permuted value. The temp pins the read to the original.
    ///
    /// Same hazard, and same remedy, as the flag-ordering invariant on
    /// the integer side, where a flag-setting instruction whose
    /// destination overlaps a source stashes the delta before the write.
    fn lift_aarch32_permute_pair(&mut self, insn: &Instruction, shape: NeonShape, kind: PairKind) {
        let (Some(first_operand), Some(second_operand)) = (
            insn.operands.first().cloned(),
            insn.operands.get(1).cloned(),
        ) else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        let Some((into_first, into_second)) = self.aarch32_paired_values(insn, shape, kind) else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        let Some(view) = shape.view_bits() else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        let stashed = self.new_temp(insn.address, view);
        self.assign(stashed.clone(), into_second);
        // Only the *value* has to predate the first write. The splice
        // inside the second write reads the parent as the first
        // assignment left it, which is what keeps `vzip.8 d0, d1` — two
        // halves of one register — from clobbering the half just
        // written.
        if !self.write_simd_dst(&first_operand, into_first, false)
            || !self.write_simd_dst(&second_operand, Expr::Var(stashed), false)
        {
            self.push_aarch32_neon_unsupported(insn);
        }
    }

    /// The two register values a paired permutation produces.
    fn aarch32_paired_values(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: PairKind,
    ) -> Option<(Expr, Expr)> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let second = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let build = |into_second: bool| {
            let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
            for index in 0..shape.lanes {
                let (which, source_lane) = paired_source(kind, index, into_second, shape.lanes)?;
                let value = match which {
                    PairSource::First => first.clone(),
                    PairSource::Second => second.clone(),
                };
                lanes.push(Self::extract_lane(value, shape.lane_bits, source_lane)?);
            }
            Self::concat_lanes(lanes)
        };
        Some((build(false)?, build(true)?))
    }

    /// `vext` — the two sources laid end to end, read from a byte
    /// offset.
    ///
    /// The second source is the *high* half of the concatenation, so a
    /// window starting inside the first source runs off its top into
    /// the second, which is what the architecture defines.
    fn aarch32_extract_window(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        byte_offset: u16,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let lo = byte_offset.checked_mul(BITS_PER_BYTE)?;
        let hi = lo.checked_add(view)?.checked_sub(1)?;
        Some(Expr::extract(Expr::concat(second, first), hi, lo))
    }

    /// `vrev` — each container's elements in the opposite order, the
    /// containers themselves staying put.
    fn aarch32_reverse_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        container_bits: u16,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let per_container = container_bits.checked_div(shape.lane_bits)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let base = index
                .checked_div(per_container)?
                .checked_mul(per_container)?;
            let within = index % per_container;
            let source_lane = base.checked_add(per_container.checked_sub(1)? - within)?;
            lanes.push(Self::extract_lane(
                source.clone(),
                shape.lane_bits,
                source_lane,
            )?);
        }
        Self::concat_lanes(lanes)
    }

    /// `vdup` — one general-purpose register's low element in every
    /// lane.
    fn aarch32_duplicate_lanes(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        let whole = self.read_register(&source)?;
        let element = match self.operand_width(&source).cmp(&shape.lane_bits) {
            Ordering::Equal => whole,
            Ordering::Greater => Expr::extract(whole, shape.lane_bits - 1, 0),
            Ordering::Less => return None,
        };
        Self::concat_lanes(vec![element; usize::from(shape.lanes)])
    }

    pub(in crate::lift::aarch32::neon) fn push_aarch32_neon_unsupported(
        &mut self,
        insn: &Instruction,
    ) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!(
                "unmodellable NEON operand at {addr} (aarch32)",
                addr = insn.address
            ),
        });
    }
}
