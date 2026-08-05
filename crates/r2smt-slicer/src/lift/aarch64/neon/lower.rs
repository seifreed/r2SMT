//! Lowering for the `AArch64` NEON families.
//!
//! The sibling module answers *what* an instruction is; this one builds
//! the expression. Splitting them keeps each file to one question, and
//! is what the resolver's promise depends on: a shape that resolves must
//! have a lowering here, or the effect table and the lifter disagree
//! about which instructions the slicer may retain.
//!
//! The bodies are grouped by the same taxonomy the resolvers use, in
//! sibling modules: [`arith`] for the operations on whole lanes,
//! [`multiply`] for those built on a lane product, [`width`] for those
//! relating two element geometries, and [`permute`] for those that only
//! move bits. What stays here is the dispatch and the handful of
//! primitives more than one family needs.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use super::geometry::operand_arrangement;
use super::{NeonOp, NeonShape};
use crate::lift::LiftCtx;
use permute::immediate_lanes;

pub(in crate::lift) use width::{convert_lane, high_narrow_lane};

impl LiftCtx {
    /// Lower a resolved NEON instruction.
    ///
    /// A decline here is sound by the free-input boundary: the slicer
    /// consumed the destination as a definition and therefore stopped
    /// tracking its upstream definitions, so emitting no assignment
    /// leaves the register a free SSA input rather than bound to a stale
    /// value.
    pub(in crate::lift::aarch64) fn lift_neon(&mut self, insn: &Instruction, shape: NeonShape) {
        if let NeonOp::Packed(op) = shape.op {
            // An `AArch64` SIMD write has no merging form, so the write
            // zeroes the register above the arrangement's view.
            self.lift_packed_vector(insn, op, shape.lane_bits, true);
            return;
        }
        let Some(destination) = insn.operands.first().cloned() else {
            self.push_neon_unsupported(insn);
            return;
        };
        let Some(value) = self.neon_value(insn, shape) else {
            self.push_neon_unsupported(insn);
            return;
        };
        let written = if shape.writes_gpr() {
            self.write_register_to(&destination, value)
        } else if matches!(shape.op, NeonOp::Insert { .. }) {
            self.write_simd_lane(&destination, value, shape.lane_bits, shape.dest_index)
        } else {
            self.write_simd_dst(&destination, value, true)
        };
        if !written {
            self.push_neon_unsupported(insn);
        }
    }

    /// Build the value a resolved NEON instruction writes.
    fn neon_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        match shape.op {
            // Handled by the caller; never reaches here.
            NeonOp::Packed(_) => None,
            NeonOp::Immediate { value, invert } => Some(immediate_lanes(&shape, value, invert)),
            NeonOp::Duplicate { from_element } => self.duplicate_lanes(insn, shape, from_element),
            NeonOp::Extract { byte_offset } => self.extract_window(insn, shape, byte_offset),
            NeonOp::Permute(kind) => self.permute_lanes(insn, shape, kind),
            NeonOp::ElementToGpr { signed } => self.element_to_gpr(insn, shape, signed),
            NeonOp::Insert { from_element } => self.insert_source(insn, shape, from_element),
            NeonOp::Widen {
                kind,
                signed,
                upper,
            } => self.widen_lanes(insn, shape, kind, signed, upper),
            NeonOp::MultiplyAccumulate(kind) => self.multiply_accumulate_lanes(insn, shape, kind),
            NeonOp::Saturating {
                kind,
                signed_sources,
                upper,
            } => self.saturating_lanes(insn, shape, kind, signed_sources, upper),
            NeonOp::Shift { kind, signed } => self.shift_lanes(insn, shape, kind, signed),
            NeonOp::ShiftInsert { left, shift } => {
                self.shift_insert_lanes(insn, shape, left, shift)
            }
            NeonOp::MixedSignAdd { destination_signed } => {
                self.mixed_sign_add_lanes(insn, shape, destination_signed)
            }
            NeonOp::DoublingLong { combine, upper } => {
                self.doubling_long_lanes(insn, shape, combine, upper)
            }
            NeonOp::Compare { kind, zero } => self.compare_lanes(insn, shape, kind, zero),
            NeonOp::Convert { kind, upper, fbits } => {
                self.convert_lanes(insn, shape, kind, upper, fbits)
            }
            NeonOp::BitwiseSelect(role) => self.bitwise_select(insn, shape, role),
            NeonOp::Reduce {
                kind,
                source_lanes,
                source_lane_bits,
            } => self.reduce_lanes(insn, shape, kind, source_lanes, source_lane_bits),
            NeonOp::ByElement { kind, upper } => self.by_element_lanes(insn, shape, kind, upper),
            NeonOp::DotProduct { signed, by_element } => {
                self.dot_product_lanes(insn, shape, signed, by_element)
            }
            NeonOp::TableLookup { keep, table_lanes } => {
                self.table_lookup_lanes(insn, shape, keep, table_lanes)
            }
            NeonOp::PolynomialMultiply { upper } => {
                self.polynomial_multiply_lanes(insn, shape, upper)
            }
            NeonOp::FusedStep(step) => self.fused_step_lanes(insn, shape, step),
            NeonOp::BitwiseUnary(kind) => self.bitwise_unary_lanes(insn, shape, kind),
            NeonOp::LaneCombine(op) => self.lane_combine_lanes(insn, shape, op),
            NeonOp::Pairwise(op) => self.pairwise_lanes(insn, shape, op),
            NeonOp::AbsoluteDifference(kind) => self.absolute_difference_lanes(insn, shape, kind),
            NeonOp::PairwiseLong { signed, accumulate } => {
                self.pairwise_long_lanes(insn, shape, signed, accumulate)
            }
            NeonOp::HighNarrow {
                subtract,
                rounding,
                upper,
            } => self.high_narrow_lanes(insn, shape, subtract, rounding, upper),
            NeonOp::Estimate => self.estimate_value(insn, shape),
        }
    }

    /// One source operand, materialised once at its own view width.
    fn widen_source(&mut self, insn: &Instruction, position: usize) -> Option<Expr> {
        let operand = insn.operands.get(position)?.clone();
        let arrangement = operand_arrangement(&operand)?;
        self.simd_operand_value(&operand, arrangement.view_bits())
    }

    /// The destination register read as an input, at the view the shape
    /// describes — what an accumulating form's prior value is.
    fn destination_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        self.simd_operand_value(&insn.operands.first()?.clone(), view)
    }

    fn push_neon_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(r2smt_ir::stmt::IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable NEON operand at {addr}", addr = insn.address),
        });
    }
}

/// The integer ordered select: the larger of the two when `max`, the
/// smaller otherwise.
///
/// Written as one comparison with the operands swapped rather than as
/// four predicates, so the two directions cannot drift apart.
fn integer_min_max(a: Expr, b: Expr, signed: bool, max: bool) -> Expr {
    let cond = if signed {
        Expr::slt(a.clone(), b.clone())
    } else {
        Expr::ult(a.clone(), b.clone())
    };
    let (taken, other) = if max { (b, a) } else { (a, b) };
    Expr::Ite {
        cond: Box::new(cond),
        then_expr: Box::new(taken),
        else_expr: Box::new(other),
    }
}

/// Extend `value` from `from` bits to `to` bits.
fn extend(value: Expr, to: u16, signed: bool) -> Expr {
    if signed {
        Expr::sign_ext(value, to)
    } else {
        Expr::zero_ext(value, to)
    }
}

/// The largest value of a `bits`-wide unsigned range, or `None` when it
/// does not fit the constant type.
///
/// A full 128-bit vector view is representable and is the width the
/// bitwise selects mask at, so only *wider* than the constant type
/// declines.
fn unsigned_max(bits: u16) -> Option<u128> {
    match bits {
        0 => None,
        128 => Some(u128::MAX),
        bits if bits < 128 => Some((1u128 << bits) - 1),
        _ => None,
    }
}

mod arith;
mod multiply;
mod permute;
mod width;
