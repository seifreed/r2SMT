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

use super::{BITS_PER_BYTE, NeonOp, NeonShape};

impl LiftCtx {
    /// Lower a resolved `AArch32` NEON instruction.
    pub(in crate::lift::aarch32) fn lift_aarch32_neon(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
    ) {
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
        }
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

    fn push_aarch32_neon_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!(
                "unmodellable NEON operand at {addr} (aarch32)",
                addr = insn.address
            ),
        });
    }
}
