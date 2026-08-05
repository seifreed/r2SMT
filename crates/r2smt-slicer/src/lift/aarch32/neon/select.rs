//! The `AArch32` NEON forms that combine a source with the destination
//! under a bit mask.
//!
//! Five mnemonics, one idea: some bits of the result come from a source
//! register and the rest are the destination's own, so every one of them
//! reads what it writes. `vbsl` / `vbit` / `vbif` take the mask from a
//! register; `vsli` / `vsri` derive it from a shift amount, which is the
//! only reason they carry an element width at all — the insertion itself
//! is bit-granular.
//!
//! Both spellings are invisible to [`super::super::neon_packed_op`], and
//! for different reasons. The selects carry no data type, and the bare
//! mnemonic table there lists only the two-source bitwise operations —
//! a select reads three registers. The shift inserts spell their element
//! as a bare width (`vsli.32`), which
//! [`super::super::neon_element_type`] rejects because it reads the
//! first character as a signedness class and a digit names none.
//!
//! Resolution and lowering live together here, as they do for the
//! structured accesses: neither half is shared with another family.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, OperandKind};

use crate::lift::LiftCtx;
use crate::lift::parse_immediate;
use crate::lift::simd::all_ones;

use super::{BITS_PER_BYTE, NeonOp, NeonShape, bare_element_bits, uniform_vector_view};

/// Which register supplies the mask of a bitwise select, and which
/// value each mask polarity picks.
///
/// All three mnemonics compute `(selected & mask) | (other & ~mask)`
/// and differ only in how the three named registers fill those roles.
/// The destination is one of them in every case, which is what makes
/// these read what they write.
#[derive(Debug, Clone, Copy)]
pub(in crate::lift::aarch32) enum SelectRole {
    /// `vbsl Dd, Dn, Dm` — the destination is the mask, the two sources
    /// the candidates.
    DestinationIsMask,
    /// `vbit Dd, Dn, Dm` — insert `Dn` where `Dm`'s bits are **set**,
    /// keeping the destination elsewhere.
    InsertWhereSet,
    /// `vbif Dd, Dn, Dm` — insert `Dn` where `Dm`'s bits are **clear**.
    InsertWhereClear,
}

/// `vbsl` / `vbit` / `vbif` — the three-register bitwise selects.
///
/// The operation is bit-granular, so the element width carries no
/// meaning beyond the view's total size and the architecture gives the
/// mnemonics no data type to spell one with. The lane geometry recorded
/// here is therefore the byte, purely so [`NeonShape::view_bits`] still
/// names the register the lowering covers.
pub(super) fn bitwise_select_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let role = match mnemonic {
        "vbsl" => SelectRole::DestinationIsMask,
        "vbit" => SelectRole::InsertWhereSet,
        "vbif" => SelectRole::InsertWhereClear,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    Some(NeonShape {
        op: NeonOp::BitwiseSelect(role),
        lane_bits: BITS_PER_BYTE,
        lanes: view.checked_div(BITS_PER_BYTE)?,
    })
}

/// `vsli` / `vsri` — shift each source element and insert it over the
/// destination, keeping the destination bits the shift vacated.
///
/// The two ranges differ by one at each end and that is the encoding
/// rather than a convention: `VSLI` shifts by `0 .. esize-1`, since a
/// left shift by the element width would insert nothing, and `VSRI` by
/// `1 .. esize`, where the top of the range keeps the destination
/// entire. Accepting the other's range would lower a shift the assembler
/// cannot spell.
pub(super) fn shift_insert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let left = match base {
        "vsli" => true,
        "vsri" => false,
        _ => return None,
    };
    let lane_bits = bare_element_bits(ty)?;
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    let amount = insn.operands.get(2)?;
    if amount.kind != OperandKind::Immediate {
        return None;
    }
    let shift = u16::try_from(parse_immediate(&amount.raw)?).ok()?;
    let in_range = if left {
        shift < lane_bits
    } else {
        (1..=lane_bits).contains(&shift)
    };
    if !in_range || view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::ShiftInsert { left, shift },
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}

/// The destination bits a shift insert keeps: the `shift` low bits of
/// the element for a left shift, its `shift` high bits for a right one.
///
/// Both degenerate ends are reachable, one from each mnemonic — `vsli`
/// by zero keeps nothing, `vsri` by the element width keeps everything —
/// and each needs its own arm because a zero-width operand is not a
/// concatenation this IR can build.
fn kept_bits_mask(left: bool, shift: u16, bits: u16) -> Option<Expr> {
    let rest = bits.checked_sub(shift)?;
    if shift == 0 {
        return Some(Expr::konst(0, bits));
    }
    let ones = all_ones(shift);
    if rest == 0 {
        return Some(ones);
    }
    let zeros = Expr::konst(0, rest);
    Some(if left {
        Expr::concat(zeros, ones)
    } else {
        Expr::concat(ones, zeros)
    })
}

impl LiftCtx {
    /// `vbsl` / `vbit` / `vbif` — one bitwise select over the whole
    /// view.
    ///
    /// Emitted once rather than once per lane: the operation is
    /// bit-granular, so splitting it would multiply the formula the
    /// solver sees by the lane count for identical bits.
    pub(super) fn aarch32_bitwise_select(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        role: SelectRole,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let (mask, selected, other) = match role {
            SelectRole::DestinationIsMask => (destination, first, second),
            SelectRole::InsertWhereSet => (second, first, destination),
            SelectRole::InsertWhereClear => (second, destination, first),
        };
        Some(Expr::bv_or(
            Expr::bv_and(selected, mask.clone()),
            Expr::bv_and(other, Expr::bv_xor(mask, all_ones(view))),
        ))
    }

    /// `vsli` / `vsri` — the shifted source over the destination bits
    /// the shift vacated.
    ///
    /// Per lane rather than over the whole view, unlike the selects: a
    /// shift moves bits across the element boundary the instruction
    /// exists to respect.
    pub(super) fn aarch32_shift_insert(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        left: bool,
        shift: u16,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let bits = shape.lane_bits;
        let amount = Expr::konst(u128::from(shift), bits);
        // A left shift keeps the destination's low bits and a right
        // shift its high ones, which is the whole difference between the
        // two mnemonics.
        let kept = kept_bits_mask(left, shift, bits)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let previous = Self::extract_lane(destination.clone(), bits, index)?;
            let value = Self::extract_lane(source.clone(), bits, index)?;
            let shifted = if left {
                Expr::shl(value, amount.clone())
            } else {
                Expr::lshr(value, amount.clone())
            };
            lanes.push(Expr::bv_or(shifted, Expr::bv_and(previous, kept.clone())));
        }
        Self::concat_lanes(lanes)
    }
}
