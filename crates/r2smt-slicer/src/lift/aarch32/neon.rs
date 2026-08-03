//! `AArch32` NEON forms the element-typed mnemonic dispatch cannot
//! reach.
//!
//! [`super::neon_packed_op`] resolves a lane-wise operation from the
//! mnemonic alone, and that works because every family it covers shares
//! one geometry across the whole instruction: the element type says how
//! wide a lane is, the destination register says how many there are,
//! and every source matches. The families here break that assumption in
//! one of three ways — a permutation spells its element as a bare width
//! the element-type parser rejects (`vzip.8`), a widening form relates
//! two different geometries, and the by-element and structured forms
//! carry shape on the operand rather than the mnemonic.
//!
//! Each gets a resolver returning one [`NeonShape`], and [`resolve`] is
//! consulted by the effect table and the per-mnemonic dispatcher alike.
//! That is what keeps them from disagreeing: an instruction the slicer
//! retains because its destination is a definition, but whose lowering
//! is then dropped, would leave a later read bound to a stale value.

use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::registers::{has_vector_arrangement, is_simd_parent, register_layout};

pub(super) mod lower;
mod permute;

/// Bits in a byte, named because `vext` measures its window in them.
pub(super) const BITS_PER_BYTE: u16 = 8;

/// What a resolved `AArch32` NEON instruction computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NeonOp {
    /// `vext` — the window of the two sources concatenated that starts
    /// at this byte offset.
    Extract { byte_offset: u16 },
    /// `vrev16` / `vrev32` / `vrev64` — reverse the element order
    /// inside each container of this many bits.
    Reverse { container_bits: u16 },
    /// `vdup` — replicate a general-purpose register's low element to
    /// every lane.
    Duplicate,
}

/// A resolved `AArch32` NEON instruction: what to compute, and at what
/// geometry.
///
/// Only destination geometry, as on `AArch64`. Every family here reads
/// its sources at the destination's width, and the families that do not
/// — the widening forms and the reductions — will carry the difference
/// on their [`NeonOp`] variant when they land, the way `AArch64` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NeonShape {
    pub(super) op: NeonOp,
    /// Element width of the destination, in bits.
    pub(super) lane_bits: u16,
    /// Number of destination lanes.
    pub(super) lanes: u16,
}

impl NeonShape {
    /// Width of the destination's view — the whole `d` or `q` register
    /// the instruction covers.
    pub(super) const fn view_bits(self) -> Option<u16> {
        self.lane_bits.checked_mul(self.lanes)
    }
}

/// Resolve `insn` into the NEON form it names, or `None` when no family
/// here claims it.
///
/// Ordered the way `AArch64`'s dispatch is: the resolvers that
/// constrain their operands most tightly go first, so a form that
/// matches several mnemonically is claimed by the one whose operand
/// shape it actually fits.
pub(super) fn resolve(insn: &Instruction) -> Option<NeonShape> {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    permute::extract_shape(insn, &mnemonic)
        .or_else(|| permute::reverse_shape(insn, &mnemonic))
        .or_else(|| permute::duplicate_shape(insn, &mnemonic))
}

/// Element width a bare `AArch32` NEON size suffix names — the `8` in
/// `vext.8`, the `32` in `vdup.32`.
///
/// The families that move bits rather than computing on them spell
/// their element as a width alone, with no signedness letter, because
/// there is no arithmetic for a sign to change.
/// [`super::neon_element_type`] rejects exactly these spellings: it
/// reads the first character as the signedness class, and a digit names
/// none.
pub(super) fn bare_element_bits(ty: &str) -> Option<u16> {
    match ty {
        "8" => Some(8),
        "16" => Some(16),
        "32" => Some(32),
        "64" => Some(64),
        _ => None,
    }
}

/// The view width a bare `AArch32` vector register operand names: 64
/// for a `d` register, 128 for a `q`.
///
/// `AArch32` leaves the operand bare, so the register spelling is the
/// only thing saying how much of the register file the instruction
/// covers — the counterpart of the arrangement `AArch64` writes on the
/// operand itself.
///
/// An operand carrying vector shape is refused rather than resolved.
/// [`register_layout`] deliberately maps `d1[1]` onto the *whole* `d1`
/// slice, so accepting it here would read all 64 bits where the
/// instruction names one lane.
pub(super) fn vector_view_bits(op: &Operand) -> Option<u16> {
    if op.kind != OperandKind::Register || has_vector_arrangement(&op.raw, Arch::Arm) {
        return None;
    }
    let layout = register_layout(op.raw.trim(), Arch::Arm)?;
    is_simd_parent(layout.parent, Arch::Arm).then(|| layout.width())
}

/// The view every operand in `positions` names, or `None` unless they
/// all name the same one.
///
/// The uniformity check is the `AArch32` counterpart of `AArch64`'s
/// "every operand yields the same arrangement": there, a mismatched
/// `.8b` against a `.16b` is the decline; here it is a `d` against a
/// `q`.
pub(super) fn uniform_vector_view(insn: &Instruction, positions: usize) -> Option<u16> {
    let view = vector_view_bits(insn.operands.first()?)?;
    for index in 1..positions {
        if vector_view_bits(insn.operands.get(index)?)? != view {
            return None;
        }
    }
    Some(view)
}

/// Whether `op` names a general-purpose register — the source `vdup`
/// broadcasts, as opposed to the vector element form.
pub(super) fn is_general_register(op: &Operand) -> bool {
    op.kind == OperandKind::Register
        && !has_vector_arrangement(&op.raw, Arch::Arm)
        && register_layout(op.raw.trim(), Arch::Arm)
            .is_some_and(|layout| !is_simd_parent(layout.parent, Arch::Arm))
}
