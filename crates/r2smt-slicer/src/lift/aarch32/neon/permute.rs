//! The `AArch32` NEON families that move bits without computing on
//! them, and whose element is spelled as a bare width.
//!
//! `vzip` / `vuzp` / `vtrn` are deliberately absent: on `AArch32` they
//! write *both* named registers, unlike `AArch64`'s single-destination
//! `zip1` / `zip2`, so they need an effect entry of their own rather
//! than the shared one every family here uses.

use r2smt_ir::program::{Instruction, OperandKind};

use crate::lift::parse_immediate;

use super::{
    BITS_PER_BYTE, NeonOp, NeonShape, bare_element_bits, is_general_register, uniform_vector_view,
    vector_view_bits,
};

/// `vext` — a window of the two sources concatenated, starting at an
/// immediate byte offset.
///
/// Only the byte-granular spelling is accepted. `VEXT.16` / `.32` /
/// `.64` are assembler aliases that scale the immediate onto the one
/// encoding the architecture has, so a disassembler prints `.8` and a
/// wider spelling would mean the immediate is in units this lowering
/// does not know.
pub(super) fn extract_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    if base != "vext" || bare_element_bits(ty)? != BITS_PER_BYTE {
        return None;
    }
    if insn.operands.len() != 4 {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    let offset = insn.operands.get(3)?;
    if offset.kind != OperandKind::Immediate {
        return None;
    }
    let byte_offset = u16::try_from(parse_immediate(&offset.raw)?).ok()?;
    // The window has to start inside the view; at the view's width it
    // would be the second source entire, which the encoding cannot
    // spell.
    if byte_offset >= view / BITS_PER_BYTE {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Extract { byte_offset },
        lane_bits: BITS_PER_BYTE,
        lanes: view / BITS_PER_BYTE,
    })
}

/// `vrev16` / `vrev32` / `vrev64` — reverse the order of the elements
/// inside each container the mnemonic names.
pub(super) fn reverse_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let container_bits = match base {
        "vrev16" => 16,
        "vrev32" => 32,
        "vrev64" => 64,
        _ => return None,
    };
    let lane_bits = bare_element_bits(ty)?;
    if insn.operands.len() != 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    // A container has to hold a whole number of elements, and more than
    // one of them, or there is nothing to reverse; the view in turn has
    // to hold a whole number of containers.
    if container_bits <= lane_bits || container_bits % lane_bits != 0 || view % container_bits != 0
    {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Reverse { container_bits },
        lane_bits,
        lanes: view / lane_bits,
    })
}

/// `vdup` — replicate a general-purpose register's low element across
/// every lane.
///
/// The element spelling `vdup.32 d0, d1[1]` is a different source and
/// declines here; it carries vector shape on the operand, which is the
/// by-element seam rather than this one.
pub(super) fn duplicate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    if base != "vdup" {
        return None;
    }
    let lane_bits = bare_element_bits(ty)?;
    // A doubleword element has no general-register source on a 32-bit
    // ISA, so the architecture does not define the form.
    if lane_bits > 32 || insn.operands.len() != 2 {
        return None;
    }
    let view = vector_view_bits(insn.operands.first()?)?;
    if !is_general_register(insn.operands.get(1)?) || view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Duplicate,
        lane_bits,
        lanes: view / lane_bits,
    })
}
