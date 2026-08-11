//! The `AArch32` NEON families that move bits without computing on
//! them, and whose element is spelled as a bare width.
//!
//! `vzip` / `vuzp` / `vtrn` sit apart from the rest even here: on
//! `AArch32` one of them rewrites *both* named registers, where
//! `AArch64` splits the same work across the single-destination `zip1`
//! and `zip2`. That is why [`PairKind`] is a separate vocabulary from
//! the one-destination forms rather than another arm beside them — the
//! effect table has to record two definitions, and the lowering has to
//! finish reading before it starts writing.

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

/// Which of the two named registers a lane comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::lift::aarch32) enum PairSource {
    First,
    Second,
}

/// The permutations that rewrite both of their operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::lift::aarch32) enum PairKind {
    /// `vzip` — interleave the two registers.
    Zip,
    /// `vuzp` — de-interleave them.
    Uzp,
    /// `vtrn` — transpose, swapping each odd element of the first
    /// register with the even element beside it in the second.
    Trn,
}

/// `vzip` / `vuzp` / `vtrn` — a permutation across the two named
/// registers, writing both.
pub(super) fn permute_pair_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let kind = match base {
        "vzip" => PairKind::Zip,
        "vuzp" => PairKind::Uzp,
        "vtrn" => PairKind::Trn,
        _ => return None,
    };
    let lane_bits = bare_element_bits(ty)?;
    if insn.operands.len() != 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    // All three are `UNDEFINED` at doubleword element size, whatever
    // the register class.
    if lane_bits == 64 {
        return None;
    }
    let lanes = view.checked_div(lane_bits)?;
    // `vzip` and `vuzp` are additionally undefined wherever a register
    // holds only two elements — `VZIP.32 Dd, Dm` — because the
    // operation there degenerates to the swap `vtrn.32` already spells,
    // and an assembler accepting that spelling emits `VTRN`.
    if lanes < 2 || (lanes == 2 && kind != PairKind::Trn) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::PermutePair(kind),
        lane_bits,
        lanes,
    })
}

/// The source lane feeding destination lane `index` of whichever
/// register `into_second` names.
///
/// Transcribed from the `AArch32` pseudocode rather than adapted from
/// `AArch64`'s `zip1` / `zip2`, which halve differently: there the
/// suffix picks which half of the *sources* to read, here one
/// instruction writes both destinations and each takes half of one
/// combined sequence.
pub(super) fn paired_source(
    kind: PairKind,
    index: u16,
    into_second: bool,
    lanes: u16,
) -> Option<(PairSource, u16)> {
    let half = lanes / 2;
    Some(match kind {
        // Laid end to end the two registers make one sequence twice as
        // long, alternating between them; the first destination takes
        // its low half and the second its high half.
        PairKind::Zip => {
            let source_lane =
                index
                    .checked_div(2)?
                    .checked_add(if into_second { half } else { 0 })?;
            if index.is_multiple_of(2) {
                (PairSource::First, source_lane)
            } else {
                (PairSource::Second, source_lane)
            }
        }
        // The inverse: the first destination collects the even elements
        // of the concatenation and the second the odd ones.
        PairKind::Uzp => {
            let offset = u16::from(into_second);
            if index < half {
                (
                    PairSource::First,
                    index.checked_mul(2)?.checked_add(offset)?,
                )
            } else {
                (
                    PairSource::Second,
                    index
                        .checked_sub(half)?
                        .checked_mul(2)?
                        .checked_add(offset)?,
                )
            }
        }
        // A 2x2 transpose: the odd element of the first register trades
        // places with the even element of the second.
        PairKind::Trn => {
            let source_lane = index
                .checked_div(2)?
                .checked_mul(2)?
                .checked_add(u16::from(into_second))?;
            if index.is_multiple_of(2) {
                (PairSource::First, source_lane)
            } else {
                (PairSource::Second, source_lane)
            }
        }
    })
}
