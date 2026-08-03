//! The `AArch32` NEON by-element forms — `vmul` / `vmla` / `vmls` and
//! their long `vmull` / `vmlal` / `vmlsl` counterparts, where the
//! second source contributes **one** element to every destination lane
//! instead of pairing lane with lane.
//!
//! These are blocked differently from the rest of this module. A
//! permutation is invisible to the mnemonic parser; an indexed form is
//! invisible to the *dispatcher*: `d2[0]` carries vector shape, so
//! [`crate::lift::vector_shape`] answered `Declined` for the whole
//! instruction before any mnemonic was looked at. That guard is not
//! defensive — [`register_layout`] deliberately maps `d2[0]` onto the
//! whole of `d2`, so an integer handler reading it would take all 64
//! bits where the instruction names one lane. Resolving the shape here
//! is what lets the guard say `Lifted` instead.
//!
//! Recognising these is load-bearing beyond adding them, for the same
//! reason `neon_packed_op`'s doc gives: `vmul.i16 d0, d1, d2[0]` has
//! the operand count and kinds the *lane-wise* multiply accepts, so a
//! form this resolver misses is not declined — it is lowered as a
//! lane-wise multiply against a whole register, which is a wrong value
//! rather than a truncated slice. The dispatch tries this module before
//! the packed arm for exactly that reason.

use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::lift::BinOp;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;
use crate::registers::{is_simd_parent, register_layout};
use r2smt_common::Arch;

use super::{NeonOp, NeonShape, vector_parent_bits, vector_view_bits};

/// Which by-element product a mnemonic names.
#[derive(Debug, Clone, Copy)]
pub(in crate::lift::aarch32) enum ByElementKind {
    /// `vmul` / `vmla` / `vmls` — the product is the same width as its
    /// sources. `combine` is set for the accumulating pair, which reads
    /// the destination.
    Same { combine: Option<BinOp> },
    /// `vmull` / `vmlal` / `vmlsl` — the product is twice the width of
    /// its sources, so it cannot overflow.
    Long {
        combine: Option<BinOp>,
        signed: bool,
    },
}

/// `vmul` / `vmla` / `vmls` / `vmull` / `vmlal` / `vmlsl` with an
/// indexed second source.
pub(super) fn by_element_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let (long, combine) = match base {
        "vmul" => (false, None),
        "vmla" => (false, Some(BinOp::Add)),
        "vmls" => (false, Some(BinOp::Sub)),
        "vmull" => (true, None),
        "vmlal" => (true, Some(BinOp::Add)),
        "vmlsl" => (true, Some(BinOp::Sub)),
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let (element, source_bits) = neon_element_type(ty)?;
    // No by-element encoding exists at byte size — the index would not
    // fit the field — and the floating-point forms are a separate
    // lowering this module does not model.
    if !matches!(source_bits, 16 | 32) || element == ElementKind::Float {
        return None;
    }
    let (indexed_view, index) = indexed_element(insn.operands.get(2)?)?;
    // The index has to address an element the register actually holds.
    if index >= indexed_view.checked_div(source_bits)? {
        return None;
    }
    let kind = if long {
        // Widening needs the extension named, so the sign-agnostic `i`
        // spelling has nothing to mean here.
        let signed = match element {
            ElementKind::Signed => true,
            ElementKind::Unsigned => false,
            _ => return None,
        };
        ByElementKind::Long { combine, signed }
    } else {
        ByElementKind::Same { combine }
    };
    let lane_bits = if long {
        source_bits.checked_mul(2)?
    } else {
        source_bits
    };
    let destination_view = vector_view_bits(insn.operands.first()?)?;
    if long && destination_view != vector_parent_bits()? {
        return None;
    }
    let lanes = destination_view.checked_div(lane_bits)?;
    // The lane-wise source holds one element per destination lane, at
    // the width the mnemonic names rather than the destination's.
    if vector_view_bits(insn.operands.get(1)?)? != source_bits.checked_mul(lanes)? || lanes < 2 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::ByElement { kind, index },
        lane_bits,
        lanes,
    })
}

/// The view width and lane index an indexed vector operand names —
/// `(64, 3)` for `d2[3]`.
///
/// Deliberately separate from [`vector_view_bits`], which refuses an
/// operand carrying vector shape: this is the one place that shape is
/// understood rather than fenced off.
pub(super) fn indexed_element(op: &Operand) -> Option<(u16, u16)> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    let (head, rest) = raw.split_once('[')?;
    let index = rest.strip_suffix(']')?.trim().parse::<u16>().ok()?;
    let layout = register_layout(head.trim(), Arch::Arm)?;
    is_simd_parent(layout.parent, Arch::Arm).then(|| (layout.width(), index))
}
