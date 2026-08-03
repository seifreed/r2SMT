//! The `AArch32` NEON families that relate two element geometries.
//!
//! These are what [`super::super::neon_packed_op`] structurally cannot
//! resolve: it returns one operation and one lane width for a whole
//! instruction, which describes a family only while every operand
//! shares that geometry. Here the destination's element is twice the
//! source's, or half it.
//!
//! No separate shape carrier is needed for that, and the
//! `AArch64` side does not have one either. `AArch32` states both
//! widths outright — the mnemonic names the **source** element
//! (`vmovl.s8 q0, d1` extends bytes) and the register spelling gives
//! each operand's view, `d` being 64 bits and `q` 128 — so the
//! resolver checks that the destination really is twice the source
//! rather than assuming it. `AArch64` has to re-parse an operand's
//! `.8h` arrangement at lowering time to recover the same fact.
//!
//! One thing absent by design: `AArch32` has no `2` suffix. `AArch64`
//! spells "operate on the upper half" as `xtn2` / `umull2`, and threads
//! an `upper` flag through every widening resolver and lowering to
//! carry it; here the register class says it instead, so none of that
//! plumbing exists.

use r2smt_ir::program::{Instruction, OperandKind};

use crate::lift::BinOp;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;
use crate::lift::parse_immediate;

use super::{NeonOp, NeonShape, vector_parent_bits, vector_view_bits};

/// How a widening or narrowing form relates its two geometries.
#[derive(Debug, Clone, Copy)]
pub(in crate::lift::aarch32) enum WidenKind {
    /// `vmovl` — extend each source element into one twice as wide.
    Long,
    /// `vaddl` / `vsubl` / `vmull`, and the `w`-suffixed `vaddw` /
    /// `vsubw` whose first source is already at the destination's
    /// width. Computed at the destination element width, so the
    /// product of two narrow elements cannot overflow.
    LongArith { op: BinOp, wide_first: bool },
    /// `vmovn` — truncate each source element to half its width.
    Narrow,
    /// `vshrn` — shift right, then truncate.
    ShiftNarrow { shift: u16 },
}

/// `vmovl` / `vaddl` / `vsubl` / `vmull` / `vaddw` / `vsubw` — a
/// destination element twice the width of its narrow sources.
pub(super) fn widen_long_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let (kind, sources) = match base {
        "vmovl" => (WidenKind::Long, 1),
        "vaddl" => (long_arith(BinOp::Add, false), 2),
        "vsubl" => (long_arith(BinOp::Sub, false), 2),
        "vmull" => (long_arith(BinOp::Mul, false), 2),
        "vaddw" => (long_arith(BinOp::Add, true), 2),
        "vsubw" => (long_arith(BinOp::Sub, true), 2),
        _ => return None,
    };
    // The extension *is* the operation, so the sign-agnostic `i`
    // spelling has nothing to mean and the architecture has no encoding
    // for it. `vmull.p8` is a polynomial multiply, a different lowering
    // entirely, and declines earlier: `p` names no element class.
    let (element, source_bits) = neon_element_type(ty)?;
    let signed = match element {
        ElementKind::Signed => true,
        ElementKind::Unsigned => false,
        ElementKind::Untyped | ElementKind::Float => return None,
    };
    let lane_bits = source_bits.checked_mul(2)?;
    let destination_view = vector_parent_bits()?;
    // A doubleword source would need a 128-bit destination element.
    if lane_bits > destination_view / 2 || insn.operands.len() != sources + 1 {
        return None;
    }
    if vector_view_bits(insn.operands.first()?)? != destination_view {
        return None;
    }
    let lanes = destination_view.checked_div(lane_bits)?;
    // Both geometries hold the same number of elements, so the narrow
    // side's view follows from the destination's rather than being
    // read off an operand suffix the way `AArch64` reads it.
    let narrow_view = source_bits.checked_mul(lanes)?;
    let wide_first = matches!(
        kind,
        WidenKind::LongArith {
            wide_first: true,
            ..
        }
    );
    for (position, operand) in insn.operands.iter().enumerate().skip(1) {
        let expected = if wide_first && position == 1 {
            destination_view
        } else {
            narrow_view
        };
        if vector_view_bits(operand)? != expected {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::Widen { kind, signed },
        lane_bits,
        lanes,
    })
}

const fn long_arith(op: BinOp, wide_first: bool) -> WidenKind {
    WidenKind::LongArith { op, wide_first }
}

/// `vmovn` / `vshrn` — a destination element half the width of its
/// source.
pub(super) fn narrow_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let shifts = match base {
        "vmovn" => false,
        "vshrn" => true,
        _ => return None,
    };
    // Truncation keeps the low half whatever the sign, which is why the
    // architecture spells both of these `I` and gives them no signed or
    // unsigned encoding to distinguish.
    let (element, source_bits) = neon_element_type(ty)?;
    if element != ElementKind::Untyped {
        return None;
    }
    let lane_bits = source_bits.checked_div(2)?;
    let source_view = vector_parent_bits()?;
    // Nothing narrower than a byte is an element.
    if lane_bits < 8 || insn.operands.len() != usize::from(shifts) + 2 {
        return None;
    }
    let destination_view = vector_view_bits(insn.operands.first()?)?;
    if destination_view != source_view / 2 {
        return None;
    }
    let lanes = destination_view.checked_div(lane_bits)?;
    if vector_view_bits(insn.operands.get(1)?)? != source_view
        || source_bits.checked_mul(lanes)? != source_view
    {
        return None;
    }
    let kind = if shifts {
        WidenKind::ShiftNarrow {
            shift: narrow_shift_amount(insn, lane_bits)?,
        }
    } else {
        WidenKind::Narrow
    };
    Some(NeonShape {
        op: NeonOp::Widen {
            kind,
            signed: false,
        },
        lane_bits,
        lanes,
    })
}

/// The shift a `vshrn` applies before truncating.
///
/// Bounded by the *destination* element width, which is what the
/// encoding allows and is also what keeps the lowering's signedness
/// question from arising: with `shift + lane_bits <= source_bits` no
/// bit shifted in from the top ever reaches the retained half, so a
/// logical shift and an arithmetic one give the same answer.
fn narrow_shift_amount(insn: &Instruction, lane_bits: u16) -> Option<u16> {
    let operand = insn.operands.get(2)?;
    if operand.kind != OperandKind::Immediate {
        return None;
    }
    let shift = u16::try_from(parse_immediate(&operand.raw)?).ok()?;
    (1..=lane_bits).contains(&shift).then_some(shift)
}
