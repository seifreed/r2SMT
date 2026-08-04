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

use super::{NeonOp, NeonShape, uniform_vector_view, vector_parent_bits, vector_view_bits};
use crate::lift::aarch64::neon::width::ConvertKind;

/// The packed lane-wise conversions `vcvt.{s32,u32}.f32` and
/// `vcvt.f32.{s32,u32}`, plain or by a fixed-point fraction width.
///
/// Resolved here rather than by [`super::super::neon_packed_op`] for
/// the same reason the scalar form needed its own parser: a conversion
/// names two element types, so the single `split_once('.')` both that
/// function and [`neon_element_type`] perform sees `"s32.f32"` and
/// matches nothing.
///
/// Claiming these is load-bearing beyond coverage. The scalar VFP arm
/// of the dispatcher recognises the identical mnemonic, so a packed
/// form this resolver misses is not declined — it is lowered as a
/// scalar conversion of lane zero with the rest of the register left
/// standing. Same hazard as the by-element multiplies, and the reason
/// both sit ahead of the families whose operand shape they overlap.
///
/// The half-precision pair `vcvt.f16.f32` / `vcvt.f32.f16` declines:
/// it halves or doubles the lane count, which is a different geometry
/// from everything here, and it renders only on Z3.
pub(super) fn convert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let (convert, dest_bits) = crate::lift::aarch32::vfp_convert(base, ty)?;
    let (source_kind, source_bits) = convert.source;
    // NEON spells only the 32-bit conversions between float and
    // integer; there is no packed `.f64` form and no packed integer to
    // integer one.
    if dest_bits != 32 || source_bits != 32 {
        return None;
    }
    let kind = match (convert.dest_kind, source_kind) {
        (ElementKind::Float, ElementKind::Float) => return None,
        (ElementKind::Float, signedness) => ConvertKind::IntToFloat {
            signed: signedness == ElementKind::Signed,
        },
        (signedness, ElementKind::Float) => ConvertKind::FloatToInt {
            signed: signedness == ElementKind::Signed,
        },
        // `vfp_convert` requires a float on one side, so this arm is
        // unreachable through the parser.
        _ => return None,
    };
    let fbits = match insn.operands.get(2) {
        // The fraction width is bounded by the element it scales, which
        // is 32 for every form reaching here.
        Some(operand) => {
            let amount = u16::try_from(parse_immediate(&operand.raw)?).ok()?;
            if amount == 0 || amount > dest_bits {
                return None;
            }
            amount
        }
        None => 0,
    };
    if insn.operands.len() != usize::from(fbits > 0) + 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    Some(NeonShape {
        op: NeonOp::Convert {
            kind,
            fbits,
            rounding: convert.to_int_rounding,
        },
        lane_bits: dest_bits,
        lanes: view.checked_div(dest_bits)?,
    })
}

/// How a widening or narrowing form relates two geometries.
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
    /// `vshrn` / `vrshrn` — shift right, then truncate. `rounding` adds
    /// half an ulp before the shift.
    ShiftNarrow { shift: u16, rounding: bool },
    /// `vqmovn` / `vqmovun` — clamp each double-width source element into
    /// the narrow range instead of truncating it. `source_signed` says
    /// how the source is read and which comparisons the clamp uses;
    /// `to_unsigned` selects the destination range — `vqmovun` reads a
    /// signed source but saturates into `[0, 2^n - 1]`, so a negative
    /// source becomes zero.
    /// A `shift` of zero is the plain `vqmovn`; the `vqshrn` /
    /// `vqrshrn` family shifts first, with `rounding` adding half an
    /// ulp before the shift.
    SaturatingNarrow {
        source_signed: bool,
        to_unsigned: bool,
        shift: u16,
        rounding: bool,
    },
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
    let (shifts, rounding) = match base {
        "vmovn" => (false, false),
        "vshrn" => (true, false),
        "vrshrn" => (true, true),
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
            rounding,
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

/// `vqmovn` / `vqmovun` — a destination element half the width of its
/// source, saturated rather than truncated.
///
/// `vqmovn.s16` / `vqmovn.u16` keep the source's signedness on both
/// sides; `vqmovun.s16` reads a signed source but clamps into the
/// unsigned destination range, so it has no unsigned-source spelling.
pub(super) fn saturating_narrow_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    // `shifts` separates the `vqmovn` pair (2 operands, no shift) from
    // the `vqshrn` family (3 operands, an immediate shift).
    let (to_unsigned, shifts, rounding) = match base {
        "vqmovn" => (false, false, false),
        "vqmovun" => (true, false, false),
        "vqshrn" => (false, true, false),
        "vqrshrn" => (false, true, true),
        "vqshrun" => (true, true, false),
        "vqrshrun" => (true, true, true),
        _ => return None,
    };
    let (element, source_bits) = neon_element_type(ty)?;
    let source_signed = match element {
        ElementKind::Signed => true,
        // `vqmovn.u` narrows an unsigned source; `vqmovun` has no
        // unsigned-source encoding.
        ElementKind::Unsigned if !to_unsigned => false,
        _ => return None,
    };
    if insn.operands.len() != usize::from(shifts) + 2 {
        return None;
    }
    let lane_bits = source_bits.checked_div(2)?;
    if lane_bits < 8 {
        return None;
    }
    let source_view = vector_parent_bits()?;
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
    let shift = if shifts {
        narrow_shift_amount(insn, lane_bits)?
    } else {
        0
    };
    Some(NeonShape {
        op: NeonOp::Widen {
            kind: WidenKind::SaturatingNarrow {
                source_signed,
                to_unsigned,
                shift,
                rounding,
            },
            signed: source_signed,
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
