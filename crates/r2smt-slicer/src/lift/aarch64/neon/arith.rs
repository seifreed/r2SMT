//! Lane arithmetic and logic.
//!
//! The families whose result is an arithmetic or logical function of
//! whole source lanes: the lane-wise arithmetic and bitwise operations,
//! the same-width shifts, the lane compares, the saturating and halving
//! forms, and the reciprocal estimates.
//!
//! The saturating family sits here rather than among the width-changing
//! ones even though half its members narrow. One resolver covers both
//! its spellings, and what defines the family is the clamp rather than
//! the width — `sqadd` and `sqxtn` differ in where they saturate, not
//! in whether they do.

use r2smt_ir::program::{Instruction, Operand};

use crate::registers::Arrangement;

use super::super::super::{BinOp, FpArithOp, PackedIntOp, PackedOp, parse_immediate};
use super::geometry::{operand_arrangement, peel_upper, spans_full_register};
use super::{NeonOp, NeonShape};

// ===================== lane-wise arithmetic and logic =====================

/// The packed operation an `AArch64` NEON data-processing mnemonic
/// computes, or `None` for a mnemonic no packed handler models.
fn packed_op(mnemonic: &str) -> Option<PackedOp> {
    Some(match mnemonic {
        "add" => PackedOp::Int(PackedIntOp::Bin(BinOp::Add)),
        "sub" => PackedOp::Int(PackedIntOp::Bin(BinOp::Sub)),
        "mul" => PackedOp::Int(PackedIntOp::Bin(BinOp::Mul)),
        "and" => PackedOp::Int(PackedIntOp::Bin(BinOp::And)),
        "orr" => PackedOp::Int(PackedIntOp::Bin(BinOp::Or)),
        "eor" => PackedOp::Int(PackedIntOp::Bin(BinOp::Xor)),
        "bic" => PackedOp::Int(PackedIntOp::BitClear),
        "mvn" | "not" => PackedOp::Int(PackedIntOp::Not),
        "mov" => PackedOp::Int(PackedIntOp::Copy),
        "fadd" => PackedOp::Fp(FpArithOp::Add),
        "fsub" => PackedOp::Fp(FpArithOp::Sub),
        "fmul" => PackedOp::Fp(FpArithOp::Mul),
        "fdiv" => PackedOp::Fp(FpArithOp::Div),
        _ => return None,
    })
}

/// The lane-wise family: every operand a vector register carrying the
/// *same* arrangement.
///
/// That is what the architecture spells for these mnemonics, and
/// requiring it is what rejects the widening forms
/// (`umlal v0.4s, v1.4h, v2.4h`), the by-element forms
/// (`mul v0.4s, v1.4s, v2.s[0]`) and the immediate ones
/// (`bic v0.4h, #0x10`) without listing any of them.
pub(super) fn packed_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let op = packed_op(mnemonic)?;
    if insn.operands.len() != op.operand_count() {
        return None;
    }
    let mut shared: Option<Arrangement> = None;
    for operand in &insn.operands {
        let arrangement = operand_arrangement(operand)?;
        if *shared.get_or_insert(arrangement) != arrangement {
            return None;
        }
    }
    let arrangement = shared?;
    // A floating-point lane has to name a float sort; `.16b` does not.
    if matches!(op, PackedOp::Fp(_)) && !matches!(arrangement.lane_bits, 16 | 32 | 64) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Packed(op),
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== same-width shifts =====================

/// The same-width shift operations.
///
/// The immediate forms and the register forms are genuinely different
/// shapes, not one shape with a different operand: an immediate shift
/// names its direction in the mnemonic, while `sshl` and `ushl` take a
/// *signed* per-lane amount whose sign chooses the direction at run
/// time.
#[derive(Debug, Clone, Copy)]
pub(super) enum ShiftKind {
    /// `shl` — shift left by an immediate.
    LeftImmediate { shift: u16 },
    /// `ushr` / `sshr` / `urshr` / `srshr` — shift right by an
    /// immediate, optionally rounding.
    RightImmediate { shift: u16, rounding: bool },
    /// `ushl` / `sshl` / `urshl` / `srshl` — shift by the second
    /// source's per-lane amount, left when positive and right when
    /// negative.
    Register { rounding: bool },
}

/// The same-width shift family.
///
/// The immediate forms carry their amount as an operand and their
/// direction in the mnemonic. The register forms carry a whole vector of
/// per-lane amounts, each read as a *signed* value whose sign chooses
/// the direction — so one lane of a `sshl` can shift left while its
/// neighbour shifts right.
pub(super) fn shift_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let immediate_shift =
        || -> Option<u16> { u16::try_from(parse_immediate(&insn.operands.get(2)?.raw)?).ok() };
    let right = |rounding| -> Option<ShiftKind> {
        Some(ShiftKind::RightImmediate {
            shift: immediate_shift()?,
            rounding,
        })
    };
    let (kind, signed) = match mnemonic {
        "shl" => (
            ShiftKind::LeftImmediate {
                shift: immediate_shift()?,
            },
            false,
        ),
        "ushr" => (right(false)?, false),
        "sshr" => (right(false)?, true),
        "urshr" => (right(true)?, false),
        "srshr" => (right(true)?, true),
        "ushl" => (ShiftKind::Register { rounding: false }, false),
        "sshl" => (ShiftKind::Register { rounding: false }, true),
        "urshl" => (ShiftKind::Register { rounding: true }, false),
        "srshl" => (ShiftKind::Register { rounding: true }, true),
        _ => return None,
    };
    let register_form = matches!(kind, ShiftKind::Register { .. });
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    // Every vector operand shares the destination's arrangement; the
    // immediate forms' third operand is not one.
    for (index, operand) in insn.operands.iter().enumerate().skip(1) {
        let Some(arrangement) = operand_arrangement(operand) else {
            if !register_form && index == 2 {
                continue;
            }
            return None;
        };
        if arrangement != destination {
            return None;
        }
    }
    // A left shift by the element width, or a right shift past it, is
    // outside the immediate encodings' range.
    let bounded = match kind {
        ShiftKind::LeftImmediate { shift } => shift < destination.lane_bits,
        ShiftKind::RightImmediate { shift, .. } => shift > 0 && shift <= destination.lane_bits,
        ShiftKind::Register { .. } => true,
    };
    if !bounded {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Shift { kind, signed },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== lane-wise compares =====================

/// The lane-wise compares.
///
/// Each writes a mask, not a condition flag: the whole point is that the
/// result feeds another vector operation.
#[derive(Debug, Clone, Copy)]
pub(super) enum CompareKind {
    /// `cmeq` / `fcmeq` — equality.
    Equal { float: bool },
    /// `cmgt` / `cmge` / `cmhi` / `cmhs` and the float `fcmgt` /
    /// `fcmge` — ordered comparison. `or_equal` picks `ge` over `gt`.
    Ordered {
        float: bool,
        signed: bool,
        or_equal: bool,
    },
    /// `cmtst` — true where the bitwise AND of the lanes is non-zero.
    TestBits,
}

/// The lane-wise compare family.
///
/// Each member has a two-operand form comparing against zero
/// (`cmgt v0.4s, v1.4s, #0`) as well as the three-operand register form,
/// and the zero form is what most opaque-predicate patterns use.
pub(super) fn compare_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let ordered = |float, signed, or_equal| CompareKind::Ordered {
        float,
        signed,
        or_equal,
    };
    let kind = match mnemonic {
        "cmeq" => CompareKind::Equal { float: false },
        "fcmeq" => CompareKind::Equal { float: true },
        "cmgt" => ordered(false, true, false),
        "cmge" => ordered(false, true, true),
        "cmhi" => ordered(false, false, false),
        "cmhs" => ordered(false, false, true),
        "fcmgt" => ordered(true, true, false),
        "fcmge" => ordered(true, true, true),
        "cmtst" => CompareKind::TestBits,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if operand_arrangement(insn.operands.get(1)?)? != destination {
        return None;
    }
    let second = insn.operands.get(2)?;
    let zero = if let Some(arrangement) = operand_arrangement(second) {
        if arrangement != destination {
            return None;
        }
        false
    } else {
        // The compare-with-zero form. `cmtst` has none.
        if matches!(kind, CompareKind::TestBits) || !is_zero_immediate(second) {
            return None;
        }
        true
    };
    // A floating-point compare needs an IEEE lane.
    let float = matches!(
        kind,
        CompareKind::Equal { float: true } | CompareKind::Ordered { float: true, .. }
    );
    if float && !matches!(destination.lane_bits, 16 | 32 | 64) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Compare { kind, zero },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// Whether an operand is the `#0` (or `#0.0`) the compare-with-zero
/// forms take.
fn is_zero_immediate(op: &Operand) -> bool {
    let raw = op.raw.trim().trim_start_matches('#');
    matches!(raw, "0" | "0.0" | "0x0" | "0.00000")
}

// ===================== saturation and rounding =====================

/// Which range a computed value is clamped into.
///
/// The distinction that matters is not the operation's signedness but
/// the *result's*: `uqsub` computes a value that can go negative and
/// clamps it into the unsigned range, which is neither of the two
/// obvious cases.
#[derive(Debug, Clone, Copy)]
pub(super) enum SaturateTo {
    /// Clamp into `[-2^(n-1), 2^(n-1) - 1]`, comparing signed.
    Signed,
    /// Clamp into `[0, 2^n - 1]` from a value that cannot be negative,
    /// comparing unsigned.
    Unsigned,
    /// Clamp a value that *can* be negative into `[0, 2^n - 1]`
    /// (`uqsub`, `sqxtun`, `sqshrun`): negatives become zero.
    SignedToUnsigned,
}

/// The saturating and rounding element operations.
#[derive(Debug, Clone, Copy)]
pub(super) enum SaturatingKind {
    /// `sqadd` / `uqadd` / `sqsub` / `uqsub` — computed one bit wider so
    /// the overflow is visible, then clamped.
    AddSub { op: BinOp, to: SaturateTo },
    /// `uhadd` / `shadd` / `urhadd` / `srhadd` — add one bit wider, then
    /// halve. The extra bit makes the sum exact, so nothing saturates.
    Halving { rounding: bool },
    /// `sqxtn` / `uqxtn` / `sqxtun` — clamp a double-width element into
    /// the narrow range.
    Narrow { to: SaturateTo },
    /// The shift-right-narrow family: shift the double-width element,
    /// optionally rounding, then narrow — clamping when the mnemonic
    /// says so and plain truncation (`shrn` / `rshrn`) when it does not.
    ShiftNarrow {
        shift: u16,
        rounding: bool,
        to: Option<SaturateTo>,
    },
    /// `sqdmulh` / `sqrdmulh` — double the product and keep its high
    /// half.
    DoublingMultiplyHigh { rounding: bool },
}

/// The same-width saturating and halving mnemonics.
fn saturating_same_width(base: &str) -> Option<(SaturatingKind, bool)> {
    use SaturateTo::{Signed, SignedToUnsigned, Unsigned};
    let add_sub = |op, to| SaturatingKind::AddSub { op, to };
    Some(match base {
        "sqadd" => (add_sub(BinOp::Add, Signed), true),
        "uqadd" => (add_sub(BinOp::Add, Unsigned), false),
        "sqsub" => (add_sub(BinOp::Sub, Signed), true),
        // An unsigned subtract can go below zero, so its clamp is the
        // signed-into-unsigned one rather than a plain upper bound.
        "uqsub" => (add_sub(BinOp::Sub, SignedToUnsigned), false),
        "shadd" => (SaturatingKind::Halving { rounding: false }, true),
        "uhadd" => (SaturatingKind::Halving { rounding: false }, false),
        "srhadd" => (SaturatingKind::Halving { rounding: true }, true),
        "urhadd" => (SaturatingKind::Halving { rounding: true }, false),
        "sqdmulh" => (
            SaturatingKind::DoublingMultiplyHigh { rounding: false },
            true,
        ),
        "sqrdmulh" => (
            SaturatingKind::DoublingMultiplyHigh { rounding: true },
            true,
        ),
        _ => return None,
    })
}

/// The narrowing saturating mnemonics, whose destination element is half
/// the source's.
fn saturating_narrowing(base: &str, insn: &Instruction) -> Option<(SaturatingKind, bool)> {
    use SaturateTo::{Signed, SignedToUnsigned, Unsigned};
    if let Some(to) = match base {
        "sqxtn" => Some(Signed),
        "uqxtn" => Some(Unsigned),
        "sqxtun" => Some(SignedToUnsigned),
        _ => None,
    } {
        let signed_sources = !matches!(to, Unsigned);
        return Some((SaturatingKind::Narrow { to }, signed_sources));
    }
    let (to, rounding, signed_sources) = match base {
        "shrn" => (None, false, true),
        "rshrn" => (None, true, true),
        "sqshrn" => (Some(Signed), false, true),
        "sqrshrn" => (Some(Signed), true, true),
        "uqshrn" => (Some(Unsigned), false, false),
        "uqrshrn" => (Some(Unsigned), true, false),
        "sqshrun" => (Some(SignedToUnsigned), false, true),
        "sqrshrun" => (Some(SignedToUnsigned), true, true),
        _ => return None,
    };
    let shift = u16::try_from(parse_immediate(&insn.operands.get(2)?.raw)?).ok()?;
    Some((
        SaturatingKind::ShiftNarrow {
            shift,
            rounding,
            to,
        },
        signed_sources,
    ))
}

/// The saturating, halving and rounding-narrow family.
pub(super) fn saturating_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let (kind, signed_sources) =
        saturating_same_width(base).or_else(|| saturating_narrowing(base, insn))?;
    let narrowing = matches!(
        kind,
        SaturatingKind::Narrow { .. } | SaturatingKind::ShiftNarrow { .. }
    );
    let expected_operands = match kind {
        SaturatingKind::Narrow { .. } => 2,
        _ => 3,
    };
    if insn.operands.len() != expected_operands {
        return None;
    }
    // Only the narrowing forms have a `2` variant.
    if upper && !narrowing {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    // `sqdmulh` doubles a product, which the architecture only encodes
    // for 16- and 32-bit elements.
    if matches!(kind, SaturatingKind::DoublingMultiplyHigh { .. })
        && !matches!(destination.lane_bits, 16 | 32)
    {
        return None;
    }
    let written = if narrowing && upper {
        destination.lanes / 2
    } else {
        destination.lanes
    };
    if written == 0 {
        return None;
    }
    if upper && !spans_full_register(destination) {
        return None;
    }
    let source_bits = if narrowing {
        destination.lane_bits.checked_mul(2)?
    } else {
        destination.lane_bits
    };
    if source_bits > 64 {
        return None;
    }
    // A shift has to leave the surviving bits inside the source element,
    // which is what makes the shift direction's signedness irrelevant to
    // the truncated result.
    if let SaturatingKind::ShiftNarrow { shift, .. } = kind
        && (shift == 0 || shift > destination.lane_bits)
    {
        return None;
    }
    for (index, operand) in insn.operands.iter().enumerate().skip(1) {
        let Some(arrangement) = operand_arrangement(operand) else {
            if matches!(kind, SaturatingKind::ShiftNarrow { .. }) && index == 2 {
                continue;
            }
            return None;
        };
        if arrangement.lane_bits != source_bits || arrangement.lanes != written {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::Saturating {
            kind,
            signed_sources,
            upper,
        },
        lane_bits: destination.lane_bits,
        lanes: written,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== estimates =====================

/// `frecpe` / `frsqrte`, the reciprocal and reciprocal-square-root
/// estimates.
///
/// Their *refinement* steps `frecps` / `frsqrts` are deliberately not
/// here. `2.0 - x*y` and `(3.0 - x*y) / 2.0` describe them arithmetically,
/// but `AArch64` computes both through `FPRecipStepFused` — one
/// rounding over the whole expression, where a separate `fmul` and
/// `fsub` round twice. That is the same objection that keeps `fmla`
/// out: the IR has no fused node, so the obvious lowering would be a
/// definite wrong value rather than a wider one. (It is expressible for
/// binary32 lanes by computing in binary64, whose 53 bits exceed the
/// `2 * 24 + 2` an exact emulation needs — but not for binary64 lanes,
/// which would need binary128, so that is its own piece of work rather
/// than a half-covered special case.)
pub(super) fn estimate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if !matches!(mnemonic, "frecpe" | "frsqrte") || insn.operands.len() != 2 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    // An estimate of an IEEE lane; `.16b` names no float format.
    if !matches!(destination.lane_bits, 16 | 32 | 64) {
        return None;
    }
    if operand_arrangement(insn.operands.get(1)?)? != destination {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Estimate,
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}
