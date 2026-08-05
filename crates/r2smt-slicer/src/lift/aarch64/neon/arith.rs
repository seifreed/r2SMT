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

use r2smt_ir::expr::RoundingMode;
use r2smt_ir::program::{Instruction, Operand};

use crate::lift::simd::CompareKind;
use crate::registers::Arrangement;

use super::super::super::{BinOp, FpArithOp, PackedIntOp, PackedOp, parse_immediate};
use super::super::FPCR_DEFAULT_ROUNDING;
use super::geometry::{
    BITS_PER_BYTE, indexed_element, operand_arrangement, peel_upper, spans_full_register,
};
use super::{NeonOp, NeonShape};

/// Element widths that name an IEEE interchange format.
///
/// Checked wherever a resolver would otherwise accept an arrangement its
/// mnemonic does not encode: `.16b` is a perfectly good arrangement and
/// a perfectly bad float sort, and reading a byte lane as a float is a
/// wrong value rather than a decline.
const fn is_float_lane(lane_bits: u16) -> bool {
    matches!(lane_bits, 16 | 32 | 64)
}

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
        "abs" => PackedOp::Int(PackedIntOp::Abs),
        "neg" => PackedOp::Int(PackedIntOp::Neg),
        // Both are sign-bit manipulations, exact at every value
        // including the NaNs, so neither needs a float sort.
        "fabs" => PackedOp::Int(PackedIntOp::SignBit { negate: false }),
        "fneg" => PackedOp::Int(PackedIntOp::SignBit { negate: true }),
        "smax" => PackedOp::Int(PackedIntOp::MinMax {
            max: true,
            signed: true,
        }),
        "smin" => PackedOp::Int(PackedIntOp::MinMax {
            max: false,
            signed: true,
        }),
        "umax" => PackedOp::Int(PackedIntOp::MinMax {
            max: true,
            signed: false,
        }),
        "umin" => PackedOp::Int(PackedIntOp::MinMax {
            max: false,
            signed: false,
        }),
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
    // `fabs` / `fneg` are checked here too even though their lowering is
    // a bit mask that would run happily on a byte lane: the architecture
    // encodes no such form, so accepting one would model an instruction
    // that cannot occur.
    let names_float_lane = matches!(
        op,
        PackedOp::Fp(_) | PackedOp::Int(PackedIntOp::SignBit { .. })
    );
    if names_float_lane && !is_float_lane(arrangement.lane_bits) {
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

// ===================== per-lane bit functions =====================

/// The unary operations whose result depends on a lane's individual
/// *bits* rather than on the value they denote.
///
/// Kept apart from the lane-wise family next door for a reason that is
/// not stylistic: the IR has no population-count, no count-leading-zeros
/// and no bit-reversal node, so each of these is a construction over
/// single-bit slices rather than one `Expr` operator.
#[derive(Debug, Clone, Copy)]
pub(super) enum BitwiseUnary {
    /// `cnt` — the number of set bits in each byte.
    PopulationCount,
    /// `clz` — the leading zero bits of each element.
    LeadingZeros,
    /// `cls` — the bits below the sign bit that repeat it, the sign bit
    /// itself excluded, so the count never reaches the element width.
    LeadingSignBits,
    /// `rbit` — each byte's bits in the opposite order.
    ReverseBits,
}

/// The per-lane bit functions.
pub(super) fn bitwise_unary_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let kind = match mnemonic {
        "cnt" => BitwiseUnary::PopulationCount,
        "clz" => BitwiseUnary::LeadingZeros,
        "cls" => BitwiseUnary::LeadingSignBits,
        "rbit" => BitwiseUnary::ReverseBits,
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if operand_arrangement(insn.operands.get(1)?)? != destination {
        return None;
    }
    // ARM ARM C7.2 — `cnt` and `rbit` encode the byte arrangements only,
    // and the two leading-bit counts stop at a 32-bit element. A wider
    // spelling is one the architecture does not produce.
    let encodable = match kind {
        BitwiseUnary::PopulationCount | BitwiseUnary::ReverseBits => {
            destination.lane_bits == BITS_PER_BYTE
        }
        BitwiseUnary::LeadingZeros | BitwiseUnary::LeadingSignBits => {
            matches!(destination.lane_bits, 8 | 16 | 32)
        }
    };
    if !encodable {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::BitwiseUnary(kind),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== round to integral =====================

/// The packed `frint*` family — round each lane to an integral value and
/// keep the float sort.
///
/// Five of the seven name their mode in the opcode and so depend on
/// nothing FPCR holds. `frinti` reads the control register and `frintx`
/// computes the same value while additionally being able to signal the
/// inexact exception, which the value model does not carry — so both pin
/// the reset default, and [`crate::lift::pins_rounding_mode`] lists them
/// by mnemonic, which covers the packed spelling for free.
pub(super) fn round_to_integral_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let rounding = match mnemonic {
        "frinta" => RoundingMode::NearestTiesAway,
        "frintn" => RoundingMode::NearestTiesEven,
        "frintp" => RoundingMode::TowardPositive,
        "frintm" => RoundingMode::TowardNegative,
        "frintz" => RoundingMode::TowardZero,
        "frinti" | "frintx" => FPCR_DEFAULT_ROUNDING,
        _ => return None,
    };
    let destination = uniform_shape(insn, 2)?;
    // Rounding a byte lane is not an encoding the architecture spells,
    // and reading one as a float would be a wrong value.
    if !is_float_lane(destination.lane_bits) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::RoundToIntegral(rounding),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== two-lane folds =====================

/// A function of two lanes.
///
/// One vocabulary for two families, because the lane-wise selects and
/// the pairwise family differ only in *which* two lanes each destination
/// lane is handed: `fmaxp` is `fmax` over the two neighbours instead of
/// over the two operands at the same index.
#[derive(Debug, Clone, Copy)]
pub(super) enum PairOp {
    /// `addp` — wrapping integer addition.
    Add,
    /// `faddp` — one IEEE lane sum.
    FloatAdd,
    /// `smax` / `umin` / `smaxp` / … — the integer ordered select.
    MinMax { signed: bool, max: bool },
    /// `fmax` / `fmin` / `fmaxp` / … — ARM's `FPMax` and `FPMin`, which
    /// propagate a NaN and combine the signs of a zero tie, or, when
    /// `number_wins`, `FPMaxNum` / `FPMinNum`, where a quiet NaN loses to
    /// a number.
    ///
    /// Neither is Intel's `MAXPS`, which is what
    /// [`super::super::super::fp_lane_result`]'s `Max` / `Min` spell and
    /// which returns its *second operand* on unordered — an ordinary
    /// number where ARM yields a NaN. That is a wrong value, not a wider
    /// one, which is why these do not reuse [`FpArithOp`].
    FloatMinMax { max: bool, number_wins: bool },
}

impl PairOp {
    /// Whether the lanes have to name an IEEE interchange format.
    const fn float(self) -> bool {
        matches!(self, Self::FloatAdd | Self::FloatMinMax { .. })
    }
}

/// The floating-point lane-wise selects.
///
/// Their integer twins (`smax` / `smin` / `umax` / `umin`) are members of
/// the lane-wise family instead, since the packed vocabulary already
/// spells an integer ordered select correctly. There is no such
/// shortcut here: the packed float vocabulary spells Intel's ordering.
pub(super) fn float_min_max_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let select = |max, number_wins| PairOp::FloatMinMax { max, number_wins };
    let op = match mnemonic {
        "fmax" => select(true, false),
        "fmin" => select(false, false),
        "fmaxnm" => select(true, true),
        "fminnm" => select(false, true),
        _ => return None,
    };
    let destination = uniform_shape(insn, 3)?;
    if !is_float_lane(destination.lane_bits) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::LaneCombine(op),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The fold a pairwise mnemonic names.
fn pairwise_op(mnemonic: &str) -> Option<PairOp> {
    let min_max = |signed, max| PairOp::MinMax { signed, max };
    let float_select = |max, number_wins| PairOp::FloatMinMax { max, number_wins };
    Some(match mnemonic {
        "addp" => PairOp::Add,
        "smaxp" => min_max(true, true),
        "sminp" => min_max(true, false),
        "umaxp" => min_max(false, true),
        "uminp" => min_max(false, false),
        "faddp" => PairOp::FloatAdd,
        "fmaxp" => float_select(true, false),
        "fminp" => float_select(false, false),
        "fmaxnmp" => float_select(true, true),
        "fminnmp" => float_select(false, true),
        _ => return None,
    })
}

/// The scalar pairwise forms: `addp d0, v1.2d`, `faddp s0, v1.2s` and
/// the float selects beside them.
///
/// The destination is a bare scalar and so carries no arrangement at
/// all, exactly as an across-lane reduction's does. The geometry
/// therefore comes from operand 1, and the destination's own width is
/// *checked* against the element it produces rather than assumed — which
/// is what stops `faddp s0, v1.2d` resolving.
///
/// The float members encode a two-lane source of any IEEE width; the
/// integer family has exactly one scalar member, `addp d0, v1.2d`, and
/// the min / max spellings have none. Accepting `smaxp d0, v1.2d` would
/// model an instruction the architecture does not encode.
pub(super) fn scalar_pairwise_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    const DOUBLEWORD_BITS: u16 = 64;
    let op = pairwise_op(mnemonic)?;
    if insn.operands.len() != 2 {
        return None;
    }
    let source = operand_arrangement(insn.operands.get(1)?)?;
    if source.lanes != 2 {
        return None;
    }
    let encodable = match op {
        PairOp::Add => source.lane_bits == DOUBLEWORD_BITS,
        PairOp::MinMax { .. } => false,
        PairOp::FloatAdd | PairOp::FloatMinMax { .. } => is_float_lane(source.lane_bits),
    };
    if !encodable {
        return None;
    }
    if super::width::scalar_vector_width(insn.operands.first()?)? != source.lane_bits {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::ScalarPairwise(op),
        lane_bits: source.lane_bits,
        lanes: 1,
        dest_index: 0,
        source_index: 0,
    })
}

/// The pairwise family: each destination lane folds two *adjacent* lanes
/// of `vn:vm` read as one vector, `vn` at the low end.
///
/// So the destination's low half comes from the first source and its
/// high half from the second — which is why an odd lane count cannot
/// occur and is rejected rather than silently halved.
pub(super) fn pairwise_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let op = pairwise_op(mnemonic)?;
    let destination = uniform_shape(insn, 3)?;
    if op.float() && !is_float_lane(destination.lane_bits) {
        return None;
    }
    if destination.lanes < 2 || destination.lanes % 2 != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Pairwise(op),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== absolute differences =====================

/// The absolute-difference family.
#[derive(Debug, Clone, Copy)]
pub(super) enum AbsDiffKind {
    /// `sabd` / `uabd`, and the accumulating `saba` / `uaba`, whose
    /// destination is an input.
    Integer { signed: bool, accumulate: bool },
    /// `fabd` — `FPAbs(FPSub(a, b))`. There is no accumulating float
    /// spelling, which is why this variant carries no flag.
    Float,
}

/// The absolute-difference family.
pub(super) fn absolute_difference_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let integer = |signed, accumulate| AbsDiffKind::Integer { signed, accumulate };
    let kind = match mnemonic {
        "sabd" => integer(true, false),
        "uabd" => integer(false, false),
        "saba" => integer(true, true),
        "uaba" => integer(false, true),
        "fabd" => AbsDiffKind::Float,
        _ => return None,
    };
    let destination = uniform_shape(insn, 3)?;
    if matches!(kind, AbsDiffKind::Float) && !is_float_lane(destination.lane_bits) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::AbsoluteDifference(kind),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The arrangement every operand of an `operands`-long instruction
/// shares, or `None` when they differ or the count is wrong.
fn uniform_shape(insn: &Instruction, operands: usize) -> Option<Arrangement> {
    if insn.operands.len() != operands {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    for operand in insn.operands.iter().skip(1) {
        if operand_arrangement(operand)? != destination {
            return None;
        }
    }
    Some(destination)
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
    /// `shl` / `sqshl` / `uqshl` / `sqshlu` — shift left by an
    /// immediate, discarding the bits that leave the element or clamping
    /// into the range `saturate` names instead.
    ///
    /// The range is carried rather than derived from the family's
    /// signedness because `sqshlu` separates the two: it reads its
    /// element *signed* and clamps into the *unsigned* range, so neither
    /// a signed nor an unsigned clamp describes it.
    LeftImmediate {
        shift: u16,
        saturate: Option<SaturateTo>,
    },
    /// `ushr` / `sshr` / `urshr` / `srshr` — shift right by an
    /// immediate, optionally rounding.
    RightImmediate { shift: u16, rounding: bool },
    /// `ushl` / `sshl` / `urshl` / `srshl` / `sqshl` / `uqshl` /
    /// `sqrshl` / `uqrshl` — shift by the second source's per-lane
    /// amount, left when positive and right when negative.
    ///
    /// `saturate` applies to the left direction only, which is not an
    /// approximation: a right shift of an `n`-bit element always fits
    /// `n` bits, so there is nothing for it to clamp.
    Register { rounding: bool, saturate: bool },
}

/// The saturating left shifts, whose one mnemonic spells two shapes.
///
/// `sqshl v0.8b, v1.8b, #3` and `sqshl v0.8b, v1.8b, v2.8b` are
/// genuinely different instructions — a fixed left shift and a per-lane
/// signed amount that can go either way — and the only thing telling
/// them apart is whether the amount operand carries an arrangement.
fn saturating_left_shift(insn: &Instruction, to: SaturateTo) -> Option<ShiftKind> {
    let amount = insn.operands.get(2)?;
    if operand_arrangement(amount).is_some() {
        return Some(ShiftKind::Register {
            rounding: false,
            saturate: true,
        });
    }
    Some(ShiftKind::LeftImmediate {
        shift: u16::try_from(parse_immediate(&amount.raw)?).ok()?,
        saturate: Some(to),
    })
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
    let register = |rounding, saturate| ShiftKind::Register { rounding, saturate };
    let (kind, signed) = match mnemonic {
        "shl" => (
            ShiftKind::LeftImmediate {
                shift: immediate_shift()?,
                saturate: None,
            },
            false,
        ),
        "ushr" => (right(false)?, false),
        "sshr" => (right(false)?, true),
        "urshr" => (right(true)?, false),
        "srshr" => (right(true)?, true),
        "ushl" => (register(false, false), false),
        "sshl" => (register(false, false), true),
        "urshl" => (register(true, false), false),
        "srshl" => (register(true, false), true),
        "sqshl" => (saturating_left_shift(insn, SaturateTo::Signed)?, true),
        "uqshl" => (saturating_left_shift(insn, SaturateTo::Unsigned)?, false),
        // The one member whose source signedness and clamp range
        // disagree, and immediate-only: the architecture spells no
        // register form of it.
        "sqshlu" => (
            ShiftKind::LeftImmediate {
                shift: immediate_shift()?,
                saturate: Some(SaturateTo::SignedToUnsigned),
            },
            true,
        ),
        "sqrshl" => (register(true, true), true),
        "uqrshl" => (register(true, true), false),
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
        ShiftKind::LeftImmediate { shift, .. } => shift < destination.lane_bits,
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

/// `sri` / `sli` — the shift-and-insert pair.
///
/// The two directions carry different immediate ranges, and both bounds
/// are the encoding's rather than a convenience: `sli` shifts left by
/// `0..n-1` and `sri` right by `1..n`, so each spells exactly the
/// amounts that leave at least one source bit in the destination. A
/// `sri` by the full element width is therefore legal and copies
/// nothing, which the lowering reproduces rather than special-cases.
pub(super) fn shift_insert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let left = match mnemonic {
        "sli" => true,
        "sri" => false,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if operand_arrangement(insn.operands.get(1)?)? != destination {
        return None;
    }
    let shift = u16::try_from(parse_immediate(&insn.operands.get(2)?.raw)?).ok()?;
    let bounded = if left {
        shift < destination.lane_bits
    } else {
        shift > 0 && shift <= destination.lane_bits
    };
    if !bounded {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::ShiftInsert { left, shift },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== lane-wise compares =====================

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
    /// `sqabs` / `sqneg` — the magnitude or the negation of one element,
    /// clamped into the signed range.
    ///
    /// The clamp is the whole reason these are not members of the
    /// lane-wise family, where `abs` and `neg` already sit: both wrap at
    /// `INT_MIN`, whose magnitude is one past the top of the range, so
    /// resolving `sqabs` through `abs` would answer `INT_MIN` where the
    /// architecture answers `INT_MAX`. A wrong value, not a decline.
    Unary { negate: bool },
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
        "sqabs" => (SaturatingKind::Unary { negate: false }, true),
        "sqneg" => (SaturatingKind::Unary { negate: true }, true),
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
        SaturatingKind::Narrow { .. } | SaturatingKind::Unary { .. } => 2,
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

/// `suqadd` / `usqadd` — the mixed-signedness accumulate.
///
/// Two operands, both the destination's arrangement, and the
/// destination is an input: it is the accumulator. What separates this
/// from the ordinary saturating add is that the two addends are read
/// with *different* signednesses and the clamp is into the range of the
/// destination, which is neither addend's.
pub(super) fn mixed_sign_add_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let destination_signed = match mnemonic {
        "suqadd" => true,
        "usqadd" => false,
        _ => return None,
    };
    let destination = uniform_shape(insn, 2)?;
    Some(NeonShape {
        op: NeonOp::MixedSignAdd { destination_signed },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// `sqdmull` / `sqdmlal` / `sqdmlsl` — the doubling long multiplies.
///
/// A product, and yet resolved here rather than beside the other
/// multiplies, for the reason the module header gives: what defines the
/// family is the clamp. `2 * INT_MIN * INT_MIN` is `2^(2n-1)`, one past
/// the top of the doubled element's signed range, and it is the only
/// input pair at which these saturate at all.
pub(super) fn doubling_long_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let combine = match base {
        "sqdmull" => None,
        "sqdmlal" => Some(BinOp::Add),
        "sqdmlsl" => Some(BinOp::Sub),
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    // The architecture encodes a 16- or 32-bit source element only, so
    // the destination is 32 or 64. Without this a `.8h` destination
    // would resolve against byte sources the encoding does not spell.
    if !matches!(destination.lane_bits, 32 | 64) {
        return None;
    }
    let source_bits = destination.lane_bits / 2;
    let expected_lanes = if upper {
        destination.lanes.checked_mul(2)?
    } else {
        destination.lanes
    };
    let first = operand_arrangement(insn.operands.get(1)?)?;
    if first.lane_bits != source_bits || first.lanes != expected_lanes {
        return None;
    }
    if upper && !spans_full_register(first) {
        return None;
    }
    // The second source is either the matching whole vector or a single
    // element. The `2` suffix never halves the indexed one: an element
    // is addressed absolutely inside the register, so `sqdmull2 v0.4s,
    // v1.8h, v2.h[7]` reads `v1`'s upper half and `v2`'s lane seven.
    let second = insn.operands.get(2)?;
    let (by_element, source_index) = if let Some(arrangement) = operand_arrangement(second) {
        if arrangement.lane_bits != source_bits || arrangement.lanes != expected_lanes {
            return None;
        }
        if upper && !spans_full_register(arrangement) {
            return None;
        }
        (false, 0)
    } else {
        let (element_bits, index) = indexed_element(second)?;
        if element_bits != source_bits {
            return None;
        }
        (true, index)
    };
    Some(NeonShape {
        op: NeonOp::DoublingLong {
            combine,
            upper,
            by_element,
        },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index,
    })
}

// ===================== estimates =====================

/// `frecpe` / `frsqrte` / `urecpe` / `ursqrte`, the reciprocal and
/// reciprocal-square-root estimates.
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
    // The unsigned pair shares the lowering because it shares the
    // objection: `urecpe` / `ursqrte` normalise the element, look the
    // reciprocal up in a table the architecture specifies to eight
    // fraction bits, and renormalise. That is a divide the IR cannot
    // spell, so the choice is a free value or a decline, and a decline
    // would truncate the slice. Unlike the float pair they encode only
    // a 32-bit element.
    let float = match mnemonic {
        "frecpe" | "frsqrte" => true,
        "urecpe" | "ursqrte" => false,
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    let encodable = if float {
        // An estimate of an IEEE lane; `.16b` names no float format.
        is_float_lane(destination.lane_bits)
    } else {
        destination.lane_bits == 32
    };
    if !encodable {
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
