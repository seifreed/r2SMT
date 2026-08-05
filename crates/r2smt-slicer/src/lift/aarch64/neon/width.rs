//! The families that relate two different element geometries.
//!
//! The widening and narrowing arithmetic, the conversions between
//! integer and float and between float widths, and the across-lane
//! reductions.
//!
//! What they have in common is the thing every other resolver may
//! assume away: that the destination's arrangement describes its
//! sources too. Here it does not, so each source is sized against the
//! destination rather than compared to it — and a reduction's
//! destination is a bare scalar carrying no arrangement at all, which
//! is why that family reads its geometry from operand 1.

use r2smt_common::Arch;
use r2smt_ir::expr::RoundingMode;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::registers::{is_simd_parent, register_layout};

use super::super::super::{BinOp, parse_immediate};
use super::geometry::{operand_arrangement, peel_upper, spans_full_register};
use super::{NeonOp, NeonShape};

// ===================== widening and narrowing =====================

/// The widening and narrowing element operations.
#[derive(Debug, Clone, Copy)]
pub(super) enum WidenKind {
    /// `ushll` / `sshll` and their `uxtl` / `sxtl` zero-shift aliases:
    /// extend each element to double width, then shift left.
    ShiftLong { shift: u16 },
    /// `xtn` — truncate each element to half its width.
    Narrow,
    /// The long and wide arithmetic: extend the narrow sources to the
    /// destination's element width and operate there. `wide_first` marks
    /// the `w`-suffixed forms, whose first source is already wide.
    Arith { op: BinOp, wide_first: bool },
}

/// The widening / narrowing operation a mnemonic names, with the
/// signedness its leading letter spells.
fn widen_kind(base: &str) -> Option<(WidenKind, bool)> {
    // The zero-shift aliases carry no immediate operand of their own.
    let arith = |op, wide_first| WidenKind::Arith { op, wide_first };
    Some(match base {
        "xtn" => (WidenKind::Narrow, false),
        "uaddl" => (arith(BinOp::Add, false), false),
        "saddl" => (arith(BinOp::Add, false), true),
        "usubl" => (arith(BinOp::Sub, false), false),
        "ssubl" => (arith(BinOp::Sub, false), true),
        "umull" => (arith(BinOp::Mul, false), false),
        "smull" => (arith(BinOp::Mul, false), true),
        "uaddw" => (arith(BinOp::Add, true), false),
        "saddw" => (arith(BinOp::Add, true), true),
        "usubw" => (arith(BinOp::Sub, true), false),
        "ssubw" => (arith(BinOp::Sub, true), true),
        _ => return None,
    })
}

/// `ushll` / `sshll` and their `uxtl` / `sxtl` aliases, whose shift is
/// an operand in the first spelling and zero in the second.
fn shift_long_kind(base: &str, insn: &Instruction) -> Option<(WidenKind, bool)> {
    let (signed, has_immediate) = match base {
        "ushll" => (false, true),
        "sshll" => (true, true),
        "uxtl" => (false, false),
        "sxtl" => (true, false),
        _ => return None,
    };
    let shift = if has_immediate {
        u16::try_from(parse_immediate(&insn.operands.get(2)?.raw)?).ok()?
    } else {
        if insn.operands.len() != 2 {
            return None;
        }
        0
    };
    Some((WidenKind::ShiftLong { shift }, signed))
}

/// The widening and narrowing family.
///
/// The destination's arrangement gives the geometry; the sources are
/// checked against it rather than trusted, because the whole family is
/// distinguished from the lane-wise one precisely by its operands having
/// *different* arrangements. A `2` form reads sources that span the full
/// register and takes their upper half.
pub(super) fn widen_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let (kind, signed) = widen_kind(base).or_else(|| shift_long_kind(base, insn))?;
    let destination = operand_arrangement(insn.operands.first()?)?;
    let expected_operands = match kind {
        WidenKind::Narrow => 2,
        WidenKind::ShiftLong { .. } => {
            if matches!(base, "uxtl" | "sxtl") {
                2
            } else {
                3
            }
        }
        WidenKind::Arith { .. } => 3,
    };
    if insn.operands.len() != expected_operands {
        return None;
    }
    let narrow_bits = match kind {
        // The destination is the narrow side; the source is wide.
        WidenKind::Narrow => destination.lane_bits.checked_mul(2)?,
        _ => destination.lane_bits / 2,
    };
    if narrow_bits == 0 || !matches!(narrow_bits, 8 | 16 | 32 | 64) {
        return None;
    }
    // Written lanes: for a narrowing `2` form the destination holds
    // twice as many as the instruction produces, the low half surviving.
    let written = match kind {
        WidenKind::Narrow if upper => destination.lanes / 2,
        _ => destination.lanes,
    };
    if written == 0 {
        return None;
    }
    // A narrowing `2` form writes the destination's upper half, so the
    // destination is the full-width operand; a widening one reads its
    // sources' upper half, which the per-operand check below sizes.
    if upper && matches!(kind, WidenKind::Narrow) && !spans_full_register(destination) {
        return None;
    }
    for (index, operand) in insn.operands.iter().enumerate().skip(1) {
        let Some(arrangement) = operand_arrangement(operand) else {
            // The shift is an immediate, not an arrangement.
            if matches!(kind, WidenKind::ShiftLong { .. }) && index == 2 {
                continue;
            }
            return None;
        };
        let wide_operand = matches!(kind, WidenKind::Narrow)
            || matches!(kind, WidenKind::Arith { wide_first: true, .. } if index == 1);
        let expected_bits = if wide_operand {
            match kind {
                WidenKind::Narrow => narrow_bits,
                _ => destination.lane_bits,
            }
        } else {
            narrow_bits
        };
        if arrangement.lane_bits != expected_bits {
            return None;
        }
        // A `2` form's narrow operands span the whole register, so they
        // carry twice the lanes the instruction consumes; a wide operand
        // is never halved.
        let expected_lanes = if upper && !wide_operand {
            written.checked_mul(2)?
        } else {
            written
        };
        if arrangement.lanes != expected_lanes {
            return None;
        }
        // A widening `2` form reads its narrow sources' upper half, so
        // those sources span the whole register.
        if upper && !wide_operand && !spans_full_register(arrangement) {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::Widen {
            kind,
            signed,
            upper,
        },
        lane_bits: destination.lane_bits,
        lanes: written,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== long pairwise addition =====================

/// `saddlp` / `uaddlp` and the accumulating `sadalp` / `uadalp`.
///
/// Adjacent source lanes are extended to twice their width and summed
/// there, so the destination holds half as many lanes of twice the size.
/// Extending first is the whole point: the same sum at the source width
/// would wrap exactly where the instruction exists not to.
pub(super) fn pairwise_long_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (signed, accumulate) = match mnemonic {
        "saddlp" => (true, false),
        "uaddlp" => (false, false),
        "sadalp" => (true, true),
        "uadalp" => (false, true),
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    let source = operand_arrangement(insn.operands.get(1)?)?;
    if source.lane_bits.checked_mul(2)? != destination.lane_bits {
        return None;
    }
    if source.lanes != destination.lanes.checked_mul(2)? {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::PairwiseLong { signed, accumulate },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== high-half narrowing =====================

/// `addhn` / `subhn` and their rounding forms `raddhn` / `rsubhn`.
///
/// Both sources are double-width and the destination keeps only the
/// *high* half of each result, which is what makes the family a
/// narrowing one without any shift being spelled.
pub(super) fn high_narrow_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let (subtract, rounding) = match base {
        "addhn" => (false, false),
        "raddhn" => (false, true),
        "subhn" => (true, false),
        "rsubhn" => (true, true),
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    // The architecture encodes `8H`/`4S`/`2D` sources, so the narrow
    // element is one of three widths and never a 64-bit one.
    if !matches!(destination.lane_bits, 8 | 16 | 32) {
        return None;
    }
    let source_bits = destination.lane_bits.checked_mul(2)?;
    // A `2` form writes the destination's upper half, so the destination
    // holds twice the lanes the instruction produces.
    let written = if upper {
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
    for operand in insn.operands.iter().skip(1) {
        let arrangement = operand_arrangement(operand)?;
        if arrangement.lane_bits != source_bits || arrangement.lanes != written {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::HighNarrow {
            subtract,
            rounding,
            upper,
        },
        lane_bits: destination.lane_bits,
        lanes: written,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== lane-wise conversions =====================

/// The lane-wise conversions.
#[derive(Debug, Clone, Copy)]
pub(in crate::lift) enum ConvertKind {
    /// `scvtf` / `ucvtf` — integer lane to float of the same width.
    IntToFloat { signed: bool },
    /// `fcvtzs` / `fcvtzu` — float lane to integer, rounding toward
    /// zero as the mnemonic's `z` spells.
    FloatToInt { signed: bool },
    /// `fcvtl` / `fcvtn` — between float widths, one lane doubling or
    /// halving in size.
    FloatToFloat { widening: bool },
}

impl ConvertKind {
    /// Whether the mnemonic has a fixed-point form, which carries the
    /// number of fractional bits as a third operand.
    ///
    /// Only the integer conversions do: `fcvtl` / `fcvtn` change the
    /// float format, and there is no fixed point on either side.
    pub(in crate::lift) const fn scales(self) -> bool {
        matches!(self, Self::IntToFloat { .. } | Self::FloatToInt { .. })
    }
}

/// A conversion mnemonic's operation, the mode a float-to-integer member
/// rounds with, and whether the architecture spells a fixed-point form
/// of it.
///
/// The three travel together because they are one decision. Only
/// `fcvtzs` / `fcvtzu` carry a fraction width: the four directed
/// spellings have register forms only, so accepting a third operand for
/// them would model an encoding that does not exist.
fn convert_kind(base: &str) -> Option<(ConvertKind, RoundingMode, bool)> {
    let to_int = |signed, rounding| {
        (
            ConvertKind::FloatToInt { signed },
            rounding,
            matches!(rounding, RoundingMode::TowardZero),
        )
    };
    // The mode a conversion out of float rounds with, spelled by the
    // mnemonic's fourth letter: `z` truncates, `a` rounds ties away, `n`
    // ties to even, `p` toward `+inf` and `m` toward `-inf`. The other
    // two directions round to nearest and ignore it.
    let nearest = RoundingMode::NearestTiesEven;
    Some(match base {
        "scvtf" => (ConvertKind::IntToFloat { signed: true }, nearest, true),
        "ucvtf" => (ConvertKind::IntToFloat { signed: false }, nearest, true),
        "fcvtzs" => to_int(true, RoundingMode::TowardZero),
        "fcvtzu" => to_int(false, RoundingMode::TowardZero),
        "fcvtas" => to_int(true, RoundingMode::NearestTiesAway),
        "fcvtau" => to_int(false, RoundingMode::NearestTiesAway),
        "fcvtns" => to_int(true, RoundingMode::NearestTiesEven),
        "fcvtnu" => to_int(false, RoundingMode::NearestTiesEven),
        "fcvtps" => to_int(true, RoundingMode::TowardPositive),
        "fcvtpu" => to_int(false, RoundingMode::TowardPositive),
        "fcvtms" => to_int(true, RoundingMode::TowardNegative),
        "fcvtmu" => to_int(false, RoundingMode::TowardNegative),
        "fcvtl" => (ConvertKind::FloatToFloat { widening: true }, nearest, false),
        "fcvtn" => (
            ConvertKind::FloatToFloat { widening: false },
            nearest,
            false,
        ),
        _ => return None,
    })
}

/// The lane-wise conversion family.
pub(super) fn convert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let (kind, rounding, scales) = convert_kind(base)?;
    let widths_differ = matches!(kind, ConvertKind::FloatToFloat { .. });
    if upper && !widths_differ {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    // The fixed-point forms carry the fraction width as a third
    // operand. ARM ARM C7.2 bounds it by `1 <= fbits <= esize`, and
    // there is no fixed-point `2` form.
    let fbits = match insn.operands.len() {
        2 => 0,
        3 if scales && !upper => {
            let raw = u16::try_from(parse_immediate(&insn.operands.get(2)?.raw)?).ok()?;
            if raw == 0 || raw > destination.lane_bits {
                return None;
            }
            raw
        }
        _ => return None,
    };
    let source = operand_arrangement(insn.operands.get(1)?)?;
    let (source_bits, written) = match kind {
        ConvertKind::FloatToFloat { widening: true } => {
            // `fcvtl` doubles the element and halves the lane count; the
            // `2` form reads the source's upper half.
            (destination.lane_bits / 2, destination.lanes)
        }
        ConvertKind::FloatToFloat { widening: false } => {
            // `fcvtn` halves the element; `fcvtn2` writes the
            // destination's upper half.
            let written = if upper {
                destination.lanes / 2
            } else {
                destination.lanes
            };
            (destination.lane_bits.checked_mul(2)?, written)
        }
        _ => (destination.lane_bits, destination.lanes),
    };
    if written == 0 || source_bits == 0 {
        return None;
    }
    let expected_lanes = match kind {
        ConvertKind::FloatToFloat { widening: true } if upper => written.checked_mul(2)?,
        _ => written,
    };
    if source.lane_bits != source_bits || source.lanes != expected_lanes {
        return None;
    }
    // `fcvtl2` reads the source's upper half, `fcvtn2` writes the
    // destination's; either way that operand spans the whole register.
    let full_width_side = match kind {
        ConvertKind::FloatToFloat { widening: true } => source,
        _ => destination,
    };
    if upper && !spans_full_register(full_width_side) {
        return None;
    }
    // Every one of these names an IEEE lane on at least one side.
    let float_bits = match kind {
        ConvertKind::FloatToInt { .. } => source_bits,
        _ => destination.lane_bits,
    };
    if !matches!(float_bits, 16 | 32 | 64) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Convert {
            kind,
            upper,
            fbits,
            rounding,
        },
        lane_bits: destination.lane_bits,
        lanes: written,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== across-lane reductions =====================

/// The integer across-lane reductions.
#[derive(Debug, Clone, Copy)]
pub(super) enum ReduceKind {
    /// `addv` — sum every lane at the element width, keeping the low
    /// bits, which is what the architecture's truncation of the
    /// unbounded sum comes to.
    Add,
    /// `uaddlv` / `saddlv` — extend each lane to twice its width and
    /// sum there. The widened sum cannot overflow for any encodable
    /// arrangement, so this is exact rather than merely truncated.
    AddLong { signed: bool },
    /// `smaxv` / `umaxv` / `sminv` / `uminv`.
    MinMax { signed: bool, max: bool },
    /// `fmaxv` / `fminv` — the same fold over IEEE lanes.
    ///
    /// A separate variant, and not `MinMax` with a `float` flag, so
    /// that nothing can reach the integer comparison by accident. The
    /// two are not the same operation with a different sort: ARM's
    /// `FPMax` propagates NaN and combines the signs of a zero tie,
    /// neither of which an integer `slt` can express. `number_wins`
    /// selects `fmaxnmv` / `fminnmv`, whose fold uses `FPMaxNum` /
    /// `FPMinNum` — a quiet NaN loses to a number rather than
    /// propagating.
    Float { max: bool, number_wins: bool },
}

/// The integer across-lane reductions.
///
/// Every other family reads its geometry from operand 0. These cannot:
/// their destination is a *scalar* register (`s0`), which carries no
/// arrangement at all, so the lane count and the source element width
/// are spelled only on operand 1. The destination's own width is then
/// checked against what the reduction produces, rather than assumed —
/// that check is what tells `addv` from `uaddlv` when a caller has
/// mistyped one for the other.
pub(super) fn reduce_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let min_max = |signed, max| ReduceKind::MinMax { signed, max };
    let kind = match mnemonic {
        "addv" => ReduceKind::Add,
        "uaddlv" => ReduceKind::AddLong { signed: false },
        "saddlv" => ReduceKind::AddLong { signed: true },
        "umaxv" => min_max(false, true),
        "smaxv" => min_max(true, true),
        "uminv" => min_max(false, false),
        "sminv" => min_max(true, false),
        "fmaxv" => ReduceKind::Float {
            max: true,
            number_wins: false,
        },
        "fminv" => ReduceKind::Float {
            max: false,
            number_wins: false,
        },
        "fmaxnmv" => ReduceKind::Float {
            max: true,
            number_wins: true,
        },
        "fminnmv" => ReduceKind::Float {
            max: false,
            number_wins: true,
        },
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let source = operand_arrangement(insn.operands.get(1)?)?;
    // A float reduction needs an IEEE lane, which rules out the byte
    // arrangements the integer members admit.
    if matches!(kind, ReduceKind::Float { .. }) && !matches!(source.lane_bits, 16 | 32) {
        return None;
    }
    // ARM ARM C7.2 — the across-lane reductions encode `8B` / `16B`,
    // `4H` / `8H` and `4S` only. A 32-bit element requires the
    // full-width arrangement (`size == 10` implies `Q == 1`) and a
    // 64-bit one has no encoding at all, so a shape outside that set is
    // a spelling the architecture does not produce.
    if !matches!(source.lane_bits, 8 | 16 | 32) {
        return None;
    }
    if source.lane_bits == 32 && !spans_full_register(source) {
        return None;
    }
    let lane_bits = match kind {
        ReduceKind::AddLong { .. } => source.lane_bits.checked_mul(2)?,
        _ => source.lane_bits,
    };
    if scalar_vector_width(insn.operands.first()?)? != lane_bits {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Reduce {
            kind,
            source_lanes: source.lanes,
            source_lane_bits: source.lane_bits,
        },
        lane_bits,
        lanes: 1,
        dest_index: 0,
        source_index: 0,
    })
}

/// Width of a bare scalar view of a vector register (`b0` → 8, `s0` →
/// 32), which is how the reductions spell their destination.
///
/// An arranged or indexed spelling is rejected: those name a lane
/// geometry, and the whole point here is an operand that names none.
///
/// Shared with the scalar pairwise forms next door, which spell their
/// destination the same way and for the same reason.
pub(super) fn scalar_vector_width(op: &Operand) -> Option<u16> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if raw.contains(['.', '[']) {
        return None;
    }
    let layout = register_layout(&raw, Arch::Aarch64)?;
    is_simd_parent(layout.parent, Arch::Aarch64).then(|| layout.width())
}
