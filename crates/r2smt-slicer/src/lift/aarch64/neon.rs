//! `AArch64` NEON (Advanced SIMD) shape resolution and lowering.
//!
//! Every NEON instruction the lifter models is resolved here, by one
//! function, into a [`NeonShape`] that fully describes the lowering. The
//! effect table and the per-mnemonic dispatcher both consult that
//! resolver, which is what keeps them from disagreeing about which
//! instructions the slicer may retain: an instruction kept because its
//! destination is a definition, but whose lowering is then dropped,
//! would leave a later read bound to a stale value.
//!
//! Resolution is deliberately strict about operand shape. NEON reuses
//! the scalar mnemonics (`add`, `fadd`, `mov`), so a shape that does not
//! match the family exactly must fall through to the next family and, in
//! the end, decline — never be guessed at.

use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::registers::{
    Arrangement, element_type_bits, is_simd_parent, parse_arrangement, parse_lane_index,
    register_layout,
};

use super::super::{BinOp, FpArithOp, PackedIntOp, PackedOp, parse_immediate};

/// Number of bits in a byte, used wherever an operand counts bytes
/// (`ext`'s index) but the model counts bits.
const BITS_PER_BYTE: u16 = 8;

/// Which of an instruction's two vector sources a permuted lane comes
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermuteSource {
    First,
    Second,
}

/// A lane permutation, expressed as the source each destination lane
/// draws from. Every member of the family is a pure rearrangement of
/// existing lanes, so one representation covers them all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermuteKind {
    /// `zip1` / `zip2` — interleave the lower or upper halves.
    Zip { upper: bool },
    /// `uzp1` / `uzp2` — deinterleave the even or odd lanes.
    Uzp { odd: bool },
    /// `trn1` / `trn2` — transpose the even or odd lanes.
    Trn { odd: bool },
    /// `rev16` / `rev32` / `rev64` — reverse the element order within
    /// each container of this many bits.
    Reverse { container_bits: u16 },
}

/// What a resolved NEON instruction computes.
#[derive(Debug, Clone, Copy)]
enum NeonOp {
    /// Lane-wise arithmetic or logic over identically shaped operands.
    Packed(PackedOp),
    /// `movi` / `mvni` — replicate an immediate to every lane,
    /// optionally inverted.
    Immediate { value: u64, invert: bool },
    /// `dup` — replicate one source value to every lane. The source is
    /// a general-purpose register, or an element named by
    /// [`NeonShape::source_index`].
    Duplicate { from_element: bool },
    /// `ext` — extract a window from the two sources concatenated, at a
    /// byte offset.
    Extract { byte_offset: u16 },
    /// A pure lane permutation over one or two sources.
    Permute(PermuteKind),
    /// `umov` / `smov` — move one element into a general-purpose
    /// register, zero- or sign-extended.
    ElementToGpr { signed: bool },
    /// `ins` — insert a general-purpose register or another vector's
    /// element into one lane of the destination.
    Insert { from_element: bool },
    /// A widening or narrowing element operation. `signed` selects the
    /// extension; `upper` is the `2` suffix, which reads the top half of
    /// its sources and, when narrowing, writes the top half of its
    /// destination.
    Widen {
        kind: WidenKind,
        signed: bool,
        upper: bool,
    },
    /// `mla` / `mls` and the long `umlal` / `smlal` / `umlsl` / `smlsl`:
    /// multiply the two sources and accumulate the product into the
    /// destination.
    MultiplyAccumulate(AccumulateKind),
    /// A saturating or rounding element operation. `signed_sources`
    /// selects how the narrow elements are extended into the width the
    /// operation is computed at; `upper` is the `2` suffix.
    Saturating {
        kind: SaturatingKind,
        signed_sources: bool,
        upper: bool,
    },
    /// A same-width shift, by an immediate or by a per-lane amount.
    Shift { kind: ShiftKind, signed: bool },
    /// A lane-wise compare, writing an all-ones or all-zeros mask per
    /// lane rather than a flag.
    Compare { kind: CompareKind, zero: bool },
    /// A lane-wise conversion between integer and floating point, or
    /// between float widths. `upper` is the `2` suffix, which only the
    /// width-changing forms carry.
    Convert { kind: ConvertKind, upper: bool },
    /// `bsl` / `bit` / `bif` — bitwise select, where one of the three
    /// registers supplies the mask and the destination is always one of
    /// the three.
    BitwiseSelect(SelectRole),
}

/// Which role the destination register plays in a bitwise select.
///
/// All three mnemonics compute `(a & mask) | (b & ~mask)`; they differ
/// only in which operand is the mask and which is selected when the mask
/// bit is set. Every one of them reads the destination.
#[derive(Debug, Clone, Copy)]
enum SelectRole {
    /// `bsl` — the destination is the mask; the sources supply the two
    /// candidate values.
    DestinationIsMask,
    /// `bit` — insert the first source where the second's bits are set,
    /// keeping the destination elsewhere.
    InsertWhereSet,
    /// `bif` — insert the first source where the second's bits are
    /// *clear*.
    InsertWhereClear,
}

/// The lane-wise compares.
///
/// Each writes a mask, not a condition flag: the whole point is that the
/// result feeds another vector operation.
#[derive(Debug, Clone, Copy)]
enum CompareKind {
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

/// The lane-wise conversions.
#[derive(Debug, Clone, Copy)]
enum ConvertKind {
    /// `scvtf` / `ucvtf` — integer lane to float of the same width.
    IntToFloat { signed: bool },
    /// `fcvtzs` / `fcvtzu` — float lane to integer, rounding toward
    /// zero as the mnemonic's `z` spells.
    FloatToInt { signed: bool },
    /// `fcvtl` / `fcvtn` — between float widths, one lane doubling or
    /// halving in size.
    FloatToFloat { widening: bool },
}

/// The same-width shift operations.
///
/// The immediate forms and the register forms are genuinely different
/// shapes, not one shape with a different operand: an immediate shift
/// names its direction in the mnemonic, while `sshl` and `ushl` take a
/// *signed* per-lane amount whose sign chooses the direction at run
/// time.
#[derive(Debug, Clone, Copy)]
enum ShiftKind {
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

/// How a multiply-accumulate reads its source elements.
///
/// A separate type rather than a `widen` flag beside a `signed` one,
/// because signedness is meaningless without widening: `mla` extends
/// nothing, so there is no extension to choose.
#[derive(Debug, Clone, Copy)]
enum AccumulateSources {
    /// `mla` / `mls` — sources are already the destination's element
    /// width.
    SameWidth,
    /// `umlal` / `smlal` / `umlsl` / `smlsl` — sources are half the
    /// destination's element width and are extended before the multiply.
    Long { signed: bool },
}

/// How a multiply-accumulate reads its sources and combines the product.
#[derive(Debug, Clone, Copy)]
struct AccumulateKind {
    /// Whether the product is added to or subtracted from the
    /// accumulator.
    combine: BinOp,
    sources: AccumulateSources,
    /// The `2` suffix — read the sources' upper half.
    upper: bool,
}

/// Which range a computed value is clamped into.
///
/// The distinction that matters is not the operation's signedness but
/// the *result's*: `uqsub` computes a value that can go negative and
/// clamps it into the unsigned range, which is neither of the two
/// obvious cases.
#[derive(Debug, Clone, Copy)]
enum SaturateTo {
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
enum SaturatingKind {
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

/// The widening and narrowing element operations.
#[derive(Debug, Clone, Copy)]
enum WidenKind {
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

/// A resolved NEON instruction: what to compute, and at what geometry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NeonShape {
    op: NeonOp,
    /// Element width of the destination, in bits.
    lane_bits: u16,
    /// Number of destination lanes. One for the forms whose destination
    /// is a single element or a general-purpose register.
    lanes: u16,
    /// Element index addressed in the destination, for the inserts.
    dest_index: u16,
    /// Element index addressed in the source, for the element-reading
    /// forms.
    source_index: u16,
}

impl NeonShape {
    /// Whether the lowering reads the destination register as well as
    /// writing it.
    ///
    /// True for the element inserts, which preserve every lane they do
    /// not write; for the accumulators and the bitwise selects, whose
    /// destination is an input; and for the narrowing `2` forms, which
    /// write only the destination's upper half. Everything else here writes the
    /// destination whole — an `AArch64` SIMD write has no merging form,
    /// so even a 64-bit arrangement replaces the register by zeroing its
    /// upper half.
    pub(crate) const fn reads_destination(&self) -> bool {
        matches!(
            self.op,
            NeonOp::Insert { .. }
                | NeonOp::MultiplyAccumulate { .. }
                | NeonOp::BitwiseSelect(_)
                | NeonOp::Widen {
                    kind: WidenKind::Narrow,
                    upper: true,
                    ..
                }
        )
    }

    /// Whether the destination is a general-purpose register rather than
    /// a vector one.
    const fn writes_gpr(&self) -> bool {
        matches!(self.op, NeonOp::ElementToGpr { .. })
    }
}

/// Resolve `insn` into the NEON lowering that models it, or `None` when
/// none does.
///
/// The families are tried in order of how tightly they constrain their
/// operands, so a mnemonic shared between two of them (`mov` is both a
/// lane-wise copy and an element insert) reaches the one whose operand
/// shape it actually matches.
pub(crate) fn shape(insn: &Instruction) -> Option<NeonShape> {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    packed_shape_of(insn, &mnemonic)
        .or_else(|| immediate_shape(insn, &mnemonic))
        .or_else(|| duplicate_shape(insn, &mnemonic))
        .or_else(|| extract_shape(insn, &mnemonic))
        .or_else(|| permute_shape(insn, &mnemonic))
        .or_else(|| element_to_gpr_shape(insn, &mnemonic))
        .or_else(|| insert_shape(insn, &mnemonic))
        .or_else(|| widen_shape(insn, &mnemonic))
        .or_else(|| multiply_accumulate_shape(insn, &mnemonic))
        .or_else(|| saturating_shape(insn, &mnemonic))
        .or_else(|| shift_shape(insn, &mnemonic))
        .or_else(|| compare_shape(insn, &mnemonic))
        .or_else(|| convert_shape(insn, &mnemonic))
        .or_else(|| bitwise_select_shape(insn, &mnemonic))
}

// ===================== bitwise select =====================

/// `bsl` / `bit` / `bif`, the three-register bitwise selects.
///
/// All three read the destination, and the architecture spells them only
/// with the byte arrangements — the operation is bit-granular, so the
/// element width carries no meaning beyond the view's total size.
fn bitwise_select_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let role = match mnemonic {
        "bsl" => SelectRole::DestinationIsMask,
        "bit" => SelectRole::InsertWhereSet,
        "bif" => SelectRole::InsertWhereClear,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if destination.lane_bits != BITS_PER_BYTE {
        return None;
    }
    for operand in insn.operands.iter().skip(1) {
        if operand_arrangement(operand)? != destination {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::BitwiseSelect(role),
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
fn compare_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
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

// ===================== lane-wise conversions =====================

/// The lane-wise conversion family.
fn convert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let kind = match base {
        "scvtf" => ConvertKind::IntToFloat { signed: true },
        "ucvtf" => ConvertKind::IntToFloat { signed: false },
        "fcvtzs" => ConvertKind::FloatToInt { signed: true },
        "fcvtzu" => ConvertKind::FloatToInt { signed: false },
        "fcvtl" => ConvertKind::FloatToFloat { widening: true },
        "fcvtn" => ConvertKind::FloatToFloat { widening: false },
        _ => return None,
    };
    let widths_differ = matches!(kind, ConvertKind::FloatToFloat { .. });
    if upper && !widths_differ {
        return None;
    }
    // The two-operand register forms only; the fixed-point variants
    // carry a third immediate operand and scale by it.
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
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
        op: NeonOp::Convert { kind, upper },
        lane_bits: destination.lane_bits,
        lanes: written,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== same-width shifts =====================

/// The same-width shift family.
///
/// The immediate forms carry their amount as an operand and their
/// direction in the mnemonic. The register forms carry a whole vector of
/// per-lane amounts, each read as a *signed* value whose sign chooses
/// the direction — so one lane of a `sshl` can shift left while its
/// neighbour shifts right.
fn shift_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
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

// ===================== saturation and rounding =====================

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
fn saturating_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
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

// ===================== multiply-accumulate =====================

/// The multiply-accumulate family.
///
/// Same-width (`mla` / `mls`) and long (`umlal` / `smlal` / `umlsl` /
/// `smlsl`) share one lowering: multiply the sources, then add or
/// subtract the product against the destination's existing lane. The
/// long forms multiply at the destination's width, so the product cannot
/// overflow the element the way a same-width multiply can.
///
/// Every member reads its destination. That is the whole point of an
/// accumulator, and it is what the effect table has to be told, or the
/// slicer will drop the definition the accumulation builds on.
fn multiply_accumulate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let long = |signed| AccumulateSources::Long { signed };
    let (combine, sources) = match base {
        "mla" => (BinOp::Add, AccumulateSources::SameWidth),
        "mls" => (BinOp::Sub, AccumulateSources::SameWidth),
        "umlal" => (BinOp::Add, long(false)),
        "smlal" => (BinOp::Add, long(true)),
        "umlsl" => (BinOp::Sub, long(false)),
        "smlsl" => (BinOp::Sub, long(true)),
        _ => return None,
    };
    let widen = matches!(sources, AccumulateSources::Long { .. });
    // The same-width forms have no `2` variant.
    if upper && !widen {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    let source_bits = if widen {
        destination.lane_bits / 2
    } else {
        destination.lane_bits
    };
    if source_bits == 0 {
        return None;
    }
    let expected_lanes = if upper {
        destination.lanes.checked_mul(2)?
    } else {
        destination.lanes
    };
    for operand in insn.operands.iter().skip(1) {
        let arrangement = operand_arrangement(operand)?;
        if arrangement.lane_bits != source_bits || arrangement.lanes != expected_lanes {
            return None;
        }
        if upper && !spans_full_register(arrangement) {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::MultiplyAccumulate(AccumulateKind {
            combine,
            sources,
            upper,
        }),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== widening and narrowing =====================

/// Whether a `2`-form's full-register operand really spans 128 bits.
///
/// The `2` suffix means "the half of the 128-bit register the base form
/// does not touch", so the operand it halves must be a full-width
/// arrangement. Without this an arrangement half that size would resolve
/// to a shape the architecture does not encode.
const fn spans_full_register(arrangement: Arrangement) -> bool {
    arrangement.view_bits() == SIMD_REGISTER_BITS
}

/// Width of an `AArch64` vector register.
const SIMD_REGISTER_BITS: u16 = 128;

/// Peel the `2` suffix that marks the half-register forms.
fn peel_upper(mnemonic: &str) -> (&str, bool) {
    mnemonic
        .strip_suffix('2')
        .map_or((mnemonic, false), |base| (base, true))
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
fn widen_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
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
fn packed_shape_of(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
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

// ===================== broadcast and permutation =====================

/// `movi vd.<T>, #imm{, lsl #shift}` and `mvni`, which replicate an
/// immediate across every lane.
///
/// The disassembler prints the *decoded* immediate, so the printed value
/// is the per-lane one and needs no re-expansion — including for `.2d`,
/// whose encoded form is a byte mask. An `msl` shift is a different
/// operation (it shifts ones in, not zeroes) and declines rather than
/// being treated as `lsl`.
fn immediate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let invert = match mnemonic {
        "movi" => false,
        "mvni" => true,
        _ => return None,
    };
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    let raw = parse_immediate(&insn.operands.get(1)?.raw)?;
    let shift = match insn.operands.len() {
        2 => 0,
        3 => shift_amount(insn.operands.get(2)?)?,
        _ => return None,
    };
    let value = raw.checked_shl(u32::from(shift))?;
    Some(NeonShape {
        op: NeonOp::Immediate { value, invert },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The `lsl #n` shift operand of a `movi`. Radare2 renders the shifting
/// modifier as one operand, so the whole thing is parsed here.
fn shift_amount(op: &Operand) -> Option<u16> {
    let raw = op.raw.trim().to_ascii_lowercase();
    let body = raw.strip_prefix("lsl")?.trim();
    u16::try_from(parse_immediate(body)?).ok()
}

/// `dup vd.<T>, rn` and `dup vd.<T>, vn.<Ts>[index]`.
///
/// The scalar-destination form (`dup d0, v1.d[1]`) declines: its
/// destination carries no arrangement, so the lane count is not spelled.
fn duplicate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if mnemonic != "dup" || insn.operands.len() != 2 {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    let source = insn.operands.get(1)?;
    let (from_element, source_index) = if let Some((element_bits, index)) = indexed_element(source)
    {
        if element_bits != arrangement.lane_bits {
            return None;
        }
        (true, index)
    } else {
        // A general-purpose source supplies the low `lane_bits`.
        if !is_general_register(source) {
            return None;
        }
        (false, 0)
    };
    Some(NeonShape {
        op: NeonOp::Duplicate { from_element },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index,
    })
}

/// `ext vd.<T>, vn.<T>, vm.<T>, #index` — a byte-granular window over
/// the two sources concatenated, with `vn` at the low end.
///
/// Only the byte arrangements are architecturally valid, and the index
/// must leave a whole view inside the concatenation.
fn extract_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if mnemonic != "ext" || insn.operands.len() != 4 {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    if arrangement.lane_bits != BITS_PER_BYTE {
        return None;
    }
    for index in 1..=2 {
        if operand_arrangement(insn.operands.get(index)?)? != arrangement {
            return None;
        }
    }
    let byte_offset = u16::try_from(parse_immediate(&insn.operands.get(3)?.raw)?).ok()?;
    if byte_offset >= arrangement.lanes {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Extract { byte_offset },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The permutation family: `zip`, `uzp`, `trn` over two sources, and
/// `rev` over one.
fn permute_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let kind = match mnemonic {
        "zip1" => PermuteKind::Zip { upper: false },
        "zip2" => PermuteKind::Zip { upper: true },
        "uzp1" => PermuteKind::Uzp { odd: false },
        "uzp2" => PermuteKind::Uzp { odd: true },
        "trn1" => PermuteKind::Trn { odd: false },
        "trn2" => PermuteKind::Trn { odd: true },
        "rev16" => PermuteKind::Reverse { container_bits: 16 },
        "rev32" => PermuteKind::Reverse { container_bits: 32 },
        "rev64" => PermuteKind::Reverse { container_bits: 64 },
        _ => return None,
    };
    let operand_count = if matches!(kind, PermuteKind::Reverse { .. }) {
        2
    } else {
        3
    };
    if insn.operands.len() != operand_count {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    for index in 1..operand_count {
        if operand_arrangement(insn.operands.get(index)?)? != arrangement {
            return None;
        }
    }
    // A reversal needs whole elements inside each container, and at
    // least two of them for the reversal to mean anything.
    if let PermuteKind::Reverse { container_bits } = kind {
        if container_bits <= arrangement.lane_bits
            || container_bits % arrangement.lane_bits != 0
            || arrangement.view_bits() % container_bits != 0
        {
            return None;
        }
    } else if arrangement.lanes % 2 != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Permute(kind),
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// `umov wd, vn.<Ts>[index]` and `smov`, which move one element into a
/// general-purpose register.
fn element_to_gpr_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let signed = match mnemonic {
        // `mov wd, vn.s[0]` is the assembler alias for `umov`.
        "umov" | "mov" => false,
        "smov" => true,
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = insn.operands.first()?;
    if !is_general_register(destination) {
        return None;
    }
    let (lane_bits, source_index) = indexed_element(insn.operands.get(1)?)?;
    Some(NeonShape {
        op: NeonOp::ElementToGpr { signed },
        lane_bits,
        lanes: 1,
        dest_index: 0,
        source_index,
    })
}

/// `ins vd.<Ts>[index], rn` and `ins vd.<Ts>[index], vn.<Ts>[index2]`,
/// which write one lane and preserve the rest.
fn insert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    // `mov v0.s[1], w0` is the assembler alias for `ins`.
    if !matches!(mnemonic, "ins" | "mov") || insn.operands.len() != 2 {
        return None;
    }
    let (lane_bits, dest_index) = indexed_element(insn.operands.first()?)?;
    let source = insn.operands.get(1)?;
    let (from_element, source_index) = if let Some((source_bits, index)) = indexed_element(source) {
        if source_bits != lane_bits {
            return None;
        }
        (true, index)
    } else {
        if !is_general_register(source) {
            return None;
        }
        (false, 0)
    };
    Some(NeonShape {
        op: NeonOp::Insert { from_element },
        lane_bits,
        lanes: 1,
        dest_index,
        source_index,
    })
}

// ===================== operand shape helpers =====================

/// The arrangement an operand carries, if it is a vector register that
/// carries one.
fn operand_arrangement(op: &Operand) -> Option<Arrangement> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if !names_vector_register(&raw) {
        return None;
    }
    let (_, body) = raw.split_once('.')?;
    parse_arrangement(body)
}

/// The `(element_bits, index)` an indexed operand names: `v0.s[1]` is a
/// 32-bit element at index 1.
fn indexed_element(op: &Operand) -> Option<(u16, u16)> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if !names_vector_register(&raw) {
        return None;
    }
    let (_, body) = raw.split_once('.')?;
    let mut chars = body.chars();
    let element_bits = element_type_bits(chars.next()?)?;
    let index = parse_lane_index(chars.as_str())?;
    // The element has to fit inside the 128-bit register.
    let lanes = crate::registers::simd_parent_bits(Arch::Aarch64)? / element_bits;
    (index < lanes).then_some((element_bits, index))
}

/// Whether an operand names a general-purpose register.
fn is_general_register(op: &Operand) -> bool {
    op.kind == OperandKind::Register
        && register_layout(&op.raw, Arch::Aarch64)
            .is_some_and(|layout| !is_simd_parent(layout.parent, Arch::Aarch64))
}

/// Whether the operand text names a vector register (before any shape
/// suffix is considered).
fn names_vector_register(raw: &str) -> bool {
    register_layout(raw, Arch::Aarch64)
        .is_some_and(|layout| is_simd_parent(layout.parent, Arch::Aarch64))
}

mod lower;
pub(crate) mod structured;
