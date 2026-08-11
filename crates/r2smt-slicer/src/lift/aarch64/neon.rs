//! `AArch64` NEON (Advanced SIMD) shape resolution.
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
//!
//! This file holds only what the rest of the lifter sees: [`NeonShape`],
//! the [`NeonOp`] it carries, and the [`shape`] dispatch. The family
//! resolvers live in sibling modules — [`arith`] for the operations on
//! whole lanes, [`multiply`] for those built on a lane product,
//! [`width`] for those relating two element geometries, and [`permute`]
//! for those that only move bits. [`geometry`] holds the operand
//! primitives all four share, and `lower` builds the expression a
//! resolved shape describes.

use r2smt_ir::expr::RoundingMode;
use r2smt_ir::program::Instruction;

use super::super::{BinOp, FusedStep, PackedOp};
use crate::lift::simd::CompareKind;
use arith::{AbsDiffKind, BitwiseUnary, PairOp, SaturatingKind, ShiftKind};
use multiply::{AccumulateKind, ByElementKind};
use permute::{PermuteKind, SelectRole};
use width::{ConvertKind, ReduceKind, WidenKind};

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
    /// `suqadd` / `usqadd` — an accumulator and a source read with
    /// *opposite* signednesses, clamped into the destination's range.
    ///
    /// Which is neither operand's range: `suqadd` adds an unsigned
    /// source onto a signed accumulator and saturates signed, `usqadd`
    /// the other way round. Reading both with one signedness, as the
    /// ordinary saturating add does, is a wrong value.
    MixedSignAdd { destination_signed: bool },
    /// `sqdmull` / `sqdmlal` / `sqdmlsl` — the *doubled* product of two
    /// narrow elements, saturated into the destination's element and
    /// optionally accumulated onto it under a second saturation.
    /// `upper` is the `2` suffix, which reads the sources' upper half.
    ///
    /// `by_element` marks the `v2.h[i]` spelling, where the second
    /// source contributes one element — named by
    /// [`NeonShape::source_index`] — to every destination lane instead of
    /// pairing each lane with its own.
    DoublingLong {
        combine: Option<BinOp>,
        upper: bool,
        by_element: bool,
    },
    /// `sri` / `sli` — shift one source lane and insert it over the
    /// destination lane, keeping the destination bits the shift vacated.
    ///
    /// Its own variant rather than a member of [`NeonOp::Shift`] because
    /// the destination is an *input* here and a plain shift's is not:
    /// the bits the shift moves out of the way are the ones that survive.
    ShiftInsert { left: bool, shift: u16 },
    /// A lane-wise compare, writing an all-ones or all-zeros mask per
    /// lane rather than a flag.
    Compare { kind: CompareKind, zero: bool },
    /// A lane-wise conversion between integer and floating point, or
    /// between float widths. `upper` is the `2` suffix, which only the
    /// width-changing forms carry.
    ///
    /// `fbits` is the fixed-point form's fraction width, and zero for
    /// the plain register forms — the integer side is then read as a
    /// scaled value, `Int(lane) / 2^fbits`.
    ///
    /// `rounding` is the mode a float-to-integer member rounds with, and
    /// it is carried rather than fixed because `AArch64` spells five of
    /// them: `fcvtz*` truncates, `fcvta*` rounds ties away, `fcvtn*`
    /// ties to even, `fcvtp*` up and `fcvtm*` down. The other two
    /// directions round to nearest and ignore this.
    Convert {
        kind: ConvertKind,
        upper: bool,
        fbits: u16,
        rounding: RoundingMode,
    },
    /// `fcvtas w0, s1` — one float element converted to an integer in a
    /// **general** register, at the destination's own width.
    ///
    /// A variant of its own rather than a flag on [`NeonOp::Convert`]
    /// because it differs in the two things a lowering is made of: where
    /// the result goes, and how wide it is. Source and destination
    /// decouple here — `fcvtas x0, s1` is a legal encoding — so there is
    /// no single `lane_bits` describing both, and `lane_bits` carries the
    /// *source* the way [`NeonOp::ElementToGpr`] already does.
    FloatToGpr {
        signed: bool,
        rounding: RoundingMode,
    },
    /// `frint<mode>` — round each lane to an integral value without
    /// leaving the float sort.
    ///
    /// Not the integer round trip `fcvtz*` followed by `scvtf`: that
    /// agrees only on the values an integer lane can hold, and turns an
    /// infinity, a NaN or an out-of-range magnitude into some in-range
    /// number.
    RoundToIntegral {
        rounding: RoundingMode,
        /// The `frint32*` / `frint64*` clamp: the signed integer width
        /// the integral result must fit, or `None` for the plain
        /// family.
        ///
        /// Independent of `lane_bits` on purpose — `FEAT_FRINTTS`
        /// encodes all four combinations, because the saturation width
        /// comes from the mnemonic and the float sort from the register.
        saturate: Option<u16>,
    },
    /// `bsl` / `bit` / `bif` — bitwise select, where one of the three
    /// registers supplies the mask and the destination is always one of
    /// the three.
    BitwiseSelect(SelectRole),
    /// An across-lane reduction, folding every source lane into the
    /// single element the scalar destination holds.
    ///
    /// The source geometry is carried here rather than on
    /// [`NeonShape`] because a reduction is the one family whose two
    /// sides genuinely differ: `uaddlv h0, v1.8b` folds eight 8-bit
    /// lanes into one 16-bit element, so neither the lane width nor
    /// the lane count is shared.
    Reduce {
        kind: ReduceKind,
        source_lanes: u16,
        source_lane_bits: u16,
    },
    /// A by-element form: the second source contributes one element,
    /// named by [`NeonShape::source_index`], multiplied into every
    /// destination lane. `upper` is the `2` suffix on the long members.
    ByElement { kind: ByElementKind, upper: bool },
    /// `sdot` / `udot` — four byte products summed into each 32-bit
    /// destination lane, accumulated onto its prior value. `by_element`
    /// marks the `v2.4b[i]` spelling, which broadcasts one four-byte
    /// group (named by [`NeonShape::source_index`]) to every lane instead
    /// of pairing each lane with its own group.
    DotProduct { signed: bool, by_element: bool },
    /// `tbl` / `tbx` — each destination byte selected from the table by
    /// the corresponding index byte. `keep` marks `tbx`, which leaves
    /// the destination byte alone where an out-of-range index makes
    /// `tbl` write zero.
    TableLookup { keep: bool, table_lanes: u16 },
    /// `pmull` / `pmull2` — a carry-less polynomial multiply, whose
    /// product is the `XOR` of the shifted multiplicand over the set
    /// bits of the multiplier. `upper` is the `2` suffix.
    PolynomialMultiply { upper: bool },
    /// `fmla` / `fmls` / `frecps` / `frsqrts` — a fused multiply step
    /// over binary32 lanes, rounded once. `MulAdd` / `MulSub` read the
    /// destination as an accumulator.
    FusedStep(FusedStep),
    /// `cnt` / `clz` / `cls` / `rbit` — a function of a lane's
    /// individual bits rather than of its value.
    BitwiseUnary(BitwiseUnary),
    /// A lane-wise fold of two lanes at the same index, for the members
    /// whose lane operation the lane-wise [`NeonOp::Packed`] family
    /// cannot spell — today the four floating-point selects, whose ARM
    /// semantics differ from the Intel ones [`PackedOp::Fp`] carries.
    LaneCombine(PairOp),
    /// `addp` / `smaxp` / `faddp` / … — the same fold applied to
    /// *adjacent* lanes of the concatenated sources rather than to the
    /// lanes at one index.
    Pairwise(PairOp),
    /// `addp d0, v1.2d` / `faddp s0, v1.2s` — the same fold applied to
    /// the *one* source's two lanes, producing a single element.
    ///
    /// Its own variant rather than [`NeonOp::Pairwise`] with one lane,
    /// because there is no second source: the vector form's lowering
    /// splits the destination between two operands, and at one lane it
    /// would take the pair out of an operand that is not there.
    ScalarPairwise(PairOp),
    /// `sabd` / `uabd` / `saba` / `uaba` / `fabd` — the magnitude of the
    /// lane difference, optionally accumulated onto the destination.
    AbsoluteDifference(AbsDiffKind),
    /// `saddlp` / `uaddlp` / `sadalp` / `uadalp` — adjacent source lanes
    /// summed into a destination element twice their width, optionally
    /// accumulated onto the destination's prior value.
    PairwiseLong { signed: bool, accumulate: bool },
    /// `addhn` / `subhn` and their rounding forms — the high half of a
    /// double-width sum or difference. `upper` is the `2` suffix, which
    /// writes the destination's top half.
    HighNarrow {
        subtract: bool,
        rounding: bool,
        upper: bool,
    },
    /// `frecpe` / `frsqrte` — the reciprocal and reciprocal-square-root
    /// estimates, whose result is a free value.
    ///
    /// Both mnemonics share one variant because they share one
    /// lowering: the architecture fixes only a relative error bound and
    /// leaves the value itself implementation-defined, so there is
    /// nothing to tell them apart *with*.
    Estimate,
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
        match self.op {
            NeonOp::Insert { .. }
            | NeonOp::MultiplyAccumulate { .. }
            | NeonOp::BitwiseSelect(_)
            | NeonOp::DotProduct { .. }
            | NeonOp::Widen {
                kind: WidenKind::Narrow,
                upper: true,
                ..
            }
            | NeonOp::AbsoluteDifference(AbsDiffKind::Integer {
                accumulate: true, ..
            })
            | NeonOp::PairwiseLong {
                accumulate: true, ..
            }
            | NeonOp::HighNarrow { upper: true, .. }
            // The saturating narrowing `2` forms (`sqxtn2`, `shrn2`,
            // `sqshrn2`, …) and the narrowing float convert `fcvtn2`
            // write only the destination's upper half and preserve its
            // lower half, so their lowering reads the destination. The
            // widening `fcvtl2` does not, hence the `widening: false`.
            | NeonOp::Saturating {
                kind: SaturatingKind::Narrow { .. } | SaturatingKind::ShiftNarrow { .. },
                upper: true,
                ..
            }
            | NeonOp::Convert {
                kind: ConvertKind::FloatToFloat { widening: false },
                upper: true,
                ..
            }
            | NeonOp::ShiftInsert { .. }
            | NeonOp::MixedSignAdd { .. }
            | NeonOp::DoublingLong {
                combine: Some(_), ..
            } => true,
            NeonOp::ByElement { kind, .. } => kind.combines(),
            NeonOp::FusedStep(step) => step.reads_accumulator(),
            // `tbx` preserves the destination byte for an out-of-range
            // index; `tbl` writes zero and reads nothing.
            NeonOp::TableLookup { keep, .. } => keep,
            _ => false,
        }
    }

    /// Whether the destination is a general-purpose register rather than
    /// a vector one.
    const fn writes_gpr(&self) -> bool {
        matches!(
            self.op,
            NeonOp::ElementToGpr { .. } | NeonOp::FloatToGpr { .. }
        )
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
    arith::packed_shape(insn, &mnemonic)
        .or_else(|| arith::bitwise_unary_shape(insn, &mnemonic))
        .or_else(|| arith::round_to_integral_shape(insn, &mnemonic))
        .or_else(|| arith::float_min_max_shape(insn, &mnemonic))
        .or_else(|| arith::pairwise_shape(insn, &mnemonic))
        .or_else(|| arith::scalar_pairwise_shape(insn, &mnemonic))
        .or_else(|| arith::absolute_difference_shape(insn, &mnemonic))
        .or_else(|| width::pairwise_long_shape(insn, &mnemonic))
        .or_else(|| width::high_narrow_shape(insn, &mnemonic))
        .or_else(|| permute::immediate_shape(insn, &mnemonic))
        .or_else(|| permute::duplicate_shape(insn, &mnemonic))
        .or_else(|| permute::extract_shape(insn, &mnemonic))
        .or_else(|| permute::permute_shape(insn, &mnemonic))
        .or_else(|| permute::element_to_gpr_shape(insn, &mnemonic))
        .or_else(|| permute::insert_shape(insn, &mnemonic))
        .or_else(|| width::widen_shape(insn, &mnemonic))
        .or_else(|| multiply::multiply_accumulate_shape(insn, &mnemonic))
        .or_else(|| arith::saturating_shape(insn, &mnemonic))
        .or_else(|| arith::saturating_scalar_shape(insn, &mnemonic))
        .or_else(|| arith::mixed_sign_add_shape(insn, &mnemonic))
        .or_else(|| arith::doubling_long_shape(insn, &mnemonic))
        .or_else(|| arith::shift_shape(insn, &mnemonic))
        .or_else(|| arith::shift_insert_shape(insn, &mnemonic))
        .or_else(|| arith::compare_shape(insn, &mnemonic))
        .or_else(|| width::convert_shape(insn, &mnemonic))
        .or_else(|| width::convert_scalar_shape(insn, &mnemonic))
        .or_else(|| width::convert_to_gpr_shape(insn, &mnemonic))
        .or_else(|| permute::bitwise_select_shape(insn, &mnemonic))
        .or_else(|| width::reduce_shape(insn, &mnemonic))
        .or_else(|| multiply::by_element_shape(insn, &mnemonic))
        .or_else(|| multiply::dot_product_shape(insn, &mnemonic))
        .or_else(|| permute::table_lookup_shape(insn, &mnemonic))
        .or_else(|| multiply::polynomial_multiply_shape(insn, &mnemonic))
        .or_else(|| multiply::fused_step_shape(insn, &mnemonic))
        .or_else(|| arith::estimate_shape(insn, &mnemonic))
}

pub(in crate::lift) use arith::round_to_integral_kind;

mod arith;
mod geometry;
pub(in crate::lift) mod lower;
mod multiply;
mod permute;
pub(crate) mod structured;
pub(in crate::lift) mod width;
