//! `AArch32` NEON forms the element-typed mnemonic dispatch cannot
//! reach.
//!
//! [`super::neon_packed_op`] resolves a lane-wise operation from the
//! mnemonic alone, and that works because every family it covers shares
//! one geometry across the whole instruction: the element type says how
//! wide a lane is, the destination register says how many there are,
//! and every source matches. The families here break that assumption in
//! one of three ways — a permutation spells its element as a bare width
//! the element-type parser rejects (`vzip.8`), a widening form relates
//! two different geometries, and the by-element and structured forms
//! carry shape on the operand rather than the mnemonic.
//!
//! Each gets a resolver returning one [`NeonShape`], and [`resolve`] is
//! consulted by the effect table and the per-mnemonic dispatcher alike.
//! That is what keeps them from disagreeing: an instruction the slicer
//! retains because its destination is a definition, but whose lowering
//! is then dropped, would leave a later read bound to a stale value.

use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::registers::{has_vector_arrangement, is_simd_parent, register_layout};

mod accumulate;
mod element;
mod estimate;
mod lanewise;
pub(super) mod lower;
mod permute;
mod select;
pub(super) mod structured;
mod table;
mod width;

/// Bits in a byte, named because `vext` measures its window in them.
pub(super) const BITS_PER_BYTE: u16 = 8;

/// What a resolved `AArch32` NEON instruction computes.
// `PartialEq` is deliberately absent: `WidenKind` carries a `BinOp`,
// which has no equality, and nothing here compares shapes.
#[derive(Debug, Clone, Copy)]
pub(super) enum NeonOp {
    /// `vext` — the window of the two sources concatenated that starts
    /// at this byte offset.
    Extract { byte_offset: u16 },
    /// `vrev16` / `vrev32` / `vrev64` — reverse the element order
    /// inside each container of this many bits.
    Reverse { container_bits: u16 },
    /// `vdup` — replicate a general-purpose register's low element to
    /// every lane.
    Duplicate,
    /// `vzip` / `vuzp` / `vtrn` — a permutation over the two named
    /// registers that writes **both** of them.
    ///
    /// `AArch64` has no counterpart: its `zip1` / `zip2` are two
    /// instructions each producing one destination, so a shape carrying
    /// a single destination describes them. Here one instruction
    /// rewrites both operands, and the effect table has to say so or
    /// the slicer drops whatever defined the second one.
    PermutePair(permute::PairKind),
    /// A widening or narrowing element operation. `signed` selects how
    /// a narrow element is extended into the width the operation is
    /// computed at, and is meaningless for the narrowing members, which
    /// discard the high half whatever the sign.
    Widen {
        kind: width::WidenKind,
        signed: bool,
    },
    /// A by-element product: the second source contributes the single
    /// element `index` names to every destination lane, rather than
    /// pairing lane with lane.
    ByElement {
        kind: element::ByElementKind,
        index: u16,
    },
    /// `vtbl` / `vtbx` — each destination byte selected from a table of
    /// `table_lanes` bytes. `keep` is `vtbx`, which preserves the
    /// destination byte for an out-of-range index where `vtbl` writes
    /// zero.
    TableLookup { keep: bool, table_lanes: u16 },
    /// `vceq` / `vcgt` / `vcge` / `vtst` — a lane-wise compare whose
    /// result is an all-ones / all-zeros mask. `zero` is the
    /// compare-against-`#0` form.
    Compare {
        kind: lanewise::CompareKind,
        zero: bool,
    },
    /// `vcvt` — a lane-wise conversion between the 32-bit float and
    /// integer formats, optionally by a fixed-point fraction width.
    ///
    /// Shares [`crate::lift::aarch64::neon::width::ConvertKind`] and
    /// the `AArch64` lowering rather than restating them: the two ISAs
    /// differ in how they *spell* the conversion, not in what it
    /// computes.
    Convert {
        kind: crate::lift::aarch64::neon::width::ConvertKind,
        /// Fraction width of the fixed-point form, zero for the plain
        /// register form.
        fbits: u16,
        /// The mode a float-to-integer conversion rounds with.
        rounding: r2smt_ir::expr::RoundingMode,
    },
    /// `vpadd` / `vpmax` / `vpmin` — a pairwise reduction of adjacent
    /// source lanes. `float` selects the `.f32` forms, whose `vpmax` /
    /// `vpmin` follow ARM `FPMax` NaN / signed-zero semantics.
    Pairwise {
        op: lanewise::PairwiseOp,
        signed: bool,
        float: bool,
    },
    /// `vcnt` / `vclz` — a per-element bit count. `leading` is `vclz`
    /// (count leading zeros) against `vcnt` (population count).
    CountBits { leading: bool },
    /// `vrecpe` / `vrsqrte` — the reciprocal and reciprocal-square-root
    /// estimates, whose result is a free value: the architecture fixes
    /// only an error bound, so there is no definite value to compute.
    Estimate,
    /// `vbsl` / `vbit` / `vbif` — a bit-granular select over the whole
    /// view, where the role carries which register supplies the mask.
    BitwiseSelect(select::SelectRole),
    /// `vsli` / `vsri` — the source element shifted by `shift` and
    /// inserted over the destination, which keeps the bits the shift
    /// vacated.
    ShiftInsert { left: bool, shift: u16 },
    /// `vabd` / `vaba` — the magnitude of the lane-wise difference,
    /// added onto the destination when `accumulate` is set.
    AbsoluteDifference { signed: bool, accumulate: bool },
    /// `vpaddl` / `vpadal` — adjacent source elements summed into one
    /// destination element twice as wide, added onto the destination
    /// when `accumulate` is set.
    PairwiseLong { signed: bool, accumulate: bool },
}

/// A resolved `AArch32` NEON instruction: what to compute, and at what
/// geometry.
///
/// Only destination geometry, as on `AArch64`. Every family here reads
/// its sources at the destination's width, and the families that do not
/// — the widening forms and the reductions — will carry the difference
/// on their [`NeonOp`] variant when they land, the way `AArch64` does.
#[derive(Debug, Clone, Copy)]
pub(super) struct NeonShape {
    pub(super) op: NeonOp,
    /// Element width of the destination, in bits.
    pub(super) lane_bits: u16,
    /// Number of destination lanes.
    pub(super) lanes: u16,
}

impl NeonShape {
    /// Width of the destination's view — the whole `d` or `q` register
    /// the instruction covers.
    pub(super) const fn view_bits(self) -> Option<u16> {
        self.lane_bits.checked_mul(self.lanes)
    }

    /// Whether the lowering writes the second operand as well as the
    /// first.
    pub(super) const fn writes_operand_pair(self) -> bool {
        matches!(self.op, NeonOp::PermutePair(_))
    }
}

/// Whether `insn` is a NEON form that writes both of its named
/// registers, so the effect table records two definitions.
pub(super) fn writes_operand_pair(insn: &Instruction) -> bool {
    resolve(insn).is_some_and(NeonShape::writes_operand_pair)
}

/// Resolve `insn` into the NEON form it names, or `None` when no family
/// here claims it.
///
/// Ordered the way `AArch64`'s dispatch is: the resolvers that
/// constrain their operands most tightly go first, so a form that
/// matches several mnemonically is claimed by the one whose operand
/// shape it actually fits.
pub(super) fn resolve(insn: &Instruction) -> Option<NeonShape> {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    permute::extract_shape(insn, &mnemonic)
        .or_else(|| permute::reverse_shape(insn, &mnemonic))
        .or_else(|| permute::duplicate_shape(insn, &mnemonic))
        .or_else(|| permute::permute_pair_shape(insn, &mnemonic))
        .or_else(|| width::widen_long_shape(insn, &mnemonic))
        .or_else(|| width::narrow_shape(insn, &mnemonic))
        .or_else(|| width::saturating_narrow_shape(insn, &mnemonic))
        // Ahead of the mnemonic families below and, more importantly,
        // ahead of the scalar VFP arm in the dispatcher: `vcvt` is
        // spelled identically in both, and the scalar handler would
        // convert lane zero and leave the rest of the register
        // standing — a wrong value rather than a decline.
        .or_else(|| width::convert_shape(insn, &mnemonic))
        .or_else(|| element::by_element_shape(insn, &mnemonic))
        .or_else(|| accumulate::absolute_difference_shape(insn, &mnemonic))
        .or_else(|| accumulate::pairwise_long_shape(insn, &mnemonic))
        .or_else(|| select::bitwise_select_shape(insn, &mnemonic))
        .or_else(|| select::shift_insert_shape(insn, &mnemonic))
        .or_else(|| table::table_lookup_shape(insn, &mnemonic))
        .or_else(|| lanewise::compare_shape(insn, &mnemonic))
        .or_else(|| lanewise::pairwise_shape(insn, &mnemonic))
        .or_else(|| lanewise::count_shape(insn, &mnemonic))
        .or_else(|| estimate::estimate_shape(insn, &mnemonic))
}

/// Architectural width of the `AArch32` vector register parent, which
/// is also the width of a `q` view and twice that of a `d`.
pub(super) fn vector_parent_bits() -> Option<u16> {
    crate::registers::simd_parent_bits(Arch::Arm)
}

/// Element width a bare `AArch32` NEON size suffix names — the `8` in
/// `vext.8`, the `32` in `vdup.32`.
///
/// The families that move bits rather than computing on them spell
/// their element as a width alone, with no signedness letter, because
/// there is no arithmetic for a sign to change.
/// [`super::neon_element_type`] rejects exactly these spellings: it
/// reads the first character as the signedness class, and a digit names
/// none.
pub(super) fn bare_element_bits(ty: &str) -> Option<u16> {
    match ty {
        "8" => Some(8),
        "16" => Some(16),
        "32" => Some(32),
        "64" => Some(64),
        _ => None,
    }
}

/// The view width a bare `AArch32` vector register operand names: 64
/// for a `d` register, 128 for a `q`.
///
/// `AArch32` leaves the operand bare, so the register spelling is the
/// only thing saying how much of the register file the instruction
/// covers — the counterpart of the arrangement `AArch64` writes on the
/// operand itself.
///
/// An operand carrying vector shape is refused rather than resolved.
/// [`register_layout`] deliberately maps `d1[1]` onto the *whole* `d1`
/// slice, so accepting it here would read all 64 bits where the
/// instruction names one lane.
pub(super) fn vector_view_bits(op: &Operand) -> Option<u16> {
    if op.kind != OperandKind::Register || has_vector_arrangement(&op.raw, Arch::Arm) {
        return None;
    }
    let layout = register_layout(op.raw.trim(), Arch::Arm)?;
    is_simd_parent(layout.parent, Arch::Arm).then(|| layout.width())
}

/// The view every operand in `positions` names, or `None` unless they
/// all name the same one.
///
/// The uniformity check is the `AArch32` counterpart of `AArch64`'s
/// "every operand yields the same arrangement": there, a mismatched
/// `.8b` against a `.16b` is the decline; here it is a `d` against a
/// `q`.
pub(super) fn uniform_vector_view(insn: &Instruction, positions: usize) -> Option<u16> {
    let view = vector_view_bits(insn.operands.first()?)?;
    for index in 1..positions {
        if vector_view_bits(insn.operands.get(index)?)? != view {
            return None;
        }
    }
    Some(view)
}

/// Whether `op` names a general-purpose register — the source `vdup`
/// broadcasts, as opposed to the vector element form.
pub(super) fn is_general_register(op: &Operand) -> bool {
    op.kind == OperandKind::Register
        && !has_vector_arrangement(&op.raw, Arch::Arm)
        && register_layout(op.raw.trim(), Arch::Arm)
            .is_some_and(|layout| !is_simd_parent(layout.parent, Arch::Arm))
}
