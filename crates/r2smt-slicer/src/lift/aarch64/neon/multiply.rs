//! The families built on a lane product.
//!
//! Multiply-accumulate, the by-element forms, the dot products and the
//! polynomial multiply. They share more than the operation: the `2`
//! suffix that reads a source's upper half, the choice between
//! multiplying at the source width and extending first, and the
//! accumulator that makes the destination an input as well as an
//! output.
//!
//! The plain lane-wise `mul` is not here. It is one member of the
//! lane-wise family next door, resolved by the arrangement every
//! operand shares rather than by anything about the product.

use r2smt_ir::program::Instruction;

use super::super::super::{BinOp, FusedStep};
use super::geometry::{
    BITS_PER_BYTE, dot_product_element, indexed_element, operand_arrangement, peel_upper,
    spans_full_register,
};
use super::{NeonOp, NeonShape};

// ===================== multiply-accumulate =====================

/// How a multiply-accumulate reads its source elements.
///
/// A separate type rather than a `widen` flag beside a `signed` one,
/// because signedness is meaningless without widening: `mla` extends
/// nothing, so there is no extension to choose.
#[derive(Debug, Clone, Copy)]
pub(super) enum AccumulateSources {
    /// `mla` / `mls` — sources are already the destination's element
    /// width.
    SameWidth,
    /// `umlal` / `smlal` / `umlsl` / `smlsl` — sources are half the
    /// destination's element width and are extended before the multiply.
    Long { signed: bool },
}

/// How a multiply-accumulate reads its sources and combines the product.
#[derive(Debug, Clone, Copy)]
pub(super) struct AccumulateKind {
    /// Whether the product is added to or subtracted from the
    /// accumulator.
    pub(super) combine: BinOp,
    pub(super) sources: AccumulateSources,
    /// The `2` suffix — read the sources' upper half.
    pub(super) upper: bool,
}

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
pub(super) fn multiply_accumulate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
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

// ===================== by-element multiplies =====================

/// The by-element multiplies.
///
/// A three-way enum rather than a struct of flags because the
/// combinations are not free: an integer accumulate is exact, a long
/// one extends first, and the float member has no accumulating sibling
/// here at all.
#[derive(Debug, Clone, Copy)]
pub(super) enum ByElementKind {
    /// `mul` / `mla` / `mls` — same-width integer lanes, the product
    /// optionally combined with the destination's prior value.
    Integer { combine: Option<BinOp> },
    /// `umull` / `smull` / `umlal` / `smlal` / `umlsl` / `smlsl` — both
    /// narrow sources extended to the destination's element width
    /// before the multiply, so the product cannot overflow it.
    Long {
        signed: bool,
        combine: Option<BinOp>,
    },
    /// `fmul` — one IEEE lane product.
    Float,
    /// `fmla` / `fmls` — the product and the accumulate rounded
    /// **together**, once.
    ///
    /// `fadd(d, fmul(a, b))` rounds twice and the two differ in real
    /// cases, so this cannot go through the `Float` arm with a `combine`
    /// bolted on: that composition would be a definite wrong value
    /// rather than a wider one. The lowering routes to
    /// [`crate::lift::simd::fused_multiply_lane`], which computes the
    /// whole step in binary64 and rounds once back — exact for a
    /// binary32 lane and the reason this kind is gated to that width.
    Fused { step: FusedStep },
}

impl ByElementKind {
    /// How the product joins the destination's prior lane, or `None`
    /// when it replaces it.
    pub(super) const fn combine(self) -> Option<BinOp> {
        match self {
            Self::Integer { combine } | Self::Long { combine, .. } => combine,
            // The fused forms accumulate inside the single rounding, so
            // there is no separate combine step to name here.
            Self::Float | Self::Fused { .. } => None,
        }
    }

    /// Whether the destination's prior value is an input.
    pub(super) const fn combines(self) -> bool {
        self.combine().is_some() || matches!(self, Self::Fused { .. })
    }

    /// Whether the sources are half the destination's element width.
    pub(super) const fn widens(self) -> bool {
        matches!(self, Self::Long { .. })
    }
}

/// The by-element multiply a mnemonic names.
fn by_element_kind(base: &str) -> Option<ByElementKind> {
    let integer = |combine| ByElementKind::Integer { combine };
    let long = |signed, combine| ByElementKind::Long { signed, combine };
    Some(match base {
        "mul" => integer(None),
        "mla" => integer(Some(BinOp::Add)),
        "mls" => integer(Some(BinOp::Sub)),
        "fmul" => ByElementKind::Float,
        "fmla" => ByElementKind::Fused {
            step: FusedStep::MulAdd,
        },
        "fmls" => ByElementKind::Fused {
            step: FusedStep::MulSub,
        },
        "umull" => long(false, None),
        "smull" => long(true, None),
        "umlal" => long(false, Some(BinOp::Add)),
        "smlal" => long(true, Some(BinOp::Add)),
        "umlsl" => long(false, Some(BinOp::Sub)),
        "smlsl" => long(true, Some(BinOp::Sub)),
        _ => return None,
    })
}

/// The by-element family: `mul v0.4s, v1.4s, v2.s[0]` and its
/// relatives, where the second source names *one* element rather than a
/// whole vector.
///
/// Nothing about the index needed building — [`indexed_element`] already
/// returns `(32, 0)` for `v2.s[0]`, and [`NeonShape`] already carries a
/// source index. What declined was the geometry check the lane-wise and
/// widening resolvers share: both require every operand to yield the
/// *same* arrangement, and an indexed operand yields none at all.
pub(super) fn by_element_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let kind = by_element_kind(base)?;
    // Only the long forms have a `2` variant.
    if upper && !kind.widens() {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    let source_bits = if kind.widens() {
        destination.lane_bits / 2
    } else {
        destination.lane_bits
    };
    // ARM ARM C7.2 — the by-element encodings name a 16- or 32-bit
    // element, and the float ones a 64-bit one as well. There is no
    // byte-element form, which is what the dot products exist for.
    let encodable = match kind {
        ByElementKind::Float => matches!(source_bits, 16 | 32 | 64),
        // A binary64 lane would need binary128 to keep the fused step
        // exact, so the whole-vector spelling declines there too.
        ByElementKind::Fused { .. } => source_bits == FUSED_STEP_LANE_BITS,
        ByElementKind::Integer { .. } | ByElementKind::Long { .. } => {
            matches!(source_bits, 16 | 32)
        }
    };
    if !encodable {
        return None;
    }
    let first = operand_arrangement(insn.operands.get(1)?)?;
    let expected_lanes = if upper {
        destination.lanes.checked_mul(2)?
    } else {
        destination.lanes
    };
    if first.lane_bits != source_bits || first.lanes != expected_lanes {
        return None;
    }
    if upper && !spans_full_register(first) {
        return None;
    }
    let (element_bits, source_index) = indexed_element(insn.operands.get(2)?)?;
    if element_bits != source_bits {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::ByElement { kind, upper },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index,
    })
}

// ===================== dot products =====================

/// Byte products summed into one destination lane by `sdot` / `udot`.
pub(super) const DOT_PRODUCT_TERMS: u16 = 4;

/// Destination element width of a dot product.
const DOT_PRODUCT_LANE_BITS: u16 = 32;

/// `sdot` / `udot` over two whole vectors, or the by-element spelling.
///
/// The by-element form (`sdot v0.4s, v1.16b, v2.4b[1]`) names an
/// arrangement *and* an index at once, which the arrangement parser does
/// not resolve; [`dot_product_element`] parses it instead, and the index
/// selects the four-byte group broadcast to every destination lane.
pub(super) fn dot_product_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let signed = match mnemonic {
        "sdot" => true,
        "udot" => false,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if destination.lane_bits != DOT_PRODUCT_LANE_BITS {
        return None;
    }
    let expected_lanes = destination.lanes.checked_mul(DOT_PRODUCT_TERMS)?;
    // The first source is always the whole-byte vector.
    let first = operand_arrangement(insn.operands.get(1)?)?;
    if first.lane_bits != BITS_PER_BYTE || first.lanes != expected_lanes {
        return None;
    }
    // The second source is either the matching whole vector or the
    // `v2.4b[i]` group selector.
    let third = insn.operands.get(2)?;
    let (by_element, source_index) = match operand_arrangement(third) {
        Some(second) if second.lane_bits == BITS_PER_BYTE && second.lanes == expected_lanes => {
            (false, 0)
        }
        Some(_) => return None,
        None => {
            let (_, index) = dot_product_element(third)?;
            (true, index)
        }
    };
    Some(NeonShape {
        op: NeonOp::DotProduct { signed, by_element },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index,
    })
}

// ===================== fused multiply steps =====================

/// Destination element width the fused steps are exactly emulable at.
const FUSED_STEP_LANE_BITS: u16 = 32;

/// `fmla` / `fmls` / `frecps` / `frsqrts` over whole binary32-lane
/// vectors.
///
/// All four round the product and the following combine together, once.
/// Modelling them as a separate `fmul` then `fadd` would round twice and
/// give a wrong value, so the lowering instead computes each lane in
/// binary64 — exact for a binary32 lane — and rounds once back. Only the
/// 32-bit arrangements (`.2s` / `.4s`) resolve: a 64-bit lane would need
/// binary128 to stay exact, so `.2d` declines.
pub(super) fn fused_step_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let step = match mnemonic {
        "fmla" => FusedStep::MulAdd,
        "fmls" => FusedStep::MulSub,
        "frecps" => FusedStep::RecipStep,
        "frsqrts" => FusedStep::RsqrtStep,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if destination.lane_bits != FUSED_STEP_LANE_BITS {
        return None;
    }
    for operand in insn.operands.iter().skip(1) {
        if operand_arrangement(operand)? != destination {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::FusedStep(step),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== polynomial multiply =====================

/// Ceiling on the partial products one polynomial multiply may expand
/// into, counted as `lanes * bits per multiplier`.
///
/// The `.1q` form -- the AES / GHASH shape, and the one that actually
/// turns up -- is 64 partial products in a single lane, and the byte
/// form is eight apiece across eight lanes. Both sit well inside the
/// bound; it is here so that a wider encoding declines and says so
/// rather than quietly emitting a formula nobody sized.
const POLYNOMIAL_MULTIPLY_TERM_CAP: u32 = 256;

/// `pmull` / `pmull2` — the carry-less product of two narrow elements.
///
/// The architecture encodes exactly two shapes: `8B` into `8H`, and
/// `1D` into `1Q` under `FEAT_PMULL`. Nothing between them exists, so
/// the narrow width is checked against that pair rather than against a
/// range.
pub(super) fn polynomial_multiply_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    if base != "pmull" || insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    let narrow = destination.lane_bits / 2;
    if !matches!(narrow, 8 | 64) {
        return None;
    }
    let expected_lanes = if upper {
        destination.lanes.checked_mul(2)?
    } else {
        destination.lanes
    };
    for operand in insn.operands.iter().skip(1) {
        let arrangement = operand_arrangement(operand)?;
        if arrangement.lane_bits != narrow || arrangement.lanes != expected_lanes {
            return None;
        }
        if upper && !spans_full_register(arrangement) {
            return None;
        }
    }
    let terms = u32::from(narrow).checked_mul(u32::from(destination.lanes))?;
    if terms > POLYNOMIAL_MULTIPLY_TERM_CAP {
        tracing::debug!(
            mnemonic,
            terms,
            cap = POLYNOMIAL_MULTIPLY_TERM_CAP,
            "declining a polynomial multiply whose partial products exceed the cap"
        );
        return None;
    }
    Some(NeonShape {
        op: NeonOp::PolynomialMultiply { upper },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}
