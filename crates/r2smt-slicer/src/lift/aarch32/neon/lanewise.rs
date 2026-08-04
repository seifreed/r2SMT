//! `AArch32` NEON families that compute one destination lane from the
//! matching source lanes: the lane compares, the pairwise reductions,
//! and the per-element bit counts.
//!
//! They share this module because they share a shape — a typed or
//! bare-width element on the mnemonic and bare vector operands — and
//! none of them widens or narrows.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::lift::LiftCtx;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;
use crate::lift::simd::all_ones;
use crate::lift::{FpArithOp, fp_lane_result, fp_propagating_max_min, fp_sort_bits_checked};

use super::{NeonOp, NeonShape, bare_element_bits, uniform_vector_view, vector_view_bits};

/// Width of a `d` register — the only view the pairwise forms take.
const DOUBLEWORD_BITS: u16 = 64;

/// Ceiling on the primitive operations a per-element count may expand
/// into (`lanes * lane_bits`). The widest real form is a 128-bit `vclz`
/// over four 32-bit lanes, 128 operations, so no encoding can bind this;
/// it is a backstop against a malformed geometry, not a real limit.
const COUNT_EXPANSION_CAP: u32 = 2048;

/// Which lane-wise predicate a compare mnemonic names.
#[derive(Debug, Clone, Copy)]
pub(in crate::lift::aarch32) enum CompareKind {
    /// `vceq` — equality. `float` reads the lane as an IEEE value.
    Equal { float: bool },
    /// `vcgt` / `vcge` — ordered comparison. `or_equal` picks `ge` over
    /// `gt`; `signed` chooses the integer comparison and is unused for a
    /// float lane.
    Ordered {
        float: bool,
        signed: bool,
        or_equal: bool,
    },
    /// `vtst` — true where the bitwise AND of the two lanes is non-zero.
    TestBits,
}

/// Which reduction a pairwise mnemonic applies to each adjacent pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::lift::aarch32) enum PairwiseOp {
    /// `vpadd`.
    Add,
    /// `vpmax`.
    Max,
    /// `vpmin`.
    Min,
}

/// `vceq` / `vcgt` / `vcge` / `vtst` — the lane-wise compares, whose
/// result is an all-ones / all-zeros mask per lane rather than a flag,
/// because it feeds another vector operation.
///
/// Each compare-against-register form has a compare-against-`#0`
/// counterpart (`vcge.s8 d0, d1, #0`), which is what most
/// opaque-predicate patterns use. `vtst` has no zero form.
pub(super) fn compare_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    if base == "vtst" {
        // `vtst` is bitwise, so it spells a bare width and has no signed
        // or float class and no zero form.
        let lane_bits = bare_element_bits(ty).or_else(|| untyped_bits(ty))?;
        return build_compare(insn, CompareKind::TestBits, lane_bits, false);
    }
    let (element, lane_bits) = neon_element_type(ty)?;
    let kind = match base {
        // Equality needs no signedness; `vceq` is spelled `i` or `f`.
        "vceq" => CompareKind::Equal {
            float: element == ElementKind::Float,
        },
        "vcgt" => ordered_kind(element, false)?,
        "vcge" => ordered_kind(element, true)?,
        _ => return None,
    };
    build_compare(insn, kind, lane_bits, true)
}

/// `vpadd` / `vpmax` / `vpmin` — the pairwise reductions, which combine
/// each adjacent pair of source lanes into one destination lane.
///
/// The two sources fill the destination in order: the first source's
/// pairwise results occupy the low half of the destination, the
/// second's the high half. Doubleword-only, as the architecture has no
/// quadword pairwise form. The `.f32` forms lower through the IEEE
/// lane helpers — `vpmax` / `vpmin` reuse the ARM `FPMax` value, never
/// Intel's `MAXPS`.
pub(super) fn pairwise_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let op = match base {
        "vpadd" => PairwiseOp::Add,
        "vpmax" => PairwiseOp::Max,
        "vpmin" => PairwiseOp::Min,
        _ => return None,
    };
    let (element, lane_bits) = neon_element_type(ty)?;
    let float = element == ElementKind::Float;
    let signed = match op {
        // Addition is sign-agnostic; `vpadd` is spelled `i` or `f`.
        PairwiseOp::Add => false,
        // Max / min need a signedness class for the integer forms; the
        // `i` spelling has no ordered encoding. Float carries its own
        // ordering, so signedness does not apply.
        PairwiseOp::Max | PairwiseOp::Min => match element {
            ElementKind::Signed => true,
            ElementKind::Unsigned | ElementKind::Float => false,
            ElementKind::Untyped => return None,
        },
    };
    // The float forms exist only at single precision.
    if float && lane_bits != 32 {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    // Pairwise is doubleword-only.
    if view != DOUBLEWORD_BITS {
        return None;
    }
    let lanes = view.checked_div(lane_bits)?;
    if lanes < 2 || lanes % 2 != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Pairwise { op, signed, float },
        lane_bits,
        lanes,
    })
}

/// `vcnt` / `vclz` — the per-element bit counts.
///
/// `vcnt` counts the set bits of each byte and exists only at byte
/// width; `vclz` counts leading zeros and exists at 8, 16 and 32 bits.
/// Both are sign-agnostic — the count does not depend on the value's
/// signedness — so they spell a bare width or the `i` class.
pub(super) fn count_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let leading = match base {
        "vcnt" => false,
        "vclz" => true,
        _ => return None,
    };
    let lane_bits =
        bare_element_bits(ty).or_else(|| neon_element_type(ty).map(|(_, bits)| bits))?;
    // `vcnt` is byte-only; `vclz` covers 8 / 16 / 32.
    let ok = if leading {
        matches!(lane_bits, 8 | 16 | 32)
    } else {
        lane_bits == 8
    };
    if !ok || insn.operands.len() != 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    let lanes = view.checked_div(lane_bits)?;
    if lanes == 0 || u32::from(lanes).checked_mul(u32::from(lane_bits))? > COUNT_EXPANSION_CAP {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::CountBits { leading },
        lane_bits,
        lanes,
    })
}

/// Width of an `i`-spelled bare element, for the `vtst.i8` spelling some
/// disassemblers print instead of `vtst.8`.
fn untyped_bits(ty: &str) -> Option<u16> {
    match neon_element_type(ty)? {
        (ElementKind::Untyped, bits) => Some(bits),
        _ => None,
    }
}

/// The ordered compare `vcgt` / `vcge` names, or `None` for the
/// sign-agnostic `i` spelling the ordered forms have no encoding for.
fn ordered_kind(element: ElementKind, or_equal: bool) -> Option<CompareKind> {
    let (float, signed) = match element {
        ElementKind::Signed => (false, true),
        ElementKind::Unsigned => (false, false),
        ElementKind::Float => (true, true),
        ElementKind::Untyped => return None,
    };
    Some(CompareKind::Ordered {
        float,
        signed,
        or_equal,
    })
}

fn build_compare(
    insn: &Instruction,
    kind: CompareKind,
    lane_bits: u16,
    allow_zero: bool,
) -> Option<NeonShape> {
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    let lanes = view.checked_div(lane_bits)?;
    if lanes == 0 {
        return None;
    }
    let third = insn.operands.get(2)?;
    let zero = if third.kind == OperandKind::Immediate {
        if !allow_zero || !is_zero_immediate(third) {
            return None;
        }
        true
    } else {
        if vector_view_bits(third)? != view {
            return None;
        }
        false
    };
    // A float compare needs an IEEE lane.
    let float = matches!(
        kind,
        CompareKind::Equal { float: true } | CompareKind::Ordered { float: true, .. }
    );
    if float && fp_sort_bits_checked(lane_bits).is_none() {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Compare { kind, zero },
        lane_bits,
        lanes,
    })
}

/// Whether `op` is the immediate zero the compare-against-zero forms
/// take.
fn is_zero_immediate(op: &Operand) -> bool {
    let raw = op.raw.trim().trim_start_matches('#');
    matches!(raw, "0" | "0x0" | "0.0" | "0.00000" | "#0")
}

impl LiftCtx {
    /// Lower a lane-wise compare into a per-lane all-ones / all-zeros
    /// mask.
    pub(in crate::lift::aarch32::neon) fn aarch32_compare_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: CompareKind,
        zero: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = if zero {
            None
        } else {
            Some(self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?)
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = Self::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = match second.as_ref() {
                Some(value) => Self::extract_lane(value.clone(), shape.lane_bits, index)?,
                None => Expr::konst(0, shape.lane_bits),
            };
            lanes.push(compare_lane(kind, a, b, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }
}

/// One destination lane of a compare: an all-ones mask where the
/// predicate holds and all-zeros where it does not.
///
/// A `gt` lowers as `slt(b, a)`, not `sgt(a, b)`, mirroring the
/// `AArch64` side and keeping one comparison direction throughout.
fn compare_lane(kind: CompareKind, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    let predicate = match kind {
        CompareKind::Equal { float: false } => Expr::eq(a, b),
        CompareKind::Equal { float: true } => {
            let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
            Expr::feq(
                Expr::bv_to_fp(a, ebits, sbits),
                Expr::bv_to_fp(b, ebits, sbits),
            )
        }
        CompareKind::Ordered {
            float: true,
            or_equal,
            ..
        } => {
            let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
            let (x, y) = (
                Expr::bv_to_fp(a, ebits, sbits),
                Expr::bv_to_fp(b, ebits, sbits),
            );
            // Ordered: unordered compares false, which `fp.lt` / `fp.leq`
            // already give.
            if or_equal {
                Expr::fle(y, x)
            } else {
                Expr::flt(y, x)
            }
        }
        CompareKind::Ordered {
            float: false,
            signed,
            or_equal,
        } => match (signed, or_equal) {
            (true, false) => Expr::slt(b, a),
            (true, true) => Expr::sle(b, a),
            (false, false) => Expr::ult(b, a),
            (false, true) => Expr::ule(b, a),
        },
        CompareKind::TestBits => Expr::ne(Expr::bv_and(a, b), Expr::konst(0, lane_bits)),
    };
    Some(Expr::Ite {
        cond: Box::new(predicate),
        then_expr: Box::new(all_ones(lane_bits)),
        else_expr: Box::new(Expr::konst(0, lane_bits)),
    })
}

impl LiftCtx {
    /// Lower a pairwise reduction: each adjacent pair of one source's
    /// lanes combined into a destination lane, the first source filling
    /// the low half of the destination and the second the high half.
    pub(in crate::lift::aarch32::neon) fn aarch32_pairwise_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        op: PairwiseOp,
        signed: bool,
        float: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let half = shape.lanes / 2;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let (source, pair) = if index < half {
                (&first, index)
            } else {
                (&second, index - half)
            };
            let a = Self::extract_lane(source.clone(), shape.lane_bits, pair.checked_mul(2)?)?;
            let b = Self::extract_lane(
                source.clone(),
                shape.lane_bits,
                pair.checked_mul(2)?.checked_add(1)?,
            )?;
            lanes.push(pairwise_combine(op, signed, float, shape.lane_bits, a, b)?);
        }
        Self::concat_lanes(lanes)
    }
}

impl LiftCtx {
    /// Lower a per-element bit count: population count (`vcnt`) or
    /// leading-zero count (`vclz`), one destination lane per source
    /// lane.
    pub(in crate::lift::aarch32::neon) fn aarch32_count_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        leading: bool,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let element = Self::extract_lane(source.clone(), shape.lane_bits, index)?;
            lanes.push(if leading {
                count_leading_zeros_lane(&element, shape.lane_bits)
            } else {
                population_count_lane(&element, shape.lane_bits)
            });
        }
        Self::concat_lanes(lanes)
    }
}

/// One destination lane of `vcnt`: the number of set bits, as the sum of
/// each bit zero-extended to the lane width.
fn population_count_lane(element: &Expr, bits: u16) -> Expr {
    let mut acc = Expr::zero_ext(Expr::extract(element.clone(), 0, 0), bits);
    for bit in 1..bits {
        acc = Expr::add(
            acc,
            Expr::zero_ext(Expr::extract(element.clone(), bit, bit), bits),
        );
    }
    acc
}

/// One destination lane of `vclz`: the number of leading zeros, as an
/// `Ite` ladder from the least-significant bit upward so the
/// most-significant bit's test is the outermost — and therefore
/// highest-priority — decision. An all-zero element yields the full
/// width.
fn count_leading_zeros_lane(element: &Expr, bits: u16) -> Expr {
    let mut result = Expr::konst(u128::from(bits), bits);
    for bit in 0..bits {
        // A set bit at position `bit` (from the LSB) is the highest set
        // bit seen so far once the loop reaches it, and leaves
        // `bits - 1 - bit` zeros above it.
        let is_set = Expr::eq(Expr::extract(element.clone(), bit, bit), Expr::konst(1, 1));
        result = Expr::Ite {
            cond: Box::new(is_set),
            then_expr: Box::new(Expr::konst(u128::from(bits - 1 - bit), bits)),
            else_expr: Box::new(result),
        };
    }
    result
}

/// One destination lane of a pairwise reduction. The float forms lower
/// through the IEEE lane helpers: `vpadd.f32` is a lane add, and
/// `vpmax` / `vpmin` use the ARM `FPMax` value (`number_wins = false`,
/// so a NaN operand propagates), never Intel's `MAXPS`.
fn pairwise_combine(
    op: PairwiseOp,
    signed: bool,
    float: bool,
    lane_bits: u16,
    a: Expr,
    b: Expr,
) -> Option<Expr> {
    if float {
        return match op {
            PairwiseOp::Add => fp_lane_result(FpArithOp::Add, a, b, lane_bits),
            PairwiseOp::Max => fp_propagating_max_min(a, b, lane_bits, true, false),
            PairwiseOp::Min => fp_propagating_max_min(a, b, lane_bits, false, false),
        };
    }
    Some(match op {
        PairwiseOp::Add => Expr::add(a, b),
        PairwiseOp::Max | PairwiseOp::Min => {
            let less = if signed {
                Expr::slt(a.clone(), b.clone())
            } else {
                Expr::ult(a.clone(), b.clone())
            };
            let (then_expr, else_expr) = if op == PairwiseOp::Max {
                // `a < b` selects `b` for the maximum.
                (b, a)
            } else {
                (a, b)
            };
            Expr::Ite {
                cond: Box::new(less),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            }
        }
    })
}
