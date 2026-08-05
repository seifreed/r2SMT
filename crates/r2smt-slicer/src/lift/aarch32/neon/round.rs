//! `vrinta` / `vrintn` / `vrintp` / `vrintm` / `vrintz` / `vrintx` in
//! their packed spelling — round every lane to an integral value
//! without leaving the float sort.
//!
//! Resolving these is a **soundness** matter and not extra coverage. The
//! scalar VFP arm of the dispatcher already claims the identical
//! mnemonic, so a packed form no resolver here claims does not decline:
//! it reaches [`crate::lift::LiftCtx::vfp_round_value`], which would
//! round lane zero and leave the rest of the register standing — the
//! wrong value the `vcvt` and by-element families document. Two defences
//! rather than one, as there: this resolver sits ahead of the scalar arm,
//! and the scalar handler independently checks each operand's register
//! class holds a single element, so a `q` operand cannot reach it
//! whatever this module does.
//!
//! Where the two families divide is the register class alone, which is
//! why the split is by lane *count*: a one-lane view is the VFP
//! encoding, and every wider one is Advanced SIMD.

use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::Instruction;

use crate::lift::LiftCtx;
use crate::lift::aarch32::{VfpOp, vfp_scalar};
use crate::lift::simd::fp_sort_bits_checked;

use super::{NeonOp, NeonShape, uniform_vector_view};

/// Element widths the Advanced SIMD rounding forms encode: binary32,
/// and binary16 with the FP16 extension. There is no packed `.f64`.
const ROUND_ELEMENT_BITS: [u16; 2] = [16, 32];

/// The packed `vrint<mode>` family — the destination geometry and the
/// mode the mnemonic names.
///
/// The mode table is [`vfp_scalar`]'s rather than a second copy, because
/// the two spellings differ in operand class and in nothing else.
/// `vrintr` is the one member excluded: it is VFP-only, the Advanced
/// SIMD space spelling its round-by-FPSCR form `vrintx`, so claiming it
/// would lift an encoding the architecture does not define.
pub(super) fn round_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, _) = mnemonic.split_once('.')?;
    if base == "vrintr" {
        return None;
    }
    let (VfpOp::Round { rounding, .. }, lane_bits) = vfp_scalar(mnemonic)? else {
        return None;
    };
    if !ROUND_ELEMENT_BITS.contains(&lane_bits) || insn.operands.len() != 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    let lanes = view.checked_div(lane_bits)?;
    // One lane is the scalar encoding. Leaving it to the VFP arm keeps
    // that path byte-identical rather than rerouting it through a
    // whole-view write that happens to agree.
    if lanes < 2 || view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::RoundToIntegral { rounding },
        lane_bits,
        lanes,
    })
}

impl LiftCtx {
    /// One `fround_to_integral` per lane, at the lane's own float sort.
    ///
    /// The result stays a float — that is what separates this family
    /// from `vcvt`, which lands in an integer format — so each lane
    /// bridges back through `fp_to_ieee_bv` rather than through a
    /// conversion.
    pub(in crate::lift::aarch32::neon) fn aarch32_round_to_integral_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        rounding: RoundingMode,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        let source = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let (ebits, sbits) = fp_sort_bits_checked(shape.lane_bits)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let bits = Self::extract_lane(source.clone(), shape.lane_bits, index)?;
            lanes.push(Expr::fp_to_ieee_bv(Expr::fround_to_integral(
                Expr::bv_to_fp(bits, ebits, sbits),
                rounding,
            )));
        }
        Self::concat_lanes(lanes)
    }
}
