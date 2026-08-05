//! The `AArch32` NEON forms that round **once** over a multiply and the
//! combine that follows it: `vfma` / `vfms` and the reciprocal
//! refinement steps `vrecps` / `vrsqrts`.
//!
//! This is the family whose failure mode is least visible. `vmla` /
//! `vmls` sit right beside these in the instruction set, take the same
//! three float operands, and compute *almost* the same thing — the
//! difference is that they round the product and then round the
//! accumulate again, where these round the whole expression once. So a
//! lowering built from a separate `fp_lane_result(Mul)` and a following
//! add would be plausible, would agree with the machine on most inputs,
//! and would be a definite wrong number on the rest. That is why these
//! go through [`fused_multiply_lane`] and not through the lane helpers
//! [`super::super::neon_packed_op`]'s float table uses.
//!
//! Only binary32 lanes resolve. The lowering is exact because it widens
//! each lane to binary64 and rounds once back, which works precisely
//! when the wider format's significand covers `2 * 24 + 2` bits — true
//! for binary64 over binary32 and false for anything over binary64. A
//! `.f64` form therefore declines, and soundly: no scalar VFP handler
//! spells `vfma` either, so nothing downstream lowers it as a lane-zero
//! operation.

use r2smt_ir::program::Instruction;

use crate::lift::FusedStep;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;

use super::{NeonOp, NeonShape, uniform_vector_view};

/// The only element width the fused steps are exactly emulable at.
const FUSED_STEP_LANE_BITS: u16 = 32;

/// `vfma` / `vfms` / `vrecps` / `vrsqrts`.
///
/// A single-lane view (`vfma.f32 s0, s1, s2`) resolves here too, and
/// deliberately: that is the VFP spelling of the same instruction, it
/// computes the same fused step, and unlike the families
/// [`super::super::neon_packed_op`] shares with the scalar handler there
/// is no existing scalar path for it to stay byte-identical with.
pub(super) fn fused_step_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let step = match base {
        "vfma" => FusedStep::MulAdd,
        "vfms" => FusedStep::MulSub,
        "vrecps" => FusedStep::RecipStep,
        "vrsqrts" => FusedStep::RsqrtStep,
        _ => return None,
    };
    let (element, lane_bits) = neon_element_type(ty)?;
    if element != ElementKind::Float || lane_bits != FUSED_STEP_LANE_BITS {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    if view % lane_bits != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::FusedStep(step),
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}
