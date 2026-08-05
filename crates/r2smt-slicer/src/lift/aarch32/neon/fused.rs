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
//! The lowering is exact because it widens each lane to the next format
//! up and rounds once back, which works precisely when the wider
//! significand covers `2 * p + 2` bits — binary64 over binary32, and
//! binary128 over binary64. So the width question is not "is it 32", it
//! is whatever [`crate::lift::fused_step_is_emulable`] answers, and
//! asking the lowering is what stops the two drifting apart again.
//!
//! Which mnemonics may be 64 bits wide is a **separate** question, and
//! the one this file gets wrong if it copies `AArch64`. Advanced SIMD on
//! `AArch32` has no packed `.f64` form at all:
//!
//! - `vfma` / `vfms` do have a binary64 spelling, but only the VFP
//!   scalar one — `vfma.f64 d0, d1, d2`, a single 64-bit lane in a `d`
//!   register. A `q` operand with `.f64` is not an encoding, so widening
//!   the gate without checking the view would resolve two lanes that the
//!   machine never produces.
//! - `vrecps` / `vrsqrts` have no binary64 form in any spelling. They
//!   stay at binary32 whatever the intermediate can express.
//!
//! It is the same asymmetry `neon/round.rs` records for `vrint*`, and
//! for the same reason: `AArch32` spells the element type on the
//! mnemonic, so a width the mnemonic can name is not evidence that the
//! encoding exists.

use r2smt_ir::program::Instruction;

use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;
use crate::lift::{FusedStep, fused_step_is_emulable};

use super::{NeonOp, NeonShape, uniform_vector_view};

/// The widest lane the packed `AArch32` spelling of a fused step
/// encodes, whatever the intermediate could express.
///
/// Advanced SIMD has no `.f64` form; only `vfma` / `vfms` reach 64 bits,
/// and only through the single-lane VFP spelling the check below allows.
const PACKED_FUSED_STEP_MAX_LANE_BITS: u16 = 32;

/// `vfma` / `vfms` / `vrecps` / `vrsqrts`.
///
/// A single-lane view (`vfma.f32 s0, s1, s2`) resolves here too, and
/// deliberately: that is the VFP spelling of the same instruction, it
/// computes the same fused step, and unlike the families
/// [`super::super::neon_packed_op`] shares with the scalar handler there
/// is no existing scalar path for it to stay byte-identical with.
///
/// A binary64 lane is that spelling and only that spelling. The module
/// doc has the encoding argument; the shape consequence is that a 64-bit
/// element must occupy the whole view, so `vfma.f64 d0, d1, d2` resolves
/// as one lane and `vfma.f64 q0, q1, q2` — which would be two — does not.
pub(super) fn fused_step_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    let (step, packed) = match base {
        "vfma" => (FusedStep::MulAdd, false),
        "vfms" => (FusedStep::MulSub, false),
        "vrecps" => (FusedStep::RecipStep, true),
        "vrsqrts" => (FusedStep::RsqrtStep, true),
        _ => return None,
    };
    let (element, lane_bits) = neon_element_type(ty)?;
    if element != ElementKind::Float || !fused_step_is_emulable(lane_bits) {
        return None;
    }
    if packed && lane_bits > PACKED_FUSED_STEP_MAX_LANE_BITS {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let view = uniform_vector_view(insn, 3)?;
    if view % lane_bits != 0 {
        return None;
    }
    // The only binary64 encoding is the one-lane VFP form, so a wider
    // view naming a 64-bit element is not an instruction.
    if lane_bits > PACKED_FUSED_STEP_MAX_LANE_BITS && view != lane_bits {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::FusedStep(step),
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}
