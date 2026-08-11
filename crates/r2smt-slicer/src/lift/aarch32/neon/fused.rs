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
//! significand covers `2 * p + 2` bits — binary32 over binary16,
//! binary64 over binary32, binary128 over binary64. So the width
//! question is not "is it 32", it is whatever
//! [`crate::lift::fused_step_is_emulable`] answers, and asking the
//! lowering is what stops the two drifting apart again.
//!
//! Which widths the ISA *encodes* is a **separate** question, and the
//! one this file gets wrong if it copies `AArch64`. It goes wrong at
//! both ends:
//!
//! - `vfma` / `vfms` do have a binary64 spelling, but only the VFP
//!   scalar one — `vfma.f64 d0, d1, d2`, a single 64-bit lane in a `d`
//!   register. A `q` operand with `.f64` is not an encoding, so widening
//!   the gate without checking the view would resolve two lanes that the
//!   machine never produces.
//! - `vrecps` / `vrsqrts` have no binary64 form in any spelling. They
//!   stay at binary32 whatever the intermediate can express.
//! - At the narrow end all four *do* have a packed `.f16` form with
//!   `FEAT_FP16`, and those resolve. The trap there is the reverse: a
//!   register width is not a lane count, because an `s` register holds
//!   one half-precision element and not two, so the VFP scalar `.f16`
//!   spelling declines rather than being divided into a lane that does
//!   not exist.
//!
//! It is the same asymmetry `neon/round.rs` records for `vrint*`, and
//! for the same reason: `AArch32` spells the element type on the
//! mnemonic, so a width the mnemonic can name is not evidence that the
//! encoding exists — nor of how many of them a register holds.

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

/// The view a VFP scalar register presents. It holds exactly one
/// element whatever the mnemonic's element width says, which is why an
/// element narrower than this cannot be resolved by dividing.
const VFP_SCALAR_VIEW_BITS: u16 = 32;

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
///
/// A binary16 lane needs the *other* half of the same idea, and there
/// getting it wrong is a wrong value rather than a decline.
/// `vector_view_bits` reports a register's width, so an `s` register
/// answers 32 whatever the element is — and dividing that by a 16-bit
/// element gives two lanes where `VFMA.F16 <Sd>, <Sn>, <Sm>` has one,
/// in the low half, as [`super::super::vfp::vfp_round_value`] records.
///
/// So that spelling declines. Resolving it as one lane does not work
/// here: the shape's view is `lanes * lane_bits`, and that is what the
/// destination write splices, so a one-lane binary16 shape would splice
/// 16 bits into an operand whose own view is 32 and lose the bits
/// between. The single-lane write that handles it correctly is the one
/// `vfp.rs` uses, on a path this resolver does not reach. The packed
/// `d` / `q` spellings, which are the NEON ones, resolve normally.
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
    // A half-precision element in an `s` register is *one* lane in the
    // low half, not the two this view would divide into — but declining
    // is the answer rather than resolving it as one, because the shape's
    // view (`lanes * lane_bits`) is what the write splices and it would
    // then disagree with the operand's own 32-bit view. Modelling it
    // needs the single-lane write `vfp.rs` uses, not this path.
    if view == VFP_SCALAR_VIEW_BITS && lane_bits < VFP_SCALAR_VIEW_BITS {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::FusedStep(step),
        lane_bits,
        lanes: view.checked_div(lane_bits)?,
    })
}
