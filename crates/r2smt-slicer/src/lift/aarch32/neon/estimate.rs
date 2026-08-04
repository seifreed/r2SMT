//! `vrecpe` / `vrsqrte` — the reciprocal and reciprocal-square-root
//! estimates.
//!
//! Their result is a **free value**, not a computed reciprocal. The
//! architecture guarantees only a relative error bound and leaves the
//! value itself implementation-defined, so `1 / x` would be a definite
//! number the machine is not required to produce — the fabrication this
//! pipeline exists to avoid. Declining is not the alternative either:
//! the slicer has already consumed the destination as a definition, so
//! a decline truncates the slice and every downstream verdict becomes
//! `Unsound`. Assigning a temp that is never defined keeps the slice
//! `Complete`, and `ssa_convert` surfaces the temp as a free input, so
//! the solver ranges over every value the estimate could take.
//!
//! This mirrors `AArch64`'s `frecpe` / `frsqrte`. The `.u32` integer
//! estimate is equally implementation-defined and maps to the same
//! free-value lowering.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use crate::lift::LiftCtx;
use crate::lift::aarch32::ElementKind;
use crate::lift::aarch32::neon_element_type;

use super::{NeonOp, NeonShape, uniform_vector_view};

/// `vrecpe` / `vrsqrte` — resolve the destination geometry. The two
/// mnemonics share one lowering because they share one contract: an
/// implementation-defined estimate.
pub(super) fn estimate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, ty) = mnemonic.split_once('.')?;
    if !matches!(base, "vrecpe" | "vrsqrte") {
        return None;
    }
    let (element, lane_bits) = neon_element_type(ty)?;
    // The float estimates take an IEEE lane; the `u32` integer estimate
    // is the only integer form.
    let ok = match element {
        ElementKind::Float => matches!(lane_bits, 16 | 32),
        ElementKind::Unsigned => lane_bits == 32,
        _ => false,
    };
    if !ok || insn.operands.len() != 2 {
        return None;
    }
    let view = uniform_vector_view(insn, 2)?;
    let lanes = view.checked_div(lane_bits)?;
    if lanes == 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Estimate,
        lane_bits,
        lanes,
    })
}

impl LiftCtx {
    /// Lower a reciprocal estimate to a fresh, never-assigned temp — the
    /// free-value contract. The source is deliberately not read: two
    /// estimates of the same input need not agree, so binding the value
    /// to the operand would claim a fact the architecture does not
    /// guarantee.
    pub(in crate::lift::aarch32::neon) fn aarch32_estimate_value(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
    ) -> Option<Expr> {
        let view = shape.view_bits()?;
        Some(Expr::Var(self.new_temp(insn.address, view)))
    }
}
