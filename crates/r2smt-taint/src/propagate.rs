//! Forward may-taint propagation over an SSA-renamed slice.
//!
//! # Soundness
//!
//! For a slice with no `Expr::Unknown`, no `IrStmt::Unsupported`, and no
//! truncation, the register data-flow taint is *exact* (SSA gives
//! def-before-use, so one forward pass is the fixed point) and the
//! memory taint is a sound over-approximation: any load may alias any
//! earlier tainted store, so a load inherits the union of every prior
//! stored taint. Under those conditions the result has no false
//! negatives — if a sink is untainted here, no modelled flow taints it.
//!
//! When an `Unknown` or `Unsupported` node is present the pass sets
//! `TaintOutcome::opaque`. An opaque outcome is best-effort: flows
//! hidden inside the unmodelled node are invisible, so the taint map is
//! no longer a guaranteed lower or upper bound. Callers that need
//! soundness must treat an opaque outcome as "taint status unknown"
//! (mirroring how the verifier declines on `Unknown`). Completing those
//! flows via concretisation is the unsound subset delegated to the
//! exploration engine, not this crate.

use std::collections::BTreeMap;

use r2smt_ir::expr::Expr;
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::SliceStatus;
use r2smt_ssa::SsaLiftedSlice;

use crate::lattice::{SourceId, TaintSet};

/// Recursion budget for the expression walker (host-safety guardrail;
/// per-instruction expression trees are shallow, so this only trips on
/// pathological IR).
const EXPR_DEPTH_BUDGET: u32 = 256;

/// Initial taint assignment: a source seeded onto named SSA variables or
/// inputs (keyed by variable name).
pub type TaintSeeds = BTreeMap<String, TaintSet>;

/// Result of a propagation pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaintOutcome {
    taint: BTreeMap<String, TaintSet>,
    opaque: bool,
}

impl TaintOutcome {
    /// The taint carried by the SSA variable named `name` (untainted if
    /// the name was never defined or seeded).
    #[must_use]
    pub fn taint_of(&self, name: &str) -> TaintSet {
        self.taint.get(name).cloned().unwrap_or_default()
    }

    /// Whether `name` may carry `source`.
    #[must_use]
    pub fn reaches(&self, name: &str, source: SourceId) -> bool {
        self.taint.get(name).is_some_and(|t| t.contains(source))
    }

    /// Whether any `Unknown` / `Unsupported` node made the result
    /// best-effort rather than sound. See the module soundness note.
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        self.opaque
    }
}

/// Propagate `seeds` forward through `ssa`, returning the taint of every
/// defined variable.
///
/// Returns `None` only if the expression recursion budget is exceeded
/// (pathological IR) — callers should treat that as "not analysable".
#[must_use]
pub fn propagate(ssa: &SsaLiftedSlice, seeds: &TaintSeeds) -> Option<TaintOutcome> {
    let mut taint = seeds.clone();
    // Coarse (alias-free) memory taint: the union of every tainted value
    // stored so far. A later load inherits it, so any load may alias any
    // earlier tainted store — a sound over-approximation.
    let mut memory_taint = TaintSet::untainted();
    let mut opaque = matches!(ssa.status, SliceStatus::Truncated { .. });

    for stmt in &ssa.statements {
        match stmt {
            IrStmt::Assign { dst, src } => {
                let t = taint_of_expr(src, &taint, &mut opaque, 0)?;
                if !t.is_untainted() {
                    taint.entry(dst.name.clone()).or_default().join_from(&t);
                }
            }
            IrStmt::LoadMem { dst, address, .. } => {
                let mut t = taint_of_expr(address, &taint, &mut opaque, 0)?;
                t.join_from(&memory_taint);
                if !t.is_untainted() {
                    taint.entry(dst.name.clone()).or_default().join_from(&t);
                }
            }
            IrStmt::StoreMem { value, .. } => {
                let t = taint_of_expr(value, &taint, &mut opaque, 0)?;
                memory_taint.join_from(&t);
            }
            IrStmt::Unsupported { .. } => opaque = true,
            IrStmt::Nop => {}
        }
    }

    Some(TaintOutcome { taint, opaque })
}

/// The taint of an expression: the union of the taints of the variables
/// it reads. An `Unknown` node contributes nothing to the set but flags
/// the outcome opaque (its hidden reads cannot be accounted for).
fn taint_of_expr(
    expr: &Expr,
    taint: &BTreeMap<String, TaintSet>,
    opaque: &mut bool,
    depth: u32,
) -> Option<TaintSet> {
    if depth > EXPR_DEPTH_BUDGET {
        return None;
    }
    let next = depth + 1;
    let out = match expr {
        Expr::Var(v) => taint.get(&v.name).cloned().unwrap_or_default(),
        Expr::Const { .. } => TaintSet::untainted(),
        Expr::Unknown(_) => {
            *opaque = true;
            TaintSet::untainted()
        }
        Expr::BoolNot(a) => taint_of_expr(a, taint, opaque, next)?,
        Expr::Extract { src, .. } | Expr::ZeroExtend { src, .. } | Expr::SignExtend { src, .. } => {
            taint_of_expr(src, taint, opaque, next)?
        }
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::UDiv(a, b)
        | Expr::URem(a, b)
        | Expr::SDiv(a, b)
        | Expr::SRem(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Shl(a, b)
        | Expr::LShr(a, b)
        | Expr::AShr(a, b)
        | Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Ult(a, b)
        | Expr::Ule(a, b)
        | Expr::Slt(a, b)
        | Expr::Sle(a, b)
        | Expr::BoolAnd(a, b)
        | Expr::BoolOr(a, b)
        | Expr::Concat { high: a, low: b } => {
            let mut t = taint_of_expr(a, taint, opaque, next)?;
            t.join_from(&taint_of_expr(b, taint, opaque, next)?);
            t
        }
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            let mut t = taint_of_expr(cond, taint, opaque, next)?;
            t.join_from(&taint_of_expr(then_expr, taint, opaque, next)?);
            t.join_from(&taint_of_expr(else_expr, taint, opaque, next)?);
            t
        }
    };
    Some(out)
}
