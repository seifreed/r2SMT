//! Solver wrapper that delivers an opaque-predicate verdict for a
//! single SSA lifted slice.

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_slicer::SliceStatus;
use r2smt_ssa::SsaLiftedSlice;
use tracing::debug;
use z3::ast::{Ast, Bool};
use z3::{ApplyResult, Goal, Params, SatResult, Solver, Tactic};

use crate::encoder::Encoder;
use crate::pretty::z3_bool_to_infix;

/// Rich outcome of [`solve_branch_with_pretty`]: pairs the
/// [`SmtResult`] verdict with the C-style infix rendering of the
/// post-`aggressive_simplify` Z3 formula, when one was produced.
#[derive(Debug, Clone)]
pub struct SolveOutcome {
    /// Verdict reported to the classifier.
    pub verdict: SmtResult,
    /// Infix rendering of the solver-simplified formula. `None` for
    /// truncated slices that short-circuit before the solver runs.
    pub formula_z3_pretty: Option<String>,
}

/// Solve a branch and report the verdict.
///
/// Slices with a truncated status are reported as
/// [`SmtResult::Unsound`] without invoking the solver — by definition
/// they do not capture the full data-flow.
///
/// When the slice was produced under
/// [`r2smt_slicer::SliceLimits::unknowns_on_truncation`], the
/// truncated result still drives the solver: SSA already surfaced the
/// unresolved roots as free symbolic [`r2smt_ir::expr::Var`]s in
/// [`SsaLiftedSlice::inputs`], so the verdict is sound. The
/// classifier marks the resulting finding with a lower confidence
/// downstream.
#[must_use]
pub fn solve_branch(slice: &SsaLiftedSlice, options: SolveOptions) -> SmtResult {
    solve_branch_with_pretty(slice, options).verdict
}

/// Variant of [`solve_branch`] that also returns the C-style infix
/// rendering of the post-`aggressive_simplify` Z3 formula. Used by
/// the classifier to populate `r2smt_core::Finding::formula_z3_pretty`.
#[must_use]
pub fn solve_branch_with_pretty(slice: &SsaLiftedSlice, options: SolveOptions) -> SolveOutcome {
    let is_complete = matches!(slice.status, SliceStatus::Complete);
    if !is_complete && !slice.treat_truncation_as_inputs {
        return SolveOutcome {
            verdict: SmtResult::Unsound,
            formula_z3_pretty: None,
        };
    }

    let solver = Solver::new();
    let mut params = Params::new();
    params.set_u32("timeout", options.timeout_ms);
    solver.set_params(&params);

    let mut encoder = Encoder::new();
    let raw = encoder.encode(slice, &solver);
    // Pre-simplify the condition before issuing SAT queries.
    //
    // The lightweight `raw.simplify()` falls short on polynomial-style
    // opaque predicates (e.g. `(x*x) - (x*x) == 0`) because it does not
    // normalise expressions into sum-of-monomials form by default.
    // Apply a tactic chain — `simplify` with `som` and `blast_eq_value`
    // enabled, then `propagate-values` and `ctx-simplify` — to fold a
    // wider class of identities before the SAT loop. The chain is
    // equivalence-preserving, so the verdict is unchanged when nothing
    // collapses; on success the solver sees a smaller formula on both
    // push/pop iterations.
    let truth = aggressive_simplify(&raw);
    let pretty = Some(z3_bool_to_infix(&truth));

    solver.push();
    solver.assert(&truth);
    let sat_true = solver.check();
    solver.pop(1);

    solver.push();
    let not_truth = truth.not();
    solver.assert(&not_truth);
    let sat_false = solver.check();
    solver.pop(1);

    let verdict = combine(sat_true, sat_false);
    debug!(
        target: "r2smt::smt",
        at = %slice.branch.address,
        ?sat_true,
        ?sat_false,
        ?verdict,
        "smt verdict"
    );
    SolveOutcome {
        verdict,
        formula_z3_pretty: pretty,
    }
}

fn aggressive_simplify(raw: &Bool) -> Bool {
    let mut simplify_params = Params::new();
    simplify_params.set_bool("som", true);
    simplify_params.set_bool("blast_eq_value", true);

    let simplify = Tactic::new("simplify").with(&simplify_params);
    let chain = simplify
        .and_then(&Tactic::new("propagate-values"))
        .and_then(&Tactic::new("ctx-simplify"));

    let goal = Goal::new(false, false, false);
    goal.assert(raw);

    match chain.apply(&goal, None) {
        Ok(result) => collapse_subgoals(result),
        Err(_) => raw.simplify(),
    }
}

fn collapse_subgoals(result: ApplyResult) -> Bool {
    let mut conjuncts: Vec<Bool> = Vec::new();
    for goal in result.list_subgoals() {
        if goal.is_decided_unsat() {
            return Bool::from_bool(false);
        }
        if goal.is_decided_sat() {
            continue;
        }
        let formulas = goal.get_formulas();
        if formulas.is_empty() {
            continue;
        }
        let refs: Vec<&Bool> = formulas.iter().collect();
        conjuncts.push(Bool::and(refs.as_slice()));
    }
    if conjuncts.is_empty() {
        return Bool::from_bool(true);
    }
    let refs: Vec<&Bool> = conjuncts.iter().collect();
    Bool::and(refs.as_slice())
}

fn combine(sat_true: SatResult, sat_false: SatResult) -> SmtResult {
    match (sat_true, sat_false) {
        (SatResult::Sat, SatResult::Unsat) => SmtResult::AlwaysTrue,
        (SatResult::Unsat, SatResult::Sat) => SmtResult::AlwaysFalse,
        (SatResult::Sat, SatResult::Sat) => SmtResult::BothPossible,
        (SatResult::Unsat, SatResult::Unsat) => SmtResult::Unsound,
        (SatResult::Unknown, _) | (_, SatResult::Unknown) => SmtResult::Timeout,
    }
}

#[cfg(test)]
mod tests;
