//! Solver-agnostic verdict types shared between the SMT backend and
//! the decision engine.
//!
//! Keeping these here (rather than inside `r2smt-smt`) lets
//! `r2smt-core` consume verdicts without taking a hard dependency on
//! any concrete solver crate.

use serde::{Deserialize, Serialize};

/// Verdict for a single branch produced by the SMT backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SmtResult {
    /// Condition is `SAT` for `cond == true` and `UNSAT` for
    /// `cond == false` — the branch is always taken.
    AlwaysTrue,
    /// The dual: branch is never taken.
    AlwaysFalse,
    /// Both polarities are satisfiable — genuine choice.
    BothPossible,
    /// Both polarities are `UNSAT`. With a sound encoding this should
    /// not happen; surface it so the caller can investigate.
    Unsound,
    /// The solver returned `UNKNOWN` for at least one polarity within
    /// the time budget.
    Timeout,
    /// The solver returned `UNKNOWN` for a non-timeout reason.
    Unknown,
}

/// Options controlling a single solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveOptions {
    /// Per-branch solver budget in milliseconds.
    pub timeout_ms: u32,
}

impl Default for SolveOptions {
    fn default() -> Self {
        // Matches SPEC.md §5.6 default budget.
        Self { timeout_ms: 500 }
    }
}
