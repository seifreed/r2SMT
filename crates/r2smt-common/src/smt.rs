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
///
/// Extended fields are purely additive: every existing call site that
/// only set `timeout_ms` keeps its behaviour by spreading
/// `..SolveOptions::default()`, and the defaults are chosen so the
/// observable verdict is unchanged unless a caller opts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveOptions {
    /// Per-branch solver budget in milliseconds (wall-clock).
    pub timeout_ms: u32,
    /// Pinned PRNG seed handed to every backend — the Z3 `random_seed`
    /// parameter and the SMT-LIB `(set-option :random-seed …)` line
    /// the subprocess backends (CVC5 / Bitwuzla) consume. Pinning it
    /// makes a given query's verdict reproducible run-to-run instead
    /// of varying with the solver's internal randomisation. Default
    /// `0` (also Z3's own default — pinning it explicitly documents
    /// the determinism intent and guards against upstream drift).
    pub random_seed: u32,
    /// Z3 deterministic resource limit (the `rlimit` parameter): a
    /// load-independent unit-of-work bound. `0` (default) leaves it
    /// unset, so the wall-clock `timeout_ms` path is byte-identical to
    /// before unless a caller opts in. When `> 0` the budget no longer
    /// depends on host load, so the `Unknown → Timeout` classification
    /// stops flickering between runs under contention.
    pub rlimit: u32,
}

impl Default for SolveOptions {
    fn default() -> Self {
        // `timeout_ms` matches SPEC.md §5.6; `random_seed` / `rlimit`
        // defaults preserve the pre-P24 observable behaviour.
        Self {
            timeout_ms: 500,
            random_seed: 0,
            rlimit: 0,
        }
    }
}
