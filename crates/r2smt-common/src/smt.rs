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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// load-independent unit-of-work bound. `0` leaves it unset, so the
    /// only budget is the wall clock and the verdict then depends on
    /// host load — the same branch classifies `real_branch` on an idle
    /// machine and `suspicious_but_unknown` on a busy one.
    ///
    /// Defaults to [`DEFAULT_RLIMIT`] rather than to `0` since
    /// 2026-08-07. Read the trade precisely, because it is not a
    /// widening: this is a *second* budget applied alongside
    /// `timeout_ms`, so it can only ever make fewer branches decide, not
    /// more. What it buys is that which ones stop deciding no longer
    /// depends on what else the host was doing.
    pub rlimit: u32,
}

/// Deterministic solver budget applied by default, in Z3 resource
/// units.
///
/// The value every measurement on this branch already used, so a run
/// with no flags and a diff taken per the `CLAUDE.md` recipe now agree
/// instead of silently measuring two different things.
pub const DEFAULT_RLIMIT: u32 = 2_000_000;

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            timeout_ms: 500,
            random_seed: 0,
            rlimit: DEFAULT_RLIMIT,
        }
    }
}
