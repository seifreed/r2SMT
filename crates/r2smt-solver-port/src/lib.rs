#![deny(missing_docs)]
//! Solver port — the narrow contract every SMT backend (Z3, CVC5,
//! Bitwuzla) and every differential oracle (radius2) implements.
//!
//! Kept in its own crate (depending only on [`r2smt_common`] and
//! [`r2smt_ssa`]) so backends are adapters wired at the composition
//! root and `r2smt-core` never depends on a concrete solver. The
//! `role` distinction is load-bearing for the reliability contract: a
//! [`SolverRole::DifferentialOracle`] may corroborate or downgrade a
//! verdict but may never decide one.

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_ssa::SsaLiftedSlice;

/// Whether a backend is an authoritative sound verdict source or a
/// corroboration-only differential oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverRole {
    /// Authoritative: its verdict may decide a `Finding`.
    Sound,
    /// Corroboration-only: may downgrade/flag, never decide a verdict.
    DifferentialOracle,
}

/// Outcome of solving a single branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverOutcome {
    /// The branch verdict.
    pub verdict: SmtResult,
    /// Optional C-style infix rendering of the post-simplify formula.
    /// Only the in-process Z3 backend fills this; others return `None`.
    pub formula_pretty: Option<String>,
}

/// Failure modes an adapter may surface at the port boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolverError {
    /// The backend binary / library is unavailable in this environment.
    Unavailable(String),
    /// The backend ran but failed or returned something unusable.
    Backend(String),
}

impl SolverError {
    /// The human-readable detail without the variant prefix. Lets the
    /// composition root format backend-specific CLI errors verbatim.
    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Self::Unavailable(d) | Self::Backend(d) => d,
        }
    }
}

impl core::fmt::Display for SolverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unavailable(d) => write!(f, "solver unavailable: {d}"),
            Self::Backend(d) => write!(f, "solver backend error: {d}"),
        }
    }
}

impl std::error::Error for SolverError {}

/// A pluggable SMT backend or differential oracle.
pub trait Solver {
    /// Decide a single branch. Implementations MUST sound-decline
    /// (return [`SmtResult::Unsound`]) rather than answer on a slice
    /// they cannot model — never fabricate a verdict.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] when the backend is unavailable or its
    /// invocation failed; this is distinct from a sound `Unsound`
    /// verdict.
    fn solve(
        &self,
        slice: &SsaLiftedSlice,
        options: SolveOptions,
    ) -> Result<SolverOutcome, SolverError>;

    /// Stable backend identifier (e.g. `"z3"`, `"cvc5"`).
    fn name(&self) -> &'static str;

    /// Whether this backend is a sound verdict source or a
    /// corroboration-only differential oracle.
    fn role(&self) -> SolverRole;
}
