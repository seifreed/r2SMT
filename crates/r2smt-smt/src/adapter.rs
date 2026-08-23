//! Port adapters wrapping the in-crate Z3 / CVC5 backends so the
//! composition root selects a solver through [`r2smt_solver_port`]
//! rather than a hard-coded match.

use r2smt_common::smt::SolveOptions;
use r2smt_solver_port::{Solver, SolverError, SolverOutcome, SolverRole};
use r2smt_ssa::SsaLiftedSlice;

use crate::{
    BitwuzlaError, Cvc5Error, solve_branch_bitwuzla, solve_branch_cvc5, solve_branch_with_pretty,
};

/// In-process Z3 backend. Authoritative and infallible by contract
/// (a slice it cannot model is reported as `Unsound`, never an error).
#[derive(Debug, Default, Clone, Copy)]
pub struct Z3Solver;

impl Solver for Z3Solver {
    fn solve(
        &self,
        slice: &SsaLiftedSlice,
        options: SolveOptions,
    ) -> Result<SolverOutcome, SolverError> {
        let outcome = solve_branch_with_pretty(slice, options);
        Ok(SolverOutcome {
            verdict: outcome.verdict,
            formula_pretty: outcome.formula_z3_pretty,
        })
    }

    fn name(&self) -> &'static str {
        "z3"
    }

    fn role(&self) -> SolverRole {
        SolverRole::Sound
    }
}

/// CVC5 subprocess backend (authoritative). The detail strings are
/// crafted so the composition root can prefix them and reproduce the
/// pre-port CLI error messages byte-for-byte.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cvc5Solver;

impl Solver for Cvc5Solver {
    fn solve(
        &self,
        slice: &SsaLiftedSlice,
        options: SolveOptions,
    ) -> Result<SolverOutcome, SolverError> {
        match solve_branch_cvc5(slice, options) {
            Ok(verdict) => Ok(SolverOutcome {
                verdict,
                formula_pretty: None,
            }),
            Err(Cvc5Error::NotFound(detail)) => Err(SolverError::Unavailable(format!(
                "cvc5 binary not found on PATH ({detail}); install it with `brew install cvc5` / `apt install cvc5`"
            ))),
            Err(Cvc5Error::SubprocessError(detail)) => {
                Err(SolverError::Backend(format!("subprocess failed: {detail}")))
            }
            Err(Cvc5Error::UnrecognisedVerdict(out)) => {
                Err(SolverError::Backend(format!("unrecognised stdout: {out}")))
            }
        }
    }

    fn name(&self) -> &'static str {
        "cvc5"
    }

    fn role(&self) -> SolverRole {
        SolverRole::Sound
    }
}

/// Bitwuzla subprocess backend (authoritative). A third independent
/// `QF_BV` opinion in the portfolio; structurally identical to
/// [`Cvc5Solver`], differing only in the underlying binary. The detail
/// strings mirror the CVC5 adapter's so the composition root can
/// prefix them uniformly.
#[derive(Debug, Default, Clone, Copy)]
pub struct BitwuzlaSolver;

impl Solver for BitwuzlaSolver {
    fn solve(
        &self,
        slice: &SsaLiftedSlice,
        options: SolveOptions,
    ) -> Result<SolverOutcome, SolverError> {
        match solve_branch_bitwuzla(slice, options) {
            Ok(verdict) => Ok(SolverOutcome {
                verdict,
                formula_pretty: None,
            }),
            Err(BitwuzlaError::NotFound(detail)) => Err(SolverError::Unavailable(format!(
                "bitwuzla binary not found on PATH ({detail}); install it with `brew install bitwuzla` or build from https://github.com/bitwuzla/bitwuzla"
            ))),
            Err(BitwuzlaError::SubprocessError(detail)) => {
                Err(SolverError::Backend(format!("subprocess failed: {detail}")))
            }
            Err(BitwuzlaError::UnrecognisedVerdict(out)) => {
                Err(SolverError::Backend(format!("unrecognised stdout: {out}")))
            }
        }
    }

    fn name(&self) -> &'static str {
        "bitwuzla"
    }

    fn role(&self) -> SolverRole {
        SolverRole::Sound
    }
}

/// Consensus backend requiring Z3, CVC5, and Bitwuzla to agree.
/// Missing backends are reported as errors; disagreements fail closed
/// to [`r2smt_common::smt::SmtResult::Unsound`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PortfolioSolver;

impl Solver for PortfolioSolver {
    fn solve(
        &self,
        slice: &SsaLiftedSlice,
        options: SolveOptions,
    ) -> Result<SolverOutcome, SolverError> {
        let z3 = Z3Solver.solve(slice, options)?;
        let cvc5 = Cvc5Solver.solve(slice, options)?;
        let bitwuzla = BitwuzlaSolver.solve(slice, options)?;
        Ok(consensus(z3, &cvc5, &bitwuzla))
    }

    fn name(&self) -> &'static str {
        "portfolio"
    }

    fn role(&self) -> SolverRole {
        SolverRole::Sound
    }
}

fn consensus(
    mut authoritative: SolverOutcome,
    cvc5: &SolverOutcome,
    bitwuzla: &SolverOutcome,
) -> SolverOutcome {
    if authoritative.verdict != cvc5.verdict || authoritative.verdict != bitwuzla.verdict {
        authoritative.verdict = r2smt_common::smt::SmtResult::Unsound;
    }
    authoritative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_advertise_sound_role_and_stable_names() {
        // The factory keys the verdict ladder off `role`; a sound
        // backend mislabelled as an oracle (or vice versa) would
        // break the reliability contract.
        assert_eq!(Z3Solver.name(), "z3");
        assert_eq!(Z3Solver.role(), SolverRole::Sound);
        assert_eq!(Cvc5Solver.name(), "cvc5");
        assert_eq!(Cvc5Solver.role(), SolverRole::Sound);
        assert_eq!(BitwuzlaSolver.name(), "bitwuzla");
        assert_eq!(BitwuzlaSolver.role(), SolverRole::Sound);
        assert_eq!(PortfolioSolver.name(), "portfolio");
        assert_eq!(PortfolioSolver.role(), SolverRole::Sound);
    }

    #[test]
    fn portfolio_requires_three_identical_verdicts() {
        let outcome = |verdict| SolverOutcome {
            verdict,
            formula_pretty: None,
        };
        let agreed = consensus(
            outcome(r2smt_common::smt::SmtResult::AlwaysTrue),
            &outcome(r2smt_common::smt::SmtResult::AlwaysTrue),
            &outcome(r2smt_common::smt::SmtResult::AlwaysTrue),
        );
        assert_eq!(agreed.verdict, r2smt_common::smt::SmtResult::AlwaysTrue);

        let disagreed = consensus(
            outcome(r2smt_common::smt::SmtResult::AlwaysTrue),
            &outcome(r2smt_common::smt::SmtResult::AlwaysFalse),
            &outcome(r2smt_common::smt::SmtResult::AlwaysTrue),
        );
        assert_eq!(disagreed.verdict, r2smt_common::smt::SmtResult::Unsound);
    }
}
