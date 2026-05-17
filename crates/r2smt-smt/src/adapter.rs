//! Port adapters wrapping the in-crate Z3 / CVC5 backends so the
//! composition root selects a solver through [`r2smt_solver_port`]
//! rather than a hard-coded match.

use r2smt_common::smt::SolveOptions;
use r2smt_solver_port::{Solver, SolverError, SolverOutcome, SolverRole};
use r2smt_ssa::SsaLiftedSlice;

use crate::{Cvc5Error, solve_branch_cvc5, solve_branch_with_pretty};

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
    }
}
