#![deny(missing_docs)]
//! Z3 backend for r2SMT.
//!
//! Consumes [`r2smt_ssa::SsaLiftedSlice`]s, translates the bit-vector
//! IR into Z3 ASTs, and reports a verdict for each branch via
//! [`SmtResult`]:
//!
//! - [`SmtResult::AlwaysTrue`]: the condition is `SAT` under
//!   `cond == 1` and `UNSAT` under `cond == 0`.
//! - [`SmtResult::AlwaysFalse`]: the dual.
//! - [`SmtResult::BothPossible`]: both polarities are satisfiable —
//!   the branch is a genuine choice.
//! - [`SmtResult::Unsound`]: both polarities are `UNSAT` (an encoding
//!   bug; should not happen with the current lifter).
//! - [`SmtResult::Timeout`] / [`SmtResult::Unknown`]: solver could not
//!   conclude within the budget.

pub mod cvc5;
pub mod encoder;
pub mod pretty;
pub mod smtlib;
pub mod solver;

pub use cvc5::{Cvc5Error, solve_branch_cvc5};
pub use pretty::z3_bool_to_infix;
pub use r2smt_common::smt::{SmtResult, SolveOptions};
pub use smtlib::{emit_preamble, emit_query};
pub use solver::{SolveOutcome, solve_branch, solve_branch_with_pretty};
