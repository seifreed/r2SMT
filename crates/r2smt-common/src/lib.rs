#![deny(missing_docs)]
//! Foundation primitives, error types, and shared DTOs for the r2SMT
//! workspace. No `r2smt-*` dependencies are allowed here.

pub mod error;
pub mod smt;
pub mod types;

pub use error::{Error, Result};
pub use smt::{SmtResult, SolveOptions};
pub use types::{Address, Arch};
