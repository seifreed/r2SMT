#![deny(missing_docs)]
//! Rename a [`r2smt_slicer::LiftedSlice`] into Static Single Assignment
//! form.
//!
//! Each [`r2smt_ir::Var`] written by the slice gets a `#N` suffix on its
//! name; reads are rewired to the most recent definition available at
//! the point of use. Variables that are read before being defined inside
//! the slice keep their plain name and are reported as `inputs` — the
//! SMT backend will treat them as free symbolic values.

pub mod convert;
pub mod optimize;
pub mod pretty;

pub use convert::{SsaLiftedSlice, ssa_convert};
pub use optimize::optimize_slice;
pub use pretty::{pretty_condition, pretty_condition_with_hints};
