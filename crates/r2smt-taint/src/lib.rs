#![deny(missing_docs)]
//! Sound may-taint analysis over the r2SMT IR (roadmap E2 / phase 34).
//!
//! Given an SSA-renamed slice and a set of seeded taint sources, this
//! crate answers "which values may carry data derived from source S?".
//! It generalises the boolean `Unknown`-taint pass in `r2smt-difflift`
//! to a multi-source join-semilattice ([`TaintSet`]), so several sources
//! can be tracked at once (e.g. `argv[1]` and `stdin` independently).
//!
//! The propagation ([`propagate()`]) is the *sound subset* of the taint
//! story: exact for register data flow and a sound over-approximation
//! for memory, with an [`TaintOutcome::is_opaque`] flag when an
//! unmodelled node (`Unknown` / `Unsupported` / truncation) makes the
//! answer best-effort. Completing opaque flows via concretisation is the
//! unsound subset, delegated to the fenced exploration engine — never
//! done here.
//!
//! Pure domain: depends only on `r2smt-common`, `r2smt-ir`,
//! `r2smt-slicer`, and `r2smt-ssa`. No solver or adapter dependency.

pub mod lattice;
pub mod propagate;

pub use lattice::{SourceId, TaintSet};
pub use propagate::{TaintOutcome, TaintSeeds, propagate};
