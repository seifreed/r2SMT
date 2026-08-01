#![deny(missing_docs)]
//! Fenced exploration engine for r2SMT (roadmap E1 / phase 31).
//!
//! The verifier core ([`r2smt_smt`](https://docs.rs) and friends) only
//! ever produces *sound* verdicts: it never claims a branch is dead
//! unless it proved it. This crate is the opposite tool — an
//! **exploration** engine that runs a symbolic executor (radius2, which
//! embeds radare2 + an SMT layer) to *search* for a concrete input that
//! drives execution to a target address, and reports the witness it
//! found. Search is best-effort and path-bounded: a negative answer
//! only means "not found within the budget", never "unreachable".
//!
//! Because its outputs are unsound, three fences keep it from
//! contaminating the verifier:
//!
//! 1. **Type fence.** The public result type is [`ExploreResult`], which
//!    carries a concrete [`Model`] witness — never a
//!    [`r2smt_common::SmtResult`]. There is deliberately no
//!    `From<ExploreResult> for SmtResult`; a compile-fail test
//!    (`tests/type_fence.rs`) proves the conversion does not exist.
//! 2. **Dependency fence.** The engine (radare2 / radius2) lives behind
//!    the off-by-default `oracle-radius2` feature and a subprocess
//!    worker, so the default dependency graph of the verify/patch crates
//!    never reaches this crate. `tests/dep_fence.rs` walks `cargo
//!    metadata` and fails if it ever does.
//! 3. **Process + budget fence.** Exploration always runs in a spawned
//!    [worker subprocess](crate::driver::run_worker) under a host-side
//!    wall-clock deadline and a path-explosion cap, so a runaway search
//!    cannot take down the host that called it.
//!
//! Rendering any explore output to a human must go through the
//! non-suppressible `r2smt_report::UNSOUND_BANNER`.

pub mod driver;
#[cfg(feature = "oracle-radius2")]
pub mod engine;
pub mod model;

pub use driver::{WaitOutcome, WorkerSpec, explore, run_worker, spawn_and_wait};
pub use model::{
    ExploreBudget, ExploreError, ExploreRequest, ExploreResult, MAX_ARGV, MAX_REGS,
    MAX_STDIN_BYTES, Model,
};
