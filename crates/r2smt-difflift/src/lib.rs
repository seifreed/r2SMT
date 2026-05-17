#![deny(missing_docs)]
//! Differential multi-lifter harness for r2SMT.
//!
//! r2SMT carries three independent instruction lowerings — Ghidra
//! SLEIGH P-code, radare2 ESIL, and the per-mnemonic Fase-C handlers.
//! In production they form a fall-through ladder
//! ([`r2smt_slicer::lift_slice`]); the first that succeeds wins. This
//! crate runs them **side by side** on the same instruction and asks
//! the SMT path whether they are *semantically* equivalent. A proven
//! disagreement is an engine-integrity defect: one of the lowerings is
//! unsound, and CI should fail on it.
//!
//! The harness is a pure-domain crate: it builds the equivalence query
//! ([`build_equivalence_query`]) but delegates the actual solve to the
//! wiring layer, so no solver dependency leaks in. See [`equiv`] for
//! the soundness posture — the harness may flag a disagreement or stay
//! silent, but it never fabricates one.

pub mod equiv;
pub mod lower;

pub use equiv::{AgreementStats, DiffVerdict, build_equivalence_query, classify_equivalence};
pub use lower::{Lowering, Lowerings, lower_all};
