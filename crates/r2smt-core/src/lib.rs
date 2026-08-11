#![deny(missing_docs)]
//! Orchestration use cases for r2SMT.
//!
//! `r2smt-core` consumes `r2smt-ir` ports and produces application-level
//! results. It must not depend on any adapter crate (`r2smt-r2pipe`,
//! future SMT bindings, …) — those are wired together at the binary
//! layer.

pub mod analyzer;
pub mod consensus;
pub mod dump;
pub mod finding;
pub mod prepare;

pub use analyzer::{Analyzer, AnalyzerConfig};
pub use consensus::{OracleReconciliation, reconcile_finding_with_oracle, reconcile_with_oracle};
pub use dump::dump_program;
pub use finding::{
    Confidence, Finding, FindingEvidence, FindingKind, OracleAgreement, classify_finding,
    classify_finding_with_hints, classify_finding_with_pretty, classify_lowered_upstream,
    lifter_disagreement_finding, reconcile_folded,
};
pub use prepare::prepare_ssa;
