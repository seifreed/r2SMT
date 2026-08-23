#![deny(missing_docs)]
//! Report renderers for r2SMT.
//!
//! Consumes a slice of [`r2smt_core::Finding`]s plus program-level
//! metadata and emits:
//!
//! - [`Report::render_json`] — stable, serde-friendly JSON.
//! - [`Report::render_markdown`] — human-readable Markdown.
//! - [`Report::render_r2_script`] — radare2 `CCu` comments and
//!   commented-out `wa` patch suggestions (one line per actionable
//!   finding), suitable for `r2 -i annotations.r2`.

pub mod batch_report;
mod patch_suggestion;
pub mod report;
pub mod unsound;

pub use batch_report::{
    BatchOutcome, BatchReport, BatchSampleEntry, BatchSampleSummary, MAX_FINDINGS_PER_SAMPLE,
};
pub use patch_suggestion::{PatchStrategy, PatchSuggestion, suggest_patch};
pub use report::{
    AnalysisMetrics, AnalysisOptions, AnalysisReasonCode, Annotation, FindingDiagnostic, IrPolicy,
    KindCounts, REPORT_SCHEMA_VERSION, Report, ReportMetadata, SolverProvenance,
};
pub use unsound::{UNSOUND_BANNER, wrap_unsound};
