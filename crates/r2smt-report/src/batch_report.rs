//! Aggregated multi-sample report for `r2smt batch`.
//!
//! A corpus sweep can produce tens of thousands of findings *per
//! sample*. Per the Host-Side Safety guardrail the aggregate keeps
//! only exact counts plus a bounded slice of the most actionable
//! findings ([`MAX_FINDINGS_PER_SAMPLE`]), so total retention stays
//! `O(samples)` rather than `O(samples × findings)`.

use std::fmt::Write as _;

use r2smt_common::Arch;
use r2smt_core::Finding;
use serde::{Deserialize, Serialize};

use crate::report::{KindCounts, Report};

/// Upper bound on actionable findings retained per sample. Counts are
/// always exact (`actionable_total`); only the detail list is capped.
pub const MAX_FINDINGS_PER_SAMPLE: usize = 200;

/// Aggregated result of analysing a directory of samples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchReport {
    /// r2SMT version string.
    pub r2smt_version: String,
    /// Display path of the swept root directory.
    pub root: String,
    /// One entry per sample file, in deterministic (sorted-path) order.
    pub samples: Vec<BatchSampleEntry>,
}

/// One sample file's result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSampleEntry {
    /// Display path of the sample.
    pub path: String,
    /// Analysis outcome.
    pub outcome: BatchOutcome,
}

/// Per-sample success or failure. Failure never aborts the sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BatchOutcome {
    /// The sample was analysed; carries its bounded summary.
    Analyzed {
        /// Bounded per-sample summary.
        summary: BatchSampleSummary,
    },
    /// The sample failed to analyse; carries a human-readable reason.
    Failed {
        /// Failure reason (formatted error chain).
        error: String,
    },
}

/// Bounded summary of a single sample's [`Report`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSampleSummary {
    /// Target architecture.
    pub arch: Arch,
    /// Pointer width in bits.
    pub bits: u8,
    /// Functions discovered by radare2.
    pub functions_analyzed: usize,
    /// Conditional branches considered.
    pub branches_analyzed: usize,
    /// Aggregated counts by `FindingKind` (always exact).
    pub summary: KindCounts,
    /// Total actionable findings before the per-sample cap.
    pub actionable_total: usize,
    /// `true` when `actionable_total > MAX_FINDINGS_PER_SAMPLE` and
    /// `top_findings` is therefore a prefix, not the full set.
    pub findings_truncated: bool,
    /// The most actionable findings (sorted by confidence then
    /// address), capped at [`MAX_FINDINGS_PER_SAMPLE`].
    pub top_findings: Vec<Finding>,
}

impl BatchSampleSummary {
    /// Downsample a full [`Report`] into the bounded summary, dropping
    /// the unbounded finding vector at the call site.
    #[must_use]
    pub fn from_report(report: &Report) -> Self {
        let mut actionable: Vec<Finding> = report
            .findings
            .iter()
            .filter(|f| f.is_actionable())
            .cloned()
            .collect();
        actionable.sort_by_key(|f| (f.confidence, f.address));
        let actionable_total = actionable.len();
        let findings_truncated = actionable_total > MAX_FINDINGS_PER_SAMPLE;
        actionable.truncate(MAX_FINDINGS_PER_SAMPLE);
        Self {
            arch: report.arch,
            bits: report.bits,
            functions_analyzed: report.functions_analyzed,
            branches_analyzed: report.branches_analyzed,
            summary: report.summary,
            actionable_total,
            findings_truncated,
            top_findings: actionable,
        }
    }
}

impl BatchReport {
    /// Assemble the aggregate from per-sample entries.
    #[must_use]
    pub fn new(
        version: impl Into<String>,
        root: impl Into<String>,
        samples: Vec<BatchSampleEntry>,
    ) -> Self {
        Self {
            r2smt_version: version.into(),
            root: root.into(),
            samples,
        }
    }

    /// Render the aggregate as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`serde_json::Error`] on serialisation failure.
    pub fn render_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render the aggregate as a Markdown table plus a totals row.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# r2SMT Batch Report");
        let _ = writeln!(s);
        let _ = writeln!(s, "| Field | Value |");
        let _ = writeln!(s, "|-------|-------|");
        let _ = writeln!(s, "| r2SMT version | {} |", self.r2smt_version);
        let _ = writeln!(s, "| Root | `{}` |", self.root);
        let _ = writeln!(s, "| Samples | {} |", self.samples.len());
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "| Sample | Status | Branches | Opaque | Dead | Const | Actionable |"
        );
        let _ = writeln!(
            s,
            "|--------|--------|----------|--------|------|-------|------------|"
        );

        let mut analyzed = 0usize;
        let mut failed = 0usize;
        let mut tot_branches = 0usize;
        let mut tot_opaque = 0usize;
        let mut tot_dead = 0usize;
        let mut tot_const = 0usize;
        let mut tot_actionable = 0usize;

        for entry in &self.samples {
            match &entry.outcome {
                BatchOutcome::Analyzed { summary } => {
                    analyzed += 1;
                    tot_branches += summary.branches_analyzed;
                    tot_opaque += summary.summary.opaque_predicate;
                    tot_dead += summary.summary.dead_branch;
                    tot_const += summary.summary.constant_condition;
                    tot_actionable += summary.actionable_total;
                    let trunc = if summary.findings_truncated { "+" } else { "" };
                    let _ = writeln!(
                        s,
                        "| `{path}` | ok | {branches} | {opaque} | {dead} | {konst} | {act}{trunc} |",
                        path = entry.path,
                        branches = summary.branches_analyzed,
                        opaque = summary.summary.opaque_predicate,
                        dead = summary.summary.dead_branch,
                        konst = summary.summary.constant_condition,
                        act = summary.actionable_total,
                    );
                }
                BatchOutcome::Failed { error } => {
                    failed += 1;
                    let _ = writeln!(
                        s,
                        "| `{path}` | FAILED: {error} | - | - | - | - | - |",
                        path = entry.path,
                        error = error.replace('|', "\\|").replace('\n', " "),
                    );
                }
            }
        }

        let _ = writeln!(
            s,
            "| **total** | {analyzed} ok / {failed} failed | {tot_branches} | {tot_opaque} | {tot_dead} | {tot_const} | {tot_actionable} |"
        );
        s
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::smt::SmtResult;
    use r2smt_common::{Address, Arch};
    use r2smt_core::{Confidence, Finding, FindingEvidence, FindingKind};
    use r2smt_slicer::condition::BranchCondition;
    use r2smt_slicer::slice::SliceStatus;

    use super::*;

    fn finding(kind: FindingKind, addr: u64) -> Finding {
        Finding {
            address: Address(addr),
            function: Address(0x40_1000),
            mnemonic: "jne".into(),
            condition: BranchCondition::NotEqual,
            formula: "ZF == 0".into(),
            formula_pretty: "(ZF == 0)".into(),
            formula_z3_pretty: None,
            verdict: SmtResult::AlwaysTrue,
            kind,
            confidence: Confidence::High,
            taken_target: Some(Address(0x40_1080)),
            fallthrough_target: Some(Address(addr + 2)),
            operands: Vec::new(),
            is_thumb: false,
            evidence: FindingEvidence {
                slice_status: SliceStatus::Complete,
                statement_count: 1,
                input_count: 0,
                inputs: vec![],
                unknown_count: 0,
                upstream_resolved_to: None,
            },
            pseudocode: None,
        }
    }

    fn report_with(n_opaque: usize) -> Report {
        let findings: Vec<Finding> = (0..n_opaque)
            .map(|i| finding(FindingKind::OpaquePredicate, 0x40_1000 + (i as u64) * 4))
            .collect();
        Report::from_findings("test", "/x/sample", Arch::X86_64, 64, 3, findings)
    }

    #[test]
    fn test_from_report_caps_findings_and_flags_truncation() {
        let report = report_with(MAX_FINDINGS_PER_SAMPLE + 50);
        let summary = BatchSampleSummary::from_report(&report);
        assert_eq!(summary.actionable_total, MAX_FINDINGS_PER_SAMPLE + 50);
        assert!(summary.findings_truncated);
        assert_eq!(summary.top_findings.len(), MAX_FINDINGS_PER_SAMPLE);
    }

    #[test]
    fn test_from_report_keeps_all_when_under_cap() {
        let report = report_with(10);
        let summary = BatchSampleSummary::from_report(&report);
        assert!(!summary.findings_truncated);
        assert_eq!(summary.top_findings.len(), 10);
    }

    #[test]
    fn test_render_json_roundtrips_and_preserves_sample_order() {
        let samples = vec![
            BatchSampleEntry {
                path: "/x/a".into(),
                outcome: BatchOutcome::Analyzed {
                    summary: BatchSampleSummary::from_report(&report_with(2)),
                },
            },
            BatchSampleEntry {
                path: "/x/b".into(),
                outcome: BatchOutcome::Failed {
                    error: "radare2 spawn failed".into(),
                },
            },
        ];
        let report = BatchReport::new("test", "/x", samples);
        let json = report.render_json().unwrap();
        let back: BatchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
        assert_eq!(back.samples[0].path, "/x/a");
        assert_eq!(back.samples[1].path, "/x/b");
    }

    #[test]
    fn test_render_markdown_lists_failed_and_total_rows() {
        let samples = vec![
            BatchSampleEntry {
                path: "/x/ok".into(),
                outcome: BatchOutcome::Analyzed {
                    summary: BatchSampleSummary::from_report(&report_with(3)),
                },
            },
            BatchSampleEntry {
                path: "/x/bad".into(),
                outcome: BatchOutcome::Failed {
                    error: "boom".into(),
                },
            },
        ];
        let md = BatchReport::new("test", "/x", samples).render_markdown();
        assert!(md.contains("FAILED: boom"));
        assert!(md.contains("| **total** | 1 ok / 1 failed |"));
    }
}
