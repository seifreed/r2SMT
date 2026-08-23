//! `Report` aggregates program metadata plus a `Vec<Finding>` and
//! exposes three renderers.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use r2smt_common::{Address, Arch};
use r2smt_core::{Finding, FindingKind};
use r2smt_ir::program::Function;
use r2smt_slicer::slice::SliceStatus;
use serde::{Deserialize, Serialize};

use crate::patch_suggestion::suggest_patch;

fn unknown_tool_version() -> String {
    "unknown".to_string()
}

/// Stable JSON schema emitted by [`Report`].
pub const REPORT_SCHEMA_VERSION: u32 = 1;

const fn report_schema_version() -> u32 {
    REPORT_SCHEMA_VERSION
}

/// Solver identity and exact budgets used for the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverProvenance {
    /// Stable backend name (`z3`, `cvc5`, or `bitwuzla`).
    pub name: String,
    /// Backend version reported by the installed executable or library.
    pub version: String,
    /// Per-query wall-clock budget.
    pub timeout_ms: u32,
    /// Deterministic resource budget (`0` means unset).
    pub rlimit: u32,
    /// Pinned backend random seed.
    pub random_seed: u32,
}

impl Default for SolverProvenance {
    fn default() -> Self {
        Self {
            name: "unknown".into(),
            version: unknown_tool_version(),
            timeout_ms: 0,
            rlimit: 0,
            random_seed: 0,
        }
    }
}

/// Ordered lowering policy used to obtain the typed IR.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrPolicy {
    /// radare2 ESIL with per-mnemonic fallback.
    #[default]
    EsilMnemonic,
    /// r2ghidra P-code, then ESIL, then per-mnemonic fallback.
    PcodeEsilMnemonic,
}

/// Analysis knobs that materially affect report results.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisOptions {
    /// Whether radare2 used the deeper `aaaa` analysis level.
    pub deep: bool,
    /// Optional single branch address requested by the caller.
    pub branch: Option<Address>,
    /// Optional single function address requested by the caller.
    pub function: Option<Address>,
    /// Maximum instructions retained per backward slice.
    pub max_slice_instructions: usize,
    /// Maximum basic blocks traversed per slice.
    pub max_basic_blocks: u32,
    /// Whether slicing follows memory effects.
    pub allow_memory: bool,
    /// Whether slicing follows calls.
    pub allow_calls: bool,
    /// Whether truncated roots become free symbolic inputs.
    pub unknowns_on_truncation: bool,
    /// Whether bounded join recovery is enabled.
    pub allow_join_merge: bool,
    /// Whether the ESIL flag-writing rung is enabled.
    pub esil_flags: bool,
    /// Whether differential lifting was requested.
    pub differential_lift: bool,
    /// Whether decompiler context was attached.
    pub with_decompiler: bool,
    /// Whether the authoritative pipeline ran in a sandboxed worker.
    pub worker_isolated: bool,
    /// Total per-sample worker deadline.
    pub worker_wall_clock_ms: u64,
    /// Aggregate worker process-group RSS limit.
    pub worker_memory_mib: u64,
    /// Maximum functions accepted from the provider.
    pub max_functions: usize,
    /// Maximum branch candidates accepted before solving.
    pub max_branches: usize,
}

/// Stable machine-readable reason for an inconclusive finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisReasonCode {
    /// Solver exhausted its time or resource budget.
    SolverTimeout,
    /// Solver returned unknown for another reason.
    SolverUnknown,
    /// The soundness gate declined the model.
    UnsoundModel,
    /// Candidate block was absent from the function model.
    BlockNotFound,
    /// Candidate instruction was absent from its block.
    InstructionNotFound,
    /// A configured slice budget was exhausted.
    SliceBudget,
    /// Backward traversal encountered a cycle.
    SliceCycle,
    /// Backward traversal reached an unresolved CFG join.
    SliceJoin,
    /// The slice crossed disabled or unmodellable memory.
    MemoryBoundary,
    /// The slice crossed a disabled or unmodellable call.
    CallBoundary,
    /// An instruction or semantic effect is unsupported.
    UnsupportedInstruction,
    /// A floating-point control word made the active mode unknown.
    FloatingPointControl,
    /// No sound definition for the branch flags was found.
    MissingFlagDefinition,
    /// A CFG predecessor referenced by radare2 was unavailable.
    MissingPredecessor,
    /// Truncation did not match a more specific stable category.
    OtherTruncation,
}

/// Stable code plus the original human-readable diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FindingDiagnostic {
    /// Finding address this diagnostic belongs to.
    pub address: Address,
    /// Stable reason code suitable for aggregation.
    pub code: AnalysisReasonCode,
    /// Additional human-readable detail, not part of the stable code.
    pub detail: Option<String>,
}

/// Metrics computed from the complete finding set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    /// Number of functions discovered by radare2.
    pub functions_analyzed: usize,
    /// Number of candidate branches considered.
    pub branches_analyzed: usize,
    /// Findings backed by complete slices.
    pub complete_slices: usize,
    /// Findings backed by truncated slices.
    pub truncated_slices: usize,
    /// Findings with a definitive true/false/both-possible verdict.
    pub definitive_findings: usize,
    /// Findings eligible for annotation or patch planning.
    pub actionable_findings: usize,
    /// Inconclusive findings grouped by stable code.
    pub unknown_by_code: BTreeMap<AnalysisReasonCode, usize>,
}

/// Provenance and options attached by the application composition root.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReportMetadata {
    /// Exact radare2 version.
    pub radare2_version: String,
    /// Exact r2ghidra version.
    pub r2ghidra_version: String,
    /// Solver identity and budgets.
    pub solver: SolverProvenance,
    /// Ordered IR fallback policy.
    pub ir_policy: IrPolicy,
    /// SHA-256 of the analyzed input.
    pub binary_sha256: String,
    /// Analysis knobs that affect the result.
    pub analysis_options: AnalysisOptions,
}

/// A single textual annotation derived from a [`Finding`].
///
/// Used both as the source of truth for the `CCu` lines emitted by
/// [`Report::render_r2_script`] and as the unit of work the CLI's
/// `annotate` subcommand sends through an [`r2smt_ir::Annotator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Annotation {
    /// Address the annotation attaches to.
    pub address: Address,
    /// Comment payload. May span multiple lines (a decompiled preview);
    /// consumers that write it into a line-oriented context must
    /// base64-encode it, as `Annotator::set_comment` and the r2-script
    /// renderer both do.
    pub text: String,
}

/// Count per `FindingKind`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindCounts {
    /// Number of [`FindingKind::OpaquePredicate`] findings.
    pub opaque_predicate: usize,
    /// Number of [`FindingKind::DeadBranch`] findings.
    pub dead_branch: usize,
    /// Number of [`FindingKind::ConstantCondition`] findings.
    pub constant_condition: usize,
    /// Number of [`FindingKind::RealBranch`] findings.
    pub real_branch: usize,
    /// Number of [`FindingKind::SuspiciousButUnknown`] (or future
    /// non-exhaustive) findings.
    pub suspicious_but_unknown: usize,
}

impl KindCounts {
    /// Sum of the actionable kinds (opaque / dead / constant).
    #[must_use]
    pub const fn actionable(self) -> usize {
        self.opaque_predicate + self.dead_branch + self.constant_condition
    }

    /// Total of every counted bucket.
    #[must_use]
    pub const fn total(self) -> usize {
        self.opaque_predicate
            + self.dead_branch
            + self.constant_condition
            + self.real_branch
            + self.suspicious_but_unknown
    }
}

/// Top-level r2SMT report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    /// Stable JSON schema version.
    #[serde(default = "report_schema_version")]
    pub schema_version: u32,
    /// r2SMT version string (typically `env!("CARGO_PKG_VERSION")` from
    /// the CLI).
    pub r2smt_version: String,
    /// Exact radare2 version used for the analysis.
    #[serde(default = "unknown_tool_version")]
    pub radare2_version: String,
    /// Exact r2ghidra version used for P-code, or `unavailable`.
    #[serde(default = "unknown_tool_version")]
    pub r2ghidra_version: String,
    /// Solver identity, version, and budgets.
    #[serde(default)]
    pub solver: SolverProvenance,
    /// Ordered IR fallback policy.
    #[serde(default)]
    pub ir_policy: IrPolicy,
    /// SHA-256 of the analyzed binary.
    #[serde(default)]
    pub binary_sha256: String,
    /// Analysis knobs that materially affect results.
    #[serde(default)]
    pub analysis_options: AnalysisOptions,
    /// Display path of the analysed binary.
    pub binary: String,
    /// Target architecture.
    pub arch: Arch,
    /// Pointer width in bits.
    pub bits: u16,
    /// Number of functions discovered by radare2.
    pub functions_analyzed: usize,
    /// Total number of conditional branches considered.
    pub branches_analyzed: usize,
    /// Aggregated counts by `FindingKind`.
    pub summary: KindCounts,
    /// Metrics derived from the complete finding set.
    #[serde(default)]
    pub metrics: AnalysisMetrics,
    /// Stable inconclusive-reason codes with human-readable detail.
    #[serde(default)]
    pub diagnostics: Vec<FindingDiagnostic>,
    /// Findings, in the same order they were emitted by the solver.
    pub findings: Vec<Finding>,
}

impl Report {
    /// Build a `Report` from program metadata and findings.
    #[must_use]
    pub fn from_findings(
        version: impl Into<String>,
        binary: impl Into<String>,
        arch: Arch,
        bits: u16,
        functions_analyzed: usize,
        findings: Vec<Finding>,
    ) -> Self {
        let summary = count_kinds(&findings);
        let branches_analyzed = findings.len();
        let (metrics, diagnostics) = analysis_metrics(functions_analyzed, &findings);
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            r2smt_version: version.into(),
            radare2_version: unknown_tool_version(),
            r2ghidra_version: unknown_tool_version(),
            solver: SolverProvenance::default(),
            ir_policy: IrPolicy::default(),
            binary_sha256: String::new(),
            analysis_options: AnalysisOptions::default(),
            binary: binary.into(),
            arch,
            bits,
            functions_analyzed,
            branches_analyzed,
            summary,
            metrics,
            diagnostics,
            findings,
        }
    }

    /// Attach the exact radare2 version used to produce this report.
    #[must_use]
    pub fn with_radare2_version(mut self, version: impl Into<String>) -> Self {
        self.radare2_version = version.into();
        self
    }

    /// Attach complete application provenance and result-affecting options.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ReportMetadata) -> Self {
        self.radare2_version = metadata.radare2_version;
        self.r2ghidra_version = metadata.r2ghidra_version;
        self.solver = metadata.solver;
        self.ir_policy = metadata.ir_policy;
        self.binary_sha256 = metadata.binary_sha256;
        self.analysis_options = metadata.analysis_options;
        self
    }

    /// Render the report as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`serde_json::Error`] if serialisation fails (e.g.
    /// a non-UTF-8 byte string sneaks into a finding — should not
    /// happen with the current types).
    pub fn render_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render the report as a Markdown document.
    #[must_use]
    pub fn render_markdown(&self, functions: &[Function]) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# r2SMT Report");
        let _ = writeln!(s);
        let _ = writeln!(s, "| Field | Value |");
        let _ = writeln!(s, "|-------|-------|");
        let _ = writeln!(s, "| r2SMT version | {} |", self.r2smt_version);
        let _ = writeln!(s, "| radare2 version | {} |", self.radare2_version);
        let _ = writeln!(s, "| Binary | `{}` |", self.binary);
        let _ = writeln!(s, "| Architecture | {:?} ({} bits) |", self.arch, self.bits);
        let _ = writeln!(s, "| Functions analyzed | {} |", self.functions_analyzed);
        let _ = writeln!(s, "| Branches analyzed | {} |", self.branches_analyzed);
        let _ = writeln!(s);
        let _ = writeln!(s, "## Summary");
        let _ = writeln!(s);
        let _ = writeln!(s, "| Kind | Count |");
        let _ = writeln!(s, "|------|-------|");
        let _ = writeln!(
            s,
            "| Opaque predicate | {} |",
            self.summary.opaque_predicate
        );
        let _ = writeln!(s, "| Dead branch | {} |", self.summary.dead_branch);
        let _ = writeln!(
            s,
            "| Constant condition | {} |",
            self.summary.constant_condition
        );
        let _ = writeln!(s, "| Real branch | {} |", self.summary.real_branch);
        let _ = writeln!(
            s,
            "| Suspicious | {} |",
            self.summary.suspicious_but_unknown
        );
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "**Actionable**: {act} / {total} ({pct:.2}%)",
            act = self.summary.actionable(),
            total = self.summary.total().max(1),
            pct = pct(self.summary.actionable(), self.summary.total()),
        );

        let mut actionables: Vec<&Finding> =
            self.findings.iter().filter(|f| f.is_actionable()).collect();
        if !actionables.is_empty() {
            actionables.sort_by_key(|f| (f.confidence, f.address));
            let _ = writeln!(s);
            let _ = writeln!(s, "## Actionable findings");
            let _ = writeln!(s);
            for f in actionables {
                render_finding_markdown(&mut s, f, functions);
            }
        }
        s
    }

    /// Build the list of annotations the script would emit, in the same
    /// order. Each entry corresponds to one actionable finding and is
    /// safe to feed through any [`r2smt_ir::Annotator`].
    #[must_use]
    pub fn annotations(&self, functions: &[Function]) -> Vec<Annotation> {
        let mut out: Vec<Annotation> = Vec::new();
        for f in &self.findings {
            if !f.is_actionable() {
                continue;
            }
            let fname = function_name(functions, f.function);
            out.push(Annotation {
                address: f.address,
                text: annotation_text(f, &fname),
            });
        }
        out
    }

    /// Render the report as a radare2 script suitable for
    /// `r2 -i annotations.r2`.
    #[must_use]
    pub fn render_r2_script(&self, functions: &[Function]) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# r2SMT annotations");
        let _ = writeln!(
            s,
            "# binary: {} ({:?}, {} bits)",
            self.binary, self.arch, self.bits
        );
        let _ = writeln!(
            s,
            "# total findings: {total}, actionable: {act}",
            total = self.branches_analyzed,
            act = self.summary.actionable(),
        );
        let _ = writeln!(s);
        for f in &self.findings {
            if !f.is_actionable() {
                continue;
            }
            let fname = function_name(functions, f.function);
            // Base64-encode exactly as the live `Annotator::set_comment`
            // path does: the payload carries newlines (decompiled preview)
            // and formula text with `;`, `@`, `>` — all r2 command
            // metacharacters that would otherwise split the line and run
            // the pseudocode as commands.
            let encoded = r2smt_common::b64::encode(annotation_text(f, &fname).as_bytes());
            let _ = writeln!(s, "CCu base64:{encoded} @ {addr}", addr = f.address);
            if let Some(patch) = suggest_patch(f) {
                let _ = writeln!(s, "#   strategy: {}", patch.strategy.as_str());
                let _ = writeln!(s, "#   rationale: {}", patch.rationale);
                let _ = writeln!(s, "#   patch: {}", patch.r2_command);
            }
            let _ = writeln!(s);
        }
        s
    }
}

/// Lines of decompiled context appended to a `CCu` annotation. Kept
/// small on purpose: the comment is for orientation, not a full
/// listing, and the base64 `CCu` payload should stay compact.
const PSEUDO_ANNOTATION_PREVIEW_LINES: usize = 6;

fn annotation_text(f: &Finding, fname: &str) -> String {
    let mut text = format!(
        "r2SMT: {kind:?}/{conf:?} -- {formula} -- {fname}",
        kind = f.kind,
        conf = f.confidence,
        formula = f.formula,
    );
    if let Some(code) = &f.pseudocode {
        let preview: Vec<&str> = code.lines().take(PSEUDO_ANNOTATION_PREVIEW_LINES).collect();
        if !preview.is_empty() {
            text.push_str("\n-- decompiled --\n");
            text.push_str(&preview.join("\n"));
            if code.lines().count() > PSEUDO_ANNOTATION_PREVIEW_LINES {
                text.push_str("\n…");
            }
        }
    }
    text
}

fn count_kinds(findings: &[Finding]) -> KindCounts {
    let mut c = KindCounts::default();
    for f in findings {
        match f.kind {
            FindingKind::OpaquePredicate => c.opaque_predicate += 1,
            FindingKind::DeadBranch => c.dead_branch += 1,
            FindingKind::ConstantCondition => c.constant_condition += 1,
            FindingKind::RealBranch => c.real_branch += 1,
            // `FindingKind` is `#[non_exhaustive]`; bucket SuspiciousButUnknown
            // and any future variants together.
            _ => c.suspicious_but_unknown += 1,
        }
    }
    c
}

fn analysis_metrics(
    functions_analyzed: usize,
    findings: &[Finding],
) -> (AnalysisMetrics, Vec<FindingDiagnostic>) {
    let mut metrics = AnalysisMetrics {
        functions_analyzed,
        branches_analyzed: findings.len(),
        ..AnalysisMetrics::default()
    };
    let mut diagnostics = Vec::new();
    for finding in findings {
        match &finding.evidence.slice_status {
            SliceStatus::Complete => metrics.complete_slices += 1,
            SliceStatus::Truncated { reason } => {
                metrics.truncated_slices += 1;
                diagnostics.push(FindingDiagnostic {
                    address: finding.address,
                    code: truncation_code(reason),
                    detail: Some(reason.clone()),
                });
            }
        }
        if matches!(
            finding.verdict,
            r2smt_common::smt::SmtResult::AlwaysTrue
                | r2smt_common::smt::SmtResult::AlwaysFalse
                | r2smt_common::smt::SmtResult::BothPossible
        ) {
            metrics.definitive_findings += 1;
        }
        if finding.is_actionable() {
            metrics.actionable_findings += 1;
        }
        if matches!(finding.evidence.slice_status, SliceStatus::Complete) {
            let code = match finding.verdict {
                r2smt_common::smt::SmtResult::Timeout => Some(AnalysisReasonCode::SolverTimeout),
                r2smt_common::smt::SmtResult::Unknown => Some(AnalysisReasonCode::SolverUnknown),
                r2smt_common::smt::SmtResult::Unsound => Some(AnalysisReasonCode::UnsoundModel),
                _ => None,
            };
            if let Some(code) = code {
                diagnostics.push(FindingDiagnostic {
                    address: finding.address,
                    code,
                    detail: None,
                });
            }
        }
    }
    for diagnostic in &diagnostics {
        *metrics.unknown_by_code.entry(diagnostic.code).or_default() += 1;
    }
    (metrics, diagnostics)
}

fn truncation_code(reason: &str) -> AnalysisReasonCode {
    let reason = reason.to_ascii_lowercase();
    if reason.contains("candidate instruction not found") {
        AnalysisReasonCode::InstructionNotFound
    } else if reason.contains("block not found") {
        AnalysisReasonCode::BlockNotFound
    } else if reason.contains("budget") || reason.contains("instruction limit") {
        AnalysisReasonCode::SliceBudget
    } else if reason.contains("cycle") || reason.contains("loop") {
        AnalysisReasonCode::SliceCycle
    } else if reason.contains("join") || reason.contains("predecessors") {
        AnalysisReasonCode::SliceJoin
    } else if reason.contains("memory") || reason.contains("load") || reason.contains("store") {
        AnalysisReasonCode::MemoryBoundary
    } else if reason.contains("call") {
        AnalysisReasonCode::CallBoundary
    } else if reason.contains("unsupported") || reason.contains("unmodellable") {
        AnalysisReasonCode::UnsupportedInstruction
    } else if reason.contains("fpu control") || reason.contains("rounding mode") {
        AnalysisReasonCode::FloatingPointControl
    } else if reason.contains("flag-defining") {
        AnalysisReasonCode::MissingFlagDefinition
    } else if reason.contains("predecessor block") {
        AnalysisReasonCode::MissingPredecessor
    } else {
        AnalysisReasonCode::OtherTruncation
    }
}

fn render_finding_markdown(s: &mut String, f: &Finding, functions: &[Function]) {
    let fname = function_name(functions, f.function);
    let _ = writeln!(
        s,
        "### {addr} — {kind:?} ({conf:?} confidence)",
        addr = f.address,
        kind = f.kind,
        conf = f.confidence,
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "- **Function**: `{fname}` @ {addr}", addr = f.function);
    let _ = writeln!(s, "- **Mnemonic**: `{mnem}`", mnem = f.mnemonic);
    let _ = writeln!(
        s,
        "- **Condition formula**: `{formula}`",
        formula = f.formula
    );
    if !f.formula_pretty.is_empty() && f.formula_pretty != f.formula {
        let _ = writeln!(s, "- **Expression**: `{pretty}`", pretty = f.formula_pretty);
    }
    // Surface the post-Z3-simplify rendering when the solver produced
    // one *and* it differs from the SSA-level pretty form (otherwise
    // the duplicated line is noise).
    if let Some(z3_pretty) = &f.formula_z3_pretty
        && !z3_pretty.is_empty()
        && z3_pretty.as_str() != f.formula_pretty
        && z3_pretty.as_str() != f.formula
    {
        let _ = writeln!(s, "- **Solver-simplified**: `{z3_pretty}`");
    }
    let _ = writeln!(s, "- **Verdict**: `{:?}`", f.verdict);
    if let Some(taken) = f.taken_target {
        let _ = writeln!(s, "- **Taken target**: {taken}");
    }
    if let Some(ft) = f.fallthrough_target {
        let _ = writeln!(s, "- **Fallthrough target**: {ft}");
    }
    if !f.evidence.inputs.is_empty() {
        let _ = writeln!(
            s,
            "- **Free inputs**: {inputs}",
            inputs = f.evidence.inputs.join(", "),
        );
    }
    let _ = writeln!(
        s,
        "- **IR statements**: {stmt}, **Unknowns**: {unk}",
        stmt = f.evidence.statement_count,
        unk = f.evidence.unknown_count,
    );
    if let Some(patch) = suggest_patch(f) {
        let _ = writeln!(
            s,
            "- **Suggested patch**: `{strategy}` — {rationale}",
            strategy = patch.strategy.as_str(),
            rationale = patch.rationale,
        );
        let _ = writeln!(s, "  - `{cmd}`", cmd = patch.r2_command);
    }
    if let Some(code) = &f.pseudocode {
        let _ = writeln!(s);
        let _ = writeln!(s, "- **Decompiled context**:");
        let _ = writeln!(s);
        let _ = writeln!(s, "```c");
        let _ = writeln!(s, "{code}");
        let _ = writeln!(s, "```");
    }
    let _ = writeln!(s);
}

fn function_name(functions: &[Function], address: Address) -> String {
    functions
        .iter()
        .find(|f| f.address == address)
        .and_then(|f| f.name.clone())
        .unwrap_or_else(|| "<anon>".to_string())
}

// `clippy::cast_precision_loss`: percentages are rendered with `.2` precision
// in the markdown report, and the maximum `usize` values we see here are
// well below 2^53, so the lossy `usize → f64` cast is intentional.
#[allow(clippy::cast_precision_loss)]
fn pct(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64) * 100.0 / (total as f64)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::smt::SmtResult;
    use r2smt_common::{Address, Arch};
    use r2smt_core::{Confidence, Finding, FindingEvidence, FindingKind};
    use r2smt_ir::program::Function;
    use r2smt_slicer::condition::BranchCondition;
    use r2smt_slicer::slice::SliceStatus;

    use super::*;

    fn finding(verdict: SmtResult, kind: FindingKind, mnem: &str, addr: u64) -> Finding {
        Finding {
            address: Address(addr),
            function: Address(0x40_1000),
            mnemonic: mnem.into(),
            condition: BranchCondition::NotEqual,
            formula: "ZF == 0".into(),
            formula_pretty: "(ZF == 0)".into(),
            formula_z3_pretty: None,
            verdict,
            kind,
            confidence: Confidence::High,
            taken_target: Some(Address(0x40_1080)),
            fallthrough_target: Some(Address(addr + 2)),
            operands: Vec::new(),
            is_thumb: false,
            evidence: FindingEvidence {
                slice_status: SliceStatus::Complete,
                statement_count: 6,
                input_count: 0,
                inputs: vec![],
                unknown_count: 0,
                upstream_resolved_to: None,
                oracle_agreement: None,
            },
            pseudocode: None,
        }
    }

    fn sample_functions() -> Vec<Function> {
        vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![],
            is_thumb: false,
        }]
    }

    fn sample_report() -> Report {
        let findings = vec![
            finding(
                SmtResult::AlwaysFalse,
                FindingKind::DeadBranch,
                "jne",
                0x40_1050,
            ),
            finding(
                SmtResult::AlwaysTrue,
                FindingKind::OpaquePredicate,
                "je",
                0x40_1060,
            ),
            finding(
                SmtResult::BothPossible,
                FindingKind::RealBranch,
                "jne",
                0x40_1070,
            ),
        ];
        Report::from_findings("0.1.0", "sample.exe", Arch::X86_64, 64, 1, findings)
    }

    #[test]
    fn kind_counts_actionable_and_total() {
        let r = sample_report();
        assert_eq!(r.summary.dead_branch, 1);
        assert_eq!(r.summary.opaque_predicate, 1);
        assert_eq!(r.summary.real_branch, 1);
        assert_eq!(r.summary.actionable(), 2);
        assert_eq!(r.summary.total(), 3);
    }

    #[test]
    fn json_round_trips() {
        let r = sample_report();
        let json = r.render_json().unwrap();
        let back: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn json_schema_carries_provenance_metrics_and_stable_reason_codes() {
        let mut f = finding(
            SmtResult::Unsound,
            FindingKind::SuspiciousButUnknown,
            "jne",
            0x40_1050,
        );
        f.evidence.slice_status = SliceStatus::truncated("block budget 2 exhausted");
        let report = Report::from_findings("0.2.0", "sample.exe", Arch::X86_64, 64, 1, vec![f])
            .with_metadata(ReportMetadata {
                radare2_version: "6.2.0".into(),
                r2ghidra_version: "6.2.0".into(),
                solver: SolverProvenance {
                    name: "z3".into(),
                    version: "4.15.3".into(),
                    timeout_ms: 500,
                    rlimit: 2_000_000,
                    random_seed: 0,
                },
                ir_policy: IrPolicy::PcodeEsilMnemonic,
                binary_sha256: "00".repeat(32),
                analysis_options: AnalysisOptions {
                    max_slice_instructions: 32,
                    max_basic_blocks: 2,
                    ..AnalysisOptions::default()
                },
            });
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["schema_version"], REPORT_SCHEMA_VERSION);
        assert_eq!(json["solver"]["name"], "z3");
        assert_eq!(json["ir_policy"], "pcode_esil_mnemonic");
        assert_eq!(json["metrics"]["truncated_slices"], 1);
        assert_eq!(json["diagnostics"][0]["code"], "slice_budget");
        assert_eq!(json["diagnostics"][0]["detail"], "block budget 2 exhausted");
    }

    #[test]
    fn truncation_reason_mapping_is_stable() {
        assert_eq!(
            truncation_code("call encountered at 0x401000"),
            AnalysisReasonCode::CallBoundary
        );
        assert_eq!(
            truncation_code("predecessor block 0x10 not present in function"),
            AnalysisReasonCode::MissingPredecessor
        );
        assert_eq!(
            truncation_code("new diagnostic text"),
            AnalysisReasonCode::OtherTruncation
        );
    }

    #[test]
    fn markdown_contains_finding_section_per_actionable() {
        let r = sample_report();
        let md = r.render_markdown(&sample_functions());
        assert!(md.contains("# r2SMT Report"));
        assert!(md.contains("Actionable findings"));
        // The two actionable findings should appear; the RealBranch
        // should not get its own section.
        assert!(md.contains("0x401050"));
        assert!(md.contains("0x401060"));
        assert!(!md.contains("### 0x401070"));
        assert!(md.contains("Suggested patch"));
        assert!(md.contains("sym.main"));
    }

    #[test]
    fn r2_script_emits_one_ccu_per_actionable() {
        let r = sample_report();
        let script = r.render_r2_script(&sample_functions());
        let ccu_lines: Vec<&str> = script
            .lines()
            .filter(|l| l.starts_with("CCu base64:"))
            .collect();
        assert_eq!(ccu_lines.len(), 2);
        // Patch suggestions are commented out (start with `#   patch:`).
        let patch_lines: Vec<&str> = script
            .lines()
            .filter(|l| l.starts_with("#   patch:"))
            .collect();
        assert_eq!(patch_lines.len(), 2);
    }

    #[test]
    fn r2_script_skips_real_branches() {
        let r = sample_report();
        let script = r.render_r2_script(&sample_functions());
        // RealBranch at 0x401070 must not appear in the script.
        assert!(!script.contains("@ 0x401070"));
    }

    #[test]
    fn markdown_surfaces_infix_expression_when_pretty_differs_from_flag_formula() {
        // Regression contract for the human-readable Expression line.
        // When the SSA substitution produces a richer infix form than
        // the bare flag predicate, the Markdown report must surface
        // both. This is the analyst-facing payoff of `pretty_condition`
        // and the `Expr::Display` infix renderer.
        let mut f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            "jne",
            0x40_1060,
        );
        f.formula = "ZF == 0".into();
        f.formula_pretty = "(((ecx * ecx) & 0x1:32) == 0x2:32)".into();
        let r = Report::from_findings("0.1.0", "sample.exe", Arch::X86_64, 64, 1, vec![f]);
        let md = r.render_markdown(&sample_functions());
        assert!(
            md.contains("**Condition formula**: `ZF == 0`"),
            "flag-level formula must remain in the markdown; got:\n{md}",
        );
        assert!(
            md.contains("**Expression**: `(((ecx * ecx) & 0x1:32) == 0x2:32)`"),
            "infix-rendered expression must follow the formula line; got:\n{md}",
        );
    }

    #[test]
    fn markdown_skips_expression_line_when_pretty_equals_formula() {
        // When the two strings coincide (no SSA substitution happened
        // — e.g. a bare `jne` with no upstream cmp), there is no extra
        // information to show and the markdown must skip the
        // duplicated line.
        let mut f = finding(
            SmtResult::BothPossible,
            FindingKind::RealBranch,
            "jne",
            0x40_1070,
        );
        f.formula = "ZF == 0".into();
        f.formula_pretty = "ZF == 0".into();
        // Promote to actionable kind so it appears in the section.
        f.kind = FindingKind::OpaquePredicate;
        let r = Report::from_findings("0.1.0", "sample.exe", Arch::X86_64, 64, 1, vec![f]);
        let md = r.render_markdown(&sample_functions());
        assert!(md.contains("**Condition formula**: `ZF == 0`"));
        assert!(
            !md.contains("**Expression**:"),
            "expression line must be suppressed when redundant; got:\n{md}",
        );
    }

    #[test]
    fn markdown_handles_zero_findings_gracefully() {
        let r = Report::from_findings("0.1.0", "empty.exe", Arch::X86, 32, 0, vec![]);
        let md = r.render_markdown(&[]);
        assert!(md.contains("# r2SMT Report"));
        assert!(md.contains("Branches analyzed | 0"));
        // No "Actionable findings" section.
        assert!(!md.contains("Actionable findings"));
    }

    #[test]
    fn annotations_returns_one_entry_per_actionable_finding() {
        let r = sample_report();
        let ann = r.annotations(&sample_functions());
        assert_eq!(ann.len(), 2);
        let addresses: Vec<Address> = ann.iter().map(|a| a.address).collect();
        assert!(addresses.contains(&Address(0x40_1050)));
        assert!(addresses.contains(&Address(0x40_1060)));
        // RealBranch must not be annotated.
        assert!(!addresses.contains(&Address(0x40_1070)));
    }

    #[test]
    fn annotation_text_contains_kind_confidence_and_function_name() {
        let r = sample_report();
        let ann = r.annotations(&sample_functions());
        let dead = ann
            .iter()
            .find(|a| a.address == Address(0x40_1050))
            .unwrap();
        assert!(dead.text.contains("DeadBranch"));
        assert!(dead.text.contains("High"));
        assert!(dead.text.contains("sym.main"));
        // Must be one line — newlines would break r2's `CCu` command.
        assert!(!dead.text.contains('\n'));
        assert!(!dead.text.contains('\r'));
    }

    #[test]
    fn markdown_emits_decompiled_context_block_when_pseudocode_present() {
        let mut f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            "je",
            0x40_1060,
        );
        f.pseudocode = Some("int main(void) {\n  return 0;\n}".into());
        let r = Report::from_findings("t", "/x", Arch::X86_64, 64, 1, vec![f]);
        let md = r.render_markdown(&sample_functions());
        assert!(md.contains("**Decompiled context**:"));
        assert!(md.contains("```c"));
        assert!(md.contains("int main(void) {"));
    }

    #[test]
    fn annotation_text_appends_bounded_decompiled_preview() {
        let mut f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            "je",
            0x40_1060,
        );
        let mut body = String::new();
        for i in 0..20 {
            body.push_str("line");
            body.push_str(&i.to_string());
            body.push('\n');
        }
        f.pseudocode = Some(body);
        let r = Report::from_findings("t", "/x", Arch::X86_64, 64, 1, vec![f]);
        let ann = r.annotations(&sample_functions());
        let entry = ann
            .iter()
            .find(|a| a.address == Address(0x40_1060))
            .unwrap();
        assert!(entry.text.contains("-- decompiled --"));
        assert!(entry.text.contains("line0"));
        assert!(entry.text.contains("line5"));
        // Bounded at PSEUDO_ANNOTATION_PREVIEW_LINES (6): line6+ excluded.
        assert!(!entry.text.contains("line6"));
        assert!(entry.text.contains('…'));
    }

    #[test]
    fn r2_script_lines_match_annotation_entries() {
        let r = sample_report();
        let ann = r.annotations(&sample_functions());
        let script = r.render_r2_script(&sample_functions());
        for a in &ann {
            let encoded = r2smt_common::b64::encode(a.text.as_bytes());
            let expected = format!("CCu base64:{encoded} @ {}", a.address);
            assert!(
                script.contains(&expected),
                "missing line in script: {expected}\nscript:\n{script}"
            );
        }
    }

    #[test]
    fn r2_script_keeps_multiline_pseudocode_on_one_base64_line() {
        // A decompiled preview embeds newlines and C metacharacters
        // (`;`, `->`) that are r2 command separators. The CCu payload must
        // be a single base64 line so r2 never runs the pseudocode as
        // commands nor applies the comment at the wrong seek.
        let mut f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            "je",
            0x40_1060,
        );
        f.pseudocode = Some("int main(void) {\n  a->b = c; return 0 > x;\n}".into());
        let r = Report::from_findings("t", "/x", Arch::X86_64, 64, 1, vec![f]);
        let script = r.render_r2_script(&sample_functions());
        let ccu_lines: Vec<&str> = script.lines().filter(|l| l.starts_with("CCu ")).collect();
        assert_eq!(ccu_lines.len(), 1, "exactly one CCu line: {script}");
        let line = ccu_lines[0];
        assert!(
            line.starts_with("CCu base64:"),
            "not base64-encoded: {line}"
        );
        assert!(line.ends_with("@ 0x401060"), "address at line end: {line}");
        // No raw pseudocode leaked as its own command line.
        assert!(
            !script.contains("int main(void)"),
            "raw code in script: {script}"
        );
        assert!(
            !script.contains("return 0 > x"),
            "raw code in script: {script}"
        );
    }
}
