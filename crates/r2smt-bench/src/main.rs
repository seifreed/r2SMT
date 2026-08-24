#![allow(clippy::print_stdout)]
//! Validate and score the public r2SMT benchmark corpus.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use r2smt_common::smt::SmtResult;
use r2smt_core::{FindingKind, OracleAgreement};
use r2smt_report::Report;
use r2smt_slicer::slice::SliceStatus;
use serde::{Deserialize, Serialize};

const REQUIRED_ARCHITECTURES: &[&str] = &["x86", "x86_64", "aarch32", "thumb", "aarch64"];
const REQUIRED_COMPILERS: &[&str] = &["gcc", "clang", "msvc"];
const REQUIRED_OPTIMIZATIONS: &[&str] = &["O0", "O2", "O3", "Os"];
const REQUIRED_FEATURES: &[&str] = &[
    "real_branches",
    "opaque_predicates",
    "mba",
    "signed_unsigned",
    "partial_flags",
    "subregisters",
    "diamonds_joins",
    "loops",
    "memory",
    "calls",
    "x87_sse_avx",
    "neon_vfp",
    "unsupported",
    "pie",
    "stripped",
    "lto",
];

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    schema_version: u32,
    architectures: Vec<String>,
    compilers: Vec<String>,
    optimizations: Vec<String>,
    features: Vec<String>,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    source: PathBuf,
    assembly: PathBuf,
    expected_branches: PathBuf,
    expected_findings: PathBuf,
    expected_patches: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ExpectedBranches {
    schema_version: u32,
    total: usize,
}

#[derive(Debug, Deserialize)]
struct ExpectedFindings {
    schema_version: u32,
    actionable: Vec<ExpectedFinding>,
}

#[derive(Debug, Deserialize)]
struct ExpectedFinding {
    kind: FindingKind,
    mnemonic: String,
}

#[derive(Debug, Deserialize)]
struct ExpectedPatches {
    schema_version: u32,
    patches: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct Metrics {
    schema_version: u32,
    expected_branches: usize,
    discovered_branches: usize,
    branch_discovery_recall: Option<f64>,
    complete_slice_percent: Option<f64>,
    definitive_percent: Option<f64>,
    actionable_precision: Option<f64>,
    false_actionable_findings: usize,
    false_negative_actionable_findings: usize,
    unknown_by_reason: BTreeMap<String, usize>,
    frontend_coverage: BTreeMap<String, usize>,
    lifter_disagreements: usize,
    solver_disagreements: usize,
    elapsed_ms_p50: Option<u64>,
    elapsed_ms_p95: Option<u64>,
    verified_patch_rate: Option<f64>,
    rollback_success_rate: Option<f64>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn missing(required: &[&str], actual: &[String]) -> Vec<String> {
    let actual: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    required
        .iter()
        .filter(|item| !actual.contains(**item))
        .map(|item| (*item).to_string())
        .collect()
}

fn validate(root: &Path) -> Result<()> {
    let manifest: CorpusManifest = read_json(&root.join("manifest.json"))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported corpus schema_version {}",
            manifest.schema_version
        );
    }
    for (label, required, actual) in [
        (
            "architectures",
            REQUIRED_ARCHITECTURES,
            &manifest.architectures,
        ),
        ("compilers", REQUIRED_COMPILERS, &manifest.compilers),
        (
            "optimizations",
            REQUIRED_OPTIMIZATIONS,
            &manifest.optimizations,
        ),
        ("features", REQUIRED_FEATURES, &manifest.features),
    ] {
        let missing = missing(required, actual);
        if !missing.is_empty() {
            bail!("corpus is missing {label}: {}", missing.join(", "));
        }
    }
    if manifest.fixtures.is_empty() {
        bail!("corpus has no fixtures");
    }
    let mut ids = BTreeSet::new();
    for fixture in manifest.fixtures {
        if !ids.insert(fixture.id.clone()) {
            bail!("duplicate fixture id: {}", fixture.id);
        }
        let source = root.join(fixture.source);
        let assembly = root.join(fixture.assembly);
        let branches_path = root.join(fixture.expected_branches);
        let findings_path = root.join(fixture.expected_findings);
        let patches_path = root.join(fixture.expected_patches);
        for path in [
            &source,
            &assembly,
            &branches_path,
            &findings_path,
            &patches_path,
        ] {
            if !path.exists() {
                bail!("fixture {} is missing {}", fixture.id, path.display());
            }
        }
        for directory in [&source, &assembly] {
            if fs::read_dir(directory)?.next().is_none() {
                bail!("fixture {} has empty {}", fixture.id, directory.display());
            }
        }
        let branches: ExpectedBranches = read_json(&branches_path)?;
        let findings: ExpectedFindings = read_json(&findings_path)?;
        let patches: ExpectedPatches = read_json(&patches_path)?;
        if branches.schema_version != 1
            || findings.schema_version != 1
            || patches.schema_version != 1
        {
            bail!(
                "fixture {} has an unsupported expectation schema",
                fixture.id
            );
        }
        let _expected_patch_count = patches.patches.len();
    }
    println!("corpus validation ok");
    Ok(())
}

fn percent(part: usize, total: usize) -> Option<f64> {
    let part = f64::from(u32::try_from(part).ok()?);
    let total = f64::from(u32::try_from(total).ok()?);
    (total != 0.0).then(|| part * 100.0 / total)
}

fn score(report: &Report, branches: &ExpectedBranches, expected: &ExpectedFindings) -> Metrics {
    let mut remaining: Vec<(FindingKind, &str)> = expected
        .actionable
        .iter()
        .map(|f| (f.kind, f.mnemonic.as_str()))
        .collect();
    let mut true_positive = 0usize;
    let mut false_positive = 0usize;
    let mut complete = 0usize;
    let mut definitive = 0usize;
    let mut unknown_by_reason = BTreeMap::new();
    let mut lifter_disagreements = 0usize;
    let mut solver_disagreements = 0usize;

    for finding in &report.findings {
        if matches!(finding.evidence.slice_status, SliceStatus::Complete) {
            complete += 1;
        } else if let SliceStatus::Truncated { reason } = &finding.evidence.slice_status {
            *unknown_by_reason
                .entry(format!("truncated:{reason}"))
                .or_insert(0) += 1;
        }
        match finding.verdict {
            SmtResult::AlwaysTrue | SmtResult::AlwaysFalse | SmtResult::BothPossible => {
                definitive += 1;
            }
            SmtResult::Timeout => *unknown_by_reason.entry("timeout".into()).or_insert(0) += 1,
            SmtResult::Unknown => {
                *unknown_by_reason
                    .entry("solver_unknown".into())
                    .or_insert(0) += 1;
            }
            SmtResult::Unsound => *unknown_by_reason.entry("unsound".into()).or_insert(0) += 1,
            _ => {
                *unknown_by_reason
                    .entry("future_verdict".into())
                    .or_insert(0) += 1;
            }
        }
        if finding.kind == FindingKind::LifterDisagreement {
            lifter_disagreements += 1;
        }
        if finding.evidence.oracle_agreement == Some(OracleAgreement::Disagreed) {
            solver_disagreements += 1;
        }
        if finding.is_actionable() {
            if let Some(index) = remaining
                .iter()
                .position(|(kind, mnemonic)| *kind == finding.kind && *mnemonic == finding.mnemonic)
            {
                remaining.swap_remove(index);
                true_positive += 1;
            } else {
                false_positive += 1;
                eprintln!(
                    "unexpected actionable finding: {:?}:{}",
                    finding.kind, finding.mnemonic
                );
            }
        }
    }

    Metrics {
        schema_version: 1,
        expected_branches: branches.total,
        discovered_branches: report.branches_analyzed,
        branch_discovery_recall: percent(
            report.branches_analyzed.min(branches.total),
            branches.total,
        ),
        complete_slice_percent: percent(complete, report.findings.len()),
        definitive_percent: percent(definitive, report.findings.len()),
        actionable_precision: percent(true_positive, true_positive + false_positive),
        false_actionable_findings: false_positive,
        false_negative_actionable_findings: remaining.len(),
        unknown_by_reason,
        frontend_coverage: BTreeMap::new(),
        lifter_disagreements,
        solver_disagreements,
        elapsed_ms_p50: None,
        elapsed_ms_p95: None,
        verified_patch_rate: None,
        rollback_success_rate: None,
    }
}

fn score_files(report_path: &Path, fixture_root: &Path) -> Result<()> {
    let report: Report = read_json(report_path)?;
    let branches: ExpectedBranches = read_json(&fixture_root.join("expected-branches.json"))?;
    let expected: ExpectedFindings = read_json(&fixture_root.join("expected-findings.json"))?;
    if branches.schema_version != 1 || expected.schema_version != 1 {
        bail!("unsupported expectation schema version");
    }
    let metrics = score(&report, &branches, &expected);
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    if metrics.false_actionable_findings != 0 {
        bail!(
            "{} false actionable finding(s)",
            metrics.false_actionable_findings
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [command, root] if command == "validate" => validate(Path::new(root)),
        [command, report, fixture] if command == "score" => {
            score_files(Path::new(report), Path::new(fixture))
        }
        _ => bail!("usage: r2smt-bench validate <corpus> | score <report.json> <fixture-dir>"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::Arch;

    use super::*;

    #[test]
    fn empty_actionable_set_has_zero_false_positives() {
        let report = Report::from_findings("test", "fixture", Arch::X86_64, 64, 1, Vec::new());
        let metrics = score(
            &report,
            &ExpectedBranches {
                schema_version: 1,
                total: 0,
            },
            &ExpectedFindings {
                schema_version: 1,
                actionable: Vec::new(),
            },
        );
        assert_eq!(metrics.false_actionable_findings, 0);
    }
}
