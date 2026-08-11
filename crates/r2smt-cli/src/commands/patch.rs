//! `patch` subcommand group: conservative byte-level patching
//! (`patch`, dry-run plan, rollback) plus its CLI knob struct and
//! default backup / manifest path helpers.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use r2smt_common::Arch;
use r2smt_core::{Confidence, Finding};
use r2smt_patch::{ApplyConfig, PatchManifest, apply_plan, build_plan, rollback_from_manifest};
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;

use crate::args::SolverArg;
use crate::render::hex_preview;
use crate::support::{compute_findings, open_provider, open_provider_writable};

pub(crate) struct PatchCli<'a> {
    pub(crate) min_confidence: Confidence,
    pub(crate) apply: bool,
    pub(crate) backup: Option<&'a Path>,
    pub(crate) manifest: Option<&'a Path>,
    pub(crate) rollback: bool,
    pub(crate) solver: SolverArg,
}

const DEFAULT_BACKUP_SUFFIX: &str = ".r2smt.bak";
const DEFAULT_MANIFEST_SUFFIX: &str = ".r2smt.manifest.json";

fn default_backup_path(file: &Path) -> PathBuf {
    let mut s = file.as_os_str().to_owned();
    s.push(DEFAULT_BACKUP_SUFFIX);
    PathBuf::from(s)
}

fn default_manifest_path(file: &Path) -> PathBuf {
    let mut s = file.as_os_str().to_owned();
    s.push(DEFAULT_MANIFEST_SUFFIX);
    PathBuf::from(s)
}

// `clippy::too_many_arguments`: same rationale as `solve` / `batch` —
// a CLI driver threading independent, read-at-distinct-stages knobs
// (`ir_pcode` is the 8th). A params struct would only relocate noise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn patch(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    options: SolveOptions,
    cfg: &PatchCli<'_>,
    ir_pcode: bool,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }
    if cfg.rollback {
        return patch_rollback(file, deep, cfg);
    }

    // Read-only pipeline first: open without write access so we can
    // plan and (optionally) bail out before any disk mutation.
    let (arch, findings) = compute_findings(
        file,
        deep,
        at,
        function_filter,
        limits,
        options,
        cfg.solver,
        false,
        ir_pcode,
    )?;
    let actionable: Vec<Finding> = findings
        .into_iter()
        .filter(Finding::is_actionable)
        .collect();
    println!(
        "candidate findings: {n} (min_confidence={mc:?})",
        n = actionable.len(),
        mc = cfg.min_confidence,
    );

    if !cfg.apply {
        println!();
        patch_dry_run_plan(file, deep, arch, &actionable, cfg)?;
        return Ok(());
    }

    // Apply path: backup → open writable → plan → apply → write manifest.
    let backup = cfg
        .backup
        .map_or_else(|| default_backup_path(file), Path::to_path_buf);
    let manifest_path = cfg
        .manifest
        .map_or_else(|| default_manifest_path(file), Path::to_path_buf);

    if backup.exists() {
        anyhow::bail!(
            "refusing to overwrite existing backup at {} — move or delete it first",
            backup.display()
        );
    }
    fs::copy(file, &backup)
        .with_context(|| format!("backing up {} to {}", file.display(), backup.display()))?;
    println!("backup: {}", backup.display());

    let mut provider = open_provider_writable(file, deep)?;
    let plan = build_plan(&actionable, cfg.min_confidence, arch, &mut provider)
        .with_context(|| "building patch plan")?;
    println!("plan: {} operations", plan.operations.len());
    for skip in &plan.skipped {
        println!("  skipped {addr}: {reason}", addr = skip.0, reason = skip.1);
    }
    if plan.operations.is_empty() {
        println!("nothing to apply");
        return Ok(());
    }

    let apply_cfg = ApplyConfig {
        binary_path: file.to_path_buf(),
        backup_path: backup.clone(),
        r2smt_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let manifest =
        apply_plan(&mut provider, &plan, &apply_cfg).with_context(|| "applying patch plan")?;
    drop(provider);

    manifest
        .write_to(&manifest_path)
        .with_context(|| format!("writing manifest to {}", manifest_path.display()))?;
    println!();
    println!("applied:  {} operations", manifest.operations.len());
    println!("manifest: {}", manifest_path.display());
    println!("before SHA-256: {}", manifest.binary_sha256_before);
    println!("after  SHA-256: {}", manifest.binary_sha256_after);
    Ok(())
}

fn patch_dry_run_plan(
    file: &Path,
    deep: bool,
    arch: Arch,
    actionable: &[Finding],
    cfg: &PatchCli<'_>,
) -> Result<()> {
    let mut provider = open_provider(file, deep)?;
    let plan = build_plan(actionable, cfg.min_confidence, arch, &mut provider)
        .with_context(|| "building patch plan")?;
    println!("planned operations: {}", plan.operations.len());
    for op in &plan.operations {
        println!(
            "  {addr}  {strategy:<22}  size={size}  → {bytes}",
            addr = op.address,
            strategy = op.strategy.as_str(),
            size = op.size,
            bytes = hex_preview(&op.new_bytes),
        );
    }
    if !plan.skipped.is_empty() {
        println!();
        println!("skipped: {}", plan.skipped.len());
        for (addr, reason) in &plan.skipped {
            println!("  {addr}  {reason}");
        }
    }
    println!();
    println!("dry-run: re-run with --apply to write the changes");
    Ok(())
}

fn patch_rollback(file: &Path, deep: bool, cfg: &PatchCli<'_>) -> Result<()> {
    let manifest_path = cfg
        .manifest
        .map_or_else(|| default_manifest_path(file), Path::to_path_buf);
    let manifest = PatchManifest::read_from(&manifest_path)
        .with_context(|| format!("reading manifest at {}", manifest_path.display()))?;
    println!(
        "rolling back {n} operation(s) from {path}",
        n = manifest.operations.len(),
        path = manifest_path.display(),
    );
    let mut provider = open_provider_writable(file, deep)?;
    rollback_from_manifest(&mut provider, &manifest).with_context(|| "rolling back manifest")?;
    drop(provider);
    println!("rollback completed");
    Ok(())
}
