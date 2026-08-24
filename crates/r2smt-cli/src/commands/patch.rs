//! `patch` subcommand group: conservative byte-level patching
//! (`patch`, dry-run plan, rollback) plus its CLI knob struct and
//! default backup / manifest path helpers.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use r2smt_common::Arch;
use r2smt_core::{Confidence, Finding};
use r2smt_ir::BinaryProvider;
use r2smt_patch::{
    ApplyConfig, PatchManifest, PatchStatus, apply_plan, build_plan, rollback_from_manifest,
    sha256_hex, verify_manifest,
};
use r2smt_report::PatchStrategy;
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;

use crate::args::SolverArg;
use crate::render::hex_preview;
use crate::support::{compute_findings, open_provider, open_provider_writable};

// These booleans mirror mutually constrained CLI switches; replacing
// them with another enum would duplicate clap's validated state.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct PatchCli<'a> {
    pub(crate) min_confidence: Confidence,
    pub(crate) apply: bool,
    pub(crate) output: Option<&'a Path>,
    pub(crate) in_place: bool,
    pub(crate) expected_sha256: Option<&'a str>,
    pub(crate) verify_only: bool,
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

fn default_output_path(file: &Path) -> PathBuf {
    let mut s = file.as_os_str().to_owned();
    s.push(".r2smt.patched");
    PathBuf::from(s)
}

// `clippy::too_many_arguments`: same rationale as `solve` / `batch` —
// a CLI driver threading independent, read-at-distinct-stages knobs
// (`ir_pcode` is the 8th). A params struct would only relocate noise.
// The apply branch is deliberately linear so the transaction order is
// visible: copy, preflight, patch, sync, reanalyse, then atomic commit.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    if cfg.verify_only {
        return patch_verify_only(file, deep, limits, options, cfg, ir_pcode);
    }

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
        println!("input SHA-256: {}", sha256_hex(file)?);
        println!("apply requires --expect-sha256 with this value");
        return Ok(());
    }

    let expected_sha256 = cfg
        .expected_sha256
        .ok_or_else(|| anyhow::anyhow!("--apply requires --expect-sha256 from the dry-run"))?
        .to_ascii_lowercase();
    if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        anyhow::bail!("--expect-sha256 must be exactly 64 hexadecimal characters");
    }
    let target = if cfg.in_place {
        file.to_path_buf()
    } else {
        cfg.output
            .map_or_else(|| default_output_path(file), Path::to_path_buf)
    };
    if !cfg.in_place && target == file {
        anyhow::bail!("--output must differ from the input; use --in-place explicitly");
    }
    if !cfg.in_place && target.exists() {
        anyhow::bail!(
            "refusing to overwrite existing output at {}",
            target.display()
        );
    }
    if cfg.backup.is_some() && !cfg.in_place {
        anyhow::bail!("--backup is only meaningful with --in-place");
    }
    let manifest_path = cfg
        .manifest
        .map_or_else(|| default_manifest_path(&target), Path::to_path_buf);
    if manifest_path.exists() {
        anyhow::bail!(
            "refusing to overwrite existing manifest at {}",
            manifest_path.display()
        );
    }
    let manifest_parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(manifest_parent)?;

    let target_parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(target_parent)?;
    let transaction = tempfile::NamedTempFile::new_in(target_parent)
        .with_context(|| format!("creating transaction file in {}", target_parent.display()))?;
    fs::copy(file, transaction.path())?;

    let mut provider = open_provider_writable(transaction.path(), deep)?;
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

    let backup = if cfg.in_place {
        cfg.backup
            .map_or_else(|| default_backup_path(file), Path::to_path_buf)
    } else {
        file.to_path_buf()
    };
    let apply_cfg = ApplyConfig {
        binary_path: transaction.path().to_path_buf(),
        target_path: target.clone(),
        backup_path: backup.clone(),
        r2smt_version: env!("CARGO_PKG_VERSION").to_string(),
        expected_sha256,
        failure_manifest_path: manifest_path.clone(),
    };
    let mut manifest =
        apply_plan(&mut provider, &plan, &apply_cfg).with_context(|| "applying patch plan")?;
    drop(provider);

    transaction.as_file().sync_all()?;
    if let Err(error) = post_patch_verify(
        transaction.path(),
        deep,
        limits,
        options,
        cfg,
        ir_pcode,
        &manifest,
    ) {
        manifest.status = PatchStatus::Recovered;
        manifest.failure = Some(format!("post-patch verification failed: {error:#}"));
        manifest.write_to(&manifest_path)?;
        return Err(error.context("post-patch verification failed; original left untouched"));
    }

    if cfg.in_place {
        if backup.exists() {
            anyhow::bail!(
                "refusing to overwrite existing backup at {} — move or delete it first",
                backup.display()
            );
        }
        fs::copy(file, &backup)?;
        fs::File::open(&backup)?.sync_all()?;
        println!("backup: {}", backup.display());
    }

    let manifest_tmp = tempfile::NamedTempFile::new_in(manifest_parent)?;
    manifest.write_to(manifest_tmp.path())?;
    manifest_tmp.as_file().sync_all()?;

    transaction
        .persist(&target)
        .map_err(|error| error.error)
        .with_context(|| format!("atomically committing {}", target.display()))?;
    if let Err(error) = manifest_tmp
        .persist(&manifest_path)
        .map_err(|error| error.error)
    {
        if cfg.in_place {
            fs::copy(&backup, &target)?;
        } else {
            fs::remove_file(&target)?;
        }
        manifest.status = PatchStatus::Recovered;
        manifest.failure = Some(format!("manifest commit failed: {error}"));
        manifest.write_to(&manifest_path)?;
        anyhow::bail!("manifest commit failed; binary recovery completed: {error}");
    }

    println!();
    println!("applied:  {} operations", manifest.operations.len());
    println!("output:   {}", target.display());
    println!("manifest: {}", manifest_path.display());
    println!("before SHA-256: {}", manifest.binary_sha256_before);
    println!("after  SHA-256: {}", manifest.binary_sha256_after);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn post_patch_verify(
    file: &Path,
    deep: bool,
    limits: &SliceLimits,
    options: SolveOptions,
    cfg: &PatchCli<'_>,
    ir_pcode: bool,
    manifest: &PatchManifest,
) -> Result<()> {
    let mut provider = open_provider(file, deep)?;
    verify_manifest(&mut provider, manifest, file)?;
    for record in &manifest.operations {
        let expected_bytes = record.patched_bytes()?;
        let original_size = record.original_bytes()?.len();
        let function = provider.load_block_at(record.address)?;
        let (block, instruction) = function
            .blocks
            .iter()
            .find_map(|block| {
                block
                    .instructions
                    .iter()
                    .find(|instruction| instruction.address == record.address)
                    .map(|instruction| (block, instruction))
            })
            .ok_or_else(|| {
                anyhow::anyhow!("patched instruction {} did not redecode", record.address)
            })?;
        if instruction.bytes != expected_bytes
            || usize::from(instruction.size) != expected_bytes.len()
        {
            anyhow::bail!(
                "patched instruction {} decoded with unexpected bytes or size",
                record.address
            );
        }
        if let Some(expected) = &record.expected_successors {
            let flow_matches = if record.strategy == PatchStrategy::NopJcc.as_str() {
                let next = record.address.get()
                    + u64::try_from(original_size)
                        .context("original patch size does not fit address")?;
                expected.as_slice() == [r2smt_common::Address(next)]
                    && function.blocks.iter().any(|candidate| {
                        candidate
                            .instructions
                            .first()
                            .is_some_and(|instruction| instruction.address.get() == next)
                            || candidate.instructions.windows(2).any(|pair| {
                                pair[0].address == record.address && pair[1].address.get() == next
                            })
                    })
            } else {
                let mut actual = block.successors.clone();
                let mut expected = expected.clone();
                actual.sort_unstable();
                expected.sort_unstable();
                actual == expected
            };
            if !flow_matches {
                anyhow::bail!("CFG postcondition failed at {}", record.address);
            }
        }
    }
    drop(provider);

    let (_, findings) = compute_findings(
        file, deep, None, None, limits, options, cfg.solver, false, ir_pcode,
    )?;
    let patched: BTreeSet<_> = manifest
        .operations
        .iter()
        .map(|record| record.address)
        .collect();
    if let Some(finding) = findings
        .iter()
        .find(|finding| finding.is_actionable() && patched.contains(&finding.address))
    {
        anyhow::bail!(
            "actionable finding remains at {} after patch",
            finding.address
        );
    }
    Ok(())
}

fn patch_verify_only(
    file: &Path,
    deep: bool,
    limits: &SliceLimits,
    options: SolveOptions,
    cfg: &PatchCli<'_>,
    ir_pcode: bool,
) -> Result<()> {
    let manifest_path = cfg
        .manifest
        .map_or_else(|| default_manifest_path(file), Path::to_path_buf);
    let manifest = PatchManifest::read_from(&manifest_path)?;
    post_patch_verify(file, deep, limits, options, cfg, ir_pcode, &manifest)?;
    println!("verification passed: {}", manifest_path.display());
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
    let restored = sha256_hex(file)?;
    if restored != manifest.binary_sha256_before {
        anyhow::bail!(
            "rollback hash mismatch: restored {restored}, expected {}",
            manifest.binary_sha256_before
        );
    }
    println!("rollback completed");
    Ok(())
}
