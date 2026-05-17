//! Apply a [`PatchPlan`] through a [`BytePatcher`] and produce the
//! durable [`PatchManifest`].

use std::path::{Path, PathBuf};

use r2smt_common::Result;
use r2smt_ir::byte_patcher::BytePatcher;
use tracing::{info, warn};

use crate::digest::sha256_hex;
use crate::manifest::{MANIFEST_VERSION, PatchManifest, PatchRecord};
use crate::plan::PatchPlan;

/// Inputs to [`apply_plan`] that are not part of the plan itself.
///
/// The caller is responsible for creating `backup_path` *before*
/// invoking the patcher — backups taken after writes have started
/// would already be corrupted.
#[derive(Debug, Clone)]
pub struct ApplyConfig {
    /// Path of the binary being patched (used to compute integrity
    /// hashes for the manifest).
    pub binary_path: PathBuf,
    /// Path of the full-file backup created before patching.
    pub backup_path: PathBuf,
    /// r2SMT version string recorded in the manifest.
    pub r2smt_version: String,
}

/// Apply every operation in `plan` through `patcher`, returning the
/// durable manifest that records what changed.
///
/// On a partial failure (any `read` or `write` returning `Err`) the
/// function aborts immediately and propagates the error; the manifest
/// for the *partial* run is *not* returned, so the caller must use
/// the backup at `config.backup_path` to recover.
///
/// # Errors
///
/// Propagates I/O failures from hashing the binary, plus any error
/// produced by the underlying [`BytePatcher`].
pub fn apply_plan(
    patcher: &mut dyn BytePatcher,
    plan: &PatchPlan,
    config: &ApplyConfig,
) -> Result<PatchManifest> {
    let binary_sha256_before = sha256_hex(&config.binary_path)?;
    info!(
        target: "r2smt::patch",
        binary = %config.binary_path.display(),
        ops = plan.operations.len(),
        skipped = plan.skipped.len(),
        sha256_before = %binary_sha256_before,
        "starting patch run"
    );

    let mut records: Vec<PatchRecord> = Vec::with_capacity(plan.operations.len());
    for op in &plan.operations {
        let original = patcher.read_bytes(op.address, op.size)?;
        if original.len() != op.new_bytes.len() {
            warn!(
                target: "r2smt::patch",
                addr = %op.address,
                original = original.len(),
                new = op.new_bytes.len(),
                "plan size disagreed with read; aborting"
            );
            return Err(r2smt_common::Error::parse(
                "patch_apply",
                format!(
                    "size mismatch at {addr}: original {orig}, new {new}",
                    addr = op.address,
                    orig = original.len(),
                    new = op.new_bytes.len(),
                ),
            ));
        }
        patcher.write_bytes(op.address, &op.new_bytes)?;
        records.push(PatchRecord {
            address: op.address,
            strategy: op.strategy.as_str().to_string(),
            kind: op.kind,
            confidence: op.confidence,
            original_bytes_hex: hex::encode(&original),
            patched_bytes_hex: hex::encode(&op.new_bytes),
            rationale: op.rationale.clone(),
        });
    }

    let binary_sha256_after = sha256_hex(&config.binary_path)?;
    info!(
        target: "r2smt::patch",
        applied = records.len(),
        sha256_after = %binary_sha256_after,
        "patch run completed"
    );

    Ok(PatchManifest {
        manifest_version: MANIFEST_VERSION,
        r2smt_version: config.r2smt_version.clone(),
        binary: config.binary_path.display().to_string(),
        binary_sha256_before,
        binary_sha256_after,
        backup_path: absolute_or_display(&config.backup_path),
        operations: records,
    })
}

fn absolute_or_display(path: &Path) -> String {
    path.canonicalize()
        .map_or_else(|_| path.display().to_string(), |p| p.display().to_string())
}

/// Restore the original bytes recorded in `manifest`, walking the
/// operations in reverse order so any chained patches are unwound
/// last-applied-first.
///
/// # Errors
///
/// Propagates [`r2smt_common::Error::Parse`] if any record has
/// malformed hex, plus any error from the underlying [`BytePatcher`].
pub fn rollback_from_manifest(
    patcher: &mut dyn BytePatcher,
    manifest: &PatchManifest,
) -> Result<()> {
    info!(
        target: "r2smt::patch",
        ops = manifest.operations.len(),
        binary = %manifest.binary,
        "starting rollback"
    );
    for record in manifest.operations.iter().rev() {
        let original = record.original_bytes()?;
        patcher.write_bytes(record.address, &original)?;
    }
    info!(target: "r2smt::patch", "rollback completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::fs;
    use std::io::Write;

    use r2smt_common::smt::SmtResult;
    use r2smt_common::{Address, Arch};
    use r2smt_core::{Confidence, Finding, FindingEvidence, FindingKind};
    use r2smt_ir::testing::InMemoryBytePatcher;
    use r2smt_report::PatchStrategy;
    use r2smt_slicer::condition::BranchCondition;
    use r2smt_slicer::slice::SliceStatus;
    use tempfile::NamedTempFile;

    use super::*;
    use crate::plan::{PlanOperation, build_plan};

    fn dead_branch_finding(address: u64, size: u64) -> Finding {
        Finding {
            address: Address(address),
            function: Address(0x40_1000),
            mnemonic: "jne".into(),
            condition: BranchCondition::NotEqual,
            formula: "ZF == 0".into(),
            formula_pretty: "(ZF == 0)".into(),
            formula_z3_pretty: None,
            verdict: SmtResult::AlwaysFalse,
            kind: FindingKind::DeadBranch,
            confidence: Confidence::High,
            taken_target: Some(Address(0x40_1080)),
            fallthrough_target: Some(Address(address + size)),
            operands: Vec::new(),
            is_thumb: false,
            evidence: FindingEvidence {
                slice_status: SliceStatus::Complete,
                statement_count: 0,
                input_count: 0,
                inputs: vec![],
                unknown_count: 0,
                upstream_resolved_to: None,
            },
            pseudocode: None,
        }
    }

    fn writable_temp_file_with_bytes(bytes: &[u8]) -> NamedTempFile {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(bytes).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn apply_records_original_and_new_bytes() {
        let bytes = vec![0x75, 0x05, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90];
        let tmp = writable_temp_file_with_bytes(&bytes);
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let finding = dead_branch_finding(0x40_1050, 2);
        let plan = build_plan(&[finding], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);

        let config = ApplyConfig {
            binary_path: tmp.path().to_path_buf(),
            backup_path: tmp.path().with_extension("bak"),
            r2smt_version: "test".into(),
        };
        let manifest = apply_plan(&mut patcher, &plan, &config).unwrap();

        assert_eq!(manifest.operations.len(), 1);
        let record = &manifest.operations[0];
        assert_eq!(record.address, Address(0x40_1050));
        assert_eq!(record.original_bytes_hex, "7505");
        assert_eq!(record.patched_bytes_hex, "9090");
        assert_eq!(record.strategy, PatchStrategy::NopJcc.as_str());
        // In-memory patcher mutates the buffer; verify the write
        // actually replaced the original bytes.
        assert_eq!(&patcher.bytes[0..2], &[0x90, 0x90]);
    }

    #[test]
    fn rollback_restores_original_bytes() {
        let bytes = vec![0x75, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let tmp = writable_temp_file_with_bytes(&bytes);
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes.clone());
        let finding = dead_branch_finding(0x40_1050, 2);
        let plan = build_plan(&[finding], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let config = ApplyConfig {
            binary_path: tmp.path().to_path_buf(),
            backup_path: tmp.path().with_extension("bak"),
            r2smt_version: "test".into(),
        };
        let manifest = apply_plan(&mut patcher, &plan, &config).unwrap();

        // The plan wrote NOPs; ensure the buffer now diverges from
        // the original.
        assert_ne!(&patcher.bytes[0..2], &bytes[0..2]);

        // Roll back and confirm the original bytes are restored.
        rollback_from_manifest(&mut patcher, &manifest).unwrap();
        assert_eq!(&patcher.bytes[0..2], &bytes[0..2]);
    }

    #[test]
    fn apply_aborts_when_patcher_write_fails() {
        // Use a tiny buffer so the second write goes past the end.
        let bytes = vec![0x75, 0x05];
        let tmp = writable_temp_file_with_bytes(&bytes);
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let mut plan = PatchPlan::default();
        plan.operations.push(PlanOperation {
            address: Address(0x40_1050),
            strategy: PatchStrategy::NopJcc,
            kind: FindingKind::DeadBranch,
            confidence: Confidence::High,
            size: 2,
            new_bytes: vec![0x90, 0x90],
            rationale: "test".into(),
        });
        // Second operation writes past the end of the in-memory
        // buffer and must trigger an Err from the patcher.
        plan.operations.push(PlanOperation {
            address: Address(0x40_1060),
            strategy: PatchStrategy::NopJcc,
            kind: FindingKind::DeadBranch,
            confidence: Confidence::High,
            size: 2,
            new_bytes: vec![0x90, 0x90],
            rationale: "test".into(),
        });
        let config = ApplyConfig {
            binary_path: tmp.path().to_path_buf(),
            backup_path: tmp.path().with_extension("bak"),
            r2smt_version: "test".into(),
        };
        let err = apply_plan(&mut patcher, &plan, &config).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("past end") || msg.contains("address"));
    }

    #[test]
    fn apply_captures_sha256_from_disk_into_manifest() {
        let bytes = vec![0x75, 0x05];
        let tmp = writable_temp_file_with_bytes(&bytes);
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let finding = dead_branch_finding(0x40_1050, 2);
        let plan = build_plan(&[finding], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let config = ApplyConfig {
            binary_path: tmp.path().to_path_buf(),
            backup_path: tmp.path().with_extension("bak"),
            r2smt_version: "test".into(),
        };

        // Capture the file's SHA-256 before apply. The in-memory
        // patcher does not write to the file, so the post hash also
        // matches `pre` — the assertion below pins that the manifest
        // truly reads from disk both times rather than just echoing
        // an in-memory value.
        let pre = sha256_hex(tmp.path()).unwrap();
        let manifest = apply_plan(&mut patcher, &plan, &config).unwrap();
        assert_eq!(manifest.binary_sha256_before, pre);
        assert_eq!(manifest.binary_sha256_after, pre);

        // Now rewrite the underlying file to simulate the effect of a
        // real disk-backed patcher and verify the manifest's hashes
        // would differ if the file actually changed between the two
        // reads.
        fs::write(tmp.path(), [0x90, 0x90]).unwrap();
        let post = sha256_hex(tmp.path()).unwrap();
        assert_ne!(pre, post, "rewriting the file must change its hash");
    }
}
