//! Patch manifest: the durable record of every byte r2SMT has changed.
//!
//! Manifests are written to JSON next to the patched binary. They
//! drive both human review and machine rollback ([`crate::apply::rollback_from_manifest`]).

use std::fs;
use std::path::Path;

use r2smt_common::{Address, Error, Result};
use r2smt_core::{Confidence, FindingKind};
use serde::{Deserialize, Serialize};

/// Wire-format version of the manifest.
///
/// Bumped on any incompatible schema change. Rollback refuses to
/// operate on manifests with an unknown version so out-of-date tools
/// cannot accidentally corrupt newer patches.
pub const MANIFEST_VERSION: u32 = 1;

/// One byte-level patch operation as recorded in a [`PatchManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchRecord {
    /// Address of the patched instruction.
    pub address: Address,
    /// Strategy name (matches `r2smt-report::PatchStrategy::as_str`).
    pub strategy: String,
    /// Finding kind that motivated the patch.
    pub kind: FindingKind,
    /// Confidence at which the patch was committed.
    pub confidence: Confidence,
    /// Lowercase hex of the bytes present at `address` *before* the
    /// patch was applied. Rollback restores these.
    pub original_bytes_hex: String,
    /// Lowercase hex of the bytes the patch wrote at `address`.
    pub patched_bytes_hex: String,
    /// Human-readable explanation forwarded from the suggestion engine.
    pub rationale: String,
}

impl PatchRecord {
    /// Decode `original_bytes_hex`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] if the hex string is malformed.
    pub fn original_bytes(&self) -> Result<Vec<u8>> {
        hex::decode(&self.original_bytes_hex)
            .map_err(|e| Error::parse("patch_record.original_bytes_hex", e.to_string()))
    }

    /// Decode `patched_bytes_hex`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] if the hex string is malformed.
    pub fn patched_bytes(&self) -> Result<Vec<u8>> {
        hex::decode(&self.patched_bytes_hex)
            .map_err(|e| Error::parse("patch_record.patched_bytes_hex", e.to_string()))
    }
}

/// Durable record of an r2SMT patch session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchManifest {
    /// Schema version (currently [`MANIFEST_VERSION`]).
    pub manifest_version: u32,
    /// r2SMT version that produced the manifest.
    pub r2smt_version: String,
    /// Display path of the patched binary at apply-time.
    pub binary: String,
    /// SHA-256 of the binary before any patch was applied.
    pub binary_sha256_before: String,
    /// SHA-256 of the binary after all operations completed.
    pub binary_sha256_after: String,
    /// Absolute path of the backup created before patching.
    pub backup_path: String,
    /// Operations applied, in execution order.
    pub operations: Vec<PatchRecord>,
}

impl PatchManifest {
    /// Default file name r2SMT uses to persist a manifest next to its
    /// binary.
    pub const DEFAULT_FILE_NAME: &'static str = "r2smt.manifest.json";

    /// Render the manifest as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Propagates serialisation failures.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| Error::parse("patch_manifest", e.to_string()))
    }

    /// Write the manifest to `path` as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Propagates I/O and serialisation failures.
    pub fn write_to(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = self.to_json()?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load a manifest from `path`.
    ///
    /// Refuses to deserialize manifests with a version r2SMT does not
    /// understand so old binaries cannot corrupt newer patches.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Parse`] if the file is missing, malformed, or
    /// written by a future schema version.
    pub fn read_from(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let parsed: Self = serde_json::from_str(&raw)
            .map_err(|e| Error::parse("patch_manifest", e.to_string()))?;
        if parsed.manifest_version != MANIFEST_VERSION {
            return Err(Error::parse(
                "patch_manifest",
                format!(
                    "unsupported manifest version {got} (this build only handles {expected})",
                    got = parsed.manifest_version,
                    expected = MANIFEST_VERSION,
                ),
            ));
        }
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use tempfile::NamedTempFile;

    use super::*;

    fn sample_manifest() -> PatchManifest {
        PatchManifest {
            manifest_version: MANIFEST_VERSION,
            r2smt_version: "0.1.0".into(),
            binary: "/tmp/sample.exe".into(),
            binary_sha256_before: "a".repeat(64),
            binary_sha256_after: "b".repeat(64),
            backup_path: "/tmp/sample.exe.r2smt.bak".into(),
            operations: vec![PatchRecord {
                address: Address(0x40_1050),
                strategy: "nop_jcc".into(),
                kind: FindingKind::DeadBranch,
                confidence: Confidence::High,
                original_bytes_hex: "7505".into(),
                patched_bytes_hex: "9090".into(),
                rationale: "jne is never taken".into(),
            }],
        }
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let original = sample_manifest();
        let json = original.to_json().unwrap();
        let back: PatchManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn manifest_round_trips_through_disk() {
        let original = sample_manifest();
        let tmp = NamedTempFile::new().unwrap();
        original.write_to(tmp.path()).unwrap();
        let back = PatchManifest::read_from(tmp.path()).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn read_from_rejects_unknown_version() {
        let mut bad = sample_manifest();
        bad.manifest_version = MANIFEST_VERSION + 1;
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), bad.to_json().unwrap()).unwrap();
        let err = PatchManifest::read_from(tmp.path()).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("unsupported manifest version"));
    }

    #[test]
    fn patch_record_decodes_hex_round_trip() {
        let record = &sample_manifest().operations[0];
        assert_eq!(record.original_bytes().unwrap(), vec![0x75, 0x05]);
        assert_eq!(record.patched_bytes().unwrap(), vec![0x90, 0x90]);
    }
}
