//! r2 `ij` (binary info) + `iej` (entrypoint) JSON parsers,
//! extracted from `parse.rs`.

use r2smt_common::{Address, Arch, Error, Result};
use serde::Deserialize;

/// Architecture metadata extracted from `ij`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryInfo {
    /// Target instruction set.
    pub arch: Arch,
    /// Pointer width in bits.
    pub bits: u8,
    /// Entry point, when reported by r2.
    pub entry: Option<Address>,
}

#[derive(Debug, Deserialize)]
struct IjBin {
    arch: String,
    bits: u8,
}

#[derive(Debug, Deserialize)]
struct IjRoot {
    bin: IjBin,
}

/// Parse the response of `ij` into [`BinaryInfo`].
///
/// Sets `entry` to `None`; the dedicated entry-point query (`iej`) feeds
/// it via [`parse_entry`].
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed or the architecture
/// is unsupported.
pub fn parse_info(json: &str) -> Result<BinaryInfo> {
    let root: IjRoot = serde_json::from_str(json).map_err(|e| Error::parse("ij", e.to_string()))?;
    let arch = arch_from_str(&root.bin.arch, root.bin.bits)?;
    Ok(BinaryInfo {
        arch,
        bits: root.bin.bits,
        entry: None,
    })
}

fn arch_from_str(name: &str, bits: u8) -> Result<Arch> {
    match (name, bits) {
        ("x86", 32) => Ok(Arch::X86),
        ("x86", 64) => Ok(Arch::X86_64),
        // radare2 reports both AArch32 and AArch64 with arch="arm" and
        // discriminates via the bits field.
        ("arm", 32) => Ok(Arch::Arm),
        ("arm", 64) => Ok(Arch::Aarch64),
        _ => Err(Error::Unsupported(format!(
            "unsupported arch '{name}' ({bits} bits)"
        ))),
    }
}

#[derive(Debug, Deserialize)]
struct IjEntry {
    vaddr: u64,
}

/// Parse the response of `iej` and return the first entry's virtual
/// address, if any.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed.
pub fn parse_entry(json: &str) -> Result<Option<Address>> {
    let entries: Vec<IjEntry> =
        serde_json::from_str(json).map_err(|e| Error::parse("iej", e.to_string()))?;
    Ok(entries.first().map(|e| Address(e.vaddr)))
}
