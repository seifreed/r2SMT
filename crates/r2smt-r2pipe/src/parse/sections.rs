//! r2 `isj` (sections) JSON parser + executable-range filter,
//! extracted from `parse.rs`.

use r2smt_common::{Error, Result};
use r2smt_ir::program::Function;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ISjSection {
    vaddr: u64,
    #[serde(default)]
    vsize: u64,
    #[serde(default)]
    perm: String,
}

/// A half-open virtual-address interval `[start, end)` of a section
/// the loader maps as executable.
pub type ExecRange = (u64, u64);

/// Parse the response of `iSj` and return the half-open virtual
/// address intervals of every section the loader maps executable
/// (permission string contains `x`).
///
/// Used to enforce the format-grounded invariant that an instruction
/// must live in an executable mapping: radare2's analysis can
/// over-extend a function's CFG into data sections (string tables,
/// rodata) and decode that data as garbage instructions. Those blocks
/// are filtered out by [`address_in_ranges`] in the adapter. The set
/// is intentionally derived from r2's own permission model so it works
/// for ELF / PE / Mach-O without hardcoding section names.
///
/// Sections with zero `vsize` are skipped (they map no bytes).
/// Returns an empty vector when r2 reports no section view (e.g. a
/// fully stripped binary); callers must treat "no ranges known" as
/// "do not filter" rather than "filter everything".
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed.
pub fn parse_executable_ranges(json: &str) -> Result<Vec<ExecRange>> {
    let sections: Vec<ISjSection> =
        serde_json::from_str(json).map_err(|e| Error::parse("iSj", e.to_string()))?;
    let mut ranges: Vec<ExecRange> = sections
        .into_iter()
        .filter(|s| s.perm.contains('x') && s.vsize > 0)
        .filter_map(|s| s.vaddr.checked_add(s.vsize).map(|end| (s.vaddr, end)))
        .collect();
    ranges.sort_unstable();
    Ok(ranges)
}

/// `true` when `addr` falls inside any executable range. `ranges` is
/// the output of [`parse_executable_ranges`]; each entry is half-open
/// (`start` inclusive, `end` exclusive). The list is small (a handful
/// of code sections), so a linear scan is both clearest and fast
/// enough on the hot load path.
#[must_use]
pub fn address_in_ranges(ranges: &[ExecRange], addr: u64) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| start <= addr && addr < end)
}

/// Drop every block of `func` whose start address is not inside an
/// executable range, returning the number of blocks removed.
///
/// `ranges` must be the executable mapping from
/// [`parse_executable_ranges`]. The caller is responsible for the
/// "no ranges known" policy: an empty `ranges` would strip every
/// block, so callers must skip this call entirely when the section
/// view is unavailable (stripped binary) rather than pass `&[]`.
/// A debug assertion guards that contract.
pub fn retain_executable_blocks(func: &mut Function, ranges: &[ExecRange]) -> usize {
    debug_assert!(
        !ranges.is_empty(),
        "retain_executable_blocks called with no executable ranges — \
         caller must skip filtering when the section view is unknown"
    );
    let before = func.blocks.len();
    func.blocks
        .retain(|b| address_in_ranges(ranges, b.address.get()));
    before - func.blocks.len()
}
