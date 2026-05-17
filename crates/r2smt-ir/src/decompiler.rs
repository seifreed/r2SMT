//! Port: a contract for fetching decompiler pseudocode.
//!
//! Sibling of [`BinaryProvider`](crate::provider::BinaryProvider)
//! (read-only program data) and [`Annotator`](crate::annotator::Annotator)
//! (write-side). [`Decompiler`] is an *optional context source*: the
//! domain never depends on it, and analysis verdicts never consume it.
//! The composition root wires a concrete implementation (live radare2
//! via r2ghidra / r2dec, an IDA Hex-Rays bridge, …) only when the user
//! opts in, and any backend absence must degrade to `Ok(None)` rather
//! than failing the run.

use r2smt_common::{Address, Result};

/// Source of human-readable decompiled pseudocode keyed by function
/// address.
pub trait Decompiler {
    /// Return decompiled pseudocode for the function at `function`, or
    /// `Ok(None)` when no decompiler backend is available.
    ///
    /// Absence of a decompiler is **not** an error: a missing backend
    /// must yield `Ok(None)` so an analysis run still completes.
    ///
    /// # Errors
    ///
    /// Returns an adapter-specific error only when the transport
    /// itself fails (broken pipe, malformed response from a backend
    /// that *is* present), never merely because no backend is loaded.
    fn pseudocode(&mut self, function: Address) -> Result<Option<String>>;
}
