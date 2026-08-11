//! Port: a contract for sources of normalized program data.
//!
//! Adapters (e.g. `r2smt-r2pipe`) implement [`BinaryProvider`]; use cases
//! in `r2smt-core` accept `&mut dyn BinaryProvider` so the domain never
//! depends on a concrete tool.

use r2smt_common::{Address, Result};

use crate::name_hints::NameHints;
use crate::program::{Function, Program};

/// Source of normalized program data.
///
/// Implementations must be idempotent for read-only operations: calling
/// `load_program` twice on the same instance must produce equal results
/// modulo unspecified ordering of sibling functions / blocks.
///
/// # Errors
///
/// Returns an error if the underlying source fails to produce a coherent
/// program model. Specific failure modes are described by
/// [`r2smt_common::Error`].
pub trait BinaryProvider {
    /// Load the whole program (architecture metadata + every discovered
    /// function and its basic blocks).
    ///
    /// # Errors
    ///
    /// Propagates I/O, transport, and parse failures from the adapter.
    fn load_program(&mut self) -> Result<Program>;

    /// Load a single function by its entry address.
    ///
    /// # Errors
    ///
    /// Returns [`r2smt_common::Error::Parse`] if the adapter cannot find
    /// the function or its disassembly is malformed.
    fn load_function(&mut self, address: Address) -> Result<Function>;

    /// Best-effort: return a [`Function`] containing `address`, even if
    /// the upstream tool has not detected a function there.
    ///
    /// Default implementation simply delegates to [`Self::load_function`].
    /// Adapters with extra heuristics (radare2 `af @ addr` retry,
    /// shellcode-style basic-block discovery, …) override this to widen
    /// the success surface. The returned `Function` is *synthetic* when
    /// the upstream tool has nothing to offer — its `address` may point
    /// to the basic-block start the heuristic settled on.
    ///
    /// # Errors
    ///
    /// Propagates the underlying adapter error if every fallback fails.
    fn load_block_at(&mut self, address: Address) -> Result<Function> {
        self.load_function(address)
    }

    /// Optional human-readable aliases for the canonical names lifted
    /// from `function`. Default returns an empty hint set so adapters
    /// without symbol info opt out for free. The report layer merges
    /// these aliases into pretty-printed expressions and finding
    /// payloads.
    ///
    /// # Errors
    ///
    /// Adapter-specific. Implementations that have no symbol channel
    /// (e.g. the in-memory test double) should return `Ok(NameHints::default())`.
    fn name_hints(&mut self, function: Address) -> Result<NameHints> {
        let _ = function;
        Ok(NameHints::default())
    }
}
