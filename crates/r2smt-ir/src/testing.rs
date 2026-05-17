//! In-memory test doubles for [`BinaryProvider`].
//!
//! Gated behind the `testing` feature so they are excluded from
//! production builds. Used by `r2smt-core` unit tests to exercise the
//! domain without a real radare2 instance.

use std::collections::BTreeMap;

use r2smt_common::{Address, Error, Result};

use crate::annotator::Annotator;
use crate::byte_patcher::BytePatcher;
use crate::decompiler::Decompiler;
use crate::program::{Function, Program};
use crate::provider::BinaryProvider;

/// A canned provider that returns a fixed [`Program`] on every call.
#[derive(Debug, Clone)]
pub struct InMemoryProvider {
    /// The program returned by [`load_program`].
    ///
    /// [`load_program`]: BinaryProvider::load_program
    pub program: Program,
}

impl InMemoryProvider {
    /// Wrap a `Program` for use in tests.
    #[must_use]
    pub fn new(program: Program) -> Self {
        Self { program }
    }
}

impl BinaryProvider for InMemoryProvider {
    fn load_program(&mut self) -> Result<Program> {
        Ok(self.program.clone())
    }

    fn load_function(&mut self, address: Address) -> Result<Function> {
        self.program
            .functions
            .iter()
            .find(|f| f.address == address)
            .cloned()
            .ok_or_else(|| Error::parse("in_memory_provider", format!("no function at {address}")))
    }
}

/// In-memory implementation of [`Annotator`] for tests.
///
/// Stores every `set_comment` call in a `BTreeMap` keyed by address so
/// tests can assert what would have been written to the underlying tool.
#[derive(Debug, Default, Clone)]
pub struct InMemoryAnnotator {
    /// Comments keyed by address.
    pub comments: BTreeMap<Address, String>,
}

impl InMemoryAnnotator {
    /// Create an empty annotator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Annotator for InMemoryAnnotator {
    fn set_comment(&mut self, address: Address, comment: &str) -> Result<()> {
        self.comments.insert(address, comment.to_string());
        Ok(())
    }
}

/// In-memory implementation of [`Decompiler`] for tests.
///
/// Returns canned pseudocode for addresses present in `sources`, and
/// `Ok(None)` for anything else — exercising both the
/// backend-available and backend-absent paths without radare2.
#[derive(Debug, Default, Clone)]
pub struct InMemoryDecompiler {
    /// Canned pseudocode keyed by function address.
    pub sources: BTreeMap<Address, String>,
}

impl InMemoryDecompiler {
    /// Create an empty decompiler (every lookup yields `Ok(None)`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Decompiler for InMemoryDecompiler {
    fn pseudocode(&mut self, function: Address) -> Result<Option<String>> {
        Ok(self.sources.get(&function).cloned())
    }
}

/// In-memory [`BytePatcher`] test double backed by a contiguous buffer.
///
/// `base_address` is the virtual address corresponding to byte zero of
/// `bytes`. All `read_bytes` / `write_bytes` calls index into `bytes`
/// relative to `base_address`. `assemble_table` is consulted by
/// [`BytePatcher::assemble`]: tests register the expected encodings
/// ahead of time (e.g. `("nop", vec![0x90])`) and the patcher returns
/// them verbatim.
///
/// This double intentionally does not contain a real assembler — the
/// production assembler lives in the r2 adapter. Tests that need to
/// exercise assembling without r2 register the expected bytes
/// explicitly.
#[derive(Debug, Clone)]
pub struct InMemoryBytePatcher {
    /// Address that maps to byte zero of `bytes`.
    pub base_address: Address,
    /// Backing byte buffer.
    pub bytes: Vec<u8>,
    /// `asm string` → encoded bytes lookup used by [`BytePatcher::assemble`].
    pub assemble_table: BTreeMap<String, Vec<u8>>,
}

impl InMemoryBytePatcher {
    /// Build a patcher that maps `base_address` to byte zero of
    /// `bytes`.
    #[must_use]
    pub fn new(base_address: Address, bytes: Vec<u8>) -> Self {
        Self {
            base_address,
            bytes,
            assemble_table: BTreeMap::new(),
        }
    }

    /// Register the bytes [`BytePatcher::assemble`] should return for `asm`.
    pub fn add_assemble(&mut self, asm: impl Into<String>, encoding: Vec<u8>) {
        self.assemble_table.insert(asm.into(), encoding);
    }

    fn offset_for(&self, address: Address) -> Result<usize> {
        if address.get() < self.base_address.get() {
            return Err(Error::parse(
                "in_memory_byte_patcher",
                format!(
                    "address {address} below base {base}",
                    base = self.base_address
                ),
            ));
        }
        let raw = address.get() - self.base_address.get();
        usize::try_from(raw).map_err(|_| {
            Error::parse(
                "in_memory_byte_patcher",
                format!("address {address} exceeds host usize"),
            )
        })
    }
}

impl BytePatcher for InMemoryBytePatcher {
    fn read_bytes(&mut self, address: Address, size: usize) -> Result<Vec<u8>> {
        let start = self.offset_for(address)?;
        let end = start.checked_add(size).ok_or_else(|| {
            Error::parse(
                "in_memory_byte_patcher",
                "size overflow on read".to_string(),
            )
        })?;
        if end > self.bytes.len() {
            return Err(Error::parse(
                "in_memory_byte_patcher",
                format!("read past end ({end} > {})", self.bytes.len()),
            ));
        }
        Ok(self.bytes[start..end].to_vec())
    }

    fn write_bytes(&mut self, address: Address, bytes: &[u8]) -> Result<()> {
        let start = self.offset_for(address)?;
        let end = start.checked_add(bytes.len()).ok_or_else(|| {
            Error::parse(
                "in_memory_byte_patcher",
                "size overflow on write".to_string(),
            )
        })?;
        if end > self.bytes.len() {
            return Err(Error::parse(
                "in_memory_byte_patcher",
                format!("write past end ({end} > {})", self.bytes.len()),
            ));
        }
        self.bytes[start..end].copy_from_slice(bytes);
        Ok(())
    }

    fn assemble(&mut self, _address: Address, asm: &str) -> Result<Vec<u8>> {
        self.assemble_table.get(asm).cloned().ok_or_else(|| {
            Error::parse(
                "in_memory_byte_patcher",
                format!("no canned encoding for '{asm}'"),
            )
        })
    }
}
