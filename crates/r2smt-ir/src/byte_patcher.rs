//! Port: a contract for reading and writing raw bytes at guest
//! addresses.
//!
//! Used by `r2smt-patch` to apply conservative binary patches without
//! depending on any concrete disassembler. Adapters (`r2smt-r2pipe`,
//! in-memory test double) implement this trait.

use r2smt_common::{Address, Result};

/// Bidirectional byte-level view of a target binary.
///
/// Read and write happen at *virtual addresses* (`Address`). The
/// underlying adapter is responsible for mapping those to file
/// offsets — `r2smt-patch` never assumes a particular file layout.
pub trait BytePatcher {
    /// Read `size` bytes starting at `address`.
    ///
    /// # Errors
    ///
    /// Returns an adapter-specific error if the address is not mapped
    /// or the read fails.
    fn read_bytes(&mut self, address: Address, size: usize) -> Result<Vec<u8>>;

    /// Overwrite the bytes at `address` with `bytes`.
    ///
    /// Implementations must write exactly `bytes.len()` bytes; they
    /// must not extend the file, change layout, or alter neighbouring
    /// bytes. Callers are responsible for preserving instruction
    /// boundaries.
    ///
    /// # Errors
    ///
    /// Returns an adapter-specific error if the write is refused or
    /// fails to complete.
    fn write_bytes(&mut self, address: Address, bytes: &[u8]) -> Result<()>;

    /// Assemble `asm` (in the target's native syntax) as if it lived
    /// at `address`, returning the encoded bytes *without* writing
    /// them. Callers use this to know in advance whether a patch will
    /// fit in the original instruction's footprint.
    ///
    /// # Errors
    ///
    /// Returns an adapter-specific error if the assembler rejects the
    /// input.
    fn assemble(&mut self, address: Address, asm: &str) -> Result<Vec<u8>>;
}
