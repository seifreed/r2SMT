//! Port: a contract for writing user annotations back to a target tool.
//!
//! `BinaryProvider` is a read-only contract; [`Annotator`] is its
//! write-side sibling. r2SMT consumes the trait so the domain remains
//! independent of any concrete annotation target (live radare2, an r2
//! project file, an IDA database, …).

use r2smt_common::{Address, Result};

/// Sink for human-readable annotations keyed by guest address.
///
/// Semantics: `set_comment(addr, text)` associates `text` with `addr`
/// in the underlying tool, replacing any annotation previously set at
/// that address by r2SMT or by the user.
pub trait Annotator {
    /// Attach `comment` to `address`, replacing any prior comment.
    ///
    /// # Errors
    ///
    /// Returns an adapter-specific error if the transport fails or the
    /// target rejects the request.
    fn set_comment(&mut self, address: Address, comment: &str) -> Result<()>;
}
