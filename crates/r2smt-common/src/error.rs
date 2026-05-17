//! Crate-wide error type and `Result` alias.
//!
//! `Error` aggregates the open-domain failures r2SMT components can
//! surface. Variants with `String` payloads carry context that cannot be
//! enumerated at compile time (tool output, file format reasons); closed
//! enumerations live in dedicated enums attached to each variant when
//! introduced.

use std::io;

use thiserror::Error;

/// Top-level r2SMT error.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Host I/O failure surfaced from the standard library.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Communication with the radare2 process failed.
    #[error("r2pipe error: {0}")]
    R2Pipe(String),

    /// Failure to parse external data (JSON, hex, addresses, ...).
    #[error("parse error in {context}: {reason}")]
    Parse {
        /// Where in the pipeline the parse failed (e.g. `"aflj"`).
        context: &'static str,
        /// Free-form description of the parse failure.
        reason: String,
    },

    /// The current operation is not supported on the requested target.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Convenience alias used across the workspace.
pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    /// Build a `Parse` error with a static context string and a free-form
    /// reason.
    #[must_use]
    pub fn parse(context: &'static str, reason: impl Into<String>) -> Self {
        Self::Parse {
            context,
            reason: reason.into(),
        }
    }

    /// Build an `R2Pipe` error from any displayable value.
    pub fn r2pipe(source: impl std::fmt::Display) -> Self {
        Self::R2Pipe(source.to_string())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn parse_builder_keeps_context_and_reason() {
        let err = Error::parse("aflj", "missing offset field");
        let rendered = err.to_string();
        assert!(rendered.contains("aflj"));
        assert!(rendered.contains("missing offset field"));
    }

    #[test]
    fn io_error_round_trip() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "boom");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
