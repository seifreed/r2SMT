#![deny(missing_docs)]
//! radare2 adapter for the r2SMT [`BinaryProvider`] port.
//!
//! Spawns a radare2 process through `r2pipe`, runs `aaa`, and translates
//! the JSON responses from `ij`, `aflj`, and `agfj` into the normalized
//! [`Program`] model owned by `r2smt-ir`.
//!
//! [`BinaryProvider`]: r2smt_ir::BinaryProvider
//! [`Program`]: r2smt_ir::Program

pub mod b64;
pub mod parse;
pub mod provider;

pub use provider::{AnalysisLevel, R2PipeProvider};
