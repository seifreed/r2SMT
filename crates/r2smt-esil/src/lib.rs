#![deny(missing_docs)]
//! ESIL (Evaluable Strings Intermediate Language) lifter for r2SMT.
//!
//! radare2 attaches an ESIL string to every disassembled instruction
//! via `aoj`. ESIL describes the instruction's semantics as a postfix
//! token stream — operands push to a stack, operators pop, intermediate
//! results stay on the stack until consumed.
//!
//! This crate turns that string into a `Vec<r2smt_ir::IrStmt>` so the
//! slicer + SSA + SMT pipeline can consume any instruction radare2
//! knows how to decode, regardless of whether the per-mnemonic
//! lifter in `r2smt-slicer` already has a handler for it.
//!
//! The lifter is intentionally a strict subset of ESIL — it returns
//! an [`EsilError`] on unrecognised tokens, control-flow markers
//! (`GOTO`, `BREAK`, `?{`), and operators that would require a memory
//! model the slicer does not yet trust. Callers are expected to fall
//! back to the per-mnemonic handler in those cases.
//!
//! See [`lift_esil`] for the entry point.

pub mod flags;
pub mod machine;
pub mod parse;

pub use machine::{EsilError, EsilLift, lift_esil};
