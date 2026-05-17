#![deny(missing_docs)]
//! Ghidra / SLEIGH P-code lifter for r2SMT.
//!
//! radare2's r2ghidra plugin emits SLEIGH P-code for a run of
//! instructions via `pdgsd N` — a decompiler-grade IR with explicit
//! SSA-style `unique` varnodes, regular ~30-opcode integer/boolean
//! semantics, and *explicit* flag derivation (no `NZCV` guessing). That
//! makes it a cleaner analysis source than per-instruction ESIL for
//! the cases it covers.
//!
//! This crate turns that text into a `Vec<r2smt_ir::IrStmt>` so the
//! existing slicer → SSA → SMT pipeline can consume it unchanged.
//!
//! Like the `r2smt_esil` crate it is intentionally a **strict
//! subset**: [`lift_pcode`] returns a [`PcodeError`] on any opcode or
//! flag construct whose lowering is not *provably* sound against the
//! IR model (notably ARM `NZCV` C/V/N polarity, which differs from the
//! per-mnemonic `AArch64` flag model — only the Z flag maps cleanly).
//! Callers fall back to the ESIL / per-mnemonic lifter on error, so an
//! unsupported construct never produces a wrong verdict — it just
//! declines the P-code path.
//!
//! See [`lift_pcode`] for the entry point and [`parse`] for the pure
//! grammar parser.

pub mod machine;
pub mod parse;

pub use machine::{PcodeError, PcodeLift, lift_pcode};
pub use parse::{PcodeInsn, PcodeOp, Varnode, parse_pcode};
