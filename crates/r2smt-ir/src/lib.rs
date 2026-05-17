#![deny(missing_docs)]
//! Program model and provider port for r2SMT.
//!
//! This crate owns the normalized representation of a binary
//! ([`Program`], [`Function`], [`BasicBlock`], [`Instruction`]) and the
//! [`BinaryProvider`] trait that adapters such as `r2smt-r2pipe`
//! implement. Domain code consumes the trait; it never depends on a
//! concrete adapter.

pub mod annotator;
pub mod byte_patcher;
pub mod decompiler;
pub mod expr;
pub mod name_hints;
pub mod program;
pub mod provider;
pub mod simplify;
pub mod stmt;

#[cfg(feature = "testing")]
pub mod testing;

pub use annotator::Annotator;
pub use byte_patcher::BytePatcher;
pub use decompiler::Decompiler;
pub use expr::{Expr, Var};
pub use name_hints::NameHints;
pub use program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
pub use provider::BinaryProvider;
pub use simplify::simplify_expr;
pub use stmt::IrStmt;
