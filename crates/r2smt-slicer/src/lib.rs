#![deny(missing_docs)]
//! Branch collection (Phase 2) and backward slicing (Phase 3) for
//! r2SMT.
//!
//! Phase 2 exposes [`collector::collect_branches`] and the supporting
//! [`condition::BranchCondition`] enum that maps every supported
//! x86 / `x86_64` conditional mnemonic to its symbolic flag predicate.
//! Backward slicing lands in a sibling module in Phase 3.

pub mod collector;
pub mod condition;
pub mod effect;
pub mod lift;
pub mod registers;
pub mod slice;

pub use collector::{BranchCandidate, collect_branches, collect_function_branches};
pub use condition::{BranchCondition, BranchKind};
pub use effect::{InstructionEffect, InstructionKind, analyze, canonical_register};
pub use lift::{LiftedSlice, lift_branch_condition, lift_slice};
pub use registers::{RegisterLayout, alias_for, register_layout};
pub use slice::{Slice, SliceLimits, SliceStatus, slice_branch};
