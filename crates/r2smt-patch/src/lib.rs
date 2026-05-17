#![deny(missing_docs)]
//! Safe binary patching for r2SMT.
//!
//! Phase 10 closes the loop from solver verdict → committed binary
//! change. Every applied patch is recorded in a [`PatchManifest`] and
//! a full-file backup is taken *before* any byte is written, so the
//! pipeline can roll back to a known-good state even if the host
//! process is killed mid-flight.
//!
//! The crate is sample-agnostic: it never inspects sample-specific
//! values or branches on opcode signatures from a single family.
//! Strategies are defined in terms of the abstract finding kinds
//! produced by `r2smt-core`.

pub mod aarch64_encoding;
pub mod apply;
pub mod arm_encoding;
pub mod digest;
pub mod manifest;
pub mod plan;
pub mod x86_encoding;

pub use apply::{ApplyConfig, apply_plan, rollback_from_manifest};
pub use arm_encoding::{ARM_INSTRUCTION_BYTES, arm_nop_buffer, arm_nop_bytes};
pub use digest::sha256_hex;
pub use manifest::{PatchManifest, PatchRecord};
pub use plan::{PatchPlan, PlanOperation, build_plan};
pub use x86_encoding::{nop_buffer, patch_cmovcc_to_mov, patch_setcc};
