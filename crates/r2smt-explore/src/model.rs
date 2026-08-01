//! Explore-domain DTOs: the request, the budget, the witness model,
//! the result enum, and the crate's error type.
//!
//! None of these types can be turned into a
//! [`r2smt_common::SmtResult`]; that omission is the type fence and is
//! enforced by `tests/type_fence.rs`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use r2smt_common::{Address, Arch};
use serde::{Deserialize, Serialize};

/// Maximum number of stdin bytes retained in a [`Model`]. Larger
/// witnesses are truncated when ingesting worker output so a runaway
/// engine cannot balloon host memory through the result channel.
pub const MAX_STDIN_BYTES: usize = 64 * 1024;

/// Maximum number of `argv` entries retained in a [`Model`].
pub const MAX_ARGV: usize = 256;

/// Maximum number of initial-register bindings retained in a [`Model`].
pub const MAX_REGS: usize = 64;

/// A concrete input witness that drives execution to the requested
/// target address. This is best-effort evidence produced by an unsound
/// search — it is *not* a proof of anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Model {
    /// Bytes to feed on standard input.
    pub stdin: Vec<u8>,
    /// Process argument vector (including `argv[0]` if the engine
    /// modelled it).
    pub argv: Vec<String>,
    /// Initial register assignments, keyed by canonical register name.
    pub regs_init: BTreeMap<String, u64>,
}

impl Model {
    /// Clamp every collection to its documented host-safety cap. Applied
    /// when ingesting worker output so untrusted engine results cannot
    /// exhaust host memory.
    #[must_use]
    pub fn clamped(mut self) -> Self {
        self.stdin.truncate(MAX_STDIN_BYTES);
        self.argv.truncate(MAX_ARGV);
        if self.regs_init.len() > MAX_REGS {
            let keep: Vec<String> = self.regs_init.keys().take(MAX_REGS).cloned().collect();
            self.regs_init.retain(|k, _| keep.contains(k));
        }
        self
    }
}

/// Outcome of an exploration run.
///
/// The absence of an `SmtResult`-bearing variant (or any conversion into
/// one) is intentional and load-bearing: see the crate-level type fence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ExploreResult {
    /// The target was reached; the witness is attached.
    ReachedWith(Model),
    /// The search exhausted its wall-clock or path budget without
    /// reaching the target. This says nothing about reachability.
    NotFoundWithinBudget {
        /// Which budget was hit, in engine-defined terms.
        reason: String,
    },
    /// The engine could not run or gave an answer that cannot be trusted
    /// even as a witness (spawn failure, malformed output, engine not
    /// compiled in).
    Inconclusive {
        /// Free-form diagnostic.
        reason: String,
    },
}

impl ExploreResult {
    /// Build an [`ExploreResult::Inconclusive`].
    #[must_use]
    pub fn inconclusive(reason: impl Into<String>) -> Self {
        Self::Inconclusive {
            reason: reason.into(),
        }
    }

    /// Build an [`ExploreResult::NotFoundWithinBudget`].
    #[must_use]
    pub fn not_found(reason: impl Into<String>) -> Self {
        Self::NotFoundWithinBudget {
            reason: reason.into(),
        }
    }

    /// Whether this result carries a concrete witness.
    #[must_use]
    pub fn is_reached(&self) -> bool {
        matches!(self, Self::ReachedWith(_))
    }
}

/// Host-side bounds on a single exploration run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExploreBudget {
    /// Wall-clock deadline, in milliseconds, enforced by the parent
    /// process (not the engine). When exceeded the worker is killed.
    pub wall_clock_ms: u64,
    /// Path-explosion cap handed to the engine: the maximum number of
    /// symbolic states it may fork before giving up.
    pub max_paths: u64,
}

impl Default for ExploreBudget {
    fn default() -> Self {
        Self {
            wall_clock_ms: 30_000,
            max_paths: 10_000,
        }
    }
}

/// A request to reach `target` in `binary_path`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExploreRequest {
    /// Path to the binary on disk.
    pub binary_path: PathBuf,
    /// Address the search must reach.
    pub target: Address,
    /// Target architecture (drives the engine's ESIL/lifter setup).
    pub arch: Arch,
}

/// Failure modes of the exploration driver. Budget exhaustion is *not*
/// an error — it surfaces as
/// [`ExploreResult::NotFoundWithinBudget`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExploreError {
    /// The worker subprocess could not be spawned.
    #[error("failed to spawn explore worker: {0}")]
    WorkerSpawn(String),
    /// An I/O error occurred while communicating with the worker.
    #[error("explore worker io error: {0}")]
    WorkerIo(String),
    /// The worker exited successfully but its output could not be parsed.
    #[error("explore worker produced malformed output: {0}")]
    MalformedOutput(String),
}
