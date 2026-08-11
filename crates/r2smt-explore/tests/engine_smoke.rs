//! Opt-in end-to-end smoke test for the radius2 engine.
//!
//! Requires the `oracle-radius2` feature (which links boolector +
//! radare2) and a local `/bin/ls`, so it is `#[ignore]`d by default per
//! the repo's Fixture Discipline. Run explicitly with:
//!
//! ```text
//! CMAKE_POLICY_VERSION_MINIMUM=3.5 \
//!   cargo test -p r2smt-explore --features oracle-radius2 -- --ignored
//! ```
//!
//! It drives the real worker subprocess through [`run_worker`], so the
//! host-side watchdog bounds the search: whatever radius2 does, the run
//! terminates within the budget and yields a well-formed
//! [`ExploreResult`]. The test asserts termination + a valid variant,
//! not a specific witness (that needs a crafted fixture — future work).
#![cfg(feature = "oracle-radius2")]
#![allow(clippy::unwrap_used)]

use r2smt_common::{Address, Arch};
use r2smt_explore::{ExploreBudget, ExploreRequest, ExploreResult, WorkerSpec, run_worker};

#[test]
#[ignore = "requires oracle-radius2 feature (boolector/radare2) and /bin/ls"]
fn test_worker_run_terminates_within_budget_and_returns_valid_result() {
    let spec = WorkerSpec {
        program: env!("CARGO_BIN_EXE_r2smt-explore-worker").to_string(),
        prefix_args: Vec::new(),
    };
    let request = ExploreRequest {
        binary_path: "/bin/ls".into(),
        target: Address::new(0x1000),
        arch: Arch::Aarch64,
    };
    let budget = ExploreBudget {
        wall_clock_ms: 4_000,
        max_paths: 50,
    };

    let result = run_worker(&spec, &request, budget).unwrap();
    assert!(matches!(
        result,
        ExploreResult::ReachedWith(_)
            | ExploreResult::NotFoundWithinBudget { .. }
            | ExploreResult::Inconclusive { .. }
    ));
}
