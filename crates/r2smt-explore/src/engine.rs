//! radius2-backed exploration engine (compiled only under
//! `oracle-radius2`).
//!
//! Models the process's standard input as a symbolic buffer and asks
//! radius2 to find an assignment that drives execution from the binary's
//! entry point to the requested target address. A hit yields the
//! concrete stdin bytes as the witness [`Model`]; a miss (the search ran
//! out of paths, or the parent watchdog is about to kill it) yields
//! [`ExploreResult::NotFoundWithinBudget`].
//!
//! The engine is deliberately sample-agnostic: it seeds symbolic stdin
//! and nothing else, so it applies to any target without embedding
//! per-family knowledge. Richer seeding (argv, environ, register state)
//! is future work.
//!
//! **Soundness posture:** this is a *search*. A witness is real evidence
//! that the target is reachable, but a negative answer never proves
//! unreachability, and the result must never be treated as a solver
//! verdict — hence the crate's type fence.

use std::collections::BTreeMap;

use radius2::{Radius, RadiusOption, Value};

use crate::model::{ExploreRequest, ExploreResult, MAX_STDIN_BYTES, Model};

/// Number of leading stdin bytes modelled symbolically.
const STDIN_SYMBOLIC_BYTES: usize = 64;

/// Standard-input file descriptor.
const STDIN_FD: usize = 0;

/// Run the radius2 search for `request`, bounding symbolic program-counter
/// fan-out by `max_paths`. The wall-clock deadline is enforced by the
/// parent process, not here.
#[must_use]
pub fn explore_engine(request: &ExploreRequest, max_paths: u64) -> ExploreResult {
    let path = request.binary_path.to_string_lossy().into_owned();
    let eval_max = usize::try_from(max_paths).unwrap_or(usize::MAX).max(1);
    let mut radius = Radius::new_with_options(
        Some(path.as_str()),
        &[
            RadiusOption::Sims(true),
            RadiusOption::SimAll(true),
            RadiusOption::EvalMax(eval_max),
        ],
    );

    let mut state = radius.entry_state();
    let symbolic: Vec<Value> = (0..STDIN_SYMBOLIC_BYTES)
        .map(|i| state.symbolic_value(&format!("stdin_{i}"), 8))
        .collect();
    state.fill_file(STDIN_FD, &symbolic);

    match radius.run_until(state, request.target.get(), &[]) {
        Some(mut reached) => {
            let stdin = concretise_stdin(&mut reached, &symbolic);
            ExploreResult::ReachedWith(Model {
                stdin,
                argv: Vec::new(),
                regs_init: BTreeMap::new(),
            })
        }
        None => ExploreResult::not_found("target not reached within the symbolic search budget"),
    }
}

/// Evaluate each symbolic stdin byte in the reached state and trim the
/// trailing NULs the solver left unconstrained.
fn concretise_stdin(reached: &mut radius2::state::State, symbolic: &[Value]) -> Vec<u8> {
    let mut stdin: Vec<u8> = symbolic
        .iter()
        .map(|byte| {
            reached
                .evaluate_bytes(byte)
                .and_then(|bytes| bytes.into_iter().next())
                .unwrap_or(0)
        })
        .collect();
    while stdin.last() == Some(&0) {
        stdin.pop();
    }
    stdin.truncate(MAX_STDIN_BYTES);
    stdin
}
