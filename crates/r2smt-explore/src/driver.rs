//! Subprocess driver: runs the exploration worker under a host-side
//! wall-clock deadline and a path-explosion cap, then parses its result.
//!
//! The watchdog ([`spawn_and_wait`]) is the safety-critical piece — it
//! polls the child and kills it the moment the deadline passes, so a
//! runaway symbolic search cannot outlive its budget or hang the caller.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::model::{ExploreBudget, ExploreError, ExploreRequest, ExploreResult};

/// How often the watchdog polls the child for exit.
const POLL_INTERVAL_MS: u64 = 10;

/// Name of the sibling worker binary produced by this crate.
const WORKER_BIN: &str = "r2smt-explore-worker";

/// How a worker subprocess is invoked. Kept explicit so tests can drive
/// the watchdog with substitute programs (e.g. `sleep`) instead of the
/// real engine.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    /// Program to execute.
    pub program: String,
    /// Arguments inserted before the driver's own `--request` /
    /// `--result` / `--max-paths` arguments.
    pub prefix_args: Vec<String>,
}

/// Terminal state of a watched child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitOutcome {
    /// The child exited on its own; the flag is `status.success()`.
    Exited(bool),
    /// The wall-clock deadline elapsed and the child was killed.
    TimedOut,
}

/// Spawn `cmd` and wait for it, killing it if it runs past
/// `wall_clock_ms`.
///
/// # Errors
///
/// Returns [`ExploreError::WorkerSpawn`] if the process cannot start and
/// [`ExploreError::WorkerIo`] if polling it fails.
pub fn spawn_and_wait(mut cmd: Command, wall_clock_ms: u64) -> Result<WaitOutcome, ExploreError> {
    let mut child = cmd
        .spawn()
        .map_err(|err| ExploreError::WorkerSpawn(err.to_string()))?;
    let deadline = Instant::now() + Duration::from_millis(wall_clock_ms);
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| ExploreError::WorkerIo(err.to_string()))?
        {
            return Ok(WaitOutcome::Exited(status.success()));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(WaitOutcome::TimedOut);
        }
        std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

/// Run an exploration worker described by `spec` on `request` under
/// `budget`, returning the parsed [`ExploreResult`].
///
/// The request is handed to the worker via a temp file and the result is
/// read back from another; nothing crosses stdio, so the parent never
/// deadlocks on a pipe buffer while the watchdog is running.
///
/// # Errors
///
/// Returns [`ExploreError`] when the worker cannot be spawned, temp I/O
/// fails, or the worker's output file is missing or malformed.
pub fn run_worker(
    spec: &WorkerSpec,
    request: &ExploreRequest,
    budget: ExploreBudget,
) -> Result<ExploreResult, ExploreError> {
    let dir = tempfile::tempdir().map_err(|err| ExploreError::WorkerIo(err.to_string()))?;
    let req_path = dir.path().join("request.json");
    let out_path = dir.path().join("result.json");
    let req_json =
        serde_json::to_vec(request).map_err(|err| ExploreError::WorkerIo(err.to_string()))?;
    std::fs::write(&req_path, req_json).map_err(|err| ExploreError::WorkerIo(err.to_string()))?;

    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.prefix_args)
        .arg("--request")
        .arg(&req_path)
        .arg("--result")
        .arg(&out_path)
        .arg("--max-paths")
        .arg(budget.max_paths.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    match spawn_and_wait(cmd, budget.wall_clock_ms)? {
        WaitOutcome::TimedOut => Ok(ExploreResult::not_found(
            "wall-clock budget exhausted before target reached",
        )),
        WaitOutcome::Exited(false) => Ok(ExploreResult::inconclusive(
            "explore worker exited non-zero",
        )),
        WaitOutcome::Exited(true) => read_result(&out_path),
    }
}

/// Read and parse the worker's result file, clamping any witness to the
/// host-safety caps.
fn read_result(out_path: &std::path::Path) -> Result<ExploreResult, ExploreError> {
    let bytes = std::fs::read(out_path)
        .map_err(|err| ExploreError::MalformedOutput(format!("result file unreadable: {err}")))?;
    let result: ExploreResult = serde_json::from_slice(&bytes)
        .map_err(|err| ExploreError::MalformedOutput(err.to_string()))?;
    Ok(clamp_result(result))
}

/// Apply [`Model::clamped`] to a reached witness; pass everything else
/// through unchanged.
fn clamp_result(result: ExploreResult) -> ExploreResult {
    match result {
        ExploreResult::ReachedWith(model) => ExploreResult::ReachedWith(model.clamped()),
        other => other,
    }
}

/// Explore `request` under `budget`, returning a witness if the target
/// was reached within the budget.
///
/// With the default feature set the engine is not compiled in, so this
/// soundly returns [`ExploreResult::Inconclusive`] without spawning
/// anything. With `oracle-radius2` it resolves the sibling worker binary
/// and drives it through [`run_worker`].
#[must_use]
pub fn explore(request: &ExploreRequest, budget: ExploreBudget) -> ExploreResult {
    #[cfg(not(feature = "oracle-radius2"))]
    {
        let _ = (request, budget);
        ExploreResult::inconclusive(
            "explore engine not compiled; rebuild r2smt-explore with --features oracle-radius2",
        )
    }
    #[cfg(feature = "oracle-radius2")]
    {
        match default_worker_spec() {
            Some(spec) => run_worker(&spec, request, budget).unwrap_or_else(|err| {
                ExploreResult::inconclusive(format!("explore worker failed: {err}"))
            }),
            None => ExploreResult::inconclusive("could not locate explore worker binary"),
        }
    }
}

/// Resolve the worker binary that sits next to the current executable.
#[cfg(feature = "oracle-radius2")]
fn default_worker_spec() -> Option<WorkerSpec> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let worker = dir.join(WORKER_BIN);
    if worker.exists() {
        Some(WorkerSpec {
            program: worker.to_string_lossy().into_owned(),
            prefix_args: Vec::new(),
        })
    } else {
        None
    }
}

// Keep `WORKER_BIN` referenced even when the engine feature is off so it
// stays the single source of truth for the worker's name.
#[cfg(not(feature = "oracle-radius2"))]
const _: &str = WORKER_BIN;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use crate::model::Model;

    fn budget(ms: u64) -> ExploreBudget {
        ExploreBudget {
            wall_clock_ms: ms,
            max_paths: 100,
        }
    }

    #[test]
    fn test_spawn_and_wait_slow_child_times_out() {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let outcome = spawn_and_wait(cmd, 50).unwrap();
        assert_eq!(outcome, WaitOutcome::TimedOut);
    }

    #[test]
    fn test_spawn_and_wait_fast_success_reports_exited_true() {
        let cmd = Command::new("true");
        let outcome = spawn_and_wait(cmd, 5_000).unwrap();
        assert_eq!(outcome, WaitOutcome::Exited(true));
    }

    #[test]
    fn test_spawn_and_wait_nonzero_exit_reports_exited_false() {
        let cmd = Command::new("false");
        let outcome = spawn_and_wait(cmd, 5_000).unwrap();
        assert_eq!(outcome, WaitOutcome::Exited(false));
    }

    #[test]
    fn test_spawn_and_wait_missing_program_is_spawn_error() {
        let cmd = Command::new("r2smt-nonexistent-binary-xyz");
        let err = spawn_and_wait(cmd, 100).unwrap_err();
        assert!(matches!(err, ExploreError::WorkerSpawn(_)));
    }

    #[test]
    fn test_run_worker_timeout_yields_not_found() {
        // `sh -c 'sleep 30'` ignores the driver's trailing --request/…
        // args (they become unused positional params), so the child
        // actually sleeps and the watchdog is what stops it.
        let spec = WorkerSpec {
            program: "sh".to_string(),
            prefix_args: vec!["-c".to_string(), "sleep 30".to_string()],
        };
        let request = ExploreRequest {
            binary_path: "/bin/ls".into(),
            target: r2smt_common::Address::new(0x1000),
            arch: r2smt_common::Arch::X86_64,
        };
        let result = run_worker(&spec, &request, budget(50)).unwrap();
        assert!(matches!(result, ExploreResult::NotFoundWithinBudget { .. }));
    }

    #[test]
    fn test_clamp_result_truncates_oversized_stdin() {
        let mut model = Model::default();
        model.stdin = vec![0u8; crate::model::MAX_STDIN_BYTES + 10];
        let clamped = clamp_result(ExploreResult::ReachedWith(model));
        match clamped {
            ExploreResult::ReachedWith(m) => {
                assert_eq!(m.stdin.len(), crate::model::MAX_STDIN_BYTES);
            }
            other => panic!("expected ReachedWith, got {other:?}"),
        }
    }
}
