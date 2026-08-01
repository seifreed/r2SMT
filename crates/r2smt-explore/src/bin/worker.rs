//! Exploration worker subprocess.
//!
//! Invoked by [`r2smt_explore::run_worker`] as
//! `r2smt-explore-worker --request <in.json> --result <out.json>
//! --max-paths <n>`. It reads the [`ExploreRequest`], runs the engine
//! (behind the `oracle-radius2` feature), and writes an
//! [`ExploreResult`] to the result file. It never writes to stdout /
//! stderr — all channels are files so the parent's watchdog can kill it
//! cleanly without a pipe deadlock.

use std::path::PathBuf;

use r2smt_explore::{ExploreError, ExploreRequest, ExploreResult};

struct Args {
    request: PathBuf,
    result: PathBuf,
    max_paths: u64,
}

fn parse_args() -> Result<Args, ExploreError> {
    let mut request: Option<PathBuf> = None;
    let mut result: Option<PathBuf> = None;
    let mut max_paths: u64 = 0;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--request" => request = it.next().map(PathBuf::from),
            "--result" => result = it.next().map(PathBuf::from),
            "--max-paths" => {
                max_paths = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| ExploreError::WorkerIo("missing --max-paths value".into()))?;
            }
            other => {
                return Err(ExploreError::WorkerIo(format!("unknown argument: {other}")));
            }
        }
    }
    Ok(Args {
        request: request.ok_or_else(|| ExploreError::WorkerIo("missing --request".into()))?,
        result: result.ok_or_else(|| ExploreError::WorkerIo("missing --result".into()))?,
        max_paths,
    })
}

fn load_request(path: &PathBuf) -> Result<ExploreRequest, ExploreError> {
    let bytes = std::fs::read(path).map_err(|err| ExploreError::WorkerIo(err.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|err| ExploreError::MalformedOutput(err.to_string()))
}

fn write_result(path: &PathBuf, result: &ExploreResult) -> Result<(), ExploreError> {
    let bytes =
        serde_json::to_vec(result).map_err(|err| ExploreError::WorkerIo(err.to_string()))?;
    std::fs::write(path, bytes).map_err(|err| ExploreError::WorkerIo(err.to_string()))
}

fn run(request: &ExploreRequest, max_paths: u64) -> ExploreResult {
    #[cfg(not(feature = "oracle-radius2"))]
    {
        let _ = (request, max_paths);
        ExploreResult::inconclusive(
            "explore engine not compiled; rebuild with --features oracle-radius2",
        )
    }
    #[cfg(feature = "oracle-radius2")]
    {
        r2smt_explore::engine::explore_engine(request, max_paths)
    }
}

fn main() -> Result<(), ExploreError> {
    let args = parse_args()?;
    let result = match load_request(&args.request) {
        Ok(request) => run(&request, args.max_paths),
        Err(err) => ExploreResult::inconclusive(format!("bad request: {err}")),
    };
    write_result(&args.result, &result)
}
