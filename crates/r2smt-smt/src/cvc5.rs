//! CVC5 backend driven by SMT-LIB2 via subprocess.
//!
//! Spawns `cvc5` for each branch query (twice — once for the taken
//! polarity, once for the not-taken). The SMT-LIB2 script comes
//! straight from [`crate::smtlib::emit_query`] so the verdict ladder
//! matches the Z3 backend's combine table.
//!
//! Requires `cvc5` to be available on `$PATH`. Distribution
//! recommendation: `brew install cvc5` on macOS,
//! `apt install cvc5` on recent Debian / Ubuntu.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::Duration;

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_slicer::SliceStatus;
use r2smt_ssa::SsaLiftedSlice;
use tracing::debug;

use crate::smtlib::emit_query;

/// Failure modes specific to the CVC5 subprocess backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cvc5Error {
    /// `cvc5` was not found on `$PATH`. Carries the io error message
    /// from the spawn attempt.
    NotFound(String),
    /// The subprocess exited but its stdout did not contain a
    /// recognisable SAT verdict (`sat` / `unsat` / `unknown`).
    UnrecognisedVerdict(String),
    /// Spawning or communicating with the subprocess failed.
    SubprocessError(String),
}

/// Solve a branch via the CVC5 subprocess. Mirrors the contract of
/// [`crate::solve_branch`]: truncated slices that did not opt into
/// `unknowns_on_truncation` are reported as
/// [`SmtResult::Unsound`] without invoking the subprocess.
///
/// # Errors
///
/// Returns [`Cvc5Error`] when the subprocess cannot be spawned or
/// its output is malformed. Caller may fall back to another solver
/// or surface the failure in the verdict.
pub fn solve_branch_cvc5(
    slice: &SsaLiftedSlice,
    options: SolveOptions,
) -> Result<SmtResult, Cvc5Error> {
    let is_complete = matches!(slice.status, SliceStatus::Complete);
    if !is_complete && !slice.treat_truncation_as_inputs {
        return Ok(SmtResult::Unsound);
    }
    let script_taken = emit_query(slice, &options, true);
    let script_not_taken = emit_query(slice, &options, false);
    let taken = run_cvc5(&script_taken, options.timeout_ms)?;
    let not_taken = run_cvc5(&script_not_taken, options.timeout_ms)?;
    let verdict = combine(taken, not_taken);
    debug!(
        target: "r2smt::cvc5",
        at = %slice.branch.address,
        ?taken,
        ?not_taken,
        ?verdict,
        "cvc5 verdict"
    );
    Ok(verdict)
}

/// `(check-sat)` outcome from a single CVC5 run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SatOutcome {
    Sat,
    Unsat,
    Unknown,
}

fn combine(taken: SatOutcome, not_taken: SatOutcome) -> SmtResult {
    match (taken, not_taken) {
        (SatOutcome::Sat, SatOutcome::Unsat) => SmtResult::AlwaysTrue,
        (SatOutcome::Unsat, SatOutcome::Sat) => SmtResult::AlwaysFalse,
        (SatOutcome::Sat, SatOutcome::Sat) => SmtResult::BothPossible,
        (SatOutcome::Unsat, SatOutcome::Unsat) => SmtResult::Unsound,
        (SatOutcome::Unknown, _) | (_, SatOutcome::Unknown) => SmtResult::Timeout,
    }
}

fn run_cvc5(script: &str, timeout_ms: u32) -> Result<SatOutcome, Cvc5Error> {
    let timeout_ms_str = timeout_ms.to_string();
    let mut child = Command::new("cvc5")
        .args(["--lang", "smt2", "--tlimit-per", &timeout_ms_str])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                Cvc5Error::NotFound(err.to_string())
            } else {
                Cvc5Error::SubprocessError(err.to_string())
            }
        })?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(script.as_bytes())
            .map_err(|err| Cvc5Error::SubprocessError(err.to_string()))?;
    }
    let _ = Duration::from_millis(u64::from(timeout_ms));
    let output = child
        .wait_with_output()
        .map_err(|err| Cvc5Error::SubprocessError(err.to_string()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_verdict(&stdout)
}

fn parse_verdict(stdout: &str) -> Result<SatOutcome, Cvc5Error> {
    for line in stdout.lines().rev() {
        match line.trim() {
            "sat" => return Ok(SatOutcome::Sat),
            "unsat" => return Ok(SatOutcome::Unsat),
            "unknown" => return Ok(SatOutcome::Unknown),
            _ => {}
        }
    }
    Err(Cvc5Error::UnrecognisedVerdict(stdout.to_string()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn parse_verdict_recognises_three_outcomes() {
        assert_eq!(parse_verdict("sat\n"), Ok(SatOutcome::Sat));
        assert_eq!(parse_verdict("unsat\n"), Ok(SatOutcome::Unsat));
        assert_eq!(parse_verdict("unknown\n"), Ok(SatOutcome::Unknown));
    }

    #[test]
    fn parse_verdict_skips_leading_diagnostics() {
        // CVC5 sometimes prints version banners or comments before
        // the verdict. The parser walks from the last line backwards.
        let output = "; comment\n; another line\nsat\n";
        assert_eq!(parse_verdict(output), Ok(SatOutcome::Sat));
    }

    #[test]
    fn parse_verdict_reports_unrecognised_output() {
        let err = parse_verdict("garbage output").expect_err("should fail");
        assert!(matches!(err, Cvc5Error::UnrecognisedVerdict(_)));
    }
}
