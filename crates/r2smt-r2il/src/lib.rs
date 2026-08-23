#![deny(missing_docs)]
//! Experimental adapter for the external `r2sleigh` CLI.
//!
//! The adapter consumes `r2sleigh run --action lift --format r2cmd`.
//! That format pairs a JSON R2IL sidecar with ESIL generated from the
//! same operation. The sidecar is validated, then the existing strict
//! ESIL lifter produces r2SMT IR. Unsupported semantics fail closed.
//!
//! `r2sleigh` is LGPL-3.0-only. r2SMT does not link or redistribute it:
//! the optional executable remains across a subprocess boundary.

use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use r2smt_common::Arch;
use r2smt_ir::stmt::IrStmt;
use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

const TIMEOUT: Duration = Duration::from_secs(30);
const MAX_OUTPUT_BYTES: u64 = 4 * 1024 * 1024;

/// Architectures shared by the current r2sleigh and r2SMT contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R2sleighArch {
    /// 64-bit x86.
    X86_64,
    /// 32-bit x86.
    X86,
    /// 32-bit ARM.
    Arm,
}

impl R2sleighArch {
    fn cli_name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86-64",
            Self::X86 => "x86",
            Self::Arm => "arm",
        }
    }

    fn r2smt_arch(self) -> Arch {
        match self {
            Self::X86_64 => Arch::X86_64,
            Self::X86 => Arch::X86,
            Self::Arm => Arch::Arm,
        }
    }
}

/// Measured result of one experimental adapter invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct R2ilLift {
    /// Number of validated R2IL sidecars.
    pub r2il_operations: usize,
    /// Number of r2SMT IR statements produced.
    pub ir_statements: Vec<IrStmt>,
    /// End-to-end subprocess and adaptation time.
    pub elapsed_ms: u128,
}

/// Failures at the external adapter boundary.
#[derive(Debug, Error)]
pub enum AdapterError {
    /// The byte string or exported contract is malformed.
    #[error("invalid r2sleigh adapter input: {0}")]
    InvalidInput(String),
    /// The optional executable is absent.
    #[error("r2sleigh is unavailable: {0}")]
    Unavailable(String),
    /// The subprocess exceeded its fixed wall-clock budget.
    #[error("r2sleigh exceeded the 30 second adapter timeout")]
    Timeout,
    /// The subprocess emitted more than the bounded output size.
    #[error("r2sleigh exceeded the 4 MiB adapter output limit")]
    OutputLimit,
    /// The subprocess or strict ESIL lowering failed.
    #[error("r2sleigh adapter failed: {0}")]
    Backend(String),
}

/// Lift one hex-encoded instruction through the installed `r2sleigh`
/// executable and adapt its R2IL export to r2SMT IR.
///
/// # Errors
///
/// Returns [`AdapterError`] when input validation, the bounded
/// subprocess, the R2IL sidecar contract, or strict ESIL lowering fails.
pub fn lift_bytes(bytes_hex: &str, arch: R2sleighArch) -> Result<R2ilLift, AdapterError> {
    validate_hex(bytes_hex)?;
    let started = Instant::now();
    let output = run_bounded(bytes_hex, arch)?;
    let (r2il_operations, ir_statements) = parse_r2cmd(&output, arch.r2smt_arch())?;
    Ok(R2ilLift {
        r2il_operations,
        ir_statements,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Adapt an already captured `lift/r2cmd` export without spawning a
/// process. Useful for fixtures, offline pipelines, and fuzzing.
///
/// # Errors
///
/// Returns [`AdapterError::InvalidInput`] for malformed sidecar pairs,
/// or [`AdapterError::Backend`] when an ESIL operation is unsupported.
pub fn parse_r2cmd(text: &str, arch: Arch) -> Result<(usize, Vec<IrStmt>), AdapterError> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let mut operations = 0;
    let mut statements = Vec::new();
    while let Some(sidecar_line) = lines.next() {
        let sidecar_raw = sidecar_line
            .trim()
            .strip_prefix("# ")
            .ok_or_else(|| AdapterError::InvalidInput("expected JSON sidecar".into()))?;
        let sidecar: Value = serde_json::from_str(sidecar_raw)
            .map_err(|error| AdapterError::InvalidInput(error.to_string()))?;
        if sidecar.get("op_index").and_then(Value::as_u64).is_none()
            || sidecar.get("op").and_then(Value::as_str).is_none()
            || !sidecar.get("op_json").is_some_and(Value::is_object)
        {
            return Err(AdapterError::InvalidInput(
                "sidecar must contain op_index, op, and op_json".into(),
            ));
        }
        let esil = lines
            .next()
            .and_then(|line| line.trim().strip_prefix("ae "))
            .ok_or_else(|| AdapterError::InvalidInput("sidecar is missing its ae line".into()))?;
        let lifted = r2smt_esil::lift_esil(esil, arch)
            .map_err(|error| AdapterError::Backend(format!("{error:?}")))?;
        statements.extend(lifted.statements);
        operations += 1;
    }
    Ok((operations, statements))
}

fn validate_hex(bytes: &str) -> Result<(), AdapterError> {
    if bytes.is_empty()
        || !bytes.len().is_multiple_of(2)
        || !bytes.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(AdapterError::InvalidInput(
            "bytes must be a non-empty, even-length hex string".into(),
        ));
    }
    Ok(())
}

fn run_bounded(bytes: &str, arch: R2sleighArch) -> Result<String, AdapterError> {
    let mut child = Command::new("r2sleigh")
        .args([
            "run",
            "--arch",
            arch.cli_name(),
            "--bytes",
            bytes,
            "--action",
            "lift",
            "--format",
            "r2cmd",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| AdapterError::Unavailable(error.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AdapterError::Backend("stdout pipe unavailable".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| AdapterError::Backend("stderr pipe unavailable".into()))?;
    let stdout = thread::spawn(move || read_bounded(stdout));
    let stderr = thread::spawn(move || read_bounded(stderr));

    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| AdapterError::Backend(error.to_string()))?
        {
            break Some(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = join_reader(stdout)?;
    let stderr = join_reader(stderr)?;
    let Some(status) = status else {
        return Err(AdapterError::Timeout);
    };
    if stdout.len() as u64 > MAX_OUTPUT_BYTES || stderr.len() as u64 > MAX_OUTPUT_BYTES {
        return Err(AdapterError::OutputLimit);
    }
    if !status.success() {
        return Err(AdapterError::Backend(
            String::from_utf8_lossy(&stderr).trim().to_string(),
        ));
    }
    String::from_utf8(stdout).map_err(|error| AdapterError::Backend(error.to_string()))
}

fn read_bounded(mut reader: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(MAX_OUTPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn join_reader(
    reader: thread::JoinHandle<std::io::Result<Vec<u8>>>,
) -> Result<Vec<u8>, AdapterError> {
    reader
        .join()
        .map_err(|_| AdapterError::Backend("output reader panicked".into()))?
        .map_err(|error| AdapterError::Backend(error.to_string()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const COPY: &str = concat!(
        "# {\"op_index\":0,\"op\":\"Copy\",\"op_json\":{\"Copy\":{}}}\n",
        "ae rax,rax,=\n"
    );

    #[test]
    fn official_r2cmd_pair_lifts_to_r2smt_ir() {
        let (operations, statements) = parse_r2cmd(COPY, Arch::X86_64).unwrap();
        assert_eq!(operations, 1);
        assert_eq!(statements.len(), 1);
    }

    #[test]
    fn malformed_or_unsupported_exports_fail_closed() {
        assert!(parse_r2cmd("ae rax,rax,=", Arch::X86_64).is_err());
        assert!(
            parse_r2cmd(
                "# {\"op_index\":0,\"op\":\"Bad\",\"op_json\":{}}\nae GOTO\n",
                Arch::X86_64
            )
            .is_err()
        );
    }
}
