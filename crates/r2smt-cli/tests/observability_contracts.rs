//! Process-boundary contracts for structured logging and safe configuration.
#![allow(clippy::expect_used)]

use std::process::{Command, Output};

use serde_json::Value;

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_r2smt"))
        .args(args)
        .env_remove("NO_COLOR")
        .env_remove("RUST_LOG")
        .env("R2SMT_LOG_FORMAT", "json")
        .env("R2SMT_ENVIRONMENT", "test")
        .env("R2SMT_CORRELATION_ID", "test-correlation-1")
        .output()
        .expect("r2smt process must start")
}

fn events(output: &Output) -> Vec<Value> {
    String::from_utf8_lossy(&output.stderr)
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

#[test]
fn test_json_log_contains_normalized_process_context() {
    let output = run(&["version"]);
    let events = events(&output);
    let first = events.first().expect("lifecycle event must be emitted");
    assert_eq!(
        (
            first["level"].as_str(),
            first["event_name"].as_str(),
            first["span"]["service"].as_str(),
            first["span"]["environment"].as_str(),
            first["span"]["operation"].as_str(),
            first["span"]["correlation_id"].as_str(),
        ),
        (
            Some("INFO"),
            Some("command.started"),
            Some("r2smt"),
            Some("test"),
            Some("version"),
            Some("test-correlation-1"),
        )
    );
}

#[test]
fn test_json_log_records_success_duration() {
    let output = run(&["version"]);
    let events = events(&output);
    let completed = events
        .iter()
        .find(|event| event["event_name"] == "command.completed")
        .expect("completion event must be emitted");
    assert!(completed["duration_ms"].is_u64());
}

#[test]
fn test_rust_log_suppresses_lower_levels() {
    let output = Command::new(env!("CARGO_BIN_EXE_r2smt"))
        .arg("version")
        .env("R2SMT_LOG_FORMAT", "json")
        .env("RUST_LOG", "error")
        .output()
        .expect("r2smt process must start");
    assert!(output.stderr.is_empty());
}

#[test]
fn test_operation_error_is_structured() {
    let output = run(&["analyze", "this-file-does-not-exist"]);
    let events = events(&output);
    let failed = events
        .iter()
        .find(|event| event["event_name"] == "command.failed")
        .expect("failure event must be emitted");
    assert_eq!(
        (
            failed["level"].as_str(),
            failed["status"].as_str(),
            failed["error_type"].as_str(),
        ),
        (Some("ERROR"), Some("error"), Some("operation_error"))
    );
}

#[test]
fn test_sensitive_error_is_redacted_from_stderr() {
    let output = run(&["analyze", "password=hunter2"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("hunter2"));
}

#[test]
fn test_invalid_format_fails_closed() {
    let output = Command::new(env!("CARGO_BIN_EXE_r2smt"))
        .arg("version")
        .env("R2SMT_LOG_FORMAT", "xml")
        .output()
        .expect("r2smt process must start");
    assert!(!output.status.success());
}

#[test]
fn test_empty_batch_emits_route_lifecycle() {
    let directory = tempfile::tempdir().expect("temporary batch directory");
    let output = run(&["batch", &directory.path().to_string_lossy()]);
    let names: Vec<_> = events(&output)
        .into_iter()
        .filter_map(|event| event["event_name"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(
        names,
        ["command.started", "batch.completed", "command.completed"]
    );
}
