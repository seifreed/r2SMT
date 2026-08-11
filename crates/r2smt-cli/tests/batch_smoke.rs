//! End-to-end smoke test for `r2smt batch`.
//!
//! Ignored by default to keep the workspace suite hermetic (Fixture
//! Discipline — no corpus, no radare2 in the default tree). Run with
//! `cargo test -p r2smt-cli -- --ignored` on a host that has
//! `radare2` on `PATH`. Exercises the parallel sweep, per-sample
//! isolation, and the bounded aggregate JSON.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn sample_source() -> Option<PathBuf> {
    let p = PathBuf::from("/bin/ls");
    p.exists().then_some(p)
}

#[test]
#[ignore = "requires radare2 on PATH"]
fn batch_sweeps_directory_and_emits_bounded_aggregate() {
    let Some(src) = sample_source() else {
        eprintln!("no sample source (/bin/ls); skipping batch smoke");
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    for name in ["sample_a", "sample_b"] {
        fs::copy(&src, dir.path().join(name)).expect("copy sample");
    }
    let out = dir.path().join("agg.json");

    let status = Command::new(env!("CARGO_BIN_EXE_r2smt"))
        .arg("batch")
        .arg(dir.path())
        .arg("--threads")
        .arg("2")
        .arg("--json")
        .arg(&out)
        .status()
        .expect("spawn r2smt");
    assert!(status.success(), "batch must never abort the sweep");

    let json = fs::read_to_string(&out).expect("aggregate written");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let samples = value
        .get("samples")
        .and_then(serde_json::Value::as_array)
        .expect("samples array");
    assert_eq!(samples.len(), 2, "one entry per sample file");
    for s in samples {
        assert!(s.get("path").is_some(), "each entry carries its path");
        assert!(
            s.get("outcome").and_then(|o| o.get("status")).is_some(),
            "each entry carries a tagged outcome"
        );
    }
}
