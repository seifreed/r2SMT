//! Dependency-fence contract: the verify/patch crates must never
//! transitively depend on `r2smt-explore`. If they did, the unsound
//! exploration engine could be linked into a code path that produces
//! verdicts or patches — exactly what the fence exists to prevent.
//!
//! This walks the resolved dependency graph from `cargo metadata` and
//! fails if the guarded closure reaches `r2smt-explore`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::Command;

use serde_json::Value;

/// Crates whose dependency closure must exclude `r2smt-explore`.
const GUARDED: &[&str] = &["r2smt-core", "r2smt-smt", "r2smt-report", "r2smt-patch"];

fn workspace_metadata() -> Value {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1"])
        .output()
        .expect("cargo metadata should run");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cargo metadata should emit valid JSON")
}

#[test]
fn test_verify_patch_closure_excludes_explore() {
    let md = workspace_metadata();

    let name_to_id: HashMap<&str, &str> = md["packages"]
        .as_array()
        .expect("packages array")
        .iter()
        .map(|p| {
            (
                p["name"].as_str().expect("package name"),
                p["id"].as_str().expect("package id"),
            )
        })
        .collect();

    let mut deps: HashMap<&str, Vec<&str>> = HashMap::new();
    for node in md["resolve"]["nodes"].as_array().expect("resolve nodes") {
        let id = node["id"].as_str().expect("node id");
        let out = node["dependencies"]
            .as_array()
            .expect("node dependencies")
            .iter()
            .map(|d| d.as_str().expect("dependency id"))
            .collect();
        deps.insert(id, out);
    }

    let explore_id = *name_to_id
        .get("r2smt-explore")
        .expect("r2smt-explore is a workspace member");

    for guarded in GUARDED {
        let start = *name_to_id
            .get(guarded)
            .unwrap_or_else(|| panic!("{guarded} is a workspace member"));
        let mut seen: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::from([start]);
        while let Some(cur) = queue.pop_front() {
            if !seen.insert(cur) {
                continue;
            }
            for dep in deps.get(cur).into_iter().flatten() {
                queue.push_back(dep);
            }
        }
        assert!(
            !seen.contains(explore_id),
            "dep fence broken: {guarded} transitively depends on r2smt-explore"
        );
    }
}
