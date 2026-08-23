//! Metamorphic ESIL lifting and live radare2 differential contracts.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::fmt::Write as _;
use std::process::Command;

use r2smt_common::{Arch, SolveOptions};
use r2smt_difflift::{DiffVerdict, build_equivalence_query, classify_equivalence};
use r2smt_smt::solve_branch;

fn compare_esil(a: &str, b: &str) -> DiffVerdict {
    let a = r2smt_esil::lift_esil(a, Arch::X86_64).expect("first ESIL form must lift");
    let b = r2smt_esil::lift_esil(b, Arch::X86_64).expect("second ESIL form must lift");
    let query = build_equivalence_query(&a.statements, &b.statements, Arch::X86_64)
        .expect("forms must define a shared output");
    classify_equivalence(solve_branch(
        &query,
        SolveOptions {
            timeout_ms: 10_000,
            ..SolveOptions::default()
        },
    ))
}

#[test]
fn algebraic_esil_metamorphisms_are_equivalent() {
    assert_eq!(
        compare_esil("rax,rax,=", "0,rax,+,rax,="),
        DiffVerdict::Agree
    );
    assert_eq!(
        compare_esil("rbx,rax,+,rcx,=", "rax,rbx,+,rcx,="),
        DiffVerdict::Agree
    );
}

#[test]
fn lifted_addition_matches_radare2_esil_on_generated_register_states() {
    if Command::new("radare2").arg("-qv").output().is_err() {
        return;
    }

    let mut command = String::from("aei;");
    let mut expected = Vec::new();
    let mut state = 0x5eed_u64;
    for _ in 0..64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        write!(command, "aer rax=0x{state:x};ae 1,rax,+,rax,=;aer rax;").expect("write command");
        expected.push(state.wrapping_add(1));
    }
    command.push('q');

    let output = Command::new("radare2")
        .args([
            "-2q",
            "-N",
            "-a",
            "x86",
            "-b",
            "64",
            "-c",
            &command,
            "malloc://64",
        ])
        .output()
        .expect("run radare2 ESIL");
    assert!(output.status.success());
    let actual: Vec<u64> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().strip_prefix("0x"))
        .map(|hex| u64::from_str_radix(hex, 16).expect("hex register value"))
        .collect();
    assert_eq!(actual, expected);

    assert_eq!(
        compare_esil("1,rax,+,rax,=", "rax,1,+,rax,="),
        DiffVerdict::Agree
    );
}
