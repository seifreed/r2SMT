//! Solver-backed contracts for the `AArch32` / Thumb control-flow forms
//! that resolve a branch predicate end to end: slicer classification,
//! live-set seeding, lifting, SSA, and solve. The corpus holds no such
//! sample, so these synthetic fixtures are the coverage.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn op(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.to_string(),
        kind,
    }
}

fn insn(addr: u64, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: Address(addr),
        size: 4,
        bytes: Vec::new(),
        mnemonic: mnemonic.to_string(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: true,
    }
}

/// A single `AArch32` block: `mov r0, #imm` then `compare-and-branch r0`.
fn compare_and_branch_program(imm: &str, branch_mnemonic: &str) -> Program {
    Program {
        arch: Arch::Arm,
        bits: 32,
        entry: Some(Address(0x1000)),
        functions: vec![Function {
            address: Address(0x1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x1000),
                instructions: vec![
                    insn(
                        0x1000,
                        "mov",
                        vec![
                            op("r0", OperandKind::Register),
                            op(imm, OperandKind::Immediate),
                        ],
                    ),
                    insn(
                        0x1004,
                        branch_mnemonic,
                        vec![
                            op("r0", OperandKind::Register),
                            op("0x1080", OperandKind::Immediate),
                        ],
                    ),
                ],
                successors: vec![],
            }],
            is_thumb: true,
        }],
    }
}

fn solve_first_branch(program: &Program) -> SmtResult {
    let candidates = collect_branches(program);
    let candidate = candidates
        .first()
        .expect("one compare-and-branch candidate");
    let function = &program.functions[0];
    let slice = slice_branch(candidate, function, &SliceLimits::default(), Arch::Arm);
    let lifted = lift_slice(&slice, Arch::Arm);
    solve_branch(&ssa_convert(&lifted), solve_opts())
}

#[test]
fn aarch32_cbz_on_a_zeroed_register_is_always_taken() {
    // `mov r0, #0; cbz r0, tgt` — cbz branches when r0 == 0, which is
    // always. The opaque predicate resolves to AlwaysTrue only if the
    // slicer seeds the live set with r0 (not a flag) and the lifter
    // derives the `r0 == 0` predicate.
    let program = compare_and_branch_program("0", "cbz");
    assert_eq!(solve_first_branch(&program), SmtResult::AlwaysTrue);
}

#[test]
fn aarch32_cbnz_on_a_zeroed_register_is_never_taken() {
    // `mov r0, #0; cbnz r0, tgt` — cbnz branches when r0 != 0, never.
    let program = compare_and_branch_program("0", "cbnz");
    assert_eq!(solve_first_branch(&program), SmtResult::AlwaysFalse);
}

#[test]
fn aarch32_cbz_on_a_nonzero_register_is_never_taken() {
    // `mov r0, #5; cbz r0, tgt` — r0 == 0 is false, so never taken.
    let program = compare_and_branch_program("5", "cbz");
    assert_eq!(solve_first_branch(&program), SmtResult::AlwaysFalse);
}
