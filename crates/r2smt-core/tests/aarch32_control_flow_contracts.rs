//! Solver-backed contracts for the `AArch32` / Thumb control-flow forms
//! that resolve a branch predicate end to end: slicer classification,
//! live-set seeding, lifting, SSA, and solve. The corpus holds no such
//! sample, so these synthetic fixtures are the coverage.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
use r2smt_slicer::{SliceLimits, SliceStatus, collect_branches, lift_slice, slice_branch};
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

/// `vcmp.f32 s0, s1; vmrs APSR_nzcv, FPSCR; b<cond> tgt`.
fn vcmp_vmrs_branch_program(branch_mnemonic: &str) -> Program {
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
                        "vcmp.f32",
                        vec![
                            op("s0", OperandKind::Register),
                            op("s1", OperandKind::Register),
                        ],
                    ),
                    insn(
                        0x1004,
                        "vmrs",
                        vec![
                            op("APSR_nzcv", OperandKind::Register),
                            op("FPSCR", OperandKind::Register),
                        ],
                    ),
                    insn(
                        0x1008,
                        branch_mnemonic,
                        vec![op("0x1080", OperandKind::Immediate)],
                    ),
                ],
                successors: vec![],
            }],
            is_thumb: true,
        }],
    }
}

fn slice_status_and_verdict(program: &Program) -> (SliceStatus, SmtResult) {
    let candidates = collect_branches(program);
    let candidate = candidates.first().expect("one branch candidate");
    let function = &program.functions[0];
    let slice = slice_branch(candidate, function, &SliceLimits::default(), Arch::Arm);
    let status = slice.status.clone();
    let lifted = lift_slice(&slice, Arch::Arm);
    (status, solve_branch(&ssa_convert(&lifted), solve_opts()))
}

#[test]
fn aarch32_vcmp_vmrs_branch_resolves_instead_of_truncating() {
    // The float-compare predicate must reach the branch through the
    // `vmrs` transfer: `vcmp.f32 s0, s1; vmrs APSR_nzcv, FPSCR; beq`.
    // With free operands the verdict is BothPossible — the point is that
    // it is NOT Unsound: an unmodelled `vmrs` truncates the slice at the
    // transfer, which reports Unsound without ever reaching the `vcmp`.
    let program = vcmp_vmrs_branch_program("beq");
    let (status, verdict) = slice_status_and_verdict(&program);
    assert_eq!(
        status,
        SliceStatus::Complete,
        "slice must not truncate at vmrs"
    );
    assert_eq!(
        verdict,
        SmtResult::BothPossible,
        "free-operand float compare is BothPossible, never Unsound",
    );
}

/// `mov r1,#5; str r1,[r0]; vldr s0,[r0]; vstr s0,[r0]; ldr r2,[r0];
/// cmp r2,#5; beq tgt` — the 32 stored bits roundtrip through a VFP
/// load and store unchanged, so r2 == 5 and the branch is always taken.
fn vldr_vstr_roundtrip_program() -> Program {
    let block = BasicBlock {
        address: Address(0x1000),
        instructions: vec![
            insn(
                0x1000,
                "mov",
                vec![
                    op("r1", OperandKind::Register),
                    op("5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x1004,
                "str",
                vec![
                    op("r1", OperandKind::Register),
                    op("[r0]", OperandKind::Memory),
                ],
            ),
            insn(
                0x1008,
                "vldr",
                vec![
                    op("s0", OperandKind::Register),
                    op("[r0]", OperandKind::Memory),
                ],
            ),
            insn(
                0x100c,
                "vstr",
                vec![
                    op("s0", OperandKind::Register),
                    op("[r0]", OperandKind::Memory),
                ],
            ),
            insn(
                0x1010,
                "ldr",
                vec![
                    op("r2", OperandKind::Register),
                    op("[r0]", OperandKind::Memory),
                ],
            ),
            insn(
                0x1014,
                "cmp",
                vec![
                    op("r2", OperandKind::Register),
                    op("5", OperandKind::Immediate),
                ],
            ),
            insn(0x1018, "beq", vec![op("0x1080", OperandKind::Immediate)]),
        ],
        successors: vec![],
    };
    Program {
        arch: Arch::Arm,
        bits: 32,
        entry: Some(Address(0x1000)),
        functions: vec![Function {
            address: Address(0x1000),
            name: Some("sym.main".into()),
            blocks: vec![block],
            is_thumb: true,
        }],
    }
}

#[test]
fn aarch32_vldr_vstr_roundtrip_preserves_the_stored_bytes() {
    let program = vldr_vstr_roundtrip_program();
    let candidates = collect_branches(&program);
    let candidate = candidates.first().expect("beq candidate");
    let limits = SliceLimits {
        allow_memory: true,
        ..SliceLimits::default()
    };
    let slice = slice_branch(candidate, &program.functions[0], &limits, Arch::Arm);
    let lifted = lift_slice(&slice, Arch::Arm);
    assert_eq!(
        solve_branch(&ssa_convert(&lifted), solve_opts()),
        SmtResult::AlwaysTrue,
        "the 32 stored bits must roundtrip through vldr/vstr unchanged",
    );
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
