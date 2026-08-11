//! Thumb-2 `IT` block contract.
//!
//! radare2 6.1.8 loses ITSTATE part-way through an `IT` block and
//! prints the *last* covered instruction with no condition suffix
//! (`itt` suffixes 1 of 2, `ittt` 2 of 3). That is not a decline the
//! slicer can absorb: the instruction lifts as an unconditional
//! assignment, and because the backward walk stops as soon as its live
//! set is satisfied, the slice can reach `Complete` without ever
//! visiting the `it` that would have truncated it. The verdict is then
//! fabricated from a value the machine never computes.
//!
//! These contracts assert on the *verdict*, end to end from the `agfj`
//! JSON radare2 emits, because the claim being pinned is about what the
//! solver concludes — not about the shape of the IR.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_core::prepare_ssa;
use r2smt_ir::program::Function;
use r2smt_ir::stmt::IrStmt;
use r2smt_r2pipe::parse::{mark_thumb, parse_function_blocks};
use r2smt_slicer::{SliceLimits, SliceStatus, collect_function_branches};
use r2smt_smt::solve_branch;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;

/// `cmp r0, r0` is always equal, so the `eq` arm of the `ite` runs and
/// the `ne` arm does not: `r2` ends up 7, and `beq` after `cmp r2, #7`
/// is always taken.
///
/// The last instruction of the block — the `ne` arm — is the one
/// radare2 prints bare, exactly as `"mov r2, 9"` here. Lifted
/// unconditionally it overwrites `r2` with 9 and the branch becomes
/// always *false*, which is why this fixture separates a correct fold
/// from a missing one by a full verdict flip rather than a confidence
/// drop.
const ITE_AGFJ: &str = r#"[
    {
        "name": "fcn.thumb",
        "addr": 4096,
        "bits": 16,
        "blocks": [
            {
                "addr": 4096,
                "jump": 4200,
                "fail": 4112,
                "ops": [
                    { "addr": 4096, "size": 2, "bytes": "8042", "opcode": "cmp r0, r0" },
                    { "addr": 4098, "size": 2, "bytes": "0cbf", "opcode": "ite eq" },
                    { "addr": 4100, "size": 2, "bytes": "0722", "opcode": "moveq r2, 7" },
                    { "addr": 4102, "size": 2, "bytes": "0922", "opcode": "mov r2, 9" },
                    { "addr": 4104, "size": 2, "bytes": "072a", "opcode": "cmp r2, 7" },
                    { "addr": 4106, "size": 2, "bytes": "00d0", "opcode": "beq 0x1068" }
                ]
            }
        ]
    }
]"#;

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn function() -> Function {
    // Thumb arrives from `aflj` via `mark_thumb`, never from `agfj` —
    // which emits no width hint at all. This fixture used to carry a
    // fabricated `"bits": 16` and rely on the parser honouring it, so it
    // exercised a path production never takes.
    let mut function = parse_function_blocks(ITE_AGFJ).unwrap();
    mark_thumb(&mut function);
    function
}

/// Slice, lift, SSA-rename and solve the sole conditional branch.
fn solve_only_branch(function: &Function) -> (SmtResult, SliceStatus) {
    let branches = collect_function_branches(function, Arch::Arm);
    let candidate = branches
        .iter()
        .find(|b| b.address == Address::new(4106))
        .expect("the beq at 0x100a should be collected as a branch");
    let limits = SliceLimits {
        max_instructions: 32,
        ..SliceLimits::default()
    };
    let ssa = prepare_ssa(function, candidate, &limits, Arch::Arm);
    let status = ssa.status.clone();
    (solve_branch(&ssa, solve_opts()), status)
}

#[test]
fn test_it_block_last_instruction_is_predicated_not_unconditional() {
    // The headline claim: with the fold, the `ne` arm does not run and
    // the branch is provably taken. Without it, `mov r2, 9` executes
    // unconditionally and the same slice concludes `AlwaysFalse` — a
    // fabricated verdict, since the machine never takes that path.
    let (verdict, _) = solve_only_branch(&function());
    assert_eq!(verdict, SmtResult::AlwaysTrue);
}

#[test]
fn test_it_block_fold_leaves_the_slice_complete() {
    // Soundness alone could be bought by truncating on every `it`. It
    // is not: folding keeps the slice `Complete`, so Thumb-2 coverage
    // survives the fix.
    let (_, status) = solve_only_branch(&function());
    assert_eq!(status, SliceStatus::Complete);
}

#[test]
fn test_it_block_fold_rewrites_the_bare_mnemonic_radare2_emitted() {
    // Pins the mechanism the two verdict contracts above depend on,
    // so a regression names the cause rather than only the symptom.
    let function = function();
    let mnemonics: Vec<&str> = function.blocks[0]
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    assert_eq!(
        mnemonics,
        vec!["cmp", "nop", "moveq", "movne", "cmp", "beq"]
    );
}

/// The predicated-move idiom that fabricated a `dead_branch` on a real
/// ARM sample.
///
/// `cmp r3, r0` reads `r0` from before the slice; `mov r0, 0` then
/// overwrites it; `movcc r0, 1` keeps that 0 on its false path. If the
/// effect table does not report a predicated instruction as *reading*
/// its destination, the walk drops the `mov r0, 0`, and SSA binds the
/// else-arm to the very same free `r0` the compare used — asserting
/// `r0 == r0` across a write that changed it.
const PREDICATED_ELSE_ARM_AGFJ: &str = r#"[
    {
        "name": "fcn.arm",
        "addr": 4096,
        "blocks": [
            {
                "addr": 4096,
                "jump": 4200,
                "fail": 4116,
                "ops": [
                    { "addr": 4096, "size": 4, "bytes": "030050e1", "opcode": "cmp r3, r0" },
                    { "addr": 4100, "size": 4, "bytes": "0000a0e3", "opcode": "mov r0, 0" },
                    { "addr": 4104, "size": 4, "bytes": "0100a033", "opcode": "movcc r0, 1" },
                    { "addr": 4108, "size": 4, "bytes": "000050e3", "opcode": "cmp r0, 0" },
                    { "addr": 4112, "size": 4, "bytes": "0000000a", "opcode": "beq 0x1068" }
                ]
            }
        ]
    }
]"#;

#[test]
fn test_predicated_else_arm_resolves_to_the_stored_constant() {
    // The else-arm must be the 0 the `mov` stored, not the free `r0`
    // the compare reads. Binding it to that free input asserts
    // `r0 == r0` across a write that changed it — which is how a real
    // ARM sample produced a `High`-confidence `dead_branch` that the
    // machine does not have.
    let function = parse_function_blocks(PREDICATED_ELSE_ARM_AGFJ).unwrap();
    let branches = collect_function_branches(&function, Arch::Arm);
    let candidate = branches
        .iter()
        .find(|b| b.address == Address::new(4112))
        .expect("the beq should be collected");
    let ssa = prepare_ssa(
        &function,
        candidate,
        &SliceLimits {
            max_instructions: 32,
            ..SliceLimits::default()
        },
        Arch::Arm,
    );
    let ite = ssa
        .statements
        .iter()
        .find_map(|s| match s {
            IrStmt::Assign { src, .. } if format!("{src:?}").contains("Ite") => {
                Some(format!("{src:?}"))
            }
            _ => None,
        })
        .expect("the predicated move must lower to an Ite");
    let else_arm = ite
        .split("else_expr: ")
        .nth(1)
        .expect("an Ite has an else arm")
        .to_string();
    assert!(
        else_arm.starts_with("Const"),
        "else-arm must be the stored constant, not a register read: {else_arm}"
    );
}

#[test]
fn test_predicated_move_keeps_the_preceding_definition_in_the_slice() {
    // The mechanism itself: `mov r0, 0` must survive the walk. It is
    // dropped exactly when the predicated move fails to report `r0` as
    // a use.
    let function = parse_function_blocks(PREDICATED_ELSE_ARM_AGFJ).unwrap();
    let branches = collect_function_branches(&function, Arch::Arm);
    let candidate = branches
        .iter()
        .find(|b| b.address == Address::new(4112))
        .expect("the beq should be collected");
    let slice = r2smt_slicer::slice_branch(
        candidate,
        &function,
        &SliceLimits {
            max_instructions: 32,
            ..SliceLimits::default()
        },
        Arch::Arm,
    );
    assert!(
        slice
            .instructions
            .iter()
            .any(|i| i.address == Address::new(4100)),
        "the `mov r0, 0` at 0x1004 must be retained: {:?}",
        slice
            .instructions
            .iter()
            .map(|i| (i.address, &i.mnemonic))
            .collect::<Vec<_>>()
    );
}
