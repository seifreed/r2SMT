//! What `ands` / `tst` leave in `C` on `AArch64`, checked through the
//! branch that reads it.
//!
//! A64 logical flag-setting *clears* the architectural `C`. The pipeline
//! stores the carry in x86 borrow polarity — the inverse of ARM's `C` —
//! so "clears C" is the stored bit written **one**, not zero. `V` is
//! stored raw and is written zero, so the two constants deliberately
//! disagree.
//!
//! Getting it backwards is not imprecision: `b.hs` after an `ands`
//! resolves to the wrong arm, and the branch is reported constant in the
//! direction the machine does not take. Which is why these assert on the
//! solved predicate and never on a flag's value — the same reason
//! `aarch32_carry_convention_contracts.rs` gives.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceStatus, lift_branch_condition,
    lift_per_mnemonic,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;

fn branch(condition: BranchCondition) -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "b.cond".to_string(),
        condition,
        formula: "b.cond".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn insn(mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands
            .iter()
            .map(|raw| Operand {
                raw: (*raw).into(),
                kind: OperandKind::Register,
            })
            .collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Lift one flag-setting logical instruction and solve the branch that
/// follows it, with both sources left free — the answer must not depend
/// on them.
fn solve_after(mnemonic: &str, operands: &[&str], condition: BranchCondition) -> SmtResult {
    let statements: Vec<IrStmt> = lift_per_mnemonic(&insn(mnemonic, operands), Arch::Aarch64);
    assert!(
        statements
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} declined: {statements:?}"
    );
    let candidate = branch(condition);
    let slice = LiftedSlice {
        condition: lift_branch_condition(&candidate, Arch::Aarch64),
        branch: candidate,
        statements,
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Aarch64,
    };
    solve_branch(
        &ssa_convert(&slice),
        SolveOptions {
            timeout_ms: TEST_SOLVE_TIMEOUT_MS,
            ..SolveOptions::default()
        },
    )
}

#[test]
fn a_branch_on_carry_set_after_ands_is_never_taken() {
    // `ands` clears ARM's C, so `b.hs` (C == 1) cannot hold. Reading the
    // stored bit as if it were ARM's C answers `AlwaysTrue` here — the
    // opposite arm, reported with the same confidence.
    assert_eq!(
        solve_after("ands", &["x0", "x1", "x2"], BranchCondition::AboveOrEqual),
        SmtResult::AlwaysFalse,
    );
}

#[test]
fn a_branch_on_carry_clear_after_ands_is_always_taken() {
    assert_eq!(
        solve_after("ands", &["x0", "x1", "x2"], BranchCondition::Below),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_branch_on_carry_set_after_tst_is_never_taken() {
    // `tst` is `ands xzr, …` and writes the flags through its own
    // handler, so it needs its own contract or the two can drift.
    assert_eq!(
        solve_after("tst", &["x1", "x2"], BranchCondition::AboveOrEqual),
        SmtResult::AlwaysFalse,
    );
}

#[test]
fn a_branch_on_overflow_after_ands_is_never_taken() {
    // `V` is stored raw, so "clears V" really is a zero here. Inverting
    // it along with the carry would flip this one instead.
    assert_eq!(
        solve_after("ands", &["x0", "x1", "x2"], BranchCondition::Overflow),
        SmtResult::AlwaysFalse,
    );
}
