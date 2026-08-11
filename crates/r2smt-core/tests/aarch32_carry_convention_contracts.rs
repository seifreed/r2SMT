//! What `CF` holds on the ARM paths, and the fact that every producer
//! and every consumer must agree about it.
//!
//! This pipeline stores the carry in **x86 borrow polarity**: `cmp`
//! emits `CF = Ult(lhs, rhs)`, which is set when the subtraction
//! borrowed — the *inverse* of ARM's `C`. That is a deliberate choice
//! (`condition.rs` documents it) and it is what lets one
//! `lift_branch_condition` serve both ISAs: ARM `cs` / `hs` maps to
//! `AboveOrEqual`, which lowers to `CF == 0`.
//!
//! The choice only works if everything obeys it. A producer that writes
//! ARM's `C` raw, or a consumer that reads `CF` as if it were ARM's `C`,
//! is not imprecise — it computes a definite wrong answer, and the
//! failure is invisible to any contract that pins a flag's *value*
//! rather than the branch it decides.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceStatus, lift_per_mnemonic,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
const WORD: u16 = 32;

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "carrytest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "carrytest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with('r') {
            OperandKind::Register
        } else {
            OperandKind::Immediate
        },
    }
}

fn insn(address: u64, mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(address),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|raw| operand(raw)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Lift a straight-line block through the per-mnemonic path with the
/// named registers bound, and report whether `observed` is necessarily
/// `expected`.
///
/// A block rather than one instruction, because the whole question is
/// whether one instruction's carry means the same thing to the next.
fn solve_block(
    block: &[(&str, &[&str])],
    bindings: &[(&str, u128, u16)],
    observed: &str,
    expected: u128,
) -> SmtResult {
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    for (index, (mnemonic, operands)) in block.iter().enumerate() {
        let address = 0x1000 + 4 * index as u64;
        let lifted = lift_per_mnemonic(&insn(address, mnemonic, operands), Arch::Arm);
        assert!(
            lifted
                .iter()
                .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
            "{mnemonic} {operands:?} declined: {lifted:?}"
        );
        statements.extend(lifted);
    }
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(Expr::var(observed, WORD), Expr::konst(expected, WORD)),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Arm,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

#[test]
fn a_carry_out_of_a_compare_is_the_carry_an_add_with_carry_reads() {
    // `cmp r1, r2` with `1 - 0`: no borrow, so ARM sets C. `adc r0, r3,
    // r4` on two zeroes must therefore give 1.
    //
    // This is the whole convention in one block. The compare writes the
    // *inverse* of ARM's C, so an `adc` that reads `CF` as if it were
    // ARM's C answers 0 here — a definite wrong value, and one that no
    // single-instruction contract can see.
    assert_eq!(
        solve_block(
            &[("cmp", &["r1", "r2"]), ("adc", &["r0", "r3", "r4"])],
            &[
                ("r1", 1, WORD),
                ("r2", 0, WORD),
                ("r3", 0, WORD),
                ("r4", 0, WORD),
            ],
            "r0",
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_borrow_out_of_a_compare_is_the_borrow_a_subtract_with_carry_reads() {
    // The mirror, on the subtracting side. `cmp r1, r2` with `0 - 1`
    // borrows, so ARM clears C, and `sbc` subtracts `NOT C` = 1:
    // `5 - 3 - 1` is 1.
    assert_eq!(
        solve_block(
            &[("cmp", &["r1", "r2"]), ("sbc", &["r0", "r3", "r4"])],
            &[
                ("r1", 0, WORD),
                ("r2", 1, WORD),
                ("r3", 5, WORD),
                ("r4", 3, WORD),
            ],
            "r0",
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_carry_out_of_a_shift_is_the_carry_an_add_with_carry_reads() {
    // Same question with the *other* producer: `lsls` shifts a one out
    // of the top, so ARM sets C and `adc` of two zeroes gives 1. The
    // shift family has to store the same inverse the compare does.
    assert_eq!(
        solve_block(
            &[("lsls", &["r1", "r1", "1"]), ("adc", &["r0", "r3", "r4"])],
            &[("r1", 0x8000_0000, WORD), ("r3", 0, WORD), ("r4", 0, WORD),],
            "r0",
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn the_carry_a_compare_leaves_is_the_bit_rrx_rotates_in() {
    // And the third consumer. `cmp r1, r2` on `1 - 0` sets ARM's C, so
    // `rrx r0, r3` brings a one into the top: `2` becomes
    // `0x8000_0001`.
    assert_eq!(
        solve_block(
            &[("cmp", &["r1", "r2"]), ("rrx", &["r0", "r3"])],
            &[("r1", 1, WORD), ("r2", 0, WORD), ("r3", 2, WORD)],
            "r0",
            0x8000_0001,
        ),
        SmtResult::AlwaysTrue,
    );
}
