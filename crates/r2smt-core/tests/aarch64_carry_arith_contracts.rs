//! `adc` / `sbc` / `ngc` on `AArch64`, checked through a block that
//! produces the carry rather than through the flag's value.
//!
//! `CF` holds ARM's `C` **inverted** (x86 borrow polarity). `adc` adds
//! ARM's `C` while `sbc` subtracts its complement — the borrow — so the
//! two want opposite sides of that inversion and the stored bit is
//! already the right one for `sbc`. Reading it raw for `adc` is a
//! definite wrong value, and the `AArch32` side shipped exactly that bug
//! once, invisibly, because the contracts of the day pinned a flag's
//! value instead of the arithmetic that consumes it.
//!
//! So every case here runs a real `cmp` first and asserts the *result*
//! of the carry-consuming instruction. Prevalence of this family was
//! measured at 0 in the sampled corpus, so these contracts are the only
//! evidence it will ever have.
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
const X: u16 = 64;

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

fn insn(address: u64, mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(address),
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

/// Run a straight-line block with the named registers bound and ask
/// whether `observed` is necessarily `expected`.
fn solve_block(
    block: &[(&str, &[&str])],
    bindings: &[(&str, u128)],
    observed: &str,
    expected: u128,
) -> SmtResult {
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value)| IrStmt::Assign {
            dst: Var::new(*name, X),
            src: Expr::konst(*value, X),
        })
        .collect();
    for (index, (mnemonic, operands)) in block.iter().enumerate() {
        let address = 0x1000 + 4 * index as u64;
        let lifted = lift_per_mnemonic(&insn(address, mnemonic, operands), Arch::Aarch64);
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
        condition: Expr::eq(Expr::var(observed, X), Expr::konst(expected, X)),
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
fn adc_adds_the_carry_a_compare_left_set() {
    // `cmp x1, x2` on `1 - 0` does not borrow, so ARM sets C, and
    // `adc x0, x3, x4` on two zeroes must give 1. Reading the stored bit
    // as if it were ARM's C answers 0 — a wrong value no single-
    // instruction contract can see.
    assert_eq!(
        solve_block(
            &[("cmp", &["x1", "x2"]), ("adc", &["x0", "x3", "x4"])],
            &[("x1", 1), ("x2", 0), ("x3", 0), ("x4", 0)],
            "x0",
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn adc_adds_nothing_when_the_compare_borrowed() {
    // The mirror: `0 - 1` borrows, ARM clears C, so `adc` of two zeroes
    // is 0. Together with the case above this pins the direction rather
    // than just the presence of a carry term.
    assert_eq!(
        solve_block(
            &[("cmp", &["x1", "x2"]), ("adc", &["x0", "x3", "x4"])],
            &[("x1", 0), ("x2", 1), ("x3", 0), ("x4", 0)],
            "x0",
            0,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn sbc_subtracts_the_borrow_a_compare_left_set() {
    // `sbc` is `Rn - Rm - NOT C`. The compare borrows, so `NOT C` is 1
    // and `5 - 3 - 1` is 1.
    assert_eq!(
        solve_block(
            &[("cmp", &["x1", "x2"]), ("sbc", &["x0", "x3", "x4"])],
            &[("x1", 0), ("x2", 1), ("x3", 5), ("x4", 3)],
            "x0",
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn sbc_subtracts_nothing_extra_when_the_compare_did_not_borrow() {
    // No borrow, so `NOT C` is 0 and `5 - 3` is 2. This is the case that
    // separates `sbc` from a plain `sub`.
    assert_eq!(
        solve_block(
            &[("cmp", &["x1", "x2"]), ("sbc", &["x0", "x3", "x4"])],
            &[("x1", 1), ("x2", 0), ("x3", 5), ("x4", 3)],
            "x0",
            2,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn ngc_is_sbc_against_the_zero_register() {
    // `ngc x0, x3` is `sbc x0, xzr, x3`: with a borrow outstanding,
    // `0 - 3 - 1` is -4.
    assert_eq!(
        solve_block(
            &[("cmp", &["x1", "x2"]), ("ngc", &["x0", "x3"])],
            &[("x1", 0), ("x2", 1), ("x3", 3)],
            "x0",
            u128::from(4_u64.wrapping_neg()),
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn adcs_sets_the_zero_flag_from_the_carried_sum() {
    // The flag-setting form still computes the same value, so a sum that
    // only reaches zero *because* of the carry must set Z. Here
    // `0xffff_ffff_ffff_ffff + 0 + 1` wraps to zero.
    assert_eq!(
        solve_block(
            &[("cmp", &["x1", "x2"]), ("adcs", &["x0", "x3", "x4"])],
            &[
                ("x1", 1),
                ("x2", 0),
                ("x3", u128::from(u64::MAX)),
                ("x4", 0)
            ],
            "x0",
            0,
        ),
        SmtResult::AlwaysTrue,
    );
}
