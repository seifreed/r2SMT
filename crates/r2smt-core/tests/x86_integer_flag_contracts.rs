//! Solver-backed contracts for the x86 **per-mnemonic** integer flags.
//!
//! These bind concrete operands and ask a solver whether a flag is
//! *necessarily* the SDM value, on the `lift_per_mnemonic` seam — the
//! path that runs when the ESIL rung is closed (`--no-esil-flags`) and
//! the path the differential harness always compares against radare2's
//! ESIL. `add`/`sub`/`cmp` used to leave `CF`/`OF`/`PF` as
//! `Expr::Unknown`; a structural test cannot see that a flag was left
//! free, so every value here would pass as `BothPossible` on the old
//! code and only `AlwaysTrue` proves the flag is pinned.
//!
//! Each expected value is the Intel SDM definition, not this
//! implementation's output; the differential harness independently
//! proves the direct forms used here agree with the ESIL machine's
//! masked `old`/`cur` forms.
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
const QWORD: u16 = 64;
const FLAG: u16 = 1;

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
        mnemonic: "flagtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "flagtest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn insn(mnemonic: &str, ops: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: ops
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

/// Lift `mnemonic ops` through the per-mnemonic x86 dispatch with the
/// named registers bound to constants, and report whether `flag` is
/// necessarily `expected`.
fn solve_flag(
    mnemonic: &str,
    ops: &[&str],
    bindings: &[(&str, u128)],
    flag: &str,
    expected: u128,
) -> SmtResult {
    let lifted = lift_per_mnemonic(&insn(mnemonic, ops), Arch::X86_64);
    assert!(!lifted.is_empty(), "`{mnemonic} {ops:?}` lifted to nothing");
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value)| IrStmt::Assign {
            dst: Var::new(*name, QWORD),
            src: Expr::konst(*value, QWORD),
        })
        .collect();
    statements.extend(lifted);
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(Expr::var(flag, FLAG), Expr::konst(expected, FLAG)),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::X86_64,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

// CF. `sub`/`cmp` borrow when the minuend is below the subtrahend;
// `add` carries out when the unsigned result wraps below the original.

#[test]
fn test_sub_borrows_when_the_minuend_is_below_the_subtrahend() {
    // sub eax, ebx with 0 - 1: borrow, CF = 1.
    assert_eq!(
        solve_flag("sub", &["eax", "ebx"], &[("rax", 0), ("rbx", 1)], "CF", 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_sub_does_not_borrow_when_the_minuend_is_at_or_above() {
    // sub eax, ebx with 5 - 5: no borrow, CF = 0.
    assert_eq!(
        solve_flag("sub", &["eax", "ebx"], &[("rax", 5), ("rbx", 5)], "CF", 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_add_carries_out_when_the_unsigned_result_wraps() {
    // add eax, ebx with 0xffffffff + 1 wraps to 0 at 32 bits: CF = 1.
    assert_eq!(
        solve_flag(
            "add",
            &["eax", "ebx"],
            &[("rax", 0xffff_ffff), ("rbx", 1)],
            "CF",
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_add_does_not_carry_when_the_sum_fits() {
    // add eax, ebx with 1 + 2 = 3: no wrap, CF = 0.
    assert_eq!(
        solve_flag("add", &["eax", "ebx"], &[("rax", 1), ("rbx", 2)], "CF", 0),
        SmtResult::AlwaysTrue
    );
}

// OF. Signed overflow — was `Expr::Unknown`, which is exactly what left
// `jge`/`jl` (whose predicate is `SF == OF`) unresolvable on this path.

#[test]
fn test_cmp_overflows_when_max_minus_negative_one() {
    // cmp eax, ebx with INT_MAX - (-1): the result wraps to INT_MIN, so
    // the signed subtraction overflowed. OF = 1.
    assert_eq!(
        solve_flag(
            "cmp",
            &["eax", "ebx"],
            &[("rax", 0x7fff_ffff), ("rbx", 0xffff_ffff)],
            "OF",
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_cmp_does_not_overflow_on_the_signed_branch_idiom() {
    // cmp eax, ebx with 0 - 4 = -4: in range, OF = 0. This is the
    // `xor esi,esi ; cmp esi,4 ; jge` idiom whose `SF != OF` makes the
    // branch dead — and it can only resolve once OF is modelled.
    assert_eq!(
        solve_flag("cmp", &["eax", "ebx"], &[("rax", 0), ("rbx", 4)], "OF", 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_add_overflows_when_two_positives_make_a_negative() {
    // add eax, ebx with INT_MAX + 1 = INT_MIN: OF = 1.
    assert_eq!(
        solve_flag(
            "add",
            &["eax", "ebx"],
            &[("rax", 0x7fff_ffff), ("rbx", 1)],
            "OF",
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_sub_overflows_when_min_minus_a_positive() {
    // sub eax, ebx with INT_MIN - 1 = INT_MAX: OF = 1.
    assert_eq!(
        solve_flag(
            "sub",
            &["eax", "ebx"],
            &[("rax", 0x8000_0000), ("rbx", 1)],
            "OF",
            1
        ),
        SmtResult::AlwaysTrue
    );
}

// PF. Even parity of the result's low byte — folded from the low 8 bits
// and nothing above them.

#[test]
fn test_add_sets_even_parity_on_the_low_byte() {
    // add eax, ebx with 1 + 2 = 3 (0b11, two set bits): PF = 1.
    assert_eq!(
        solve_flag("add", &["eax", "ebx"], &[("rax", 1), ("rbx", 2)], "PF", 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_add_clears_on_odd_parity() {
    // add eax, ebx with 3 + 4 = 7 (0b111, three set bits): PF = 0.
    assert_eq!(
        solve_flag("add", &["eax", "ebx"], &[("rax", 3), ("rbx", 4)], "PF", 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_cmp_folds_the_low_byte_to_even_parity() {
    // cmp eax, ebx with 7 - 4 = 3: PF = 1.
    assert_eq!(
        solve_flag("cmp", &["eax", "ebx"], &[("rax", 7), ("rbx", 4)], "PF", 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_parity_reads_only_the_low_byte() {
    // add eax, ebx with 0x100 + 2 = 0x102: the low byte is 0x02 (one set
    // bit, odd) so PF = 0, even though the whole value has two set bits.
    // A parity that folded more than the low byte would answer 1.
    assert_eq!(
        solve_flag(
            "add",
            &["eax", "ebx"],
            &[("rax", 0x100), ("rbx", 2)],
            "PF",
            0
        ),
        SmtResult::AlwaysTrue
    );
}
