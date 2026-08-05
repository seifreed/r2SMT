//! What an `AArch32` logical flag-setting instruction does to C and V.
//!
//! `ANDS` / `ORRS` / `EORS` / `BICS` / `TST` / `TEQ` take C from the
//! **shifter carry-out of Operand2** and do not write V at all (ARM DDI
//! 0487, A32). With no shift specifier the shifter carry-out *is* the
//! carry-in, so C is unchanged. `AArch64` `ANDS` is the one that clears
//! both, and a shared lowering that clears them on A32 fabricates two
//! flags rather than losing precision — the failure mode a downstream
//! confidence downgrade cannot catch.
//!
//! Same for `MULS` from ARM v6 on: C and V unchanged.
//!
//! Every assertion here is on a *consumer* of the flag rather than on the
//! flag's rendered value, because that is the only way to see the carry
//! convention: `CF` is stored in x86 borrow polarity, the inverse of
//! ARM's `C` (see `aarch32_carry_convention_contracts.rs`).
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

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with('r') || raw.starts_with('x') || raw.starts_with('w') {
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

/// Lift a straight-line block with the named registers bound, and report
/// whether `observed` is necessarily `expected`.
fn solve_block(
    arch: Arch,
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
        let lifted = lift_per_mnemonic(&insn(address, mnemonic, operands), arch);
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
        arch,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

/// A block whose carry is *cleared* by `cmp 0 - 1` (ARM `C` = 0), then
/// read by `sbc r0, r3, r4`, which subtracts `NOT C` = 1, so that
/// `5 - 3 - 1` yields one. That is the answer exactly when the
/// instruction in between left the carry alone.
///
/// A cleared carry rather than a set one, and that choice is the whole
/// point. `CF` is stored as ARM's `C` inverted, so the constant these
/// instructions used to write — `CF = 0` — *is* ARM `C = 1`, and a block
/// that starts from a set carry would survive it by coincidence. Only the
/// cleared side tells "unchanged" apart from "fabricated", and it catches
/// a free value in the same assertion.
fn carry_survives(arch: Arch, between: &[(&str, &[&str])]) -> SmtResult {
    let mut block: Vec<(&str, &[&str])> = vec![("cmp", &["r1", "r2"])];
    block.extend_from_slice(between);
    block.push(("sbc", &["r0", "r3", "r4"]));
    solve_block(
        arch,
        &block,
        &[
            ("r1", 0, WORD),
            ("r2", 1, WORD),
            ("r3", 5, WORD),
            ("r4", 3, WORD),
            ("r5", 0xffff_ffff, WORD),
            ("r6", 0xffff_ffff, WORD),
        ],
        "r0",
        1,
    )
}

#[test]
fn an_unshifted_ands_leaves_a_cleared_carry_alone() {
    assert_eq!(
        carry_survives(Arch::Arm, &[("ands", &["r7", "r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_unshifted_ands_leaves_a_set_carry_alone() {
    // The other direction, which the shared helper cannot express: `cmp
    // 1 - 0` does not borrow, so ARM sets C and `adc` of two zeroes gives
    // 1. Both directions are needed because each is blind to one way of
    // getting C wrong — this one to a fabricated `CF = 0`, the other to a
    // fabricated `CF = 1`.
    assert_eq!(
        solve_block(
            Arch::Arm,
            &[
                ("cmp", &["r1", "r2"]),
                ("ands", &["r7", "r5", "r6"]),
                ("adc", &["r0", "r3", "r4"]),
            ],
            &[
                ("r1", 1, WORD),
                ("r2", 0, WORD),
                ("r3", 0, WORD),
                ("r4", 0, WORD),
                ("r5", 0xffff_ffff, WORD),
                ("r6", 0xffff_ffff, WORD),
            ],
            "r0",
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_unshifted_orrs_leaves_the_carry_alone() {
    // A32 has `ORRS`; A64 does not, so the shared logical arm that wrote
    // a constant here was reachable only from `AArch32` and was
    // unconditionally wrong.
    assert_eq!(
        carry_survives(Arch::Arm, &[("orrs", &["r7", "r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_unshifted_eors_leaves_the_carry_alone() {
    assert_eq!(
        carry_survives(Arch::Arm, &[("eors", &["r7", "r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_unshifted_bics_leaves_the_carry_alone() {
    assert_eq!(
        carry_survives(Arch::Arm, &[("bics", &["r7", "r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_unshifted_tst_leaves_the_carry_alone() {
    assert_eq!(
        carry_survives(Arch::Arm, &[("tst", &["r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_unshifted_teq_leaves_the_carry_alone() {
    // The comment this replaced claimed `teq` clears C and V "as
    // architectural behaviour". It does neither.
    assert_eq!(
        carry_survives(Arch::Arm, &[("teq", &["r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_armv6_muls_leaves_the_carry_alone() {
    // `muls` fell to the catch-all arm and wrote a free value into C,
    // which destroys a carry the slice already holds — precision lost
    // rather than a flag fabricated, but lost for no architectural
    // reason at all.
    assert_eq!(
        carry_survives(Arch::Arm, &[("muls", &["r7", "r5", "r6"])]),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_shifted_ands_does_not_claim_the_carry_is_unchanged() {
    // `ands r7, r5, r6, lsl 1` takes C from the shift, so the carry-in
    // does *not* survive. This lowering does not compute the carry-out
    // bit and writes a free value instead: the verdict must therefore be
    // `BothPossible`, never `AlwaysTrue`. Asserting the *absence* of a
    // definite answer is what stops the unshifted rule being applied one
    // operand too far.
    assert_eq!(
        carry_survives(Arch::Arm, &[("ands", &["r7", "r5", "r6", "lsl 1"])]),
        SmtResult::BothPossible,
    );
}

/// Which flags a single instruction assigns, by name.
fn flags_written(arch: Arch, mnemonic: &str, operands: &[&str]) -> Vec<String> {
    lift_per_mnemonic(&insn(0x1000, mnemonic, operands), arch)
        .iter()
        .filter_map(|stmt| match stmt {
            IrStmt::Assign { dst, .. } if dst.bits == 1 => Some(dst.name.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn an_aarch32_logical_form_never_writes_the_overflow_flag() {
    // V is the half no consumer in this pipeline can observe cheaply —
    // `cmp` leaves OF free on `AArch32`, so there is no producer to build
    // a solver block against. Stating the rule on the emitted statements
    // is what keeps it checkable at all, and the failure it guards is a
    // *fabricated* V rather than a lost one.
    for (mnemonic, operands) in [
        ("ands", &["r7", "r5", "r6"][..]),
        ("orrs", &["r7", "r5", "r6"][..]),
        ("eors", &["r7", "r5", "r6"][..]),
        ("bics", &["r7", "r5", "r6"][..]),
        ("muls", &["r7", "r5", "r6"][..]),
        ("tst", &["r5", "r6"][..]),
        ("teq", &["r5", "r6"][..]),
    ] {
        let written = flags_written(Arch::Arm, mnemonic, operands);
        assert!(
            !written.iter().any(|flag| flag == "OF"),
            "{mnemonic} wrote V on AArch32: {written:?}"
        );
    }
}

#[test]
fn an_aarch64_ands_still_writes_both_carry_and_overflow() {
    // The other ISA must not be dragged along by the A32 rule: A64 `ANDS`
    // *does* define C and V, so dropping the assignments here would turn
    // a defined flag into a stale one.
    let written = flags_written(Arch::Aarch64, "ands", &["x7", "x5", "x6"]);
    assert!(
        written.iter().any(|flag| flag == "CF") && written.iter().any(|flag| flag == "OF"),
        "AArch64 ands stopped defining C/V: {written:?}"
    );
}
