//! `AArch32` shift and rotate contracts.
//!
//! The family is one handler because three of its properties are not the
//! arithmetic family's, and each of them fails as a wrong value or a
//! fabricated flag rather than as a decline:
//!
//! 1. `rrx` is a **33-bit** rotate through the carry, not a rotate of
//!    the register. `Ror(x, 1)` brings the register's own low bit back
//!    round instead of the carry — a wrong result, and the pair of
//!    contracts below binds `CF` both ways to separate them.
//! 2. **C is the last bit shifted out.** The shared arithmetic helper
//!    cannot say that and left it free, so `lsls` + `bcs` — the ARM way
//!    of testing a bit — could not resolve.
//! 3. **A register amount is `Rs[7:0]`.** Shifting by `0x100` moves
//!    nothing on the machine and everything in a bit-vector solver,
//!    where the shift saturates the register to zero.
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
        mnemonic: "shifttest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "shifttest".to_string(),
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

/// Lift `mnemonic operands` with every named register and `CF` bound to
/// a constant, and report whether `observed` is necessarily `expected`.
fn solve_shift(
    mnemonic: &str,
    operands: &[&str],
    bindings: &[(&str, u128, u16)],
    observed: (&str, u16),
    expected: u128,
) -> SmtResult {
    let insn = Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|raw| operand(raw)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    };
    let lifted = lift_per_mnemonic(&insn, Arch::Arm);
    assert!(
        lifted
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} declined: {lifted:?}"
    );
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    statements.extend(lifted);
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::var(observed.0, observed.1),
            Expr::konst(expected, observed.1),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Arm,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

#[test]
fn rrx_brings_a_clear_carry_into_the_top_bit() {
    // 2 >> 1 is 1, and with C clear nothing enters above it.
    assert_eq!(
        solve_shift(
            "rrx",
            &["r0", "r1"],
            &[("r1", 2, WORD), ("CF", 0, 1)],
            ("r0", WORD),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn rrx_brings_a_set_carry_into_the_top_bit() {
    // The teeth of the family: the same source with C set gives
    // `0x8000_0001`. A lowering as `Ror(x, 1)` answers `1` here, since
    // the register's own low bit would come back round instead of the
    // carry.
    assert_eq!(
        solve_shift(
            "rrx",
            &["r0", "r1"],
            &[("r1", 2, WORD), ("CF", 1, 1)],
            ("r0", WORD),
            0x8000_0001,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn rrxs_puts_the_bit_that_fell_off_into_the_carry() {
    // The bit `rrx` discards is the one `rrxs` keeps. Bound with C set
    // so the answer cannot be the incoming carry echoed back.
    assert_eq!(
        solve_shift(
            "rrxs",
            &["r0", "r1"],
            &[("r1", 2, WORD), ("CF", 1, 1)],
            ("CF", 1),
            0,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn rrx_without_the_s_suffix_leaves_the_carry_alone() {
    // A shift that does not set flags must not touch C, and the guard
    // matters here more than elsewhere: `rrx` *reads* C, so a handler
    // that wrote it back would look plausible.
    assert_eq!(
        solve_shift(
            "rrx",
            &["r0", "r1"],
            &[("r1", 3, WORD), ("CF", 1, 1)],
            ("CF", 1),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn ror_rotates_rather_than_shifts() {
    // `0x0000_00ff` rotated right by 4 keeps every bit: the low nibble
    // reappears at the top. A logical shift would lose it.
    assert_eq!(
        solve_shift(
            "ror",
            &["r0", "r1", "4"],
            &[("r1", 0xff, WORD)],
            ("r0", WORD),
            0xf000_000f,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn lsls_carries_out_the_bit_that_left_the_top() {
    // The precision this family gained: C is the last bit shifted out,
    // where the shared arithmetic helper left it free. `0x8000_0000`
    // shifted left by one sets C and clears the register.
    assert_eq!(
        solve_shift(
            "lsls",
            &["r0", "r1", "1"],
            &[("r1", 0x8000_0000, WORD)],
            ("CF", 1),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn lsrs_carries_out_the_bit_that_left_the_bottom() {
    // The mirror, and the pair is what pins *which end* the carry comes
    // from: the same source shifted right by one carries out zero.
    assert_eq!(
        solve_shift(
            "lsrs",
            &["r0", "r1", "1"],
            &[("r1", 0x8000_0001, WORD)],
            ("CF", 1),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn asrs_by_the_full_width_carries_out_the_sign() {
    // `asr #32` is spelled with an amount of 32 and takes `src[31]`,
    // which is the edge the `n - 1` rule has to reach without running
    // off the register.
    assert_eq!(
        solve_shift(
            "asrs",
            &["r0", "r1", "32"],
            &[("r1", 0x8000_0000, WORD)],
            ("CF", 1),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_shift_by_a_register_amount_uses_only_its_low_byte() {
    // `Rs[7:0]`, and the byte is load-bearing: shifting by `0x100` moves
    // nothing on the machine, while a bit-vector shift by 256 clears the
    // register. This is a wrong value rather than a decline, which is
    // why it is bound rather than asserted on the IR shape.
    assert_eq!(
        solve_shift(
            "lsl",
            &["r0", "r1", "r2"],
            &[("r1", 0x1234, WORD), ("r2", 0x100, WORD)],
            ("r0", WORD),
            0x1234,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_shift_by_a_register_amount_still_shifts_by_the_low_byte() {
    // The other half of the mask: the low byte is used, not ignored.
    assert_eq!(
        solve_shift(
            "lsl",
            &["r0", "r1", "r2"],
            &[("r1", 0x1234, WORD), ("r2", 0x104, WORD)],
            ("r0", WORD),
            0x1_2340,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_shift_by_zero_leaves_the_carry_it_found() {
    // The architecture leaves C alone for a zero amount, so the handler
    // emits nothing for it and the carry the slice already computed
    // survives. Assigning a free value here would lose it.
    assert_eq!(
        solve_shift(
            "lsls",
            &["r0", "r1", "0"],
            &[("r1", 1, WORD), ("CF", 1, 1)],
            ("CF", 1),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}
