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

/// ARM's carry, expressed in the polarity this pipeline stores.
///
/// `CF` here is the *inverse* of ARM's `C` — x86 borrow polarity, so the
/// one `lift_branch_condition` can serve both ISAs. Contracts say what
/// the architecture does and convert here, rather than repeating a
/// flipped literal that reads like a typo. See
/// `aarch32_carry_convention_contracts.rs`.
const fn stored(arm_carry: u128) -> u128 {
    arm_carry ^ 1
}

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with('r') {
            OperandKind::Register
        } else if raw.contains(char::is_whitespace) {
            // What the real parser gives a shift specifier: it splits on
            // top-level commas only, so `lsl 2` arrives whole and its
            // internal space makes `classify_operand` answer `Unknown`.
            OperandKind::Unknown
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

/// Lift `mnemonic operands` with `bindings` applied, then ask whether
/// the branch predicate `condition` necessarily holds — the same
/// lowering a real `b<cond>` goes through.
///
/// The value harness above cannot answer this. A flag's *value* and the
/// *branch it decides* are two questions, and this pipeline deliberately
/// stores `CF` in x86 polarity (`condition.rs` documents it: the lifter
/// inverts ARM's C so `lift_branch_condition` needs no per-arch
/// dispatch), so a contract that only pins the value passes while the
/// branch resolves backwards.
fn solve_predicate(
    mnemonic: &str,
    operands: &[&str],
    bindings: &[(&str, u128, u16)],
    condition: BranchCondition,
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
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    statements.extend(lift_per_mnemonic(&insn, Arch::Arm));
    let mut candidate = branch();
    candidate.condition = condition;
    let slice = LiftedSlice {
        branch: candidate.clone(),
        statements,
        condition: r2smt_slicer::lift_branch_condition(&candidate, Arch::Arm),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Arm,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

#[test]
fn a_carry_out_of_a_shift_decides_a_carry_set_branch() {
    // `lsls r0, r1, #1` on `0x8000_0000` shifts a one out of the top, so
    // ARM sets C and `bcs` is taken.
    //
    // Getting the *value* of `CF` right is not enough to get this right,
    // and that is the whole point of the test: this pipeline stores the
    // inverse of ARM's C, because every other flag producer here follows
    // x86 borrow polarity and `b.cs` lowers to `CF == 0`. A shifter
    // carry-out written raw makes the branch resolve backwards — a
    // fabricated verdict, not a lost one.
    assert_eq!(
        solve_predicate(
            "lsls",
            &["r0", "r1", "1"],
            &[("r1", 0x8000_0000, WORD)],
            BranchCondition::AboveOrEqual,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn no_carry_out_of_a_shift_decides_a_carry_clear_branch() {
    // The other side, so the pair cannot be satisfied by a constant:
    // shifting a zero out leaves ARM's C clear and `bcc` is taken.
    assert_eq!(
        solve_predicate(
            "lsls",
            &["r0", "r1", "1"],
            &[("r1", 0x4000_0000, WORD)],
            BranchCondition::Below,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn rrx_brings_a_clear_carry_into_the_top_bit() {
    // 2 >> 1 is 1, and with C clear nothing enters above it.
    assert_eq!(
        solve_shift(
            "rrx",
            &["r0", "r1"],
            &[("r1", 2, WORD), ("CF", stored(0), 1)],
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
            &[("r1", 2, WORD), ("CF", stored(1), 1)],
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
            &[("r1", 2, WORD), ("CF", stored(1), 1)],
            ("CF", 1),
            stored(0),
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
            &[("r1", 3, WORD), ("CF", stored(1), 1)],
            ("CF", 1),
            stored(1),
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
            stored(1),
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
            stored(1),
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
            stored(1),
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
            &[("r1", 1, WORD), ("CF", stored(1), 1)],
            ("CF", 1),
            stored(1),
        ),
        SmtResult::AlwaysTrue,
    );
}

// the specifier that follows a register operand

#[test]
fn an_arithmetic_operand_carries_its_shift() {
    // `eor r0, r1, r2, lsl 2` is **four** operands, one more than the
    // three-operand handler reads. Dropping the fourth computes
    // `r1 ^ r2` — a wrong value, not a decline, and the reason this
    // whole family needed auditing.
    assert_eq!(
        solve_shift(
            "eor",
            &["r0", "r1", "r2", "lsl 2"],
            &[("r1", 0, WORD), ("r2", 3, WORD)],
            ("r0", WORD),
            0xc,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_compare_carries_its_shift_into_the_flags() {
    // `cmp r0, r1, lsl 1` is three operands where the handler reads two,
    // so the shape hides the extra one. With `r0 == 4` and `r1 == 2` the
    // shifted compare is equal and `ZF` is set; ignoring the shift makes
    // it `4 - 2` and clears `ZF`.
    assert_eq!(
        solve_shift(
            "cmp",
            &["r0", "r1", "lsl 1"],
            &[("r0", 4, WORD), ("r1", 2, WORD)],
            ("ZF", 1),
            1,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_reverse_subtract_shifts_the_operand_it_subtracts_from() {
    // `rsb Rd, Rn, Op` is `Op - Rn`, and the specifier belongs to `Op`.
    // The handler used to swap the operands by cloning the instruction,
    // which puts `Op` where nothing folds its shift — so this asserts
    // the reversal and the shift at once: `(1 << 4) - 6` is 10.
    assert_eq!(
        solve_shift(
            "rsb",
            &["r0", "r1", "r2", "lsl 4"],
            &[("r1", 6, WORD), ("r2", 1, WORD)],
            ("r0", WORD),
            10,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn an_extend_rotates_before_it_takes_the_byte() {
    // `sxtb r0, r1, ror 8` rotates first and extends the byte the
    // rotation brought down, which is the entire purpose of the
    // optional rotate. `0x0000_ff00` rotated right by 8 is `0xff`, and
    // the signed byte extends to `0xffff_ffff`; ignoring the rotate
    // would extend the original low byte, which is zero.
    assert_eq!(
        solve_shift(
            "sxtb",
            &["r0", "r1", "ror 8"],
            &[("r1", 0x0000_ff00, WORD)],
            ("r0", WORD),
            0xffff_ffff,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_shifted_operand_can_take_its_amount_from_a_register() {
    // The register-amount spelling, which is also the one whose shift
    // register has to survive into the effect table's `uses`.
    assert_eq!(
        solve_shift(
            "eor",
            &["r0", "r1", "r2", "lsl r3"],
            &[("r1", 0, WORD), ("r2", 1, WORD), ("r3", 5, WORD)],
            ("r0", WORD),
            0x20,
        ),
        SmtResult::AlwaysTrue,
    );
}
