//! `AArch64` NEON contracts for the packed float-to-integer conversions
//! that name their rounding mode in the opcode.
//!
//! `fcvtz*` was the only spelling the lifter modelled, and it is the one
//! whose mode is easiest to assume: round toward zero. The other four —
//! `fcvta*` ties away, `fcvtn*` ties to even, `fcvtp*` up, `fcvtm*` down
//! — differ from it and from each other on inputs an ordinary program
//! produces, and every one of them fails as a *wrong value* rather than
//! as a decline. So the evidence is a solver agreeing the destination
//! equals a hand-computed result, on inputs chosen so each mode is
//! contrasted with the one it would plausibly be confused for.
//!
//! `±2.5` does most of that work alone: it separates ties-to-even from
//! ties-away *and* toward-zero from toward-negative. It does not
//! separate ties-to-even from toward-zero, which agree on it, so `±1.5`
//! appears for that pair.
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

const TEST_SOLVE_TIMEOUT_MS: u32 = 20_000;
const VECTOR_BITS: u16 = 128;
/// Binding for the destination parent, so that "the result is exactly
/// this" also answers whether a half-width arrangement zeroed the rest
/// of the register.
const PARENT_PRESET: u128 = u128::MAX;

// IEEE binary32 patterns.
const S_2_5: u128 = 0x4020_0000;
const S_NEG_2_5: u128 = 0xc020_0000;
const S_1_5: u128 = 0x3fc0_0000;
const S_NEG_1_5: u128 = 0xbfc0_0000;
const S_3_5: u128 = 0x4060_0000;

// Two's-complement 32-bit results.
const I_1: u128 = 0x0000_0001;
const I_2: u128 = 0x0000_0002;
const I_3: u128 = 0x0000_0003;
const I_4: u128 = 0x0000_0004;
const I_5: u128 = 0x0000_0005;
const I_NEG_1: u128 = 0xffff_ffff;
const I_NEG_2: u128 = 0xffff_fffe;
const I_NEG_3: u128 = 0xffff_fffd;

fn reg(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: OperandKind::Register,
    }
}

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "fcvttest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "fcvttest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn instruction(mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|o| reg(o)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Pack 32-bit lanes little-endian into one vector value.
fn packed(lanes: &[u128]) -> u128 {
    let mut value = 0u128;
    for (index, lane) in lanes.iter().enumerate() {
        value |= lane << (32 * index);
    }
    value
}

fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = instruction(mnemonic, operands);
    let lifted = lift_per_mnemonic(&insn, Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} declined: {lifted:?}"
    );
    let mut statements: Vec<IrStmt> = sources
        .iter()
        .map(|(name, value)| IrStmt::Assign {
            dst: Var::new(*name, VECTOR_BITS),
            src: Expr::konst(*value, VECTOR_BITS),
        })
        .collect();
    statements.extend(lifted);
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::Var(Var::new("v0", VECTOR_BITS)),
            Expr::konst(expected, VECTOR_BITS),
        ),
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

/// Convert `±2.5` and `±1.5` in one four-lane instruction, so a lowering
/// that rounded lane zero and copied the rest fails alongside one that
/// picked the wrong mode.
fn assert_converts(mnemonic: &str, expected: &[u128; 4]) {
    assert_eq!(
        solve_lowering(
            mnemonic,
            &["v0.4s", "v1.4s"],
            &[("v1", packed(&[S_2_5, S_NEG_2_5, S_1_5, S_NEG_1_5]))],
            packed(expected),
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} of 2.5 / -2.5 / 1.5 / -1.5 must give {expected:x?}"
    );
}

fn assert_declines(mnemonic: &str, operands: &[&str]) {
    let lifted = lift_per_mnemonic(&instruction(mnemonic, operands), Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} should decline, got {lifted:?}"
    );
}

#[test]
fn fcvtns_breaks_a_tie_to_even() {
    assert_converts("fcvtns", &[I_2, I_NEG_2, I_2, I_NEG_2]);
}

#[test]
fn fcvtas_breaks_the_same_tie_away_from_zero() {
    // Ties away is a magnitude rule, not a direction: both halves of the
    // 2.5 pair move outward, and 1.5 does too.
    assert_converts("fcvtas", &[I_3, I_NEG_3, I_2, I_NEG_2]);
}

#[test]
fn fcvtps_rounds_toward_positive_infinity() {
    assert_converts("fcvtps", &[I_3, I_NEG_2, I_2, I_NEG_1]);
}

#[test]
fn fcvtms_rounds_toward_negative_infinity() {
    // The mirror of `fcvtps` on the same four lanes, which is what makes
    // the pair impossible to satisfy with one mode.
    assert_converts("fcvtms", &[I_2, I_NEG_3, I_1, I_NEG_2]);
}

#[test]
fn fcvtzs_still_truncates_toward_zero() {
    // The mode that was already modelled, asserted here so widening the
    // family cannot quietly move it. Truncation agrees with ties-to-even
    // on ±2.5 and disagrees on ±1.5.
    assert_converts("fcvtzs", &[I_2, I_NEG_2, I_1, I_NEG_1]);
}

#[test]
fn fcvtnu_breaks_a_tie_to_even_on_the_unsigned_side() {
    // The unsigned half goes through the signed conversion node with one
    // extra bit of range, so the mode has to survive that detour too.
    assert_eq!(
        solve_lowering(
            "fcvtnu",
            &["v0.4s", "v1.4s"],
            &[("v1", packed(&[S_2_5, S_3_5, S_1_5, S_2_5]))],
            packed(&[I_2, I_4, I_2, I_2]),
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn fcvtau_breaks_the_same_tie_away_from_zero() {
    // 3.5 rounds to 4 either way; 2.5 is what separates the two modes.
    assert_eq!(
        solve_lowering(
            "fcvtau",
            &["v0.4s", "v1.4s"],
            &[("v1", packed(&[S_2_5, S_3_5, S_1_5, S_2_5]))],
            packed(&[I_3, I_4, I_2, I_3]),
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn a_half_width_arrangement_zeroes_the_rest_of_the_register() {
    // An `AArch64` vector write has no merging form. The preset above
    // the `.2s` view must be gone.
    assert_eq!(
        solve_lowering(
            "fcvtms",
            &["v0.2s", "v1.2s"],
            &[("v0", PARENT_PRESET), ("v1", packed(&[S_2_5, S_NEG_2_5]))],
            packed(&[I_2, I_NEG_3]),
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn the_fixed_point_form_survives_beside_the_directed_ones() {
    // `fcvtzs v0.4s, v1.4s, #1` reads the float as a value to be scaled
    // by `2^1` before truncating, so 2.5 becomes 5 and -1.5 becomes -3.
    // The fraction width is the one operand the directed spellings must
    // *not* accept, so the two facts are checked side by side.
    assert_eq!(
        solve_lowering(
            "fcvtzs",
            &["v0.4s", "v1.4s", "#1"],
            &[("v1", packed(&[S_2_5, S_NEG_2_5, S_1_5, S_NEG_1_5]))],
            packed(&[I_5, 0xffff_fffb, I_3, I_NEG_3]),
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn the_directed_conversions_decline_a_fraction_width() {
    // ARM ARM C7.2 spells no fixed-point form of these — only `fcvtz*`
    // carries one. Accepting a third operand would model an encoding
    // that does not exist.
    for mnemonic in ["fcvtas", "fcvtnu", "fcvtps", "fcvtmu"] {
        assert_declines(mnemonic, &["v0.4s", "v1.4s", "#1"]);
    }
}

#[test]
fn the_directed_conversions_decline_a_byte_arrangement() {
    // `.16b` is a perfectly good arrangement and a perfectly bad float
    // sort; reading a byte lane as a float is a wrong value.
    assert_declines("fcvtns", &["v0.16b", "v1.16b"]);
}

#[test]
fn the_directed_conversions_decline_the_upper_half_spelling() {
    // Only the width-changing `fcvtl` / `fcvtn` carry a `2` suffix.
    // `fcvtns2` is not an instruction, and peeling the digit would make
    // it resolve as one.
    assert_declines("fcvtns2", &["v0.4s", "v1.4s"]);
}
