//! `AArch64` contracts for the scalar pairwise forms — `addp d0,
//! v1.2d`, `faddp s0, v1.2s` and the float selects beside them.
//!
//! These are the shape the vector pairwise family cannot express: one
//! source, two lanes, one element out. The vector lowering splits its
//! destination between *two* operands, so reaching it at one lane would
//! take the pair out of an operand that is not there — which is why the
//! variant is its own and why the fixtures below bind only `v1`.
//!
//! The float selects carry the same NaN and signed-zero hazard the rest
//! of the ARM min / max family does: `FPMax` propagates a NaN and
//! combines the signs of a zero tie, where Intel's `MAXPS` returns its
//! second operand in both cases. The `±0` fixture is the one that
//! separates them without asserting on a NaN payload.
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
/// Binding for the destination parent, so every expectation also answers
/// whether the scalar write zeroed the rest of the register.
const PARENT_PRESET: u128 = u128::MAX;

// IEEE binary32 patterns.
const S_1_0: u128 = 0x3f80_0000;
const S_2_0: u128 = 0x4000_0000;
const S_3_0: u128 = 0x4040_0000;
const S_PLUS_ZERO: u128 = 0x0000_0000;
const S_NEG_ZERO: u128 = 0x8000_0000;

// IEEE binary64 patterns.
const D_1_0: u128 = 0x3ff0_0000_0000_0000;
const D_2_0: u128 = 0x4000_0000_0000_0000;

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
        mnemonic: "pairtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "pairtest".to_string(),
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

/// Two lanes of `lane_bits` packed little-endian.
fn two(lane_bits: u32, low: u128, high: u128) -> u128 {
    low | (high << lane_bits)
}

fn assert_folds(
    mnemonic: &str,
    operands: &[&str],
    lane_bits: u32,
    lanes: (u128, u128),
    expected: u128,
) {
    assert_eq!(
        solve_lowering(
            mnemonic,
            operands,
            &[
                ("v0", PARENT_PRESET),
                ("v1", two(lane_bits, lanes.0, lanes.1))
            ],
            expected,
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} over {lanes:x?} must give {expected:#x} in v0"
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
fn addp_folds_the_single_sources_two_lanes() {
    // The one integer member with a scalar form. The expectation is the
    // whole 128-bit register, so the write's zeroing of everything above
    // `d0` is asserted alongside the sum.
    assert_folds("addp", &["d0", "v1.2d"], 64, (3, 4), 7);
}

#[test]
fn addp_wraps_inside_the_element_rather_than_widening() {
    // `-1 + 2` at 64 bits is `1`. A lowering that widened would answer
    // `0x1_0000_0000_0000_0001`, which does not fit `d0` at all.
    assert_folds("addp", &["d0", "v1.2d"], 64, (0xffff_ffff_ffff_ffff, 2), 1);
}

#[test]
fn faddp_sums_the_two_binary32_lanes() {
    assert_folds("faddp", &["s0", "v1.2s"], 32, (S_1_0, S_2_0), S_3_0);
}

#[test]
fn faddp_at_double_precision_reads_the_destination_letter() {
    // The geometry comes from operand 1 and the destination's width is
    // checked against it, so the same mnemonic covers both widths
    // without either being assumed.
    assert_folds("faddp", &["d0", "v1.2d"], 64, (D_1_0, D_1_0), D_2_0);
}

#[test]
fn fminp_selects_the_smaller_lane() {
    assert_folds("fminp", &["d0", "v1.2d"], 64, (D_2_0, D_1_0), D_1_0);
}

#[test]
fn fmaxp_combines_the_signs_of_a_zero_tie_rather_than_taking_the_second_lane() {
    // ARM's `FPMax` resolves `+0` against `-0` by combining the two
    // signs — `AND` for max, so `+0`. Intel's `MAXPS`, which
    // `fp_lane_result` spells, returns its *second* operand and would
    // answer `-0` here. Same bit width, different bit pattern: a wrong
    // value, not a wider one.
    assert_folds(
        "fmaxp",
        &["s0", "v1.2s"],
        32,
        (S_PLUS_ZERO, S_NEG_ZERO),
        S_PLUS_ZERO,
    );
}

#[test]
fn fminp_combines_the_same_tie_the_other_way() {
    // `OR` for min, so `-0` — the mirror image, which one shared
    // implementation cannot get right in only one direction.
    assert_folds(
        "fminp",
        &["s0", "v1.2s"],
        32,
        (S_PLUS_ZERO, S_NEG_ZERO),
        S_NEG_ZERO,
    );
}

#[test]
fn the_scalar_pairwise_forms_decline_a_source_that_is_not_two_lanes() {
    // Four lanes is the vector form's shape, and folding it into one
    // element would drop half the source.
    assert_declines("faddp", &["s0", "v1.4s"]);
}

#[test]
fn the_scalar_pairwise_forms_decline_a_mismatched_destination_width() {
    // The destination's width is checked against the element the fold
    // produces rather than assumed, which is what rejects this.
    assert_declines("faddp", &["s0", "v1.2d"]);
}

#[test]
fn the_integer_pairwise_selects_have_no_scalar_form() {
    // ARM ARM C7.2 spells a scalar `addp` and no scalar `smaxp` /
    // `uminp` / …, so accepting one would model an instruction that
    // cannot occur.
    for mnemonic in ["smaxp", "sminp", "umaxp", "uminp"] {
        assert_declines(mnemonic, &["d0", "v1.2d"]);
    }
}

#[test]
fn scalar_addp_encodes_the_doubleword_source_only() {
    // The integer scalar form is `addp d0, v1.2d` and nothing else.
    assert_declines("addp", &["s0", "v1.2s"]);
}
