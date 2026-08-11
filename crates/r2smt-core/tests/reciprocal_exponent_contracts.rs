//! `AArch64` `frecpx` contracts — solved, never asserted structurally.
//!
//! The hazard this file exists for is a naming collision, not an
//! arithmetic one. `frecpe` and `frecpx` differ by a letter and are
//! opposite kinds of thing: the estimate's value is
//! implementation-defined inside a relative error bound, so it lowers to
//! a free value, while `FPRecpX` is defined exactly. Confusing them in
//! either direction is a bug — a free value here throws away a
//! guaranteed fact, and an exact value there fabricates one.
//!
//! So the fixtures are chosen to fail against every plausible
//! misreading. `frecpx(1.0)` is `2.0` and not `1.0`, and `frecpx(2.0)`
//! is `1.0` and not `0.5`, so a lowering that computed an actual
//! reciprocal fails both. A zero exponent maps to `Ones(E) - 1` rather
//! than to its complement, so the subnormal fixture fails a lowering
//! that complemented unconditionally — it would answer infinity. And an
//! infinity maps to a zero of the same sign, which is the same
//! complement seen from the other end.
//!
//! Every fixture binds the destination's vector parent to all-ones, so
//! the expectation is only met if the scalar write zeroed everything
//! above the destination view, as `AArch64` SIMD&FP semantics require.
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
/// Width of the synthetic vector parent every `AArch64` register view
/// maps to.
const VECTOR_BITS: u16 = 128;
/// Binding for the destination parent, chosen so that "the result is
/// exactly this" also answers whether the write zeroed the bits it does
/// not cover.
const PARENT_PRESET: u128 = u128::MAX;

// IEEE binary64 patterns.
const D_1_0: u128 = 0x3ff0_0000_0000_0000;
const D_2_0: u128 = 0x4000_0000_0000_0000;
const D_NEG_2_0: u128 = 0xc000_0000_0000_0000;
const D_NEG_1_0: u128 = 0xbff0_0000_0000_0000;
const D_PLUS_ZERO: u128 = 0x0000_0000_0000_0000;
const D_NEG_ZERO: u128 = 0x8000_0000_0000_0000;
const D_PLUS_INF: u128 = 0x7ff0_0000_0000_0000;
const D_QNAN: u128 = 0x7ff8_0000_0000_0000;
const D_SNAN: u128 = 0x7ff0_0000_0000_0001;
/// `2^1023`, the value a zero exponent maps to: sign, `Ones(11) - 1`,
/// cleared significand.
const D_TWO_POW_1023: u128 = 0x7fe0_0000_0000_0000;

// IEEE binary32 patterns.
const S_1_0: u128 = 0x3f80_0000;
const S_2_0: u128 = 0x4000_0000;
/// The smallest positive subnormal — a zero exponent field with a
/// non-zero significand, which is the case a NaN test on the exponent
/// alone would get wrong.
const S_MIN_SUBNORMAL: u128 = 0x0000_0001;
/// `2^127` in binary32: `Ones(8) - 1` in the exponent field.
const S_TWO_POW_127: u128 = 0x7f00_0000;

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
        mnemonic: "frecpxtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "frecpxtest".to_string(),
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

/// Lift `mnemonic operands`, bind every named vector parent, and ask the
/// solver whether `v0` is necessarily `expected`.
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

/// `frecpx d0, d1` over a binary64 pattern.
fn assert_double(source: u128, expected: u128) {
    assert_eq!(
        solve_lowering(
            "frecpx",
            &["d0", "d1"],
            &[("v0", PARENT_PRESET), ("v1", source)],
            expected,
        ),
        SmtResult::AlwaysTrue,
        "frecpx of {source:#x} must give {expected:#x} in v0"
    );
}

#[test]
fn frecpx_of_one_is_two_and_not_the_reciprocal() {
    // The single fixture that separates this instruction from the
    // reciprocal it is named after: `1 / 1.0` is `1.0`, and negating the
    // unbiased exponent of `1.0` gives `2.0`.
    assert_double(D_1_0, D_2_0);
}

#[test]
fn frecpx_of_two_is_one_and_not_a_half() {
    // The other direction of the same confusion: `1 / 2.0` is `0.5`.
    assert_double(D_2_0, D_1_0);
}

#[test]
fn frecpx_keeps_the_sign() {
    // The sign bit is copied, not derived — the exponent field alone is
    // complemented.
    assert_double(D_NEG_2_0, D_NEG_1_0);
}

#[test]
fn frecpx_maps_a_zero_exponent_to_the_largest_finite_one() {
    // `Ones(E) - 1`, not `NOT(0)`. Complementing would give the all-ones
    // field, turning a zero into an infinity.
    assert_double(D_PLUS_ZERO, D_TWO_POW_1023);
}

#[test]
fn frecpx_of_negative_zero_keeps_the_sign_too() {
    assert_double(D_NEG_ZERO, D_TWO_POW_1023 | D_NEG_ZERO);
}

#[test]
fn frecpx_of_an_infinity_is_a_zero_of_the_same_sign() {
    // The complement seen from the other end: an all-ones exponent maps
    // to zero, and the significand is cleared either way.
    assert_double(D_PLUS_INF, D_PLUS_ZERO);
}

#[test]
fn frecpx_returns_a_quiet_nan_unchanged() {
    // `FPProcessNaN` under the reset FPCR (`DN == 0`) returns the
    // operand itself for a quiet NaN. Falling into the exponent arm
    // would clear the significand and turn it into an infinity.
    assert_double(D_QNAN, D_QNAN);
}

#[test]
fn frecpx_quietens_a_signalling_nan() {
    // The same path with the quiet bit forced on, which is the whole
    // difference between the two NaN classes here.
    assert_double(D_SNAN, D_SNAN | (1 << 51));
}

#[test]
fn frecpx_at_single_precision_reads_the_register_letter() {
    // The exponent field is five, eight or eleven bits wide depending on
    // the lane, so a lowering that hardcoded one width answers a
    // different number here.
    assert_eq!(
        solve_lowering(
            "frecpx",
            &["s0", "s1"],
            &[("v0", PARENT_PRESET), ("v1", S_1_0)],
            S_2_0,
        ),
        SmtResult::AlwaysTrue,
        "frecpx s0, s1 must complement the binary32 exponent"
    );
}

#[test]
fn frecpx_of_a_subnormal_takes_the_zero_exponent_path_not_the_nan_one() {
    // A subnormal has a zero exponent and a non-zero significand. A NaN
    // test that looked only at the significand would quieten it instead,
    // and one that complemented the exponent would answer infinity.
    assert_eq!(
        solve_lowering(
            "frecpx",
            &["s0", "s1"],
            &[("v0", PARENT_PRESET), ("v1", S_MIN_SUBNORMAL)],
            S_TWO_POW_127,
        ),
        SmtResult::AlwaysTrue,
        "frecpx of the smallest subnormal must give 2^127"
    );
}

#[test]
fn frecpe_stays_a_free_value_beside_it() {
    // The other half of the pair, asserted here so the two cannot drift
    // together. `frecpe` of `2.0` could plausibly be `0.5`, and a
    // lowering that computed it would make this `AlwaysTrue`; the
    // estimate is a free input instead, so `0.5` stays merely possible.
    assert_eq!(
        solve_lowering(
            "frecpe",
            &["v0.2d", "v1.2d"],
            &[("v1", D_2_0)],
            0x3fe0_0000_0000_0000,
        ),
        SmtResult::BothPossible,
    );
}

#[test]
fn frecpx_declines_the_arranged_spelling_the_architecture_does_not_encode() {
    // `FPRecpX` is scalar-only. An arranged operand must keep failing
    // closed rather than being lowered as one wide scalar.
    let lifted = lift_per_mnemonic(&instruction("frecpx", &["v0.4s", "v1.4s"]), Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "packed frecpx must decline, got {lifted:?}"
    );
}
