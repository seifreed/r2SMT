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
const S_3_0: u128 = 0x4040_0000;
const S_NEG_1_0: u128 = 0xbf80_0000;

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

// the scalar spelling
//
// `fcvtas s0, s1` carries no arrangement on either operand, so it
// reaches the resolvers only through the scalar-family allowlist in
// `vector_shape`. Before that it fell out of the dispatch's unknown-
// mnemonic arm — and, unlike `sqabs`, with no decline contract naming it
// in either direction. These pin the mode, because getting the mode
// wrong here is a wrong integer rather than a decline.

/// One scalar convert of `1.5`, whose result separates all four modes:
/// ties-to-even gives 2, ties-away 2, toward `+inf` 2, toward `-inf` 1
/// and truncation 1. Paired with `-1.5` below to split the first two.
fn assert_scalar_converts(mnemonic: &str, source: u128, expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, &["s0", "s1"], &[("v1", source)], expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} of {source:#x} must give {expected:#x}"
    );
}

#[test]
fn scalar_fcvtns_breaks_a_tie_to_even() {
    // 2.5 -> 2, which is what separates ties-to-even from ties-away.
    assert_scalar_converts("fcvtns", S_2_5, I_2);
}

#[test]
fn scalar_fcvtas_breaks_the_same_tie_away_from_zero() {
    assert_scalar_converts("fcvtas", S_2_5, I_3);
}

#[test]
fn scalar_fcvtps_rounds_toward_positive_infinity() {
    // -1.5 toward `+inf` is -1; toward `-inf` it is -2. Using a negative
    // input is what makes this fail for a lowering that truncates.
    assert_scalar_converts("fcvtps", S_NEG_1_5, I_NEG_1);
}

#[test]
fn scalar_fcvtms_rounds_toward_negative_infinity() {
    assert_scalar_converts("fcvtms", S_NEG_1_5, I_NEG_2);
}

#[test]
fn scalar_fcvtnu_covers_the_unsigned_range_rather_than_the_signed_one() {
    // 3e9 is above `INT_MAX` and below `UINT_MAX`. The IR has no
    // unsigned conversion node, so the lowering goes through the signed
    // one with an extra bit of range; dropping that would saturate at
    // 0x7fffffff, a wrong value rather than a lost one.
    const S_3E9: u128 = 0x4f32_d05e;
    assert_scalar_converts("fcvtnu", S_3E9, 0xb2d0_5e00);
}

#[test]
fn a_scalar_convert_zeroes_the_rest_of_the_vector_register() {
    // The write is not a merge. `v0` is preset to all ones, so a
    // lowering that preserved the upper bits fails here.
    assert_eq!(
        solve_lowering(
            "fcvtas",
            &["s0", "s1"],
            &[("v0", PARENT_PRESET), ("v1", S_2_5)],
            I_3,
        ),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn the_scalar_converts_that_already_had_a_handler_still_lift() {
    // `fcvtzs` / `fcvtzu` and `scvtf` / `ucvtf` lift through the
    // scalar-FP arm of the dispatch, so they are deliberately outside
    // the allowlist: claiming them here would give one instruction two
    // lowerings. This is the cheap guard against the allowlist growing
    // to swallow them.
    //
    // Only that they lift, which is the property the allowlist change
    // could break. What they *compute* is pinned for `scvtf` by the
    // contract below; it could not be pinned here while the scalar-FP
    // path named `v0` at the pointer width, because the 128-bit binding
    // this harness uses could not reach it. See
    // `.planning/2026-08-05-parent-width-scalar-fp.md`.
    for (mnemonic, operands) in [
        ("fcvtzs", ["s0", "s1"]),
        ("fcvtzu", ["s0", "s1"]),
        ("scvtf", ["s0", "s1"]),
        ("ucvtf", ["s0", "s1"]),
    ] {
        let lifted = lift_per_mnemonic(&instruction(mnemonic, &operands), Arch::Aarch64);
        assert!(
            lifted
                .iter()
                .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
            "{mnemonic} {operands:?} should still lift: {lifted:?}"
        );
    }
}

#[test]
fn scvtf_reads_the_integer_out_of_the_vector_register() {
    // `scvtf s0, s1` converts an integer held *in* the vector file, so
    // both ends of it are `v` registers. Reading the source through the
    // general-register path would name `v1` at the pointer width — 64
    // here, where the register is 128 — and this harness binds `v1` at
    // 128, so that spelling cannot see the binding at all. The value is
    // what proves the read reaches it: 3 becomes binary32 `3.0`, with
    // the rest of the destination cleared as an `AArch64` scalar SIMD
    // write does.
    assert_eq!(
        solve_lowering("scvtf", &["s0", "s1"], &[("v1", I_3)], S_3_0),
        SmtResult::AlwaysTrue,
    );
}

#[test]
fn the_scalar_directed_converts_reject_shapes_the_encoding_lacks() {
    // No fixed-point form for the directed spellings, mismatched views
    // are not an encoding, and a byte view names no IEEE format.
    assert_declines("fcvtas", &["s0", "s1", "#1"]);
    assert_declines("fcvtas", &["s0", "d1"]);
    assert_declines("fcvtas", &["b0", "b1"]);
}

// the general-register spelling
//
// `fcvtas w0, s1` is the form neither other resolver can express,
// because the two ends decouple: `x0, s1` and `w0, d1` are both legal,
// so no single width describes the instruction. What these pin is the
// decoupling itself and the unsigned range, which are the two places a
// lowering that assumed the widths matched would produce a number rather
// than a decline.

const GPR_BITS: u16 = 64;

fn solve_gpr_lowering(
    mnemonic: &str,
    operands: &[&str],
    source: u128,
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
    let mut statements = vec![IrStmt::Assign {
        dst: Var::new("v1", VECTOR_BITS),
        src: Expr::konst(source, VECTOR_BITS),
    }];
    statements.extend(lifted);
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::Var(Var::new("x0", GPR_BITS)),
            Expr::konst(expected, GPR_BITS),
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

fn assert_gpr_converts(mnemonic: &str, operands: &[&str], source: u128, expected: u128) {
    assert_eq!(
        solve_gpr_lowering(mnemonic, operands, source, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} of {source:#x} must give {expected:#x}"
    );
}

#[test]
fn gpr_fcvtas_rounds_ties_away_into_a_word_register() {
    assert_gpr_converts("fcvtas", &["w0", "s1"], S_2_5, I_3);
}

#[test]
fn gpr_fcvtms_and_fcvtps_disagree_on_a_negative_half() {
    // -1.5 toward `-inf` is -2 and toward `+inf` is -1. A `w` write
    // zero-extends the parent, so the 64-bit view keeps the 32-bit
    // pattern rather than sign-extending it — which is the ARM rule and
    // also the thing a lowering that wrote `x0` directly would get wrong.
    assert_gpr_converts("fcvtms", &["w0", "s1"], S_NEG_1_5, 0xffff_fffe);
    assert_gpr_converts("fcvtps", &["w0", "s1"], S_NEG_1_5, 0xffff_ffff);
}

#[test]
fn gpr_conversion_widths_decouple_between_source_and_destination() {
    // A binary32 source into a 64-bit register: 3e9 does not fit a
    // signed word, and the destination is not the source's width. A
    // resolver that required the two to agree would decline this, and
    // one that took the destination's width for the source's would read
    // the wrong format.
    const S_3E9: u128 = 0x4f32_d05e;
    assert_gpr_converts("fcvtns", &["x0", "s1"], S_3E9, 0xb2d0_5e00);
}

#[test]
fn gpr_fcvtnu_covers_the_upper_half_of_the_unsigned_range() {
    // 2^63 as a double. The unsigned conversion must reach it; the
    // signed one saturates at 2^63 - 1, so a lowering that dropped the
    // extra bit of range answers 0x7fffffffffffffff here.
    const D_TWO_POW_63: u128 = 0x43e0_0000_0000_0000;
    assert_gpr_converts("fcvtnu", &["x0", "d1"], D_TWO_POW_63, 0x8000_0000_0000_0000);
}

#[test]
fn gpr_fcvtas_reads_a_double_source_as_binary64() {
    // The mirror of the decoupling test: a `d` source into a `w`
    // destination. Reading it as binary32 would give a different number
    // entirely rather than a decline.
    const D_NEG_1_5: u128 = 0xbff8_0000_0000_0000;
    assert_gpr_converts("fcvtas", &["w0", "d1"], D_NEG_1_5, 0xffff_fffe);
}

#[test]
fn gpr_fcvtzu_saturates_a_negative_source_to_zero() {
    // `fcvtzu w0, s1` of -1.0. ARM `FCVTZU` saturates a negative source
    // to 0; the signed `fp.to_sbv` the lowering is built on would instead
    // wrap it to 0xffffffff (the unsigned maximum). Regression for the
    // missing low-end clamp.
    assert_gpr_converts("fcvtzu", &["w0", "s1"], S_NEG_1_0, 0x0000_0000);
}

#[test]
fn packed_fcvtzu_saturates_negative_lanes_to_zero() {
    // `fcvtzu v0.4s, v1.4s` truncates toward zero and saturates negatives
    // to 0: [-1.0, -2.5, 1.5, 3.0] -> [0, 0, 1, 3]. A lane that wrapped a
    // negative would show 0xffffffff / 0xfffffffe instead of 0.
    assert_eq!(
        solve_lowering(
            "fcvtzu",
            &["v0.4s", "v1.4s"],
            &[("v1", packed(&[S_NEG_1_0, S_NEG_2_5, S_1_5, S_3_0]))],
            packed(&[0, 0, I_1, I_3]),
        ),
        SmtResult::AlwaysTrue,
        "fcvtzu must saturate negative lanes to 0 and truncate the rest"
    );
}
