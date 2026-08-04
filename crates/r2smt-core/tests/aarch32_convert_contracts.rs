//! `AArch32` VFP `vcvt` contracts, solved rather than asserted
//! structurally.
//!
//! The corpus cannot supply evidence here: its 32-bit ARM samples
//! exercise no floating-point conversion inside an analysed branch
//! predicate, so a verdict diff would be silent whether the lowering is
//! right or wrong. What a wrong lowering produces is a *wrong value* —
//! a rounding mode taken from the wrong end, an unsigned range folded
//! into a signed one — so the only evidence worth having is a solver
//! agreeing the destination equals a hand-computed ARM result.
//!
//! Each test below is aimed at a specific way the lowering could be
//! wrong while still lifting cleanly.
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
const VECTOR_BITS: u16 = 128;

/// `s0`–`s3` and `d0` are views of the synthetic parent `v0`; `s4` is
/// the first view of `v1`. Sourcing from a different parent than the
/// destination keeps the binding and the expectation independent.
const DEST: &str = "s0";
const SOURCE: &str = "s4";

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with('#') {
            OperandKind::Immediate
        } else {
            OperandKind::Register
        },
    }
}

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "converttest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "converttest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands` on `Arch::Arm`, bind every named parent to a
/// concrete value, and ask the solver whether the low 64 bits of `v0`
/// are necessarily `expected`.
///
/// An `AArch32` VFP write merges, so every test binds `v0` as well as
/// its source — the bits the conversion does not cover come from that
/// binding rather than being free.
fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|o| operand(o)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    };
    let lifted = lift_per_mnemonic(&insn, Arch::Arm);
    assert!(
        lifted
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} declined: {lifted:?}"
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
            Expr::extract(Expr::Var(Var::new("v0", VECTOR_BITS)), 63, 0),
            Expr::konst(expected, 64),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Arm,
    };
    solve_branch(
        &ssa_convert(&slice),
        SolveOptions {
            timeout_ms: TEST_SOLVE_TIMEOUT_MS,
            ..SolveOptions::default()
        },
    )
}

/// Assert the lowering computes exactly `expected` in the low 64 bits
/// of `v0`.
fn assert_computes(mnemonic: &str, operands: &[&str], sources: &[(&str, u128)], expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

/// The bindings for a conversion out of `SOURCE` into `DEST`: the
/// source parent holds `bits`, and the destination parent is zero so
/// the merge contributes nothing to the expectation.
fn from_source(bits: u128) -> [(&'static str, u128); 2] {
    [("v0", 0), ("v1", bits)]
}

#[test]
fn vcvt_to_unsigned_covers_the_range_above_signed_maximum() {
    // 3e9 is representable exactly in binary32 and sits above
    // `i32::MAX`. The IR carries no unsigned conversion, so the lowering
    // converts through the signed node one bit wider and truncates —
    // which covers the unsigned range exactly. Going through the signed
    // node at the destination's own width would saturate at 0x7fffffff.
    assert_computes(
        "vcvt.u32.f32",
        &[DEST, SOURCE],
        &from_source(0x4f32_d05e),
        0xb2d0_5e00,
    );
}

#[test]
fn vcvt_to_signed_rounds_toward_zero() {
    // Plain `vcvt` into an integer carries round-toward-zero in the
    // opcode: -1.5 becomes -1. Rounding to nearest would give -2, and
    // nothing about the lift would look wrong.
    assert_computes(
        "vcvt.s32.f32",
        &[DEST, SOURCE],
        &from_source(0xbfc0_0000),
        0xffff_ffff,
    );
}

#[test]
fn vcvtm_rounds_toward_negative_infinity() {
    // The companion direction, so the two forms are genuinely
    // distinguished rather than both landing on round-toward-zero:
    // `vcvtm` of -1.5 is -2 where `vcvt` is -1.
    assert_computes(
        "vcvtm.s32.f32",
        &[DEST, SOURCE],
        &from_source(0xbfc0_0000),
        0xffff_fffe,
    );
}

#[test]
fn vcvtn_breaks_a_tie_to_even() {
    // 2.5 is exactly between 2 and 3. `vcvtn` is round-to-nearest with
    // ties to even, so it gives 2.
    assert_computes(
        "vcvtn.s32.f32",
        &[DEST, SOURCE],
        &from_source(0x4020_0000),
        2,
    );
}

#[test]
fn vcvta_breaks_the_same_tie_away_from_zero() {
    // The one input that tells `vcvta` from `vcvtn`: ties away from zero
    // gives 3 where ties to even gives 2. Without this pair the two
    // modes are indistinguishable on every test above.
    assert_computes(
        "vcvta.s32.f32",
        &[DEST, SOURCE],
        &from_source(0x4020_0000),
        3,
    );
}

#[test]
fn vcvt_from_signed_integer_reads_the_source_as_signed() {
    // `0xffffffff` is -1 read signed and 4294967295 read unsigned, so a
    // lowering that lost the signedness would produce a large positive
    // float rather than -1.0.
    assert_computes(
        "vcvt.f32.s32",
        &[DEST, SOURCE],
        &from_source(0xffff_ffff),
        0xbf80_0000,
    );
}

#[test]
fn vcvt_from_unsigned_integer_reads_the_same_bits_as_unsigned() {
    // The companion direction on the identical input: 4294967295 rounds
    // to 4294967296.0 in binary32, whose pattern is 0x4f800000.
    assert_computes(
        "vcvt.f32.u32",
        &[DEST, SOURCE],
        &from_source(0xffff_ffff),
        0x4f80_0000,
    );
}

#[test]
fn vcvt_between_float_widths_converts_the_value_not_the_bits() {
    // `vcvt.f64.f32 d0, s4` widens 1.5f to 1.5. Reinterpreting the bit
    // pattern instead would give a denormal, and reading the source at
    // the destination's width would take 64 bits of `v1`.
    assert_computes(
        "vcvt.f64.f32",
        &["d0", SOURCE],
        &from_source(0x3fc0_0000),
        0x3ff8_0000_0000_0000,
    );
}

#[test]
fn packed_vcvt_converts_every_lane_not_just_the_first() {
    // The scalar VFP arm recognises the same mnemonic, so a packed form
    // the NEON resolver missed would convert lane zero and leave the
    // rest of the register standing. Two different lanes with two
    // different values is what tells the two lowerings apart: 1.5 and
    // 2.5 truncate to 1 and 2.
    assert_computes(
        "vcvt.s32.f32",
        &["d0", "d2"],
        &[("v0", 0), ("v1", 0x4020_0000_3fc0_0000)],
        0x0000_0002_0000_0001,
    );
}

#[test]
fn packed_vcvt_from_integer_reads_lanes_as_signed() {
    // -1 and 2 in adjacent lanes: reading them unsigned would make the
    // low lane 4294967296.0 rather than -1.0.
    assert_computes(
        "vcvt.f32.s32",
        &["d0", "d2"],
        &[("v0", 0), ("v1", 0x0000_0002_ffff_ffff)],
        0x4000_0000_bf80_0000,
    );
}

#[test]
fn vcvt_fixed_point_scales_by_the_fraction_width() {
    // The fixed-point form reads the low 16 bits as a signed value
    // scaled by 2^-8: 384 / 256 = 1.5. Ignoring the third operand would
    // give 384.0 instead, and the instruction would still lift.
    //
    // ARM requires one register for both operands here, so the
    // destination parent carries the source bits.
    assert_computes(
        "vcvt.f32.s16",
        &[DEST, DEST, "#8"],
        &[("v0", 0x0180)],
        0x3fc0_0000,
    );
}
