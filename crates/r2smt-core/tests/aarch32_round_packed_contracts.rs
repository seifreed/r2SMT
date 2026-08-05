//! The packed `AArch32` forms whose rounding mode the mnemonic names —
//! `vrint<mode>` and the directed `vcvta` / `vcvtn` / `vcvtp` /
//! `vcvtm` — solved rather than asserted structurally.
//!
//! This family is where a missing resolver is a **wrong value** and not
//! a decline. `vrinta.f32` names the VFP form and the Advanced SIMD one
//! with the same mnemonic, separated only by the operand's register
//! class, and the scalar handler was already in the dispatch — so a
//! packed form nothing claims would round lane zero and leave every
//! other lane of the register standing.
//!
//! Every test below therefore poisons the destination parent with a
//! pattern no correct lane can hold and asserts the *whole* view, which
//! is what makes a lane-zero lowering visible. The rounding modes are
//! pinned against each other on inputs that separate them: a value at a
//! tie tells `vrintn` from `vrinta`, and a negative half tells `vrintp`
//! from `vrintm` from `vrintz`.
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

/// A pattern no rounded binary32 lane can hold: both halves have a
/// non-zero significand under an exponent that would make them huge, so
/// neither is an integral value any input rounds to. Binding the
/// destination to it is what turns "lane one was never written" into a
/// failed assertion.
const POISON: u128 = 0xdead_beef_dead_beef;

/// `d0` / `q0` view the parent `v0`; `d2` / `q1` view `v1`. Sourcing
/// from a different parent than the destination keeps the binding and
/// the expectation independent.
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
        mnemonic: "roundtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "roundtest".to_string(),
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
        operands: operands.iter().map(|o| operand(o)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Lift `mnemonic operands` on `Arch::Arm`, bind every named parent to a
/// concrete value, and ask the solver whether the low `width` bits of
/// `v0` are necessarily `expected`.
fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
    width: u16,
) -> SmtResult {
    let lifted = lift_per_mnemonic(&instruction(mnemonic, operands), Arch::Arm);
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
            Expr::extract(Expr::Var(Var::new("v0", VECTOR_BITS)), width - 1, 0),
            Expr::konst(expected, width),
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

/// Assert a two-lane `d` form writes both lanes: the destination parent
/// starts poisoned, so an expectation covering the whole 64-bit view
/// fails unless every lane was rounded.
fn assert_doubleword(mnemonic: &str, source: u128, expected: u128) {
    assert_eq!(
        solve_lowering(
            mnemonic,
            &["d0", "d2"],
            &[("v0", POISON), ("v1", source)],
            expected,
            64,
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} d0, d2 over {source:#x} should give {expected:#x}"
    );
}

/// Assert the instruction declines — no statement the lifter emits may
/// be anything but `Unsupported`.
fn assert_declines(mnemonic: &str, operands: &[&str]) {
    let lifted = lift_per_mnemonic(&instruction(mnemonic, operands), Arch::Arm);
    assert!(
        lifted
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} should decline, got {lifted:?}"
    );
}

/// 1.5 in the low lane, 2.5 in the high one — the pair that separates
/// every rounding mode this family spells.
const HALVES: u128 = 0x4020_0000_3fc0_0000;
/// The same magnitudes negated, which is what tells the two directed
/// modes apart from round-toward-zero.
const NEGATIVE_HALVES: u128 = 0xc020_0000_bfc0_0000;

#[test]
fn packed_vrintn_rounds_every_lane_not_just_the_first() {
    // The soundness contract of the whole family. 1.5 and 2.5 both round
    // to 2.0 under ties-to-even, so a lane-zero lowering would leave
    // `0xdeadbeef` standing in the high lane and this expectation would
    // fail on exactly that.
    assert_doubleword("vrintn.f32", HALVES, 0x4000_0000_4000_0000);
}

#[test]
fn packed_vrinta_breaks_the_tie_away_from_zero() {
    // The one input separating `vrinta` from `vrintn`: 2.5 becomes 3.0
    // where ties-to-even gives 2.0. Without this pair the two modes are
    // indistinguishable.
    assert_doubleword("vrinta.f32", HALVES, 0x4040_0000_4000_0000);
}

#[test]
fn packed_vrintz_truncates_toward_zero_in_both_directions() {
    // -1.5 becomes -1.0 and -2.5 becomes -2.0: toward zero moves the
    // negative lanes *up*, which is what a lowering that silently used
    // round-toward-negative would get wrong.
    assert_doubleword("vrintz.f32", NEGATIVE_HALVES, 0xc000_0000_bf80_0000);
}

#[test]
fn packed_vrintm_rounds_toward_negative_infinity() {
    // The companion of the test above on the identical input: -1.5 goes
    // to -2.0 rather than -1.0.
    assert_doubleword("vrintm.f32", NEGATIVE_HALVES, 0xc040_0000_c000_0000);
}

#[test]
fn packed_vrintp_rounds_toward_positive_infinity() {
    // Third reading of the same lanes, so all three directed modes are
    // pinned against each other rather than each against zero.
    assert_doubleword("vrintp.f32", NEGATIVE_HALVES, 0xc000_0000_bf80_0000);
}

#[test]
fn packed_vrintx_takes_the_control_word_default() {
    // `vrintx` rounds by FPSCR, which the lifter pins at ties-to-even —
    // the mode `vrintn` names — so it agrees with `vrintn` and not with
    // `vrinta` on the tie. `writes_rounding_control` truncates a slice
    // whose function reprograms FPSCR, which is what makes the pin
    // sound rather than assumed.
    assert_doubleword("vrintx.f32", HALVES, 0x4000_0000_4000_0000);
}

#[test]
fn packed_vrint_over_a_quadword_writes_all_four_lanes() {
    // A `q` destination is four lanes, and the poison covers the upper
    // half of the parent — so this fails if the lowering resolved the
    // view as a `d`.
    assert_eq!(
        solve_lowering(
            "vrintz.f32",
            &["q0", "q1"],
            &[
                ("v0", POISON),
                ("v1", 0x3fc0_0000_bfc0_0000_4020_0000_3fc0_0000)
            ],
            0x3f80_0000_bf80_0000_4000_0000_3f80_0000,
            128,
        ),
        SmtResult::AlwaysTrue,
        "vrintz.f32 q0, q1 should round all four lanes"
    );
}

#[test]
fn scalar_vrint_still_merges_rather_than_zeroing_the_register() {
    // The other half of the split: a single-lane view is the VFP
    // encoding, which this resolver deliberately leaves to the scalar
    // arm. An `AArch32` VFP write merges, so the poison above `s0`
    // survives — and would not if the packed resolver had claimed the
    // form and written the whole view.
    assert_eq!(
        solve_lowering(
            "vrintz.f32",
            &["s0", "s4"],
            &[("v0", POISON), ("v1", 0x3fc0_0000)],
            0xdead_beef_3f80_0000,
            64,
        ),
        SmtResult::AlwaysTrue,
        "vrintz.f32 s0, s4 should write only the addressed lane"
    );
}

/// 2.5 and -2.5 in adjacent binary32 lanes — the pair on which all four
/// directed conversion modes give four different answers, and on which
/// plain `vcvt`'s round-toward-zero gives a fifth.
const SIGNED_TIES: u128 = 0xc020_0000_4020_0000;

#[test]
fn packed_vcvta_converts_every_lane_with_ties_away_from_zero() {
    // The directed conversions reach the packed path through
    // `convert_shape`, which already claimed the plain `vcvt`; nothing
    // pinned the *mode* surviving that route until here. Ties away from
    // zero sends 2.5 to 3 and -2.5 to -3, which no other mode produces.
    assert_doubleword("vcvta.s32.f32", SIGNED_TIES, 0xffff_fffd_0000_0003);
}

#[test]
fn packed_vcvtm_converts_every_lane_toward_negative_infinity() {
    // The same lanes read the other way: 2 and -3. Between this and the
    // test above, a mode swapped for round-toward-zero (2 and -2) or for
    // ties-to-even (2 and -2) fails one of the two.
    assert_doubleword("vcvtm.s32.f32", SIGNED_TIES, 0xffff_fffd_0000_0002);
}

#[test]
fn packed_vrintr_declines_because_advanced_simd_has_no_such_encoding() {
    // `vrintr` is VFP-only — Advanced SIMD spells its round-by-FPSCR
    // form `vrintx` — so the packed resolver excludes it by name. It
    // then reaches the scalar arm, whose register-class check refuses a
    // `q` operand, and the instruction fails closed.
    assert_declines("vrintr.f32", &["q0", "q1"]);
}

#[test]
fn packed_vrint_at_double_precision_declines() {
    // There is no `.f64` Advanced SIMD rounding form. A `q` destination
    // with a 64-bit element would be two lanes if the resolver took the
    // width from the mnemonic alone, so this is the guard on the
    // element table rather than on the operand.
    assert_declines("vrintz.f64", &["q0", "q1"]);
}
