//! Contracts for the scalar round-to-integral families —  `AArch64`
//! `frint*`, `AArch32` VFP `vrint*` and x87 `frndint` — solved rather
//! than asserted structurally.
//!
//! Every one of these lowerings fails as a *wrong value* rather than as
//! a decline: a mode read off the wrong end of the table still lifts,
//! still solves, and still produces a number. So the evidence worth
//! having is a solver agreeing the destination equals a hand-computed
//! architectural result, and the inputs are chosen so each mode is
//! contrasted with the one it would plausibly be confused for.
//!
//! `±2.5` does most of that work alone: it is the smallest input that
//! separates ties-to-even from ties-away *and* toward-zero from
//! toward-negative. It does not separate ties-to-even from toward-zero,
//! which agree on it, so `1.5` appears once for that pair.
//!
//! The register-write shape is pinned by the same assertions rather
//! than by tests of their own: every fixture binds the destination's
//! vector parent to all-ones, so an `AArch64` expectation is only met
//! if the write zeroed the rest of the register and an `AArch32` one
//! only if it preserved it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceLimits, SliceStatus,
    collect_branches, lift_per_mnemonic, lift_slice, slice_branch,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 20_000;
/// Width of the synthetic vector parent every ARM register view maps to.
const VECTOR_BITS: u16 = 128;

/// Binding for a destination parent, chosen so that "the result is
/// exactly this" also answers whether the write zeroed or preserved the
/// bits it does not cover.
const PARENT_PRESET: u128 = u128::MAX;

// IEEE binary64 patterns.
const D_2_5: u128 = 0x4004_0000_0000_0000;
const D_NEG_2_5: u128 = 0xc004_0000_0000_0000;
const D_1_5: u128 = 0x3ff8_0000_0000_0000;
const D_1_0: u128 = 0x3ff0_0000_0000_0000;
const D_2_0: u128 = 0x4000_0000_0000_0000;
const D_3_0: u128 = 0x4008_0000_0000_0000;
const D_NEG_2_0: u128 = 0xc000_0000_0000_0000;
const D_NEG_3_0: u128 = 0xc008_0000_0000_0000;

// IEEE binary32 patterns.
const S_2_5: u128 = 0x4020_0000;
const S_NEG_2_5: u128 = 0xc020_0000;
const S_1_5: u128 = 0x3fc0_0000;
const S_1_0: u128 = 0x3f80_0000;
const S_2_0: u128 = 0x4000_0000;
const S_3_0: u128 = 0x4040_0000;
const S_NEG_2_0: u128 = 0xc000_0000;
const S_NEG_3_0: u128 = 0xc040_0000;

fn operand(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn reg(raw: &str) -> Operand {
    operand(raw, OperandKind::Register)
}

fn imm(raw: &str) -> Operand {
    operand(raw, OperandKind::Immediate)
}

fn mem(raw: &str) -> Operand {
    operand(raw, OperandKind::Memory)
}

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn synthetic_branch(arch: Arch) -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "rinttest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "rinttest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: matches!(arch, Arch::Arm),
    }
}

fn instruction(mnemonic: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Lift one ARM instruction, bind every named vector parent to a
/// concrete value, and ask the solver whether `v0` is necessarily
/// `expected`.
fn solve_arm_lowering(
    arch: Arch,
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = instruction(mnemonic, operands.iter().map(|o| reg(o)).collect());
    let lifted = lift_per_mnemonic(&insn, arch);
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
        branch: synthetic_branch(arch),
        statements,
        condition: Expr::eq(
            Expr::Var(Var::new("v0", VECTOR_BITS)),
            Expr::konst(expected, VECTOR_BITS),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

/// `AArch64`: round `source` out of `v1` into `d0`, and require the
/// whole of `v0` to be `expected` — so the scalar write's zeroing of
/// everything above the destination view is asserted alongside the
/// value.
fn assert_aarch64_rounds(mnemonic: &str, source: u128, expected: u128) {
    assert_eq!(
        solve_arm_lowering(
            Arch::Aarch64,
            mnemonic,
            &["d0", "d1"],
            &[("v0", PARENT_PRESET), ("v1", source)],
            expected,
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} of {source:#x} must give {expected:#x} in v0"
    );
}

/// `AArch32`: round `source` out of `s4` into `s0`. An `AArch32` VFP
/// write merges, so the expectation keeps the preset above the
/// destination's 32-bit view.
fn assert_aarch32_rounds(mnemonic: &str, source: u128, expected_lane: u128) {
    let expected = (PARENT_PRESET & !u128::from(u32::MAX)) | expected_lane;
    assert_eq!(
        solve_arm_lowering(
            Arch::Arm,
            mnemonic,
            &["s0", "s4"],
            &[("v0", PARENT_PRESET), ("v1", source)],
            expected,
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} of {source:#x} must give {expected_lane:#x} in s0"
    );
}

#[test]
fn frintn_breaks_a_tie_to_even() {
    assert_aarch64_rounds("frintn", D_2_5, D_2_0);
}

#[test]
fn frinta_breaks_the_same_tie_away_from_zero() {
    // The one input that tells `frinta` from `frintn`. Without the pair
    // the two modes are indistinguishable on every other fixture here.
    assert_aarch64_rounds("frinta", D_2_5, D_3_0);
}

#[test]
fn frintp_rounds_toward_positive_infinity() {
    assert_aarch64_rounds("frintp", D_2_5, D_3_0);
}

#[test]
fn frintm_rounds_toward_negative_infinity() {
    // The companion of `frintp` on the same input: 2 where it gives 3.
    assert_aarch64_rounds("frintm", D_2_5, D_2_0);
}

#[test]
fn frintz_rounds_a_negative_toward_zero_not_toward_negative_infinity() {
    // `frintz` and `frintm` agree on every positive input, so only a
    // negative one separates them: -2 against -3.
    assert_aarch64_rounds("frintz", D_NEG_2_5, D_NEG_2_0);
}

#[test]
fn frintm_rounds_a_negative_away_from_zero() {
    assert_aarch64_rounds("frintm", D_NEG_2_5, D_NEG_3_0);
}

#[test]
fn frinta_rounds_a_negative_away_from_zero_too() {
    // Ties away from zero is a magnitude rule, not a direction: -2.5
    // goes to -3 where ties-to-even gives -2.
    assert_aarch64_rounds("frinta", D_NEG_2_5, D_NEG_3_0);
}

#[test]
fn frintn_rounds_a_negative_tie_to_even() {
    assert_aarch64_rounds("frintn", D_NEG_2_5, D_NEG_2_0);
}

#[test]
fn frintz_truncates_where_frintn_rounds_up() {
    // ±2.5 cannot separate ties-to-even from toward-zero — both give 2.
    // 1.5 does: nearest gives 2, truncation gives 1.
    assert_aarch64_rounds("frintz", D_1_5, D_1_0);
}

#[test]
fn frintn_rounds_one_and_a_half_to_two() {
    assert_aarch64_rounds("frintn", D_1_5, D_2_0);
}

#[test]
fn frinti_uses_the_pinned_control_word_default() {
    // FPCR out of reset is round-to-nearest-ties-even, which is what
    // the lowering pins. Pinning ties-away instead would give 3.
    assert_aarch64_rounds("frinti", D_2_5, D_2_0);
}

#[test]
fn frintx_lowers_to_the_same_value_as_frinti() {
    // `frintx` differs only in being able to signal the inexact
    // exception, which the value model does not carry.
    assert_aarch64_rounds("frintx", D_2_5, D_2_0);
}

#[test]
fn frintn_at_single_precision_reads_the_register_letter() {
    // The width comes from the operand on `AArch64`: an `s` register is
    // a binary32 lane, and reading it as binary64 would take 64 bits of
    // a register holding a 32-bit pattern.
    assert_eq!(
        solve_arm_lowering(
            Arch::Aarch64,
            "frintn",
            &["s0", "s1"],
            &[("v0", PARENT_PRESET), ("v1", S_2_5)],
            S_2_0,
        ),
        SmtResult::AlwaysTrue,
        "frintn s0, s1 must round the binary32 lane"
    );
}

#[test]
fn aarch64_packed_frint_still_declines() {
    // The packed forms are out of scope, and an arranged operand must
    // keep failing closed rather than being lowered as one wide scalar.
    let insn = instruction("frinta", vec![reg("v0.4s"), reg("v1.4s")]);
    assert!(
        lift_per_mnemonic(&insn, Arch::Aarch64)
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "packed frinta must decline"
    );
}

#[test]
fn vrintn_breaks_a_tie_to_even() {
    assert_aarch32_rounds("vrintn.f32", S_2_5, S_2_0);
}

#[test]
fn vrinta_breaks_the_same_tie_away_from_zero() {
    assert_aarch32_rounds("vrinta.f32", S_2_5, S_3_0);
}

#[test]
fn vrintp_rounds_toward_positive_infinity() {
    assert_aarch32_rounds("vrintp.f32", S_2_5, S_3_0);
}

#[test]
fn vrintm_rounds_a_negative_away_from_zero() {
    assert_aarch32_rounds("vrintm.f32", S_NEG_2_5, S_NEG_3_0);
}

#[test]
fn vrintz_rounds_a_negative_toward_zero() {
    // The pair that separates truncation from toward-negative-infinity.
    assert_aarch32_rounds("vrintz.f32", S_NEG_2_5, S_NEG_2_0);
}

#[test]
fn vrintz_truncates_where_vrintn_rounds_up() {
    assert_aarch32_rounds("vrintz.f32", S_1_5, S_1_0);
}

#[test]
fn vrintr_uses_the_pinned_control_word_default() {
    // FPSCR out of reset is round-to-nearest-ties-even.
    assert_aarch32_rounds("vrintr.f32", S_2_5, S_2_0);
}

#[test]
fn vrintn_at_double_precision_reads_the_mnemonic_not_the_operand() {
    // `AArch32` spells the width on the mnemonic. `d0` is the low half
    // of the same parent `s0` views, so a lowering that took the width
    // from the register would round 64 bits of `v1` as one binary64
    // value in both cases and never be caught by the `.f32` fixtures.
    assert_eq!(
        solve_arm_lowering(
            Arch::Arm,
            "vrintn.f64",
            &["d0", "d2"],
            &[("v0", PARENT_PRESET), ("v1", D_2_5)],
            (PARENT_PRESET & !u128::from(u64::MAX)) | D_2_0,
        ),
        SmtResult::AlwaysTrue,
        "vrintn.f64 d0, d2 must round the binary64 lane and preserve d1"
    );
}

#[test]
fn aarch32_packed_vrint_rounds_every_lane_rather_than_lane_zero() {
    // `vrinta.f32 q0, q1` is the NEON form of the same mnemonic, and
    // nothing but the operand's register class separates it from the VFP
    // one. It used to decline here, on the strength of the scalar
    // handler's register-class check alone; the NEON seam now resolves
    // it ahead of that arm, so the requirement is that it lifts. What
    // each lane computes is bound against hand-computed ARM values in
    // `aarch32_round_packed_contracts.rs`.
    for operands in [["q0", "q1"], ["d0", "d2"]] {
        let insn = instruction(
            "vrinta.f32",
            operands.iter().map(|o| reg(o)).collect::<Vec<_>>(),
        );
        assert!(
            lift_per_mnemonic(&insn, Arch::Arm)
                .iter()
                .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
            "packed vrinta.f32 {operands:?} must lift"
        );
    }
}

/// Lay a straight-line body out at consecutive addresses inside one
/// function and slice back from its last instruction, which must be the
/// conditional branch.
fn slice_of(arch: Arch, body: Vec<(&str, Vec<Operand>)>) -> r2smt_slicer::Slice {
    let base = 0x40_1000;
    let mut instructions = Vec::new();
    for (index, (mnemonic, operands)) in body.into_iter().enumerate() {
        let mut insn = instruction(mnemonic, operands);
        insn.address = Address::new(base + 4 * index as u64);
        instructions.push(insn);
    }
    let program = Program {
        arch,
        bits: if matches!(arch, Arch::Arm) { 32 } else { 64 },
        entry: Some(Address::new(base)),
        functions: vec![Function {
            address: Address::new(base),
            name: Some("sym.fixture".into()),
            blocks: vec![BasicBlock {
                address: Address::new(base),
                instructions,
                successors: vec![],
            }],
            is_thumb: false,
        }],
    };
    let candidates = collect_branches(&program);
    let candidate = candidates.first().expect("fixture has a branch");
    let function = program.functions.first().expect("fixture has a function");
    slice_branch(
        candidate,
        function,
        &SliceLimits {
            allow_memory: true,
            ..SliceLimits::default()
        },
        arch,
    )
}

/// The x87 fixture: park a binary64 pattern in memory, load it onto the
/// stack, round it, store it back and compare the stored bits.
///
/// A memory round trip rather than a direct read because the x87 stack
/// is not a register file the IR can bind — it exists only inside the
/// lifter, so the only way in and out is the instruction stream.
fn x87_body(source: u128, expected: u128) -> Vec<(&'static str, Vec<Operand>)> {
    vec![
        ("mov", vec![reg("rax"), imm(&format!("{source:#x}"))]),
        ("mov", vec![mem("qword [rbp - 0x10]"), reg("rax")]),
        ("fld", vec![mem("qword [rbp - 0x10]")]),
        ("frndint", vec![]),
        ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ("mov", vec![reg("rcx"), mem("qword [rbp - 0x20]")]),
        ("cmp", vec![reg("rcx"), imm(&format!("{expected:#x}"))]),
        ("je", vec![imm("0x402000")]),
    ]
}

/// `frndint` of `source` must store `expected` back to memory.
fn assert_frndint_rounds(source: u128, expected: u128) {
    let slice = slice_of(Arch::X86_64, x87_body(source, expected));
    assert_eq!(
        slice.status,
        SliceStatus::Complete,
        "the x87 fixture must slice cleanly"
    );
    let lifted = lift_slice(&slice, Arch::X86_64);
    assert_eq!(
        solve_branch(&ssa_convert(&lifted), solve_opts()),
        SmtResult::AlwaysTrue,
        "frndint of {source:#x} must give {expected:#x}"
    );
}

#[test]
fn frndint_breaks_a_tie_to_even_under_the_pinned_control_word() {
    // The x87 control word's RC field is round-to-nearest-ties-even out
    // of reset, which the lowering pins: 2.5 becomes 2, not 3.
    assert_frndint_rounds(D_2_5, D_2_0);
}

#[test]
fn frndint_breaks_a_negative_tie_to_even_too() {
    // The negative half of the same tie, which a lowering that pinned
    // truncation would also answer -2 — hence the 1.5 fixture below.
    assert_frndint_rounds(D_NEG_2_5, D_NEG_2_0);
}

#[test]
fn frndint_rounds_one_and_a_half_up_rather_than_truncating() {
    // Separates the pinned nearest mode from round-toward-zero, which
    // ±2.5 cannot: nearest gives 2, truncation would give 1.
    assert_frndint_rounds(D_1_5, D_2_0);
}

/// The `AArch64` chain that reaches `mnemonic`'s destination through a
/// general register, optionally followed by a write to FPCR.
fn aarch64_rounding_chain(mnemonic: &str, writer: bool) -> Vec<(&str, Vec<Operand>)> {
    let mut body = vec![
        (mnemonic, vec![reg("d0"), reg("d1")]),
        ("fmov", vec![reg("x0"), reg("d0")]),
        ("cmp", vec![reg("x0"), imm("0")]),
        ("b.eq", vec![imm("0x402000")]),
    ];
    if writer {
        body.push(("msr", vec![reg("fpcr"), reg("x1")]));
    }
    body
}

#[test]
fn frinti_truncates_when_the_function_reprograms_fpcr() {
    // The decision this pins: `frinti` reads FPCR, and the lowering
    // pins the reset default rather than declining. That is only sound
    // because the slice truncates when something in the function writes
    // the control register — the walk itself cannot see it, since FPCR
    // has no register name the live set could require.
    assert!(
        matches!(
            slice_of(Arch::Aarch64, aarch64_rounding_chain("frinti", true)).status,
            SliceStatus::Truncated { .. }
        ),
        "an FPCR write must invalidate the mode `frinti` pinned"
    );
}

#[test]
fn frinti_is_complete_without_an_fpcr_write() {
    // Teeth for the guard above: untouched, the same chain resolves.
    assert_eq!(
        slice_of(Arch::Aarch64, aarch64_rounding_chain("frinti", false)).status,
        SliceStatus::Complete
    );
}

#[test]
fn frintz_survives_an_fpcr_write() {
    // The directed forms name their mode in the opcode, so they depend
    // on nothing FPCR holds and must not be dragged into the
    // truncation. Registering the whole family would cost every
    // `frint*` slice in a function that touches FPCR for no soundness
    // gain.
    assert_eq!(
        slice_of(Arch::Aarch64, aarch64_rounding_chain("frintz", true)).status,
        SliceStatus::Complete
    );
}

/// The `AArch32` counterpart: `vcmp` writes FPSCR's flags and `vmrs`
/// transfers them, giving the walk a path back to the rounded register.
fn aarch32_rounding_chain(mnemonic: &str, writer: bool) -> Vec<(&str, Vec<Operand>)> {
    let mut body = vec![
        (mnemonic, vec![reg("s0"), reg("s4")]),
        ("vcmp.f32", vec![reg("s0"), reg("s8")]),
        ("vmrs", vec![reg("APSR_nzcv"), reg("FPSCR")]),
        ("beq", vec![imm("0x402000")]),
    ];
    if writer {
        body.push(("vmsr", vec![reg("fpscr"), reg("r1")]));
    }
    body
}

#[test]
fn vrintr_truncates_when_the_function_reprograms_fpscr() {
    assert!(
        matches!(
            slice_of(Arch::Arm, aarch32_rounding_chain("vrintr.f32", true)).status,
            SliceStatus::Truncated { .. }
        ),
        "an FPSCR write must invalidate the mode `vrintr` pinned"
    );
}

#[test]
fn vrintr_is_complete_without_an_fpscr_write() {
    assert_eq!(
        slice_of(Arch::Arm, aarch32_rounding_chain("vrintr.f32", false)).status,
        SliceStatus::Complete
    );
}

#[test]
fn vrintz_survives_an_fpscr_write() {
    assert_eq!(
        slice_of(Arch::Arm, aarch32_rounding_chain("vrintz.f32", true)).status,
        SliceStatus::Complete
    );
}

#[test]
fn frndint_truncates_when_the_function_reloads_the_control_word() {
    // x87's control word is the stronger hazard of the three: it
    // carries precision control beside the rounding mode, and no fixed
    // sort can express a narrowed significand at all.
    let mut body = x87_body(D_2_5, D_2_0);
    body.push(("fldcw", vec![mem("word [rbp - 0x40]")]));
    assert!(
        matches!(
            slice_of(Arch::X86_64, body).status,
            SliceStatus::Truncated { .. }
        ),
        "`fldcw` must invalidate the mode `frndint` pinned"
    );
}
