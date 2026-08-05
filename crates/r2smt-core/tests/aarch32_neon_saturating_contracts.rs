//! `AArch32` NEON contracts for the families that clamp, that keep a
//! high half, and that round once over a fused multiply.
//!
//! Every family here fails by producing a *wrong number* rather than by
//! declining, and each failure is a plausible one. A doubling multiply
//! computed at the double width instead of one bit above it wraps
//! exactly at the corner it exists to saturate. A `vqdmlal` clamped once
//! at the end instead of twice lets a saturated product cancel against a
//! negative accumulator and land back inside the range. A high-half
//! narrow given the extra bit the saturating narrows need would still be
//! right, but one given a *rounding* term at the wrong place would not.
//! And `vfma` lowered as a separate multiply and add rounds twice, which
//! is the entire difference between it and the `vmla` beside it.
//!
//! So the assertions bind concrete vector values, lift the real
//! instruction, and ask the solver whether the destination is
//! *necessarily* the number the ARM definition gives — never whether the
//! IR has a particular shape.
//!
//! Each family also carries a "covers the whole view" contract. On
//! `AArch32` the scalar and packed handlers share mnemonics and are told
//! apart by the register class alone, so a packed form the NEON seam
//! fails to claim does not decline — it reaches a handler that computes
//! lane zero and leaves the rest of the register standing. Those tests
//! poison the destination with a value no lane of the correct result
//! holds, so a lowering covering part of the view cannot pass.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, InstructionKind, LiftedSlice, SliceStatus,
    analyze, lift_per_mnemonic,
};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 20_000;
/// Width of the synthetic `AArch32` vector parent every `q` / `d` view
/// is a slice of.
const VECTOR_BITS: u16 = 128;

/// Every byte of a 128-bit value set to `byte`.
fn splat_byte(byte: u8) -> u128 {
    u128::from_ne_bytes([byte; 16])
}

/// Every 16-bit lane of a 128-bit value set to `half`.
fn splat_halfword(half: u16) -> u128 {
    (0..8).fold(0, |acc, lane| acc | (u128::from(half) << (lane * 16)))
}

/// Every 32-bit lane of a 128-bit value set to `word`.
fn splat_word(word: u32) -> u128 {
    (0..4).fold(0, |acc, lane| acc | (u128::from(word) << (lane * 32)))
}

/// Classify a fixture operand the way the radare2 adapter's parser does.
fn operand(raw: &str) -> Operand {
    let kind = if raw.trim_start().starts_with('[') {
        OperandKind::Memory
    } else if raw.starts_with(|c: char| c.is_ascii_digit()) {
        OperandKind::Immediate
    } else {
        OperandKind::Register
    };
    Operand {
        raw: raw.into(),
        kind,
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

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "sattest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "sattest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands` under `Arch::Arm`, bind every named vector
/// parent to a concrete value, and ask the solver whether `v0` is
/// necessarily `expected`.
///
/// The bindings are prepended and run through the real SSA pass, so a
/// merging write reads the bound parent as its prior version rather than
/// contradicting it.
fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = instruction(mnemonic, operands);
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
            Expr::Var(Var::new("v0", VECTOR_BITS)),
            Expr::konst(expected, VECTOR_BITS),
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

fn assert_computes(mnemonic: &str, operands: &[&str], sources: &[(&str, u128)], expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

/// Whether the effect table and the lifter both refuse `mnemonic`.
///
/// Checked together on purpose: a mnemonic the slicer retains but the
/// lifter drops leaves the destination silently free.
fn declines(mnemonic: &str, operands: &[&str]) -> bool {
    let insn = instruction(mnemonic, operands);
    analyze(&insn, Arch::Arm).kind == InstructionKind::Other
        && lift_per_mnemonic(&insn, Arch::Arm)
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
}

/// The full-register three-operand shape: `q0`, `q1` and `q2` are `v0`,
/// `v1` and `v2` entire, so nothing merges.
const QUADS: [&str; 3] = ["q0", "q1", "q2"];
/// The one-source shape.
const QUAD_PAIR: [&str; 2] = ["q0", "q1"];
/// A widening shape: a `q` destination over two `d` sources that are
/// the low halves of *different* parents, so each binds independently.
const LONG: [&str; 3] = ["q0", "d2", "d4"];
/// A narrowing shape: a `d` destination — the *low* half of `v0`, so the
/// high half has to survive the write — over two `q` sources.
const NARROW: [&str; 3] = ["d0", "q1", "q2"];

/// A mask with the low 64 bits set — the low `d` half of a `q`.
const LOW_HALF_MASK: u128 = u64::MAX as u128;

/// What a `d0` destination leaves behind: `written` in the low half of
/// the parent and `poison` untouched in the high half.
///
/// An `AArch32` NEON write merges, so a lowering that zeroed the upper
/// half — which is what `AArch64` does — fails every assertion built
/// from this.
fn merged_low_half(poison: u128, written: u128) -> u128 {
    (poison & !LOW_HALF_MASK) | (written & LOW_HALF_MASK)
}

// ---------------------------------------------------------------
// `vqabs` / `vqneg` — the one value whose magnitude the element
// cannot hold
// ---------------------------------------------------------------

#[test]
fn vqabs_saturates_the_most_negative_element() {
    // `|-128|` is `128`, one past the top of a signed byte. Computing
    // the magnitude at the element's own width gives `-128` back, and a
    // clamp applied there would see a negative number and leave it
    // alone.
    assert_computes(
        "vqabs.s8",
        &QUAD_PAIR,
        &[("v0", splat_byte(0xaa)), ("v1", splat_byte(0x80))],
        splat_byte(0x7f),
    );
}

#[test]
fn vqabs_leaves_a_positive_element_alone() {
    // The teeth for the pair below: `vqabs` and `vqneg` agree on every
    // negative input and differ on every positive one, so a lowering
    // that confused them passes the saturation tests and fails here.
    assert_computes(
        "vqabs.s8",
        &QUAD_PAIR,
        &[("v0", splat_byte(0xaa)), ("v1", splat_byte(0x05))],
        splat_byte(0x05),
    );
}

#[test]
fn vqneg_negates_a_positive_element() {
    assert_computes(
        "vqneg.s8",
        &QUAD_PAIR,
        &[("v0", splat_byte(0xaa)), ("v1", splat_byte(0x05))],
        splat_byte(0xfb),
    );
}

#[test]
fn vqneg_saturates_the_most_negative_element() {
    assert_computes(
        "vqneg.s16",
        &QUAD_PAIR,
        &[
            ("v0", splat_halfword(0xdead)),
            ("v1", splat_halfword(0x8000)),
        ],
        splat_halfword(0x7fff),
    );
}

#[test]
fn vqabs_covers_the_whole_view_rather_than_one_lane() {
    // The `AArch32` scalar-handler trap. The destination is poisoned
    // with a word no lane of the correct answer holds, so a lowering
    // that wrote only the low element would leave `0xdeadbeef` standing
    // in the other three.
    assert_computes(
        "vqabs.s32",
        &QUAD_PAIR,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_word(0xffff_fffb)),
        ],
        splat_word(5),
    );
}

#[test]
fn vqabs_declines_an_unsigned_element() {
    // The architecture has no unsigned encoding: a magnitude is a
    // statement about a sign. Declining is the sound answer, and it must
    // be a decline in the effect table too.
    assert!(declines("vqabs.u8", &QUAD_PAIR));
}

// ---------------------------------------------------------------
// `vqdmulh` / `vqrdmulh` — doubling before keeping the high half
// ---------------------------------------------------------------

#[test]
fn vqdmulh_keeps_the_high_half_of_the_doubled_product() {
    // `2 * 0x4000 * 3 = 0x18000`; the high halfword of that is `1`.
    assert_computes(
        "vqdmulh.s16",
        &QUADS,
        &[
            ("v0", splat_halfword(0xdead)),
            ("v1", splat_halfword(0x4000)),
            ("v2", splat_halfword(3)),
        ],
        splat_halfword(1),
    );
}

#[test]
fn vqrdmulh_rounds_the_discarded_half_up() {
    // The teeth for the test above: same operands, and the discarded
    // low half is exactly `0x8000` — half an ulp — so the rounding form
    // must give `2` where the plain one gives `1`. A lowering that
    // dropped the rounding term, or added it after the shift instead of
    // before, passes one of this pair.
    assert_computes(
        "vqrdmulh.s16",
        &QUADS,
        &[
            ("v0", splat_halfword(0xdead)),
            ("v1", splat_halfword(0x4000)),
            ("v2", splat_halfword(3)),
        ],
        splat_halfword(2),
    );
}

#[test]
fn vqdmulh_saturates_the_most_negative_square() {
    // `INT_MIN * INT_MIN` doubled is `2^31`, whose high halfword is
    // `0x8000` — one past the element. Computed at `2n` bits rather than
    // `2n + 1` the doubling wraps to zero and the answer becomes `0`
    // instead of `INT_MAX`.
    assert_computes(
        "vqdmulh.s16",
        &QUADS,
        &[
            ("v0", splat_halfword(0xdead)),
            ("v1", splat_halfword(0x8000)),
            ("v2", splat_halfword(0x8000)),
        ],
        splat_halfword(0x7fff),
    );
}

#[test]
fn vqdmulh_covers_the_whole_view_rather_than_one_lane() {
    assert_computes(
        "vqdmulh.s32",
        &QUADS,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_word(0x4000_0000)),
            ("v2", splat_word(4)),
        ],
        splat_word(2),
    );
}

#[test]
fn vqdmulh_declines_a_byte_element() {
    // The architecture encodes halfword and word elements only.
    assert!(declines("vqdmulh.s8", &QUADS));
}

#[test]
fn vqdmulh_declines_the_by_element_form() {
    // `d2[0]` names one lane, and no resolver here understands it. The
    // decline is the sound answer — reading `d2` entire would be a wrong
    // value, not a wider one.
    assert!(declines("vqdmulh.s16", &["q0", "q1", "d2[0]"]));
}

// ---------------------------------------------------------------
// `vqdmull` / `vqdmlal` / `vqdmlsl` — the doubled product kept whole
// ---------------------------------------------------------------

#[test]
fn vqdmull_doubles_the_widened_product() {
    // `2 * 3 * 5 = 30`, in a word element over halfword sources.
    assert_computes(
        "vqdmull.s16",
        &LONG,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_halfword(3)),
            ("v2", splat_halfword(5)),
        ],
        splat_word(30),
    );
}

#[test]
fn vqdmull_keeps_a_negative_product_signed() {
    // The teeth for the widening: a source read unsigned would give
    // `2 * 65535 * 2` rather than `-4`.
    assert_computes(
        "vqdmull.s16",
        &LONG,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_halfword(0xffff)),
            ("v2", splat_halfword(2)),
        ],
        splat_word(0xffff_fffc),
    );
}

#[test]
fn vqdmull_saturates_the_most_negative_square() {
    // `2 * (-32768)^2` is `2^31`, one past a signed word.
    assert_computes(
        "vqdmull.s16",
        &LONG,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_halfword(0x8000)),
            ("v2", splat_halfword(0x8000)),
        ],
        splat_word(0x7fff_ffff),
    );
}

#[test]
fn vqdmlal_adds_the_doubled_product_onto_the_destination() {
    assert_computes(
        "vqdmlal.s16",
        &LONG,
        &[
            ("v0", splat_word(10)),
            ("v1", splat_halfword(3)),
            ("v2", splat_halfword(5)),
        ],
        splat_word(40),
    );
}

#[test]
fn vqdmlal_saturates_the_product_before_accumulating() {
    // The teeth for the *order* of the two saturations. The product
    // saturates to `INT_MAX`, and `INT_MIN + INT_MAX` is `-1`. A
    // lowering that clamped once at the end would add the unsaturated
    // `2^31` to `INT_MIN` and land on `0` — inside the range, and a
    // different number from the one the machine produces.
    assert_computes(
        "vqdmlal.s16",
        &LONG,
        &[
            ("v0", splat_word(0x8000_0000)),
            ("v1", splat_halfword(0x8000)),
            ("v2", splat_halfword(0x8000)),
        ],
        splat_word(0xffff_ffff),
    );
}

#[test]
fn vqdmlsl_subtracts_the_doubled_product_from_the_destination() {
    assert_computes(
        "vqdmlsl.s16",
        &LONG,
        &[
            ("v0", splat_word(10)),
            ("v1", splat_halfword(3)),
            ("v2", splat_halfword(5)),
        ],
        splat_word(0xffff_ffec),
    );
}

#[test]
fn vqdmull_covers_the_whole_view_rather_than_one_lane() {
    assert_computes(
        "vqdmull.s32",
        &LONG,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_word(7)),
            ("v2", splat_word(9)),
        ],
        (0x7e_u128 << 64) | 0x7e_u128,
    );
}

#[test]
fn vqdmull_declines_a_doubleword_source() {
    // A doubled word product would need a 128-bit destination element,
    // which the architecture does not encode.
    assert!(declines("vqdmull.s64", &LONG));
}

// ---------------------------------------------------------------
// `vaddhn` / `vsubhn` — the high half, and the window a wrap keeps
// ---------------------------------------------------------------

#[test]
fn vaddhn_keeps_the_high_half_of_the_sum() {
    // `0x1234 + 0x1111 = 0x2345`, whose high byte is `0x23`. The
    // destination is `d0`, the low half of `v0`, so the poisoned high
    // half has to survive: an `AArch32` NEON write merges.
    assert_computes(
        "vaddhn.i16",
        &NARROW,
        &[
            ("v0", splat_byte(0xaa)),
            ("v1", splat_halfword(0x1234)),
            ("v2", splat_halfword(0x1111)),
        ],
        merged_low_half(splat_byte(0xaa), splat_byte(0x23)),
    );
}

#[test]
fn vaddhn_keeps_the_window_a_wrapping_sum_preserves() {
    // `0xffff + 0x0002` is `0x10001` on the unbounded integer and
    // `0x0001` wrapped, and the window kept — bits `<15:8>` — is `0x00`
    // either way. This is where the family differs from the saturating
    // narrows: no headroom bit is needed, because the carry that leaves
    // the top never touches a bit the window keeps.
    assert_computes(
        "vaddhn.i16",
        &NARROW,
        &[
            ("v0", splat_byte(0xaa)),
            ("v1", splat_halfword(0xffff)),
            ("v2", splat_halfword(0x0002)),
        ],
        merged_low_half(splat_byte(0xaa), 0),
    );
}

#[test]
fn vsubhn_keeps_the_high_half_of_the_difference() {
    // The teeth for the operation: `vaddhn` over the same operands gives
    // `0x0004`, so a lowering that added where it should subtract passes
    // the test above and fails here.
    assert_computes(
        "vsubhn.i32",
        &NARROW,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_word(0x0003_0000)),
            ("v2", splat_word(0x0001_0000)),
        ],
        merged_low_half(splat_word(0xdead_beef), splat_halfword(0x0002)),
    );
}

#[test]
fn vraddhn_rounds_the_discarded_half_up() {
    // `0x1280`'s discarded low byte is exactly half an ulp, so the
    // rounding form gives `0x13` where the plain one gives `0x12`. The
    // teeth for the rounding term, and for adding it before the window
    // is taken rather than after.
    assert_computes(
        "vraddhn.i16",
        &NARROW,
        &[
            ("v0", splat_byte(0xaa)),
            ("v1", splat_halfword(0x1280)),
            ("v2", splat_halfword(0x0000)),
        ],
        merged_low_half(splat_byte(0xaa), splat_byte(0x13)),
    );
}

#[test]
fn vaddhn_does_not_round() {
    // The other half of that pair.
    assert_computes(
        "vaddhn.i16",
        &NARROW,
        &[
            ("v0", splat_byte(0xaa)),
            ("v1", splat_halfword(0x1280)),
            ("v2", splat_halfword(0x0000)),
        ],
        merged_low_half(splat_byte(0xaa), splat_byte(0x12)),
    );
}

#[test]
fn vaddhn_covers_the_whole_destination_view_rather_than_one_lane() {
    // The poison is a word no lane of the answer holds, and it must
    // survive in the *upper* half of the parent while every one of the
    // four written halfword lanes changes.
    assert_computes(
        "vaddhn.i32",
        &NARROW,
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_word(0x0007_0000)),
            ("v2", splat_word(0x0002_0000)),
        ],
        merged_low_half(splat_word(0xdead_beef), splat_halfword(0x0009)),
    );
}

#[test]
fn vaddhn_declines_a_signed_element() {
    // Keeping a high half discards the low one whatever the sign, so the
    // architecture spells this family `I` alone.
    assert!(declines("vaddhn.s16", &NARROW));
}
