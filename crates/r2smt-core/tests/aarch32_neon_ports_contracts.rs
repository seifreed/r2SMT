//! `AArch32` NEON contracts for the families ported from `AArch64`:
//! the bitwise selects, the shift inserts, the absolute differences and
//! the pairwise-long sums.
//!
//! Every one of these fails by producing a *wrong value* rather than by
//! declining, so a structural assertion would prove nothing. `vbif` and
//! `vbit` are the same expression with two registers swapped, `vabd.s8`
//! and `vabd.u8` differ by one boolean, and `vsli` against `vsri`
//! differ in which end of the element the destination keeps. Each test
//! therefore binds concrete vector values, lifts the real instruction,
//! and asks the solver whether the destination is *necessarily* the
//! number the ARM definition gives.
//!
//! Two `AArch32` facts shape the fixtures. The register file is one
//! synthetic 128-bit parent per `q`, so `q0` is `v0` and `d0` is its low
//! half while `d1` is its *high* half. And a NEON write **merges**: the
//! parent bits outside the destination's view survive, where `AArch64`
//! would zero them.
//!
//! Each family also carries a "not a scalar" contract. On `AArch32` the
//! scalar and packed handlers share mnemonics and are told apart by the
//! register class alone, so a packed form the resolver fails to claim
//! does not decline — it reaches a handler that computes lane zero and
//! leaves the rest of the register standing. Those tests poison the
//! destination with a value no lane of the correct result holds, so a
//! lowering that writes only part of the view cannot pass.
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

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
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
///
/// `AArch32` NEON shift counts matter here: radare2 prints them bare
/// (`vsli.32 q0, q1, 4`) rather than with the manual's `#`.
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
        mnemonic: "porttest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "porttest".to_string(),
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

/// A mask with the low 64 bits set — the low `d` half of a `q`.
const LOW_HALF_MASK: u128 = u64::MAX as u128;

// ---------------------------------------------------------------
// `vbsl` / `vbit` / `vbif` — which register is the mask
// ---------------------------------------------------------------

#[test]
fn vbsl_takes_its_mask_from_the_destination() {
    // `VBSL Dd, Dn, Dm` is `(Dn AND Dd) OR (Dm AND NOT Dd)`: the
    // destination selects, so the low half comes from `q1` and the high
    // half from `q2`.
    assert_computes(
        "vbsl",
        &QUADS,
        &[
            ("v0", LOW_HALF_MASK),
            ("v1", splat_byte(0x11)),
            ("v2", splat_byte(0x22)),
        ],
        (splat_byte(0x22) & !LOW_HALF_MASK) | (splat_byte(0x11) & LOW_HALF_MASK),
    );
}

#[test]
fn vbsl_selects_the_second_source_where_the_mask_is_clear() {
    // The teeth for the test above: inverting the mask has to swap the
    // halves. A lowering that read the mask's polarity backwards passes
    // one of these and fails the other.
    assert_computes(
        "vbsl",
        &QUADS,
        &[
            ("v0", !LOW_HALF_MASK),
            ("v1", splat_byte(0x11)),
            ("v2", splat_byte(0x22)),
        ],
        (splat_byte(0x11) & !LOW_HALF_MASK) | (splat_byte(0x22) & LOW_HALF_MASK),
    );
}

#[test]
fn vbit_inserts_the_first_source_where_the_second_is_set() {
    // `VBIT Dd, Dn, Dm` is `(Dn AND Dm) OR (Dd AND NOT Dm)`: the mask is
    // the *second source*, and the destination survives where it is
    // clear.
    assert_computes(
        "vbit",
        &QUADS,
        &[
            ("v0", splat_byte(0x33)),
            ("v1", splat_byte(0x11)),
            ("v2", LOW_HALF_MASK),
        ],
        (splat_byte(0x33) & !LOW_HALF_MASK) | (splat_byte(0x11) & LOW_HALF_MASK),
    );
}

#[test]
fn vbif_inserts_the_first_source_where_the_second_is_clear() {
    // The teeth for `vbit`: same three registers, same mask, and the
    // two halves have to come out swapped. `vbif` and `vbit` are one
    // expression with two operands exchanged, so a lowering that mixed
    // them up passes one of this pair.
    assert_computes(
        "vbif",
        &QUADS,
        &[
            ("v0", splat_byte(0x33)),
            ("v1", splat_byte(0x11)),
            ("v2", LOW_HALF_MASK),
        ],
        (splat_byte(0x11) & !LOW_HALF_MASK) | (splat_byte(0x33) & LOW_HALF_MASK),
    );
}

#[test]
fn vbsl_covers_the_whole_view_rather_than_one_lane() {
    // The `AArch32` scalar-handler trap. A packed form the NEON seam
    // fails to claim reaches a handler that computes the low element and
    // leaves the rest of the register standing, which is a wrong value
    // and not a decline. An all-ones mask makes the answer `q1` entire,
    // so any byte still holding the poisoned destination fails.
    assert_computes(
        "vbsl",
        &QUADS,
        &[
            ("v0", u128::MAX),
            ("v1", splat_byte(0x5a)),
            ("v2", splat_byte(0xa5)),
        ],
        splat_byte(0x5a),
    );
}

// ---------------------------------------------------------------
// `vsli` / `vsri` — which end of the element the destination keeps
// ---------------------------------------------------------------

#[test]
fn vsli_keeps_the_destination_bits_below_the_shift() {
    // `(0x12345678 << 4) | (0x0000000a & 0xf)` in every 32-bit lane.
    assert_computes(
        "vsli.32",
        &["q0", "q1", "4"],
        &[
            ("v0", splat_word(0x0000_000a)),
            ("v1", splat_word(0x1234_5678)),
        ],
        splat_word(0x2345_678a),
    );
}

#[test]
fn vsri_keeps_the_destination_bits_above_the_shift() {
    // The teeth for `vsli`: `vsri` shifts the other way and preserves
    // the *high* nibble of the destination. Its low nibble is set too,
    // so a lowering that kept the wrong end gives `0x0123456b`.
    assert_computes(
        "vsri.32",
        &["q0", "q1", "4"],
        &[
            ("v0", splat_word(0xa000_000b)),
            ("v1", splat_word(0x1234_5678)),
        ],
        splat_word(0xa123_4567),
    );
}

#[test]
fn vsri_at_the_element_width_keeps_the_destination_entire() {
    // The top of `VSRI`'s encoded range, which `VSLI` does not have: a
    // shift equal to the element width vacates nothing, so every bit of
    // the result is the destination's. A lowering that let the shift
    // amount wrap to zero would answer with the source instead.
    assert_computes(
        "vsri.32",
        &["q0", "q1", "32"],
        &[
            ("v0", splat_word(0xdead_beef)),
            ("v1", splat_word(0x1234_5678)),
        ],
        splat_word(0xdead_beef),
    );
}

#[test]
fn vsli_covers_every_lane_rather_than_one() {
    // The scalar-handler trap for the shift inserts. The destination is
    // poisoned with a word no correct lane holds.
    assert_computes(
        "vsli.32",
        &["q0", "q1", "8"],
        &[
            ("v0", splat_word(0xffff_ffff)),
            ("v1", splat_word(0x0000_0001)),
        ],
        splat_word(0x0000_01ff),
    );
}

// ---------------------------------------------------------------
// `vabd` / `vaba` — the difference needs a bit the element has not
// ---------------------------------------------------------------

#[test]
fn vabd_signed_computes_the_magnitude_one_bit_wider_than_the_element() {
    // `|(-128) - 127|` is `255`, which no signed byte holds. Computing
    // the difference at the element width wraps it to `1`, and the
    // answer is the unsigned byte `0xff`.
    assert_computes(
        "vabd.s8",
        &QUADS,
        &[("v1", splat_byte(0x80)), ("v2", splat_byte(0x7f))],
        splat_byte(0xff),
    );
}

#[test]
fn vabd_unsigned_reads_the_same_bytes_as_magnitudes() {
    // The teeth for the test above: one boolean apart, the same bytes
    // are `128` and `127` and the difference is `1`.
    assert_computes(
        "vabd.u8",
        &QUADS,
        &[("v1", splat_byte(0x80)), ("v2", splat_byte(0x7f))],
        splat_byte(0x01),
    );
}

#[test]
fn vaba_adds_the_magnitude_onto_the_destination() {
    // `|10 - 3| + 5` is `12`. Overwriting the destination instead of
    // accumulating gives `7`.
    assert_computes(
        "vaba.u8",
        &QUADS,
        &[
            ("v0", splat_byte(0x05)),
            ("v1", splat_byte(0x0a)),
            ("v2", splat_byte(0x03)),
        ],
        splat_byte(0x0c),
    );
}

#[test]
fn vaba_wraps_the_accumulation_at_the_element_width() {
    // `0xff + 2` is `0x101`, and `VABA` is not in the saturating family:
    // the byte keeps `0x01`. A lowering that clamped would answer
    // `0xff`.
    assert_computes(
        "vaba.s8",
        &QUADS,
        &[
            ("v0", splat_byte(0x02)),
            ("v1", splat_byte(0x80)),
            ("v2", splat_byte(0x7f)),
        ],
        splat_byte(0x01),
    );
}

#[test]
fn vabd_into_a_doubleword_preserves_the_other_half_of_the_register() {
    // An `AArch32` NEON write merges. `d0` is the low half of `v0`, so
    // the high half has to survive untouched — on `AArch64` the same
    // instruction would zero it.
    let untouched = 0x1234_5678_9abc_def0_u128 << 64;
    assert_computes(
        "vabd.u8",
        &["d0", "d2", "d4"],
        &[
            ("v0", untouched | splat_byte(0xee) & LOW_HALF_MASK),
            ("v1", splat_byte(0x0a)),
            ("v2", splat_byte(0x03)),
        ],
        untouched | (splat_byte(0x07) & LOW_HALF_MASK),
    );
}

#[test]
fn vabd_covers_every_lane_rather_than_one() {
    // The scalar-handler trap for the absolute differences.
    assert_computes(
        "vabd.u8",
        &QUADS,
        &[
            ("v0", u128::MAX),
            ("v1", splat_byte(0x0a)),
            ("v2", splat_byte(0x03)),
        ],
        splat_byte(0x07),
    );
}

// ---------------------------------------------------------------
// `vpaddl` / `vpadal` — the sum is defined at twice the source element
// ---------------------------------------------------------------

#[test]
fn vpaddl_signed_extends_each_source_byte_before_summing() {
    // Two bytes of `0xff` are `-1` each, so every halfword is `-2`.
    assert_computes(
        "vpaddl.s8",
        &QUAD_PAIR,
        &[("v1", splat_byte(0xff))],
        splat_halfword(0xfffe),
    );
}

#[test]
fn vpaddl_unsigned_reads_the_same_bytes_as_magnitudes() {
    // The teeth for the test above: `255 + 255` is `510`, which needs
    // the destination's full sixteen bits and is where the family's
    // widening earns its name.
    assert_computes(
        "vpaddl.u8",
        &QUAD_PAIR,
        &[("v1", splat_byte(0xff))],
        splat_halfword(0x01fe),
    );
}

#[test]
fn vpadal_adds_the_pairwise_sum_onto_the_destination() {
    // `510 + 1` per halfword. Overwriting instead of accumulating gives
    // `0x01fe`, which the test above already pins.
    assert_computes(
        "vpadal.u8",
        &QUAD_PAIR,
        &[("v0", splat_halfword(0x0001)), ("v1", splat_byte(0xff))],
        splat_halfword(0x01ff),
    );
}

#[test]
fn vpaddl_covers_every_lane_rather_than_one() {
    // The scalar-handler trap for the pairwise-long family.
    assert_computes(
        "vpaddl.u8",
        &QUAD_PAIR,
        &[("v0", u128::MAX), ("v1", splat_byte(0x02))],
        splat_halfword(0x0004),
    );
}

// ---------------------------------------------------------------
// The shapes that stay declined
// ---------------------------------------------------------------

#[test]
fn bitwise_select_declines_a_mismatched_register_class() {
    // The mask and the candidates have to name the same view, or the
    // lowering would combine a 64-bit value with a 128-bit one.
    assert!(declines("vbsl", &["q0", "q1", "d4"]));
    assert!(declines("vbit", &["d0", "q1", "q2"]));
}

#[test]
fn shift_insert_declines_a_shift_outside_its_encoded_range() {
    // `VSLI` shifts by `0 .. esize-1` and `VSRI` by `1 .. esize`. Each
    // rejects the other's endpoint.
    assert!(declines("vsli.32", &["q0", "q1", "32"]));
    assert!(declines("vsri.32", &["q0", "q1", "0"]));
}

#[test]
fn shift_insert_declines_a_register_shift_amount() {
    // Both forms take an immediate. A vector register there would be
    // read as a shift count, which is not an instruction.
    assert!(declines("vsli.32", &QUADS));
}

#[test]
fn shift_insert_declines_a_typed_element_spelling() {
    // The insertion is bit-granular, so the architecture spells the
    // element as a bare width and gives it no signedness to carry.
    assert!(declines("vsli.s32", &["q0", "q1", "4"]));
    assert!(declines("vsri.u16", &["q0", "q1", "4"]));
}

#[test]
fn absolute_difference_declines_the_untyped_element_spelling() {
    // `|a - b|` is not sign-agnostic the way add and multiply are, so
    // the architecture has no `.i` encoding for it.
    assert!(declines("vabd.i8", &QUADS));
    assert!(declines("vaba.i16", &QUADS));
}

#[test]
fn absolute_difference_declines_the_float_element_spelling() {
    // `VABD.F32` exists — `FPAbs(FPSub(a, b))` — and is a different lane
    // path this seam does not model. Declining is sound; the scalar VFP
    // table does not name `vabd` either, so nothing else claims it.
    assert!(declines("vabd.f32", &QUADS));
}

#[test]
fn absolute_difference_declines_a_doubleword_element() {
    // There is no `VABD.S64`.
    assert!(declines("vabd.s64", &QUADS));
}

#[test]
fn pairwise_long_declines_a_doubleword_source_element() {
    // A doubleword source would need a 128-bit destination element,
    // which the register file has no room for.
    assert!(declines("vpaddl.s64", &QUAD_PAIR));
    assert!(declines("vpadal.u64", &QUAD_PAIR));
}

#[test]
fn pairwise_long_declines_a_mismatched_register_class() {
    // Both operands name the same view: the element doubles, so half as
    // many of them fill the same register.
    assert!(declines("vpaddl.u8", &["q0", "d2"]));
}
