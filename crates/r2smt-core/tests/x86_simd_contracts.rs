//! x86 packed-SIMD contracts, solved rather than asserted structurally.
//!
//! These lowerings fail by producing a *wrong value*, not by declining:
//! a compare that picks the wrong lane width, or masks with 1 instead of
//! all-ones, still yields a perfectly plausible vector. So the evidence
//! that means anything is a solver agreeing the destination equals a
//! hand-computed value.
//!
//! This is the first solver-backed x86 SIMD contract file; the existing
//! coverage asserts on IR shape in the slicer's own unit tests.
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
const VECTOR_BITS: u16 = 512;

fn operand(raw: &str) -> Operand {
    Operand {
        raw: raw.into(),
        kind: if raw.starts_with(|c: char| c.is_ascii_digit()) {
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
        mnemonic: "simdtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "simdtest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands`, bind each named vector register to a
/// concrete value, and ask the solver whether the low 128 bits of the
/// destination are necessarily `expected`.
///
/// x86 SIMD views all collapse onto a 512-bit `zmm` parent, so the
/// bindings and the expectation are stated against that parent and the
/// comparison is taken over the `xmm` view.
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
    let lifted = lift_per_mnemonic(&insn, Arch::X86_64);
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
            Expr::extract(Expr::Var(Var::new("zmm0", VECTOR_BITS)), 127, 0),
            Expr::konst(expected, 128),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::X86_64,
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

/// An all-ones mask of `count` lanes `bits` wide, with the lanes named
/// in `zeros` cleared.
///
/// Every lane an operand does not name is bound to zero, and `0 == 0`
/// matches — so the expected result of a `pcmpeq*` over a partially
/// specified vector is all-ones *except* where the test deliberately
/// mismatches.
fn mask_except(bits: u16, count: u16, zeros: &[u16]) -> u128 {
    (0..count).filter(|i| !zeros.contains(i)).fold(0, |acc, i| {
        acc | (lane_ones(bits) << (usize::from(bits) * usize::from(i)))
    })
}

fn lane_ones(bits: u16) -> u128 {
    (1u128 << bits) - 1
}

/// Pack `values` into consecutive `bits`-wide lanes, least significant
/// first.
fn packed(bits: u16, values: &[u128]) -> u128 {
    values
        .iter()
        .enumerate()
        .fold(0, |acc, (i, v)| acc | (v << (usize::from(bits) * i)))
}

#[test]
fn pcmpeqb_writes_all_ones_where_bytes_match_and_zero_where_they_do_not() {
    // The mask is all-ones per *byte*, not 1 — anything else silently
    // breaks the `pmovmskb` idiom that consumes it.
    assert_computes(
        "pcmpeqb",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0x11, 0x22, 0x33])),
            ("zmm1", packed(8, &[0x11, 0x99, 0x33])),
        ],
        mask_except(8, 16, &[1]),
    );
}

#[test]
fn pcmpeqd_compares_whole_doublewords_not_bytes() {
    // Two dwords differing only in their top byte must compare unequal.
    // A byte-wise lowering would return a partially-set mask instead.
    assert_computes(
        "pcmpeqd",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(32, &[0x0000_0001, 0x1234_5678])),
            ("zmm1", packed(32, &[0x0000_0001, 0xff34_5678])),
        ],
        mask_except(32, 4, &[1]),
    );
}

#[test]
fn pcmpgtb_is_signed() {
    // 0xff is -1 and 0x01 is 1, so `0xff > 0x01` is false signed and
    // true unsigned. x86 spells only the signed form, so reading these
    // lanes unsigned would invert every comparison against a negative
    // byte.
    assert_computes(
        "pcmpgtb",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0xff, 0x05])),
            ("zmm1", packed(8, &[0x01, 0x03])),
        ],
        packed(8, &[0x00, 0xff]),
    );
}

#[test]
fn pcmpeqb_is_a_read_modify_write_in_the_two_operand_form() {
    // The 2-operand form compares the destination against the source,
    // so binding `zmm0` has to reach the comparison rather than being
    // overwritten before it is read.
    assert_computes(
        "pcmpeqb",
        &["xmm0", "xmm1"],
        &[("zmm0", packed(8, &[0x42])), ("zmm1", packed(8, &[0x42]))],
        mask_except(8, 16, &[]),
    );
}

#[test]
fn vpcmpeqb_three_operand_form_reads_its_two_explicit_sources() {
    // The VEX form must ignore the destination's prior value: here
    // `zmm0` is bound to a value that would produce a different mask if
    // it were read as the first source.
    assert_computes(
        "vpcmpeqb",
        &["xmm0", "xmm1", "xmm2"],
        &[
            ("zmm0", packed(8, &[0x99])),
            ("zmm1", packed(8, &[0x42])),
            ("zmm2", packed(8, &[0x42])),
        ],
        mask_except(8, 16, &[]),
    );
}

/// Solve for a *general* register rather than the vector parent.
fn solve_gpr(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    gpr: &str,
    gpr_bits: u16,
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
    let lifted = lift_per_mnemonic(&insn, Arch::X86_64);
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
            Expr::Var(Var::new(gpr, gpr_bits)),
            Expr::konst(expected, gpr_bits),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::X86_64,
    };
    solve_branch(
        &ssa_convert(&slice),
        SolveOptions {
            timeout_ms: TEST_SOLVE_TIMEOUT_MS,
            ..SolveOptions::default()
        },
    )
}

#[test]
fn pmovmskb_gathers_the_sign_bit_of_every_byte() {
    // Bytes 0 and 2 have their top bit set, byte 1 does not, and the
    // remaining 13 are zero — so the mask is 0b101.
    assert_eq!(
        solve_gpr(
            "pmovmskb",
            &["eax", "xmm1"],
            &[("zmm1", packed(8, &[0x80, 0x7f, 0xff]))],
            "rax",
            64,
            0b101,
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn pmovmskb_takes_the_sign_bit_not_the_low_bit() {
    // 0x01 has its low bit set and its sign bit clear; 0x80 the
    // reverse. Sampling the wrong end would invert this mask and still
    // look entirely plausible.
    assert_eq!(
        solve_gpr(
            "pmovmskb",
            &["eax", "xmm1"],
            &[("zmm1", packed(8, &[0x01, 0x80]))],
            "rax",
            64,
            0b10,
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn movd_into_a_vector_zeroes_the_rest_of_the_register() {
    // `movd xmm0, eax` clears bits 127:32. Merging instead would leave
    // whatever the register held, which is the wrong-value failure this
    // pins.
    assert_computes(
        "movd",
        &["xmm0", "eax"],
        &[("zmm0", u128::MAX), ("rax", 0)],
        0,
    );
}

#[test]
fn movq_into_a_vector_keeps_the_low_quadword_and_zeroes_above_it() {
    assert_computes(
        "movq",
        &["xmm0", "xmm1"],
        &[("zmm0", u128::MAX), ("zmm1", 0x1234_5678_9abc_def0)],
        0x1234_5678_9abc_def0,
    );
}

#[test]
fn paddb_wraps_within_each_byte_lane() {
    // 0xff + 0x02 is 0x01 in a byte and 0x101 in anything wider. If the
    // lanes were not isolated the carry would bleed into the next one.
    assert_computes(
        "paddb",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0xff, 0x00])),
            ("zmm1", packed(8, &[0x02, 0x00])),
        ],
        packed(8, &[0x01, 0x00]),
    );
}

#[test]
fn paddq_carries_across_the_whole_quadword() {
    // The companion direction: at 64-bit lanes the same addition must
    // carry, so a lowering stuck at byte width would be visibly wrong.
    assert_computes(
        "paddq",
        &["xmm0", "xmm1"],
        &[("zmm0", 0xff), ("zmm1", 0x02)],
        0x101,
    );
}

#[test]
fn psrldq_slides_the_whole_register_by_bytes() {
    // `psrldq` is not a lane shift: it moves the entire 128-bit value
    // down by whole bytes, so byte 4 becomes byte 0.
    assert_computes(
        "psrldq",
        &["xmm0", "4"],
        &[("zmm0", packed(8, &[0, 0, 0, 0, 0xaa, 0xbb]))],
        packed(8, &[0xaa, 0xbb]),
    );
}

#[test]
fn psrld_shifts_each_doubleword_independently() {
    // The lane form, for contrast with psrldq: each dword shifts on its
    // own and nothing crosses the boundary.
    assert_computes(
        "psrld",
        &["xmm0", "4"],
        &[("zmm0", packed(32, &[0xf0, 0xff00]))],
        packed(32, &[0x0f, 0x0ff0]),
    );
}

#[test]
fn psraw_replicates_the_sign_bit() {
    // Arithmetic right shift of 0x8000 by 4 is 0xf800, not 0x0800 —
    // using a logical shift here is a wrong value, not a decline.
    assert_computes(
        "psraw",
        &["xmm0", "4"],
        &[("zmm0", packed(16, &[0x8000]))],
        packed(16, &[0xf800]),
    );
}

#[test]
fn psllw_past_the_lane_width_clears_the_lane() {
    // x86 saturates rather than masking the count: a shift of 16 on a
    // word lane yields zero, where masking to 0 would leave the lane
    // untouched.
    assert_computes(
        "psllw",
        &["xmm0", "16"],
        &[("zmm0", packed(16, &[0x1234]))],
        0,
    );
}

#[test]
fn pavgb_rounds_up_rather_than_truncating() {
    // PAVGB is (a + b + 1) >> 1, so 3 and 4 average to 4. A plain
    // halving gives 3 — a wrong value, not a decline.
    assert_computes(
        "pavgb",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0x03, 0xff])),
            ("zmm1", packed(8, &[0x04, 0xff])),
        ],
        packed(8, &[0x04, 0xff]),
    );
}

#[test]
fn paddusb_saturates_at_the_unsigned_maximum() {
    // 0xf0 + 0x20 is 0x110, which clamps to 0xff. Wrapping would give
    // 0x10 and reading the lanes signed would clamp to 0x7f.
    assert_computes(
        "paddusb",
        &["xmm0", "xmm1"],
        &[("zmm0", packed(8, &[0xf0])), ("zmm1", packed(8, &[0x20]))],
        packed(8, &[0xff]),
    );
}

#[test]
fn paddsb_saturates_at_the_signed_maximum() {
    // The same lanes read signed: 0x70 + 0x20 is 0x90 unsigned, but as
    // a signed byte it overflows and clamps to 0x7f.
    assert_computes(
        "paddsb",
        &["xmm0", "xmm1"],
        &[("zmm0", packed(8, &[0x70])), ("zmm1", packed(8, &[0x20]))],
        packed(8, &[0x7f]),
    );
}

#[test]
fn psubusb_saturates_at_zero_rather_than_wrapping() {
    // 0x10 - 0x20 underflows; unsigned saturation floors it at zero
    // where wrapping would give 0xf0.
    assert_computes(
        "psubusb",
        &["xmm0", "xmm1"],
        &[("zmm0", packed(8, &[0x10])), ("zmm1", packed(8, &[0x20]))],
        packed(8, &[0x00]),
    );
}

#[test]
fn pmaxub_and_pmaxsb_disagree_on_a_negative_byte() {
    // 0xff is 255 unsigned and -1 signed, so the unsigned max is 0xff
    // and the signed max is 0x01. Reading the wrong signedness here is
    // a plausible-looking number rather than a decline.
    let sources = [("zmm0", packed(8, &[0xff])), ("zmm1", packed(8, &[0x01]))];
    assert_computes("pmaxub", &["xmm0", "xmm1"], &sources, packed(8, &[0xff]));
    assert_computes("pmaxsb", &["xmm0", "xmm1"], &sources, packed(8, &[0x01]));
}

#[test]
fn pminsw_selects_the_signed_minimum() {
    // 0x8000 is the most negative word, so it wins a signed minimum
    // against 0x0001 — unsigned it would lose.
    assert_computes(
        "pminsw",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(16, &[0x8000])),
            ("zmm1", packed(16, &[0x0001])),
        ],
        packed(16, &[0x8000]),
    );
}

#[test]
fn pabsb_maps_the_most_negative_byte_to_itself() {
    // PABSB is non-saturating: |-128| is not representable in a signed
    // byte, and the architecture leaves the wrapped 0x80 rather than
    // clamping to 0x7f.
    assert_computes(
        "pabsb",
        &["xmm0", "xmm1"],
        &[("zmm1", packed(8, &[0x80, 0xfe]))],
        packed(8, &[0x80, 0x02]),
    );
}

#[test]
fn pabsw_writes_its_destination_from_the_source_alone() {
    // The 2-operand form is not read-modify-write: whatever xmm0 held
    // is discarded. Binding it to a value that would change the answer
    // is what gives this contract teeth.
    assert_computes(
        "pabsw",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(16, &[0x7777, 0x7777])),
            ("zmm1", packed(16, &[0xfffb, 0x0003])),
        ],
        packed(16, &[0x0005, 0x0003]),
    );
}
