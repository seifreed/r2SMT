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
    solve_lowering_at(mnemonic, operands, sources, expected, 0)
}

/// As [`solve_lowering`], but comparing the 128-bit block starting at
/// `lo` rather than the low one — the only way to state a contract about
/// the upper half of a 256-bit view.
fn solve_lowering_at(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    expected: u128,
    lo: u16,
) -> SmtResult {
    let bindings: Vec<(&str, Expr)> = sources
        .iter()
        .map(|(name, value)| (*name, Expr::konst(*value, VECTOR_BITS)))
        .collect();
    solve_with_bindings(mnemonic, operands, &bindings, expected, lo)
}

/// A `VECTOR_BITS`-wide constant whose two low 128-bit blocks are `low`
/// and `high`.
///
/// Needed because a 256-bit source does not fit the `u128` the other
/// helpers bind, and the block-confinement contracts are exactly the
/// ones that need data in the upper block.
fn two_blocks(low: u128, high: u128) -> Expr {
    Expr::concat(
        Expr::konst(0, VECTOR_BITS - 256),
        Expr::concat(Expr::konst(high, 128), Expr::konst(low, 128)),
    )
}

fn solve_with_bindings(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, Expr)],
    expected: u128,
    lo: u16,
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
            src: value.clone(),
        })
        .collect();
    statements.extend(lifted);

    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::extract(Expr::Var(Var::new("zmm0", VECTOR_BITS)), lo + 127, lo),
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

#[test]
fn punpcklbw_interleaves_the_low_bytes_of_both_sources() {
    // Destination byte 2i comes from the first source and 2i+1 from the
    // second, both from the *low* half. Concatenating the halves
    // instead would put all of xmm0's bytes below all of xmm1's.
    assert_computes(
        "punpcklbw",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0x11, 0x22, 0x33, 0x44])),
            ("zmm1", packed(8, &[0xaa, 0xbb, 0xcc, 0xdd])),
        ],
        packed(8, &[0x11, 0xaa, 0x22, 0xbb, 0x33, 0xcc, 0x44, 0xdd]),
    );
}

#[test]
fn punpckhqdq_takes_the_upper_quadword_of_each_source() {
    // The high form reads the top half of each source, so the result is
    // xmm0's upper quadword under xmm1's.
    assert_computes(
        "punpckhqdq",
        &["xmm0", "xmm1"],
        &[
            (
                "zmm0",
                packed(64, &[0x1111_1111_1111_1111, 0x2222_2222_2222_2222]),
            ),
            (
                "zmm1",
                packed(64, &[0x3333_3333_3333_3333, 0x4444_4444_4444_4444]),
            ),
        ],
        packed(64, &[0x2222_2222_2222_2222, 0x4444_4444_4444_4444]),
    );
}

#[test]
fn pshufd_selects_each_doubleword_by_its_immediate_field() {
    // 0x1b is 00_01_10_11, so the destination doublewords come from
    // source doublewords 3, 2, 1, 0 — a reversal.
    assert_computes(
        "pshufd",
        &["xmm0", "xmm1", "0x1b"],
        &[(
            "zmm1",
            packed(32, &[0x0000_000a, 0x0000_000b, 0x0000_000c, 0x0000_000d]),
        )],
        packed(32, &[0x0000_000d, 0x0000_000c, 0x0000_000b, 0x0000_000a]),
    );
}

#[test]
fn pshufd_writes_its_destination_from_the_source_alone() {
    // Three operands, but the third is the selector — this is not a
    // read-modify-write, so whatever xmm0 held is discarded.
    assert_computes(
        "pshufd",
        &["xmm0", "xmm1", "0x00"],
        &[
            (
                "zmm0",
                packed(32, &[0x7777_7777, 0x7777_7777, 0x7777_7777, 0x7777_7777]),
            ),
            ("zmm1", packed(32, &[0x0000_0005, 0, 0, 0])),
        ],
        packed(32, &[5, 5, 5, 5]),
    );
}

#[test]
fn pshuflw_leaves_the_high_quadword_standing() {
    // Only the low four words are permuted; the top four are copied
    // verbatim. Permuting all eight would be a wrong value.
    assert_computes(
        "pshuflw",
        &["xmm0", "xmm1", "0x1b"],
        &[("zmm1", packed(16, &[1, 2, 3, 4, 5, 6, 7, 8]))],
        packed(16, &[4, 3, 2, 1, 5, 6, 7, 8]),
    );
}

#[test]
fn pshufhw_permutes_only_the_high_quadword() {
    // The mirror image: the low four words are copied and the immediate
    // selects among the high four, whose indices start at 4.
    assert_computes(
        "pshufhw",
        &["xmm0", "xmm1", "0x1b"],
        &[("zmm1", packed(16, &[1, 2, 3, 4, 5, 6, 7, 8]))],
        packed(16, &[1, 2, 3, 4, 8, 7, 6, 5]),
    );
}

#[test]
fn vpshufd_permutes_each_128_bit_block_independently() {
    // The load-bearing property of the 256-bit forms: AVX widened these
    // by running the same 128-bit permutation in each half, so a
    // selector never reaches across the halfway line. Reading a ymm as
    // one flat vector of eight doublewords would move data between
    // halves — a wrong value rather than a decline.
    //
    // Source doublewords are 0..8 and the selector 0x1b reverses each
    // group of four, so the low block reverses 0,1,2,3 and the upper
    // block reverses 4,5,6,7 *within itself*.
    let source = [(
        "zmm1",
        two_blocks(packed(32, &[0, 1, 2, 3]), packed(32, &[4, 5, 6, 7])),
    )];
    let operands = ["ymm0", "ymm1", "0x1b"];
    assert_eq!(
        solve_with_bindings("vpshufd", &operands, &source, packed(32, &[3, 2, 1, 0]), 0),
        SmtResult::AlwaysTrue,
        "vpshufd should reverse the low block within itself"
    );
    assert_eq!(
        solve_with_bindings(
            "vpshufd",
            &operands,
            &source,
            packed(32, &[7, 6, 5, 4]),
            128
        ),
        SmtResult::AlwaysTrue,
        "vpshufd should reverse the upper block within itself"
    );
}

#[test]
fn vpunpcklqdq_interleaves_within_each_128_bit_block() {
    // Same block confinement for the interleaves: the upper block pairs
    // the two sources' upper-block low quadwords, not their overall
    // ones.
    let sources = [
        (
            "zmm1",
            two_blocks(packed(64, &[0x0a, 0x0b]), packed(64, &[0x0c, 0x0d])),
        ),
        (
            "zmm2",
            two_blocks(packed(64, &[0x1a, 0x1b]), packed(64, &[0x1c, 0x1d])),
        ),
    ];
    let operands = ["ymm0", "ymm1", "ymm2"];
    assert_eq!(
        solve_with_bindings(
            "vpunpcklqdq",
            &operands,
            &sources,
            packed(64, &[0x0a, 0x1a]),
            0
        ),
        SmtResult::AlwaysTrue,
        "vpunpcklqdq should pair the low quadwords of each source's low block"
    );
    assert_eq!(
        solve_with_bindings(
            "vpunpcklqdq",
            &operands,
            &sources,
            packed(64, &[0x0c, 0x1c]),
            128
        ),
        SmtResult::AlwaysTrue,
        "vpunpcklqdq should pair the low quadwords of each source's upper block"
    );
}

#[test]
fn pshufb_zeroes_a_byte_whose_index_has_bit_seven_set() {
    // The clear-on-negative rule is the whole difference between
    // `pshufb` and an ordinary table lookup: an index of 0x80 writes
    // zero rather than selecting source byte 0.
    //
    // Every index byte the test does not name is zero, and a zero index
    // *selects source byte 0* rather than writing zero — so the tail of
    // the expectation is 0xaa, not 0x00. Getting that wrong looks
    // exactly like a lifter bug and is not one.
    let mut expected = vec![0xcc, 0x00];
    expected.resize(16, 0xaa);
    assert_computes(
        "pshufb",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0xaa, 0xbb, 0xcc])),
            ("zmm1", packed(8, &[0x02, 0x80, 0x00])),
        ],
        packed(8, &expected),
    );
}

#[test]
fn pshufb_uses_only_the_low_four_index_bits() {
    // Index 0x11 selects source byte 1, not byte 17 — the masking is
    // what confines the lookup to its own block.
    // The unnamed index bytes are zero and so select source byte 0.
    let mut expected = vec![0x11];
    expected.resize(16, 0x10);
    assert_computes(
        "pshufb",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(8, &[0x10, 0x11, 0x12])),
            ("zmm1", packed(8, &[0x11])),
        ],
        packed(8, &expected),
    );
}

#[test]
fn packuswb_saturates_a_negative_word_to_zero() {
    // The trap in the family: PACKUSWB reads its sources as *signed*
    // words and saturates them into the *unsigned* byte range, so
    // 0xffff (-1) becomes 0x00. Reading the source unsigned would give
    // 0xff, and clamping to the signed byte range would give 0x80.
    // 0x0140 (320) clamps to 0xff at the top of that range.
    assert_computes(
        "packuswb",
        &["xmm0", "xmm1"],
        &[("zmm0", packed(16, &[0xffff, 0x0140, 0x0012])), ("zmm1", 0)],
        packed(8, &[0x00, 0xff, 0x12]),
    );
}

#[test]
fn packsswb_saturates_into_the_signed_byte_range() {
    // The same lanes under signed destination bounds: -1 stays -1
    // (0xff as a byte) and 320 clamps to 0x7f.
    assert_computes(
        "packsswb",
        &["xmm0", "xmm1"],
        &[("zmm0", packed(16, &[0xffff, 0x0140, 0x0012])), ("zmm1", 0)],
        packed(8, &[0xff, 0x7f, 0x12]),
    );
}

#[test]
fn packssdw_fills_the_low_half_from_the_first_source_and_the_high_from_the_second() {
    // The operand order is observable: the first source's four
    // doublewords become the low four words and the second's the high
    // four.
    assert_computes(
        "packssdw",
        &["xmm0", "xmm1"],
        &[
            ("zmm0", packed(32, &[0x0000_0001, 0x0000_0002, 0, 0])),
            ("zmm1", packed(32, &[0x0000_0003, 0x0000_0004, 0, 0])),
        ],
        packed(16, &[1, 2, 0, 0, 3, 4, 0, 0]),
    );
}

/// Solve `ptest`'s flag output: bind the two vectors and ask whether the
/// named flag is necessarily `expected`.
fn assert_ptest_flag(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    flag: &str,
    expected: u128,
) {
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
        condition: Expr::eq(Expr::Var(Var::new(flag, 1)), Expr::konst(expected, 1)),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::X86_64,
    };
    assert_eq!(
        solve_branch(
            &ssa_convert(&slice),
            SolveOptions {
                timeout_ms: TEST_SOLVE_TIMEOUT_MS,
                ..SolveOptions::default()
            },
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should leave {flag} = {expected}"
    );
}

#[test]
fn ptest_sets_zf_when_the_operands_share_no_bits() {
    // ZF is the AND being zero, which is the whole point of the
    // `ptest xmm0, xmm0 ; jz` idiom this exists to resolve.
    assert_ptest_flag(
        "ptest",
        &["xmm0", "xmm1"],
        &[("zmm0", 0x00ff), ("zmm1", 0xff00)],
        "ZF",
        1,
    );
    assert_ptest_flag(
        "ptest",
        &["xmm0", "xmm1"],
        &[("zmm0", 0x0ff0), ("zmm1", 0xff00)],
        "ZF",
        0,
    );
}

#[test]
fn ptest_of_a_register_against_itself_tests_that_register_for_zero() {
    // The common idiom: `ptest xmm0, xmm0` sets ZF exactly when xmm0 is
    // all zero, because `x AND x` is `x`.
    assert_ptest_flag("ptest", &["xmm0", "xmm0"], &[("zmm0", 0)], "ZF", 1);
    assert_ptest_flag("ptest", &["xmm0", "xmm0"], &[("zmm0", 0x8000)], "ZF", 0);
}

#[test]
fn ptest_complements_its_first_operand_for_cf_not_its_second() {
    // CF is `src AND NOT dst`. With dst = 0x00ff and src = 0xff00,
    // NOT dst covers every bit of src, so CF is 0; swapping the roles
    // gives NOT dst = ~0xff00, which shares no bit with 0x00ff... it
    // does, so CF flips. Reversing the complemented operand is a wrong
    // flag rather than a decline.
    assert_ptest_flag(
        "ptest",
        &["xmm0", "xmm1"],
        &[("zmm0", 0x00ff), ("zmm1", 0xff00)],
        "CF",
        0,
    );
    assert_ptest_flag(
        "ptest",
        &["xmm1", "xmm0"],
        &[("zmm0", 0x00ff), ("zmm1", 0xff00)],
        "CF",
        0,
    );
    // A case where the two orders genuinely disagree: src is a subset
    // of dst, so `src AND NOT dst` is empty one way round and not the
    // other.
    assert_ptest_flag(
        "ptest",
        &["xmm0", "xmm1"],
        &[("zmm0", 0x00ff), ("zmm1", 0x000f)],
        "CF",
        1,
    );
    assert_ptest_flag(
        "ptest",
        &["xmm1", "xmm0"],
        &[("zmm0", 0x00ff), ("zmm1", 0x000f)],
        "CF",
        0,
    );
}

#[test]
fn ptest_clears_the_flags_it_does_not_compute() {
    for flag in ["OF", "SF", "PF"] {
        assert_ptest_flag(
            "ptest",
            &["xmm0", "xmm1"],
            &[("zmm0", 0x1234), ("zmm1", 0x5678)],
            flag,
            0,
        );
    }
}
