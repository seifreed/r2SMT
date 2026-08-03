//! `AArch64` NEON contracts for the families that resolve their
//! geometry from something other than operand 0.
//!
//! These are the lowerings whose failure mode is a *wrong value* rather
//! than a decline: an across-lane reduction that reads the destination's
//! width as the lane width, a by-element form that picks the wrong lane,
//! a NaN-aware select that reuses x86's `MAXPS` ordering. None of that
//! shows up in a structural assertion on the emitted IR, so each test
//! here binds concrete lanes, lifts the real instruction, and solves the
//! destination against a value computed by hand from the ARM definition.
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

fn operand(raw: &str) -> Operand {
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
        mnemonic: "neontail".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "neontail".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `mnemonic operands`, bind every named register to a concrete
/// vector value, and ask the solver whether the destination is
/// necessarily `expected`.
///
/// The binding assignments run through the real SSA pass ahead of the
/// lifted statements, so an instruction that *reads* its destination
/// sees the bound value rather than contradicting it.
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
    let lifted = lift_per_mnemonic(&insn, Arch::Aarch64);
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

/// Assert the lowering computes exactly `expected` for these inputs.
fn assert_computes(mnemonic: &str, operands: &[&str], sources: &[(&str, u128)], expected: u128) {
    assert_eq!(
        solve_lowering(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

// ===================== across-lane reductions =====================

/// Pack `lanes` little-endian into one vector value.
fn packed(lane_bits: u32, lanes: &[u128]) -> u128 {
    let mut value = 0u128;
    let mut offset = 0u32;
    for lane in lanes {
        value |= lane << offset;
        offset += lane_bits;
    }
    value
}

#[test]
fn addv_sums_every_lane_of_the_source_arrangement() {
    // 1 + 2 + 3 + 4. The destination is `s0`, so a resolver that took
    // the lane width from operand 0 would read one 32-bit lane and stop.
    assert_computes(
        "addv",
        &["s0", "v1.4s"],
        &[("v1", packed(32, &[1, 2, 3, 4]))],
        10,
    );
}

#[test]
fn addv_keeps_the_low_bits_of_an_overflowing_sum() {
    // 0xff + 0x02 is 0x101 in eight bits, and ARM truncates the
    // unbounded sum into the element — a widened result would be wrong.
    assert_computes(
        "addv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0xff, 0x02]))],
        0x01,
    );
}

#[test]
fn uaddlv_widens_before_summing() {
    // Eight lanes of 0xff sum to 2 040, which does not fit the source
    // element. The `l` in the mnemonic is exactly that: the destination
    // is twice as wide, and the sum is exact rather than truncated.
    assert_computes(
        "uaddlv",
        &["h0", "v1.8b"],
        &[("v1", packed(8, &[0xff; 8]))],
        0x7f8,
    );
}

#[test]
fn saddlv_sign_extends_each_lane_before_summing() {
    // One lane of 0xff is -1 signed, so the 16-bit sum is 0xffff. Read
    // unsigned it would be 0xff — the same bits, a different number.
    assert_computes(
        "saddlv",
        &["h0", "v1.8b"],
        &[("v1", packed(8, &[0xff]))],
        0xffff,
    );
}

#[test]
fn smaxv_compares_lanes_signed() {
    // 0x80 is -128, so the signed maximum of the two non-zero lanes is
    // 0x7f — and every other lane is zero, which does not beat it.
    assert_computes(
        "smaxv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x7f,
    );
}

#[test]
fn umaxv_compares_the_same_lanes_unsigned() {
    assert_computes(
        "umaxv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x80,
    );
}

#[test]
fn sminv_compares_lanes_signed() {
    assert_computes(
        "sminv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x80,
    );
}

#[test]
fn uminv_compares_the_same_lanes_unsigned() {
    // The six zero lanes are the unsigned minimum, which is what makes
    // this the mirror of `sminv` rather than a restatement of it.
    assert_computes(
        "uminv",
        &["b0", "v1.8b"],
        &[("v1", packed(8, &[0x7f, 0x80]))],
        0x00,
    );
}

// ===================== `movi` with `msl`, and fixed point =====================

/// [`solve_lowering`] for an instruction carrying immediate operands,
/// which the all-register helper would misclassify.
fn solve_mixed(
    mnemonic: &str,
    operands: &[(&str, OperandKind)],
    sources: &[(&str, u128)],
    expected: u128,
) -> SmtResult {
    let insn = Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands
            .iter()
            .map(|(raw, kind)| Operand {
                raw: (*raw).into(),
                kind: *kind,
            })
            .collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    };
    let lifted = lift_per_mnemonic(&insn, Arch::Aarch64);
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

fn assert_computes_mixed(
    mnemonic: &str,
    operands: &[(&str, OperandKind)],
    sources: &[(&str, u128)],
    expected: u128,
) {
    assert_eq!(
        solve_mixed(mnemonic, operands, sources, expected),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} with {sources:x?} should give {expected:#x}"
    );
}

const REG: OperandKind = OperandKind::Register;
const IMM: OperandKind = OperandKind::Immediate;

#[test]
fn movi_with_a_ones_shift_fills_the_vacated_bits_with_ones() {
    // `msl 8` on an immediate of 1 gives 0x1ff, not 0x100 — the whole
    // difference from `lsl`, and the reason it is a separate mnemonic.
    assert_computes_mixed(
        "movi",
        &[("v0.4s", REG), ("1", IMM), ("msl 8", IMM)],
        &[],
        packed(32, &[0x1ff; 4]),
    );
}

#[test]
fn mvni_inverts_the_ones_shifted_immediate() {
    assert_computes_mixed(
        "mvni",
        &[("v0.4s", REG), ("1", IMM), ("msl 8", IMM)],
        &[],
        packed(32, &[0xffff_fe00; 4]),
    );
}

#[test]
fn movi_with_a_zeroes_shift_still_fills_with_zeroes() {
    // The regression guard on the pre-existing `lsl` path: it shares
    // the parser with `msl` now, and must not have picked up its fill.
    assert_computes_mixed(
        "movi",
        &[("v0.4s", REG), ("1", IMM), ("lsl 8", IMM)],
        &[],
        packed(32, &[0x100; 4]),
    );
}

#[test]
fn scvtf_with_a_fraction_width_divides_by_that_power_of_two() {
    // 3 read as a fixed-point value with one fractional bit is 1.5,
    // whose binary32 pattern is 0x3fc00000. Without the scale it would
    // be 3.0 (0x40400000) — a different number, not a rounding.
    assert_computes_mixed(
        "scvtf",
        &[("v0.4s", REG), ("v1.4s", REG), ("1", IMM)],
        &[("v1", 3)],
        0x3fc0_0000,
    );
}

#[test]
fn ucvtf_with_a_fraction_width_reads_the_lane_unsigned() {
    // 0xffffffff over four is 1073741823.75, which rounds to 2^30
    // (0x4e800000). Read signed the same lane is -0.25 (0xbe800000),
    // so this pins the signedness as well as the scale.
    assert_computes_mixed(
        "ucvtf",
        &[("v0.4s", REG), ("v1.4s", REG), ("2", IMM)],
        &[("v1", 0xffff_ffff)],
        0x4e80_0000,
    );
}

#[test]
fn fcvtzs_with_a_fraction_width_multiplies_before_truncating() {
    // 1.5 with two fractional bits is the integer 6. Truncating first
    // and scaling after would give 4.
    assert_computes_mixed(
        "fcvtzs",
        &[("v0.4s", REG), ("v1.4s", REG), ("2", IMM)],
        &[("v1", 0x3fc0_0000)],
        6,
    );
}

#[test]
fn scvtf_scales_a_half_precision_lane_through_a_subnormal_factor() {
    // The corner that decides how the scale is built: 2^16 is infinity
    // in binary16, so a lowering that divided by `2^fbits` would give
    // zero here. Multiplying by 2^-16 — a representable subnormal —
    // gives 1/65536, whose binary16 pattern is 0x0100.
    assert_computes_mixed(
        "scvtf",
        &[("v0.4h", REG), ("v1.4h", REG), ("16", IMM)],
        &[("v1", 1)],
        0x0100,
    );
}

// ===================== by-element and dot products =====================

#[test]
fn mul_by_element_broadcasts_the_named_lane() {
    // The element is `v2` lane 1, which is 20. Picking lane 0 instead
    // would give a perfectly plausible [10, 20, 30, 40] — a wrong value
    // rather than a decline, which is why this is solved and not
    // asserted structurally.
    assert_computes(
        "mul",
        &["v0.4s", "v1.4s", "v2.s[1]"],
        &[
            ("v1", packed(32, &[1, 2, 3, 4])),
            ("v2", packed(32, &[10, 20, 30, 40])),
        ],
        packed(32, &[20, 40, 60, 80]),
    );
}

#[test]
fn umlal_by_element_accumulates_onto_the_destination() {
    // Each lane is its prior value plus the product of the halfword
    // source lane and `v2` lane 3.
    assert_computes(
        "umlal",
        &["v0.4s", "v1.4h", "v2.h[3]"],
        &[
            ("v0", packed(32, &[1, 1, 1, 1])),
            ("v1", packed(16, &[2, 3, 4, 5])),
            ("v2", packed(16, &[0, 0, 0, 7])),
        ],
        packed(32, &[15, 22, 29, 36]),
    );
}

#[test]
fn umlal2_by_element_reads_the_first_source_upper_half() {
    // The low four halfwords are 100 apiece precisely so that reading
    // them would be visible: the `2` suffix takes lanes 4..7.
    assert_computes(
        "umlal2",
        &["v0.4s", "v1.8h", "v2.h[3]"],
        &[
            ("v0", 0),
            ("v1", packed(16, &[100, 100, 100, 100, 2, 3, 4, 5])),
            ("v2", packed(16, &[0, 0, 0, 7])),
        ],
        packed(32, &[14, 21, 28, 35]),
    );
}

#[test]
fn smlal_by_element_sign_extends_both_narrow_sources() {
    // The element is 0xffff, which is -1. Read unsigned the product
    // would be 131070 rather than -2.
    assert_computes(
        "smlal",
        &["v0.4s", "v1.4h", "v2.h[0]"],
        &[
            ("v0", 0),
            ("v1", packed(16, &[2])),
            ("v2", packed(16, &[0xffff])),
        ],
        0xffff_fffe,
    );
}

#[test]
fn fmul_by_element_multiplies_every_lane_by_the_named_float() {
    // 2.0 times `v2` lane 1, which is 3.0, is 6.0 (0x40c00000).
    assert_computes(
        "fmul",
        &["v0.4s", "v1.4s", "v2.s[1]"],
        &[
            ("v1", packed(32, &[0x4000_0000])),
            ("v2", packed(32, &[0x3f80_0000, 0x4040_0000])),
        ],
        packed(32, &[0x40c0_0000]),
    );
}

#[test]
fn sdot_sums_four_byte_products_onto_the_destination_lane() {
    // 1*10 + 2*20 + 3*30 + 4*40 is 300, on top of a prior lane of 5.
    assert_computes(
        "sdot",
        &["v0.4s", "v1.16b", "v2.16b"],
        &[
            ("v0", packed(32, &[5])),
            ("v1", packed(8, &[1, 2, 3, 4])),
            ("v2", packed(8, &[10, 20, 30, 40])),
        ],
        packed(32, &[305]),
    );
}

#[test]
fn sdot_sign_extends_its_byte_elements() {
    // 0xff is -1, so the product is -2 and the lane is 0xfffffffe.
    assert_computes(
        "sdot",
        &["v0.4s", "v1.16b", "v2.16b"],
        &[
            ("v0", 0),
            ("v1", packed(8, &[0xff])),
            ("v2", packed(8, &[2])),
        ],
        0xffff_fffe,
    );
}

#[test]
fn udot_zero_extends_the_same_byte_elements() {
    // The mirror: 255 * 2 is 510, which is what makes the signed test
    // above a signedness contract rather than a restatement.
    assert_computes(
        "udot",
        &["v0.4s", "v1.16b", "v2.16b"],
        &[
            ("v0", 0),
            ("v1", packed(8, &[0xff])),
            ("v2", packed(8, &[2])),
        ],
        510,
    );
}

// ===================== ARM `FPMax` / `FPMin` =====================
//
// The whole point of these: the family *looks* like one more reduction
// over the existing `FpArithOp::Max` lane helper, and that helper is
// Intel's `MAXPS` — "the second operand wins on unordered and on
// equality". ARM's `FPMax` propagates NaN and combines the signs of a
// zero tie instead. Every expectation below is one the reused helper
// gets wrong, so a silent reuse fails here rather than passing.

/// Binary32 patterns the max / min contracts bind.
const F32_ONE: u128 = 0x3f80_0000;
const F32_FIVE: u128 = 0x40a0_0000;
const F32_TWO: u128 = 0x4000_0000;
const F32_THREE: u128 = 0x4040_0000;
const F32_NEGATIVE_ZERO: u128 = 0x8000_0000;
const F32_QUIET_NAN: u128 = 0x7fc0_0000;
const F32_SIGNALLING_NAN: u128 = 0x7f80_0001;

#[test]
fn fmaxv_reduces_to_the_largest_lane() {
    assert_computes(
        "fmaxv",
        &["s0", "v1.4s"],
        &[("v1", packed(32, &[F32_ONE, F32_FIVE, F32_TWO, F32_THREE]))],
        F32_FIVE,
    );
}

#[test]
fn fminv_reduces_to_the_smallest_lane() {
    assert_computes(
        "fminv",
        &["s0", "v1.4s"],
        &[("v1", packed(32, &[F32_ONE, F32_FIVE, F32_TWO, F32_THREE]))],
        F32_ONE,
    );
}

#[test]
fn fmaxv_propagates_a_nan_lane_instead_of_selecting_a_number() {
    // The trap. `MAXPS` returns its second operand when the comparison
    // is unordered, so reusing that helper would fold the NaN away and
    // give 1.0 here. ARM propagates the NaN.
    assert_computes(
        "fmaxv",
        &["s0", "v1.4s"],
        &[(
            "v1",
            packed(32, &[F32_QUIET_NAN, F32_ONE, F32_ONE, F32_ONE]),
        )],
        F32_QUIET_NAN,
    );
}

#[test]
fn fmax_propagates_a_nan_operand() {
    // The same trap on the scalar handler, which reached for the x86
    // helper by name and had been wrong on this input since it landed.
    assert_computes(
        "fmax",
        &["s0", "s1", "s2"],
        &[("v1", F32_QUIET_NAN), ("v2", F32_ONE)],
        F32_QUIET_NAN,
    );
}

#[test]
fn fmax_quiets_a_signalling_nan_operand() {
    // `FPProcessNaN` with `FPCR.DN` at its reset value returns the
    // operand with the leading significand bit set, so the payload
    // survives and only the quiet bit changes.
    assert_computes(
        "fmax",
        &["s0", "s1", "s2"],
        &[("v1", F32_ONE), ("v2", F32_SIGNALLING_NAN)],
        F32_QUIET_NAN | 1,
    );
}

#[test]
fn fmax_prefers_a_signalling_second_operand_over_a_quiet_first() {
    // ARM's NaN priority is signalling before quiet, and only then
    // first operand before second — so a plain "the first NaN wins"
    // would return the quiet one and be wrong here.
    assert_computes(
        "fmax",
        &["s0", "s1", "s2"],
        &[("v1", F32_QUIET_NAN), ("v2", F32_SIGNALLING_NAN)],
        F32_QUIET_NAN | 1,
    );
}

#[test]
fn fmin_of_a_negative_and_a_positive_zero_keeps_the_negative_one() {
    // Neither compares less than the other, so the architecture
    // combines the signs — `OR` for min. `MINPS` would take its second
    // operand and give +0.0.
    assert_computes(
        "fmin",
        &["s0", "s1", "s2"],
        &[("v1", F32_NEGATIVE_ZERO), ("v2", 0)],
        F32_NEGATIVE_ZERO,
    );
}

#[test]
fn fmax_of_a_positive_and_a_negative_zero_keeps_the_positive_one() {
    // The mirror: `AND` for max. `MAXPS` would give -0.0.
    assert_computes(
        "fmax",
        &["s0", "s1", "s2"],
        &[("v1", 0), ("v2", F32_NEGATIVE_ZERO)],
        0,
    );
}

// ===================== table lookup and polynomial multiply =====================

/// A register list, which radare2 renders as one braced operand and
/// which classifies as neither a register nor a memory reference.
const LIST: OperandKind = OperandKind::Unknown;

/// The identity table: byte `i` holds `i`, so a looked-up byte equals
/// the index that selected it and a wrong index is visible in the
/// answer.
fn identity_table() -> u128 {
    packed(8, &(0..16).collect::<Vec<u128>>())
}

#[test]
fn tbl_selects_the_table_byte_the_index_names() {
    // Indices 3 and 1 select table bytes 3 and 1; index 20 is past the
    // end of a single-register table, which `tbl` answers with zero.
    assert_computes_mixed(
        "tbl",
        &[("v0.8b", REG), ("{v1.16b}", LIST), ("v2.8b", REG)],
        &[
            ("v1", identity_table()),
            ("v2", packed(8, &[3, 1, 20, 0, 0, 0, 0, 0])),
        ],
        packed(8, &[3, 1, 0, 0, 0, 0, 0, 0]),
    );
}

#[test]
fn tbx_keeps_the_destination_byte_for_an_out_of_range_index() {
    // The one difference from `tbl`, and the reason `tbx` reads its
    // destination: byte 1's index is past the table, so the prior 0xaa
    // survives where `tbl` would have written zero.
    assert_computes_mixed(
        "tbx",
        &[("v0.8b", REG), ("{v1.16b}", LIST), ("v2.8b", REG)],
        &[
            ("v0", packed(8, &[0xaa; 8])),
            ("v1", identity_table()),
            ("v2", packed(8, &[3, 20, 0, 0, 0, 0, 0, 0])),
        ],
        packed(8, &[3, 0xaa, 0, 0, 0, 0, 0, 0]),
    );
}

#[test]
fn tbl_spans_a_two_register_table_low_to_high() {
    // The two members concatenate into a 32-byte table: index 3 hits
    // member 0's byte 3, index 20 hits member 1's byte 4 (0x24), and
    // index 40 is past the 32-byte table so `tbl` answers with zero.
    let upper = packed(8, &(0x20..0x30).collect::<Vec<u128>>());
    assert_computes_mixed(
        "tbl",
        &[("v0.8b", REG), ("{v1.16b, v2.16b}", LIST), ("v3.8b", REG)],
        &[
            ("v1", identity_table()),
            ("v2", upper),
            ("v3", packed(8, &[3, 20, 40, 0, 0, 0, 0, 0])),
        ],
        packed(8, &[3, 0x24, 0, 0, 0, 0, 0, 0]),
    );
}

#[test]
fn pmull_multiplies_without_carries() {
    // 3 times 3 carry-less is `(3 << 0) XOR (3 << 1)` = 5. An ordinary
    // multiply gives 9, so this is the whole contract in one lane: the
    // two are different functions, not one an approximation of the
    // other.
    assert_computes(
        "pmull",
        &["v0.8h", "v1.8b", "v2.8b"],
        &[("v1", packed(8, &[3])), ("v2", packed(8, &[3]))],
        packed(16, &[5]),
    );
}

#[test]
fn pmull2_multiplies_the_upper_halves() {
    // The low eight bytes are 0xff apiece so that reading them would be
    // visible; the `2` suffix takes bytes 8..15.
    let mut low = vec![0xffu128; 8];
    low.push(3);
    assert_computes(
        "pmull2",
        &["v0.8h", "v1.16b", "v2.16b"],
        &[("v1", packed(8, &low)), ("v2", packed(8, &low))],
        packed(16, &[5]),
    );
}

#[test]
fn pmull_of_two_doublewords_fills_the_quadword_lane() {
    // The AES / GHASH shape. `2^63` squared carry-less is `2^126`,
    // which only fits because the destination is the one-lane 128-bit
    // arrangement — the reason `q` had to become a real element type.
    assert_computes(
        "pmull",
        &["v0.1q", "v1.1d", "v2.1d"],
        &[("v1", 1u128 << 63), ("v2", 1u128 << 63)],
        1u128 << 126,
    );
}

// ===================== estimates =====================

#[test]
fn frecpe_leaves_the_destination_free_rather_than_computing_a_reciprocal() {
    // The anti-fabrication contract. `FDiv(1.0, 2.0)` is 0.5, and a
    // lowering that computed it would make this `AlwaysTrue` — for a
    // value the architecture does not require the machine to produce.
    // The estimate is a free input instead, so 0.5 remains *possible*
    // and is not asserted.
    assert_eq!(
        solve_lowering(
            "frecpe",
            &["v0.4s", "v1.4s"],
            &[("v1", packed(32, &[F32_TWO; 4]))],
            packed(32, &[0x3f00_0000; 4]),
        ),
        SmtResult::BothPossible,
    );
}

#[test]
fn frecpe_keeps_the_slice_complete_rather_than_truncating() {
    // The other half of the same decision: a decline would have made
    // the destination undefined and the verdict `Unsound`. A verdict
    // that ranges over every value the estimate could take is worth
    // more than no verdict at all.
    assert_ne!(
        solve_lowering(
            "frecpe",
            &["v0.4s", "v1.4s"],
            &[("v1", packed(32, &[F32_TWO; 4]))],
            packed(32, &[0x3f00_0000; 4]),
        ),
        SmtResult::Unsound,
    );
}
