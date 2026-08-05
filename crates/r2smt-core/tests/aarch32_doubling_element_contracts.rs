//! The `AArch32` doubling multiplies in their by-element spelling, plus
//! the ARMv8.1 `vqrdmlah` / `vqrdmlsh` pair, solved rather than asserted
//! structurally.
//!
//! Two distinct failure modes are pinned here, and neither is a decline.
//!
//! A by-element form has exactly the operand count and register kinds
//! the lane-wise one accepts, so a resolver that misses it does not fail
//! closed — it reads the whole of `d3` where the instruction names one
//! of its lanes. Every by-element test below therefore chooses a second
//! source whose other lanes carry values a lane-wise reading would
//! visibly produce.
//!
//! `vqrdmlah` saturates **once**, over the accumulator and the doubled
//! product together, where `vqdmlal` saturates the product and then the
//! sum. Lowering it as `vqrdmulh` followed by a saturating add is the
//! natural mistake and gives a value one ulp away on the corner where it
//! matters, which is what the first test binds.
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

/// Bits above a `d` destination, which an `AArch32` NEON write merges
/// rather than zeroing. Present so a lowering that wrote the whole
/// parent would fail the expectation.
const UPPER: u128 = 0xdead_beef_dead_beef;

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
        mnemonic: "doublingtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "doublingtest".to_string(),
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

/// Lift `mnemonic operands` on `Arch::Arm`, bind `v0` and `v1`, and ask
/// the solver whether the low `width` bits of `v0` are necessarily
/// `expected`.
///
/// `d0` / `q0` view `v0`; `d2` is the low half of `v1` and `d3` the high
/// one, so a whole 128-bit binding of `v1` supplies both sources at
/// once.
fn solve_lowering(
    mnemonic: &str,
    operands: &[&str],
    destination: u128,
    sources: u128,
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

    let mut statements = vec![
        IrStmt::Assign {
            dst: Var::new("v0", VECTOR_BITS),
            src: Expr::konst(destination, VECTOR_BITS),
        },
        IrStmt::Assign {
            dst: Var::new("v1", VECTOR_BITS),
            src: Expr::konst(sources, VECTOR_BITS),
        },
    ];
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

/// Assert a `d`-destination form computes `expected` across all four
/// halfword lanes.
fn assert_doubleword(
    mnemonic: &str,
    operands: &[&str],
    accumulator: u128,
    sources: u128,
    expected: u128,
) {
    assert_eq!(
        solve_lowering(
            mnemonic,
            operands,
            (UPPER << 64) | accumulator,
            sources,
            expected,
            64,
        ),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} should give {expected:#x}"
    );
}

fn assert_declines(mnemonic: &str, operands: &[&str]) {
    let lifted = lift_per_mnemonic(&instruction(mnemonic, operands), Arch::Arm);
    assert!(
        lifted
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} should decline, got {lifted:?}"
    );
}

/// `d2` holds `0x4000, 0x8000, 0x0002, 0x2000` low lane first — a
/// positive half-scale value, `INT_MIN`, a small value and a quarter —
/// and `d3` holds `0x0001, 0x4000, 0x0001, 0x0001`, so lane 1 is the
/// element the indexed tests select and every other lane is a value a
/// lane-wise misreading would visibly produce.
const SOURCES: u128 = 0x0001_0001_4000_0001_2000_0002_8000_4000;

#[test]
fn by_element_vqdmulh_multiplies_every_lane_by_the_named_element() {
    // `0x4000` is 0.5 in the Q15 reading this instruction implements, so
    // each lane halves: 0x4000→0x2000, INT_MIN→0xc000, 2→1, 0x2000→
    // 0x1000. A lane-wise reading against `d3` would give
    // `0x0000_0000_c000_0000` — only the lane whose `d3` element happens
    // to be the selected one survives.
    assert_doubleword(
        "vqdmulh.s16",
        &["d0", "d2", "d3[1]"],
        0,
        SOURCES,
        0x1000_0001_c000_2000,
    );
}

#[test]
fn by_element_vqdmulh_saturates_the_int_min_square() {
    // `d3[0]` is 1, so nothing saturates; selecting `d2`'s own INT_MIN
    // lane as the element is what reaches the one corner this family
    // clamps. `2 * INT_MIN * INT_MIN >> 15` is `INT_MAX + 1`, and the
    // clamp turns it into 0x7fff.
    //
    // Lane 1 of the source is INT_MIN, so `d2[1]` is the element and the
    // lane pairing it with itself is lane 1 of the result.
    assert_doubleword(
        "vqdmulh.s16",
        &["d0", "d2", "d2[1]"],
        0,
        SOURCES,
        0xe000_fffe_7fff_c000,
    );
}

#[test]
fn by_element_vqdmull_keeps_the_doubled_product_whole() {
    // The long form's destination element is twice the source's, so
    // nothing is discarded and nothing saturates: each 32-bit lane is
    // `2 * a * 0x4000` exactly. A lane-wise reading would zero three of
    // the four lanes.
    assert_eq!(
        solve_lowering(
            "vqdmull.s16",
            &["q0", "d2", "d3[1]"],
            0,
            SOURCES,
            0x1000_0000_0001_0000_c000_0000_2000_0000,
            128,
        ),
        SmtResult::AlwaysTrue,
        "vqdmull.s16 q0, d2, d3[1] should double every lane product"
    );
}

#[test]
fn by_element_vqdmlal_saturates_the_product_before_accumulating() {
    // The long accumulating form saturates twice, and in this order: the
    // doubled product is clamped into the destination element first,
    // then the sum with the accumulator is clamped again. Here nothing
    // reaches either bound, so the test pins the arithmetic and the
    // element selection; the ordering itself is covered on the lane-wise
    // spelling.
    assert_eq!(
        solve_lowering(
            "vqdmlal.s16",
            &["q0", "d2", "d3[1]"],
            0x0000_0001_0000_0000_0000_0000_0000_0002,
            SOURCES,
            0x1000_0001_0001_0000_c000_0000_2000_0002,
            128,
        ),
        SmtResult::AlwaysTrue,
        "vqdmlal.s16 should add the doubled product onto the destination"
    );
}

#[test]
fn vqrdmlah_saturates_once_over_the_whole_expression() {
    // The teeth of the family. Lane 0 pairs INT_MIN with itself against
    // an accumulator of -1. Saturating the rounded doubled product on
    // its own gives 0x7fff, and adding -1 to that gives 0x7ffe;
    // scaling the accumulator up and clamping the total once — which is
    // what the architecture defines — gives 0x7fff. One ulp apart, and
    // both lift cleanly.
    //
    // The other lanes pin the ordinary arithmetic: 0x4000 squared is a
    // half-scale product of 0x2000 added onto 0x0010, the third lane
    // rounds a tiny product away, and the fourth carries a bare
    // accumulator through the shift.
    assert_doubleword(
        "vqrdmlah.s16",
        &["d0", "d2", "d3"],
        0x0001_0000_0010_ffff,
        0x0000_0001_4000_8000_0000_0001_4000_8000,
        0x0001_0000_2010_7fff,
    );
}

#[test]
fn vqrdmlsh_subtracts_the_doubled_product_from_the_accumulator() {
    // The companion direction on an input that undoes the test above:
    // an accumulator of 0x2010 less the 0x2000 half-scale product leaves
    // 0x0010. A lowering that added would give 0x4010.
    assert_doubleword(
        "vqrdmlsh.s16",
        &["d0", "d2", "d3"],
        0x0000_0000_0000_2010,
        0x0000_0000_0000_4000_0000_0000_0000_4000,
        0x0000_0000_0000_0010,
    );
}

#[test]
fn by_element_vqrdmlah_selects_the_element_rather_than_pairing_lanes() {
    // The by-element spelling of the same family. `d3[1]` is 0x4000 and
    // every other `d3` lane is zero, so a lane-wise reading would leave
    // three lanes holding their accumulator alone — 0x0010 instead of
    // 0x2010 in lane 0, which is exactly the difference this asserts.
    //
    // The top lane pins the rounding term as well: `2 * 1 * 0x4000` is
    // exactly half an element, so the added half ulp carries it to 1
    // where dropping the term would floor it to 0. Every other lane
    // here is insensitive to it.
    assert_doubleword(
        "vqrdmlah.s16",
        &["d0", "d2", "d3[1]"],
        0x0000_0000_0000_0010,
        0x0000_0000_4000_0000_0001_8000_0002_4000,
        0x0001_c000_0001_2010,
    );
}

#[test]
fn an_out_of_range_element_index_declines() {
    // A `d` register holds four halfwords, so `[8]` names nothing. The
    // resolver refuses rather than wrapping the index, and no other
    // resolver claims the mnemonic, so the instruction fails closed.
    assert_declines("vqdmulh.s16", &["d0", "d2", "d3[8]"]);
}

#[test]
fn an_unsigned_doubling_multiply_declines() {
    // Doubling and saturating at `INT_MAX` are statements about a
    // two's-complement sign, so the architecture gives this family no
    // unsigned encoding and there is nothing for the spelling to mean.
    assert_declines("vqrdmlah.u16", &["d0", "d2", "d3"]);
}
