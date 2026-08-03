//! `AArch64` NEON structured load / store contract.
//!
//! A de-interleaving load that puts the right *number* of bytes in the
//! right *register* but the wrong element in the wrong lane produces a
//! wrong value, not a decline — and it looks entirely plausible in the
//! IR. So these tests solve the lowering instead of reading it.
//!
//! Each one seeds memory with concrete bytes through the byte-granular
//! model, lifts the real instruction, and asks the solver whether the
//! destination register necessarily equals a value computed by hand
//! from the ARM definition. The store direction goes the other way: the
//! instruction writes memory and a hand-written load reads it back.
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
const POINTER_BITS: u16 = 64;
const BASE_REGISTER: &str = "x0";

/// The concrete address every fixture's base register holds. Any value
/// works; a fixed one keeps the seeded stores and the lifted loads
/// talking about the same bytes without a symbolic base.
const BASE_ADDRESS: u128 = 0x4000;

fn operand(raw: &str) -> Operand {
    let kind = if raw.starts_with('{') {
        OperandKind::Unknown
    } else if raw.starts_with('[') {
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

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "structuredtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "structuredtest".to_string(),
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
        operands: operands.iter().map(|raw| operand(raw)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

fn lift(mnemonic: &str, operands: &[&str]) -> Vec<IrStmt> {
    let lifted = lift_per_mnemonic(&instruction(mnemonic, operands), Arch::Aarch64);
    assert!(
        lifted
            .iter()
            .all(|stmt| !matches!(stmt, IrStmt::Unsupported { .. })),
        "{mnemonic} {operands:?} declined: {lifted:?}"
    );
    lifted
}

fn address_at(offset: u128) -> Expr {
    Expr::konst(BASE_ADDRESS + offset, POINTER_BITS)
}

/// Pin the base register to [`BASE_ADDRESS`] and seed `bytes` starting
/// there, one `StoreMem` per element.
fn seed(elements: &[(u128, u16)]) -> Vec<IrStmt> {
    let mut statements = vec![IrStmt::Assign {
        dst: Var::new(BASE_REGISTER, POINTER_BITS),
        src: Expr::konst(BASE_ADDRESS, POINTER_BITS),
    }];
    let mut offset = 0u128;
    for (value, bits) in elements {
        statements.push(IrStmt::StoreMem {
            address: address_at(offset),
            value: Expr::konst(*value, *bits),
            bits: *bits,
        });
        offset += u128::from(*bits / 8);
    }
    statements
}

fn solve(statements: Vec<IrStmt>, condition: Expr) -> SmtResult {
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition,
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

/// Seed memory, run the load, and assert the named vector register
/// necessarily holds `expected`.
fn assert_loads(
    mnemonic: &str,
    operands: &[&str],
    elements: &[(u128, u16)],
    register: &str,
    expected: u128,
) {
    let mut statements = seed(elements);
    statements.extend(lift(mnemonic, operands));
    let condition = Expr::eq(
        Expr::Var(Var::new(register, VECTOR_BITS)),
        Expr::konst(expected, VECTOR_BITS),
    );
    assert_eq!(
        solve(statements, condition),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} should leave {register} = {expected:#x}"
    );
}

/// Bind the listed vector registers, run the store, and assert the
/// `bits`-wide unit at `offset` bytes from the base necessarily holds
/// `expected`.
fn assert_stores(
    mnemonic: &str,
    operands: &[&str],
    sources: &[(&str, u128)],
    unit: (u128, u16),
    expected: u128,
) {
    let (offset, bits) = unit;
    let mut statements = vec![IrStmt::Assign {
        dst: Var::new(BASE_REGISTER, POINTER_BITS),
        src: Expr::konst(BASE_ADDRESS, POINTER_BITS),
    }];
    statements.extend(sources.iter().map(|(name, value)| IrStmt::Assign {
        dst: Var::new(*name, VECTOR_BITS),
        src: Expr::konst(*value, VECTOR_BITS),
    }));
    statements.extend(lift(mnemonic, operands));
    let readback = Var::new("readback", bits);
    statements.push(IrStmt::LoadMem {
        dst: readback.clone(),
        address: address_at(offset),
        bits,
    });
    let condition = Expr::eq(Expr::Var(readback), Expr::konst(expected, bits));
    assert_eq!(
        solve(statements, condition),
        SmtResult::AlwaysTrue,
        "{mnemonic} {operands:?} should leave {bits} bits at +{offset} = {expected:#x}"
    );
}

/// Sixteen distinct bytes, one word each, as the de-interleaving
/// fixtures' memory image: element `n` holds `0xn0`.
fn words(count: u16) -> Vec<(u128, u16)> {
    (0..count).map(|n| (u128::from(n) * 0x10, 32)).collect()
}

// ===================== ld1 / st1, whole registers =====================

#[test]
fn test_ld1_single_register_loads_the_whole_view() {
    assert_loads(
        "ld1",
        &["{v0.16b}", "[x0]"],
        &[(0x1122_3344_5566_7788, 64), (0x99aa_bbcc_ddee_ff00, 64)],
        "v0",
        0x99aa_bbcc_ddee_ff00_1122_3344_5566_7788,
    );
}

#[test]
fn test_ld1_half_width_arrangement_zeroes_the_upper_half() {
    // Every `AArch64` SIMD write covers the whole register: a `.8b`
    // load writes 64 bits and zeroes the 64 above them.
    assert_loads(
        "ld1",
        &["{v0.8b}", "[x0]"],
        &[(0x1122_3344_5566_7788, 64), (0xffff_ffff_ffff_ffff, 64)],
        "v0",
        0x1122_3344_5566_7788,
    );
}

#[test]
fn test_ld1_second_register_reads_the_next_view() {
    // A two-register `ld1` interleaves nothing: `v1` takes the sixteen
    // bytes above `v0`'s.
    assert_loads(
        "ld1",
        &["{v0.16b, v1.16b}", "[x0]"],
        &[
            (0, 64),
            (0, 64),
            (0x0011_2233_4455_6677, 64),
            (0x8899_aabb_ccdd_eeff, 64),
        ],
        "v1",
        0x8899_aabb_ccdd_eeff_0011_2233_4455_6677,
    );
}

#[test]
fn test_ld1_first_register_is_unaffected_by_a_second_member() {
    assert_loads(
        "ld1",
        &["{v0.16b, v1.16b}", "[x0]"],
        &[
            (0x1122_3344_5566_7788, 64),
            (0x99aa_bbcc_ddee_ff00, 64),
            (0xffff_ffff_ffff_ffff, 64),
            (0xffff_ffff_ffff_ffff, 64),
        ],
        "v0",
        0x99aa_bbcc_ddee_ff00_1122_3344_5566_7788,
    );
}

#[test]
fn test_st1_writes_the_whole_view_at_the_base() {
    assert_stores(
        "st1",
        &["{v0.16b}", "[x0]"],
        &[("v0", 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100)],
        (0, 64),
        0x0706_0504_0302_0100,
    );
}

#[test]
fn test_st1_second_register_lands_one_view_above_the_first() {
    assert_stores(
        "st1",
        &["{v0.8b, v1.8b}", "[x0]"],
        &[("v0", 0x1111_1111_1111_1111), ("v1", 0x2222_2222_2222_2222)],
        (8, 64),
        0x2222_2222_2222_2222,
    );
}

// ===================== the single-element forms =====================

#[test]
fn test_ld1_single_element_writes_only_the_addressed_lane() {
    // Lane 1 takes the loaded word; the three lanes around it keep the
    // value `v0` was bound to.
    let mut statements = seed(&[(0xdead_beef, 32)]);
    statements.push(IrStmt::Assign {
        dst: Var::new("v0", VECTOR_BITS),
        src: Expr::konst(0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa, VECTOR_BITS),
    });
    statements.extend(lift("ld1", &["{v0.s}[1]", "[x0]"]));
    let condition = Expr::eq(
        Expr::Var(Var::new("v0", VECTOR_BITS)),
        Expr::konst(0xaaaa_aaaa_aaaa_aaaa_dead_beef_aaaa_aaaa_u128, VECTOR_BITS),
    );
    assert_eq!(solve(statements, condition), SmtResult::AlwaysTrue);
}

#[test]
fn test_st1_single_element_stores_the_addressed_lane() {
    assert_stores(
        "st1",
        &["{v0.s}[3]", "[x0]"],
        &[("v0", 0x4444_4444_3333_3333_2222_2222_1111_1111)],
        (0, 32),
        0x4444_4444,
    );
}

#[test]
fn test_ld2_single_element_loads_consecutive_elements_into_each_register() {
    // `ld2 {v0.s, v1.s}[1]` reads two words: the first goes to `v0`'s
    // lane 1, the second to `v1`'s.
    let mut statements = seed(&[(0x1111_1111, 32), (0x2222_2222, 32)]);
    for name in ["v0", "v1"] {
        statements.push(IrStmt::Assign {
            dst: Var::new(name, VECTOR_BITS),
            src: Expr::konst(0, VECTOR_BITS),
        });
    }
    statements.extend(lift("ld2", &["{v0.s, v1.s}[1]", "[x0]"]));
    let condition = Expr::eq(
        Expr::Var(Var::new("v1", VECTOR_BITS)),
        Expr::konst(0x2222_2222_0000_0000, VECTOR_BITS),
    );
    assert_eq!(solve(statements, condition), SmtResult::AlwaysTrue);
}

// ===================== de-interleaving =====================

#[test]
fn test_ld2_first_destination_takes_the_even_elements() {
    // Memory holds e0..e7; `v0` gets e0, e2, e4, e6.
    assert_loads(
        "ld2",
        &["{v0.4s, v1.4s}", "[x0]"],
        &words(8),
        "v0",
        0x0000_0060_0000_0040_0000_0020_0000_0000,
    );
}

#[test]
fn test_ld2_second_destination_takes_the_odd_elements() {
    assert_loads(
        "ld2",
        &["{v0.4s, v1.4s}", "[x0]"],
        &words(8),
        "v1",
        0x0000_0070_0000_0050_0000_0030_0000_0010,
    );
}

#[test]
fn test_ld3_middle_destination_takes_every_third_element_from_one() {
    // e1, e4, e7, e10 for `v1` of a 3-way interleave.
    assert_loads(
        "ld3",
        &["{v0.4s, v1.4s, v2.4s}", "[x0]"],
        &words(12),
        "v1",
        0x0000_00a0_0000_0070_0000_0040_0000_0010,
    );
}

#[test]
fn test_ld4_last_destination_takes_every_fourth_element_from_three() {
    // e3, e7, e11, e15 for `v3` of a 4-way interleave.
    assert_loads(
        "ld4",
        &["{v0.4s, v1.4s, v2.4s, v3.4s}", "[x0]"],
        &words(16),
        "v3",
        0x0000_00f0_0000_00b0_0000_0070_0000_0030,
    );
}

#[test]
fn test_ld2_half_width_arrangement_zeroes_the_upper_half() {
    assert_loads(
        "ld2",
        &["{v0.2s, v1.2s}", "[x0]"],
        &words(4),
        "v0",
        0x0000_0020_0000_0000,
    );
}

#[test]
fn test_st2_interleaves_the_two_sources_into_memory() {
    // `v0`'s lane 0 goes to element 0 and `v1`'s lane 0 to element 1,
    // so the first doubleword is `v1.s[0] : v0.s[0]`.
    assert_stores(
        "st2",
        &["{v0.4s, v1.4s}", "[x0]"],
        &[
            ("v0", 0x0000_0004_0000_0003_0000_0002_0000_0001),
            ("v1", 0x0000_0040_0000_0030_0000_0020_0000_0010),
        ],
        (0, 64),
        0x0000_0010_0000_0001,
    );
}

#[test]
fn test_st2_second_structure_follows_the_first() {
    assert_stores(
        "st2",
        &["{v0.4s, v1.4s}", "[x0]"],
        &[
            ("v0", 0x0000_0004_0000_0003_0000_0002_0000_0001),
            ("v1", 0x0000_0040_0000_0030_0000_0020_0000_0010),
        ],
        (8, 64),
        0x0000_0020_0000_0002,
    );
}

#[test]
fn test_st4_writes_one_structure_per_lane_index() {
    // Lane 0 of all four sources lands in the first four words.
    assert_stores(
        "st4",
        &["{v0.4s, v1.4s, v2.4s, v3.4s}", "[x0]"],
        &[
            ("v0", 0x1111_1111),
            ("v1", 0x2222_2222),
            ("v2", 0x3333_3333),
            ("v3", 0x4444_4444),
        ],
        (0, 128),
        0x4444_4444_3333_3333_2222_2222_1111_1111,
    );
}

// ===================== replication =====================

#[test]
fn test_ld1r_broadcasts_one_element_to_every_lane() {
    assert_loads(
        "ld1r",
        &["{v0.4s}", "[x0]"],
        &[(0xdead_beef, 32)],
        "v0",
        0xdead_beef_dead_beef_dead_beef_dead_beef,
    );
}

#[test]
fn test_ld1r_half_width_arrangement_zeroes_the_upper_half() {
    assert_loads(
        "ld1r",
        &["{v0.2s}", "[x0]"],
        &[(0xdead_beef, 32)],
        "v0",
        0xdead_beef_dead_beef,
    );
}

#[test]
fn test_ld2r_second_register_broadcasts_the_second_element() {
    assert_loads(
        "ld2r",
        &["{v0.8b, v1.8b}", "[x0]"],
        &[(0xaa, 8), (0xbb, 8)],
        "v1",
        0xbbbb_bbbb_bbbb_bbbb,
    );
}

#[test]
fn test_ld4r_last_register_broadcasts_the_fourth_element() {
    assert_loads(
        "ld4r",
        &["{v0.4h, v1.4h, v2.4h, v3.4h}", "[x0]"],
        &[(0x1111, 16), (0x2222, 16), (0x3333, 16), (0x4444, 16)],
        "v3",
        0x4444_4444_4444_4444,
    );
}

// ===================== post-index writeback =====================

#[test]
fn test_ld1_immediate_post_index_advances_the_base() {
    let mut statements = seed(&[(0, 64), (0, 64)]);
    statements.extend(lift("ld1", &["{v0.16b}", "[x0]", "16"]));
    let condition = Expr::eq(
        Expr::Var(Var::new(BASE_REGISTER, POINTER_BITS)),
        Expr::konst(BASE_ADDRESS + 16, POINTER_BITS),
    );
    assert_eq!(solve(statements, condition), SmtResult::AlwaysTrue);
}

#[test]
fn test_ld1_register_post_index_advances_the_base_by_that_register() {
    // The stride is a run-time value, which is the whole reason a
    // writeback delta is an expression rather than a constant.
    let mut statements = seed(&[(0, 64), (0, 64)]);
    statements.push(IrStmt::Assign {
        dst: Var::new("x3", POINTER_BITS),
        src: Expr::konst(0x30, POINTER_BITS),
    });
    statements.extend(lift("ld1", &["{v0.16b}", "[x0]", "x3"]));
    let condition = Expr::eq(
        Expr::Var(Var::new(BASE_REGISTER, POINTER_BITS)),
        Expr::konst(BASE_ADDRESS + 0x30, POINTER_BITS),
    );
    assert_eq!(solve(statements, condition), SmtResult::AlwaysTrue);
}

#[test]
fn test_post_index_load_reads_the_pre_update_base() {
    // The writeback is emitted after the transfer, so the address the
    // load used is the base before the bump.
    assert_loads(
        "ld1",
        &["{v0.8b}", "[x0]", "8"],
        &[(0x1122_3344_5566_7788, 64)],
        "v0",
        0x1122_3344_5566_7788,
    );
}
