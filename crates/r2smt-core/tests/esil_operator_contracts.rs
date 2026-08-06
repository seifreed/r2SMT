//! Solver-backed contracts for the ESIL stack machine.
//!
//! The machine had ~45 tests before this file and all of them assert on
//! the *shape* of an `Expr` tree; the per-mnemonic handlers have ~750
//! that bind concrete values and ask a solver. That asymmetry is the
//! whole reason `:=` is still ungated: lexing it moves the entire
//! flag-setting population of every ISA off the handlers and onto this
//! machine, so the machine has to be worth the move first.
//!
//! Every string below is one radare2 6.1.8 emits or a minimal probe of
//! one, and every expected value was measured with
//! `r2 -q -a x86 -b 64 -c aei -c 'ar rax=<v>' -c '"ae <esil>"' -c 'ar rax'`
//! rather than derived from this implementation. Two traps in that
//! recipe, both paid for: the `"…"` quoting is mandatory because `>`,
//! `>=` and `>>` are r2's output redirection and vanish silently
//! without it, and the VM's flag state persists between `ae` calls
//! inside one session, so each probe needs its own process.
//!
//! Note what is deliberately *not* here: the multi-flag strings r2
//! really emits (`1,eax,==,$z,zf,:=,32,$b,cf,:=,…`). Each flag write in
//! those is `:=` precisely so it does not clobber the context the next
//! flag token still reads, and `:=` does not lex yet — so a contract
//! written with `=` can pin one flag per string and no more. That is a
//! limit of the gate, not of the model, and these extend to the real
//! strings the day it opens.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_esil::lift_esil;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceStatus};
use r2smt_smt::solve_branch;
use r2smt_ssa::ssa_convert;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;
const QWORD: u16 = 64;

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn branch() -> BranchCandidate {
    let at = Address::new(0x1000);
    BranchCandidate {
        address: at,
        function: at,
        block: at,
        kind: BranchKind::Jcc,
        mnemonic: "esiltest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "esiltest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Lift `esil` with every named register bound to a constant, and report
/// whether `observed` is *necessarily* `expected`.
///
/// `AlwaysTrue` is the only passing answer: `BothPossible` means the
/// machine left the value free, which is the failure mode a structural
/// test cannot see.
fn solve_esil(
    esil: &str,
    bindings: &[(&str, u128, u16)],
    observed: (&str, u16),
    expected: u128,
) -> SmtResult {
    let lifted = lift_esil(esil, Arch::X86_64)
        .unwrap_or_else(|e| panic!("`{esil}` must lift, got {e:?}"))
        .statements;
    assert!(!lifted.is_empty(), "`{esil}` lifted to nothing");
    let mut statements: Vec<IrStmt> = bindings
        .iter()
        .map(|(name, value, bits)| IrStmt::Assign {
            dst: Var::new(*name, *bits),
            src: Expr::konst(*value, *bits),
        })
        .collect();
    statements.extend(lifted);
    let slice = LiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::eq(
            Expr::var(observed.0, observed.1),
            Expr::konst(expected, observed.1),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::X86_64,
    };
    solve_branch(&ssa_convert(&slice), solve_opts())
}

// ---------------------------------------------------------------------
// Sub-register spelling. The machine places a name against its
// architectural parent, so its output is the same data-flow node the
// per-mnemonic handlers read. Both write rules are separated here
// because picking the wrong one is a wrong value, not a lost one.
// ---------------------------------------------------------------------

#[test]
fn test_a_32_bit_alias_write_zero_extends_the_whole_parent() {
    // `mov eax, 1` clears bits 63..32 of `rax` — the `x86_64` rule. A
    // parent bound to all-ones before the write must come out as 1.
    assert_eq!(
        solve_esil(
            "1,eax,=",
            &[("rax", u128::from(u64::MAX), QWORD)],
            ("rax", QWORD),
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_byte_write_preserves_the_bits_above_it() {
    // `al` is not a 32-bit alias, so the same write preserves 63..8.
    assert_eq!(
        solve_esil(
            "0,al,=",
            &[("rax", u128::from(u64::MAX), QWORD)],
            ("rax", QWORD),
            0xffff_ffff_ffff_ff00
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_high_byte_write_lands_at_bits_15_to_8() {
    // `ah` is the one alias that does not start at bit 0, so it is what
    // separates a correct splice from one that ignores the offset.
    assert_eq!(
        solve_esil("0xff,ah,=", &[("rax", 0, QWORD)], ("rax", QWORD), 0xff00),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_subregister_read_takes_the_low_bits_of_the_parent() {
    assert_eq!(
        solve_esil(
            "eax,rbx,=",
            &[("rax", 0x1122_3344_5566_7788, QWORD)],
            ("rbx", QWORD),
            0x5566_7788
        ),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// The flag context. Seeded by writes and compares, never by bare
// arithmetic — the model asserted the exact inverse until 2026-08-06.
// ---------------------------------------------------------------------

#[test]
fn test_a_write_seeds_the_zero_flag() {
    assert_eq!(
        solve_esil("0,rax,=,$z,zf,=", &[], ("ZF", 1), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compare_seeds_the_zero_flag() {
    // `ae 1,rax,==,7,$s` answers from the context `==` leaves, so the
    // compare seeds with `lhs - rhs` and pushes nothing.
    assert_eq!(
        solve_esil("5,rax,==,$z,zf,=", &[("rax", 5, QWORD)], ("ZF", 1), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_zero_flag_masks_to_the_written_register_width() {
    // `0x100,al,=,$z` is 1: the byte written is zero.
    //
    // The parent's other bits are bound to **ones**, and that is the
    // whole fixture. With `rax` bound to zero this passes on a model
    // that seeds the context with the parent-width write instead of the
    // slice — measured, by making exactly that change and watching the
    // test stay green. The high bits are what separate the two.
    assert_eq!(
        solve_esil(
            "0x100,al,=,$z,zf,=",
            &[("rax", 0xffff_ffff_ffff_ff00, QWORD)],
            ("ZF", 1),
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_same_value_written_wide_leaves_the_zero_flag_clear() {
    assert_eq!(
        solve_esil("0x100,rax,=,$z,zf,=", &[], ("ZF", 1), 0),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// The bit index rides on the stack. `$s7` / `$b4` / `$c5` are not
// radare2 tokens at all — they come back unresolved — so an index in
// the token text was a syntax nothing ever emits.
// ---------------------------------------------------------------------

#[test]
fn test_the_sign_flag_reads_the_bit_the_stack_names() {
    // `ae 0x80,rax,=,7,$s` → 1.
    assert_eq!(
        solve_esil("0x80,rax,=,7,$s,sf,=", &[], ("SF", 1), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_different_index_reads_a_different_bit() {
    // `ae 0x80,rax,=,6,$s` → 0. The twin above passes on an
    // implementation that ignores the index entirely; this one does not.
    assert_eq!(
        solve_esil("0x80,rax,=,6,$s,sf,=", &[], ("SF", 1), 0),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// Carry, borrow and overflow: `old` against `cur` under a mask, and
// nothing else. The formula they replaced was a XOR of carries over the
// *operands*, which is a different function rather than a rounding
// error, and which needed an `ArithKind` to know whether the seeding
// operation added or subtracted. radare2 does not make that
// distinction.
// ---------------------------------------------------------------------

#[test]
fn test_the_carry_flag_is_the_masked_result_below_the_masked_original() {
    // old = 0xff, cur = 0, mask = ones(8): `0 <u 0xff` is 1.
    assert_eq!(
        solve_esil("0,rax,=,7,$c,cf,=", &[("rax", 0xff, QWORD)], ("CF", 1), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_borrow_flag_is_the_same_comparison_reversed() {
    // Same context, the other direction: `0xff <u 0` is 0. Asserted as
    // a pair with the carry so a symmetric-but-wrong implementation —
    // one that answers the same for both — fails one of them.
    assert_eq!(
        solve_esil("0,rax,=,7,$b,cf,=", &[("rax", 0xff, QWORD)], ("CF", 1), 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_carry_does_not_depend_on_how_the_context_was_produced() {
    // The claim that deletes `ArithKind`: `$c` is a function of the
    // (old, cur) pair alone. Reaching (10, 14) by adding and by writing
    // the result outright must answer the same.
    let by_adding = solve_esil("4,rax,+=,7,$c,cf,=", &[("rax", 10, QWORD)], ("CF", 1), 0);
    let by_writing = solve_esil("14,rax,=,7,$c,cf,=", &[("rax", 10, QWORD)], ("CF", 1), 0);
    assert_eq!(
        (by_adding, by_writing),
        (SmtResult::AlwaysTrue, SmtResult::AlwaysTrue)
    );
}

#[test]
fn test_the_overflow_flag_is_the_two_adjacent_carries_disagreeing() {
    // old = 0x7f, cur = 0x80 at 8 bits.
    //   $c(6): mask ones(7) → `0x00 <u 0x7f` = 1
    //   $c(7): mask ones(8) → `0x80 <u 0x7f` = 0
    // so `$o(7)` is 1 — the signed-overflow shape.
    assert_eq!(
        solve_esil("0x80,al,=,7,$o,of,=", &[("rax", 0x7f, QWORD)], ("OF", 1), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_parity_flag_folds_the_low_byte_to_even_parity() {
    // 0b11 has two set bits, so even parity is 1.
    assert_eq!(
        solve_esil("3,rax,=,$p,pf,=", &[], ("PF", 1), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_odd_parity_clears_the_flag() {
    assert_eq!(
        solve_esil("1,rax,=,$p,pf,=", &[], ("PF", 1), 0),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// Operand order. `ae 4,10,<op>` answers `10 <op> 4` — the second token
// is the left-hand side — and getting it backwards is a wrong value on
// every non-commutative operator. These run in production today: 611 of
// 893 ESIL strings in one x86-64 sample carry neither `:=` nor `?{` and
// so take the ESIL rung, and 290 of those contain a non-commutative
// operator. Until now none had a value test.
// ---------------------------------------------------------------------

#[test]
fn test_a_bare_subtract_takes_the_second_token_as_the_left_hand_side() {
    assert_eq!(
        solve_esil("4,10,-,rax,=", &[], ("rax", QWORD), 6),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_shift_shifts_the_second_token_by_the_first() {
    assert_eq!(
        solve_esil("4,10,<<,rax,=", &[], ("rax", QWORD), 160),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compound_subtract_takes_the_target_as_the_left_hand_side() {
    // Measured `aer rax=10; ae 4,rax,-=; aer rax` → 6. The only
    // compound-assign test that predates this file uses `+=`, which is
    // commutative and therefore passes on an inverted implementation.
    assert_eq!(
        solve_esil("4,rax,-=", &[("rax", 10, QWORD)], ("rax", QWORD), 6),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compound_shift_shifts_the_target_by_the_value() {
    // Measured → 0xa0, i.e. `10 << 4` and never `4 << 10`.
    assert_eq!(
        solve_esil("4,rax,<<=", &[("rax", 10, QWORD)], ("rax", QWORD), 0xa0),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// Memory. ~6 200 stores and ~7 000 loads across three samples take the
// ESIL rung, and none of them had a value test either — only structural
// ones about which stack slot the address comes from.
// ---------------------------------------------------------------------

#[test]
fn test_a_store_is_read_back_by_a_load_of_the_same_address() {
    assert_eq!(
        solve_esil(
            "0x2a,rbx,=[8],rbx,[8],rax,=",
            &[("rbx", 0x7fff_0000, QWORD)],
            ("rax", QWORD),
            0x2a
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_store_lays_its_bytes_out_little_endian() {
    // The low byte of the stored word sits at the *base* address, so a
    // one-byte load there reads 0x88 and not 0x11. A big-endian byte
    // order is a wrong value that the round-trip above cannot see.
    assert_eq!(
        solve_esil(
            "0x1122334455667788,rbx,=[8],rbx,[1],rax,=",
            &[("rbx", 0x7fff_0000, QWORD)],
            ("rax", QWORD),
            0x88
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_load_of_an_untouched_address_is_free_rather_than_zero() {
    // The soundness direction. Nothing wrote there, so the value must
    // range over every byte pattern — an implementation that answered
    // zero would fabricate.
    assert_eq!(
        solve_esil(
            "rbx,[8],rax,=",
            &[("rbx", 0x7fff_0000, QWORD)],
            ("rax", QWORD),
            0
        ),
        SmtResult::BothPossible
    );
}

// ---------------------------------------------------------------------
// `:=` — assignment without seeding.
//
// The operator radare2 writes every flag of every ISA with, and the one
// the ladder in `r2smt_slicer::lift` gates on. What it *is* belongs
// here; whether a flag-setting instruction may take the ESIL rung is a
// separate decision one layer up.
// ---------------------------------------------------------------------

#[test]
fn test_an_unseeding_assignment_preserves_the_compare_context() {
    // `==(equal) ; 1,rcx,:= ; $z` → 1. The write to `rcx` passes over
    // the context untouched, so `$z` still describes the compare.
    assert_eq!(
        solve_esil(
            "5,rax,==,1,rcx,:=,$z,zf,=",
            &[("rax", 5, QWORD)],
            ("ZF", 1),
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_seeding_assignment_overwrites_the_compare_context() {
    // The same string with `=` instead answers 0, because the write of
    // a non-zero value re-seeds. This pair is the entire difference
    // between the two operators, and either one alone passes on an
    // implementation that treats them as synonyms.
    assert_eq!(
        solve_esil(
            "5,rax,==,1,rcx,=,$z,zf,=",
            &[("rax", 5, QWORD)],
            ("ZF", 1),
            0
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_an_unseeding_assignment_still_writes_its_destination() {
    // It is an assignment, not a no-op: only the seeding differs.
    assert_eq!(
        solve_esil("0x33,rcx,:=", &[("rcx", 0x11, QWORD)], ("rcx", QWORD), 0x33),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// The gate itself. `lift_slice_with`'s third argument opens the ESIL
// rung for flag-setting instructions — on x86 only.
// ---------------------------------------------------------------------

/// One `cmp eax, 1` carrying the ESIL string radare2 6.1.8 emits for it,
/// lifted as a whole slice so the ladder's decision is what is observed.
fn cmp_slice(esil: &str, mnemonic: &str, ops: &[&str]) -> r2smt_slicer::Slice {
    use r2smt_ir::program::{Instruction, Operand, OperandKind};
    r2smt_slicer::Slice {
        branch: branch(),
        instructions: vec![Instruction {
            address: Address::new(0x1000),
            size: 4,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands: ops
                .iter()
                .map(|raw| Operand {
                    raw: (*raw).into(),
                    kind: OperandKind::Register,
                })
                .collect(),
            esil: Some(esil.into()),
            pcode: None,
            is_thumb: false,
        }],
        status: SliceStatus::Complete,
        roots: Vec::new(),
        merges: Vec::new(),
        treat_truncation_as_inputs: false,
    }
}

/// Which rung lifted this `cmp`, told apart by a difference that is
/// itself the point: the per-mnemonic x86 handler leaves `PF` (and `OF`)
/// as `Expr::Unknown`, while radare2's ESIL spells `$p` out and the
/// machine models it exactly. So an `Unknown` in the result means the
/// per-mnemonic handler ran.
fn took_the_esil_rung(lifted: &r2smt_slicer::LiftedSlice) -> bool {
    !format!("{:?}", lifted.statements).contains("Unknown")
}

#[test]
fn test_the_gate_off_sends_a_flag_setting_x86_instruction_to_its_handler() {
    let slice = cmp_slice(
        "1,eax,==,$z,zf,:=,32,$b,cf,:=,$p,pf,:=,31,$s,sf,:=",
        "cmp",
        &["eax", "1"],
    );
    assert!(!took_the_esil_rung(&r2smt_slicer::lift_slice_with(
        &slice,
        Arch::X86_64,
        false
    )));
}

#[test]
fn test_the_gate_on_lets_a_flag_setting_x86_instruction_take_the_esil_rung() {
    let slice = cmp_slice(
        "1,eax,==,$z,zf,:=,32,$b,cf,:=,$p,pf,:=,31,$s,sf,:=",
        "cmp",
        &["eax", "1"],
    );
    assert!(took_the_esil_rung(&r2smt_slicer::lift_slice_with(
        &slice,
        Arch::X86_64,
        true
    )));
}

#[test]
fn test_the_gate_on_still_refuses_arm() {
    // The one contract that cannot be relaxed. radare2 emits ARM's
    // architectural carry (`64,$b,!,cf,:=`) where this pipeline stores
    // its inverse, so lifting this faithfully sends every unsigned ARM
    // branch to the opposite arm — a wrong verdict with full confidence,
    // not a lost one.
    let slice = cmp_slice(
        "x1,x0,==,$z,zf,:=,63,$s,nf,:=,64,$b,!,cf,:=,63,$o,vf,:=",
        "cmp",
        &["x0", "x1"],
    );
    assert!(!took_the_esil_rung(&r2smt_slicer::lift_slice_with(
        &slice,
        Arch::Aarch64,
        true
    )));
}

// ---------------------------------------------------------------------
// The flag-ordering invariant, arriving in the ESIL machine.
//
// `ssa_convert` renames reads by statement *position*, and every flag
// statement is emitted after the destination write, so a flag whose
// expression is inlined has its references to the destination rewritten
// to the version that write just created.
//
// Found by the differential harness rather than by a test, and the
// reason is worth keeping: each output is individually correct. The
// destination alone agrees with the per-mnemonic handler, and the flag
// alone agrees. Only comparing the two *together* exposes it, which no
// single-output contract can do. It was 399 of 540 disagreements on one
// x86-64 sample.
// ---------------------------------------------------------------------

#[test]
fn test_a_flag_reads_the_value_written_and_not_the_value_written_twice() {
    // The real `sub rsp, 0x20` string. With `rsp` bound to 32 the
    // result is 0, so `ZF` is 1. Reading the post-write `rsp` instead
    // computes `(32 - 32) - 32`, which is not zero — the destination
    // subtracted twice.
    assert_eq!(
        solve_esil(
            "32,rsp,-=,63,$s,sf,:=,$z,zf,=",
            &[("rsp", 32, QWORD)],
            ("ZF", 1),
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_destination_itself_is_written_once() {
    // The other half of the pair, and the reason the bug survived: this
    // side was always right. A contract on the destination alone passes
    // on the broken ordering.
    assert_eq!(
        solve_esil(
            "32,rsp,-=,$z,zf,=",
            &[("rsp", 96, QWORD)],
            ("rsp", QWORD),
            64
        ),
        SmtResult::AlwaysTrue
    );
}
