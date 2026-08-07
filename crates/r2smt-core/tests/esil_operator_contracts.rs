//! Solver-backed contracts for the ESIL stack machine.
//!
//! The machine had ~45 tests before this file and all of them assert on
//! the *shape* of an `Expr` tree; the per-mnemonic handlers have ~750
//! that bind concrete values and ask a solver. That asymmetry was the
//! argument for keeping `:=` gated — and since 2026-08-07 the gate is
//! *open* by default on x86, so it stopped being a reason to wait and
//! became debt on the path that actually runs. These contracts are the
//! repayment: every operator the machine supports now has one.
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
//! The multi-flag strings r2 really emits
//! (`1,eax,==,$z,zf,:=,32,$b,cf,:=,…`) are reachable now: each flag
//! write in those is `:=` precisely so it does not clobber the context
//! the next flag token still reads, and `:=` lexes since `4965d879`.
//! The `AArch64` carry contract at the bottom of this file is one such
//! string, verbatim.
//!
//! One trap of measurement is worth naming twice, because it has now
//! produced two wrong beliefs about this machine. r2's own output
//! redirection eats `>`, `>=` and `>>`, so an unquoted `ae 4,rax,>>=`
//! writes a file named `=` and leaves the register untouched — which is
//! exactly the evidence that put `>>=` on the "radare2 leaves it
//! intact" list it did not belong on. Quote every probe.
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
    solve_esil_arch(esil, Arch::X86_64, bindings, observed, expected)
}

fn solve_esil_arch(
    esil: &str,
    arch: Arch,
    bindings: &[(&str, u128, u16)],
    observed: (&str, u16),
    expected: u128,
) -> SmtResult {
    let lifted = lift_esil(esil, arch)
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
        arch,
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

// ---------------------------------------------------------------------
// The width a comparison seeds its context at.
//
// radare2 spells every immediate as a `ut64`, so `cmp ecx, 0xffffffff`
// arrives as `18446744073709551615,ecx,==`. Taking the operation width
// as the wider of the two operands lets that spelling decide the
// semantics, and the machine then asked whether a zero-extended `ecx`
// could equal `2^64 - 1` — unsatisfiable, so `ZF` was a constant zero
// where the architecture sets it. Found by the differential harness on
// six real `cmp` sites of one x86-64 sample, five of them against
// memory.
// ---------------------------------------------------------------------

#[test]
fn test_a_comparison_seeds_at_the_destination_width_not_the_immediate_spelling() {
    // The real string from `cmp ecx, 0xffffffff`. `ecx` holding
    // `0xffffffff` *is* equal to the immediate at 32 bits, so `ZF` is 1.
    assert_eq!(
        solve_esil(
            "18446744073709551615,ecx,==,$z,zf,:=",
            &[("rcx", 0xffff_ffff, QWORD)],
            ("ZF", 1),
            1
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_comparison_at_the_destination_width_ignores_the_parent_above_it() {
    // The other direction, so the fix cannot be "answer 1 regardless":
    // bits 63..32 of the parent are outside a 32-bit compare, and `ecx`
    // is zero here.
    assert_eq!(
        solve_esil(
            "18446744073709551615,ecx,==,$z,zf,:=",
            &[("rcx", 0xffff_ffff_0000_0000, QWORD)],
            ("ZF", 1),
            0
        ),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// AArch64. Two rules the machine did not have, both found by the
// differential harness on one sample and both *reachable in production*
// on this ISA in the first case: `mov x8, xzr` carries no `:=`, so its
// ESIL wins the ladder over the per-mnemonic handler.
// ---------------------------------------------------------------------

#[test]
fn test_the_zero_register_reads_as_zero_rather_than_a_free_value() {
    // `mov x8, xzr` is how AArch64 spells "zero this register", and its
    // ESIL is a plain `xzr,x8,=`. Resolving `xzr` as an ordinary
    // variable left `x8` free, so `mov x0, xzr; cbz x0, …` — the opaque
    // predicate the per-mnemonic path resolves precisely — came back
    // `BothPossible`. 142 of 234 disagreements on one sample.
    assert_eq!(
        solve_esil_arch(
            "xzr,x8,=",
            Arch::Aarch64,
            &[("x8", u128::from(u64::MAX), QWORD)],
            ("x8", QWORD),
            0
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_arm_carry_is_stored_as_its_complement() {
    // The real `cmp x9, x0` string. radare2 writes ARM's architectural
    // `C` — here `!borrow` — and this pipeline stores the complement so
    // one `lift_branch_condition` can serve both ISAs. With `x9 < x0`
    // unsigned the architecture clears `C`, so the stored bit is 1.
    //
    // Getting this wrong inverts the branch rather than losing it: a
    // `b.hs` after the compare resolves to the arm the machine does not
    // take. 77 of the 92 remaining disagreements on that sample.
    assert_eq!(
        solve_esil_arch(
            "x0,x9,==,$z,zf,:=,63,$s,nf,:=,64,$b,!,cf,:=",
            Arch::Aarch64,
            &[("x9", 1, QWORD), ("x0", 2, QWORD)],
            ("CF", 1),
            1
        ),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// The comparators. `machine.rs` implements `>` and `>=` by *swapping*
// operands rather than by a distinct node, and nothing pinned either
// that swap or the signedness. Every expectation below was measured on
// r2 6.1.8, with the `"…"` quoting that `>` / `>=` require.
// ---------------------------------------------------------------------

#[test]
fn test_the_less_than_takes_the_second_token_as_the_left_hand_side() {
    // `ae 4,10,<` → 0, i.e. `10 < 4` and never `4 < 10`.
    assert_eq!(
        solve_esil("4,10,<,rax,=", &[], ("rax", QWORD), 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_less_than_is_signed_and_not_unsigned() {
    // `ae 1,0x8000000000000000,<` → 1. Signed, `INT64_MIN < 1` holds;
    // read unsigned the same bits are the largest value there is and the
    // answer flips. A whole half of the number line, silently.
    assert_eq!(
        solve_esil("1,0x8000000000000000,<,rax,=", &[], ("rax", QWORD), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_less_or_equal_admits_equality_where_less_than_does_not() {
    // `ae 7,7,<=` → 1 against `ae 7,7,<` → 0. The boundary is the only
    // thing separating the two arms.
    assert_eq!(
        solve_esil("7,7,<=,rax,=", &[], ("rax", QWORD), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_less_or_equal_is_signed_and_not_unsigned() {
    assert_eq!(
        solve_esil("1,0x8000000000000000,<=,rax,=", &[], ("rax", QWORD), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_greater_than_takes_the_second_token_as_the_left_hand_side() {
    // `ae 4,10,>` → 1. This is the operand swap; inverted it answers 0.
    assert_eq!(
        solve_esil("4,10,>,rax,=", &[], ("rax", QWORD), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_greater_than_is_signed_and_not_unsigned() {
    // `ae 1,0x8000000000000000,>` → 0.
    assert_eq!(
        solve_esil("1,0x8000000000000000,>,rax,=", &[], ("rax", QWORD), 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_greater_or_equal_admits_equality_where_greater_than_does_not() {
    // `ae 7,7,>=` → 1 against `ae 7,7,>` → 0.
    assert_eq!(
        solve_esil("7,7,>=,rax,=", &[], ("rax", QWORD), 1),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_greater_or_equal_is_signed_and_not_unsigned() {
    assert_eq!(
        solve_esil("1,0x8000000000000000,>=,rax,=", &[], ("rax", QWORD), 0),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// `!=` — the negating assignment. radare2's own help calls it "negate
// all bits", which is wrong: `ar rax=0x0f; ae rax,!=` leaves 0, not
// 0xfffffffffffffff0. It is a logical NOT written back to the register.
// The semantics were corrected against the tool and then had no test.
// ---------------------------------------------------------------------

#[test]
fn test_the_negating_assignment_is_a_logical_not_and_not_a_bit_complement() {
    // A bit complement of 0x0f would be 0xfffffffffffffff0.
    assert_eq!(
        solve_esil("rax,!=", &[("rax", 0x0f, QWORD)], ("rax", QWORD), 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_the_negating_assignment_turns_zero_into_one() {
    // The other half: without it a "write zero" implementation passes
    // the test above and is still wrong.
    assert_eq!(
        solve_esil("rax,!=", &[("rax", 0, QWORD)], ("rax", QWORD), 1),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// Predicated blocks. The highest-prevalence untested family by a wide
// margin: 494 of the 5 339 ESIL strings in one x86-64 sample carry
// `?{`, against zero for the comparators above. The machine lowers a
// block by wrapping each `Assign` inside it in `Ite(cond, new, old)`,
// and nothing pinned either arm.
// ---------------------------------------------------------------------

#[test]
fn test_a_taken_block_writes_its_body() {
    // `ar rax=7; ae 1,?{,0x2a,rax,=,}` → 0x2a.
    assert_eq!(
        solve_esil(
            "1,?{,0x2a,rax,=,}",
            &[("rax", 7, QWORD)],
            ("rax", QWORD),
            0x2a
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_an_untaken_block_leaves_the_destination_alone() {
    // The other arm of the `Ite`, and the one that fabricates if wrong:
    // a block lowered unconditionally would answer 0x2a here.
    assert_eq!(
        solve_esil("0,?{,0x2a,rax,=,}", &[("rax", 7, QWORD)], ("rax", QWORD), 7),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_block_guard_is_zero_versus_non_zero_and_not_the_low_bit() {
    // `ar rbx=2; ae rbx,?{,…}` takes the block: the guard is "non-zero",
    // so reading it as bit 0 would skip every even condition.
    assert_eq!(
        solve_esil(
            "rbx,?{,0x2a,rax,=,}",
            &[("rax", 7, QWORD), ("rbx", 2, QWORD)],
            ("rax", QWORD),
            0x2a
        ),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// The bare binaries. `ae 4,10,<op>` answers `10 <op> 4`. `-` and `<<`
// were already pinned; these are the rest, non-commutative first
// because those are where the order is a wrong value rather than a
// no-op. `>>` is one of the three operators r2's own output
// redirection eats when a probe is not quoted.
// ---------------------------------------------------------------------

#[test]
fn test_a_bare_divide_takes_the_second_token_as_the_dividend() {
    // `ae 4,10,/` → 2, i.e. `10 / 4` and never `4 / 10` (which is 0).
    assert_eq!(
        solve_esil("4,10,/,rax,=", &[], ("rax", QWORD), 2),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_remainder_takes_the_second_token_as_the_dividend() {
    // `ae 4,10,%` → 2, against `4 % 10` = 4.
    assert_eq!(
        solve_esil("4,10,%,rax,=", &[], ("rax", QWORD), 2),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_logical_shift_right_shifts_the_second_token() {
    // `ae 2,160,>>` → 0x28. Needs the `"…"` quoting to probe at all.
    assert_eq!(
        solve_esil("2,160,>>,rax,=", &[], ("rax", QWORD), 40),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_shift_right_is_logical_and_not_arithmetic() {
    // The sign bit must shift out as zero. An arithmetic shift would
    // answer 0xffffffffffffffff — a wrong value on every negative
    // operand, and invisible to the positive case above.
    assert_eq!(
        solve_esil(
            "4,0x8000000000000000,>>,rax,=",
            &[],
            ("rax", QWORD),
            0x0800_0000_0000_0000
        ),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_add_sums_both_tokens() {
    assert_eq!(
        solve_esil("4,10,+,rax,=", &[], ("rax", QWORD), 14),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_multiply_multiplies_both_tokens() {
    assert_eq!(
        solve_esil("4,10,*,rax,=", &[], ("rax", QWORD), 40),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_and_masks_both_tokens() {
    assert_eq!(
        solve_esil("4,10,&,rax,=", &[], ("rax", QWORD), 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_or_merges_both_tokens() {
    assert_eq!(
        solve_esil("4,10,|,rax,=", &[], ("rax", QWORD), 14),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_bare_xor_differences_both_tokens() {
    assert_eq!(
        solve_esil("4,10,^,rax,=", &[], ("rax", QWORD), 14),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// The remaining compound assigns. `-=` and `<<=` were pinned; these
// five were not. `>>=` is here rather than on the out-by-decision list
// — see the note on it below.
// ---------------------------------------------------------------------

#[test]
fn test_a_compound_multiply_multiplies_the_target() {
    // `ar rax=10; ae 4,rax,*=` → 0x28.
    assert_eq!(
        solve_esil("4,rax,*=", &[("rax", 10, QWORD)], ("rax", QWORD), 40),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compound_and_masks_the_target() {
    assert_eq!(
        solve_esil("4,rax,&=", &[("rax", 10, QWORD)], ("rax", QWORD), 0),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compound_or_merges_into_the_target() {
    assert_eq!(
        solve_esil("4,rax,|=", &[("rax", 10, QWORD)], ("rax", QWORD), 14),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compound_xor_differences_the_target() {
    assert_eq!(
        solve_esil("4,rax,^=", &[("rax", 10, QWORD)], ("rax", QWORD), 14),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_compound_shift_right_shifts_the_target_by_the_value() {
    // `ar rax=0xff; ae 4,rax,>>=` → 0x0f.
    //
    // This operator was on the out-by-decision list, on the grounds that
    // r2 6.1.8 "leaves the register intact for every input probed". It
    // does not, and the claim is an artifact of the trap documented at
    // the top of this file: unquoted, the `>>` in `ae 4,rax,>>=` is r2's
    // own output redirection, which writes a file named `=` and leaves
    // the register untouched. Reproduced both ways before writing this.
    assert_eq!(
        solve_esil("4,rax,>>=", &[("rax", 0xff, QWORD)], ("rax", QWORD), 0x0f),
        SmtResult::AlwaysTrue
    );
}

// ---------------------------------------------------------------------
// Division by zero. Pinned as a *divergence*, not matched: r2 answers
// zero, SMT-LIB defines `bvudiv` by zero as all-ones and `bvurem` by
// zero as the dividend. Matching r2 would mean an `Ite` guard on every
// divide to model a case that raises #DE on the real machine, so the
// contract records what this pipeline does and why the gap is
// deliberate.
// ---------------------------------------------------------------------

#[test]
fn test_a_divide_by_zero_takes_the_smtlib_value_and_not_radare2_s() {
    assert_eq!(
        solve_esil("0,10,/,rax,=", &[], ("rax", QWORD), u128::from(u64::MAX)),
        SmtResult::AlwaysTrue
    );
}

#[test]
fn test_a_remainder_by_zero_yields_the_dividend() {
    assert_eq!(
        solve_esil("0,10,%,rax,=", &[], ("rax", QWORD), 10),
        SmtResult::AlwaysTrue
    );
}
