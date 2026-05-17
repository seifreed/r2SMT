#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::Address;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use super::*;

fn op(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn insn(mnem: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: Address(0),
        size: 0,
        bytes: vec![],
        mnemonic: mnem.into(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Test-only adapter so the existing x86 assertions stay terse.
/// `AArch64`-specific tests use `analyze(..., Arch::Aarch64)`.
fn ax86(i: &Instruction) -> InstructionEffect {
    analyze(i, Arch::X86_64)
}

#[test]
fn canonical_register_covers_aliases() {
    for alias in ["rax", "eax", "ax", "al", "ah"] {
        assert_eq!(canonical_register(alias, Arch::X86_64), Some("rax"));
    }
    assert_eq!(canonical_register("r8d", Arch::X86_64), Some("r8"));
    assert_eq!(canonical_register("xmm0", Arch::X86_64), None);
    assert_eq!(canonical_register("ptr", Arch::X86_64), None);
    assert_eq!(canonical_register("0x10", Arch::X86_64), None);
}

#[test]
fn registers_in_operand_extracts_from_memory_expression() {
    let memory = op("dword ptr [rbp + rax*2 + 8]", OperandKind::Memory);
    let regs = registers_in_operand(&memory, Arch::X86_64);
    assert!(regs.contains(&"rbp"));
    assert!(regs.contains(&"rax"));
    assert_eq!(regs.len(), 2);
}

#[test]
fn canonical_register_dispatches_on_arch() {
    // `sp` is the 16-bit alias of `rsp` on x86, the 64-bit stack
    // pointer on AArch64, and an alias of `r13` on AArch32. Same
    // string, three different parents — proves the arch parameter
    // is consulted instead of an ISA-blind table.
    assert_eq!(canonical_register("sp", Arch::X86_64), Some("rsp"));
    assert_eq!(canonical_register("sp", Arch::Aarch64), Some("sp"));
    assert_eq!(canonical_register("sp", Arch::Arm), Some("r13"));
}

#[test]
fn canonical_register_rejects_names_from_other_isas() {
    // x86 names must not resolve under AArch64 / AArch32 and
    // vice versa, otherwise cross-ISA disassembly noise pollutes
    // the data-flow graph.
    assert_eq!(canonical_register("rax", Arch::Aarch64), None);
    assert_eq!(canonical_register("rax", Arch::Arm), None);
    assert_eq!(canonical_register("x0", Arch::X86_64), None);
    assert_eq!(canonical_register("x0", Arch::Arm), None);
}

#[test]
fn registers_in_operand_dispatches_on_arch() {
    // `[x0, x1]` is an AArch64 memory expression. Tokenising it
    // under X86_64 yields nothing because x0/x1 are not x86 GPR
    // names; under AArch64 it yields both registers.
    let memory = op("[x0, x1]", OperandKind::Memory);
    assert!(registers_in_operand(&memory, Arch::X86_64).is_empty());
    let aa64 = registers_in_operand(&memory, Arch::Aarch64);
    assert!(aa64.contains(&"x0"));
    assert!(aa64.contains(&"x1"));
}

#[test]
fn registers_in_operand_surfaces_arm_simd_names() {
    // `v1.2d` is a common AArch64 NEON operand spelling. The
    // tokenizer splits on the dot, so `v1` should surface even
    // when paired with a width suffix the slicer doesn't model.
    let neon = op("v1.2d", OperandKind::Register);
    let aa64 = registers_in_operand(&neon, Arch::Aarch64);
    assert!(aa64.contains(&"v1"));
    // Under AArch32 the same string `v1` is the AAPCS alias for
    // r4 (not a NEON register — NEON is qN/dN/sN under AArch32).
    // The tokenizer therefore surfaces r4, not a synthetic v
    // parent.
    let arm = registers_in_operand(&neon, Arch::Arm);
    assert!(arm.contains(&"r4"));
    assert!(!arm.contains(&"v1"));
}

#[test]
fn registers_in_operand_collapses_arm32_d_to_v_parent() {
    // A NEON load list like `{d0, d1}` should surface a single
    // parent (v0) — d0 and d1 are both halves of v0. The slicer
    // can then see that subsequent reads of q0 or s2 also touch
    // the same data-flow node.
    let list = op("{d0, d1}", OperandKind::Register);
    let arm = registers_in_operand(&list, Arch::Arm);
    assert_eq!(arm, vec!["v0"]);
}

#[test]
fn canonical_register_recognises_aarch64_simd_aliases() {
    for alias in ["v0", "q0", "d0", "s0", "h0", "b0"] {
        assert_eq!(canonical_register(alias, Arch::Aarch64), Some("v0"));
    }
    assert_eq!(canonical_register("d31", Arch::Aarch64), Some("v31"));
    // x86 SIMD still resolves to None — adding ARM SIMD does not
    // accidentally widen the x86 table.
    assert_eq!(canonical_register("xmm0", Arch::X86_64), None);
}

#[test]
fn mov_reg_imm_defines_no_flags() {
    let e = ax86(&insn(
        "mov",
        vec![
            op("eax", OperandKind::Register),
            op("0x10", OperandKind::Immediate),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Mov);
    assert_eq!(e.defs, vec!["rax"]);
    assert!(e.uses.is_empty());
    assert!(!e.defines_flags);
}

#[test]
fn xor_same_register_is_zero_idiom() {
    let e = ax86(&insn(
        "xor",
        vec![
            op("eax", OperandKind::Register),
            op("eax", OperandKind::Register),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Xor);
    assert_eq!(e.defs, vec!["rax"]);
    assert!(e.uses.is_empty(), "zero idiom must not depend on prior eax");
    assert!(e.defines_flags);
}

#[test]
fn xor_different_registers_uses_both() {
    let e = ax86(&insn(
        "xor",
        vec![
            op("eax", OperandKind::Register),
            op("ebx", OperandKind::Register),
        ],
    ));
    assert_eq!(e.defs, vec!["rax"]);
    assert_eq!(e.uses, vec!["rax", "rbx"]);
    assert!(e.defines_flags);
}

#[test]
fn cmp_uses_both_operands_no_def() {
    let e = ax86(&insn(
        "cmp",
        vec![
            op("eax", OperandKind::Register),
            op("2", OperandKind::Immediate),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Cmp);
    assert!(e.defs.is_empty());
    assert_eq!(e.uses, vec!["rax"]);
    assert!(e.defines_flags);
}

#[test]
fn test_uses_both_operands_no_def() {
    let e = ax86(&insn(
        "test",
        vec![
            op("eax", OperandKind::Register),
            op("eax", OperandKind::Register),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Test);
    assert!(e.defs.is_empty());
    assert_eq!(e.uses, vec!["rax"]);
    assert!(e.defines_flags);
}

#[test]
fn lea_does_not_access_memory() {
    let e = ax86(&insn(
        "lea",
        vec![
            op("eax", OperandKind::Register),
            op("[rbp - 4]", OperandKind::Memory),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Lea);
    assert_eq!(e.defs, vec!["rax"]);
    assert_eq!(e.uses, vec!["rbp"]);
    assert!(!e.has_memory_access);
    assert!(!e.defines_flags);
}

#[test]
fn mov_load_from_unresolved_memory_flags_has_memory_access() {
    // `[rax]` is an indirect deref — not a recognised stack slot,
    // so the slicer still truncates on it.
    let e = ax86(&insn(
        "mov",
        vec![
            op("eax", OperandKind::Register),
            op("[rax]", OperandKind::Memory),
        ],
    ));
    assert!(e.has_memory_access);
    assert!(e.stack_uses.is_empty());
}

#[test]
fn mov_load_from_stack_slot_uses_virtual_slot() {
    // `[rbp - 4]` is a Phase C stack slot — surfaced via
    // `stack_uses`, not `has_memory_access`.
    let e = ax86(&insn(
        "mov",
        vec![
            op("eax", OperandKind::Register),
            op("[rbp - 4]", OperandKind::Memory),
        ],
    ));
    assert!(!e.has_memory_access);
    assert_eq!(e.stack_uses, vec!["stk_rbp_-4".to_string()]);
    assert!(e.defs.contains(&"rax"));
}

#[test]
fn mov_store_to_stack_slot_is_a_virtual_def() {
    let e = ax86(&insn(
        "mov",
        vec![
            op("dword ptr [rbp - 8]", OperandKind::Memory),
            op("5", OperandKind::Immediate),
        ],
    ));
    assert!(!e.has_memory_access);
    assert_eq!(e.stack_defs, vec!["stk_rbp_-8".to_string()]);
    assert!(
        e.defs.is_empty(),
        "stack-slot stores must not define a register"
    );
}

#[test]
fn stack_slot_rejects_dynamic_indexing() {
    let dyn_op = op("[rbp + rax*4]", OperandKind::Memory);
    assert!(stack_slot(&dyn_op).is_none());
    let abs_op = op("[rax]", OperandKind::Memory);
    assert!(stack_slot(&abs_op).is_none());
}

#[test]
fn stack_slot_recognises_widths() {
    let (name, bits) = stack_slot(&op("byte ptr [rbp - 1]", OperandKind::Memory)).unwrap();
    assert_eq!(name, "stk_rbp_-1");
    assert_eq!(bits, 8);
    let (_, bits) = stack_slot(&op("qword ptr [rsp + 0x10]", OperandKind::Memory)).unwrap();
    assert_eq!(bits, 64);
}

#[test]
fn xor_sub_register_is_not_zero_idiom() {
    // `xor ah, al` mixes two distinct sub-registers of rax. It is
    // NOT a zero idiom; the result depends on the current bytes of
    // rax. Regression for a Phase D false-positive that surfaced
    // on APT10 ANELLOADER (every `xor ah, al` was being treated
    // as a constant zero).
    let e = ax86(&insn(
        "xor",
        vec![
            op("ah", OperandKind::Register),
            op("al", OperandKind::Register),
        ],
    ));
    // Treated as plain arithmetic — defines rax (canonical of ah),
    // uses rax (both operands canonicalise to it), sets flags.
    assert!(!e.uses.is_empty(), "xor ah, al must read rax");
    assert!(e.defines_flags);
    // The defs set has rax because ah's write touches the rax
    // virtual register; uses must also include rax (the source).
    assert!(e.defs.contains(&"rax"));
    assert!(e.uses.contains(&"rax"));
}

#[test]
fn xor_eax_eax_is_still_zero_idiom() {
    let e = ax86(&insn(
        "xor",
        vec![
            op("eax", OperandKind::Register),
            op("eax", OperandKind::Register),
        ],
    ));
    assert_eq!(e.uses, Vec::<&'static str>::new());
    assert_eq!(e.defs, vec!["rax"]);
    assert!(e.defines_flags);
}

#[test]
fn imul_two_operand_is_arithmetic() {
    let e = ax86(&insn(
        "imul",
        vec![
            op("eax", OperandKind::Register),
            op("eax", OperandKind::Register),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Imul);
    assert_eq!(e.defs, vec!["rax"]);
    assert_eq!(e.uses, vec!["rax"]);
    assert!(e.defines_flags);
}

#[test]
fn call_is_flagged() {
    let e = ax86(&insn("call", vec![op("0x401000", OperandKind::Immediate)]));
    assert_eq!(e.kind, InstructionKind::Call);
    assert!(e.is_call);
}

#[test]
fn unknown_mnemonic_is_other() {
    let e = ax86(&insn("vpxor", vec![]));
    assert_eq!(e.kind, InstructionKind::Other);
    assert!(!e.is_call);
}

// --- AArch64 ---

fn aa64(i: &Instruction) -> InstructionEffect {
    analyze(i, Arch::Aarch64)
}

#[test]
fn aarch64_mov_defines_destination_without_flags() {
    let e = aa64(&insn(
        "mov",
        vec![
            op("x0", OperandKind::Register),
            op("x1", OperandKind::Register),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Mov);
    assert_eq!(e.defs, vec!["x0"]);
    assert_eq!(e.uses, vec!["x1"]);
    assert!(!e.defines_flags);
}

#[test]
fn aarch64_add_is_3op_no_flags_adds_sets_flags() {
    let plain = aa64(&insn(
        "add",
        vec![
            op("x0", OperandKind::Register),
            op("x1", OperandKind::Register),
            op("x2", OperandKind::Register),
        ],
    ));
    assert_eq!(plain.kind, InstructionKind::Add);
    assert_eq!(plain.defs, vec!["x0"]);
    assert_eq!(plain.uses, vec!["x1", "x2"]);
    assert!(!plain.defines_flags);

    let flag_set = aa64(&insn(
        "adds",
        vec![
            op("x0", OperandKind::Register),
            op("x1", OperandKind::Register),
            op("x2", OperandKind::Register),
        ],
    ));
    assert!(flag_set.defines_flags);
}

#[test]
fn aarch64_cmp_uses_both_no_def() {
    let e = aa64(&insn(
        "cmp",
        vec![
            op("x0", OperandKind::Register),
            op("#0", OperandKind::Immediate),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Cmp);
    assert!(e.defs.is_empty());
    assert_eq!(e.uses, vec!["x0"]);
    assert!(e.defines_flags);
}

#[test]
fn aarch64_b_cond_is_jcc() {
    let e = aa64(&insn("b.eq", vec![op("0x401080", OperandKind::Immediate)]));
    assert_eq!(e.kind, InstructionKind::Jcc);
}

#[test]
fn aarch64_unconditional_b_is_jmp() {
    let e = aa64(&insn("b", vec![op("0x401080", OperandKind::Immediate)]));
    assert_eq!(e.kind, InstructionKind::Jmp);
}

#[test]
fn aarch64_bl_is_call() {
    let e = aa64(&insn("bl", vec![op("0x402000", OperandKind::Immediate)]));
    assert_eq!(e.kind, InstructionKind::Call);
    assert!(e.is_call);
}

#[test]
fn aarch64_w_subregister_canonicalises_to_x() {
    let e = aa64(&insn(
        "mov",
        vec![
            op("w0", OperandKind::Register),
            op("w1", OperandKind::Register),
        ],
    ));
    // AArch64 32-bit subregisters share the parent name; defs/uses
    // collapse onto the 64-bit family for slicing.
    assert_eq!(e.defs, vec!["x0"]);
    assert_eq!(e.uses, vec!["x1"]);
}

#[test]
fn x86_mnemonics_under_aarch64_are_other() {
    // `xor` is x86; AArch64 uses `eor`. Under Arch::Aarch64 the
    // analyzer must classify `xor` as Other so the slicer
    // truncates instead of misinterpreting it.
    let e = aa64(&insn(
        "xor",
        vec![
            op("x0", OperandKind::Register),
            op("x0", OperandKind::Register),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Other);
}

#[test]
fn shifts_define_dest_and_flags() {
    let e = ax86(&insn(
        "shl",
        vec![
            op("eax", OperandKind::Register),
            op("4", OperandKind::Immediate),
        ],
    ));
    assert_eq!(e.kind, InstructionKind::Shl);
    assert_eq!(e.defs, vec!["rax"]);
    assert_eq!(e.uses, vec!["rax"]);
    assert!(e.defines_flags);
}
