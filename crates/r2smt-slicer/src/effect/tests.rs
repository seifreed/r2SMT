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
    // `xmm` now resolves (P40-b); MMX stays unmodelled. `st0` is the
    // ESIL spelling of the x87 stack, which resolves only under the
    // bare `st` token that the disassembly spelling `st(0)` yields.
    assert_eq!(canonical_register("st0", Arch::X86_64), None);
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
    // Adding ARM SIMD does not accidentally widen the x86 table: MMX
    // stays None, and so does the ESIL-only `st0` spelling of the x87
    // stack (whose disassembly spelling `st(0)` tokenises to `st`).
    assert_eq!(canonical_register("mm0", Arch::X86_64), None);
    assert_eq!(canonical_register("st0", Arch::X86_64), None);
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
fn mov_load_from_register_indirect_is_modellable_memory() {
    // `[rax]` is register-indirect — a memory access the byte model
    // can build, so it flags `has_memory_access` but is NOT
    // unmodellable (resolves without `--allow-memory`). The address
    // register is a data-flow input.
    let mem = op("[rax]", OperandKind::Memory);
    let e = ax86(&insn(
        "mov",
        vec![op("eax", OperandKind::Register), mem.clone()],
    ));
    assert!(e.has_memory_access);
    assert!(!has_unmodellable_memory(
        std::slice::from_ref(&mem),
        Arch::X86_64
    ));
    assert!(e.uses.contains(&"rax"));
}

#[test]
fn mov_load_from_stack_slot_is_modellable_memory() {
    // `[rbp - 4]` now lifts through the byte model like any other
    // memory operand: `has_memory_access` is set and the base register
    // `rbp` is a data-flow input, not a virtual stack slot.
    let e = ax86(&insn(
        "mov",
        vec![
            op("eax", OperandKind::Register),
            op("[rbp - 4]", OperandKind::Memory),
        ],
    ));
    assert!(e.has_memory_access);
    assert!(e.uses.contains(&"rbp"));
    assert!(e.defs.contains(&"rax"));
}

#[test]
fn mov_store_to_stack_slot_uses_address_register_no_reg_def() {
    // `mov [rbp - 8], 5` — a byte-model store: no register def, the
    // address register `rbp` is a use.
    let e = ax86(&insn(
        "mov",
        vec![
            op("dword ptr [rbp - 8]", OperandKind::Memory),
            op("5", OperandKind::Immediate),
        ],
    ));
    assert!(e.has_memory_access);
    assert!(e.uses.contains(&"rbp"));
    assert!(
        e.defs.is_empty(),
        "memory stores must not define a register"
    );
}

#[test]
fn segment_prefixed_memory_is_unmodellable() {
    // `fs:[0x30]` addresses `fs_base + 0x30`; the byte model declines
    // it, so it must be gated (unmodellable) rather than resolved as
    // an absolute `[0x30]`.
    let seg = op("qword fs:[0x30]", OperandKind::Memory);
    assert!(has_unmodellable_memory(
        std::slice::from_ref(&seg),
        Arch::X86_64
    ));
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
    let e = ax86(&insn("cpuid", vec![]));
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

#[test]
fn memory_operand_width_reads_simd_size_prefixes() {
    assert_eq!(memory_operand_width("xmmword ptr [rsi]"), Some(128));
    assert_eq!(memory_operand_width("ymmword [rsi]"), Some(256));
    assert_eq!(memory_operand_width("zmmword ptr [rsi]"), Some(512));
    assert_eq!(memory_operand_width("qword ptr [rbp - 8]"), Some(64));
    assert_eq!(memory_operand_width("dword [rax]"), Some(32));
    assert_eq!(memory_operand_width("byte ptr [rdi]"), Some(8));
}

#[test]
fn memory_operand_width_ignores_size_keywords_inside_symbols() {
    // A symbol whose name merely contains a size keyword is not a sized
    // access — the specifier must be the leading token.
    assert_eq!(memory_operand_width("[obj.dword_table]"), None);
    assert_eq!(memory_operand_width("[byte_count]"), None);
    assert_eq!(memory_operand_width("[rax]"), None);
}

#[test]
fn aarch64_packed_add_defines_its_vector_destination() {
    // The soundness contract this replaces: an arranged destination used
    // to canonicalise to `None`, so the instruction passed with empty
    // `defs` while `v1`/`v2` still resolved as `uses`. The slicer then
    // neither truncated nor kept it, and a later read of `v0` bound to a
    // stale definition. Now the packed handler models it, so the
    // definition is reported.
    let i = insn(
        "add",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.defs, vec!["v0"]);
}

#[test]
fn aarch64_packed_add_reads_both_vector_sources() {
    let i = insn(
        "add",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).uses, vec!["v1", "v2"]);
}

#[test]
fn aarch64_packed_add_is_classified_as_simd() {
    let i = insn(
        "add",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).kind, InstructionKind::Simd);
}

#[test]
fn aarch64_packed_floating_point_defines_its_vector_destination() {
    let i = insn(
        "fadd",
        vec![
            op("v0.2d", OperandKind::Register),
            op("v1.2d", OperandKind::Register),
            op("v2.2d", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.defs, vec!["v0"]);
}

#[test]
fn aarch64_packed_vector_instruction_sets_no_flags() {
    let i = insn(
        "add",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    assert!(!analyze(&i, Arch::Aarch64).defines_flags);
}

#[test]
fn aarch64_unmodelled_vector_mnemonic_declines() {
    // Not modelled, so the effect table has to fail closed — otherwise
    // the slicer retains a definition the lifter drops. `fmulx` is the
    // canary now that `fmla` lifts: it is a float multiply extended,
    // which no family claims.
    let i = insn(
        "fmulx",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).kind, InstructionKind::Other);
}

#[test]
fn aarch64_indexed_lane_operand_declines() {
    let i = insn(
        "mov",
        vec![
            op("w0", OperandKind::Register),
            op("v0.s[1]", OperandKind::Memory),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).kind, InstructionKind::Other);
}

#[test]
fn aarch64_multi_register_list_operand_recovers_every_parent() {
    let i = insn(
        "ld4",
        vec![
            op("{v16.4s, v17.4s, v18.4s, v19.4s}", OperandKind::Unknown),
            op("[x2]", OperandKind::Memory),
        ],
    );
    assert_eq!(
        analyze(&i, Arch::Aarch64).defs,
        vec!["v16", "v17", "v18", "v19"]
    );
}

#[test]
fn aarch64_structured_list_of_an_unmodelled_shape_still_declines() {
    // A list the resolver cannot place -- here one whose registers are
    // not consecutive -- must truncate the slice rather than pass with
    // an empty `defs`.
    let i = insn(
        "ld4",
        vec![
            op("{v16.4s, v18.4s, v20.4s, v22.4s}", OperandKind::Unknown),
            op("[x2]", OperandKind::Memory),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).kind, InstructionKind::Other);
}

#[test]
fn aarch64_scalar_arithmetic_is_unaffected_by_the_arrangement_guard() {
    let i = insn(
        "add",
        vec![
            op("x0", OperandKind::Register),
            op("x1", OperandKind::Register),
            op("x2", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.kind, InstructionKind::Add);
    assert_eq!(effect.defs, vec!["x0"]);
}

#[test]
fn aarch64_scalar_floating_point_is_unaffected_by_the_arrangement_guard() {
    let i = insn(
        "fadd",
        vec![
            op("s0", OperandKind::Register),
            op("s1", OperandKind::Register),
            op("s2", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.kind, InstructionKind::Simd);
    assert_eq!(effect.defs, vec!["v0"]);
}

#[test]
fn aarch32_indexed_vector_lane_declines() {
    // AArch32 spells the indexed form without a dot (`d0[1]`), so it
    // reaches the integer arms by a different route than AArch64's.
    let i = insn(
        "vmov",
        vec![
            op("r0", OperandKind::Register),
            op("d0[1]", OperandKind::Memory),
        ],
    );
    assert_eq!(analyze(&i, Arch::Arm).kind, InstructionKind::Other);
}

#[test]
fn aarch32_aapcs_vector_alias_is_still_a_general_purpose_register() {
    // `v1` on AArch32 is the AAPCS alias for `r4`, not a vector
    // register: a bare name carries no arrangement and must survive.
    let i = insn(
        "add",
        vec![
            op("v1", OperandKind::Register),
            op("v2", OperandKind::Register),
            op("v3", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Arm);
    assert_eq!(effect.kind, InstructionKind::Add);
    assert_eq!(effect.defs, vec!["r4"]);
}

#[test]
fn aarch32_packed_integer_defines_its_vector_destination() {
    let i = insn(
        "vadd.i32",
        vec![
            op("q0", OperandKind::Register),
            op("q1", OperandKind::Register),
            op("q2", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Arm);
    assert_eq!(effect.defs, vec!["v0"]);
}

#[test]
fn aarch32_packed_integer_also_reads_its_destination() {
    // An AArch32 vector write merges, so the prior value stays live.
    let i = insn(
        "vadd.i32",
        vec![
            op("d0", OperandKind::Register),
            op("d1", OperandKind::Register),
            op("d2", OperandKind::Register),
        ],
    );
    assert!(analyze(&i, Arch::Arm).uses.contains(&"v0"));
}

#[test]
fn aarch32_untyped_bitwise_neon_is_classified_as_simd() {
    let i = insn(
        "vand",
        vec![
            op("q0", OperandKind::Register),
            op("q1", OperandKind::Register),
            op("q2", OperandKind::Register),
        ],
    );
    assert_eq!(analyze(&i, Arch::Arm).kind, InstructionKind::Simd);
}

#[test]
fn aarch32_general_purpose_push_list_is_still_data_movement() {
    // A GPR register list is spelled like a NEON one; judging it by the
    // braces alone would truncate every AArch32 stack-frame setup.
    let i = insn("push", vec![op("{r4, r5, lr}", OperandKind::Unknown)]);
    assert_eq!(analyze(&i, Arch::Arm).kind, InstructionKind::Mov);
}

#[test]
fn aarch32_general_purpose_register_list_is_not_a_vector_shape() {
    // AArch32 spells push / pop / ldm operands with the same braces as
    // an AArch64 vector list, so an arrangement guard that keys on the
    // brace alone fails every stack-frame setup closed. The corpus
    // cannot catch this: it holds no 32-bit ARM sample at all.
    let i = insn("push", vec![op("{r4, r5, lr}", OperandKind::Unknown)]);
    let effect = analyze(&i, Arch::Arm);
    assert_eq!(effect.kind, InstructionKind::Mov);
}

#[test]
fn aarch64_contiguous_load_defines_its_listed_register() {
    let i = insn(
        "ld1",
        vec![
            op("{v0.16b}", OperandKind::Unknown),
            op("[x8]", OperandKind::Memory),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.defs, vec!["v0"]);
}

#[test]
fn aarch64_contiguous_store_reads_its_listed_registers() {
    let i = insn(
        "st1",
        vec![
            op("{v0.4s, v1.4s}", OperandKind::Unknown),
            op("[x8]", OperandKind::Memory),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).uses, vec!["x8", "v0", "v1"]);
}

#[test]
fn aarch64_structured_load_is_a_memory_access() {
    let i = insn(
        "ld1",
        vec![
            op("{v0.16b}", OperandKind::Unknown),
            op("[x8]", OperandKind::Memory),
        ],
    );
    assert!(analyze(&i, Arch::Aarch64).has_memory_access);
}

#[test]
fn aarch64_structured_post_index_defines_its_base_register() {
    let i = insn(
        "ld1",
        vec![
            op("{v0.16b}", OperandKind::Unknown),
            op("[x8]", OperandKind::Memory),
            op("16", OperandKind::Immediate),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).defs, vec!["v0", "x8"]);
}

#[test]
fn aarch64_structured_single_element_load_reads_its_destination() {
    // One lane is replaced and the rest of the register preserved, so
    // the prior definition is still live.
    let i = insn(
        "ld1",
        vec![
            op("{v0.s}[1]", OperandKind::Unknown),
            op("[x8]", OperandKind::Memory),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert!(effect.uses.contains(&"v0"), "{effect:?}");
}

#[test]
fn aarch64_deinterleaving_store_defines_no_register() {
    let i = insn(
        "st4",
        vec![
            op("{v16.4s, v17.4s, v18.4s, v19.4s}", OperandKind::Unknown),
            op("[x8]", OperandKind::Memory),
        ],
    );
    assert!(analyze(&i, Arch::Aarch64).defs.is_empty());
}

// --- N3a: NEON broadcast and permutation ---

#[test]
fn aarch64_element_insert_reads_its_destination() {
    // `ins` preserves every lane it does not write, so the register's
    // prior definition is still live and the slicer must keep it.
    let i = insn(
        "ins",
        vec![
            op("v0.s[1]", OperandKind::Register),
            op("w1", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert!(effect.uses.contains(&"v0"), "{effect:?}");
}

#[test]
fn aarch64_element_insert_still_defines_its_destination() {
    let i = insn(
        "ins",
        vec![
            op("v0.s[1]", OperandKind::Register),
            op("w1", OperandKind::Register),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).defs, vec!["v0"]);
}

#[test]
fn aarch64_permutation_does_not_read_its_destination() {
    // A permutation writes every lane, so the prior value is dead.
    let i = insn(
        "zip1",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert!(!effect.uses.contains(&"v0"), "{effect:?}");
}

#[test]
fn aarch64_element_to_general_register_defines_the_general_register() {
    let i = insn(
        "umov",
        vec![
            op("w0", OperandKind::Register),
            op("v1.s[1]", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.defs, vec!["x0"]);
    assert_eq!(effect.uses, vec!["v1"]);
}

#[test]
fn aarch64_broadcast_immediate_reads_no_register() {
    let i = insn(
        "movi",
        vec![
            op("v0.4s", OperandKind::Register),
            op("0x1", OperandKind::Immediate),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.defs, vec!["v0"]);
    assert!(effect.uses.is_empty(), "{effect:?}");
}

#[test]
fn aarch64_deinterleaving_load_defines_every_listed_register() {
    let i = insn(
        "ld2",
        vec![
            op("{v0.4s, v1.4s}", OperandKind::Unknown),
            op("[x2]", OperandKind::Memory),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).defs, vec!["v0", "v1"]);
}

#[test]
fn aarch64_deinterleaving_load_reads_only_its_base_register() {
    let i = insn(
        "ld2",
        vec![
            op("{v0.4s, v1.4s}", OperandKind::Unknown),
            op("[x2]", OperandKind::Memory),
        ],
    );
    assert_eq!(analyze(&i, Arch::Aarch64).uses, vec!["x2"]);
}

#[test]
fn aarch64_narrowing_two_form_reads_its_destination() {
    // `xtn2` writes only the destination's upper half, so the lower half
    // survives and its prior definition is still live.
    let i = insn(
        "xtn2",
        vec![
            op("v0.8h", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
        ],
    );
    assert!(analyze(&i, Arch::Aarch64).uses.contains(&"v0"));
}

#[test]
fn aarch64_narrowing_base_form_does_not_read_its_destination() {
    let i = insn(
        "xtn",
        vec![
            op("v0.4h", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
        ],
    );
    assert!(!analyze(&i, Arch::Aarch64).uses.contains(&"v0"));
}

#[test]
fn aarch64_widening_long_defines_its_destination() {
    let i = insn(
        "umull",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4h", OperandKind::Register),
            op("v2.4h", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert_eq!(effect.defs, vec!["v0"]);
    assert_eq!(effect.kind, InstructionKind::Simd);
}

#[test]
fn aarch64_multiply_accumulate_reads_its_destination() {
    // The accumulator's prior definition is live; without this the
    // slicer drops it and the accumulation reads a free input.
    let i = insn(
        "mla",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4s", OperandKind::Register),
            op("v2.4s", OperandKind::Register),
        ],
    );
    let effect = analyze(&i, Arch::Aarch64);
    assert!(effect.uses.contains(&"v0"), "{effect:?}");
}

#[test]
fn aarch64_long_multiply_accumulate_reads_its_destination() {
    let i = insn(
        "umlal",
        vec![
            op("v0.4s", OperandKind::Register),
            op("v1.4h", OperandKind::Register),
            op("v2.4h", OperandKind::Register),
        ],
    );
    assert!(analyze(&i, Arch::Aarch64).uses.contains(&"v0"));
}

#[test]
fn aarch64_bitwise_select_reads_its_destination() {
    // All three selects use the destination as an input — as the mask
    // for `bsl`, as the surviving value for `bit` / `bif`.
    for mnemonic in ["bsl", "bit", "bif"] {
        let i = insn(
            mnemonic,
            vec![
                op("v0.16b", OperandKind::Register),
                op("v1.16b", OperandKind::Register),
                op("v2.16b", OperandKind::Register),
            ],
        );
        let effect = analyze(&i, Arch::Aarch64);
        assert!(effect.uses.contains(&"v0"), "{mnemonic}: {effect:?}");
    }
}
