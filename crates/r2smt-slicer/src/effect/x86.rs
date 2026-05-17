//! x86 per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, first_register,
    has_unresolved_memory, other_effect, registers_in_operand, stack_slot,
};
use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, OperandKind};

fn arith_effect(insn: &Instruction, kind: InstructionKind) -> InstructionEffect {
    // Two-operand RMW arithmetic / logical: `op dst, src` —
    // defines flags, defines dst (if register), uses dst + src.
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first() {
        if let Some(reg) = canonical_register(&dst.raw, Arch::X86_64) {
            defs.push(reg);
            uses.push(reg);
        } else {
            uses.extend(registers_in_operand(dst, Arch::X86_64));
        }
    }
    if let Some(src) = insn.operands.get(1) {
        for r in registers_in_operand(src, Arch::X86_64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind,
        defs,
        uses,
        defines_flags: true,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn mov_effect(insn: &Instruction) -> InstructionEffect {
    // `mov dst, src` — defines dst, uses src registers, no flag effect.
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    let mut stack_defs: Vec<String> = Vec::new();
    let mut stack_uses: Vec<String> = Vec::new();

    if let Some(dst) = insn.operands.first() {
        if let Some(reg) = canonical_register(&dst.raw, Arch::X86_64) {
            defs.push(reg);
        } else if let Some((slot, _bits)) = stack_slot(dst) {
            // `mov [rbp - K], src` — the stack slot becomes a def, like
            // a virtual register. The base register (rbp/rsp) is not a
            // data input.
            stack_defs.push(slot);
        } else {
            // Memory destination we cannot resolve: the registers in
            // the expression are address inputs, not data inputs.
            uses.extend(registers_in_operand(dst, Arch::X86_64));
        }
    }
    if let Some(src) = insn.operands.get(1) {
        if src.kind == OperandKind::Memory {
            if let Some((slot, _bits)) = stack_slot(src) {
                stack_uses.push(slot);
            } else {
                for r in registers_in_operand(src, Arch::X86_64) {
                    if !uses.contains(&r) {
                        uses.push(r);
                    }
                }
            }
        } else {
            for r in registers_in_operand(src, Arch::X86_64) {
                if !uses.contains(&r) {
                    uses.push(r);
                }
            }
        }
    }

    InstructionEffect {
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: has_unresolved_memory(&insn.operands),
        is_call: false,
        stack_defs,
        stack_uses,
        reads_flags: false,
    }
}

fn lea_effect(insn: &Instruction) -> InstructionEffect {
    // `lea dst, [expr]` — defines dst with the *address* of `expr`,
    // never accesses memory. Uses the registers inside `expr`.
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::X86_64)
    {
        defs.push(reg);
    }
    if let Some(src) = insn.operands.get(1) {
        uses.extend(registers_in_operand(src, Arch::X86_64));
    }
    InstructionEffect {
        kind: InstructionKind::Lea,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: false,
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn xor_effect(insn: &Instruction) -> InstructionEffect {
    // The zero idiom requires both operands to be the *same* register
    // textually (e.g. `xor eax, eax`). Comparing canonical names would
    // incorrectly fold sub-register pairs like `xor ah, al` (which
    // XORs the second-lowest byte with the lowest byte of rax) onto
    // a zero idiom — leading to false constant-condition findings.
    let lhs_raw = insn
        .operands
        .first()
        .map(|o| o.raw.trim().to_ascii_lowercase());
    let rhs_raw = insn
        .operands
        .get(1)
        .map(|o| o.raw.trim().to_ascii_lowercase());
    if let (Some(l), Some(r)) = (lhs_raw, rhs_raw)
        && l == r
        && let Some(canon) = canonical_register(&l, Arch::X86_64)
    {
        // True zero idiom: `xor reg, reg` sets the register to 0 and
        // clears ZF/CF/SF/OF/PF without depending on the previous value.
        return InstructionEffect {
            kind: InstructionKind::Xor,
            defs: vec![canon],
            uses: vec![],
            defines_flags: true,
            has_memory_access: false,
            is_call: false,
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        };
    }
    arith_effect(insn, InstructionKind::Xor)
}

fn cmp_or_test_effect(insn: &Instruction, kind: InstructionKind) -> InstructionEffect {
    // `cmp lhs, rhs` and `test lhs, rhs`: read both, define no register,
    // write flags.
    let mut uses = Vec::new();
    let mut stack_uses: Vec<String> = Vec::new();
    for op in &insn.operands {
        if op.kind == OperandKind::Memory {
            if let Some((slot, _bits)) = stack_slot(op) {
                if !stack_uses.contains(&slot) {
                    stack_uses.push(slot);
                }
                continue;
            }
        }
        for r in registers_in_operand(op, Arch::X86_64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind,
        defs: Vec::new(),
        uses,
        defines_flags: true,
        has_memory_access: has_unresolved_memory(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses,
        reads_flags: false,
    }
}

fn shift_effect(insn: &Instruction, kind: InstructionKind) -> InstructionEffect {
    arith_effect(insn, kind)
}

fn jcc_effect(insn: &Instruction) -> InstructionEffect {
    // jcc reads flags, defines nothing, and the only register it might
    // reference is in an indirect target operand (rare for jcc).
    let mut uses = Vec::new();
    for op in &insn.operands {
        for r in registers_in_operand(op, Arch::X86_64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind: InstructionKind::Jcc,
        defs: Vec::new(),
        uses,
        defines_flags: false,
        has_memory_access: false,
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn setcc_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    if let Some(reg) = first_register(&insn.operands) {
        defs.push(reg);
    }
    InstructionEffect {
        kind: InstructionKind::SetCc,
        defs,
        uses: Vec::new(),
        defines_flags: false,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn cmovcc_effect(insn: &Instruction) -> InstructionEffect {
    // `cmovcc dst, src` — *conditionally* writes dst; treat as both a
    // def and a use of dst so a non-taken cmov keeps the prior value
    // alive in the slice.
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::X86_64)
    {
        defs.push(reg);
        uses.push(reg);
    }
    if let Some(src) = insn.operands.get(1) {
        for r in registers_in_operand(src, Arch::X86_64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind: InstructionKind::CMovCc,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

pub(super) fn analyze_x86(insn: &Instruction) -> InstructionEffect {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    match mnemonic.as_str() {
        "mov" | "movzx" | "movsx" | "movsxd" => mov_effect(insn),
        "lea" => lea_effect(insn),
        "xor" => xor_effect(insn),
        "and" => arith_effect(insn, InstructionKind::And),
        "or" => arith_effect(insn, InstructionKind::Or),
        "add" => arith_effect(insn, InstructionKind::Add),
        "sub" => arith_effect(insn, InstructionKind::Sub),
        "imul" => imul_effect(insn),
        "cmp" => cmp_or_test_effect(insn, InstructionKind::Cmp),
        "test" => cmp_or_test_effect(insn, InstructionKind::Test),
        "shl" | "sal" => shift_effect(insn, InstructionKind::Shl),
        "shr" => shift_effect(insn, InstructionKind::Shr),
        "sar" => shift_effect(insn, InstructionKind::Sar),
        "jmp" => InstructionEffect {
            kind: InstructionKind::Jmp,
            defs: Vec::new(),
            uses: insn
                .operands
                .iter()
                .flat_map(|op| registers_in_operand(op, Arch::X86_64))
                .collect(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        },
        "call" => InstructionEffect {
            kind: InstructionKind::Call,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: true,
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        },
        "ret" | "retn" | "retf" => InstructionEffect {
            kind: InstructionKind::Ret,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        },
        m if m.starts_with('j') => jcc_effect(insn),
        m if m.starts_with("set") => setcc_effect(insn),
        m if m.starts_with("cmov") => cmovcc_effect(insn),
        _ => other_effect(insn),
    }
}

fn imul_effect(insn: &Instruction) -> InstructionEffect {
    // `imul` has three forms:
    //   1-operand:  imul src           — defs rax:rdx, uses rax + src
    //   2-operand:  imul dst, src      — defs dst, uses dst + src
    //   3-operand:  imul dst, src, imm — defs dst, uses src
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    match insn.operands.len() {
        1 => {
            defs.push("rax");
            defs.push("rdx");
            uses.push("rax");
            uses.extend(registers_in_operand(&insn.operands[0], Arch::X86_64));
        }
        2 => {
            return arith_effect(insn, InstructionKind::Imul);
        }
        3 => {
            if let Some(reg) = canonical_register(&insn.operands[0].raw, Arch::X86_64) {
                defs.push(reg);
            }
            uses.extend(registers_in_operand(&insn.operands[1], Arch::X86_64));
        }
        _ => {}
    }
    InstructionEffect {
        kind: InstructionKind::Imul,
        defs,
        uses,
        defines_flags: true,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}
