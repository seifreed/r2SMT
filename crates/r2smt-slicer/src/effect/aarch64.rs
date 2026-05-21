//! `AArch64` per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, other_effect,
    registers_in_operand,
};
use r2smt_common::Arch;
use r2smt_ir::program::Instruction;

pub(super) fn analyze_aarch64(insn: &Instruction) -> InstructionEffect {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    match mnemonic.as_str() {
        // 2-operand data movement: dst, src.
        "mov" | "movz" => aarch64_mov_effect(insn),
        // 3-operand arithmetic / logical: dst, src1, src2. The
        // flag-setting `s` suffix flips `defines_flags`.
        "add" => aarch64_arith3_effect(insn, InstructionKind::Add, false),
        "adds" => aarch64_arith3_effect(insn, InstructionKind::Add, true),
        "sub" => aarch64_arith3_effect(insn, InstructionKind::Sub, false),
        "subs" => aarch64_arith3_effect(insn, InstructionKind::Sub, true),
        "and" => aarch64_arith3_effect(insn, InstructionKind::And, false),
        "ands" => aarch64_arith3_effect(insn, InstructionKind::And, true),
        "orr" => aarch64_arith3_effect(insn, InstructionKind::Or, false),
        "eor" => aarch64_arith3_effect(insn, InstructionKind::Xor, false),
        // `mul`, `udiv`, `sdiv` share the 3-operand shape and never
        // set NZCV on AArch64 (no `s`-suffixed sibling). The lifter
        // tells them apart semantically via `Expr::udiv` / `sdiv`.
        "mul" | "udiv" | "sdiv" => aarch64_arith3_effect(insn, InstructionKind::Imul, false),
        // 3-operand shifts: dst, src, count.
        "lsl" => aarch64_arith3_effect(insn, InstructionKind::Shl, false),
        "lsr" => aarch64_arith3_effect(insn, InstructionKind::Shr, false),
        "asr" => aarch64_arith3_effect(insn, InstructionKind::Sar, false),
        // 2-operand compare / test: flags only, no destination.
        "cmp" => aarch64_cmp_test_effect(insn, InstructionKind::Cmp),
        "tst" => aarch64_cmp_test_effect(insn, InstructionKind::Test),
        // Control flow.
        "b" => InstructionEffect {
            kind: InstructionKind::Jmp,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        },
        "bl" | "blr" => InstructionEffect {
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
        "ret" => InstructionEffect {
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
        // Conditional branches read NZCV without writing any
        // register. Same shape as x86 jcc.
        m if m.starts_with("b.") => InstructionEffect {
            kind: InstructionKind::Jcc,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        },
        // Compare-and-branch (`cbz`/`cbnz`/`tbz`/`tbnz`) — does not
        // touch NZCV, but reads the operand register. The slicer
        // keeps a definition of that register alive.
        "cbz" | "cbnz" | "tbz" | "tbnz" => {
            let mut uses = Vec::new();
            if let Some(op) = insn.operands.first()
                && let Some(reg) = canonical_register(&op.raw, Arch::Aarch64)
            {
                uses.push(reg);
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
        // Conditional select family. The 4-operand `csel Rd, Rn, Rm,
        // cond` shape reads `Rn`, `Rm`, and NZCV; writes `Rd`. The
        // 2-operand `cset Rd, cond` shape only reads flags.
        // Conditional select family — full 4-operand forms plus the
        // 3-operand aliases (`cinc Rd, Rn, cond` etc.) that route
        // through the same effect because they read `Rd`'s parent,
        // `Rn`, and NZCV the same way.
        "csel" | "csinc" | "csinv" | "csneg" | "cinc" | "cinv" | "cneg" => {
            aarch64_csel_effect(insn)
        }
        "cset" | "csetm" => aarch64_cset_effect(insn),
        // P26 — memory loads / stores. `ldr` defines its destination
        // register; `str` does not define a register but mutates
        // memory state which downstream `ldr`s consume, so the
        // slicer's memory-aware pass keeps it (gated on
        // `allow_memory`). Flag-setting is `false` for both — these
        // are the plain `ldr` / `str` family, not the comparison
        // forms (`tst` / `cmp` handled above).
        "ldr" => aarch64_ldr_effect(insn),
        "str" => aarch64_str_effect(insn),
        _ => other_effect(insn),
    }
}

fn aarch64_ldr_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Aarch64)
    {
        defs.push(reg);
    }
    if let Some(mem) = insn.operands.get(1) {
        for r in registers_in_operand(mem, Arch::Aarch64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn aarch64_str_effect(insn: &Instruction) -> InstructionEffect {
    let mut uses = Vec::new();
    if let Some(src) = insn.operands.first()
        && let Some(reg) = canonical_register(&src.raw, Arch::Aarch64)
    {
        uses.push(reg);
    }
    if let Some(mem) = insn.operands.get(1) {
        for r in registers_in_operand(mem, Arch::Aarch64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        // `Mov` keeps the slicer out of the `Other`-truncation path
        // even though `str` has no register destination — it is
        // semantically "data movement", and the memory side-effect is
        // surfaced through `has_memory_access`. The memory-aware
        // slice walker keeps it for any kept downstream `ldr`.
        kind: InstructionKind::Mov,
        defs: Vec::new(),
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn aarch64_csel_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Aarch64)
    {
        defs.push(reg);
    }
    for src in insn.operands.iter().skip(1).take(2) {
        if let Some(reg) = canonical_register(&src.raw, Arch::Aarch64)
            && !uses.contains(&reg)
        {
            uses.push(reg);
        }
    }
    InstructionEffect {
        kind: InstructionKind::CMovCc,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: false,
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        // csel / csinc / csinv / csneg (and the cinc / cinv / cneg
        // aliases that route through here) all consume NZCV.
        reads_flags: true,
    }
}

fn aarch64_cset_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Aarch64)
    {
        defs.push(reg);
    }
    InstructionEffect {
        kind: InstructionKind::SetCc,
        defs,
        uses: Vec::new(),
        defines_flags: false,
        has_memory_access: false,
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        // `cset` / `csetm` only read NZCV — the predicate decides
        // whether to write 1 / -1 or 0.
        reads_flags: true,
    }
}

fn aarch64_mov_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Aarch64)
    {
        defs.push(reg);
    }
    if let Some(src) = insn.operands.get(1) {
        uses.extend(registers_in_operand(src, Arch::Aarch64));
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
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

fn aarch64_arith3_effect(
    insn: &Instruction,
    kind: InstructionKind,
    sets_flags: bool,
) -> InstructionEffect {
    // 3-operand: dst, src1, src2. Read-only sources, write-only dst —
    // unlike x86's RMW 2-operand form. Zero idiom `mov x, xzr` is
    // expressed by reading from `xzr`; the lifter handles that
    // semantically.
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Aarch64)
    {
        defs.push(reg);
    }
    for src in insn.operands.iter().skip(1) {
        for r in registers_in_operand(src, Arch::Aarch64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind,
        defs,
        uses,
        defines_flags: sets_flags,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn aarch64_cmp_test_effect(insn: &Instruction, kind: InstructionKind) -> InstructionEffect {
    // `cmp Rn, Operand` / `tst Rn, Operand` — read both, define no
    // register, write NZCV. Same shape as x86 cmp/test but with
    // AArch64 operand sets.
    let mut uses = Vec::new();
    for op in &insn.operands {
        for r in registers_in_operand(op, Arch::Aarch64) {
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
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}
