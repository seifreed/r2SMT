//! `AArch32` per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, other_effect,
    registers_in_operand,
};
use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, Operand};

pub(super) fn analyze_aarch32(insn: &Instruction) -> InstructionEffect {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    // Conditional-execution suffix: `<base><cond>` (e.g. `addeq`,
    // `moveq`) collapses for slicing purposes to the base mnemonic.
    // The actual predication lives in the lifter via Ite-wrapping
    // around every Assign — but the slice still needs to follow the
    // same register def / use chain as the unpredicated instruction
    // *and* keep the upstream flag-defining instruction alive so the
    // predicate is sound.
    let (dispatch_mnemonic, is_predicated) = if let Some((base, _)) =
        crate::lift::strip_aarch32_cond_suffix(&mnemonic)
        && crate::lift::is_aarch32_base_supported(base)
    {
        (base.to_string(), true)
    } else {
        (mnemonic.clone(), false)
    };
    let mut effect = analyze_aarch32_base(insn, &dispatch_mnemonic);
    if is_predicated {
        effect.reads_flags = true;
    }
    effect
}

fn analyze_aarch32_base(insn: &Instruction, dispatch_mnemonic: &str) -> InstructionEffect {
    // See the note in the `AArch64` table: an operand carrying vector
    // shape resolves as a `use` but not as a `def`, so it must fail
    // closed above the dispatch. `AArch32` NEON is `v`-prefixed with a
    // type suffix and so collides less, but the indexed form (`d0[1]`)
    // reaches the integer arms exactly the same way.
    if crate::lift::vector_shape(insn, Arch::Arm) == crate::lift::VectorShape::Declined {
        return other_effect(insn);
    }
    match dispatch_mnemonic {
        // 2-operand `mov Rd, Rn/imm` and `mvn Rd, Op` (bitwise NOT).
        "mov" | "mvn" => aarch32_mov_effect(insn),
        // 3-operand arithmetic / logical. The `s` suffix sets flags.
        "add" => aarch32_arith_effect(insn, InstructionKind::Add, false),
        "adds" => aarch32_arith_effect(insn, InstructionKind::Add, true),
        "sub" => aarch32_arith_effect(insn, InstructionKind::Sub, false),
        "subs" => aarch32_arith_effect(insn, InstructionKind::Sub, true),
        "rsb" | "rsbs" => {
            aarch32_arith_effect(insn, InstructionKind::Sub, dispatch_mnemonic.ends_with('s'))
        }
        // `and` / `ands` and the bit-clear variants (`bic`/`bics` ≡
        // `and(Rd, Rn, NOT(Operand))`) share the same data-flow
        // signature for slicing — register uses, defs, and the
        // flag-setting `s` suffix.
        "and" | "bic" => aarch32_arith_effect(insn, InstructionKind::And, false),
        "ands" | "bics" => aarch32_arith_effect(insn, InstructionKind::And, true),
        "orr" => aarch32_arith_effect(insn, InstructionKind::Or, false),
        "orrs" => aarch32_arith_effect(insn, InstructionKind::Or, true),
        "eor" => aarch32_arith_effect(insn, InstructionKind::Xor, false),
        "eors" => aarch32_arith_effect(insn, InstructionKind::Xor, true),
        // `mul`, `udiv`, `sdiv` share the 3-operand shape and never
        // set NZCV in their plain (no-`s`) form. `muls` toggles flags.
        "mul" | "udiv" | "sdiv" => aarch32_arith_effect(insn, InstructionKind::Imul, false),
        "muls" => aarch32_arith_effect(insn, InstructionKind::Imul, true),
        "lsl" | "lsls" => aarch32_arith_effect(
            insn,
            InstructionKind::Shl,
            dispatch_mnemonic.ends_with('s') && dispatch_mnemonic != "lsl",
        ),
        "lsr" | "lsrs" => aarch32_arith_effect(
            insn,
            InstructionKind::Shr,
            dispatch_mnemonic.ends_with('s') && dispatch_mnemonic != "lsr",
        ),
        "asr" | "asrs" => aarch32_arith_effect(
            insn,
            InstructionKind::Sar,
            dispatch_mnemonic.ends_with('s') && dispatch_mnemonic != "asr",
        ),
        // `cmp` / `cmn` set flags from a subtract / add and have
        // identical register-flow shape; `tst` / `teq` are the
        // logical counterparts.
        "cmp" | "cmn" => aarch32_cmp_test_effect(insn, InstructionKind::Cmp),
        "tst" | "teq" => aarch32_cmp_test_effect(insn, InstructionKind::Test),
        "b" => InstructionEffect {
            kind: InstructionKind::Jmp,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            reads_flags: false,
        },
        "bl" | "blx" => InstructionEffect {
            kind: InstructionKind::Call,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: true,
            reads_flags: false,
        },
        "bx" => InstructionEffect {
            // `bx lr` is the conventional AArch32 return.
            kind: InstructionKind::Ret,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            reads_flags: false,
        },
        m if m.starts_with('b') && m.len() == 3 => InstructionEffect {
            // `b<cond>` family — recognised by the classifier; here
            // we just tag it as a Jcc with no reg side effects.
            kind: InstructionKind::Jcc,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            reads_flags: false,
        },
        // Memory: `ldr` defines its destination register and reads the
        // base; `str` reads its source plus the base. Both flag
        // `has_memory_access` so the memory-aware slice walker keeps
        // them under `--allow-memory`.
        "ldr" | "ldrb" | "ldrh" => aarch32_ldr_effect(insn),
        "str" | "strb" | "strh" => aarch32_str_effect(insn),
        // Register-list multiple: `push`/`pop` use the implicit `sp`
        // base; `ldm`/`stm` an explicit base (index 0) plus the list
        // (index 1). Both touch memory so the memory-aware slice walker
        // keeps them.
        "push" => aarch32_push_pop_effect(insn, false),
        "pop" => aarch32_push_pop_effect(insn, true),
        "ldm" | "ldmia" => aarch32_ldm_stm_effect(insn, true),
        "stm" | "stmia" => aarch32_ldm_stm_effect(insn, false),
        // VFP scalar floating point and NEON packed data processing.
        // `vcmp` writes the flags and defines no register; everything
        // else defines its destination. Unlike AArch64, an AArch32
        // vector write preserves the rest of the register file (`d1`
        // survives a write to `d0`, both halves of `q0`), so the
        // destination is a use as well as a def.
        m if crate::lift::vfp_scalar(m).is_some()
            || crate::lift::is_aarch32_packed_instruction(insn) =>
        {
            aarch32_vfp_effect(insn)
        }
        _ => other_effect(insn),
    }
}

/// Canonical register names inside a `{r4, r5, lr}` list operand.
fn reglist_registers(op: &Operand) -> Vec<&'static str> {
    let raw = op.raw.trim();
    let body = raw
        .strip_prefix('{')
        .and_then(|b| b.strip_suffix('}'))
        .unwrap_or(raw);
    body.split(',')
        .filter_map(|s| canonical_register(s.trim(), Arch::Arm))
        .collect()
}

fn aarch32_push_pop_effect(insn: &Instruction, is_pop: bool) -> InstructionEffect {
    let regs = insn
        .operands
        .first()
        .map(reglist_registers)
        .unwrap_or_default();
    let mut defs = vec!["sp"];
    let mut uses = vec!["sp"];
    for r in regs {
        if is_pop {
            if !defs.contains(&r) {
                defs.push(r);
            }
        } else if !uses.contains(&r) {
            uses.push(r);
        }
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_ldm_stm_effect(insn: &Instruction, is_load: bool) -> InstructionEffect {
    let base = insn
        .operands
        .first()
        .and_then(|o| canonical_register(o.raw.trim().trim_end_matches('!').trim(), Arch::Arm));
    let regs = insn
        .operands
        .get(1)
        .map(reglist_registers)
        .unwrap_or_default();
    // The base is read for the address and may be written back; listing
    // it as both use and def is a sound superset for the slicer.
    let mut defs: Vec<&'static str> = base.into_iter().collect();
    let mut uses: Vec<&'static str> = base.into_iter().collect();
    for r in regs {
        if is_load {
            if !defs.contains(&r) {
                defs.push(r);
            }
        } else if !uses.contains(&r) {
            uses.push(r);
        }
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_ldr_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Arm)
    {
        defs.push(reg);
    }
    let mut uses = Vec::new();
    if let Some(mem) = insn.operands.get(1) {
        for r in registers_in_operand(mem, Arch::Arm) {
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
        reads_flags: false,
    }
}

fn aarch32_str_effect(insn: &Instruction) -> InstructionEffect {
    let mut uses = Vec::new();
    if let Some(src) = insn.operands.first()
        && let Some(reg) = canonical_register(&src.raw, Arch::Arm)
    {
        uses.push(reg);
    }
    if let Some(mem) = insn.operands.get(1) {
        for r in registers_in_operand(mem, Arch::Arm) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs: Vec::new(),
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_mov_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Arm)
    {
        defs.push(reg);
    }
    if let Some(src) = insn.operands.get(1) {
        uses.extend(registers_in_operand(src, Arch::Arm));
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_arith_effect(
    insn: &Instruction,
    kind: InstructionKind,
    sets_flags: bool,
) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Arm)
    {
        defs.push(reg);
    }
    for src in insn.operands.iter().skip(1) {
        for r in registers_in_operand(src, Arch::Arm) {
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
        reads_flags: false,
    }
}

fn aarch32_vfp_effect(insn: &Instruction) -> InstructionEffect {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    if matches!(
        crate::lift::vfp_scalar(&mnemonic),
        Some((crate::lift::VfpOp::Compare, _))
    ) {
        return aarch32_cmp_test_effect(insn, InstructionKind::Cmp);
    }
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Arm)
    {
        defs.push(reg);
        // The write only covers the addressed slice, so the rest of
        // the vector register survives and the prior value stays live.
        uses.push(reg);
    }
    for src in insn.operands.iter().skip(1) {
        for r in registers_in_operand(src, Arch::Arm) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind: InstructionKind::Simd,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_cmp_test_effect(insn: &Instruction, kind: InstructionKind) -> InstructionEffect {
    let mut uses = Vec::new();
    for op in &insn.operands {
        for r in registers_in_operand(op, Arch::Arm) {
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
        reads_flags: false,
    }
}
