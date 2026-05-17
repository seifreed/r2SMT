//! `AArch32` per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, other_effect,
    registers_in_operand,
};
use r2smt_common::Arch;
use r2smt_ir::program::Instruction;

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
            stack_defs: Vec::new(),
            stack_uses: Vec::new(),
            reads_flags: false,
        },
        "bl" | "blx" => InstructionEffect {
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
        "bx" => InstructionEffect {
            // `bx lr` is the conventional AArch32 return.
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
        m if m.starts_with('b') && m.len() == 3 => InstructionEffect {
            // `b<cond>` family — recognised by the classifier; here
            // we just tag it as a Jcc with no reg side effects.
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
        _ => other_effect(insn),
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
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
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
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
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
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}
