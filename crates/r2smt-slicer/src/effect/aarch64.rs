//! `AArch64` per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, other_effect,
    registers_in_operand,
};
use r2smt_common::Arch;
use r2smt_ir::program::Instruction;

pub(super) fn analyze_aarch64(insn: &Instruction) -> InstructionEffect {
    if let Some(effect) = aarch64_vector_effect(insn) {
        return effect;
    }
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    match mnemonic.as_str() {
        // 2-operand data movement: dst, src.
        "mov" | "movz" => aarch64_mov_effect(insn),
        // 3-operand arithmetic / logical: dst, src1, src2. The
        // flag-setting `s` suffix flips `defines_flags`.
        // The complemented-Operand2 logicals (`bic` / `bics` / `orn` /
        // `eon`) and the two-operand spellings of a three-operand
        // instruction (`mvn` = `orn Rd, xzr, Op`, `neg` = `sub Rd, xzr,
        // Op`) share their siblings' arms: what they complement is a
        // value the lifter computes, and none of it changes which
        // registers are read or written. `aarch64_arith3_effect` covers
        // both arities — it reads every operand past the first — so a
        // trailing shift specifier lands in `uses` either way.
        "add" => aarch64_arith3_effect(insn, InstructionKind::Add, false),
        "adds" => aarch64_arith3_effect(insn, InstructionKind::Add, true),
        "sub" | "neg" => aarch64_arith3_effect(insn, InstructionKind::Sub, false),
        "subs" | "negs" => aarch64_arith3_effect(insn, InstructionKind::Sub, true),
        "and" | "bic" => aarch64_arith3_effect(insn, InstructionKind::And, false),
        "ands" | "bics" => aarch64_arith3_effect(insn, InstructionKind::And, true),
        "orr" | "orn" | "mvn" => aarch64_arith3_effect(insn, InstructionKind::Or, false),
        "eor" | "eon" => aarch64_arith3_effect(insn, InstructionKind::Xor, false),
        // `mul`, `udiv`, `sdiv` share the 3-operand shape and never
        // set NZCV on AArch64 (no `s`-suffixed sibling). The lifter
        // tells them apart semantically via `Expr::udiv` / `sdiv`.
        "mul" | "udiv" | "sdiv" => aarch64_arith3_effect(insn, InstructionKind::Imul, false),
        // 3-operand shifts: dst, src, count.
        "lsl" => aarch64_arith3_effect(insn, InstructionKind::Shl, false),
        "lsr" => aarch64_arith3_effect(insn, InstructionKind::Shr, false),
        "asr" => aarch64_arith3_effect(insn, InstructionKind::Sar, false),
        // 2-operand compare / test: flags only, no destination.
        // Scalar floating point. The destination is written whole (a
        // scalar write zeroes the rest of the vector register), so it
        // is a def and not also a use — the 3-operand shape, not x86's
        // read-modify-write one. `canonical_register` already resolves
        // `s0` / `d0` / `h0` to their `v<n>` parent, so the register
        // family is shared with the rest of the vector file and the
        // slicer sees the aliasing.
        "fadd" | "fsub" | "fmul" | "fmulx" | "fdiv" | "fmax" | "fmin" | "fsqrt" | "fabs"
        | "fneg" | "fmov" | "fcvt" | "scvtf" | "ucvtf" | "fcvtzs" | "fcvtzu" | "frinta"
        | "frintn" | "frintp" | "frintm" | "frintz" | "frinti" | "frintx" | "frint32z"
        | "frint32x" | "frint64z" | "frint64x" | "frecpx" => {
            aarch64_arith3_effect(insn, InstructionKind::Simd, false)
        }
        // `fcmp` writes NZCV and defines no register — `cmp`'s shape.
        "cmp" | "cmn" | "fcmp" | "fcmpe" => aarch64_cmp_test_effect(insn, InstructionKind::Cmp),
        "tst" => aarch64_cmp_test_effect(insn, InstructionKind::Test),
        // Control flow.
        "b" => InstructionEffect {
            kind: InstructionKind::Jmp,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            reads_flags: false,
        },
        "bl" | "blr" => InstructionEffect {
            kind: InstructionKind::Call,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: true,
            reads_flags: false,
        },
        "ret" => InstructionEffect {
            kind: InstructionKind::Ret,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
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
        "ldr" | "ldrb" | "ldrh" | "ldrsb" | "ldrsh" | "ldrsw" => aarch64_ldr_effect(insn),
        "str" | "strb" | "strh" => aarch64_str_effect(insn),
        // Paired forms: `ldp` defines both `Rt`/`Rt2`, `stp` uses them;
        // both read the base register in the memory operand (index 2).
        "ldp" => aarch64_ldp_effect(insn),
        "stp" => aarch64_stp_effect(insn),
        _ => other_effect(insn),
    }
}

/// The vector-shape guard, which has to sit above the mnemonic
/// dispatch rather than in arms of its own.
///
/// NEON reuses the integer and scalar-FP mnemonics (`add v0.4s`,
/// `fadd v0.2d`), which are already claimed below. Without the guard
/// the destination's arrangement makes `canonical_register` return
/// `None` and the instruction passes with empty `defs` while its
/// sources still resolve as `uses`: the slicer neither truncates nor
/// keeps it, and a later read of the destination binds to a stale
/// definition.
///
/// `None` means no operand carries vector shape and the scalar
/// dispatch applies.
fn aarch64_vector_effect(insn: &Instruction) -> Option<InstructionEffect> {
    match crate::lift::vector_shape(insn, Arch::Aarch64) {
        crate::lift::VectorShape::None => None,
        // A packed write covers the destination's whole arrangement and
        // zeroes the vector register above it, so the destination is a
        // pure def — the 3-operand shape, not x86's read-modify-write
        // one. Handled here rather than by falling through to the
        // integer arms, which is what keeps the packed spelling of a
        // mnemonic off the scalar arm regardless of what that arm does.
        // The reason recorded here used to be that `mvn` and `bic` have
        // no scalar `AArch64` form; they do — `bic x0, x1, x2` and
        // `mvn x0, x1` are both real A64 — and both are now dispatched
        // below.
        //
        // An element insert is the exception: it preserves every lane it
        // does not write, so its destination is a use as well, and the
        // slicer has to keep whatever defined the register before it.
        crate::lift::VectorShape::Lifted { reads_destination } => {
            let mut effect = aarch64_arith3_effect(insn, InstructionKind::Simd, false);
            if reads_destination {
                for def in effect.defs.clone() {
                    if !effect.uses.contains(&def) {
                        effect.uses.push(def);
                    }
                }
            }
            Some(effect)
        }
        // A structured load / store moves bytes rather than computing
        // them, so its entry is the memory-shaped one: the register
        // list on whichever side the transfer's direction puts it, the
        // addressing registers as uses, and the base as a def when the
        // access advances it.
        crate::lift::VectorShape::LiftedMemory(effect) => {
            Some(aarch64_structured_effect(insn, effect))
        }
        crate::lift::VectorShape::Declined => Some(other_effect(insn)),
    }
}

/// The structured load / store family (`ld1`…`ld4`, `st1`…`st4`,
/// `ld1r`…`ld4r`).
///
/// The listed registers come from the operand text rather than from the
/// resolver: [`registers_in_operand`] already recovers every parent a
/// brace list names, which is the same route `AArch32`'s `ldm` / `stm`
/// entry takes. What the resolver supplies is the side each of them
/// sits on — a single-element load reads its destinations as well as
/// writing them, because it replaces one lane and preserves the rest.
fn aarch64_structured_effect(
    insn: &Instruction,
    effect: crate::lift::StructuredEffect,
) -> InstructionEffect {
    let listed = insn
        .operands
        .first()
        .map(|op| registers_in_operand(op, Arch::Aarch64))
        .unwrap_or_default();
    let addressing: Vec<&'static str> = insn
        .operands
        .iter()
        .skip(1)
        .flat_map(|op| registers_in_operand(op, Arch::Aarch64))
        .collect();
    let base = insn
        .operands
        .get(1)
        .map(|op| registers_in_operand(op, Arch::Aarch64))
        .unwrap_or_default();
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    extend_unique(&mut uses, &addressing);
    if effect.reads_list {
        extend_unique(&mut uses, &listed);
    }
    if effect.writes_list {
        extend_unique(&mut defs, &listed);
    }
    if effect.writes_base {
        extend_unique(&mut defs, &base);
    }
    InstructionEffect {
        // `Mov` keeps the slicer out of the `Other`-truncation path;
        // the memory side-effect is surfaced through
        // `has_memory_access`, exactly as `ldr` / `str` do.
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

/// Append the registers of `from` that `into` does not already list.
fn extend_unique(into: &mut Vec<&'static str>, from: &[&'static str]) {
    for register in from {
        if !into.contains(register) {
            into.push(register);
        }
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
        reads_flags: false,
    }
}

fn aarch64_ldp_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    for op in insn.operands.iter().take(2) {
        if let Some(reg) = canonical_register(&op.raw, Arch::Aarch64)
            && !defs.contains(&reg)
        {
            defs.push(reg);
        }
    }
    let mut uses = Vec::new();
    if let Some(mem) = insn.operands.get(2) {
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
        reads_flags: false,
    }
}

fn aarch64_stp_effect(insn: &Instruction) -> InstructionEffect {
    let mut uses = Vec::new();
    for op in insn.operands.iter().take(2) {
        if let Some(reg) = canonical_register(&op.raw, Arch::Aarch64)
            && !uses.contains(&reg)
        {
            uses.push(reg);
        }
    }
    if let Some(mem) = insn.operands.get(2) {
        for r in registers_in_operand(mem, Arch::Aarch64) {
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
        reads_flags: false,
    }
}
