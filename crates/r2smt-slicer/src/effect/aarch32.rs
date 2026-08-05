//! `AArch32` per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, other_effect,
    registers_in_operand,
};
use crate::lift::aarch32_memory_reads_carry;
use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, Operand};

pub(super) fn analyze_aarch32(insn: &Instruction) -> InstructionEffect {
    let mnemonic_full = insn.mnemonic.trim().to_ascii_lowercase();
    // Peel a Thumb-2 `.w` / `.n` encoding-width hint so `add.w` follows
    // the same def / use chain as `add`; mirrors the lifter dispatch.
    let mnemonic = crate::lift::strip_thumb_width_suffix(&mnemonic_full).to_string();
    // Conditional-execution suffix: `<base><cond>` (e.g. `addeq`,
    // `moveq`) collapses for slicing purposes to the base mnemonic.
    // The actual predication lives in the lifter via Ite-wrapping
    // around every Assign — but the slice still needs to follow the
    // same register def / use chain as the unpredicated instruction
    // *and* keep the upstream flag-defining instruction alive so the
    // predicate is sound.
    // Only peel when the full mnemonic is not itself a supported base:
    // flag-setting `s`-forms like `lsls`/`bics` end in a cond spelling
    // (`ls`/`cs`), so peeling first mis-dispatches them as a predicated
    // base. Mirrors the guard in `lift_instruction_aarch32`.
    let (dispatch_mnemonic, is_predicated) = if !crate::lift::is_aarch32_base_supported(&mnemonic)
        && crate::lift::vfp_scalar(&mnemonic).is_none()
        && let Some((base, _)) = crate::lift::strip_aarch32_cond_suffix(&mnemonic)
        && crate::lift::aarch32_predicable_base(&base)
    {
        (base, true)
    } else {
        (mnemonic.clone(), false)
    };
    let mut effect = analyze_aarch32_base(insn, &dispatch_mnemonic);
    if is_predicated {
        effect.reads_flags = true;
        // A predicated instruction lifts to `Ite(cond, new, Var(dst))`,
        // so on the false path the destination keeps the value it
        // already held — it is read as well as written.
        //
        // Omitting this **fabricates verdicts**; it does not merely lose
        // precision. The walk treats the predicated definition as
        // satisfying the destination's liveness, drops whatever defined
        // the previous value, and SSA then binds the else-arm to the
        // *same* free input that earlier instructions in the slice read
        // from that register. Widening to a free value is only sound
        // when the value is fresh; here it collapses onto an existing
        // one and asserts an equality the machine does not have. A real
        // ARM sample carried a `High`-confidence `dead_branch` from
        // exactly this — `Ite(cond, 1, r0)` where the architecture says
        // the else-arm is the 0 an intervening `mov r0, #0` had stored,
        // and the spurious `r0 == r0` is what let the solver prove the
        // branch never taken.
        //
        // The same def/use asymmetry the `AArch64` table documents.
        for def in effect.defs.clone() {
            if !effect.uses.contains(&def) {
                effect.uses.push(def);
            }
        }
    }
    effect
}

fn analyze_aarch32_base(insn: &Instruction, dispatch_mnemonic: &str) -> InstructionEffect {
    // See the note in the `AArch64` table: an operand carrying vector
    // shape resolves as a `use` but not as a `def`, so it must be
    // answered above the dispatch. `AArch32` NEON is `v`-prefixed with
    // a type suffix and so collides less, but the indexed form
    // (`d0[1]`) reaches the integer arms exactly the same way.
    // `vpush` / `vpop` carry a VFP register list that reads as a vector
    // arrangement, so they are answered before the vector-shape prelude
    // that would otherwise decline them to `Other`.
    match dispatch_mnemonic {
        "vpush" => return aarch32_vfp_push_pop_effect(insn, false),
        "vpop" => return aarch32_vfp_push_pop_effect(insn, true),
        _ => {}
    }
    if let Some(effect) = aarch32_vector_shape_effect(insn) {
        return effect;
    }
    aarch32_dispatch_mnemonic(insn, dispatch_mnemonic)
}

/// The vector-shape prelude shared by every `AArch32` mnemonic: a
/// structured access is the memory-shaped entry, a declined shape is
/// `Other`, and `None` / `Lifted` fall through to the mnemonic dispatch.
fn aarch32_vector_shape_effect(insn: &Instruction) -> Option<InstructionEffect> {
    match crate::lift::vector_shape(insn, Arch::Arm) {
        crate::lift::VectorShape::LiftedMemory(effect) => {
            Some(aarch32_structured_effect(insn, effect))
        }
        crate::lift::VectorShape::Declined => Some(other_effect(insn)),
        // `None` means no operand carries vector shape and the mnemonic
        // dispatch applies — which is also where a `Lifted` form lands,
        // at the guard arm that builds the vector entry. The two share
        // this arm because on `AArch32` the packed families are
        // recognised from the mnemonic either way.
        crate::lift::VectorShape::None | crate::lift::VectorShape::Lifted { .. } => None,
    }
}

/// The data-processing half of the `AArch32` dispatch: everything whose
/// effect is "define `Rd` from the source operands", with or without
/// the flag side effect. Split out from [`aarch32_dispatch_mnemonic`]
/// only to keep each under the line limit; both are pure dispatch.
fn aarch32_data_processing_effect(
    insn: &Instruction,
    dispatch_mnemonic: &str,
) -> Option<InstructionEffect> {
    Some(match dispatch_mnemonic {
        // 2-operand `mov Rd, Rn/imm` and `mvn Rd, Op` (bitwise NOT).
        // `movs` is the Thumb flag-setting move (N/Z from the value).
        // `movw` overwrites the whole register, like `mov`; `movt`
        // replaces only the top half, so it reads `Rd` too.
        // The 2-operand whole-register writes: `Rd` is defined from the
        // source and never read. Deliberately not `aarch32_arith_effect`,
        // whose 2-operand arm reads `Rd` back for the Thumb narrow form
        // (`add r0, r1` ≡ `add r0, r0, r1`) — `sxtb r0, r1` overwrites.
        "mov" | "mvn" | "movw" | "sxtb" | "sxth" | "uxtb" | "uxth" | "rev" => {
            aarch32_mov_effect(insn, false)
        }
        "movs" => aarch32_mov_effect(insn, true),
        "movt" => aarch32_mov_half_top_effect(insn),
        // `umull`/`smull` write *two* registers; reporting only the low
        // one would let the walk drop a later read of the high half.
        "umull" | "smull" => aarch32_long_multiply_effect(insn),
        "adc" | "sbc" | "adcs" | "sbcs" => aarch32_carry_arith_effect(insn, dispatch_mnemonic),
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
        // `mla Rd, Rn, Rm, Ra` carries four operands, so the narrow-form
        // arm of `aarch32_arith_effect` cannot misfire on it.
        "mul" | "udiv" | "sdiv" | "mla" | "mls" => {
            aarch32_arith_effect(insn, InstructionKind::Imul, false)
        }
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
        // `InstructionKind` spells no rotate; the kind is descriptive
        // here, and every consumer of it groups the right-moving shifts
        // together anyway.
        "ror" | "rors" => {
            aarch32_arith_effect(insn, InstructionKind::Shr, dispatch_mnemonic.ends_with('s'))
        }
        // `rrx Rd, Rm` **reads** C — it is a 33-bit rotate with the flag
        // as the extra bit. Without that, the backward walk drops
        // whatever defined C and SSA binds it to a stale free input,
        // which is a fabricated verdict rather than a lost one. The
        // 2-operand shape overwrites `Rd`, so it takes the mov-shaped
        // def/use rather than the narrow-form-aware arithmetic one.
        "rrx" | "rrxs" => {
            let mut effect = aarch32_mov_effect(insn, dispatch_mnemonic.ends_with('s'));
            effect.kind = InstructionKind::Shr;
            effect.reads_flags = true;
            effect
        }
        _ => return None,
    })
}

fn aarch32_dispatch_mnemonic(insn: &Instruction, dispatch_mnemonic: &str) -> InstructionEffect {
    if let Some(effect) = aarch32_data_processing_effect(insn, dispatch_mnemonic) {
        return effect;
    }
    match dispatch_mnemonic {
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
        // Thumb compare-and-branch: does not touch NZCV, but reads the
        // operand register, which the slicer keeps a definition of
        // alive. Thumb has no `tbz` / `tbnz`.
        "cbz" | "cbnz" => aarch32_compare_branch_effect(insn),
        m if m.starts_with('b')
            && crate::lift::strip_aarch32_cond_suffix(m).is_some_and(|(base, _)| base == "b") =>
        {
            // `b<cond>` family — `b` plus a recognised condition suffix
            // (`beq`, `bls`, `blt`, …). Recognised by the classifier;
            // here we just tag it as a Jcc with no reg side effects.
            // The precise condition check keeps the 3-char bitfield
            // mnemonics `bfi` / `bfc` (which define `Rd`) out — a length
            // gate would swallow them as flag-free jumps and let the
            // slicer step over a definition, binding a later read to a
            // stale value.
            InstructionEffect {
                kind: InstructionKind::Jcc,
                defs: Vec::new(),
                uses: Vec::new(),
                defines_flags: false,
                has_memory_access: false,
                is_call: false,
                reads_flags: false,
            }
        }
        // Memory: `ldr` defines its destination register and reads the
        // base; `str` reads its source plus the base. Both flag
        // `has_memory_access` so the memory-aware slice walker keeps
        // them under `--allow-memory`.
        "ldr" | "ldrb" | "ldrh" | "ldrsb" | "ldrsh" => aarch32_ldr_effect(insn),
        "str" | "strb" | "strh" => aarch32_str_effect(insn),
        // The doubleword forms name two registers before the memory
        // operand, so the single-register shape above cannot read them.
        "ldrd" => aarch32_pair_effect(insn, true),
        "strd" => aarch32_pair_effect(insn, false),
        // Register-list multiple: `push`/`pop` use the implicit `sp`
        // base; `ldm`/`stm` an explicit base (index 0) plus the list
        // (index 1). Both touch memory so the memory-aware slice walker
        // keeps them.
        "push" => aarch32_push_pop_effect(insn, false),
        "pop" => aarch32_push_pop_effect(insn, true),
        _ if let Some(is_load) = crate::lift::aarch32_multiple_is_load(dispatch_mnemonic) => {
            aarch32_ldm_stm_effect(insn, is_load)
        }
        // VFP scalar floating point and NEON packed data processing.
        // `vcmp` writes the flags and defines no register; everything
        // else defines its destination. Unlike AArch64, an AArch32
        // vector write preserves the rest of the register file (`d1`
        // survives a write to `d0`, both halves of `q0`), so the
        // destination is a use as well as a def.
        // `vmrs APSR_nzcv, FPSCR` transfers the FP compare flags into the
        // integer flags. `vcmp` wrote NZCV directly in this model, so the
        // transfer is an identity — but it must be kept (defines and
        // reads the flags) rather than falling to `other_effect`, or the
        // slice truncates here instead of walking to the `vcmp`.
        "vmrs" if crate::lift::aarch32_vmrs_transfers_flags(insn) => {
            aarch32_vmrs_flag_transfer_effect()
        }
        // Scalar VFP load / store. `vldr` defines its vector register
        // (and reads it — the write merges); `vstr` reads it. Both touch
        // memory so the memory-aware slice walker keeps them.
        "vldr" => aarch32_vfp_mem_effect(insn, true),
        "vstr" => aarch32_vfp_mem_effect(insn, false),
        m if crate::lift::vfp_scalar(m).is_some()
            || crate::lift::is_aarch32_packed_instruction(insn)
            || crate::lift::is_aarch32_neon_instruction(insn) =>
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
        // An `rrx` index shift rotates *through* the carry, so the
        // address depends on `CF`. Reporting `false` here would drop
        // whatever defined it from the slice and let SSA bind the
        // address's carry to a stale free input — the same fabrication
        // this file's predication wrapper documents at the top.
        reads_flags: aarch32_memory_reads_carry(insn),
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
        // An `rrx` index shift rotates *through* the carry, so the
        // address depends on `CF`. Reporting `false` here would drop
        // whatever defined it from the slice and let SSA bind the
        // address's carry to a stale free input — the same fabrication
        // this file's predication wrapper documents at the top.
        reads_flags: aarch32_memory_reads_carry(insn),
    }
}

/// The base register of an `AArch32` memory operand — the first
/// register named inside the brackets.
fn aarch32_memory_base(mem: &Operand) -> Option<&'static str> {
    let raw = mem.raw.trim();
    let body = raw.strip_suffix('!').unwrap_or(raw);
    let body = body.strip_prefix('[')?.strip_suffix(']')?;
    canonical_register(body.split(',').next()?.trim(), Arch::Arm)
}

/// The base register a memory access writes back to, when it has a
/// pre-index (`[Rn, #imm]!`) or post-index (`[Rn], #imm`) form.
///
/// The lifter emits `Rn := Rn + delta` for both, so the effect table
/// has to report `Rn` as a definition. Without it the walk steps past
/// this instruction while `Rn` is live, binds the read to an earlier
/// definition, and the slice carries a base value the machine never
/// held.
fn aarch32_writeback_base_at(insn: &Instruction, memory_index: usize) -> Option<&'static str> {
    let mem = insn.operands.get(memory_index)?;
    if !mem.raw.trim().ends_with('!') && insn.operands.len() <= memory_index + 1 {
        return None;
    }
    aarch32_memory_base(mem)
}

fn aarch32_writeback_base(insn: &Instruction) -> Option<&'static str> {
    aarch32_writeback_base_at(insn, 1)
}

/// `ldrd Rt, Rt2, [mem]` / `strd Rt, Rt2, [mem]`: both named registers
/// are defined (load) or read (store), and the memory operand sits at
/// index 2 rather than 1.
fn aarch32_pair_effect(insn: &Instruction, is_load: bool) -> InstructionEffect {
    const MEMORY_INDEX: usize = 2;
    let pair: Vec<&'static str> = insn
        .operands
        .iter()
        .take(MEMORY_INDEX)
        .filter_map(|op| canonical_register(&op.raw, Arch::Arm))
        .collect();
    let mut uses: Vec<&'static str> = Vec::new();
    if !is_load {
        uses.extend(pair.iter().copied());
    }
    for op in insn.operands.iter().skip(MEMORY_INDEX) {
        for r in registers_in_operand(op, Arch::Arm) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    let mut defs: Vec<&'static str> = if is_load { pair } else { Vec::new() };
    if let Some(base) = aarch32_writeback_base_at(insn, MEMORY_INDEX)
        && !defs.contains(&base)
    {
        defs.push(base);
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        // An `rrx` index shift rotates *through* the carry, so the
        // address depends on `CF`. Reporting `false` here would drop
        // whatever defined it from the slice and let SSA bind the
        // address's carry to a stale free input — the same fabrication
        // this file's predication wrapper documents at the top.
        reads_flags: aarch32_memory_reads_carry(insn),
    }
}

/// Registers named by the operands *after* the bracketed one — the
/// post-index delta lives outside the brackets (`ldr r0, [r1], r2`), so
/// `registers_in_operand` over the memory operand alone misses it.
fn aarch32_post_index_uses(insn: &Instruction) -> Vec<&'static str> {
    insn.operands
        .iter()
        .skip(2)
        .flat_map(|op| registers_in_operand(op, Arch::Arm))
        .collect()
}

fn aarch32_ldr_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    if let Some(dst) = insn.operands.first()
        && let Some(reg) = canonical_register(&dst.raw, Arch::Arm)
    {
        defs.push(reg);
    }
    if let Some(base) = aarch32_writeback_base(insn)
        && !defs.contains(&base)
    {
        defs.push(base);
    }
    let mut uses = Vec::new();
    if let Some(mem) = insn.operands.get(1) {
        for r in registers_in_operand(mem, Arch::Arm) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    for r in aarch32_post_index_uses(insn) {
        if !uses.contains(&r) {
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
        // An `rrx` index shift rotates *through* the carry, so the
        // address depends on `CF`. Reporting `false` here would drop
        // whatever defined it from the slice and let SSA bind the
        // address's carry to a stale free input — the same fabrication
        // this file's predication wrapper documents at the top.
        reads_flags: aarch32_memory_reads_carry(insn),
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
    for r in aarch32_post_index_uses(insn) {
        if !uses.contains(&r) {
            uses.push(r);
        }
    }
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs: aarch32_writeback_base(insn).into_iter().collect(),
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_vfp_push_pop_effect(insn: &Instruction, is_pop: bool) -> InstructionEffect {
    let parents = insn
        .operands
        .first()
        .and_then(|o| crate::lift::parse_vfp_reglist(&o.raw))
        .map(|(regs, _)| {
            let mut out: Vec<&'static str> = Vec::new();
            for reg in &regs {
                if let Some(canon) = canonical_register(&reg.raw, Arch::Arm)
                    && !out.contains(&canon)
                {
                    out.push(canon);
                }
            }
            out
        })
        .unwrap_or_default();
    let mut defs = vec!["sp"];
    let mut uses = vec!["sp"];
    for r in parents {
        if is_pop {
            if !defs.contains(&r) {
                defs.push(r);
            }
            // A VFP write merges, so the destination is also a use.
            if !uses.contains(&r) {
                uses.push(r);
            }
        } else if !uses.contains(&r) {
            uses.push(r);
        }
    }
    InstructionEffect {
        kind: InstructionKind::Simd,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_vfp_mem_effect(insn: &Instruction, is_load: bool) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(reg) = insn.operands.first()
        && let Some(canonical) = canonical_register(&reg.raw, Arch::Arm)
    {
        if is_load {
            defs.push(canonical);
            // The load covers only the addressed slice, so the rest of
            // the vector register survives and the prior value stays live.
            uses.push(canonical);
        } else {
            uses.push(canonical);
        }
    }
    if let Some(mem) = insn.operands.get(1) {
        for r in registers_in_operand(mem, Arch::Arm) {
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
        has_memory_access: true,
        is_call: false,
        reads_flags: false,
    }
}

fn aarch32_vmrs_flag_transfer_effect() -> InstructionEffect {
    InstructionEffect {
        kind: InstructionKind::Cmp,
        defs: Vec::new(),
        uses: Vec::new(),
        defines_flags: true,
        has_memory_access: false,
        is_call: false,
        reads_flags: true,
    }
}

fn aarch32_compare_branch_effect(insn: &Instruction) -> InstructionEffect {
    let mut uses = Vec::new();
    if let Some(op) = insn.operands.first()
        && let Some(reg) = canonical_register(&op.raw, Arch::Arm)
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

fn aarch32_mov_effect(insn: &Instruction, sets_flags: bool) -> InstructionEffect {
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
        defines_flags: sets_flags,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        reads_flags: false,
    }
}

/// `umull RdLo, RdHi, Rn, Rm` defines both destination registers and
/// reads only the two factors.
fn aarch32_long_multiply_effect(insn: &Instruction) -> InstructionEffect {
    const FACTOR_INDEX: usize = 2;
    let defs: Vec<&'static str> = insn
        .operands
        .iter()
        .take(FACTOR_INDEX)
        .filter_map(|op| canonical_register(&op.raw, Arch::Arm))
        .collect();
    let mut uses: Vec<&'static str> = Vec::new();
    for op in insn.operands.iter().skip(FACTOR_INDEX) {
        for r in registers_in_operand(op, Arch::Arm) {
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
        has_memory_access: false,
        is_call: false,
        reads_flags: false,
    }
}

/// `adc` / `sbc` read C on top of their register operands.
fn aarch32_carry_arith_effect(insn: &Instruction, mnemonic: &str) -> InstructionEffect {
    let kind = if mnemonic.starts_with("adc") {
        InstructionKind::Add
    } else {
        InstructionKind::Sub
    };
    let mut effect = aarch32_arith_effect(insn, kind, mnemonic.ends_with('s'));
    effect.reads_flags = true;
    effect
}

/// `movt Rd, #imm16` is a partial write: the bottom half of `Rd`
/// survives, so the destination is a use as well as a def. Reporting
/// only the def would let the walk drop whatever set the low half.
fn aarch32_mov_half_top_effect(insn: &Instruction) -> InstructionEffect {
    let dst = insn
        .operands
        .first()
        .and_then(|op| canonical_register(&op.raw, Arch::Arm));
    InstructionEffect {
        kind: InstructionKind::Mov,
        defs: dst.into_iter().collect(),
        uses: dst.into_iter().collect(),
        defines_flags: false,
        has_memory_access: false,
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
        // Thumb 2-operand narrow form: `add r0, r1` ≡ `add r0, r0, r1`,
        // so the destination is also a source. In the 3-operand ISA
        // form these mnemonics always carry three operands, so a
        // 2-operand shape is unambiguously the short form.
        if insn.operands.len() == 2 {
            uses.push(reg);
        }
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
    // `vzip` / `vuzp` / `vtrn` rewrite both named registers, so the
    // second operand is a definition too. Recorded here rather than in
    // an arm of its own because it is the only way these differ from
    // every other vector entry: the second operand is already a use,
    // since an `AArch32` vector write merges.
    if crate::lift::aarch32_neon_writes_operand_pair(insn)
        && let Some(second) = insn.operands.get(1)
        && let Some(reg) = canonical_register(&second.raw, Arch::Arm)
        && !defs.contains(&reg)
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

/// The `AArch32` structured load / store family (`vld1`–`vld4`,
/// `vst1`–`vst4`).
///
/// The listed registers come from the operand text, exactly as the
/// `ldm` / `push` entry above takes them: [`registers_in_operand`]
/// already recovers every parent a brace list names. What the resolver
/// supplies is the side each of them sits on.
///
/// Both `d0` and `d1` resolve to the parent `v0`, so a two-register
/// list collapses to one entry here — which is right, since they are
/// two halves of one register.
fn aarch32_structured_effect(
    insn: &Instruction,
    effect: crate::lift::StructuredEffect,
) -> InstructionEffect {
    let listed = insn
        .operands
        .first()
        .map(|op| registers_in_operand(op, Arch::Arm))
        .unwrap_or_default();
    let base = insn
        .operands
        .get(1)
        .map(|op| registers_in_operand(op, Arch::Arm))
        .unwrap_or_default();
    let mut defs: Vec<&'static str> = Vec::new();
    let mut uses: Vec<&'static str> = base.clone();
    // A register post-index (`[r0], r1`) reads `r1` to advance the base,
    // so any operand past the base is a use.
    for op in insn.operands.iter().skip(2) {
        for register in registers_in_operand(op, Arch::Arm) {
            if !uses.contains(&register) {
                uses.push(register);
            }
        }
    }
    if effect.reads_list {
        for register in &listed {
            if !uses.contains(register) {
                uses.push(register);
            }
        }
    }
    if effect.writes_list {
        for register in &listed {
            if !defs.contains(register) {
                defs.push(register);
            }
        }
    }
    if effect.writes_base {
        for register in &base {
            if !defs.contains(register) {
                defs.push(register);
            }
        }
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
