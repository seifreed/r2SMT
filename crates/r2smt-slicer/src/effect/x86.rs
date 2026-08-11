//! x86 per-instruction effect tables.

use super::{
    InstructionEffect, InstructionKind, any_memory_operand, canonical_register, other_effect,
    registers_in_operand,
};
use crate::lift::X86SimdShape;
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
        reads_flags: false,
    }
}

fn mov_effect(insn: &Instruction) -> InstructionEffect {
    // `mov dst, src` — defines dst, uses src registers, no flag effect.
    let mut defs = Vec::new();
    let mut uses = Vec::new();

    if let Some(dst) = insn.operands.first() {
        if let Some(reg) = canonical_register(&dst.raw, Arch::X86_64) {
            defs.push(reg);
        } else {
            // Memory destination (stack or otherwise): the registers in
            // the address expression are inputs (the store address),
            // not defs — the byte-granular memory model resolves the
            // stored value.
            uses.extend(registers_in_operand(dst, Arch::X86_64));
        }
    }
    if let Some(src) = insn.operands.get(1) {
        // `registers_in_operand` covers both a register source and the
        // address registers of a memory source (`[rbp - 4]` → rbp).
        for r in registers_in_operand(src, Arch::X86_64) {
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
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
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
            reads_flags: false,
        };
    }
    arith_effect(insn, InstructionKind::Xor)
}

fn cmp_or_test_effect(insn: &Instruction, kind: InstructionKind) -> InstructionEffect {
    // `cmp lhs, rhs` and `test lhs, rhs`: read both operands (including
    // any memory-address registers), define no register, write flags.
    // Memory operands are resolved by the byte-granular model.
    let mut uses = Vec::new();
    for op in &insn.operands {
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
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
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
        reads_flags: false,
    }
}

/// `set<cc> dst` — writes 0 or 1 from a flag it **reads**.
///
/// Both fields below were wrong in the direction that fabricates. With
/// `reads_flags: false` the backward walk never marks the flags live, so
/// the `cmp` that produced them is not retained; if some *other*
/// flag-setting instruction is retained for an unrelated register, SSA
/// binds this instruction's flag read to **that** one — a definite wrong
/// value deciding a branch, not a lost one. And scanning no operands at
/// all lost the address registers of a memory destination.
///
/// `AArch64` and `AArch32` have said `reads_flags: true` for exactly
/// this shape all along; x86 said it nowhere.
fn setcc_effect(insn: &Instruction) -> InstructionEffect {
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first() {
        if let Some(reg) = canonical_register(&dst.raw, Arch::X86_64) {
            defs.push(reg);
        } else {
            // A memory destination defines no register but reads the
            // ones that form its address.
            uses.extend(registers_in_operand(dst, Arch::X86_64));
        }
    }
    InstructionEffect {
        kind: InstructionKind::SetCc,
        defs,
        uses,
        defines_flags: false,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        reads_flags: true,
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
        // The condition is a flag read, and its ESIL lowering really
        // does emit one (`zf,!,?{,…,}`), so leaving this false let the
        // walk drop the flag's definer while the lifted IR still read
        // it — the same fabrication as `setcc` next door.
        reads_flags: true,
    }
}

/// `pxor`/`vpxor` of a register with itself is a zero idiom — the
/// result is 0 independent of the register's prior contents. The two
/// XOR'd operands are `[0],[1]` in the 2-operand form and `[1],[2]` in
/// the 3-operand VEX form.
fn xor_zero_idiom(insn: &Instruction) -> bool {
    let ops = &insn.operands;
    let same = |a: &r2smt_ir::program::Operand, b: &r2smt_ir::program::Operand| {
        let ca = canonical_register(&a.raw, Arch::X86_64);
        ca.is_some() && ca == canonical_register(&b.raw, Arch::X86_64)
    };
    match ops.len() {
        2 => same(&ops[0], &ops[1]),
        3 => same(&ops[1], &ops[2]),
        _ => false,
    }
}

fn simd_effect(insn: &Instruction, shape: X86SimdShape) -> InstructionEffect {
    // SIMD memory resolves through the byte-granular model, but only
    // where the lifter can build the address. An operand it would
    // decline must fall back to `Other` so the slicer truncates instead
    // of keeping an instruction whose store the lifter then drops —
    // that would let a later load read a stale prior store, which is
    // unsound. This condition and the lifter's own memory guard have to
    // stay the same condition; both are `x86_memory_modellable`.
    if insn
        .operands
        .iter()
        .any(|op| op.kind == OperandKind::Memory && !crate::lift::x86_memory_modellable(op))
    {
        return other_effect(insn);
    }
    let mnem = insn.mnemonic.trim().to_ascii_lowercase();
    let is_xor = mnem == "pxor" || mnem == "vpxor";
    let zero_idiom = is_xor && xor_zero_idiom(insn);

    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(dst) = insn.operands.first() {
        if let Some(reg) = canonical_register(&dst.raw, Arch::X86_64) {
            defs.push(reg);
            // A legacy read-modify-write reads its destination; the VEX
            // form names its two sources explicitly and does not, and
            // the zero idiom depends on nothing.
            //
            // Keyed on the encoding rather than on the operand count.
            // The two agree for every mnemonic modelled when this was
            // written, and diverge for the first family that is both
            // read-modify-write and 3-operand: `palignr xmm0, xmm1, 5`
            // reads `xmm0` as the high half of its source pair, so an
            // operand-count proxy would drop it from `uses` and the
            // backward walk would discard whatever produced it. The
            // 4-operand VEX form breaks the proxy the other way, adding
            // a use it does not have.
            if matches!(shape, X86SimdShape::ReadModifyWrite)
                && !crate::lift::is_vex(insn)
                && !zero_idiom
            {
                uses.push(reg);
            }
        } else {
            // Memory destination: address registers are inputs.
            uses.extend(registers_in_operand(dst, Arch::X86_64));
        }
    }
    if !zero_idiom {
        for src in insn.operands.iter().skip(1) {
            for r in registers_in_operand(src, Arch::X86_64) {
                if !uses.contains(&r) {
                    uses.push(r);
                }
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
            reads_flags: false,
        },
        "call" => InstructionEffect {
            kind: InstructionKind::Call,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: true,
            reads_flags: false,
        },
        "ret" | "retn" | "retf" => InstructionEffect {
            kind: InstructionKind::Ret,
            defs: Vec::new(),
            uses: Vec::new(),
            defines_flags: false,
            has_memory_access: false,
            is_call: false,
            reads_flags: false,
        },
        // Membership and operand shape both come from the lifter's
        // table, so the slicer cannot retain a mnemonic no handler
        // models. See `x86_simd_shape`.
        _ if let Some(shape) = crate::lift::x86_simd_shape(&mnemonic) => simd_effect(insn, shape),
        // The scalar FP compares share `cmp`'s shape, not the SIMD one:
        // they read both operands, define no register, and write flags.
        // `ptest` and the scalar FP compares share `cmp`'s shape: they
        // read both operands, define no register and write flags.
        // Routing them through `simd_effect` instead would claim operand
        // 0 as a definition and drop whatever produced that vector.
        m if crate::lift::is_x86_vector_flag_compare(m) => {
            cmp_or_test_effect(insn, InstructionKind::Cmp)
        }
        // The compare family writes a mask into its destination, so it
        // is a SIMD def like the arithmetic — not a flag-setting `cmp`.
        m if crate::lift::is_fp_compare_mnemonic(m) => {
            simd_effect(insn, X86SimdShape::ReadModifyWrite)
        }
        // The scalar moves merge one lane and leave the rest of the
        // destination standing, so the destination is a use as well as a
        // def — the `Bitwise` shape, not `Move`. The 3-operand VEX form
        // does overwrite the whole register, which `simd_effect` already
        // distinguishes. Recognised through the lifter's predicate
        // because `movsd` also names the string instruction.
        _ if crate::lift::sse_scalar_move_lane(insn).is_some() => {
            simd_effect(insn, X86SimdShape::ReadModifyWrite)
        }
        "sahf" => sahf_effect(),
        m if m.starts_with('j') => jcc_effect(insn),
        m if m.starts_with("set") => setcc_effect(insn),
        m if m.starts_with("cmov") => cmovcc_effect(insn),
        _ => match crate::lift::x87_effect(insn) {
            // Recognised through the lifter's own classifier so the two
            // cannot drift: an x87 shape the lifter would decline must
            // stay `Other` here, or the slicer keeps an instruction
            // whose stack effect the lifter then drops.
            Some(shape) => x87_effect(insn, &shape),
            None => other_effect(insn),
        },
    }
}

/// Almost every x87 instruction reads and writes the whole register
/// stack.
///
/// TOP rotates under almost every one of them, so an instruction that
/// merely reads `ST(0)` still renames every other slot; and the flat
/// register model has no way to say "slot 1, as numbered before the
/// previous push". Collapsing the file onto the single pseudo-register
/// `st` and marking it both a def and a use is the conservative
/// resolution: the slicer keeps the entire chain back to the values'
/// origin rather than stepping over the instruction that moved TOP
/// underneath it. Slot-level precision is recovered at lift time by the
/// slice-scoped stack in `lift/x87.rs`, which sees the chain in
/// execution order and so needs no numbering at all.
///
/// The exceptions come from `lift/x87.rs` rather than being decided
/// here, so the footprint and the lowering cannot drift: the `fcom`
/// family also defines the status word `fsw`, the `fcomi` family writes
/// EFLAGS in its place, and `fnstsw` reads `fsw` into a register
/// without touching the data registers at all.
fn x87_effect(insn: &Instruction, shape: &crate::lift::X87Effect) -> InstructionEffect {
    let mut defs = shape.defs.clone();
    if let Some(reg) = shape.register_def {
        defs.push(reg);
    }
    let mut uses = shape.uses.clone();
    for op in &insn.operands {
        for r in registers_in_operand(op, Arch::X86_64) {
            if !uses.contains(&r) {
                uses.push(r);
            }
        }
    }
    InstructionEffect {
        kind: InstructionKind::X87,
        defs,
        uses,
        defines_flags: shape.defines_flags,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        reads_flags: shape.reads_flags,
    }
}

/// `sahf` — load SF, ZF, AF, PF and CF from AH.
///
/// An integer instruction, listed here because it is the second half of
/// the x87 compare idiom: `fnstsw ax` puts the status word's condition
/// codes at exactly the AH positions this transfers. OF is
/// deliberately untouched
/// — `sahf` does not write it, so whatever defined it upstream still
/// holds.
fn sahf_effect() -> InstructionEffect {
    InstructionEffect {
        kind: InstructionKind::Sahf,
        defs: Vec::new(),
        uses: vec!["rax"],
        defines_flags: true,
        has_memory_access: false,
        is_call: false,
        reads_flags: false,
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
        reads_flags: false,
    }
}
