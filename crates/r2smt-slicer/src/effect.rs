//! Per-instruction effect analysis used by the backward slicer.
//!
//! For each supported x86 / `x86_64`, `AArch64` and `AArch32` mnemonic
//! we report the registers the instruction defines, the registers it
//! uses, whether it writes the CPU flags, and whether it touches
//! memory. Unsupported mnemonics (SIMD, FPU, system instructions, and
//! any mnemonic outside the recognised set) are tagged
//! [`InstructionKind::Other`] so the slicer can decide whether to
//! truncate or skip them.
//!
//! Registers are canonicalised to their parent family name so partial
//! writes (`al`, `ax`, `eax` on x86; `w0` on `AArch64`; `a1` / `v1`
//! AAPCS aliases on `AArch32`) are treated as touching the same
//! data-flow node as the full register (`rax`, `x0`, `r0`). This is a
//! safe over-approximation for slicing.

use r2smt_common::Arch;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

/// Coarse classification of an instruction by mnemonic family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InstructionKind {
    /// `mov` / `movzx` / `movsx` / `movsxd`.
    Mov,
    /// `xor` (with the zero-idiom recognised as a special case).
    Xor,
    /// `and`.
    And,
    /// `or`.
    Or,
    /// `add`.
    Add,
    /// `sub`.
    Sub,
    /// Any `imul` form (1-, 2- or 3-operand).
    Imul,
    /// `cmp` — sets flags, no register def.
    Cmp,
    /// `test` — sets flags, no register def.
    Test,
    /// `shl` / `sal`.
    Shl,
    /// `shr`.
    Shr,
    /// `sar`.
    Sar,
    /// `lea` — defines a register from a memory expression without
    /// performing an actual load.
    Lea,
    /// Conditional jump (`jcc`).
    Jcc,
    /// `setcc`.
    SetCc,
    /// `cmovcc`.
    CMovCc,
    /// `jmp` (unconditional).
    Jmp,
    /// `call`.
    Call,
    /// `ret`.
    Ret,
    /// Anything the slicer cannot reason about yet (push/pop, SIMD,
    /// FPU, string ops, system instructions, ...).
    Other,
}

/// Static description of what an instruction reads, writes, and
/// observes.
///
/// Allows four independent structural booleans (`defines_flags`,
/// `reads_flags`, `has_memory_access`, `is_call`) — each describes an
/// orthogonal property the slicer queries, and bundling them into an
/// enum or state machine would either lose information or duplicate
/// every shared field of the struct. The `struct_excessive_bools`
/// lint is suppressed locally with that rationale; treat further
/// flag-style fields as a signal to migrate to a flags enum instead.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct InstructionEffect {
    /// Classification of the mnemonic.
    pub kind: InstructionKind,
    /// Canonical register names defined by the instruction.
    pub defs: Vec<&'static str>,
    /// Canonical register names read by the instruction.
    pub uses: Vec<&'static str>,
    /// `true` if the instruction writes any flag relevant to a `jcc`
    /// (`ZF` / `CF` / `SF` / `OF` / `PF`).
    pub defines_flags: bool,
    /// `true` if the instruction *reads* NZCV — its output depends on
    /// the current flag state without being a regular flag-based
    /// branch. Examples: `AArch64` conditional-select family
    /// (`csel`, `csinc`, …) and `AArch32` predicated execution
    /// (`addeq`, `moveq`, …). The slicer treats these as
    /// flag consumers: keeping the upstream flag-defining
    /// instruction in the slice even when the live register set is
    /// already satisfied.
    pub reads_flags: bool,
    /// `true` if the instruction touches memory through a load or store
    /// the slicer cannot model. Stack-slot accesses recognised by
    /// [`stack_slot`] (e.g. `[rbp - 4]`) do **not** set this flag —
    /// they are surfaced via [`Self::stack_defs`] / [`Self::stack_uses`]
    /// so the slicer can resolve them like a virtual register.
    /// `lea` is **not** a memory access for this purpose.
    pub has_memory_access: bool,
    /// `true` if the instruction is a call.
    pub is_call: bool,
    /// Stack slots written by the instruction (e.g. `"stk_rbp_-4"`).
    pub stack_defs: Vec<String>,
    /// Stack slots read by the instruction.
    pub stack_uses: Vec<String>,
}

/// Canonicalise a register operand name to its parent register family
/// under the supplied ISA.
///
/// Returns `None` for non-registers (immediates, memory expressions,
/// segment selectors, debug / control regs, x86 SIMD/FPU stacks like
/// `xmm0` / `st0`) and for any token that does not resolve in `arch`'s
/// register table. ARM SIMD / FPU aliases (`vN`/`qN`/`dN`/`sN`/`hN`/
/// `bN` on `AArch64`; `vN`/`qN`/`dN`/`sN` on `AArch32`) are
/// recognised and collapse to the synthetic `vN` parent.
///
/// Delegates to [`crate::registers::register_layout`] so the single
/// source of truth for register naming lives in `registers.rs`. The
/// same name can mean different things in different ISAs (e.g. `sp`
/// is the 16-bit alias of `rsp` on x86, a 64-bit stack pointer on
/// `AArch64`, and an alias of `r13` on `AArch32`); the `arch`
/// parameter selects the right resolution.
#[must_use]
pub fn canonical_register(name: &str, arch: Arch) -> Option<&'static str> {
    crate::registers::register_layout(name, arch).map(|layout| layout.parent)
}

/// Extract every register name referenced by an operand under `arch`,
/// including registers used in a memory expression like
/// `[rbp + rax*2 + 8]`.
///
/// Tokens that are not register names (immediates, the `ptr` keyword,
/// segment selectors, …) are skipped. Names that exist in another ISA
/// but not in `arch` are also skipped — this keeps cross-ISA noise
/// (`xor` in an `AArch64` disassembly, `eax` in an ARM listing) from
/// polluting the data-flow graph.
#[must_use]
pub fn registers_in_operand(op: &Operand, arch: Arch) -> Vec<&'static str> {
    let mut out = Vec::new();
    for token in op
        .raw
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
    {
        if let Some(canon) = canonical_register(token, arch)
            && !out.contains(&canon)
        {
            out.push(canon);
        }
    }
    out
}

fn first_register(operands: &[Operand]) -> Option<&'static str> {
    operands
        .first()
        .and_then(|o| canonical_register(&o.raw, Arch::X86_64))
}

fn any_memory_operand(operands: &[Operand]) -> bool {
    operands.iter().any(|o| o.kind == OperandKind::Memory)
}

/// Reports whether the operand list contains a memory access that
/// is *not* a recognised stack slot. Used by the slicer to decide
/// whether to truncate.
fn has_unresolved_memory(operands: &[Operand]) -> bool {
    operands
        .iter()
        .filter(|o| o.kind == OperandKind::Memory)
        .any(|o| stack_slot(o).is_none())
}

/// Pointer width of a recognised stack slot, in bits.
///
/// Inferred from the `byte ptr` / `word ptr` / `dword ptr` /
/// `qword ptr` prefix when present. Without an explicit width prefix
/// we default to 64-bit so the slot is interoperable with native
/// pointer-sized accesses.
fn stack_slot_width(raw: &str) -> u8 {
    let lower = raw.to_ascii_lowercase();
    if lower.contains("qword") {
        64
    } else if lower.contains("dword") {
        32
    } else if lower.contains("word") {
        16
    } else if lower.contains("byte") {
        8
    } else {
        64
    }
}

fn parse_integer(raw: &str) -> Option<i64> {
    let s = raw.trim();
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
        (-1i64, rest.trim())
    } else if let Some(rest) = s.strip_prefix('+') {
        (1i64, rest.trim())
    } else {
        (1i64, s)
    };
    let value = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()?
    } else {
        body.parse::<i64>().ok()?
    };
    Some(sign * value)
}

/// Parse an operand of the form `[rbp ± K]` / `[rsp ± K]` (with
/// optional `qword/dword/word/byte ptr` prefix) into a canonical
/// stack-slot name and its width in bits.
///
/// Returns `None` for memory expressions that depend on dynamic
/// indexing (`[rbp + rax*4]`), non-stack-base registers (`[rax]`),
/// or non-memory operands.
#[must_use]
pub fn stack_slot(operand: &Operand) -> Option<(String, u8)> {
    if operand.kind != OperandKind::Memory {
        return None;
    }
    let raw = operand.raw.trim();
    let lower = raw.to_ascii_lowercase();
    let lb = lower.find('[')?;
    let rb = lower.find(']')?;
    if rb <= lb {
        return None;
    }
    let inner = lower[lb + 1..rb].trim();

    // Reject expressions with scaling (`*`) — those are dynamic
    // indexing and cannot collapse to a constant offset.
    if inner.contains('*') {
        return None;
    }

    let (base_token, offset) = if let Some(idx) = inner.find('+') {
        let base = inner[..idx].trim();
        let rest = inner[idx + 1..].trim();
        let off = parse_integer(rest)?;
        (base, off)
    } else if let Some(idx) = inner.find('-') {
        let base = inner[..idx].trim();
        let rest = inner[idx + 1..].trim();
        let off = parse_integer(rest)?;
        (base, -off)
    } else {
        (inner, 0i64)
    };

    let base_canon = canonical_register(base_token, Arch::X86_64)?;
    if base_canon != "rbp" && base_canon != "rsp" {
        return None;
    }

    // Reject any inner token that is a register but not the base —
    // catches accidental matches like `[rbp + rax]`.
    let extras: Vec<&'static str> = registers_in_operand(operand, Arch::X86_64)
        .into_iter()
        .filter(|r| *r != base_canon)
        .collect();
    if !extras.is_empty() {
        return None;
    }

    let name = format!("stk_{base_canon}_{offset}");
    let width = stack_slot_width(&lower);
    Some((name, width))
}

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

/// Classify and report the effect of a single instruction under
/// `arch`.
///
/// Dispatches on the ISA family: x86 / `x86_64` use the legacy
/// `jcc` / `setcc` / `cmovcc` mnemonic set; `AArch64` uses the
/// 3-operand arithmetic family plus `b.<cond>` branches and the
/// flag-setting `s` suffix (`adds`, `subs`, `ands`). Anything outside
/// the recognised set lands as [`InstructionKind::Other`].
#[must_use]
pub fn analyze(insn: &Instruction, arch: Arch) -> InstructionEffect {
    match arch {
        Arch::X86 | Arch::X86_64 => analyze_x86(insn),
        Arch::Aarch64 => analyze_aarch64(insn),
        Arch::Arm => analyze_aarch32(insn),
        _ => other_effect(insn),
    }
}

fn analyze_aarch32(insn: &Instruction) -> InstructionEffect {
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

fn other_effect(insn: &Instruction) -> InstructionEffect {
    InstructionEffect {
        kind: InstructionKind::Other,
        defs: Vec::new(),
        uses: Vec::new(),
        defines_flags: false,
        has_memory_access: any_memory_operand(&insn.operands),
        is_call: false,
        stack_defs: Vec::new(),
        stack_uses: Vec::new(),
        reads_flags: false,
    }
}

fn analyze_x86(insn: &Instruction) -> InstructionEffect {
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

fn analyze_aarch64(insn: &Instruction) -> InstructionEffect {
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
        _ => other_effect(insn),
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

#[cfg(test)]
mod tests;
