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
mod tests {
    #![allow(clippy::unwrap_used)]

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
}
