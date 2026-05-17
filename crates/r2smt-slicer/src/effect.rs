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

mod aarch32;
mod aarch64;
mod x86;

use aarch32::analyze_aarch32;
use aarch64::analyze_aarch64;
use x86::analyze_x86;

#[cfg(test)]
mod tests;
