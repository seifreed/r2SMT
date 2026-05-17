//! Bit-precise register layouts for the supported ISAs.
//!
//! [`canonical_register`](crate::canonical_register) reports *which*
//! parent register a name belongs to (x86 only today). This module
//! reports *which bits* of that parent the name addresses, so the
//! lifter can replace its historical "treat every sub-register as the
//! full parent" shortcut with bit-precise [`Expr::Extract`] /
//! [`Expr::Concat`] / [`Expr::ZeroExtend`] rewrites.
//!
//! The table is sample-agnostic, pure ISA. References:
//! - x86 / `x86_64`: Intel SDM Vol. 1 §3.4.
//! - `AArch64`: ARM ARM Vol. C §B1.2.1 (general-purpose registers).
//! - `AArch32` (ARM, `ARMv7-A`): ARM ARM Vol. C §B1.3.2.
//!
//! Sub-register write semantics:
//! - **`x86_64`**: writing the 32-bit alias (`eax`, `ecx`, `r8d`, …)
//!   zero-extends into the 64-bit parent; 16/8-bit writes preserve
//!   the surrounding bits.
//! - **`AArch64`**: writing the 32-bit alias (`w0`, `w1`, `wsp`, …)
//!   zero-extends into the 64-bit parent. There are no 16/8-bit GPR
//!   aliases on `AArch64` — sub-byte work happens through the SIMD
//!   register file (see below).
//! - **`AArch32`**: GPRs are 32-bit with no architectural
//!   sub-register slices; `sp`/`lr`/`pc` are ABI aliases for
//!   `r13`/`r14`/`r15`. The full AAPCS alias set (ARM IHI 0042 §5.1.1)
//!   is also recognised — `a1..a4` for `r0..r3`, `v1..v8` for
//!   `r4..r11`, plus `sb` (`r9`), `sl` (`r10`), `fp` (`r11`), `ip`
//!   (`r12`). All collapse to the bare `rN` parent so the slicer
//!   sees AAPCS-named operands as the same data-flow node as the
//!   architectural register.
//! - **`x86` (32-bit)**: shares the `x86_64` layout, but the parent
//!   register is itself 32 bits. The [`RegisterLayout::hi`] field is
//!   interpreted relative to the parent's width supplied by the
//!   caller, so a lifter running at `bits = 32` treats `eax` as a
//!   full write (no zero-extension needed).
//!
//! SIMD / FPU registers (ARM only):
//! - **`AArch64`**: 32 × 128-bit `V` registers (ARM ARM Vol. C
//!   §B1.2.2). The aliases `vN`, `qN`, `dN`, `sN`, `hN`, `bN` all
//!   address slice 127..0 / 63..0 / 31..0 / 15..0 / 7..0 of the same
//!   physical 128-bit register, modelled as a synthetic parent
//!   `v0..v31`. Writes through `D`/`S`/`H`/`B` aliases zero the
//!   upper bits of the parent per `AArch64` SIMD&FP semantics; we
//!   record the slice geometry here and defer the Concat / zero-
//!   extend modelling to the SIMD lifter (not implemented yet).
//! - **`AArch32`**: 32 × 64-bit `D` registers (ARM ARM Vol. C
//!   §A2.6). Aliases `qN` (n=0..15, 128-bit), `dN` (n=0..31, 64-bit)
//!   and `sN` (n=0..31, 32-bit) overlap as `Q_n` = (`D_{2n}`,
//!   `D_{2n+1}`) and `S_n` = half of `D_{⌊n/2⌋}`. We mirror the
//!   `AArch64` model with a synthetic `v0..v15` parent identifier
//!   that holds 128 bits each; D and S aliases land at the
//!   appropriate sub-slices of that parent. `AArch32` has no
//!   `bN`/`hN` register naming, and `vN` is not real `AArch32` NEON
//!   syntax — those names are reserved for the AAPCS GPR aliases
//!   above. The internal `vN` parent identifier is therefore only
//!   surfaced by [`alias_for`] reverse lookups, never by
//!   [`register_layout`] forward queries.
//! - **`x86_64`**: SIMD / FPU stacks (`xmm0`, `ymm0`, `zmm0`,
//!   `st0`…`st7`, MMX `mm0`…) are out of scope and still resolve to
//!   `None`. Adding them is a separate exercise gated on an x86
//!   SIMD lifter.
//!
//! Name disambiguation across ISAs is critical: `AArch64` `sp` is
//! the 64-bit stack pointer; x86 `sp` is the 16-bit alias of `rsp`;
//! `AArch32` `sp` is the alias of `r13`. The same string genuinely
//! means three different things, so [`register_layout`] takes an
//! [`Arch`] parameter to pick the right table.
//!
//! [`Expr::Extract`]: r2smt_ir::expr::Expr::Extract
//! [`Expr::Concat`]: r2smt_ir::expr::Expr::Concat
//! [`Expr::ZeroExtend`]: r2smt_ir::expr::Expr::ZeroExtend

use r2smt_common::Arch;

// ===================== shared: RegisterLayout + dispatchers + const builders =====================

/// Layout of a named register against its canonical parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterLayout {
    /// Canonical parent name (e.g. `"rax"`, `"x0"`, `"r0"`).
    pub parent: &'static str,
    /// Inclusive low bit offset within the parent (0 for `al`, 8 for
    /// `ah`, 0 for `eax`).
    pub lo: u8,
    /// Inclusive high bit offset within the parent (7 for `al`, 15 for
    /// `ah`, 31 for `eax`, 63 for `rax`).
    pub hi: u8,
    /// `true` if this alias is the 32-bit doubleword form whose write
    /// zero-extends into a 64-bit parent on `x86_64` / `AArch64`.
    /// False for 64/16/8-bit aliases.
    pub zero_extends_parent_64: bool,
}

impl RegisterLayout {
    /// Width of the register slice in bits (`hi - lo + 1`).
    #[must_use]
    pub const fn width(&self) -> u8 {
        self.hi - self.lo + 1
    }
}

/// Look up the [`RegisterLayout`] for a register operand name under
/// the given ISA.
///
/// Returns `None` for non-GPR registers (SIMD `xmm0`, FPU `st0`,
/// segment selectors, debug / control registers, …) and for any token
/// that does not match a known register alias in `arch`.
/// Case-insensitive; leading / trailing whitespace is trimmed.
#[must_use]
pub fn register_layout(name: &str, arch: Arch) -> Option<RegisterLayout> {
    let lower = name.trim().to_ascii_lowercase();
    match arch {
        Arch::X86 | Arch::X86_64 => x86_layout(&lower),
        Arch::Aarch64 => aarch64_layout(&lower),
        Arch::Arm => arm32_layout(&lower),
        // `Arch` is `#[non_exhaustive]`; any future ISA falls back to
        // no-match until its layout table is added.
        _ => None,
    }
}

/// Reverse lookup: given a `(parent, hi, lo)` triple, return the
/// canonical analyst-facing alias if one exists in `arch`.
///
/// Used by the pretty-printer to render `Extract(rax, 7, 0)` as `al`
/// or `Extract(x0, 31, 0)` as `w0`. Returns `None` for slices that
/// do not correspond to a named sub-register.
#[must_use]
pub fn alias_for(parent: &str, hi: u8, lo: u8, arch: Arch) -> Option<&'static str> {
    match arch {
        Arch::X86 | Arch::X86_64 => x86_alias(parent, hi, lo),
        Arch::Aarch64 => aarch64_alias(parent, hi, lo),
        Arch::Arm => arm32_alias(parent, hi, lo),
        _ => None,
    }
}

const fn full(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 63,
        zero_extends_parent_64: false,
    }
}

const fn dword(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 31,
        zero_extends_parent_64: true,
    }
}

const fn word(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 15,
        zero_extends_parent_64: false,
    }
}

const fn low_byte(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 7,
        zero_extends_parent_64: false,
    }
}

const fn high_byte(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 8,
        hi: 15,
        zero_extends_parent_64: false,
    }
}

const fn aarch64_full(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 63,
        zero_extends_parent_64: false,
    }
}

const fn aarch64_dword(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 31,
        zero_extends_parent_64: true,
    }
}

const fn aarch64_vector(parent: &'static str, lo: u8, hi: u8) -> RegisterLayout {
    // SIMD slice — `zero_extends_parent_64` is GPR-specific (32→64
    // dword zero-extension) and does not capture the SIMD write
    // semantic of zero-extending to 128. We leave it `false` here
    // and defer the SIMD-write modelling to the lifter.
    RegisterLayout {
        parent,
        lo,
        hi,
        zero_extends_parent_64: false,
    }
}

const fn arm32_full(parent: &'static str) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo: 0,
        hi: 31,
        zero_extends_parent_64: false,
    }
}

const fn arm32_vector(parent: &'static str, lo: u8, hi: u8) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo,
        hi,
        zero_extends_parent_64: false,
    }
}

// ===================== x86 / x86_64 =====================

fn x86_layout(lower: &str) -> Option<RegisterLayout> {
    let layout = match lower {
        "rax" => full("rax"),
        "eax" => dword("rax"),
        "ax" => word("rax"),
        "al" => low_byte("rax"),
        "ah" => high_byte("rax"),

        "rbx" => full("rbx"),
        "ebx" => dword("rbx"),
        "bx" => word("rbx"),
        "bl" => low_byte("rbx"),
        "bh" => high_byte("rbx"),

        "rcx" => full("rcx"),
        "ecx" => dword("rcx"),
        "cx" => word("rcx"),
        "cl" => low_byte("rcx"),
        "ch" => high_byte("rcx"),

        "rdx" => full("rdx"),
        "edx" => dword("rdx"),
        "dx" => word("rdx"),
        "dl" => low_byte("rdx"),
        "dh" => high_byte("rdx"),

        "rsi" => full("rsi"),
        "esi" => dword("rsi"),
        "si" => word("rsi"),
        "sil" => low_byte("rsi"),

        "rdi" => full("rdi"),
        "edi" => dword("rdi"),
        "di" => word("rdi"),
        "dil" => low_byte("rdi"),

        "rbp" => full("rbp"),
        "ebp" => dword("rbp"),
        "bp" => word("rbp"),
        "bpl" => low_byte("rbp"),

        "rsp" => full("rsp"),
        "esp" => dword("rsp"),
        "sp" => word("rsp"),
        "spl" => low_byte("rsp"),

        "rip" => full("rip"),
        "eip" => dword("rip"),
        "ip" => word("rip"),

        "r8" => full("r8"),
        "r8d" => dword("r8"),
        "r8w" => word("r8"),
        "r8b" => low_byte("r8"),

        "r9" => full("r9"),
        "r9d" => dword("r9"),
        "r9w" => word("r9"),
        "r9b" => low_byte("r9"),

        "r10" => full("r10"),
        "r10d" => dword("r10"),
        "r10w" => word("r10"),
        "r10b" => low_byte("r10"),

        "r11" => full("r11"),
        "r11d" => dword("r11"),
        "r11w" => word("r11"),
        "r11b" => low_byte("r11"),

        "r12" => full("r12"),
        "r12d" => dword("r12"),
        "r12w" => word("r12"),
        "r12b" => low_byte("r12"),

        "r13" => full("r13"),
        "r13d" => dword("r13"),
        "r13w" => word("r13"),
        "r13b" => low_byte("r13"),

        "r14" => full("r14"),
        "r14d" => dword("r14"),
        "r14w" => word("r14"),
        "r14b" => low_byte("r14"),

        "r15" => full("r15"),
        "r15d" => dword("r15"),
        "r15w" => word("r15"),
        "r15b" => low_byte("r15"),

        _ => return None,
    };
    Some(layout)
}

fn x86_alias(parent: &str, hi: u8, lo: u8) -> Option<&'static str> {
    match (parent, hi, lo) {
        ("rax", 63, 0) => Some("rax"),
        ("rax", 31, 0) => Some("eax"),
        ("rax", 15, 0) => Some("ax"),
        ("rax", 7, 0) => Some("al"),
        ("rax", 15, 8) => Some("ah"),

        ("rbx", 63, 0) => Some("rbx"),
        ("rbx", 31, 0) => Some("ebx"),
        ("rbx", 15, 0) => Some("bx"),
        ("rbx", 7, 0) => Some("bl"),
        ("rbx", 15, 8) => Some("bh"),

        ("rcx", 63, 0) => Some("rcx"),
        ("rcx", 31, 0) => Some("ecx"),
        ("rcx", 15, 0) => Some("cx"),
        ("rcx", 7, 0) => Some("cl"),
        ("rcx", 15, 8) => Some("ch"),

        ("rdx", 63, 0) => Some("rdx"),
        ("rdx", 31, 0) => Some("edx"),
        ("rdx", 15, 0) => Some("dx"),
        ("rdx", 7, 0) => Some("dl"),
        ("rdx", 15, 8) => Some("dh"),

        ("rsi", 63, 0) => Some("rsi"),
        ("rsi", 31, 0) => Some("esi"),
        ("rsi", 15, 0) => Some("si"),
        ("rsi", 7, 0) => Some("sil"),

        ("rdi", 63, 0) => Some("rdi"),
        ("rdi", 31, 0) => Some("edi"),
        ("rdi", 15, 0) => Some("di"),
        ("rdi", 7, 0) => Some("dil"),

        ("rbp", 63, 0) => Some("rbp"),
        ("rbp", 31, 0) => Some("ebp"),
        ("rbp", 15, 0) => Some("bp"),
        ("rbp", 7, 0) => Some("bpl"),

        ("rsp", 63, 0) => Some("rsp"),
        ("rsp", 31, 0) => Some("esp"),
        ("rsp", 15, 0) => Some("sp"),
        ("rsp", 7, 0) => Some("spl"),

        ("rip", 63, 0) => Some("rip"),
        ("rip", 31, 0) => Some("eip"),
        ("rip", 15, 0) => Some("ip"),

        (p, 63, 0) => extended_alias(p, ""),
        (p, 31, 0) => extended_alias(p, "d"),
        (p, 15, 0) => extended_alias(p, "w"),
        (p, 7, 0) => extended_alias(p, "b"),

        _ => None,
    }
}

fn extended_alias(parent: &str, suffix: &str) -> Option<&'static str> {
    match (parent, suffix) {
        ("r8", "") => Some("r8"),
        ("r8", "d") => Some("r8d"),
        ("r8", "w") => Some("r8w"),
        ("r8", "b") => Some("r8b"),
        ("r9", "") => Some("r9"),
        ("r9", "d") => Some("r9d"),
        ("r9", "w") => Some("r9w"),
        ("r9", "b") => Some("r9b"),
        ("r10", "") => Some("r10"),
        ("r10", "d") => Some("r10d"),
        ("r10", "w") => Some("r10w"),
        ("r10", "b") => Some("r10b"),
        ("r11", "") => Some("r11"),
        ("r11", "d") => Some("r11d"),
        ("r11", "w") => Some("r11w"),
        ("r11", "b") => Some("r11b"),
        ("r12", "") => Some("r12"),
        ("r12", "d") => Some("r12d"),
        ("r12", "w") => Some("r12w"),
        ("r12", "b") => Some("r12b"),
        ("r13", "") => Some("r13"),
        ("r13", "d") => Some("r13d"),
        ("r13", "w") => Some("r13w"),
        ("r13", "b") => Some("r13b"),
        ("r14", "") => Some("r14"),
        ("r14", "d") => Some("r14d"),
        ("r14", "w") => Some("r14w"),
        ("r14", "b") => Some("r14b"),
        ("r15", "") => Some("r15"),
        ("r15", "d") => Some("r15d"),
        ("r15", "w") => Some("r15w"),
        ("r15", "b") => Some("r15b"),
        _ => None,
    }
}

// ===================== AArch64 =====================

fn aarch64_layout(lower: &str) -> Option<RegisterLayout> {
    // x0..x30 / w0..w30
    if let Some(stripped) = lower.strip_prefix('x')
        && let Ok(n) = stripped.parse::<u8>()
        && n <= 30
    {
        return Some(aarch64_full(aarch64_x_name(n)));
    }
    if let Some(stripped) = lower.strip_prefix('w')
        && let Ok(n) = stripped.parse::<u8>()
        && n <= 30
    {
        return Some(aarch64_dword(aarch64_x_name(n)));
    }
    // SIMD / FPU: vN / qN (128) / dN (64) / sN (32) / hN (16) / bN (8).
    // All n ∈ 0..=31. Every alias collapses to the synthetic `vN`
    // parent so the slicer detects aliasing across views.
    if let Some(layout) = aarch64_simd_layout(lower) {
        return Some(layout);
    }
    match lower {
        // Stack pointer.
        "sp" => Some(aarch64_full("sp")),
        "wsp" => Some(aarch64_dword("sp")),
        // Zero register.
        "xzr" => Some(aarch64_full("xzr")),
        "wzr" => Some(aarch64_dword("xzr")),
        // Program counter (64-bit on AArch64, no 32-bit alias).
        "pc" => Some(aarch64_full("pc")),
        // ABI aliases — fall through to the bare register so SSA
        // renames stay consistent regardless of the disassembler's
        // spelling.
        "lr" => Some(aarch64_full("x30")),
        "fp" => Some(aarch64_full("x29")),
        _ => None,
    }
}

fn aarch64_simd_layout(lower: &str) -> Option<RegisterLayout> {
    let prefix = lower.chars().next()?;
    let hi = match prefix {
        'v' | 'q' => 127u8,
        'd' => 63,
        's' => 31,
        'h' => 15,
        'b' => 7,
        _ => return None,
    };
    let stripped = &lower[prefix.len_utf8()..];
    let n: u8 = stripped.parse().ok()?;
    if n > 31 {
        return None;
    }
    Some(aarch64_vector(aarch64_v_name(n), 0, hi))
}

fn aarch64_alias(parent: &str, hi: u8, lo: u8) -> Option<&'static str> {
    // SIMD parents start with 'v' and never collide with GPR parents,
    // so dispatching first keeps the GPR catch-all (`(p, 31, 0) => ...`)
    // from swallowing v0(31, 0) and returning None.
    if lo == 0 && parent.starts_with('v') {
        return aarch64_simd_alias(parent, hi);
    }
    match (parent, hi, lo) {
        ("sp", 63, 0) => Some("sp"),
        ("sp", 31, 0) => Some("wsp"),
        ("xzr", 63, 0) => Some("xzr"),
        ("xzr", 31, 0) => Some("wzr"),
        ("pc", 63, 0) => Some("pc"),
        ("x29", 63, 0) => Some("fp"),
        ("x30", 63, 0) => Some("lr"),
        (parent, 63, 0) => aarch64_xn_alias(parent),
        (parent, 31, 0) => aarch64_wn_alias(parent),
        _ => None,
    }
}

fn aarch64_simd_alias(parent: &str, hi: u8) -> Option<&'static str> {
    let stripped = parent.strip_prefix('v')?;
    let n: u8 = stripped.parse().ok()?;
    if n > 31 {
        return None;
    }
    match hi {
        127 => aarch64_vn_alias(n),
        63 => aarch64_dn_alias(n),
        31 => aarch64_sn_alias(n),
        15 => aarch64_hn_alias(n),
        7 => aarch64_bn_alias(n),
        _ => None,
    }
}

const fn aarch64_x_name(n: u8) -> &'static str {
    match n {
        0 => "x0",
        1 => "x1",
        2 => "x2",
        3 => "x3",
        4 => "x4",
        5 => "x5",
        6 => "x6",
        7 => "x7",
        8 => "x8",
        9 => "x9",
        10 => "x10",
        11 => "x11",
        12 => "x12",
        13 => "x13",
        14 => "x14",
        15 => "x15",
        16 => "x16",
        17 => "x17",
        18 => "x18",
        19 => "x19",
        20 => "x20",
        21 => "x21",
        22 => "x22",
        23 => "x23",
        24 => "x24",
        25 => "x25",
        26 => "x26",
        27 => "x27",
        28 => "x28",
        29 => "x29",
        _ => "x30",
    }
}

fn aarch64_xn_alias(parent: &str) -> Option<&'static str> {
    match parent {
        "x0" => Some("x0"),
        "x1" => Some("x1"),
        "x2" => Some("x2"),
        "x3" => Some("x3"),
        "x4" => Some("x4"),
        "x5" => Some("x5"),
        "x6" => Some("x6"),
        "x7" => Some("x7"),
        "x8" => Some("x8"),
        "x9" => Some("x9"),
        "x10" => Some("x10"),
        "x11" => Some("x11"),
        "x12" => Some("x12"),
        "x13" => Some("x13"),
        "x14" => Some("x14"),
        "x15" => Some("x15"),
        "x16" => Some("x16"),
        "x17" => Some("x17"),
        "x18" => Some("x18"),
        "x19" => Some("x19"),
        "x20" => Some("x20"),
        "x21" => Some("x21"),
        "x22" => Some("x22"),
        "x23" => Some("x23"),
        "x24" => Some("x24"),
        "x25" => Some("x25"),
        "x26" => Some("x26"),
        "x27" => Some("x27"),
        "x28" => Some("x28"),
        // x29/x30 are returned as their ABI aliases (fp/lr) by the
        // outer match.
        _ => None,
    }
}

fn aarch64_wn_alias(parent: &str) -> Option<&'static str> {
    match parent {
        "x0" => Some("w0"),
        "x1" => Some("w1"),
        "x2" => Some("w2"),
        "x3" => Some("w3"),
        "x4" => Some("w4"),
        "x5" => Some("w5"),
        "x6" => Some("w6"),
        "x7" => Some("w7"),
        "x8" => Some("w8"),
        "x9" => Some("w9"),
        "x10" => Some("w10"),
        "x11" => Some("w11"),
        "x12" => Some("w12"),
        "x13" => Some("w13"),
        "x14" => Some("w14"),
        "x15" => Some("w15"),
        "x16" => Some("w16"),
        "x17" => Some("w17"),
        "x18" => Some("w18"),
        "x19" => Some("w19"),
        "x20" => Some("w20"),
        "x21" => Some("w21"),
        "x22" => Some("w22"),
        "x23" => Some("w23"),
        "x24" => Some("w24"),
        "x25" => Some("w25"),
        "x26" => Some("w26"),
        "x27" => Some("w27"),
        "x28" => Some("w28"),
        "x29" => Some("w29"),
        "x30" => Some("w30"),
        _ => None,
    }
}

const fn aarch64_v_name(n: u8) -> &'static str {
    match n {
        0 => "v0",
        1 => "v1",
        2 => "v2",
        3 => "v3",
        4 => "v4",
        5 => "v5",
        6 => "v6",
        7 => "v7",
        8 => "v8",
        9 => "v9",
        10 => "v10",
        11 => "v11",
        12 => "v12",
        13 => "v13",
        14 => "v14",
        15 => "v15",
        16 => "v16",
        17 => "v17",
        18 => "v18",
        19 => "v19",
        20 => "v20",
        21 => "v21",
        22 => "v22",
        23 => "v23",
        24 => "v24",
        25 => "v25",
        26 => "v26",
        27 => "v27",
        28 => "v28",
        29 => "v29",
        30 => "v30",
        _ => "v31",
    }
}

fn aarch64_vn_alias(n: u8) -> Option<&'static str> {
    (n <= 31).then(|| aarch64_v_name(n))
}

fn aarch64_dn_alias(n: u8) -> Option<&'static str> {
    if n > 31 {
        return None;
    }
    Some(AARCH64_D_NAMES[n as usize])
}

fn aarch64_sn_alias(n: u8) -> Option<&'static str> {
    if n > 31 {
        return None;
    }
    Some(AARCH64_S_NAMES[n as usize])
}

fn aarch64_hn_alias(n: u8) -> Option<&'static str> {
    if n > 31 {
        return None;
    }
    Some(AARCH64_H_NAMES[n as usize])
}

fn aarch64_bn_alias(n: u8) -> Option<&'static str> {
    if n > 31 {
        return None;
    }
    Some(AARCH64_B_NAMES[n as usize])
}

const AARCH64_D_NAMES: [&str; 32] = [
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
    "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
    "d28", "d29", "d30", "d31",
];

const AARCH64_S_NAMES: [&str; 32] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    "s28", "s29", "s30", "s31",
];

const AARCH64_H_NAMES: [&str; 32] = [
    "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "h13", "h14",
    "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23", "h24", "h25", "h26", "h27",
    "h28", "h29", "h30", "h31",
];

const AARCH64_B_NAMES: [&str; 32] = [
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12", "b13", "b14",
    "b15", "b16", "b17", "b18", "b19", "b20", "b21", "b22", "b23", "b24", "b25", "b26", "b27",
    "b28", "b29", "b30", "b31",
];

// ===================== AArch32 =====================

fn arm32_layout(lower: &str) -> Option<RegisterLayout> {
    // r0..r15
    if let Some(stripped) = lower.strip_prefix('r')
        && let Ok(n) = stripped.parse::<u8>()
        && n <= 15
    {
        return Some(arm32_full(arm32_r_name(n)));
    }
    // AAPCS GPR aliases (ARM IHI 0042 §5.1.1): a1..a4 are the
    // argument / result registers (r0..r3); v1..v8 are the
    // callee-saved variable registers (r4..r11). These are NOT
    // separate physical registers — they alias r0..r11 and the
    // slicer treats them as such. Real AArch32 disassemblers emit
    // these names when the binary is built against AAPCS-aware
    // toolchains; SIMD / NEON in AArch32 always uses qN / dN / sN
    // spelling, so `vN` here is unambiguously a GPR.
    if let Some(parent) = arm32_aapcs_alias(lower) {
        return Some(arm32_full(parent));
    }
    // SIMD / FPU: q0..q15 / d0..d31 / s0..s31. (`vN` was a synthetic
    // synonym in an earlier revision but collides with the AAPCS
    // GPR alias above — real AArch32 NEON syntax does not use `vN`.)
    if let Some(layout) = arm32_simd_layout(lower) {
        return Some(layout);
    }
    match lower {
        "sp" => Some(arm32_full("r13")),
        "lr" => Some(arm32_full("r14")),
        "pc" => Some(arm32_full("r15")),
        _ => None,
    }
}

fn arm32_aapcs_alias(lower: &str) -> Option<&'static str> {
    match lower {
        "a1" => Some("r0"),
        "a2" => Some("r1"),
        "a3" => Some("r2"),
        "a4" => Some("r3"),
        "v1" => Some("r4"),
        "v2" => Some("r5"),
        "v3" => Some("r6"),
        "v4" => Some("r7"),
        "v5" => Some("r8"),
        "v6" | "sb" => Some("r9"),
        "v7" | "sl" => Some("r10"),
        "v8" | "fp" => Some("r11"),
        "ip" => Some("r12"),
        _ => None,
    }
}

fn arm32_simd_layout(lower: &str) -> Option<RegisterLayout> {
    let prefix = lower.chars().next()?;
    let stripped = &lower[prefix.len_utf8()..];
    let n: u8 = stripped.parse().ok()?;
    match prefix {
        'q' if n <= 15 => Some(arm32_vector(arm32_v_name(n), 0, 127)),
        'd' if n <= 31 => {
            let parent = arm32_v_name(n / 2);
            let lo = (n % 2) * 64;
            Some(arm32_vector(parent, lo, lo + 63))
        }
        's' if n <= 31 => {
            let parent = arm32_v_name(n / 4);
            let lo = (n % 4) * 32;
            Some(arm32_vector(parent, lo, lo + 31))
        }
        _ => None,
    }
}

fn arm32_alias(parent: &str, hi: u8, lo: u8) -> Option<&'static str> {
    if let Some(stripped) = parent.strip_prefix('v')
        && let Ok(k) = stripped.parse::<u8>()
        && k <= 15
    {
        return arm32_simd_alias(k, hi, lo);
    }
    if hi != 31 || lo != 0 {
        return None;
    }
    match parent {
        "r13" => Some("sp"),
        "r14" => Some("lr"),
        "r15" => Some("pc"),
        p => arm32_rn_alias(p),
    }
}

fn arm32_simd_alias(k: u8, hi: u8, lo: u8) -> Option<&'static str> {
    // `qN` is preferred over the synthetic `vN` since q-form is the
    // 128-bit name the AArch32 disassembler actually emits.
    match (hi, lo) {
        (127, 0) => arm32_q_alias(k),
        (63, 0) => arm32_d_alias(2 * k),
        (127, 64) => arm32_d_alias(2 * k + 1),
        (31, 0) if k < 8 => arm32_s_alias(4 * k),
        (63, 32) if k < 8 => arm32_s_alias(4 * k + 1),
        (95, 64) if k < 8 => arm32_s_alias(4 * k + 2),
        (127, 96) if k < 8 => arm32_s_alias(4 * k + 3),
        _ => None,
    }
}

const fn arm32_r_name(n: u8) -> &'static str {
    match n {
        0 => "r0",
        1 => "r1",
        2 => "r2",
        3 => "r3",
        4 => "r4",
        5 => "r5",
        6 => "r6",
        7 => "r7",
        8 => "r8",
        9 => "r9",
        10 => "r10",
        11 => "r11",
        12 => "r12",
        13 => "r13",
        14 => "r14",
        _ => "r15",
    }
}

fn arm32_rn_alias(parent: &str) -> Option<&'static str> {
    match parent {
        "r0" => Some("r0"),
        "r1" => Some("r1"),
        "r2" => Some("r2"),
        "r3" => Some("r3"),
        "r4" => Some("r4"),
        "r5" => Some("r5"),
        "r6" => Some("r6"),
        "r7" => Some("r7"),
        "r8" => Some("r8"),
        "r9" => Some("r9"),
        "r10" => Some("r10"),
        "r11" => Some("r11"),
        "r12" => Some("r12"),
        // r13/r14/r15 fall through to sp/lr/pc in the outer match.
        _ => None,
    }
}

const fn arm32_v_name(n: u8) -> &'static str {
    match n {
        0 => "v0",
        1 => "v1",
        2 => "v2",
        3 => "v3",
        4 => "v4",
        5 => "v5",
        6 => "v6",
        7 => "v7",
        8 => "v8",
        9 => "v9",
        10 => "v10",
        11 => "v11",
        12 => "v12",
        13 => "v13",
        14 => "v14",
        _ => "v15",
    }
}

fn arm32_q_alias(n: u8) -> Option<&'static str> {
    if n > 15 {
        return None;
    }
    Some(ARM32_Q_NAMES[n as usize])
}

fn arm32_d_alias(n: u8) -> Option<&'static str> {
    if n > 31 {
        return None;
    }
    Some(ARM32_D_NAMES[n as usize])
}

fn arm32_s_alias(n: u8) -> Option<&'static str> {
    if n > 31 {
        return None;
    }
    Some(ARM32_S_NAMES[n as usize])
}

const ARM32_Q_NAMES: [&str; 16] = [
    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14",
    "q15",
];

const ARM32_D_NAMES: [&str; 32] = [
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14",
    "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
    "d28", "d29", "d30", "d31",
];

const ARM32_S_NAMES: [&str; 32] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27",
    "s28", "s29", "s30", "s31",
];

#[cfg(test)]
mod tests;
