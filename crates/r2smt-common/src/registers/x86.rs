//! x86 / `x86_64` register layout + alias tables, extracted from
//! `registers.rs`. Const-fn `RegisterLayout` builders stay in the
//! parent module (reached via `super::`, ancestor-private).

use super::{RegisterLayout, dword, full, high_byte, low_byte, simd_slice, word};

pub(super) fn x86_layout(lower: &str) -> Option<RegisterLayout> {
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

        // The x87 register *stack*. `st` is not an architectural
        // register name: it is the single canonical data-flow node the
        // whole stack collapses onto, reached by tokenising the
        // disassembler spelling `st(0)`..`st(7)` (see
        // `r2smt_slicer::effect::registers_in_operand`). The index is
        // deliberately lost — TOP rotates under every push and pop, so
        // an index-keyed node would alias two different physical
        // registers. The lifter models the individual slots with its own
        // slice-scoped stack instead (`lift/x87.rs`).
        "st" => full("st"),

        // The x87 status word, likewise a synthetic node rather than an
        // operand spelling: `fcom` writes it and `fnstsw` reads it, but
        // no instruction names it. It resolves here so the slicer can
        // canonicalise it wherever a live-set entry has to round-trip
        // through the register table.
        "fsw" => word("fsw"),

        _ => return x86_simd_layout(lower),
    };
    Some(layout)
}

/// x86 SIMD register layout. `xmm<n>` / `ymm<n>` / `zmm<n>` map to bits
/// `[127:0]` / `[255:0]` / `[511:0]` of a synthetic `zmm<n>` parent —
/// the widest architectural view — so every view lands on the same
/// data-flow node without renaming the parent. `mm<n>` (MMX) stays
/// `None` until its own lifter lands, and so does `st<n>`: that is the
/// spelling radare2 uses in *ESIL*, never in disassembly, so resolving
/// it would only ever canonicalise a token no operand carries.
fn x86_simd_layout(lower: &str) -> Option<RegisterLayout> {
    let hi: u16 = match lower.get(..3)? {
        "xmm" => 127,
        "ymm" => 255,
        "zmm" => 511,
        _ => return None,
    };
    let n: u8 = lower[3..].parse().ok()?;
    if n > 31 {
        return None;
    }
    Some(simd_slice(x86_zmm_name(n), 0, hi))
}

const fn x86_zmm_name(n: u8) -> &'static str {
    match n {
        0 => "zmm0",
        1 => "zmm1",
        2 => "zmm2",
        3 => "zmm3",
        4 => "zmm4",
        5 => "zmm5",
        6 => "zmm6",
        7 => "zmm7",
        8 => "zmm8",
        9 => "zmm9",
        10 => "zmm10",
        11 => "zmm11",
        12 => "zmm12",
        13 => "zmm13",
        14 => "zmm14",
        15 => "zmm15",
        16 => "zmm16",
        17 => "zmm17",
        18 => "zmm18",
        19 => "zmm19",
        20 => "zmm20",
        21 => "zmm21",
        22 => "zmm22",
        23 => "zmm23",
        24 => "zmm24",
        25 => "zmm25",
        26 => "zmm26",
        27 => "zmm27",
        28 => "zmm28",
        29 => "zmm29",
        30 => "zmm30",
        _ => "zmm31",
    }
}

fn x86_xmm_alias(parent: &str) -> Option<&'static str> {
    let n: u8 = parent.strip_prefix("zmm")?.parse().ok()?;
    Some(match n {
        0 => "xmm0",
        1 => "xmm1",
        2 => "xmm2",
        3 => "xmm3",
        4 => "xmm4",
        5 => "xmm5",
        6 => "xmm6",
        7 => "xmm7",
        8 => "xmm8",
        9 => "xmm9",
        10 => "xmm10",
        11 => "xmm11",
        12 => "xmm12",
        13 => "xmm13",
        14 => "xmm14",
        15 => "xmm15",
        16 => "xmm16",
        17 => "xmm17",
        18 => "xmm18",
        19 => "xmm19",
        20 => "xmm20",
        21 => "xmm21",
        22 => "xmm22",
        23 => "xmm23",
        24 => "xmm24",
        25 => "xmm25",
        26 => "xmm26",
        27 => "xmm27",
        28 => "xmm28",
        29 => "xmm29",
        30 => "xmm30",
        31 => "xmm31",
        _ => return None,
    })
}

fn x86_zmm_alias(parent: &str) -> Option<&'static str> {
    let n: u8 = parent.strip_prefix("zmm")?.parse().ok()?;
    (n <= 31).then(|| x86_zmm_name(n))
}

fn x86_ymm_alias(parent: &str) -> Option<&'static str> {
    let n: u8 = parent.strip_prefix("zmm")?.parse().ok()?;
    Some(match n {
        0 => "ymm0",
        1 => "ymm1",
        2 => "ymm2",
        3 => "ymm3",
        4 => "ymm4",
        5 => "ymm5",
        6 => "ymm6",
        7 => "ymm7",
        8 => "ymm8",
        9 => "ymm9",
        10 => "ymm10",
        11 => "ymm11",
        12 => "ymm12",
        13 => "ymm13",
        14 => "ymm14",
        15 => "ymm15",
        16 => "ymm16",
        17 => "ymm17",
        18 => "ymm18",
        19 => "ymm19",
        20 => "ymm20",
        21 => "ymm21",
        22 => "ymm22",
        23 => "ymm23",
        24 => "ymm24",
        25 => "ymm25",
        26 => "ymm26",
        27 => "ymm27",
        28 => "ymm28",
        29 => "ymm29",
        30 => "ymm30",
        31 => "ymm31",
        _ => return None,
    })
}

pub(super) fn x86_alias(parent: &str, hi: u16, lo: u16) -> Option<&'static str> {
    // SIMD parents (`zmm<n>`) never collide with GPR parents; the low
    // slice width selects the disassembler-visible view: `[127:0]` →
    // `xmm`, `[255:0]` → `ymm`, `[511:0]` → the `zmm` parent itself.
    if lo == 0 && parent.starts_with("zmm") {
        return match hi {
            127 => x86_xmm_alias(parent),
            255 => x86_ymm_alias(parent),
            511 => x86_zmm_alias(parent),
            _ => None,
        };
    }
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
