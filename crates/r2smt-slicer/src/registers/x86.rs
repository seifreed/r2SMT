//! x86 / `x86_64` register layout + alias tables, extracted from
//! `registers.rs`. Const-fn `RegisterLayout` builders stay in the
//! parent module (reached via `super::`, ancestor-private).

use super::{RegisterLayout, dword, full, high_byte, low_byte, word};

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

        _ => return None,
    };
    Some(layout)
}

pub(super) fn x86_alias(parent: &str, hi: u8, lo: u8) -> Option<&'static str> {
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
