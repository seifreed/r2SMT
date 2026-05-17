//! `AArch64` register layout + alias tables, extracted from
//! `registers.rs`.

use super::{RegisterLayout, aarch64_dword, aarch64_full, aarch64_vector};

pub(super) fn aarch64_layout(lower: &str) -> Option<RegisterLayout> {
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

pub(super) fn aarch64_alias(parent: &str, hi: u8, lo: u8) -> Option<&'static str> {
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
