//! `AArch32` register layout + alias tables, extracted from
//! `registers.rs`.

use super::{RegisterLayout, arm32_full, arm32_vector};

pub(super) fn arm32_layout(lower: &str) -> Option<RegisterLayout> {
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

pub(super) fn arm32_alias(parent: &str, hi: u8, lo: u8) -> Option<&'static str> {
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
