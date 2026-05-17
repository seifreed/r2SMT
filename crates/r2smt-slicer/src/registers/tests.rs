#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;

#[test]
fn rax_family_widths_are_correct() {
    for name in ["rax", "eax", "ax", "al", "ah"] {
        let layout = register_layout(name, Arch::X86_64).unwrap();
        assert_eq!(layout.parent, "rax");
    }
    assert_eq!(register_layout("rax", Arch::X86_64).unwrap().width(), 64);
    assert_eq!(register_layout("eax", Arch::X86_64).unwrap().width(), 32);
    assert_eq!(register_layout("ax", Arch::X86_64).unwrap().width(), 16);
    assert_eq!(register_layout("al", Arch::X86_64).unwrap().width(), 8);
    assert_eq!(register_layout("ah", Arch::X86_64).unwrap().width(), 8);
}

#[test]
fn ah_addresses_bits_8_to_15() {
    let layout = register_layout("ah", Arch::X86_64).unwrap();
    assert_eq!(layout.parent, "rax");
    assert_eq!(layout.lo, 8);
    assert_eq!(layout.hi, 15);
    assert!(!layout.zero_extends_parent_64);
}

#[test]
fn eax_zero_extends_parent_on_x86_64() {
    let layout = register_layout("eax", Arch::X86_64).unwrap();
    assert_eq!(layout.parent, "rax");
    assert_eq!(layout.lo, 0);
    assert_eq!(layout.hi, 31);
    assert!(layout.zero_extends_parent_64);
}

#[test]
fn extended_gpr_r8_family_resolves() {
    let r8 = register_layout("r8", Arch::X86_64).unwrap();
    assert_eq!(r8.parent, "r8");
    assert_eq!(r8.width(), 64);
    let r8d = register_layout("r8d", Arch::X86_64).unwrap();
    assert_eq!(r8d.parent, "r8");
    assert_eq!(r8d.width(), 32);
    assert!(r8d.zero_extends_parent_64);
    assert_eq!(register_layout("r15b", Arch::X86_64).unwrap().width(), 8);
}

#[test]
fn case_and_whitespace_insensitive() {
    assert_eq!(
        register_layout(" Eax ", Arch::X86_64).map(|l| l.width()),
        Some(32)
    );
    assert_eq!(register_layout("AH", Arch::X86_64).unwrap().lo, 8);
}

#[test]
fn non_gpr_returns_none() {
    assert!(register_layout("xmm0", Arch::X86_64).is_none());
    assert!(register_layout("st0", Arch::X86_64).is_none());
    assert!(register_layout("ptr", Arch::X86_64).is_none());
    assert!(register_layout("0x10", Arch::X86_64).is_none());
    assert!(register_layout("", Arch::X86_64).is_none());
}

#[test]
fn alias_for_round_trips_named_subregisters() {
    assert_eq!(alias_for("rax", 7, 0, Arch::X86_64), Some("al"));
    assert_eq!(alias_for("rax", 15, 8, Arch::X86_64), Some("ah"));
    assert_eq!(alias_for("rax", 15, 0, Arch::X86_64), Some("ax"));
    assert_eq!(alias_for("rax", 31, 0, Arch::X86_64), Some("eax"));
    assert_eq!(alias_for("rax", 63, 0, Arch::X86_64), Some("rax"));
    assert_eq!(alias_for("r8", 31, 0, Arch::X86_64), Some("r8d"));
    assert_eq!(alias_for("rsi", 7, 0, Arch::X86_64), Some("sil"));
}

#[test]
fn alias_for_returns_none_for_arbitrary_slices() {
    // Bits 23..16 of rax — no standard mnemonic for "third byte".
    assert_eq!(alias_for("rax", 23, 16, Arch::X86_64), None);
    // Bogus parent name.
    assert_eq!(alias_for("xyz", 7, 0, Arch::X86_64), None);
}

// --- AArch64 ---

#[test]
fn aarch64_x_and_w_family_widths() {
    let x0 = register_layout("x0", Arch::Aarch64).unwrap();
    assert_eq!(x0.parent, "x0");
    assert_eq!(x0.width(), 64);
    assert!(!x0.zero_extends_parent_64);
    let w0 = register_layout("w0", Arch::Aarch64).unwrap();
    assert_eq!(w0.parent, "x0");
    assert_eq!(w0.width(), 32);
    assert!(w0.zero_extends_parent_64);
}

#[test]
fn aarch64_x30_and_w30_resolve() {
    assert_eq!(register_layout("x30", Arch::Aarch64).unwrap().parent, "x30");
    assert_eq!(register_layout("w30", Arch::Aarch64).unwrap().parent, "x30");
}

#[test]
fn aarch64_sp_wsp_xzr_wzr_resolve() {
    assert_eq!(register_layout("sp", Arch::Aarch64).unwrap().parent, "sp");
    assert_eq!(register_layout("sp", Arch::Aarch64).unwrap().width(), 64);
    assert!(
        register_layout("wsp", Arch::Aarch64)
            .unwrap()
            .zero_extends_parent_64
    );
    assert_eq!(register_layout("xzr", Arch::Aarch64).unwrap().parent, "xzr");
    assert_eq!(register_layout("wzr", Arch::Aarch64).unwrap().parent, "xzr");
}

#[test]
fn aarch64_abi_aliases_resolve_to_underlying_xn() {
    let lr = register_layout("lr", Arch::Aarch64).unwrap();
    assert_eq!(lr.parent, "x30");
    let fp = register_layout("fp", Arch::Aarch64).unwrap();
    assert_eq!(fp.parent, "x29");
}

#[test]
fn sp_means_different_things_in_x86_and_aarch64() {
    // Same string, different ISA, different layout.
    let x86_sp = register_layout("sp", Arch::X86_64).unwrap();
    assert_eq!(x86_sp.parent, "rsp");
    assert_eq!(x86_sp.width(), 16);
    let aarch64_sp = register_layout("sp", Arch::Aarch64).unwrap();
    assert_eq!(aarch64_sp.parent, "sp");
    assert_eq!(aarch64_sp.width(), 64);
}

#[test]
fn aarch64_does_not_recognise_x86_names() {
    assert!(register_layout("rax", Arch::Aarch64).is_none());
    assert!(register_layout("ah", Arch::Aarch64).is_none());
    assert!(register_layout("eax", Arch::Aarch64).is_none());
}

#[test]
fn aarch64_alias_for_round_trips() {
    assert_eq!(alias_for("x0", 63, 0, Arch::Aarch64), Some("x0"));
    assert_eq!(alias_for("x0", 31, 0, Arch::Aarch64), Some("w0"));
    assert_eq!(alias_for("sp", 63, 0, Arch::Aarch64), Some("sp"));
    assert_eq!(alias_for("sp", 31, 0, Arch::Aarch64), Some("wsp"));
    assert_eq!(alias_for("xzr", 31, 0, Arch::Aarch64), Some("wzr"));
    assert_eq!(alias_for("x29", 63, 0, Arch::Aarch64), Some("fp"));
    assert_eq!(alias_for("x30", 63, 0, Arch::Aarch64), Some("lr"));
    // Bogus parent under AArch64.
    assert_eq!(alias_for("rax", 7, 0, Arch::Aarch64), None);
}

// --- AArch32 ---

#[test]
fn arm32_r_n_full_widths_are_32() {
    for n in 0u8..=15 {
        let name = format!("r{n}");
        let layout = register_layout(&name, Arch::Arm).unwrap();
        let expected = [
            "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13",
            "r14", "r15",
        ][usize::from(n)];
        assert_eq!(layout.parent, expected);
        assert_eq!(layout.width(), 32);
        assert!(!layout.zero_extends_parent_64);
    }
}

#[test]
fn arm32_sp_lr_pc_alias_r13_r14_r15() {
    assert_eq!(register_layout("sp", Arch::Arm).unwrap().parent, "r13");
    assert_eq!(register_layout("lr", Arch::Arm).unwrap().parent, "r14");
    assert_eq!(register_layout("pc", Arch::Arm).unwrap().parent, "r15");
}

#[test]
fn r10_disambiguates_across_x86_and_arm() {
    let x86_r10 = register_layout("r10", Arch::X86_64).unwrap();
    assert_eq!(x86_r10.parent, "r10");
    assert_eq!(x86_r10.width(), 64);
    let arm_r10 = register_layout("r10", Arch::Arm).unwrap();
    assert_eq!(arm_r10.parent, "r10");
    assert_eq!(arm_r10.width(), 32);
}

#[test]
fn arm32_alias_for_abi_aliases() {
    assert_eq!(alias_for("r13", 31, 0, Arch::Arm), Some("sp"));
    assert_eq!(alias_for("r14", 31, 0, Arch::Arm), Some("lr"));
    assert_eq!(alias_for("r15", 31, 0, Arch::Arm), Some("pc"));
    assert_eq!(alias_for("r0", 31, 0, Arch::Arm), Some("r0"));
    // Non-full slices have no ARM32 alias.
    assert_eq!(alias_for("r0", 15, 0, Arch::Arm), None);
}

// --- AArch64 SIMD / FPU ---

#[test]
fn aarch64_simd_v_q_d_s_h_b_collapse_to_vn() {
    for alias in ["v0", "q0", "d0", "s0", "h0", "b0"] {
        let layout = register_layout(alias, Arch::Aarch64).unwrap();
        assert_eq!(layout.parent, "v0", "{alias} should collapse to v0");
    }
    for alias in ["v31", "q31", "d31", "s31", "h31", "b31"] {
        let layout = register_layout(alias, Arch::Aarch64).unwrap();
        assert_eq!(layout.parent, "v31", "{alias} should collapse to v31");
    }
}

#[test]
fn aarch64_simd_aliases_have_correct_widths() {
    assert_eq!(register_layout("v0", Arch::Aarch64).unwrap().width(), 128);
    assert_eq!(register_layout("q0", Arch::Aarch64).unwrap().width(), 128);
    assert_eq!(register_layout("d0", Arch::Aarch64).unwrap().width(), 64);
    assert_eq!(register_layout("s0", Arch::Aarch64).unwrap().width(), 32);
    assert_eq!(register_layout("h0", Arch::Aarch64).unwrap().width(), 16);
    assert_eq!(register_layout("b0", Arch::Aarch64).unwrap().width(), 8);
}

#[test]
fn aarch64_simd_slices_start_at_bit_zero() {
    // AArch64 SIMD aliases address the low bits of the 128-bit V
    // parent — there is no `ah`-style high-byte alias.
    for alias in ["d0", "s0", "h0", "b0", "d17", "s23", "h7", "b29"] {
        let layout = register_layout(alias, Arch::Aarch64).unwrap();
        assert_eq!(layout.lo, 0, "{alias} should start at bit 0");
    }
}

#[test]
fn aarch64_simd_rejects_out_of_range() {
    assert!(register_layout("v32", Arch::Aarch64).is_none());
    assert!(register_layout("q40", Arch::Aarch64).is_none());
    assert!(register_layout("d99", Arch::Aarch64).is_none());
}

#[test]
fn aarch64_simd_alias_for_round_trips() {
    assert_eq!(alias_for("v0", 127, 0, Arch::Aarch64), Some("v0"));
    assert_eq!(alias_for("v0", 63, 0, Arch::Aarch64), Some("d0"));
    assert_eq!(alias_for("v0", 31, 0, Arch::Aarch64), Some("s0"));
    assert_eq!(alias_for("v0", 15, 0, Arch::Aarch64), Some("h0"));
    assert_eq!(alias_for("v0", 7, 0, Arch::Aarch64), Some("b0"));
    assert_eq!(alias_for("v17", 127, 0, Arch::Aarch64), Some("v17"));
    assert_eq!(alias_for("v17", 63, 0, Arch::Aarch64), Some("d17"));
    // Slices that do not correspond to a named SIMD alias.
    assert_eq!(alias_for("v0", 95, 64, Arch::Aarch64), None);
    assert_eq!(alias_for("v32", 127, 0, Arch::Aarch64), None);
}

// --- AArch32 SIMD / FPU ---

#[test]
fn arm32_q_d_s_canonicalise_to_vn() {
    // q0 / d0 / d1 / s0..s3 all live in v0 (128-bit synthetic
    // parent). Same parent → slicer sees them as one data-flow
    // node, capturing physical aliasing across views. `vN`
    // itself is NOT a SIMD alias under AArch32 — that namespace
    // is reserved for AAPCS GPRs (see `arm32_aapcs_v_aliases`).
    for alias in ["q0", "d0", "d1", "s0", "s1", "s2", "s3"] {
        let layout = register_layout(alias, Arch::Arm).unwrap();
        assert_eq!(layout.parent, "v0", "{alias} should collapse to v0");
    }
}

#[test]
fn arm32_vn_resolves_to_gpr_not_simd() {
    // Real AArch32 NEON syntax uses qN/dN/sN — never vN. So
    // `register_layout("v1", Arch::Arm)` must return the AAPCS
    // GPR alias (r4), not a 128-bit SIMD layout.
    let v1 = register_layout("v1", Arch::Arm).unwrap();
    assert_eq!(v1.parent, "r4");
    assert_eq!(v1.width(), 32);
    // The internal SIMD parent identifier `v1` is still used by
    // alias_for reverse lookups (see arm32_simd_alias_*), but it
    // never appears as forward-resolved layout output.
}

#[test]
fn arm32_d1_is_upper_half_of_v0() {
    // Q_n = (D_{2n} lower, D_{2n+1} upper). So d1 maps to the
    // upper 64 bits of v0.
    let d1 = register_layout("d1", Arch::Arm).unwrap();
    assert_eq!(d1.parent, "v0");
    assert_eq!(d1.lo, 64);
    assert_eq!(d1.hi, 127);
    assert_eq!(d1.width(), 64);
}

#[test]
fn arm32_s_aliasing_into_quad_register() {
    // S_n is a 32-bit slice of D_{⌊n/2⌋}, which is itself half
    // of V_{⌊n/4⌋}. Spot-check the geometry on s5: parent v1,
    // bits 32..63 (lower half of d2, upper 32-bit slot of v1's
    // lower 64 bits).
    let s5 = register_layout("s5", Arch::Arm).unwrap();
    assert_eq!(s5.parent, "v1");
    assert_eq!(s5.lo, 32);
    assert_eq!(s5.hi, 63);
    assert_eq!(s5.width(), 32);
}

#[test]
fn arm32_q15_d31_s31_are_valid() {
    // Cardinality boundary: AArch32 has 16 Q, 32 D, 32 S regs.
    let q15 = register_layout("q15", Arch::Arm).unwrap();
    assert_eq!(q15.parent, "v15");
    assert_eq!(q15.width(), 128);
    let d31 = register_layout("d31", Arch::Arm).unwrap();
    assert_eq!(d31.parent, "v15");
    assert_eq!(d31.lo, 64);
    assert_eq!(d31.hi, 127);
    // s31 maps to v7 bits 96..127 (s31 = 4·7 + 3).
    let s31 = register_layout("s31", Arch::Arm).unwrap();
    assert_eq!(s31.parent, "v7");
    assert_eq!(s31.lo, 96);
    assert_eq!(s31.hi, 127);
}

#[test]
fn arm32_simd_rejects_out_of_range() {
    assert!(register_layout("q16", Arch::Arm).is_none());
    assert!(register_layout("d32", Arch::Arm).is_none());
    assert!(register_layout("s32", Arch::Arm).is_none());
    // AArch32 has no `bN`/`hN` register naming — those are
    // AArch64-only.
    assert!(register_layout("b0", Arch::Arm).is_none());
    assert!(register_layout("h0", Arch::Arm).is_none());
}

#[test]
fn arm32_simd_alias_for_prefers_qn_over_vn() {
    // qN is the spelling AArch32 disassemblers actually emit, so
    // `alias_for` returns `qN` for the full 128-bit slice. `vN`
    // is the synthetic parent identifier — the reverse lookup
    // should never resurface it.
    assert_eq!(alias_for("v0", 127, 0, Arch::Arm), Some("q0"));
    assert_eq!(alias_for("v15", 127, 0, Arch::Arm), Some("q15"));
}

#[test]
fn arm32_simd_alias_for_recovers_d_and_s() {
    assert_eq!(alias_for("v0", 63, 0, Arch::Arm), Some("d0"));
    assert_eq!(alias_for("v0", 127, 64, Arch::Arm), Some("d1"));
    assert_eq!(alias_for("v0", 31, 0, Arch::Arm), Some("s0"));
    assert_eq!(alias_for("v0", 63, 32, Arch::Arm), Some("s1"));
    assert_eq!(alias_for("v0", 95, 64, Arch::Arm), Some("s2"));
    assert_eq!(alias_for("v0", 127, 96, Arch::Arm), Some("s3"));
    // s* exists only for v0..v7 (s0..s31). v8+ has no s alias.
    assert_eq!(alias_for("v8", 31, 0, Arch::Arm), None);
}

// --- AArch32 AAPCS aliases ---

#[test]
fn arm32_aapcs_a_aliases() {
    // a1..a4 are the AAPCS argument / result registers and
    // alias r0..r3.
    for (alias, expected) in [("a1", "r0"), ("a2", "r1"), ("a3", "r2"), ("a4", "r3")] {
        let layout = register_layout(alias, Arch::Arm).unwrap();
        assert_eq!(layout.parent, expected);
        assert_eq!(layout.width(), 32);
    }
}

#[test]
fn arm32_aapcs_v_aliases() {
    // v1..v8 are the AAPCS callee-saved variable registers and
    // alias r4..r11.
    for (alias, expected) in [
        ("v1", "r4"),
        ("v2", "r5"),
        ("v3", "r6"),
        ("v4", "r7"),
        ("v5", "r8"),
        ("v6", "r9"),
        ("v7", "r10"),
        ("v8", "r11"),
    ] {
        let layout = register_layout(alias, Arch::Arm).unwrap();
        assert_eq!(layout.parent, expected);
        assert_eq!(layout.width(), 32);
    }
}

#[test]
fn arm32_aapcs_named_synonyms() {
    // sb/sl/fp/ip are AAPCS named-register synonyms. sb shares
    // r9 with v6; sl shares r10 with v7; fp shares r11 with v8.
    for (alias, expected) in [("sb", "r9"), ("sl", "r10"), ("fp", "r11"), ("ip", "r12")] {
        let layout = register_layout(alias, Arch::Arm).unwrap();
        assert_eq!(layout.parent, expected);
        assert_eq!(layout.width(), 32);
    }
}

#[test]
fn arm32_aapcs_aliases_do_not_collide_with_x86_or_aarch64() {
    // AAPCS aliases must be Arch::Arm-only. The same strings
    // could resolve under x86 (e.g. `sb` happens to look like a
    // segment register but isn't one) or AArch64 — verify they
    // don't.
    assert!(register_layout("a1", Arch::X86_64).is_none());
    assert!(register_layout("v1", Arch::X86_64).is_none());
    assert!(register_layout("sb", Arch::X86_64).is_none());
    assert!(register_layout("ip", Arch::Aarch64).is_none());
    // x86 `ip`/`eip`/`rip` are the instruction pointer — under
    // AArch32 we deliberately rebind `ip` to r12 (AAPCS scratch).
    assert_eq!(register_layout("ip", Arch::X86_64).unwrap().parent, "rip");
}

#[test]
fn d0_disambiguates_across_aarch64_and_arm() {
    // Same name, different parent widths: AArch64 d0 lives in a
    // 128-bit v0; AArch32 d0 lives in a 128-bit v0 too, but the
    // surrounding register file is half the size (16 Q regs vs
    // 32). The width and parent should agree.
    let a64 = register_layout("d0", Arch::Aarch64).unwrap();
    assert_eq!(a64.parent, "v0");
    assert_eq!(a64.width(), 64);
    let arm = register_layout("d0", Arch::Arm).unwrap();
    assert_eq!(arm.parent, "v0");
    assert_eq!(arm.width(), 64);
}
