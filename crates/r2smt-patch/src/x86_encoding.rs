//! Byte-level rewrites for `setcc` / `cmovcc` on x86 / `x86_64`.
//!
//! All functions in this module take the raw byte sequence of the
//! original instruction (as read via [`r2smt_ir::BytePatcher::read_bytes`])
//! and produce a same-length replacement that preserves any prefix
//! bytes, ModR/M, SIB, and displacement bytes. Same-length output is
//! enforced because the patcher writes back over the original
//! footprint; a longer or shorter instruction would either clobber
//! the next instruction or leave invalid bytes behind.
//!
//! Encoding references:
//!
//! - `SETcc r/m8` — `[REX]? 0F 9x /0 r/m8`
//! - MOV r/m8, imm8 — `[REX]? C6 /0 r/m8 imm8`
//! - `CMOVcc r, r/m` — `[REX]? 0F 4x /r`
//! - MOV r, r/m — `[REX]? 8B /r`
//!
//! Both pairs share their addressing-byte layout (ModR/M + optional
//! SIB + optional displacement), so the rewrites only swap opcode
//! bytes and either append `imm8` (setcc) or a `nop` (cmovcc).

use r2smt_common::{Error, Result};

const NOP: u8 = 0x90;

/// Single-byte NOP opcode used for padding patched instructions.
#[must_use]
pub const fn nop_byte() -> u8 {
    NOP
}

/// Rewrite a `SETcc` instruction so it unconditionally writes
/// `value as u8` to its destination operand.
///
/// `original` is the verbatim instruction byte sequence (typically
/// 3 bytes for `setcc r/m8` or 4 bytes with a REX prefix; longer for
/// memory operands with SIB / displacement). The returned vector is
/// the same length and contains the equivalent `MOV r/m8, imm8`.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the byte sequence does not look like
/// a `SETcc` encoding (no `0F 9x` opcode pair, REG field of ModR/M not
/// `/0`, or the buffer is too short).
pub fn patch_setcc(original: &[u8], value: bool) -> Result<Vec<u8>> {
    if original.len() < 3 {
        return Err(Error::parse(
            "x86_encoding.setcc",
            format!("setcc must be at least 3 bytes, got {}", original.len()),
        ));
    }

    let mut idx = 0usize;
    let mut out = Vec::with_capacity(original.len());

    // Optional REX prefix (0x40-0x4F).
    if (original[idx] & 0xF0) == 0x40 {
        out.push(original[idx]);
        idx += 1;
        if idx + 2 >= original.len() {
            return Err(Error::parse(
                "x86_encoding.setcc",
                "buffer too short after REX prefix",
            ));
        }
    }

    // Expect the SETcc opcode pair `0F 9x`.
    if original[idx] != 0x0F || (original[idx + 1] & 0xF0) != 0x90 {
        return Err(Error::parse(
            "x86_encoding.setcc",
            format!(
                "not a SETcc opcode at offset {idx}: {:02x} {:02x}",
                original[idx],
                original[idx + 1]
            ),
        ));
    }
    idx += 2;

    // ModR/M follows the opcode. SETcc encodes /0, i.e. the REG bits
    // (5-3) of ModR/M must be zero.
    if idx >= original.len() {
        return Err(Error::parse("x86_encoding.setcc", "missing ModR/M byte"));
    }
    let modrm = original[idx];
    if (modrm >> 3) & 0x7 != 0 {
        return Err(Error::parse(
            "x86_encoding.setcc",
            format!("SETcc ModR/M REG field is non-zero: 0x{modrm:02x}"),
        ));
    }

    // Build `[REX]? C6 ModR/M [SIB] [disp] imm8`. MOV r/m8, imm8
    // uses the same /0 ModR/M and the same addressing bytes, so we
    // simply copy everything from the ModR/M onward and append imm8.
    out.push(0xC6);
    out.extend_from_slice(&original[idx..]);
    out.push(u8::from(value));

    if out.len() != original.len() {
        return Err(Error::parse(
            "x86_encoding.setcc",
            format!(
                "size mismatch: original {}, patched {}",
                original.len(),
                out.len()
            ),
        ));
    }
    Ok(out)
}

/// Rewrite a `CMOVcc` instruction as an unconditional `MOV r, r/m`.
///
/// The opcode byte pair `0F 4x` becomes a single-byte `8B`; the
/// remaining ModR/M, SIB, and displacement bytes are preserved
/// verbatim. The freed byte at the end of the instruction is replaced
/// with a `nop` so the total footprint stays identical.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the input does not match a
/// `[REX]? 0F 4x ModR/M ...` layout.
pub fn patch_cmovcc_to_mov(original: &[u8]) -> Result<Vec<u8>> {
    let mut idx = 0usize;
    let mut out = Vec::with_capacity(original.len());

    // Optional 16-bit operand-size override (Intel SDM §2.1.1). The
    // matching `MOV r16, r/m16` form is `66 8B /r`, so we simply copy
    // the prefix through verbatim.
    if original.first() == Some(&0x66) {
        out.push(0x66);
        idx += 1;
    }

    // Optional REX prefix (Intel SDM §2.2.1). Any byte in `0x40..=0x4F`
    // counts. We never reinterpret REX.W/R/X/B here — both `0F 4x /r`
    // (cmovcc) and `8B /r` (mov) honour the same prefix verbatim.
    if let Some(&b) = original.get(idx)
        && (b & 0xF0) == 0x40
    {
        out.push(b);
        idx += 1;
    }

    // 2-byte opcode (`0F 4x`) + at least one ModR/M byte must follow.
    if idx + 2 >= original.len() {
        return Err(Error::parse(
            "x86_encoding.cmovcc",
            format!(
                "cmovcc body must be ≥3 bytes after prefixes at offset {idx}, got {}",
                original.len(),
            ),
        ));
    }

    if original[idx] != 0x0F || (original[idx + 1] & 0xF0) != 0x40 {
        return Err(Error::parse(
            "x86_encoding.cmovcc",
            format!(
                "not a CMOVcc opcode at offset {idx}: {:02x} {:02x}",
                original[idx],
                original[idx + 1]
            ),
        ));
    }
    idx += 2;

    // `8B /r` swallows the same ModR/M-driven addressing bytes — and
    // the same REX/operand-size prefixes — as `0F 4x /r`. After
    // copying ModR/M+SIB+displacement verbatim the encoding is one
    // byte shorter than the original; pad the tail with NOPs so the
    // overall footprint matches.
    out.push(0x8B);
    out.extend_from_slice(&original[idx..]);
    while out.len() < original.len() {
        out.push(NOP);
    }
    debug_assert_eq!(
        out.len(),
        original.len(),
        "cmovcc rewrite produced wrong length",
    );
    Ok(out)
}

/// Return a buffer of `len` NOPs (`0x90`). Used both for fully-NOPed
/// `setcc` / `cmovcc` instructions (always-false outcome) and for
/// NOP-padding shorter replacements.
#[must_use]
pub fn nop_buffer(len: usize) -> Vec<u8> {
    vec![NOP; len]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;

    // ----- setcc tests --------------------------------------------------

    #[test]
    fn setcc_reg_no_rex_value_true() {
        // sete al — 0F 94 C0
        let bytes = [0x0F, 0x94, 0xC0];
        let patched = patch_setcc(&bytes, true).unwrap();
        // mov al, 1 — C6 C0 01
        assert_eq!(patched, vec![0xC6, 0xC0, 0x01]);
    }

    #[test]
    fn setcc_reg_no_rex_value_false() {
        // setne bl — 0F 95 C3
        let bytes = [0x0F, 0x95, 0xC3];
        let patched = patch_setcc(&bytes, false).unwrap();
        // mov bl, 0 — C6 C3 00
        assert_eq!(patched, vec![0xC6, 0xC3, 0x00]);
    }

    #[test]
    fn setcc_reg_with_rex_preserves_prefix() {
        // sete sil — 40 0F 94 C6
        let bytes = [0x40, 0x0F, 0x94, 0xC6];
        let patched = patch_setcc(&bytes, true).unwrap();
        // mov sil, 1 — 40 C6 C6 01
        assert_eq!(patched, vec![0x40, 0xC6, 0xC6, 0x01]);
    }

    #[test]
    fn setcc_memory_operand_preserves_addressing() {
        // sete byte ptr [rbp - 4] — 0F 94 45 FC
        // mod = 01, reg = 000, r/m = 101  → ModR/M = 0x45
        // disp8 = 0xFC (-4)
        let bytes = [0x0F, 0x94, 0x45, 0xFC];
        let patched = patch_setcc(&bytes, true).unwrap();
        // mov byte ptr [rbp - 4], 1 — C6 45 FC 01
        assert_eq!(patched, vec![0xC6, 0x45, 0xFC, 0x01]);
    }

    #[test]
    fn setcc_memory_sib_operand_preserves_layout() {
        // sete byte ptr [rax + rcx*1] — 0F 94 04 08
        // mod=00, reg=000, r/m=100 → ModR/M = 0x04; SIB = 0x08
        let bytes = [0x0F, 0x94, 0x04, 0x08];
        let patched = patch_setcc(&bytes, false).unwrap();
        // mov byte ptr [rax+rcx], 0 — C6 04 08 00
        assert_eq!(patched, vec![0xC6, 0x04, 0x08, 0x00]);
    }

    #[test]
    fn setcc_size_is_preserved() {
        let bytes = [0x0F, 0x94, 0xC0];
        let patched = patch_setcc(&bytes, true).unwrap();
        assert_eq!(patched.len(), bytes.len());
    }

    #[test]
    fn setcc_rejects_too_short() {
        let bytes = [0x0F, 0x94];
        assert!(patch_setcc(&bytes, true).is_err());
    }

    #[test]
    fn setcc_rejects_wrong_opcode() {
        // 0F 80 = JO (near jump) — not SETcc.
        let bytes = [0x0F, 0x80, 0x00];
        assert!(patch_setcc(&bytes, true).is_err());
    }

    #[test]
    fn setcc_rejects_non_zero_reg_field() {
        // ModR/M with reg = 010 (non-zero) — invalid SETcc encoding.
        // 0F 94 D0 — reg field = 010
        let bytes = [0x0F, 0x94, 0xD0];
        assert!(patch_setcc(&bytes, true).is_err());
    }

    // ----- cmovcc tests -------------------------------------------------

    #[test]
    fn cmovcc_reg_no_rex_to_mov() {
        // cmove eax, ebx — 0F 44 C3
        let bytes = [0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        // mov eax, ebx — 8B C3 + NOP padding
        assert_eq!(patched, vec![0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_reg_with_rex_w_preserves_prefix() {
        // cmove rax, rbx — 48 0F 44 C3
        let bytes = [0x48, 0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        // mov rax, rbx — 48 8B C3 + NOP
        assert_eq!(patched, vec![0x48, 0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_memory_with_disp8() {
        // cmove eax, dword ptr [rbp - 4] — 0F 44 45 FC
        let bytes = [0x0F, 0x44, 0x45, 0xFC];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        // mov eax, dword ptr [rbp - 4] — 8B 45 FC + NOP
        assert_eq!(patched, vec![0x8B, 0x45, 0xFC, NOP]);
    }

    #[test]
    fn cmovcc_size_is_preserved() {
        let bytes = [0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched.len(), bytes.len());
        let _ = patched.iter().last().unwrap();
    }

    #[test]
    fn cmovcc_rejects_wrong_opcode() {
        // 0F 84 = JZ near — not CMOVcc.
        let bytes = [0x0F, 0x84, 0x00, 0x00, 0x00, 0x00];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    #[test]
    fn cmovcc_rejects_too_short() {
        let bytes = [0x0F, 0x44];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    #[test]
    fn cmovcc_disp8_memory_operand() {
        // cmove rax, [rbx + 0x10] — 48 0F 44 43 10
        //   ModR/M = 0x43 (mod=01, reg=000, r/m=011 → [rbx + disp8])
        let bytes = [0x48, 0x0F, 0x44, 0x43, 0x10];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        // mov rax, [rbx + 0x10] — 48 8B 43 10 + NOP
        assert_eq!(patched, vec![0x48, 0x8B, 0x43, 0x10, NOP]);
    }

    #[test]
    fn cmovcc_sib_disp32_memory_operand() {
        // cmove rax, [rbx + rcx*4 + 0x12345678]
        //   REX.W=1, opcode=0F 44, ModR/M=0x84 (mod=10, reg=000,
        //   r/m=100 → SIB-with-disp32), SIB=0x8B (scale=10, idx=001,
        //   base=011), disp32 little-endian
        let bytes = [0x48, 0x0F, 0x44, 0x84, 0x8B, 0x78, 0x56, 0x34, 0x12];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(
            patched,
            vec![0x48, 0x8B, 0x84, 0x8B, 0x78, 0x56, 0x34, 0x12, NOP],
        );
    }

    #[test]
    fn cmovcc_rex_b_extended_source() {
        // cmove rax, r11 — 49 0F 44 C3 (REX.B picks r11 as r/m)
        let bytes = [0x49, 0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x49, 0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_rex_r_extended_dest() {
        // cmove r8, rbx — 4C 0F 44 C3 (REX.R picks r8 as reg)
        let bytes = [0x4C, 0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x4C, 0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_rex_rb_both_extended() {
        // cmove r8, r11 — 4D 0F 44 C3 (REX.R + REX.B)
        let bytes = [0x4D, 0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x4D, 0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_operand_size_16bit() {
        // cmove ax, bx — 66 0F 44 C3. The 16-bit operand-size
        // override prefix `66` must be preserved verbatim; the
        // matching `MOV r16, r/m16` form is `66 8B /r`.
        let bytes = [0x66, 0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x66, 0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_32bit_no_rex() {
        // cmove eax, ebx — 0F 44 C3
        let bytes = [0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_64bit_with_rex_w() {
        // cmove rax, rbx — 48 0F 44 C3 (REX.W = 1)
        let bytes = [0x48, 0x0F, 0x44, 0xC3];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x48, 0x8B, 0xC3, NOP]);
    }

    #[test]
    fn cmovcc_condition_code_variety() {
        // Loop over every legal condition code (`0F 40` .. `0F 4F`).
        // All of them must rewrite identically to `8B C3 NOP`.
        for cond in 0x40u8..=0x4Fu8 {
            let bytes = [0x0F, cond, 0xC3];
            let patched = patch_cmovcc_to_mov(&bytes)
                .unwrap_or_else(|e| panic!("cond byte {cond:#x} failed: {e}"));
            assert_eq!(patched, vec![0x8B, 0xC3, NOP], "cond byte {cond:#x}");
        }
    }

    #[test]
    fn cmovcc_rip_relative() {
        // cmove rax, [rip + 0x12345678] — 48 0F 44 05 78 56 34 12
        //   ModR/M=0x05 (mod=00, reg=000, r/m=101 → RIP-relative
        //   in 64-bit mode), disp32 little-endian.
        let bytes = [0x48, 0x0F, 0x44, 0x05, 0x78, 0x56, 0x34, 0x12];
        let patched = patch_cmovcc_to_mov(&bytes).unwrap();
        assert_eq!(patched, vec![0x48, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12, NOP],);
    }

    #[test]
    fn cmovcc_too_short_no_prefix() {
        // Single byte cannot contain even the opcode pair.
        let bytes = [0x0F];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    #[test]
    fn cmovcc_too_short_after_rex() {
        // REX + 1 opcode byte — missing the condition code byte.
        let bytes = [0x48, 0x0F];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    #[test]
    fn cmovcc_wrong_first_byte() {
        // 0x90 (NOP) followed by garbage — must fail.
        let bytes = [0x90, 0x44, 0xC3];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    #[test]
    fn cmovcc_wrong_second_nibble() {
        // `0F 90 C3` is `seto`, not a cmovcc. Must reject.
        let bytes = [0x0F, 0x90, 0xC3];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    #[test]
    fn cmovcc_invalid_rex_followed_by_garbage() {
        // 48 (REX.W) AA BB CC — second byte is not 0F so opcode
        // detection must reject.
        let bytes = [0x48, 0xAA, 0xBB, 0xCC];
        assert!(patch_cmovcc_to_mov(&bytes).is_err());
    }

    // ----- nop_buffer ---------------------------------------------------

    #[test]
    fn nop_buffer_emits_requested_length() {
        let buf = nop_buffer(7);
        assert_eq!(buf, vec![NOP; 7]);
    }
}
