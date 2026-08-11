//! Byte-level rewrites for ARM conditional branches.
//!
//! Both `AArch64` and `AArch32` (ARM mode) are fixed 4-byte
//! instruction encodings, so the v0 rewrite strategy mirrors x86
//! `nop_jcc` / `replace_jcc_with_jmp`:
//!
//! - **Always-false branch** → replace the 4-byte conditional
//!   instruction with a 4-byte architectural NOP.
//! - **Always-true branch** → assemble an unconditional `b <target>`
//!   over the same 4 bytes.
//!
//! Mnemonic coverage (planner-side, classified by `plan::classify_mnemonic`):
//!
//! - `AArch64`: `b.<cond>` and the compare-and-branch family
//!   `cbz` / `cbnz` / `tbz` / `tbnz`.
//! - `AArch32` (ARM mode): `b<cond>` for the standard condition
//!   suffixes (`eq`/`ne`/`cs`/`hs`/`cc`/`lo`/`mi`/`pl`/`vs`/`vc`/
//!   `hi`/`ls`/`ge`/`lt`/`gt`/`le`). Unconditional `b`, link forms
//!   `bl`/`blx`, and indirect `bx` are excluded.
//!
//! Encoding references:
//!
//! - `AArch64` NOP: `D503201F` (ARM ARM Vol. C §C6.2.182). Little-
//!   endian byte layout: `1F 20 03 D5`.
//! - `AArch32` NOP (ARMv6T2+): `E320F000` (ARM ARM Vol. C §A8.8.119).
//!   Little-endian byte layout: `00 F0 20 E3`.
//!
//! Both NOP forms are architectural hint instructions, not the
//! historical `mov rN, rN` idiom — they are explicitly recognised
//! as NOPs by the CPU's instruction decoder and have zero side
//! effects on flags or registers.
//!
//! Thumb-mode `AArch32` (2-byte / 4-byte mixed encoding) is **out of
//! scope** for this rewrite; callers must reject Thumb mnemonics
//! upstream. The slicer / planner currently classifies `b<cond>`
//! purely on the textual mnemonic and does not yet attempt to detect
//! Thumb vs ARM mode, so any caller invoking the planner against a
//! Thumb function should expect the resulting plan to be byte-
//! incorrect — fixing that is a follow-up gated on Thumb mode
//! detection in the slicer / r2pipe adapter.

use r2smt_common::{Arch, Error, Result};

/// Length, in bytes, of any `AArch64` or `AArch32` (ARM-mode)
/// instruction.
pub const ARM_INSTRUCTION_BYTES: usize = 4;

/// Canonical NOP encoding for `AArch64` (`D503201F`, little-endian).
const AARCH64_NOP_LE: [u8; 4] = [0x1F, 0x20, 0x03, 0xD5];

/// Canonical NOP encoding for `AArch32` (`E320F000`, little-endian).
const AARCH32_NOP_LE: [u8; 4] = [0x00, 0xF0, 0x20, 0xE3];

/// Length, in bytes, of a Thumb 16-bit instruction half-word.
pub const THUMB_HALFWORD_BYTES: usize = 2;

/// Thumb NOP encoding (`BF00`, little-endian). `ARMv6T2` introduced this
/// as a proper hint instruction; older Thumb encodings fell back to
/// `MOV r8, r8` which still functions as a NOP but is harder to
/// recognise.
pub const THUMB_NOP_LE: [u8; 2] = [0x00, 0xBF];

/// Return the architectural NOP encoding for `arch`.
///
/// # Errors
///
/// Returns [`Error::Parse`] if `arch` is not an ARM ISA. Callers
/// should not invoke this for x86 — the x86 NOP is a single
/// `0x90` byte and lives in [`crate::x86_encoding`].
pub fn arm_nop_bytes(arch: Arch) -> Result<[u8; 4]> {
    match arch {
        Arch::Aarch64 => Ok(AARCH64_NOP_LE),
        Arch::Arm => Ok(AARCH32_NOP_LE),
        other => Err(Error::parse(
            "arm_encoding.nop",
            format!("{other:?} is not an ARM ISA"),
        )),
    }
}

/// Fill `len` bytes with the architectural NOP encoding for `arch`.
///
/// `len` must be a multiple of [`ARM_INSTRUCTION_BYTES`]; otherwise
/// the resulting tail bytes would be a partial instruction and the
/// CPU would fault on execution.
///
/// # Errors
///
/// Returns [`Error::Parse`] if `arch` is not an ARM ISA or `len` is
/// not a multiple of 4.
pub fn arm_nop_buffer(arch: Arch, len: usize) -> Result<Vec<u8>> {
    if len % ARM_INSTRUCTION_BYTES != 0 {
        return Err(Error::parse(
            "arm_encoding.nop_buffer",
            format!("{len} is not a multiple of {ARM_INSTRUCTION_BYTES}"),
        ));
    }
    let nop = arm_nop_bytes(arch)?;
    let count = len / ARM_INSTRUCTION_BYTES;
    let mut out = Vec::with_capacity(len);
    for _ in 0..count {
        out.extend_from_slice(&nop);
    }
    Ok(out)
}

/// Fill `len` bytes with the Thumb 16-bit NOP hint.
///
/// `len` must be a multiple of [`THUMB_HALFWORD_BYTES`] so the
/// resulting buffer ends on an instruction boundary. Used by the
/// patcher to NOP-out Thumb conditional branches whose original
/// footprint is 2 or 4 bytes (a single Thumb half-word or a Thumb-2
/// 32-bit branch).
///
/// # Errors
///
/// Returns [`Error::Parse`] if `len` is not a multiple of 2.
pub fn thumb_nop_buffer(len: usize) -> Result<Vec<u8>> {
    if len % THUMB_HALFWORD_BYTES != 0 {
        return Err(Error::parse(
            "arm_encoding.thumb_nop_buffer",
            format!("{len} is not a multiple of {THUMB_HALFWORD_BYTES}"),
        ));
    }
    let count = len / THUMB_HALFWORD_BYTES;
    let mut out = Vec::with_capacity(len);
    for _ in 0..count {
        out.extend_from_slice(&THUMB_NOP_LE);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn aarch64_nop_is_canonical_hint_encoding() {
        let nop = arm_nop_bytes(Arch::Aarch64).unwrap();
        // D503201F in little-endian byte order.
        assert_eq!(nop, [0x1F, 0x20, 0x03, 0xD5]);
        // Decoded back to a u32 big-endian: 0xD503201F.
        let word = u32::from_le_bytes(nop);
        assert_eq!(word, 0xD503_201F);
    }

    #[test]
    fn aarch32_nop_is_canonical_hint_encoding() {
        let nop = arm_nop_bytes(Arch::Arm).unwrap();
        // E320F000 in little-endian byte order.
        assert_eq!(nop, [0x00, 0xF0, 0x20, 0xE3]);
        let word = u32::from_le_bytes(nop);
        assert_eq!(word, 0xE320_F000);
    }

    #[test]
    fn arm_nop_bytes_rejects_non_arm_arch() {
        assert!(arm_nop_bytes(Arch::X86_64).is_err());
        assert!(arm_nop_bytes(Arch::X86).is_err());
    }

    #[test]
    fn arm_nop_buffer_tiles_for_aarch64() {
        let buf = arm_nop_buffer(Arch::Aarch64, 8).unwrap();
        assert_eq!(buf.len(), 8);
        assert_eq!(&buf[..4], &AARCH64_NOP_LE);
        assert_eq!(&buf[4..], &AARCH64_NOP_LE);
    }

    #[test]
    fn arm_nop_buffer_handles_single_instruction() {
        let buf = arm_nop_buffer(Arch::Arm, 4).unwrap();
        assert_eq!(buf, AARCH32_NOP_LE);
    }

    #[test]
    fn arm_nop_buffer_rejects_misaligned_length() {
        // 6 is not a multiple of 4 — partial instruction would crash
        // the CPU at execution time.
        assert!(arm_nop_buffer(Arch::Aarch64, 6).is_err());
        assert!(arm_nop_buffer(Arch::Arm, 5).is_err());
    }

    #[test]
    fn arm_nop_buffer_zero_length_yields_empty() {
        let buf = arm_nop_buffer(Arch::Aarch64, 0).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn thumb_nop_2byte_matches_bf00() {
        let buf = thumb_nop_buffer(2).unwrap();
        assert_eq!(buf, vec![0x00, 0xBF]);
    }

    #[test]
    fn thumb_nop_4byte_tiles_two_halfwords() {
        let buf = thumb_nop_buffer(4).unwrap();
        assert_eq!(buf, vec![0x00, 0xBF, 0x00, 0xBF]);
    }

    #[test]
    fn thumb_nop_buffer_rejects_odd_length() {
        assert!(thumb_nop_buffer(3).is_err());
        assert!(thumb_nop_buffer(5).is_err());
    }

    #[test]
    fn thumb_nop_buffer_zero_length_yields_empty() {
        assert!(thumb_nop_buffer(0).unwrap().is_empty());
    }
}
