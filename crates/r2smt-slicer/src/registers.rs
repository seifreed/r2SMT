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
//! - **`x86_64`**: SIMD vector registers (`xmm<n>` / `ymm<n>` /
//!   `zmm<n>`, n=0..31) resolve to bits `[127:0]` / `[255:0]` /
//!   `[511:0]` of a synthetic `zmm<n>` parent, so the three views
//!   alias on one data-flow node. The x87 stack (`st0`…`st7`) and MMX
//!   (`mm0`…) stay `None`, gated on their own lifters.
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
    pub lo: u16,
    /// Inclusive high bit offset within the parent (7 for `al`, 15 for
    /// `ah`, 31 for `eax`, 63 for `rax`, 511 for `zmm`).
    pub hi: u16,
    /// `true` if this alias is the 32-bit doubleword form whose write
    /// zero-extends into a 64-bit parent on `x86_64` / `AArch64`.
    /// False for 64/16/8-bit aliases.
    pub zero_extends_parent_64: bool,
}

impl RegisterLayout {
    /// Width of the register slice in bits (`hi - lo + 1`).
    #[must_use]
    pub const fn width(&self) -> u16 {
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
pub fn alias_for(parent: &str, hi: u16, lo: u16, arch: Arch) -> Option<&'static str> {
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

const fn aarch64_vector(parent: &'static str, lo: u16, hi: u16) -> RegisterLayout {
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

/// SIMD register slice, ISA-neutral. `zero_extends_parent_64` is a
/// GPR concept (32→64 dword zero-extension) and does not describe SIMD
/// write semantics, so it stays `false`; the lifter models the
/// zero-extension-to-vector-width behaviour explicitly.
const fn simd_slice(parent: &'static str, lo: u16, hi: u16) -> RegisterLayout {
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

const fn arm32_vector(parent: &'static str, lo: u16, hi: u16) -> RegisterLayout {
    RegisterLayout {
        parent,
        lo,
        hi,
        zero_extends_parent_64: false,
    }
}

mod aarch32;
mod aarch64;
mod x86;

use aarch32::{arm32_alias, arm32_layout};
use aarch64::{aarch64_alias, aarch64_layout};
use x86::{x86_alias, x86_layout};

#[cfg(test)]
mod tests;
