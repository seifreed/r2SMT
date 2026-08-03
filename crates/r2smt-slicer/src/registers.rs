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
//!   alias on one data-flow node. MMX (`mm0`…) stays `None`, gated on
//!   its own lifter. The x87 stack collapses onto the single synthetic
//!   parent `st`: TOP rotates under every push and pop, so `st(1)`
//!   names a different physical register after each `fld` and an
//!   index-keyed node would alias two of them. Slot-level precision
//!   lives in the lifter's slice-scoped stack instead
//!   (`lift/x87.rs`). The bare `st0`…`st7` spelling stays `None` — it
//!   is what radare2 emits in ESIL, never in disassembly.
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
/// Returns `None` for registers outside the modelled files (MMX `mm0`,
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

/// Full architectural width of a vector register parent under `arch`.
///
/// Every SIMD view is a slice of one synthetic parent held at this
/// width, so a 128-bit and a 256-bit view of the same register share a
/// data-flow node instead of colliding as two different-width variables
/// of the same name. x86 carries `zmm<n>` at AVX-512 width; both ARM
/// ISAs carry `v<n>` at the 128-bit architectural maximum.
#[must_use]
pub const fn simd_parent_bits(arch: Arch) -> Option<u16> {
    match arch {
        Arch::X86 | Arch::X86_64 => Some(512),
        Arch::Aarch64 | Arch::Arm => Some(128),
        _ => None,
    }
}

/// Whether `parent` names a vector register parent under `arch`.
///
/// Checked against the *resolved* parent rather than the operand text,
/// which matters on `AArch32`: `v1` as an operand is an AAPCS alias for
/// the general-purpose `r4`, and [`register_layout`] resolves it there,
/// so a parent that still spells `v1` came from the SIMD table.
#[must_use]
pub fn is_simd_parent(parent: &str, arch: Arch) -> bool {
    let prefix = match arch {
        Arch::X86 | Arch::X86_64 => "zmm",
        Arch::Aarch64 | Arch::Arm => "v",
        _ => return false,
    };
    parent
        .strip_prefix(prefix)
        .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// Whether an operand names an ARM vector register in a shape the
/// single-parent register model cannot express on its own: an element
/// arrangement (`v0.4s`), an indexed lane (`v0.s[1]`, `d0[1]`), or a
/// multi-register list (`{v0.4s, v1.4s}`).
///
/// Purely syntactic, and deliberately so: it answers "does this operand
/// carry vector shape beyond the bare register name", not "can the
/// lifter model it". [`register_layout`] now resolves the arranged and
/// indexed spellings to their parent slice, so this detector — which
/// inspects only the leading register token — is what keeps the effect
/// tables and the lifter dispatchers routing vector-shaped operands
/// through the vector seams instead of letting a scalar handler treat
/// `v0.4s` as one wide register.
///
/// A register list is judged by what it lists, not by its braces:
/// `AArch32` spells its GPR push / pop / ldm lists the same way
/// (`{r4, r5, lr}`), and those are ordinary data movement the lifter
/// models. Only a list of vector registers carries vector shape.
///
/// Always `false` outside the ARM ISAs: no x86 register spelling carries
/// an arrangement, and the `AArch32` AAPCS aliases (`v1` for `r4`) are
/// bare names that never reach the remainder test.
#[must_use]
pub fn has_vector_arrangement(raw: &str, arch: Arch) -> bool {
    if !matches!(arch, Arch::Aarch64 | Arch::Arm) {
        return false;
    }
    let lower = raw.trim().to_ascii_lowercase();
    if let Some(body) = lower.strip_prefix('{') {
        let first = body.split([',', '}']).next().unwrap_or_default().trim();
        // A structured single-element list spells its lane index once,
        // *outside* the brace (`{v0.s}[1]`), which leaves each member an
        // incomplete spelling the register table does not resolve on its
        // own. Its head still names a vector register, and carrying a
        // suffix at all is what makes the list vector-shaped.
        return names_vector_register(first, arch) || carries_vector_shape(first, arch);
    }
    carries_vector_shape(&lower, arch)
}

/// Whether an operand spelling is a vector register name followed by
/// shape of some kind — an arrangement, a lane index, or the bare
/// element type a structured list's member carries.
fn carries_vector_shape(spelling: &str, arch: Arch) -> bool {
    let head_len = spelling
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(spelling.len());
    let (head, rest) = spelling.split_at(head_len);
    !rest.is_empty() && names_vector_register(head, arch)
}

/// Whether `name` resolves to a slice of a vector register parent under
/// `arch`.
fn names_vector_register(name: &str, arch: Arch) -> bool {
    register_layout(name, arch).is_some_and(|layout| is_simd_parent(layout.parent, arch))
}

/// Element arrangement of an ARM vector operand: the `4s` in `v0.4s`.
///
/// Kept outside [`RegisterLayout`] deliberately — the layout describes
/// *which bits* of the parent a name addresses, while the arrangement
/// describes *how the lifter must cut* those bits into lanes. Folding
/// lane geometry into the layout would also break [`alias_for`], whose
/// reverse lookup assumes plain `(hi, lo)` slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Arrangement {
    /// Number of elements in the view.
    pub lanes: u16,
    /// Width of one element in bits.
    pub lane_bits: u16,
}

impl Arrangement {
    /// Total width of the arranged view in bits (`lanes * lane_bits`).
    #[must_use]
    pub const fn view_bits(&self) -> u16 {
        self.lanes * self.lane_bits
    }
}

/// Width in bits of an ARM vector element-type letter.
///
/// `q` is a genuine element type and not only a register spelling:
/// `pmull v0.1q, v1.1d, v2.1d` names a one-lane, 128-bit arrangement,
/// which is the only shape a 64-by-64 carry-less product can be written
/// in. It admits exactly one lane — `2q` would span 256 bits, which
/// [`parse_arrangement`] rejects on total width.
pub(crate) const fn element_type_bits(letter: char) -> Option<u16> {
    match letter {
        'b' => Some(8),
        'h' => Some(16),
        's' => Some(32),
        'd' => Some(64),
        'q' => Some(128),
        _ => None,
    }
}

/// Parse the `<lanes><type>` body of an `AArch64` element arrangement
/// (`4s`, `16b`, `2d`, `1d` — ARM ARM C1.2.4, "Vector formats").
///
/// Valid arrangements cover exactly half or all of the 128-bit vector
/// register, so any `<lanes><type>` whose product is neither 64 nor 128
/// is rejected rather than guessed at.
#[must_use]
pub fn parse_arrangement(body: &str) -> Option<Arrangement> {
    let split = body.find(|c: char| !c.is_ascii_digit())?;
    let (digits, letter) = body.split_at(split);
    let lanes: u16 = digits.parse().ok()?;
    let mut letters = letter.chars();
    let lane_bits = element_type_bits(letters.next()?)?;
    if letters.next().is_some() {
        return None;
    }
    let arrangement = Arrangement { lanes, lane_bits };
    matches!(arrangement.view_bits(), 64 | 128).then_some(arrangement)
}

/// Parse the `[N]` lane index of an indexed vector operand (`v0.s[1]`,
/// `d0[1]`).
///
/// The bound is architectural, not per-type: 128-bit vector registers
/// with a minimum 8-bit element admit indices `0..=15`, so anything
/// larger is a malformed spelling regardless of the element width the
/// caller knows about. Callers with a concrete element type tighten the
/// check further.
#[must_use]
pub fn parse_lane_index(body: &str) -> Option<u16> {
    const MAX_VECTOR_LANE_INDEX: u16 = 15;
    let digits = body.strip_prefix('[')?.strip_suffix(']')?;
    let index: u16 = digits.parse().ok()?;
    (index <= MAX_VECTOR_LANE_INDEX).then_some(index)
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
