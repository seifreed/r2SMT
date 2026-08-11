//! Operand geometry for the `AArch64` NEON resolvers.
//!
//! What an operand *names*, before any question of what the instruction
//! does with it: the arrangement suffix a vector register carries, the
//! element an indexed one addresses, and whether a name belongs to the
//! vector file at all. Every family resolver starts here, which is why
//! these primitives sit in one place rather than beside whichever
//! family first needed them.
//!
//! The `2`-suffix pair lives here too. [`peel_upper`] reads the half a
//! mnemonic asks for and [`spans_full_register`] checks that the
//! operand it applies to has two halves to choose between; they are one
//! question asked on the two sides of the instruction.

use r2smt_common::Arch;
use r2smt_ir::program::{Operand, OperandKind};

use crate::registers::{
    Arrangement, element_type_bits, is_simd_parent, parse_arrangement, parse_lane_index,
    register_layout,
};

/// Number of bits in a byte, used wherever an operand counts bytes
/// (`ext`'s index) but the model counts bits.
pub(super) const BITS_PER_BYTE: u16 = 8;

/// Whether a `2`-form's full-register operand really spans 128 bits.
///
/// The `2` suffix means "the half of the 128-bit register the base form
/// does not touch", so the operand it halves must be a full-width
/// arrangement. Without this an arrangement half that size would resolve
/// to a shape the architecture does not encode.
pub(super) const fn spans_full_register(arrangement: Arrangement) -> bool {
    arrangement.view_bits() == SIMD_REGISTER_BITS
}

/// Width of an `AArch64` vector register.
const SIMD_REGISTER_BITS: u16 = 128;

/// Peel the `2` suffix that marks the half-register forms.
pub(super) fn peel_upper(mnemonic: &str) -> (&str, bool) {
    mnemonic
        .strip_suffix('2')
        .map_or((mnemonic, false), |base| (base, true))
}

/// The arrangement an operand carries, if it is a vector register that
/// carries one.
pub(super) fn operand_arrangement(op: &Operand) -> Option<Arrangement> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if !names_vector_register(&raw) {
        return None;
    }
    let (_, body) = raw.split_once('.')?;
    parse_arrangement(body)
}

/// The `(element_bits, index)` an indexed operand names: `v0.s[1]` is a
/// 32-bit element at index 1.
pub(super) fn indexed_element(op: &Operand) -> Option<(u16, u16)> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if !names_vector_register(&raw) {
        return None;
    }
    let (_, body) = raw.split_once('.')?;
    let mut chars = body.chars();
    let element_bits = element_type_bits(chars.next()?)?;
    let index = parse_lane_index(chars.as_str())?;
    // The element has to fit inside the 128-bit register.
    let lanes = crate::registers::simd_parent_bits(Arch::Aarch64)? / element_bits;
    (index < lanes).then_some((element_bits, index))
}

/// The `(register, group_index)` a `sdot` / `udot` by-element source
/// names: `v2.4b[1]` selects the four-byte group at index 1 of `v2`.
///
/// This is the one NEON operand that carries an element shape *and* an
/// index at once, which both [`operand_arrangement`] and
/// [`indexed_element`] reject, so it has its own parser. The register is
/// re-spelled bare so the SIMD readers, which resolve names through the
/// register table, can materialise it.
pub(super) fn dot_product_element(op: &Operand) -> Option<(Operand, u16)> {
    // Four four-byte groups in the 128-bit register.
    const DOT_PRODUCT_GROUPS: u16 = SIMD_REGISTER_BITS / 32;
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    let (register, body) = raw.split_once('.')?;
    if !names_vector_register(register) {
        return None;
    }
    let index = parse_lane_index(body.strip_prefix("4b")?)?;
    (index < DOT_PRODUCT_GROUPS).then_some((
        Operand {
            raw: register.to_string(),
            kind: OperandKind::Register,
        },
        index,
    ))
}

/// Whether an operand names a general-purpose register.
pub(super) fn is_general_register(op: &Operand) -> bool {
    op.kind == OperandKind::Register
        && register_layout(&op.raw, Arch::Aarch64)
            .is_some_and(|layout| !is_simd_parent(layout.parent, Arch::Aarch64))
}

/// Whether the operand text names a vector register (before any shape
/// suffix is considered).
fn names_vector_register(raw: &str) -> bool {
    register_layout(raw, Arch::Aarch64)
        .is_some_and(|layout| is_simd_parent(layout.parent, Arch::Aarch64))
}
