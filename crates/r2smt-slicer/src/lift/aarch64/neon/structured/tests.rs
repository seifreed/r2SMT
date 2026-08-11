//! Resolution contracts for the `AArch64` structured load / store
//! family. What the lowering *computes* is checked by solving, in
//! `r2smt-core/tests/neon_structured_contracts.rs`; these fix which
//! operand shapes are admitted at all.

use super::{ListElement, StructuredEffect, resolve};
use r2smt_common::Address;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

fn operand(raw: &str) -> Operand {
    let kind = if raw.starts_with('{') {
        OperandKind::Unknown
    } else if raw.starts_with('[') {
        OperandKind::Memory
    } else if raw.starts_with(|c: char| c.is_ascii_digit()) || raw.starts_with('#') {
        OperandKind::Immediate
    } else {
        OperandKind::Register
    };
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn insn(mnemonic: &str, operands: &[&str]) -> Instruction {
    Instruction {
        address: Address::new(0x1000),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands: operands.iter().map(|raw| operand(raw)).collect(),
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

fn effect_of(mnemonic: &str, operands: &[&str]) -> Option<StructuredEffect> {
    resolve(&insn(mnemonic, operands)).map(|access| access.effect())
}

#[test]
fn test_resolve_contiguous_load_writes_the_listed_registers() {
    assert_eq!(
        effect_of("ld1", &["{v0.16b}", "[x0]"]),
        Some(StructuredEffect {
            reads_list: false,
            writes_list: true,
            writes_base: false,
        })
    );
}

#[test]
fn test_resolve_contiguous_store_reads_the_listed_registers() {
    assert_eq!(
        effect_of("st1", &["{v0.4s, v1.4s}", "[x0]"]),
        Some(StructuredEffect {
            reads_list: true,
            writes_list: false,
            writes_base: false,
        })
    );
}

#[test]
fn test_resolve_single_element_load_also_reads_its_destination() {
    // The load replaces one lane and preserves the rest, so the slicer
    // has to keep whatever defined the register before it.
    assert_eq!(
        effect_of("ld1", &["{v0.s}[1]", "[x8]"]),
        Some(StructuredEffect {
            reads_list: true,
            writes_list: true,
            writes_base: false,
        })
    );
}

#[test]
fn test_resolve_post_index_marks_the_base_written() {
    assert_eq!(
        effect_of("ld1", &["{v0.16b}", "[x0]", "16"]).map(|effect| effect.writes_base),
        Some(true)
    );
}

#[test]
fn test_resolve_register_post_index_marks_the_base_written() {
    assert_eq!(
        effect_of("ld1", &["{v0.16b}", "[x0]", "x3"]).map(|effect| effect.writes_base),
        Some(true)
    );
}

#[test]
fn test_resolve_accepts_a_list_wrapping_past_the_last_register() {
    // ARM ARM C7.2: the listed registers are consecutive modulo 32.
    assert!(resolve(&insn("ld1", &["{v31.2d, v0.2d}", "[x0]"])).is_some());
}

#[test]
fn test_resolve_rejects_a_non_consecutive_list() {
    assert!(resolve(&insn("ld1", &["{v0.4s, v2.4s}", "[x0]"])).is_none());
}

#[test]
fn test_resolve_rejects_mixed_arrangements_in_one_list() {
    assert!(resolve(&insn("ld1", &["{v0.4s, v1.8b}", "[x0]"])).is_none());
}

#[test]
fn test_resolve_rejects_a_deinterleaving_list_of_the_wrong_length() {
    // `ld2` interleaves exactly two structures and names exactly two
    // registers; one register is a shape the architecture has no
    // encoding for.
    assert!(resolve(&insn("ld2", &["{v0.4s}", "[x0]"])).is_none());
}

#[test]
fn test_resolve_accepts_a_four_register_contiguous_load() {
    // `ld1` is the exception: it moves one to four whole registers.
    assert!(resolve(&insn("ld1", &["{v0.16b, v1.16b, v2.16b, v3.16b}", "[x0]"])).is_some());
}

#[test]
fn test_resolve_rejects_a_single_lane_arrangement_for_deinterleaving() {
    // `1d` gives one element per register, which LD3 cannot
    // de-interleave; the architecture does not encode it.
    assert!(resolve(&insn("ld3", &["{v0.1d, v1.1d, v2.1d}", "[x0]"])).is_none());
}

#[test]
fn test_resolve_rejects_a_replicating_store() {
    // A replicate reads one element and broadcasts it; there is no
    // store dual, and `st1r` is not a mnemonic.
    assert!(resolve(&insn("st1r", &["{v0.4s}", "[x0]"])).is_none());
}

#[test]
fn test_resolve_rejects_a_replicating_single_element_list() {
    assert!(resolve(&insn("ld1r", &["{v0.s}[1]", "[x0]"])).is_none());
}

#[test]
fn test_resolve_rejects_an_out_of_range_lane_index() {
    // A doubleword register holds two 64-bit elements.
    assert!(resolve(&insn("ld1", &["{v0.d}[2]", "[x0]"])).is_none());
}

#[test]
fn test_resolve_rejects_the_scalar_load_family() {
    for mnemonic in ["ldr", "ldrb", "ldrsw", "ldp", "str", "stp", "ld", "st"] {
        assert!(
            resolve(&insn(mnemonic, &["x0", "[x1]"])).is_none(),
            "{mnemonic} is not a structured access"
        );
    }
}

#[test]
fn test_resolve_rejects_a_non_memory_second_operand() {
    assert!(resolve(&insn("ld1", &["{v0.16b}", "x0"])).is_none());
}

#[test]
fn test_resolve_reads_the_replicating_load_arrangement() {
    assert_eq!(
        resolve(&insn("ld2r", &["{v0.4s, v1.4s}", "[x0]"])).map(|access| access.element),
        Some(ListElement::Whole(crate::registers::Arrangement {
            lanes: 4,
            lane_bits: 32,
        }))
    );
}
