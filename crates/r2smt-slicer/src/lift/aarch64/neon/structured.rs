//! `AArch64` NEON structured loads and stores.
//!
//! The family radare2 spells `ld1`…`ld4` / `st1`…`st4` / `ld1r`…`ld4r`:
//! a brace-delimited list of consecutive vector registers on one side
//! and a memory operand on the other. Resolution and lowering live in
//! one file because the family is one question — how many bytes move
//! between which register lanes and which addresses — unlike the
//! data-processing families next door, which split *what* an
//! instruction is from *how* it is built.
//!
//! Three shapes exist, and the mnemonic alone does not tell them apart:
//! the list does. `{v0.4s, v1.4s}` names whole arranged views,
//! `{v0.s, v1.s}[1]` names one element of each, and the replicating
//! `r` mnemonics read one element and broadcast it. Everything the
//! resolver cannot place in one of the three declines, which the
//! effect table sees as [`crate::effect::InstructionKind::Other`] and
//! the slicer as a truncation.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::{Arrangement, element_type_bits, parse_arrangement, parse_lane_index};

use super::super::super::{LiftCtx, MemAccess};
use super::super::aarch64_mem_access;

/// Bits in a byte, wherever the model counts bits but an address counts
/// bytes.
const BITS_PER_BYTE: u16 = 8;

/// Vector registers an `AArch64` structured list may name. The
/// architecture encodes one to four consecutive registers.
const MAX_LIST_REGISTERS: usize = 4;

/// Number of vector registers in the `AArch64` register file. A list
/// that runs past `v31` wraps to `v0` (ARM ARM C7.2, "the registers are
/// consecutive, modulo 32").
const SIMD_REGISTER_COUNT: u8 = 32;

/// Width of an `AArch64` vector register.
const SIMD_REGISTER_BITS: u16 = 128;

/// What each list member transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListElement {
    /// `{v0.4s, v1.4s}` — a whole arranged view per member.
    Whole(Arrangement),
    /// `{v0.s, v1.s}[1]` — one element per member, at the index the
    /// list spells once for all of them.
    Indexed { lane_bits: u16, index: u16 },
}

/// Which registers a structured access reads and writes, in the terms
/// the effect table needs.
///
/// Kept separate from the resolved access so [`crate::lift::VectorShape`]
/// can carry it without carrying the member list: the effect table
/// recovers the register *names* from the operand text through
/// [`crate::effect::registers_in_operand`], and only needs to be told
/// which side of the transfer they sit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StructuredEffect {
    /// The listed registers are read — every store, and the
    /// single-element load, which writes one lane and preserves the
    /// rest of the register.
    pub(crate) reads_list: bool,
    /// The listed registers are written.
    pub(crate) writes_list: bool,
    /// The base register is advanced by the access.
    pub(crate) writes_base: bool,
}

/// The family a structured mnemonic names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Family {
    /// The `N` in `ldN` — how many structures the access interleaves.
    /// One for `ld1` / `st1`, which is contiguous and interleaves
    /// nothing, so a two-register `ld1` is simply two consecutive
    /// loads.
    interleave: u16,
    /// The `r` suffix: read one element per register and broadcast it
    /// across every lane.
    replicate: bool,
    /// Whether the access writes memory from the registers.
    stores: bool,
}

/// One unit of a structured transfer: `bits` wide at `base + offset`,
/// moving either the member's whole view or the single lane `lane`
/// names.
struct TransferUnit<'a> {
    member: &'a Operand,
    offset: u32,
    bits: u16,
    lane: Option<u16>,
}

/// A resolved structured load or store.
pub(crate) struct Structured {
    /// One operand per list member, in list order, spelled so the
    /// register table resolves it.
    members: Vec<Operand>,
    element: ListElement,
    family: Family,
    /// Whether the access advances its base register — the post-index
    /// forms, which carry a third operand.
    writes_base: bool,
}

impl Structured {
    /// Which registers this access reads and writes.
    pub(crate) const fn effect(&self) -> StructuredEffect {
        let indexed_load =
            matches!(self.element, ListElement::Indexed { .. }) && !self.family.stores;
        StructuredEffect {
            reads_list: self.family.stores || indexed_load,
            writes_list: !self.family.stores,
            writes_base: self.writes_base,
        }
    }
}

/// Resolve `insn` into the structured access it performs, or `None`
/// when it is not one this module models.
///
/// Every constraint the architecture places on the family is checked
/// here rather than during lowering, so a resolved access lowers
/// completely or the register it failed to write stays a free input.
pub(crate) fn resolve(insn: &Instruction) -> Option<Structured> {
    let family = family(&insn.mnemonic.trim().to_ascii_lowercase())?;
    // De-interleaving (`ld2` / `ld3` / `ld4`) and replication (`ldNr`)
    // are recognised as families but have no lowering here yet, so they
    // decline at resolution rather than at lowering: the effect table
    // consults this same function, and a shape it retained but the
    // lifter dropped would leave a later read bound to a stale value.
    if family.interleave > 1 || family.replicate {
        return None;
    }
    let list = insn.operands.first()?;
    let memory = insn.operands.get(1)?;
    if memory.kind != OperandKind::Memory {
        return None;
    }
    let post = insn.operands.get(2);
    if insn.operands.len() > 3 || post.is_some_and(|op| !is_post_index_operand(op)) {
        return None;
    }
    let (members, element) = parse_vector_reglist(list)?;
    // `ld1` / `st1` move one to four whole registers; the
    // single-element form moves one element of one register.
    let permitted = match element {
        ListElement::Whole(_) => MAX_LIST_REGISTERS,
        ListElement::Indexed { .. } => 1,
    };
    if members.is_empty() || members.len() > permitted {
        return None;
    }
    Some(Structured {
        members,
        element,
        family,
        writes_base: post.is_some(),
    })
}

/// The structured family a mnemonic names.
///
/// `ldr` / `ldp` / `str` / `stp` share the `ld` / `st` stem, so the
/// remainder has to be exactly a digit (optionally followed by the
/// replicating `r`) rather than merely start with one.
fn family(mnemonic: &str) -> Option<Family> {
    let (body, stores) = match (mnemonic.strip_prefix("ld"), mnemonic.strip_prefix("st")) {
        (Some(rest), _) => (rest, false),
        (_, Some(rest)) => (rest, true),
        _ => return None,
    };
    let (digits, replicate) = match body.strip_suffix('r') {
        // There is no replicating store: the operation reads one
        // element and broadcasts it, which has no store dual.
        Some(_) if stores => return None,
        Some(digits) => (digits, true),
        None => (body, false),
    };
    let interleave = match digits {
        "1" => 1,
        "2" => 2,
        "3" => 3,
        "4" => 4,
        _ => return None,
    };
    Some(Family {
        interleave,
        replicate,
        stores,
    })
}

/// Whether a trailing operand is one a post-index access can carry: the
/// immediate stride, or the register holding a variable one.
fn is_post_index_operand(op: &Operand) -> bool {
    matches!(op.kind, OperandKind::Immediate | OperandKind::Register)
}

/// Parse a `{v0.4s, v1.4s}` or `{v0.s, v1.s}[1]` vector register list.
///
/// Returns one operand per member — re-spelled with the list's element
/// index attached, since the register table resolves `v0.s[1]` and not
/// the bare `v0.s` a single-element list writes — together with the
/// element shape every member shares.
///
/// Sibling of `AArch32`'s `parse_reglist`, and different from it in the
/// two ways the ISAs differ: the members carry an arrangement, and the
/// list order is architectural rather than sorted, because a structured
/// access assigns consecutive addresses to consecutive list positions.
fn parse_vector_reglist(op: &Operand) -> Option<(Vec<Operand>, ListElement)> {
    let raw = op.raw.trim().to_ascii_lowercase();
    let (body, tail) = raw.strip_prefix('{')?.split_once('}')?;
    let index = match tail.trim() {
        "" => None,
        indexed => Some(parse_lane_index(indexed)?),
    };
    let names: Vec<&str> = body.split(',').map(str::trim).collect();
    if names.is_empty() || names.len() > MAX_LIST_REGISTERS {
        return None;
    }
    let mut members = Vec::with_capacity(names.len());
    let mut shared: Option<ListElement> = None;
    let mut first_number = None;
    for (position, name) in names.iter().enumerate() {
        let (register, shape) = name.split_once('.')?;
        let number = vector_register_number(register)?;
        let expected = wrapped_register(*first_number.get_or_insert(number), position)?;
        if number != expected {
            return None;
        }
        let element = list_element(shape, index)?;
        if *shared.get_or_insert(element) != element {
            return None;
        }
        members.push(Operand {
            raw: match index {
                Some(lane) => format!("{name}[{lane}]"),
                None => (*name).to_string(),
            },
            kind: OperandKind::Register,
        });
    }
    Some((members, shared?))
}

/// The element shape a list member's `.`-suffix names, given whether
/// the list as a whole carried an index.
///
/// An indexed list spells a bare element type (`v0.s`) because the
/// index applies to every member at once; a plain one spells a full
/// arrangement (`v0.4s`). Mixing the two is not a shape the
/// architecture has.
fn list_element(shape: &str, index: Option<u16>) -> Option<ListElement> {
    let Some(index) = index else {
        return Some(ListElement::Whole(parse_arrangement(shape)?));
    };
    let mut letters = shape.chars();
    let lane_bits = element_type_bits(letters.next()?)?;
    if letters.next().is_some() {
        return None;
    }
    // The addressed element has to lie inside the register.
    (index < SIMD_REGISTER_BITS / lane_bits).then_some(ListElement::Indexed { lane_bits, index })
}

/// The number of the `vN` a list member names.
fn vector_register_number(name: &str) -> Option<u8> {
    name.strip_prefix('v')?
        .parse::<u8>()
        .ok()
        .filter(|number| *number < SIMD_REGISTER_COUNT)
}

/// The register `position` places after `first`, wrapping at `v31`.
fn wrapped_register(first: u8, position: usize) -> Option<u8> {
    let offset = u8::try_from(position).ok()?;
    Some((first.wrapping_add(offset)) % SIMD_REGISTER_COUNT)
}

impl LiftCtx {
    /// Lower a resolved structured load or store.
    ///
    /// The base-register writeback is emitted last so every address in
    /// the transfer reads the pre-update base under SSA rename, exactly
    /// as [`LiftCtx::emit_writeback`]'s scalar callers arrange.
    ///
    /// A decline is sound by the free-input boundary: the slicer
    /// consumed the listed registers as definitions and stopped
    /// tracking what defined them upstream, so a register left unwritten
    /// is a free SSA input rather than a stale value.
    pub(in crate::lift::aarch64) fn lift_structured(
        &mut self,
        insn: &Instruction,
        access: &Structured,
    ) {
        let Some(memory) = insn.operands.get(1) else {
            self.push_structured_unsupported(insn);
            return;
        };
        let Some(MemAccess { address, writeback }) =
            aarch64_mem_access(memory, insn.operands.get(2), self.bits)
        else {
            self.push_structured_unsupported(insn);
            return;
        };
        let transferred = match access.element {
            ListElement::Whole(arrangement) => {
                self.transfer_contiguous(insn, access, &address, arrangement)
            }
            ListElement::Indexed { lane_bits, index } => {
                self.transfer_elements(insn, access, &address, lane_bits, index)
            }
        };
        if !transferred {
            self.push_structured_unsupported(insn);
            return;
        }
        self.emit_writeback(writeback);
    }

    /// `ld1` / `st1` over whole registers: one access per member, each
    /// a whole view above the previous.
    fn transfer_contiguous(
        &mut self,
        insn: &Instruction,
        access: &Structured,
        base: &Expr,
        arrangement: Arrangement,
    ) -> bool {
        let view = arrangement.view_bits();
        let Some(stride) = byte_width(view) else {
            return false;
        };
        for (position, member) in access.members.iter().enumerate() {
            let Some(offset) = list_position(position).and_then(|p| element_offset(p, stride))
            else {
                return false;
            };
            let unit = TransferUnit {
                member,
                offset,
                bits: view,
                lane: None,
            };
            if !self.transfer_unit(insn, base, &unit, access.family.stores) {
                return false;
            }
        }
        true
    }

    /// The single-element forms: one element per member, at consecutive
    /// addresses, addressing the lane the list's index names. A load
    /// writes that lane and preserves the rest of the register.
    fn transfer_elements(
        &mut self,
        insn: &Instruction,
        access: &Structured,
        base: &Expr,
        lane_bits: u16,
        index: u16,
    ) -> bool {
        let Some(stride) = byte_width(lane_bits) else {
            return false;
        };
        for (position, member) in access.members.iter().enumerate() {
            let Some(offset) = list_position(position).and_then(|p| element_offset(p, stride))
            else {
                return false;
            };
            let unit = TransferUnit {
                member,
                offset,
                bits: lane_bits,
                lane: Some(index),
            };
            if !self.transfer_unit(insn, base, &unit, access.family.stores) {
                return false;
            }
        }
        true
    }

    /// One unit of a transfer, in whichever direction `stores` names.
    fn transfer_unit(
        &mut self,
        insn: &Instruction,
        base: &Expr,
        unit: &TransferUnit<'_>,
        stores: bool,
    ) -> bool {
        let TransferUnit {
            member,
            offset,
            bits,
            lane,
        } = *unit;
        let address = self.offset_address(base, offset);
        if stores {
            let value = match lane {
                Some(index) => self.read_simd_lane_bits(member, bits, index),
                None => self.simd_operand_value(member, bits),
            };
            let Some(value) = value else {
                return false;
            };
            self.stmts.push(IrStmt::StoreMem {
                address,
                value,
                bits,
            });
            return true;
        }
        let temp = self.new_temp(insn.address, bits);
        self.stmts.push(IrStmt::LoadMem {
            dst: temp.clone(),
            address,
            bits,
        });
        match lane {
            // An `AArch64` SIMD write covers the whole register: the
            // arrangement's view is written and everything above it
            // zeroed.
            None => self.write_simd_dst(member, Expr::Var(temp), true),
            Some(index) => self.write_simd_lane(member, Expr::Var(temp), bits, index),
        }
    }

    /// `base + offset` at the pointer width, or the base alone when the
    /// offset is zero.
    fn offset_address(&self, base: &Expr, offset: u32) -> Expr {
        if offset == 0 {
            return base.clone();
        }
        Expr::add(base.clone(), Expr::konst(u128::from(offset), self.bits))
    }

    fn push_structured_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!(
                "unmodellable structured access at {addr}",
                addr = insn.address
            ),
        });
    }
}

/// Byte width of a `bits`-wide unit, or `None` when it is not a whole
/// number of bytes.
fn byte_width(bits: u16) -> Option<u16> {
    if bits == 0 || bits % BITS_PER_BYTE != 0 {
        return None;
    }
    Some(bits / BITS_PER_BYTE)
}

/// Byte offset of the `position`-th `stride`-byte unit of a transfer.
fn element_offset(position: u32, stride: u16) -> Option<u32> {
    position.checked_mul(u32::from(stride))
}

/// A list position as a transfer position.
fn list_position(position: usize) -> Option<u32> {
    u32::try_from(position).ok()
}

#[cfg(test)]
mod tests;
