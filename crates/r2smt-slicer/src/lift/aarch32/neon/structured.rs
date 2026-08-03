//! The `AArch32` structured accesses — `vld1`–`vld4` and `vst1`–`vst4`,
//! which move bytes between memory and a list of `d` registers,
//! de-interleaving them on the way.
//!
//! Resolution and lowering live in one file, as they do on the
//! `AArch64` side: the transfer is described by the same three numbers
//! that perform it, and splitting them across modules would mean
//! restating the geometry.
//!
//! `AArch32` spells its list members bare — `{d0, d1}` where `AArch64`
//! writes `{v0.8b, v1.8b}` — so the element width comes off the
//! mnemonic and each member's view off its register class. The list is
//! also *not* sorted here, unlike the `ldm` / `push` parser this sits
//! beside: consecutive list positions take consecutive addresses, so
//! reordering them would transfer the right bytes to the wrong
//! registers.

use r2smt_common::Arch;
use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::lift::{LiftCtx, StructuredEffect, Writeback};
use crate::registers::{is_simd_parent, register_layout};

use super::{BITS_PER_BYTE, bare_element_bits};

/// Width of a `d` register, the only view a structured list member may
/// name.
const DOUBLEWORD_BITS: u16 = 64;
/// The architecture lists at most four registers.
const MAX_LIST_REGISTERS: usize = 4;

/// A resolved structured access.
pub(in crate::lift::aarch32) struct Structured {
    /// The listed registers, in the order written — which is the order
    /// their bytes appear at, so this is never sorted.
    members: Vec<Operand>,
    /// Element width in bits, from the mnemonic.
    element_bits: u16,
    /// The `N` in `vldN`: how many structures the access interleaves.
    /// One for `vld1` / `vst1`, which is contiguous.
    interleave: u16,
    /// Whether the access writes memory from the registers.
    stores: bool,
    /// Base register, already resolved to its parent.
    base: String,
    /// The access advances the base by the bytes it transferred.
    writes_base: bool,
}

impl Structured {
    /// How the slicer should read this access's operands.
    ///
    /// `reads_list` is unconditionally true, unlike `AArch64`, and for
    /// the reason every `AArch32` vector entry says so: a write to `d0`
    /// preserves the other half of its parent, so the parent's prior
    /// value stays live even on a pure load.
    pub(in crate::lift::aarch32) const fn effect(&self) -> StructuredEffect {
        StructuredEffect {
            reads_list: true,
            writes_list: !self.stores,
            writes_base: self.writes_base,
        }
    }

    /// Total bytes moved — one whole `d` register per list member.
    fn transferred_bytes(&self) -> Option<i64> {
        i64::try_from(self.members.len())
            .ok()?
            .checked_mul(i64::from(DOUBLEWORD_BITS / BITS_PER_BYTE))
    }

    /// Elements each member holds.
    fn lanes(&self) -> Option<u16> {
        (self.element_bits != 0 && DOUBLEWORD_BITS % self.element_bits == 0)
            .then(|| DOUBLEWORD_BITS / self.element_bits)
    }
}

/// Resolve `insn` into the structured access it performs, or `None`
/// when it is not one this module models.
pub(in crate::lift::aarch32) fn resolve(insn: &Instruction) -> Option<Structured> {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    let (base_mnemonic, ty) = mnemonic.split_once('.')?;
    let (interleave, stores) = family(base_mnemonic)?;
    let element_bits = bare_element_bits(ty)?;
    // Only the contiguous family has a doubleword element; an
    // interleaved one would have a single structure fill the register.
    if interleave > 1 && element_bits >= DOUBLEWORD_BITS {
        return None;
    }
    let members = parse_members(insn.operands.first()?)?;
    // A `vld2` names exactly two registers here, a `vld3` three, a
    // `vld4` four. The stride-two spellings (`{d0, d2}`) and the
    // double-width `vld2 {d0,d1,d2,d3}` are different encodings and
    // decline rather than being read as these.
    if interleave > 1 && usize::from(interleave) != members.len() {
        return None;
    }
    let (base, writes_base) = parse_base(insn.operands.get(1)?)?;
    // Only the immediate post-index is modelled; a register post-index
    // would advance the base by a value the writeback cannot spell as a
    // constant.
    if insn.operands.len() > 2 {
        return None;
    }
    Some(Structured {
        members,
        element_bits,
        interleave,
        stores,
        base,
        writes_base,
    })
}

/// The interleave and direction a structured mnemonic names.
fn family(base: &str) -> Option<(u16, bool)> {
    let (stem, stores) = match base.strip_prefix("vld") {
        Some(rest) => (rest, false),
        None => (base.strip_prefix("vst")?, true),
    };
    match stem {
        "1" => Some((1, stores)),
        "2" => Some((2, stores)),
        "3" => Some((3, stores)),
        "4" => Some((4, stores)),
        _ => None,
    }
}

/// Parse `{d0, d1}` into its members, in list order.
///
/// Declines the single-element (`{d0[3]}`) and replicating (`{d0[]}`)
/// shapes, which transfer one element rather than whole registers, and
/// any list whose members are not consecutive — the stride-two forms
/// are a separate encoding.
fn parse_members(op: &Operand) -> Option<Vec<Operand>> {
    if op.kind != OperandKind::Register && op.kind != OperandKind::Memory {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    let body = raw.strip_prefix('{')?.strip_suffix('}')?;
    let mut members = Vec::new();
    let mut numbers = Vec::new();
    for part in body.split(',') {
        let name = part.trim();
        if name.contains('[') {
            return None;
        }
        let layout = register_layout(name, Arch::Arm)?;
        if !is_simd_parent(layout.parent, Arch::Arm) || layout.width() != DOUBLEWORD_BITS {
            return None;
        }
        numbers.push(doubleword_number(name)?);
        members.push(Operand {
            raw: name.to_string(),
            kind: OperandKind::Register,
        });
    }
    if members.is_empty() || members.len() > MAX_LIST_REGISTERS {
        return None;
    }
    let first = *numbers.first()?;
    for (position, number) in numbers.iter().enumerate() {
        if u16::from(*number) != u16::from(first).checked_add(u16::try_from(position).ok()?)? {
            return None;
        }
    }
    Some(members)
}

/// The `N` in `dN`.
fn doubleword_number(name: &str) -> Option<u8> {
    let number: u8 = name.strip_prefix('d')?.parse().ok()?;
    (number < 32).then_some(number)
}

/// Split `[rN]` or `[rN]!` into its base parent and whether the access
/// advances it.
///
/// Deliberately narrow. An alignment specifier (`[r0:64]`, `[r0@128]`)
/// and any offset are outside the modelled subset, and a disassembler
/// spells alignment more than one way, so anything but a bare register
/// inside the brackets declines rather than being guessed at.
fn parse_base(op: &Operand) -> Option<(String, bool)> {
    if op.kind != OperandKind::Memory {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    let (body, writes_base) = match raw.strip_suffix('!') {
        Some(inner) => (inner.trim().to_string(), true),
        None => (raw, false),
    };
    let inner = body.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.contains([':', '@', ',', '#', '+', '-', ' ']) {
        return None;
    }
    let layout = register_layout(inner, Arch::Arm)?;
    if is_simd_parent(layout.parent, Arch::Arm) {
        return None;
    }
    Some((layout.parent.to_string(), writes_base))
}

impl LiftCtx {
    /// Lower a resolved structured access.
    pub(in crate::lift::aarch32) fn lift_aarch32_structured(
        &mut self,
        insn: &Instruction,
        access: &Structured,
    ) {
        let transferred = access.transferred_bytes();
        let moved = if access.interleave == 1 {
            self.transfer_aarch32_contiguous(insn, access)
        } else {
            self.transfer_aarch32_interleaved(insn, access)
        };
        let Some(bytes) = transferred else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        if !moved {
            self.push_aarch32_neon_unsupported(insn);
            return;
        }
        if access.writes_base {
            self.emit_writeback(Some(Writeback::by_constant(&access.base, bytes, self.bits)));
        }
    }

    /// `vld1` / `vst1` — the list is one contiguous block, so each
    /// member is a whole-register transfer and the element width says
    /// nothing about the layout.
    fn transfer_aarch32_contiguous(&mut self, insn: &Instruction, access: &Structured) -> bool {
        let stride = i64::from(DOUBLEWORD_BITS / BITS_PER_BYTE);
        for (position, member) in access.members.iter().enumerate() {
            let Ok(index) = i64::try_from(position) else {
                return false;
            };
            let Some(offset) = index.checked_mul(stride) else {
                return false;
            };
            if !self.transfer_aarch32_unit(insn, access, member, offset, DOUBLEWORD_BITS, None) {
                return false;
            }
        }
        true
    }

    /// `vld2` – `vld4` — consecutive elements in memory belong to
    /// consecutive *registers*, so element `e` of member `s` sits at
    /// element position `e * interleave + s`.
    fn transfer_aarch32_interleaved(&mut self, insn: &Instruction, access: &Structured) -> bool {
        let (Some(lanes), Some(element_bytes)) = (
            access.lanes(),
            Some(i64::from(access.element_bits / BITS_PER_BYTE)),
        ) else {
            return false;
        };
        for (structure, member) in access.members.iter().enumerate() {
            for lane in 0..lanes {
                let Some(offset) = i64::from(lane)
                    .checked_mul(i64::from(access.interleave))
                    .and_then(|p| i64::try_from(structure).ok().and_then(|s| p.checked_add(s)))
                    .and_then(|p| p.checked_mul(element_bytes))
                else {
                    return false;
                };
                if !self.transfer_aarch32_unit(
                    insn,
                    access,
                    member,
                    offset,
                    access.element_bits,
                    Some(lane),
                ) {
                    return false;
                }
            }
        }
        true
    }

    /// One unit of a structured transfer: `bits` wide at
    /// `base + offset`, moving either the member's whole view or the
    /// single lane `lane` names.
    fn transfer_aarch32_unit(
        &mut self,
        insn: &Instruction,
        access: &Structured,
        member: &Operand,
        offset: i64,
        bits: u16,
        lane: Option<u16>,
    ) -> bool {
        let address = self.aarch32_structured_address(&access.base, offset);
        if access.stores {
            let value = match lane {
                Some(lane) => self.read_simd_lane_bits(member, bits, lane),
                None => self.read_simd_operand(member),
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
        let destination = self.new_temp(insn.address, bits);
        self.stmts.push(IrStmt::LoadMem {
            dst: destination.clone(),
            address,
            bits,
        });
        let loaded = Expr::Var(destination);
        match lane {
            // A NEON write merges, so a lane write preserves the rest
            // of the register and a whole-register one preserves the
            // other half of its parent.
            Some(lane) => self.write_simd_lane(member, loaded, bits, lane),
            None => self.write_simd_dst(member, loaded, false),
        }
    }

    fn aarch32_structured_address(&self, base: &str, offset: i64) -> Expr {
        crate::lift::aarch32::aarch32_addr_from(base, offset, self.bits)
    }
}
