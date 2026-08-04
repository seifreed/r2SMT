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
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::lift::{LiftCtx, StructuredEffect, Writeback};
use crate::registers::{is_simd_parent, register_layout};

use super::{BITS_PER_BYTE, bare_element_bits, is_general_register};

/// Width of a `d` register, the only view a structured list member may
/// name.
const DOUBLEWORD_BITS: u16 = 64;
/// The architecture lists at most four registers.
const MAX_LIST_REGISTERS: usize = 4;

/// What each list member transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ListShape {
    /// `{d0, d1}` — the member's whole `d` register.
    Whole,
    /// `{d0[3]}` — one element, at this lane index.
    Element(u16),
    /// `{d0[]}` — one element, broadcast to every lane. Load-only.
    Replicate,
}

/// How the access advances its base register afterward.
#[derive(Debug, Clone)]
enum BaseAdvance {
    /// No writeback.
    None,
    /// `[r0]!` — advance by the bytes transferred, a constant.
    Bytes,
    /// `[r0], r1` — advance by the value of a general register.
    Register(String),
}

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
    /// What each member transfers — the whole register, one element, or
    /// one element broadcast.
    shape: ListShape,
    /// How the base advances after the transfer.
    advance: BaseAdvance,
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
            writes_base: !matches!(self.advance, BaseAdvance::None),
        }
    }

    /// Bytes per member: a whole `d` register for the whole-register
    /// shape, one element for the single-element and replicating ones.
    fn bytes_per_member(&self) -> u16 {
        match self.shape {
            ListShape::Whole => DOUBLEWORD_BITS / BITS_PER_BYTE,
            ListShape::Element(_) | ListShape::Replicate => self.element_bits / BITS_PER_BYTE,
        }
    }

    /// Total bytes moved.
    fn transferred_bytes(&self) -> Option<i64> {
        i64::try_from(self.members.len())
            .ok()?
            .checked_mul(i64::from(self.bytes_per_member()))
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
    // Only the contiguous whole-register family has a doubleword element;
    // an interleaved or single-element one would have a single structure
    // fill the register.
    if interleave > 1 && element_bits >= DOUBLEWORD_BITS {
        return None;
    }
    let (members, shape, stride) = parse_list(insn.operands.first()?)?;
    // A `vldN` names exactly `N` registers, whichever shape they carry.
    // `vld1` may name one to four.
    if interleave > 1 && usize::from(interleave) != members.len() {
        return None;
    }
    // Only the interleaved family has a stride-two spelling; a contiguous
    // `vld1 {d0, d2}` is not an encoding.
    if stride != 1 && interleave == 1 {
        return None;
    }
    validate_shape(shape, element_bits, stores)?;
    let (base, immediate_advance) = parse_base(insn.operands.get(1)?)?;
    let advance = base_advance(insn, immediate_advance)?;
    Some(Structured {
        members,
        element_bits,
        interleave,
        stores,
        base,
        shape,
        advance,
    })
}

/// The base-register writeback the access performs.
///
/// A register post-index (`[r0], r1`) and an immediate `!` are mutually
/// exclusive: the base advances by a register value or by the transfer
/// size, never both.
fn base_advance(insn: &Instruction, immediate: bool) -> Option<BaseAdvance> {
    match insn.operands.get(2) {
        None => Some(if immediate {
            BaseAdvance::Bytes
        } else {
            BaseAdvance::None
        }),
        Some(post) => {
            // A register post-index cannot also carry the `!` immediate
            // writeback.
            if immediate || !is_general_register(post) {
                return None;
            }
            let layout = register_layout(post.raw.trim(), Arch::Arm)?;
            Some(BaseAdvance::Register(layout.parent.to_string()))
        }
    }
}

/// Reject the shapes a mnemonic cannot carry: a replicating store (there
/// is no such encoding) and a single-element index outside the register.
fn validate_shape(shape: ListShape, element_bits: u16, stores: bool) -> Option<()> {
    match shape {
        ListShape::Whole => Some(()),
        // A replicating access reads one element and broadcasts it, which
        // has no store dual.
        ListShape::Replicate => (!stores).then_some(()),
        ListShape::Element(index) => {
            let lanes = (element_bits != 0 && DOUBLEWORD_BITS % element_bits == 0)
                .then(|| DOUBLEWORD_BITS / element_bits)?;
            (index < lanes).then_some(())
        }
    }
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
///
/// Shared with the `vtbl` / `vtbx` table parser, whose register list is
/// the same shape (consecutive whole `d` registers).
pub(in crate::lift::aarch32::neon) fn parse_members(op: &Operand) -> Option<Vec<Operand>> {
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

/// Parse a structured list into its members, the shape each carries, and
/// the register stride.
///
/// Unlike [`parse_members`] (which the `vtbl` table parser shares, and
/// which admits only consecutive whole `d` registers), this admits the
/// single-element and replicating spellings and the stride-two lists the
/// interleaved family allows. The members are never sorted — consecutive
/// list positions take consecutive addresses.
fn parse_list(op: &Operand) -> Option<(Vec<Operand>, ListShape, u16)> {
    if op.kind != OperandKind::Register && op.kind != OperandKind::Memory {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    let body = raw.strip_prefix('{')?.strip_suffix('}')?;
    let mut members = Vec::new();
    let mut numbers = Vec::new();
    let mut shape: Option<ListShape> = None;
    for part in body.split(',') {
        let (reg_name, member_shape) = parse_list_member(part.trim())?;
        // Every member must carry the same shape — mixing `{d0, d1[2]}`
        // is not an encoding.
        match shape {
            None => shape = Some(member_shape),
            Some(existing) if existing == member_shape => {}
            Some(_) => return None,
        }
        let layout = register_layout(reg_name, Arch::Arm)?;
        if !is_simd_parent(layout.parent, Arch::Arm) || layout.width() != DOUBLEWORD_BITS {
            return None;
        }
        numbers.push(doubleword_number(reg_name)?);
        members.push(Operand {
            raw: reg_name.to_string(),
            kind: OperandKind::Register,
        });
    }
    if members.is_empty() || members.len() > MAX_LIST_REGISTERS {
        return None;
    }
    Some((members, shape?, list_stride(&numbers)?))
}

/// Split one list member into its register name and its transfer shape:
/// `d0` → whole, `d0[3]` → element 3, `d0[]` → replicate.
fn parse_list_member(name: &str) -> Option<(&str, ListShape)> {
    match name.split_once('[') {
        None => Some((name, ListShape::Whole)),
        Some((head, rest)) => {
            let inside = rest.strip_suffix(']')?.trim();
            let shape = if inside.is_empty() {
                ListShape::Replicate
            } else {
                ListShape::Element(inside.parse::<u16>().ok()?)
            };
            Some((head.trim(), shape))
        }
    }
}

/// The uniform register stride of a list — 1 for `{d0, d1}`, 2 for the
/// interleaved `{d0, d2}`. Any other spacing declines.
fn list_stride(numbers: &[u8]) -> Option<u16> {
    let first = u16::from(*numbers.first()?);
    let Some(second) = numbers.get(1) else {
        return Some(1);
    };
    let stride = u16::from(*second).checked_sub(first)?;
    if stride != 1 && stride != 2 {
        return None;
    }
    for (position, number) in numbers.iter().enumerate() {
        let expected = first.checked_add(u16::try_from(position).ok()?.checked_mul(stride)?)?;
        if u16::from(*number) != expected {
            return None;
        }
    }
    Some(stride)
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
    // An alignment specifier (`[r0:64]`, `[r0@128]`) is a hint the value
    // model ignores, so strip it and read the base register alone. An
    // offset or shift is a different addressing mode and still declines.
    let register = inner.split([':', '@']).next()?.trim();
    if register.contains([',', '#', '+', '-', ' ']) {
        return None;
    }
    let layout = register_layout(register, Arch::Arm)?;
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
        let moved = match access.shape {
            ListShape::Whole if access.interleave == 1 => {
                self.transfer_aarch32_contiguous(insn, access)
            }
            ListShape::Whole => self.transfer_aarch32_interleaved(insn, access),
            ListShape::Element(index) => self.transfer_aarch32_single_element(insn, access, index),
            ListShape::Replicate => self.transfer_aarch32_replicate(insn, access),
        };
        let Some(bytes) = transferred else {
            self.push_aarch32_neon_unsupported(insn);
            return;
        };
        if !moved {
            self.push_aarch32_neon_unsupported(insn);
            return;
        }
        match &access.advance {
            BaseAdvance::None => {}
            BaseAdvance::Bytes => {
                self.emit_writeback(Some(Writeback::by_constant(&access.base, bytes, self.bits)));
            }
            // The register post-index advances by a run-time value, which
            // the `Expr`-valued writeback delta carries directly.
            BaseAdvance::Register(register) => {
                self.emit_writeback(Some(Writeback {
                    base: access.base.clone(),
                    delta: Expr::Var(Var::new(register.clone(), self.bits)),
                }));
            }
        }
    }

    /// The single-element shapes (`{d0[3]}`, `{d0[3], d1[3]}`): each
    /// member receives one element at lane `index`, from consecutive
    /// addresses in list order.
    fn transfer_aarch32_single_element(
        &mut self,
        insn: &Instruction,
        access: &Structured,
        index: u16,
    ) -> bool {
        let element_bytes = i64::from(access.element_bits / BITS_PER_BYTE);
        for (structure, member) in access.members.iter().enumerate() {
            let Some(offset) = i64::try_from(structure)
                .ok()
                .and_then(|s| s.checked_mul(element_bytes))
            else {
                return false;
            };
            if !self.transfer_aarch32_unit(
                insn,
                access,
                member,
                offset,
                access.element_bits,
                Some(index),
            ) {
                return false;
            }
        }
        true
    }

    /// The replicating shapes (`{d0[]}`, `{d0[], d1[]}`): each member
    /// loads one element and broadcasts it across every lane. Load-only.
    fn transfer_aarch32_replicate(&mut self, insn: &Instruction, access: &Structured) -> bool {
        let Some(lanes) = access.lanes() else {
            return false;
        };
        let element_bytes = i64::from(access.element_bits / BITS_PER_BYTE);
        for (structure, member) in access.members.iter().enumerate() {
            let Some(offset) = i64::try_from(structure)
                .ok()
                .and_then(|s| s.checked_mul(element_bytes))
            else {
                return false;
            };
            let address = self.aarch32_structured_address(&access.base, offset);
            let temp = self.new_temp(insn.address, access.element_bits);
            self.stmts.push(IrStmt::LoadMem {
                dst: temp.clone(),
                address,
                bits: access.element_bits,
            });
            let Some(value) = Self::concat_lanes(vec![Expr::Var(temp); usize::from(lanes)]) else {
                return false;
            };
            if !self.write_simd_dst(member, value, false) {
                return false;
            }
        }
        true
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
