//! The families that move bits without computing on them.
//!
//! The immediate broadcasts, `dup`, `ext`, the permutations, the
//! element moves and inserts, the table lookups and the bitwise
//! selects. Every destination lane here is some source lane, some
//! literal, or a bit-granular choice between two of them; none is an
//! arithmetic function of what it selects.
//!
//! That makes their resolution strict in a different way from the
//! arithmetic families. The geometry is rarely in doubt — most of these
//! require one arrangement throughout — and what has to be checked is
//! that an index, a byte offset or a container size addresses something
//! that exists.

use r2smt_ir::program::{Instruction, Operand};

use super::super::super::parse_immediate;
use super::geometry::{
    BITS_PER_BYTE, indexed_element, is_general_register, operand_arrangement, spans_full_register,
};
use super::{NeonOp, NeonShape};

/// `movi vd.<T>, #imm{, lsl|msl #shift}` and `mvni`, which replicate an
/// immediate across every lane.
///
/// The disassembler prints the *decoded* immediate, so the printed value
/// is the per-lane one and needs no re-expansion — including for `.2d`,
/// whose encoded form is a byte mask.
pub(super) fn immediate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let invert = match mnemonic {
        "movi" => false,
        "mvni" => true,
        _ => return None,
    };
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    let raw = parse_immediate(&insn.operands.get(1)?.raw)?;
    let shifted = match insn.operands.len() {
        2 => raw,
        3 => {
            let (kind, shift) = shift_modifier(insn.operands.get(2)?)?;
            // ARM ARM C7.2 (MOVI / MVNI, shifting ones): the `msl` form
            // is encoded for 32-bit elements only, and only for a shift
            // of 8 or 16.
            if kind == ShiftModifier::Ones
                && (arrangement.lane_bits != 32 || !matches!(shift, 8 | 16))
            {
                return None;
            }
            apply_shift_modifier(kind, raw, shift)?
        }
        _ => return None,
    };
    Some(NeonShape {
        op: NeonOp::Immediate {
            value: shifted,
            invert,
        },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// Which bit an immediate shift feeds in from the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShiftModifier {
    /// `lsl` — zeroes.
    Zeroes,
    /// `msl` — ones. The whole reason it is a separate mnemonic: it
    /// builds the `0x0000ffff`-style masks a bare `lsl` cannot.
    Ones,
}

/// The `lsl #n` / `msl #n` shift operand of a `movi`. Radare2 renders
/// the shifting modifier as one operand, so the whole thing is parsed
/// here.
fn shift_modifier(op: &Operand) -> Option<(ShiftModifier, u16)> {
    let raw = op.raw.trim().to_ascii_lowercase();
    let (kind, body) = if let Some(body) = raw.strip_prefix("lsl") {
        (ShiftModifier::Zeroes, body)
    } else {
        (ShiftModifier::Ones, raw.strip_prefix("msl")?)
    };
    Some((kind, u16::try_from(parse_immediate(body.trim())?).ok()?))
}

/// Apply a shift modifier to the printed immediate.
///
/// Both are computed here, in Rust, rather than lowered as IR: every
/// operand of a `movi` is a literal, so the lane value is known at lift
/// time and emitting a shift node would leave the solver folding a
/// constant.
fn apply_shift_modifier(kind: ShiftModifier, raw: u64, shift: u16) -> Option<u64> {
    let shifted = raw.checked_shl(u32::from(shift))?;
    Some(match kind {
        ShiftModifier::Zeroes => shifted,
        // The vacated bits come in set, so the low `shift` bits are all
        // ones rather than all zeroes.
        ShiftModifier::Ones => shifted | (1u64.checked_shl(u32::from(shift))? - 1),
    })
}

/// `dup vd.<T>, rn` and `dup vd.<T>, vn.<Ts>[index]`.
///
/// The scalar-destination form (`dup d0, v1.d[1]`) declines: its
/// destination carries no arrangement, so the lane count is not spelled.
pub(super) fn duplicate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if mnemonic != "dup" || insn.operands.len() != 2 {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    let source = insn.operands.get(1)?;
    let (from_element, source_index) = if let Some((element_bits, index)) = indexed_element(source)
    {
        if element_bits != arrangement.lane_bits {
            return None;
        }
        (true, index)
    } else {
        // A general-purpose source supplies the low `lane_bits`.
        if !is_general_register(source) {
            return None;
        }
        (false, 0)
    };
    Some(NeonShape {
        op: NeonOp::Duplicate { from_element },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index,
    })
}

/// `ext vd.<T>, vn.<T>, vm.<T>, #index` — a byte-granular window over
/// the two sources concatenated, with `vn` at the low end.
///
/// Only the byte arrangements are architecturally valid, and the index
/// must leave a whole view inside the concatenation.
pub(super) fn extract_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if mnemonic != "ext" || insn.operands.len() != 4 {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    if arrangement.lane_bits != BITS_PER_BYTE {
        return None;
    }
    for index in 1..=2 {
        if operand_arrangement(insn.operands.get(index)?)? != arrangement {
            return None;
        }
    }
    let byte_offset = u16::try_from(parse_immediate(&insn.operands.get(3)?.raw)?).ok()?;
    if byte_offset >= arrangement.lanes {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Extract { byte_offset },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// Which of an instruction's two vector sources a permuted lane comes
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PermuteSource {
    First,
    Second,
}

/// A lane permutation, expressed as the source each destination lane
/// draws from. Every member of the family is a pure rearrangement of
/// existing lanes, so one representation covers them all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PermuteKind {
    /// `zip1` / `zip2` — interleave the lower or upper halves.
    Zip { upper: bool },
    /// `uzp1` / `uzp2` — deinterleave the even or odd lanes.
    Uzp { odd: bool },
    /// `trn1` / `trn2` — transpose the even or odd lanes.
    Trn { odd: bool },
    /// `rev16` / `rev32` / `rev64` — reverse the element order within
    /// each container of this many bits.
    Reverse { container_bits: u16 },
}

/// The permutation family: `zip`, `uzp`, `trn` over two sources, and
/// `rev` over one.
pub(super) fn permute_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let kind = match mnemonic {
        "zip1" => PermuteKind::Zip { upper: false },
        "zip2" => PermuteKind::Zip { upper: true },
        "uzp1" => PermuteKind::Uzp { odd: false },
        "uzp2" => PermuteKind::Uzp { odd: true },
        "trn1" => PermuteKind::Trn { odd: false },
        "trn2" => PermuteKind::Trn { odd: true },
        "rev16" => PermuteKind::Reverse { container_bits: 16 },
        "rev32" => PermuteKind::Reverse { container_bits: 32 },
        "rev64" => PermuteKind::Reverse { container_bits: 64 },
        _ => return None,
    };
    let operand_count = if matches!(kind, PermuteKind::Reverse { .. }) {
        2
    } else {
        3
    };
    if insn.operands.len() != operand_count {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    for index in 1..operand_count {
        if operand_arrangement(insn.operands.get(index)?)? != arrangement {
            return None;
        }
    }
    // A reversal needs whole elements inside each container, and at
    // least two of them for the reversal to mean anything.
    if let PermuteKind::Reverse { container_bits } = kind {
        if container_bits <= arrangement.lane_bits
            || container_bits % arrangement.lane_bits != 0
            || arrangement.view_bits() % container_bits != 0
        {
            return None;
        }
    } else if arrangement.lanes % 2 != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Permute(kind),
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// `umov wd, vn.<Ts>[index]` and `smov`, which move one element into a
/// general-purpose register.
pub(super) fn element_to_gpr_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let signed = match mnemonic {
        // `mov wd, vn.s[0]` is the assembler alias for `umov`.
        "umov" | "mov" => false,
        "smov" => true,
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = insn.operands.first()?;
    if !is_general_register(destination) {
        return None;
    }
    let (lane_bits, source_index) = indexed_element(insn.operands.get(1)?)?;
    Some(NeonShape {
        op: NeonOp::ElementToGpr { signed },
        lane_bits,
        lanes: 1,
        dest_index: 0,
        source_index,
    })
}

/// `ins vd.<Ts>[index], rn` and `ins vd.<Ts>[index], vn.<Ts>[index2]`,
/// which write one lane and preserve the rest.
pub(super) fn insert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    // `mov v0.s[1], w0` is the assembler alias for `ins`.
    if !matches!(mnemonic, "ins" | "mov") || insn.operands.len() != 2 {
        return None;
    }
    let (lane_bits, dest_index) = indexed_element(insn.operands.first()?)?;
    let source = insn.operands.get(1)?;
    let (from_element, source_index) = if let Some((source_bits, index)) = indexed_element(source) {
        if source_bits != lane_bits {
            return None;
        }
        (true, index)
    } else {
        if !is_general_register(source) {
            return None;
        }
        (false, 0)
    };
    Some(NeonShape {
        op: NeonOp::Insert { from_element },
        lane_bits,
        lanes: 1,
        dest_index,
        source_index,
    })
}

/// Ceiling on the selects one table lookup may expand into.
///
/// Every destination byte is an `Ite` chain over the whole table, so
/// the formula grows as `destination bytes * table bytes` — 256 for the
/// widest single-register form, and 1 024 for the four-register one.
/// The bound exists so that growth is a decision rather than a
/// surprise: past it the instruction declines, and the slicer truncates
/// as it does for anything unmodelled.
const TABLE_LOOKUP_SELECT_CAP: u32 = 1024;

/// `tbl vd.<T>, {vn.16b[, …]}, vm.<T>` and `tbx`.
///
/// The table is one to four consecutive whole `.16b` registers whose
/// bytes concatenate low-to-high into a single lookup table; the
/// register-list length is bounded by the structured-access parser, and
/// the resulting select chain by [`TABLE_LOOKUP_SELECT_CAP`].
pub(super) fn table_lookup_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let keep = match mnemonic {
        "tbl" => false,
        "tbx" => true,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if destination.lane_bits != BITS_PER_BYTE {
        return None;
    }
    // Each table register is always the full 128 bits, whatever the
    // destination's arrangement; the byte count is their sum.
    let tables = table_registers(insn.operands.get(1)?)?;
    let mut table_lanes: u16 = 0;
    for table in &tables {
        table_lanes = table_lanes.checked_add(operand_arrangement(table)?.lanes)?;
    }
    if operand_arrangement(insn.operands.get(2)?)? != destination {
        return None;
    }
    let selects = u32::from(destination.lanes).checked_mul(u32::from(table_lanes))?;
    if selects > TABLE_LOOKUP_SELECT_CAP {
        tracing::debug!(
            mnemonic,
            selects,
            cap = TABLE_LOOKUP_SELECT_CAP,
            "declining a table lookup whose select chain exceeds the cap"
        );
        return None;
    }
    Some(NeonShape {
        op: NeonOp::TableLookup { keep, table_lanes },
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The one-to-four consecutive registers a `{vN.16b[, …]}` table list
/// names, as plain register operands the SIMD readers accept.
///
/// Radare2 renders a register list as a single braced operand, which
/// classifies as neither a register nor a memory reference. The
/// consecutive-register rule is enforced by the shared list parser; each
/// member must be a whole `.16b` register.
pub(super) fn table_registers(op: &Operand) -> Option<Vec<Operand>> {
    let members = super::structured::parse_reglist_members(op)?;
    for member in &members {
        let arrangement = operand_arrangement(member)?;
        if arrangement.lane_bits != BITS_PER_BYTE || !spans_full_register(arrangement) {
            return None;
        }
    }
    Some(members)
}

/// Which role the destination register plays in a bitwise select.
///
/// All three mnemonics compute `(a & mask) | (b & ~mask)`; they differ
/// only in which operand is the mask and which is selected when the mask
/// bit is set. Every one of them reads the destination.
#[derive(Debug, Clone, Copy)]
pub(super) enum SelectRole {
    /// `bsl` — the destination is the mask; the sources supply the two
    /// candidate values.
    DestinationIsMask,
    /// `bit` — insert the first source where the second's bits are set,
    /// keeping the destination elsewhere.
    InsertWhereSet,
    /// `bif` — insert the first source where the second's bits are
    /// *clear*.
    InsertWhereClear,
}

/// `bsl` / `bit` / `bif`, the three-register bitwise selects.
///
/// All three read the destination, and the architecture spells them only
/// with the byte arrangements — the operation is bit-granular, so the
/// element width carries no meaning beyond the view's total size.
pub(super) fn bitwise_select_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let role = match mnemonic {
        "bsl" => SelectRole::DestinationIsMask,
        "bit" => SelectRole::InsertWhereSet,
        "bif" => SelectRole::InsertWhereClear,
        _ => return None,
    };
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    if destination.lane_bits != BITS_PER_BYTE {
        return None;
    }
    for operand in insn.operands.iter().skip(1) {
        if operand_arrangement(operand)? != destination {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::BitwiseSelect(role),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}
