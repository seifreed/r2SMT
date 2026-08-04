//! `vtbl` / `vtbx` — the `AArch32` table lookup.
//!
//! Each destination byte selects a byte from a table of one to four
//! consecutive `d` registers, indexed by the matching byte of the index
//! register. An index that names no table byte falls through to a base:
//! zero for `vtbl`, the destination's prior byte for `vtbx`.
//!
//! `AArch64`'s `tbl` / `tbx` is the same lowering over 128-bit `q`
//! table registers; here the table registers are 64-bit `d`s, so the
//! byte count is `8 * members` rather than `16 * members`, and the
//! select chain is short enough that no cap can bind — the widest form
//! is eight destination bytes over a 32-byte table, 256 selects.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::Instruction;

use crate::lift::LiftCtx;

use super::structured::parse_members;
use super::{BITS_PER_BYTE, NeonOp, NeonShape, vector_view_bits};

/// Width of a `d` register — the view of a table register, the index
/// register, and the destination alike.
const DOUBLEWORD_BITS: u16 = 64;
/// Bytes a `d` register holds, which is the table stride.
const DOUBLEWORD_BYTES: u16 = DOUBLEWORD_BITS / BITS_PER_BYTE;

/// `vtbl` / `vtbx` — resolve the destination geometry and table size.
pub(super) fn table_lookup_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let base = mnemonic.split('.').next()?;
    let keep = match base {
        "vtbl" => false,
        "vtbx" => true,
        _ => return None,
    };
    // The table lookup is byte-granular; a type suffix, if the
    // disassembler prints one, can only be `.8`.
    if let Some((_, ty)) = mnemonic.split_once('.')
        && ty != "8"
    {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    if vector_view_bits(insn.operands.first()?)? != DOUBLEWORD_BITS
        || vector_view_bits(insn.operands.get(2)?)? != DOUBLEWORD_BITS
    {
        return None;
    }
    let members = parse_members(insn.operands.get(1)?)?;
    let table_lanes = u16::try_from(members.len())
        .ok()?
        .checked_mul(DOUBLEWORD_BYTES)?;
    Some(NeonShape {
        op: NeonOp::TableLookup { keep, table_lanes },
        lane_bits: BITS_PER_BYTE,
        lanes: DOUBLEWORD_BYTES,
    })
}

impl LiftCtx {
    /// Lower a resolved `vtbl` / `vtbx`: one `Ite` chain per destination
    /// byte over the whole table, with the out-of-range answer at its
    /// base so an unmatched index falls through to it.
    pub(in crate::lift::aarch32::neon) fn aarch32_table_lookup_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        keep: bool,
        table_lanes: u16,
    ) -> Option<Expr> {
        let members = parse_members(insn.operands.get(1)?)?;
        let member_values = members
            .iter()
            .map(|member| self.simd_operand_value(member, DOUBLEWORD_BITS))
            .collect::<Option<Vec<_>>>()?;
        // Members concatenate low-to-high, so byte `i` of member `m`
        // sits at table byte `m * 8 + i`.
        let table = Self::concat_lanes(member_values)?;
        let indices = self.simd_operand_value(&insn.operands.get(2)?.clone(), DOUBLEWORD_BITS)?;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let prior = if keep {
            Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?)
        } else {
            None
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let selector = Self::extract_lane(indices.clone(), BITS_PER_BYTE, index)?;
            let mut value = match prior.as_ref() {
                Some(destination) => Self::extract_lane(destination.clone(), BITS_PER_BYTE, index)?,
                None => Expr::konst(0, BITS_PER_BYTE),
            };
            for entry in 0..table_lanes {
                let byte = Self::extract_lane(table.clone(), BITS_PER_BYTE, entry)?;
                value = Expr::Ite {
                    cond: Box::new(Expr::eq(
                        selector.clone(),
                        Expr::konst(u128::from(entry), BITS_PER_BYTE),
                    )),
                    then_expr: Box::new(byte),
                    else_expr: Box::new(value),
                };
            }
            lanes.push(value);
        }
        Self::concat_lanes(lanes)
    }
}
