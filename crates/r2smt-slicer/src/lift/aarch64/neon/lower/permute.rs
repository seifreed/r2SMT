//! Lowering for the `AArch64` NEON families that move bits without
//! computing on them.
//!
//! `movi` / `mvni`, `dup`, `ext`, the permutations, `umov` / `smov` /
//! `ins`, `tbl` / `tbx`, and the bitwise selects.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand};

use super::unsigned_max;
use crate::lift::LiftCtx;
use crate::lift::aarch64::neon::NeonShape;
use crate::lift::aarch64::neon::geometry::BITS_PER_BYTE;
use crate::lift::aarch64::neon::permute::{
    PermuteKind, PermuteSource, SelectRole, table_registers,
};

impl LiftCtx {
    /// `tbl` / `tbx` — each destination byte selected from the table by
    /// the byte at the same position of the index vector.
    ///
    /// The chain is built with the out-of-range answer at its base and
    /// the table entries layered over it, so an index no entry matches
    /// falls through to that base: zero for `tbl`, the destination's
    /// prior byte for `tbx`. The resolver caps how long the chain may
    /// get, since it is `destination bytes * table bytes` long.
    pub(super) fn table_lookup_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        keep: bool,
        table_lanes: u16,
    ) -> Option<Expr> {
        // Each table register is a full 128-bit view; their bytes
        // concatenate low-to-high, so byte `i` of member `m` is table
        // byte `m * 16 + i`.
        const TABLE_REGISTER_BITS: u16 = 128;
        let members = table_registers(insn.operands.get(1)?)?;
        let member_values = members
            .iter()
            .map(|operand| self.simd_operand_value(operand, TABLE_REGISTER_BITS))
            .collect::<Option<Vec<_>>>()?;
        let table = Self::concat_lanes(member_values)?;
        let indices = self.widen_source(insn, 2)?;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let prior = if keep {
            Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?)
        } else {
            None
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let selector = LiftCtx::extract_lane(indices.clone(), BITS_PER_BYTE, index)?;
            let mut value = match prior.as_ref() {
                Some(destination) => {
                    LiftCtx::extract_lane(destination.clone(), BITS_PER_BYTE, index)?
                }
                None => Expr::konst(0, BITS_PER_BYTE),
            };
            for entry in 0..table_lanes {
                let byte = LiftCtx::extract_lane(table.clone(), BITS_PER_BYTE, entry)?;
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

    /// `bsl` / `bit` / `bif` — one bitwise select over the whole view.
    ///
    /// Bit-granular, so there is nothing to do per lane: the mask, the
    /// selected value and the alternative are each read whole and
    /// combined once.
    pub(super) fn bitwise_select(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        role: SelectRole,
    ) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        // `(selected & mask) | (other & ~mask)` in every case; only the
        // assignment of the three registers to those roles differs.
        let (mask, selected, other) = match role {
            SelectRole::DestinationIsMask => (destination, first, second),
            SelectRole::InsertWhereSet => (second, first, destination),
            SelectRole::InsertWhereClear => (second, destination, first),
        };
        let ones = all_ones(view)?;
        Some(Expr::bv_or(
            Expr::bv_and(selected, mask.clone()),
            Expr::bv_and(other, Expr::bv_xor(mask, ones)),
        ))
    }

    /// `dup` — one value replicated to every destination lane.
    pub(super) fn duplicate_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        from_element: bool,
    ) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        let element = if from_element {
            self.read_simd_lane_bits(&source, shape.lane_bits, shape.source_index)?
        } else {
            // The low `lane_bits` of a general-purpose register.
            self.read_general_element(&source, shape.lane_bits)?
        };
        Self::concat_lanes(vec![element; usize::from(shape.lanes)])
    }

    /// `ext` — a window over `vm:vn` starting `byte_offset` bytes up,
    /// with `vn` at the low end of the concatenation.
    pub(super) fn extract_window(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        byte_offset: u16,
    ) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let low = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let high = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let pair = Expr::concat(high, low);
        let lo = byte_offset.checked_mul(BITS_PER_BYTE)?;
        let hi = lo.checked_add(view)?.checked_sub(1)?;
        Some(Expr::extract(pair, hi, lo))
    }

    /// The permutation family — every destination lane is some source
    /// lane, so one loop over [`permuted_source`] covers all of them.
    pub(super) fn permute_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: PermuteKind,
    ) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = match insn.operands.get(2) {
            Some(operand) => Some(self.simd_operand_value(&operand.clone(), view)?),
            None => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let (which, source_lane) = permuted_source(kind, index, shape)?;
            let value = match which {
                PermuteSource::First => first.clone(),
                PermuteSource::Second => second.clone()?,
            };
            lanes.push(Self::extract_lane(value, shape.lane_bits, source_lane)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `umov` / `smov` — one element widened into a general-purpose
    /// register.
    pub(super) fn element_to_gpr(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
    ) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        let element = self.read_simd_lane_bits(&source, shape.lane_bits, shape.source_index)?;
        let destination_bits = self.operand_width(insn.operands.first()?);
        Some(match destination_bits.cmp(&shape.lane_bits) {
            std::cmp::Ordering::Equal => element,
            std::cmp::Ordering::Greater if signed => Expr::sign_ext(element, destination_bits),
            std::cmp::Ordering::Greater => Expr::zero_ext(element, destination_bits),
            std::cmp::Ordering::Less => return None,
        })
    }

    /// `ins` — the value written into one destination lane.
    pub(super) fn insert_source(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        from_element: bool,
    ) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        if from_element {
            return self.read_simd_lane_bits(&source, shape.lane_bits, shape.source_index);
        }
        self.read_general_element(&source, shape.lane_bits)
    }

    /// The low `lane_bits` of a general-purpose register operand, as the
    /// element `dup` replicates and `ins` writes.
    fn read_general_element(&self, op: &Operand, lane_bits: u16) -> Option<Expr> {
        let whole = self.read_register(op)?;
        let natural = self.operand_width(op);
        Some(match natural.cmp(&lane_bits) {
            std::cmp::Ordering::Equal => whole,
            std::cmp::Ordering::Greater => Expr::extract(whole, lane_bits - 1, 0),
            std::cmp::Ordering::Less => return None,
        })
    }
}

/// An all-ones bit-vector of `bits`.
fn all_ones(bits: u16) -> Option<Expr> {
    Some(Expr::konst(unsigned_max(bits)?, bits))
}

/// The replicated lane value of a `movi` / `mvni`.
pub(super) fn immediate_lanes(shape: &NeonShape, value: u64, invert: bool) -> Expr {
    let mask = if shape.lane_bits >= u16::try_from(u64::BITS).unwrap_or(u16::MAX) {
        u64::MAX
    } else {
        (1u64 << shape.lane_bits) - 1
    };
    let lane = if invert { !value } else { value } & mask;
    let element = Expr::konst(u128::from(lane), shape.lane_bits);
    LiftCtx::concat_lanes(vec![element; usize::from(shape.lanes)])
        .unwrap_or_else(|| Expr::konst(u128::from(lane), shape.lane_bits))
}

/// Which source lane feeds destination lane `index`.
///
/// - `zip1` / `zip2` interleave one half of each source: destination
///   lane `2i` takes source-one lane `i` (plus half the lane count for
///   `zip2`), lane `2i+1` takes source two's.
/// - `uzp1` / `uzp2` deinterleave: the low half of the destination walks
///   the even (or odd) lanes of source one, the high half those of
///   source two.
/// - `trn1` / `trn2` transpose: even destination lanes take even (or
///   odd) lanes of source one, odd ones the same lanes of source two.
/// - `rev` reverses element order inside each container, which is a
///   single-source permutation.
fn permuted_source(
    kind: PermuteKind,
    index: u16,
    shape: NeonShape,
) -> Option<(PermuteSource, u16)> {
    let lanes = shape.lanes;
    let half = lanes / 2;
    Some(match kind {
        PermuteKind::Zip { upper } => {
            let base = if upper { half } else { 0 };
            let pair = index / 2;
            if index.is_multiple_of(2) {
                (PermuteSource::First, base.checked_add(pair)?)
            } else {
                (PermuteSource::Second, base.checked_add(pair)?)
            }
        }
        PermuteKind::Uzp { odd } => {
            let offset = u16::from(odd);
            if index < half {
                (
                    PermuteSource::First,
                    index.checked_mul(2)?.checked_add(offset)?,
                )
            } else {
                let local = index - half;
                (
                    PermuteSource::Second,
                    local.checked_mul(2)?.checked_add(offset)?,
                )
            }
        }
        PermuteKind::Trn { odd } => {
            let pair = index / 2;
            let source_lane = pair.checked_mul(2)?.checked_add(u16::from(odd))?;
            if index.is_multiple_of(2) {
                (PermuteSource::First, source_lane)
            } else {
                (PermuteSource::Second, source_lane)
            }
        }
        PermuteKind::Reverse { container_bits } => {
            let per_container = container_bits / shape.lane_bits;
            let container = index / per_container;
            let within = index % per_container;
            let source_lane = container
                .checked_mul(per_container)?
                .checked_add(per_container - 1 - within)?;
            (PermuteSource::First, source_lane)
        }
    })
}
