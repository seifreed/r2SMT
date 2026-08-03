//! Lowering for the `AArch64` NEON families.
//!
//! The sibling module answers *what* an instruction is; this one builds
//! the expression. Splitting them keeps each file to one question, and
//! is what the resolver's promise depends on: a shape that resolves must
//! have a lowering here, or the effect table and the lifter disagree
//! about which instructions the slicer may retain.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand};

use super::super::super::{FpArithOp, LiftCtx, fp_lane_result, fp_propagating_max_min};
use super::arith::{CompareKind, SaturateTo, SaturatingKind, ShiftKind};
use super::geometry::{BITS_PER_BYTE, dot_product_element, operand_arrangement};
use super::multiply::{AccumulateKind, AccumulateSources, ByElementKind, DOT_PRODUCT_TERMS};
use super::permute::{PermuteKind, PermuteSource, SelectRole, table_registers};
use super::width::{ConvertKind, ReduceKind, WidenKind};
use super::{NeonOp, NeonShape};

impl LiftCtx {
    /// Lower a resolved NEON instruction.
    ///
    /// A decline here is sound by the free-input boundary: the slicer
    /// consumed the destination as a definition and therefore stopped
    /// tracking its upstream definitions, so emitting no assignment
    /// leaves the register a free SSA input rather than bound to a stale
    /// value.
    pub(in crate::lift::aarch64) fn lift_neon(&mut self, insn: &Instruction, shape: NeonShape) {
        if let NeonOp::Packed(op) = shape.op {
            // An `AArch64` SIMD write has no merging form, so the write
            // zeroes the register above the arrangement's view.
            self.lift_packed_vector(insn, op, shape.lane_bits, true);
            return;
        }
        let Some(destination) = insn.operands.first().cloned() else {
            self.push_neon_unsupported(insn);
            return;
        };
        let Some(value) = self.neon_value(insn, shape) else {
            self.push_neon_unsupported(insn);
            return;
        };
        let written = if shape.writes_gpr() {
            self.write_register_to(&destination, value)
        } else if matches!(shape.op, NeonOp::Insert { .. }) {
            self.write_simd_lane(&destination, value, shape.lane_bits, shape.dest_index)
        } else {
            self.write_simd_dst(&destination, value, true)
        };
        if !written {
            self.push_neon_unsupported(insn);
        }
    }

    /// Build the value a resolved NEON instruction writes.
    fn neon_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        match shape.op {
            // Handled by the caller; never reaches here.
            NeonOp::Packed(_) => None,
            NeonOp::Immediate { value, invert } => Some(immediate_lanes(&shape, value, invert)),
            NeonOp::Duplicate { from_element } => self.duplicate_lanes(insn, shape, from_element),
            NeonOp::Extract { byte_offset } => self.extract_window(insn, shape, byte_offset),
            NeonOp::Permute(kind) => self.permute_lanes(insn, shape, kind),
            NeonOp::ElementToGpr { signed } => self.element_to_gpr(insn, shape, signed),
            NeonOp::Insert { from_element } => self.insert_source(insn, shape, from_element),
            NeonOp::Widen {
                kind,
                signed,
                upper,
            } => self.widen_lanes(insn, shape, kind, signed, upper),
            NeonOp::MultiplyAccumulate(kind) => self.multiply_accumulate_lanes(insn, shape, kind),
            NeonOp::Saturating {
                kind,
                signed_sources,
                upper,
            } => self.saturating_lanes(insn, shape, kind, signed_sources, upper),
            NeonOp::Shift { kind, signed } => self.shift_lanes(insn, shape, kind, signed),
            NeonOp::Compare { kind, zero } => self.compare_lanes(insn, shape, kind, zero),
            NeonOp::Convert { kind, upper, fbits } => {
                self.convert_lanes(insn, shape, kind, upper, fbits)
            }
            NeonOp::BitwiseSelect(role) => self.bitwise_select(insn, shape, role),
            NeonOp::Reduce {
                kind,
                source_lanes,
                source_lane_bits,
            } => self.reduce_lanes(insn, shape, kind, source_lanes, source_lane_bits),
            NeonOp::ByElement { kind, upper } => self.by_element_lanes(insn, shape, kind, upper),
            NeonOp::DotProduct { signed, by_element } => {
                self.dot_product_lanes(insn, shape, signed, by_element)
            }
            NeonOp::TableLookup { keep, table_lanes } => {
                self.table_lookup_lanes(insn, shape, keep, table_lanes)
            }
            NeonOp::PolynomialMultiply { upper } => {
                self.polynomial_multiply_lanes(insn, shape, upper)
            }
            NeonOp::Estimate => self.estimate_value(insn, shape),
        }
    }

    /// `frecpe` / `frsqrte` — a fresh value that is never assigned.
    ///
    /// The architecture guarantees only a relative error bound for
    /// these and leaves the value itself implementation-defined, so
    /// `FDiv(1.0, x)` would not be an approximation of the result: it
    /// would be a *definite* number the machine is not required to
    /// produce, which is the fabrication this pipeline exists to avoid.
    ///
    /// Emitting nothing is not the alternative either. The slicer has
    /// already consumed the destination as a definition, so a decline
    /// truncates the slice and every downstream verdict becomes
    /// `Unsound`. Assigning a temp that is never defined keeps the slice
    /// `Complete` instead: `ssa_convert` surfaces a variable read before
    /// it is written as a free input, so the solver considers every
    /// value the estimate could take and an `AlwaysTrue` verdict is
    /// still one that holds for all of them.
    ///
    /// The temp is fresh per instruction, which loses one true fact —
    /// two `frecpe`s over the same input agree, since the estimate is a
    /// function. That is a widening, not an unsoundness, and recovering
    /// it would mean naming the value by its operand rather than by its
    /// address.
    fn estimate_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        Some(Expr::Var(self.new_temp(insn.address, view)))
    }

    /// `tbl` / `tbx` — each destination byte selected from the table by
    /// the byte at the same position of the index vector.
    ///
    /// The chain is built with the out-of-range answer at its base and
    /// the table entries layered over it, so an index no entry matches
    /// falls through to that base: zero for `tbl`, the destination's
    /// prior byte for `tbx`. The resolver caps how long the chain may
    /// get, since it is `destination bytes * table bytes` long.
    fn table_lookup_lanes(
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

    /// `pmull` / `pmull2` — the carry-less product of two narrow
    /// elements, which is the `XOR` of the multiplicand shifted to
    /// every set bit of the multiplier.
    ///
    /// Carry-less is the whole point: an ordinary multiply is the same
    /// sum of shifted copies but with carries propagating between them,
    /// so `Expr::mul` is not an approximation of this, it is a
    /// different function. The product of two `n`-bit elements needs
    /// `2n - 1` bits, so the destination element always holds it.
    fn polynomial_multiply_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        upper: bool,
    ) -> Option<Expr> {
        let narrow = shape.lane_bits / 2;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let multiplicand = Expr::zero_ext(
                LiftCtx::extract_lane(first.clone(), narrow, source_lane)?,
                shape.lane_bits,
            );
            let multiplier = LiftCtx::extract_lane(second.clone(), narrow, source_lane)?;
            let mut product = Expr::konst(0, shape.lane_bits);
            for bit in 0..narrow {
                let shifted = Expr::shl(
                    multiplicand.clone(),
                    Expr::konst(u128::from(bit), shape.lane_bits),
                );
                product = Expr::bv_xor(
                    product,
                    Expr::Ite {
                        cond: Box::new(Expr::eq(
                            Expr::extract(multiplier.clone(), bit, bit),
                            Expr::konst(1, 1),
                        )),
                        then_expr: Box::new(shifted),
                        else_expr: Box::new(Expr::konst(0, shape.lane_bits)),
                    },
                );
            }
            lanes.push(product);
        }
        Self::concat_lanes(lanes)
    }

    /// The by-element multiplies — one source element multiplied into
    /// every destination lane.
    ///
    /// The element is read once, outside the loop. Reading it per lane
    /// would emit the same extract `lanes` times over, and for a memory
    /// operand it would be a load per lane.
    fn by_element_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ByElementKind,
        upper: bool,
    ) -> Option<Expr> {
        let source_bits = if kind.widens() {
            shape.lane_bits / 2
        } else {
            shape.lane_bits
        };
        let first = self.widen_source(insn, 1)?;
        let element = self.read_simd_lane_bits(
            &insn.operands.get(2)?.clone(),
            source_bits,
            shape.source_index,
        )?;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = match kind.combine() {
            Some(_) => Some(self.simd_operand_value(&insn.operands.first()?.clone(), view)?),
            None => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            // A `2` form reads the first source's upper half; the
            // element operand is indexed absolutely and is never halved.
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let a = LiftCtx::extract_lane(first.clone(), source_bits, source_lane)?;
            let product = match kind {
                ByElementKind::Long { signed, .. } => Expr::mul(
                    extend(a, shape.lane_bits, signed),
                    extend(element.clone(), shape.lane_bits, signed),
                ),
                ByElementKind::Integer { .. } => Expr::mul(a, element.clone()),
                ByElementKind::Float => {
                    fp_lane_result(FpArithOp::Mul, a, element.clone(), shape.lane_bits)?
                }
            };
            lanes.push(match (kind.combine(), accumulator.as_ref()) {
                (Some(combine), Some(previous)) => combine.apply(
                    LiftCtx::extract_lane(previous.clone(), shape.lane_bits, index)?,
                    product,
                ),
                _ => product,
            });
        }
        Self::concat_lanes(lanes)
    }

    /// `sdot` / `udot` — four byte products summed into each 32-bit
    /// destination lane, on top of that lane's prior value.
    ///
    /// The products are formed at the destination's width rather than
    /// the byte's, which is what the architecture's extension of each
    /// element before the multiply comes to; the accumulation then wraps
    /// at 32 bits exactly as ARM defines it.
    fn dot_product_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
        by_element: bool,
    ) -> Option<Expr> {
        // Each source register is a full 128-bit view; a by-element
        // source re-spells its group selector to the bare register.
        const REGISTER_BITS: u16 = 128;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.widen_source(insn, 1)?;
        let second = if by_element {
            let (register, _) = dot_product_element(insn.operands.get(2)?)?;
            self.simd_operand_value(&register, REGISTER_BITS)?
        } else {
            self.widen_source(insn, 2)?
        };
        let read = |value: &Expr, byte: u16| -> Option<Expr> {
            let raw = LiftCtx::extract_lane(value.clone(), BITS_PER_BYTE, byte)?;
            Some(extend(raw, shape.lane_bits, signed))
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let mut sum = LiftCtx::extract_lane(accumulator.clone(), shape.lane_bits, index)?;
            for term in 0..DOT_PRODUCT_TERMS {
                let first_byte = index.checked_mul(DOT_PRODUCT_TERMS)?.checked_add(term)?;
                // By-element broadcasts one group to every lane; the
                // whole-vector form pairs each lane with its own.
                let second_group = if by_element {
                    shape.source_index
                } else {
                    index
                };
                let second_byte = second_group
                    .checked_mul(DOT_PRODUCT_TERMS)?
                    .checked_add(term)?;
                sum = Expr::add(
                    sum,
                    Expr::mul(read(&first, first_byte)?, read(&second, second_byte)?),
                );
            }
            lanes.push(sum);
        }
        Self::concat_lanes(lanes)
    }

    /// The across-lane reductions — every source lane folded into the
    /// one element the scalar destination holds.
    ///
    /// The fold is left-associative, which the architecture permits to
    /// matter for none of these: integer addition and the min / max
    /// selects are associative, so no ordering is observable.
    fn reduce_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ReduceKind,
        source_lanes: u16,
        source_lane_bits: u16,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let mut accumulator: Option<Expr> = None;
        for index in 0..source_lanes {
            let raw = LiftCtx::extract_lane(source.clone(), source_lane_bits, index)?;
            // The widening forms extend before the fold; the same-width
            // ones already sit at the destination's element width.
            let element = match kind {
                ReduceKind::AddLong { signed } => extend(raw, shape.lane_bits, signed),
                ReduceKind::Add | ReduceKind::MinMax { .. } | ReduceKind::Float { .. } => raw,
            };
            accumulator = Some(match accumulator {
                None => element,
                Some(previous) => reduce_step(kind, previous, element, shape.lane_bits)?,
            });
        }
        accumulator
    }

    /// `bsl` / `bit` / `bif` — one bitwise select over the whole view.
    ///
    /// Bit-granular, so there is nothing to do per lane: the mask, the
    /// selected value and the alternative are each read whole and
    /// combined once.
    fn bitwise_select(
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

    /// The lane-wise compares, each writing an all-ones or all-zeros
    /// mask per lane.
    fn compare_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: CompareKind,
        zero: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = if zero {
            None
        } else {
            Some(self.widen_source(insn, 2)?)
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?;
            let b = match second.as_ref() {
                Some(value) => LiftCtx::extract_lane(value.clone(), shape.lane_bits, index)?,
                None => Expr::konst(0, shape.lane_bits),
            };
            lanes.push(compare_lane(kind, a, b, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The lane-wise conversions.
    fn convert_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ConvertKind,
        upper: bool,
        fbits: u16,
    ) -> Option<Expr> {
        let source = self.widen_source(insn, 1)?;
        let source_bits = match kind {
            ConvertKind::FloatToFloat { widening: true } => shape.lane_bits / 2,
            ConvertKind::FloatToFloat { widening: false } => shape.lane_bits.checked_mul(2)?,
            _ => shape.lane_bits,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            // `fcvtl2` reads the source's upper half; `fcvtn2` reads the
            // same lanes `fcvtn` does and moves only the destination.
            let source_lane =
                if upper && matches!(kind, ConvertKind::FloatToFloat { widening: true }) {
                    index.checked_add(shape.lanes)?
                } else {
                    index
                };
            let element = LiftCtx::extract_lane(source.clone(), source_bits, source_lane)?;
            lanes.push(convert_lane(
                kind,
                element,
                source_bits,
                shape.lane_bits,
                fbits,
            )?);
        }
        let converted = Self::concat_lanes(lanes)?;
        if !upper || !matches!(kind, ConvertKind::FloatToFloat { widening: false }) {
            return Some(converted);
        }
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            converted,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// The same-width shift family.
    fn shift_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: ShiftKind,
        signed: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let amounts = match kind {
            ShiftKind::Register { .. } => Some(self.widen_source(insn, 2)?),
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let value = LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?;
            let amount = match amounts.as_ref() {
                Some(vector) => Some(LiftCtx::extract_lane(
                    vector.clone(),
                    shape.lane_bits,
                    index,
                )?),
                None => None,
            };
            lanes.push(shift_lane(kind, signed, value, amount, shape.lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The saturating, halving and rounding-narrow family.
    ///
    /// Every member computes its lane at a width where the result cannot
    /// overflow, then brings it back down — by clamping, by halving, or
    /// by truncating. Doing the arithmetic at the destination's width
    /// and clamping afterwards would be too late: the overflow the
    /// instruction exists to detect would already have wrapped.
    fn saturating_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: SaturatingKind,
        signed_sources: bool,
        upper: bool,
    ) -> Option<Expr> {
        let narrowing = matches!(
            kind,
            SaturatingKind::Narrow { .. } | SaturatingKind::ShiftNarrow { .. }
        );
        let source_bits = if narrowing {
            shape.lane_bits.checked_mul(2)?
        } else {
            shape.lane_bits
        };
        let first = self.widen_source(insn, 1)?;
        let second = match insn.operands.get(2) {
            Some(operand) if operand_arrangement(operand).is_some() => {
                Some(self.widen_source(insn, 2)?)
            }
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let a = LiftCtx::extract_lane(first.clone(), source_bits, index)?;
            let b = match second.as_ref() {
                Some(value) => Some(LiftCtx::extract_lane(value.clone(), source_bits, index)?),
                None => None,
            };
            lanes.push(saturating_lane(
                kind,
                a,
                b,
                shape.lane_bits,
                signed_sources,
            )?);
        }
        let narrowed = Self::concat_lanes(lanes)?;
        if !upper {
            return Some(narrowed);
        }
        // A `2` form writes the destination's upper half and preserves
        // the lower one.
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            narrowed,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// The multiply-accumulate family: each destination lane is its own
    /// prior value plus or minus the product of the two source lanes.
    fn multiply_accumulate_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: AccumulateKind,
    ) -> Option<Expr> {
        let AccumulateKind {
            combine,
            sources,
            upper,
        } = kind;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let source_bits = match sources {
            AccumulateSources::SameWidth => shape.lane_bits,
            AccumulateSources::Long { .. } => shape.lane_bits / 2,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let read = |value: &Expr| -> Option<Expr> {
                let element = LiftCtx::extract_lane(value.clone(), source_bits, source_lane)?;
                Some(match sources {
                    AccumulateSources::SameWidth => element,
                    AccumulateSources::Long { signed: true } => {
                        Expr::sign_ext(element, shape.lane_bits)
                    }
                    AccumulateSources::Long { signed: false } => {
                        Expr::zero_ext(element, shape.lane_bits)
                    }
                })
            };
            let product = Expr::mul(read(&first)?, read(&second)?);
            let previous = LiftCtx::extract_lane(accumulator.clone(), shape.lane_bits, index)?;
            lanes.push(combine.apply(previous, product));
        }
        Self::concat_lanes(lanes)
    }

    /// The widening and narrowing family: read each source element,
    /// extend or truncate it to the destination's width, and operate
    /// there.
    ///
    /// Extending *before* operating is the whole point of the family —
    /// it is what stops the result overflowing the element, which is
    /// exactly what the same-width lane-wise form would do.
    fn widen_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: WidenKind,
        signed: bool,
        upper: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = match insn.operands.get(2) {
            Some(operand) if operand_arrangement(operand).is_some() => {
                Some(self.widen_source(insn, 2)?)
            }
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            lanes.push(Self::widen_lane(
                &first,
                second.as_ref(),
                shape,
                kind,
                signed,
                index,
                upper,
            )?);
        }
        let narrowed = Self::concat_lanes(lanes)?;
        if !matches!(kind, WidenKind::Narrow) || !upper {
            return Some(narrowed);
        }
        // `xtn2` writes the destination's upper half and preserves the
        // lower one.
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            narrowed,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// One source operand, materialised once at its own view width.
    fn widen_source(&mut self, insn: &Instruction, position: usize) -> Option<Expr> {
        let operand = insn.operands.get(position)?.clone();
        let arrangement = operand_arrangement(&operand)?;
        self.simd_operand_value(&operand, arrangement.view_bits())
    }

    /// One destination lane of a widening or narrowing operation.
    ///
    /// `upper` shifts the lane read out of a *narrow* source by half a
    /// register, which is what the `2` suffix means. A wide source is
    /// never halved — `uaddw2 v0.4s, v1.4s, v2.8h` takes `v1`'s lane `i`
    /// and `v2`'s lane `i + 4`.
    fn widen_lane(
        first: &Expr,
        second: Option<&Expr>,
        shape: NeonShape,
        kind: WidenKind,
        signed: bool,
        index: u16,
        upper: bool,
    ) -> Option<Expr> {
        let extend = |value: Expr| {
            if signed {
                Expr::sign_ext(value, shape.lane_bits)
            } else {
                Expr::zero_ext(value, shape.lane_bits)
            }
        };
        let narrow_lane = if upper {
            index.checked_add(shape.lanes)?
        } else {
            index
        };
        let narrow = shape.lane_bits / 2;
        match kind {
            WidenKind::Narrow => {
                // Truncation needs no signedness: the low half of a
                // two's-complement value is the same bits either way.
                // The source is wide, so it is never halved — `xtn2`
                // narrows the same lanes `xtn` does and only the
                // *destination* half differs.
                let wide = shape.lane_bits.checked_mul(2)?;
                let element = LiftCtx::extract_lane(first.clone(), wide, index)?;
                Some(Expr::extract(element, shape.lane_bits - 1, 0))
            }
            WidenKind::ShiftLong { shift } => {
                let element = LiftCtx::extract_lane(first.clone(), narrow, narrow_lane)?;
                let widened = extend(element);
                if shift == 0 {
                    return Some(widened);
                }
                Some(Expr::shl(
                    widened,
                    Expr::konst(u128::from(shift), shape.lane_bits),
                ))
            }
            WidenKind::Arith { op, wide_first } => {
                // A `w`-form's first source is already at the
                // destination's width and is read at its own lane index.
                let lhs = if wide_first {
                    LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?
                } else {
                    extend(LiftCtx::extract_lane(first.clone(), narrow, narrow_lane)?)
                };
                let rhs = extend(LiftCtx::extract_lane(second?.clone(), narrow, narrow_lane)?);
                Some(op.apply(lhs, rhs))
            }
        }
    }

    /// `dup` — one value replicated to every destination lane.
    fn duplicate_lanes(
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
    fn extract_window(
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
    fn permute_lanes(
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
    fn element_to_gpr(
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
    fn insert_source(
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

    fn push_neon_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(r2smt_ir::stmt::IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable NEON operand at {addr}", addr = insn.address),
        });
    }
}

/// Fold one more lane into an across-lane reduction's accumulator.
///
/// `addv` sums at the element width and keeps the low bits, which is
/// what the architecture's truncation of the unbounded sum comes to; a
/// wrapping add reproduces it exactly. `uaddlv` / `saddlv` arrive here
/// already extended, and no encodable arrangement can overflow the
/// doubled width (sixteen bytes reach 4 080, eight halfwords 524 280),
/// so their sum is exact rather than truncated too.
/// `fmaxv` / `fminv` fold with [`fp_propagating_max_min`] and *not*
/// with the integer comparison beside it, nor with
/// [`super::super::super::fp_lane_result`]'s `Max` / `Min`, which is
/// Intel's `MAXPS`: ARM's `FPMax` propagates NaN where `MAXPS` returns
/// its second operand, and combines the signs of a zero tie where
/// `MAXPS` again takes the second. Both differences are wrong values,
/// not wider ones.
fn reduce_step(kind: ReduceKind, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    Some(match kind {
        ReduceKind::Add | ReduceKind::AddLong { .. } => Expr::add(a, b),
        ReduceKind::Float { max, number_wins } => {
            return fp_propagating_max_min(a, b, lane_bits, max, number_wins);
        }
        ReduceKind::MinMax { signed, max } => {
            let cond = if signed {
                Expr::slt(a.clone(), b.clone())
            } else {
                Expr::ult(a.clone(), b.clone())
            };
            let (taken, other) = if max { (b, a) } else { (a, b) };
            Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(taken),
                else_expr: Box::new(other),
            }
        }
    })
}

/// Extend `value` from `from` bits to `to` bits.
fn extend(value: Expr, to: u16, signed: bool) -> Expr {
    if signed {
        Expr::sign_ext(value, to)
    } else {
        Expr::zero_ext(value, to)
    }
}

/// The largest value of a `bits`-wide unsigned range, or `None` when it
/// does not fit the constant type.
///
/// A full 128-bit vector view is representable and is the width the
/// bitwise selects mask at, so only *wider* than the constant type
/// declines.
fn unsigned_max(bits: u16) -> Option<u128> {
    match bits {
        0 => None,
        128 => Some(u128::MAX),
        bits if bits < 128 => Some((1u128 << bits) - 1),
        _ => None,
    }
}

/// Clamp `value`, computed at `wide` bits, into the `narrow`-bit range
/// `to` names, returning a `narrow`-bit result.
///
/// The comparison's signedness follows the *value*, not the target: an
/// unsigned sum is compared unsigned, while `uqsub`'s difference can be
/// negative and so is compared signed even though it clamps into the
/// unsigned range.
fn clamp(value: Expr, wide: u16, narrow: u16, to: SaturateTo) -> Option<Expr> {
    let konst = |v: u128| Expr::konst(v, wide);
    let clamped = match to {
        SaturateTo::Signed => {
            let magnitude = 1u128.checked_shl(u32::from(narrow.checked_sub(1)?))?;
            let high = konst(magnitude - 1);
            // The lower bound is `-2^(narrow-1)` in `wide`-bit two's
            // complement.
            let low = konst(unsigned_max(wide)?.checked_add(1)?.checked_sub(magnitude)?);
            let above = Expr::Ite {
                cond: Box::new(Expr::slt(high.clone(), value.clone())),
                then_expr: Box::new(high),
                else_expr: Box::new(value.clone()),
            };
            Expr::Ite {
                cond: Box::new(Expr::slt(value, low.clone())),
                then_expr: Box::new(low),
                else_expr: Box::new(above),
            }
        }
        SaturateTo::Unsigned => {
            let high = konst(unsigned_max(narrow)?);
            Expr::Ite {
                cond: Box::new(Expr::ult(high.clone(), value.clone())),
                then_expr: Box::new(high),
                else_expr: Box::new(value),
            }
        }
        SaturateTo::SignedToUnsigned => {
            let high = konst(unsigned_max(narrow)?);
            let above = Expr::Ite {
                cond: Box::new(Expr::slt(high.clone(), value.clone())),
                then_expr: Box::new(high),
                else_expr: Box::new(value.clone()),
            };
            Expr::Ite {
                cond: Box::new(Expr::slt(value, konst(0))),
                then_expr: Box::new(konst(0)),
                else_expr: Box::new(above),
            }
        }
    };
    Some(Expr::extract(clamped, narrow - 1, 0))
}

/// One destination lane of a saturating, halving or rounding-narrow
/// operation.
///
/// `a` and `b` arrive at the *source* width, which for the narrowing
/// members is twice `lane_bits`.
fn saturating_lane(
    kind: SaturatingKind,
    a: Expr,
    b: Option<Expr>,
    lane_bits: u16,
    signed_sources: bool,
) -> Option<Expr> {
    match kind {
        SaturatingKind::AddSub { op, to } => {
            // One extra bit makes the sum or difference exact, so the
            // clamp sees the true value rather than a wrapped one.
            let wide = lane_bits.checked_add(1)?;
            let value = op.apply(
                extend(a, wide, signed_sources),
                extend(b?, wide, signed_sources),
            );
            clamp(value, wide, lane_bits, to)
        }
        SaturatingKind::Halving { rounding } => {
            let wide = lane_bits.checked_add(1)?;
            let mut sum = Expr::add(
                extend(a, wide, signed_sources),
                extend(b?, wide, signed_sources),
            );
            if rounding {
                sum = Expr::add(sum, Expr::konst(1, wide));
            }
            // The exact sum needs no clamp — halving it always fits.
            let halved = if signed_sources {
                Expr::ashr(sum, Expr::konst(1, wide))
            } else {
                Expr::lshr(sum, Expr::konst(1, wide))
            };
            Some(Expr::extract(halved, lane_bits - 1, 0))
        }
        SaturatingKind::Narrow { to } => clamp(a, lane_bits.checked_mul(2)?, lane_bits, to),
        SaturatingKind::ShiftNarrow {
            shift,
            rounding,
            to,
        } => {
            // One bit above the source width. The rounding term is added
            // *before* the shift, and at the source's own width that
            // addition can carry into the sign bit — `0x7fff + 8` in
            // sixteen bits is negative, which would turn a saturation at
            // the top of the range into one at the bottom. ARM defines
            // the rounding on the unbounded integer, and this is the
            // narrowest width that reproduces it.
            let wide = lane_bits.checked_mul(2)?.checked_add(1)?;
            let mut value = extend(a, wide, signed_sources);
            if rounding {
                // Half an ulp of the shift, added before the low bits
                // are discarded.
                let half = 1u128.checked_shl(u32::from(shift - 1))?;
                value = Expr::add(value, Expr::konst(half, wide));
            }
            let shifted = if signed_sources {
                Expr::ashr(value, Expr::konst(u128::from(shift), wide))
            } else {
                Expr::lshr(value, Expr::konst(u128::from(shift), wide))
            };
            match to {
                Some(target) => clamp(shifted, wide, lane_bits, target),
                // `shrn` / `rshrn` truncate: with the shift bounded by
                // the destination's element width, every surviving bit
                // is a real bit of the source rather than fill.
                None => Some(Expr::extract(shifted, lane_bits - 1, 0)),
            }
        }
        SaturatingKind::DoublingMultiplyHigh { rounding } => {
            // The product needs `2 * lane_bits`, and doubling it needs
            // one more — which is exactly the `INT_MIN * INT_MIN` corner
            // where this instruction saturates.
            let wide = lane_bits.checked_mul(2)?.checked_add(1)?;
            let product = Expr::mul(extend(a, wide, true), extend(b?, wide, true));
            let mut doubled = Expr::shl(product, Expr::konst(1, wide));
            if rounding {
                let half = 1u128.checked_shl(u32::from(lane_bits - 1))?;
                doubled = Expr::add(doubled, Expr::konst(half, wide));
            }
            let high = Expr::ashr(doubled, Expr::konst(u128::from(lane_bits), wide));
            clamp(high, wide, lane_bits, SaturateTo::Signed)
        }
    }
}

/// Shift `value` right by `amount`, rounding when asked.
///
/// Rounding adds half an ulp of the shift *before* the low bits are
/// discarded, and at the element's own width that addition can carry out
/// — `0xff + 1` in eight bits is zero. The sum is therefore taken one
/// bit wider, which is the narrowest width that reproduces ARM's
/// unbounded-integer definition.
fn shift_right(
    value: Expr,
    amount: &Expr,
    lane_bits: u16,
    signed: bool,
    rounding: bool,
) -> Option<Expr> {
    let shift_by = |v: Expr, by: Expr| {
        if signed {
            Expr::ashr(v, by)
        } else {
            Expr::lshr(v, by)
        }
    };
    if !rounding {
        return Some(shift_by(value, amount.clone()));
    }
    let wide = lane_bits.checked_add(1)?;
    let wide_amount = Expr::zero_ext(amount.clone(), wide);
    let half = Expr::shl(
        Expr::konst(1, wide),
        Expr::sub(wide_amount.clone(), Expr::konst(1, wide)),
    );
    let sum = Expr::add(extend(value, wide, signed), half);
    Some(Expr::extract(shift_by(sum, wide_amount), lane_bits - 1, 0))
}

/// One destination lane of a same-width shift.
///
/// The register forms read a *signed* per-lane amount whose sign chooses
/// the direction, so both directions are built and an `Ite` selects
/// between them. An out-of-range amount needs no special case: a
/// bit-vector shift by more than the width already yields zero, or all
/// sign bits for an arithmetic right shift, which is what the
/// architecture specifies.
fn shift_lane(
    kind: ShiftKind,
    signed: bool,
    value: Expr,
    amount: Option<Expr>,
    lane_bits: u16,
) -> Option<Expr> {
    match kind {
        ShiftKind::LeftImmediate { shift } => {
            Some(Expr::shl(value, Expr::konst(u128::from(shift), lane_bits)))
        }
        ShiftKind::RightImmediate { shift, rounding } => shift_right(
            value,
            &Expr::konst(u128::from(shift), lane_bits),
            lane_bits,
            signed,
            rounding,
        ),
        ShiftKind::Register { rounding } => {
            // Only the low byte of the amount element is read, as a
            // signed value.
            let raw = amount?;
            let signed_amount = if lane_bits > BITS_PER_BYTE {
                Expr::sign_ext(Expr::extract(raw, BITS_PER_BYTE - 1, 0), lane_bits)
            } else {
                raw
            };
            let left = Expr::shl(value.clone(), signed_amount.clone());
            let negated = Expr::sub(Expr::konst(0, lane_bits), signed_amount.clone());
            let right = shift_right(value, &negated, lane_bits, signed, rounding)?;
            Some(Expr::Ite {
                cond: Box::new(Expr::slt(signed_amount, Expr::konst(0, lane_bits))),
                then_expr: Box::new(right),
                else_expr: Box::new(left),
            })
        }
    }
}

/// One destination lane of a compare: an all-ones mask where the
/// predicate holds and all-zeros where it does not.
///
/// A vector compare produces a *value*, not a flag, because its result
/// feeds another vector operation — which is why this cannot reuse the
/// scalar compare path that writes NZCV.
fn compare_lane(kind: CompareKind, a: Expr, b: Expr, lane_bits: u16) -> Option<Expr> {
    let predicate = match kind {
        CompareKind::Equal { float: false } => Expr::eq(a, b),
        CompareKind::Equal { float: true } => {
            let (ebits, sbits) = fp_sort(lane_bits)?;
            Expr::feq(
                Expr::bv_to_fp(a, ebits, sbits),
                Expr::bv_to_fp(b, ebits, sbits),
            )
        }
        CompareKind::Ordered {
            float: true,
            or_equal,
            ..
        } => {
            let (ebits, sbits) = fp_sort(lane_bits)?;
            let (x, y) = (
                Expr::bv_to_fp(a, ebits, sbits),
                Expr::bv_to_fp(b, ebits, sbits),
            );
            // `fcmgt`/`fcmge` are *ordered*: unordered compares false,
            // which `fp.lt` / `fp.leq` already give.
            if or_equal {
                Expr::fle(y, x)
            } else {
                Expr::flt(y, x)
            }
        }
        CompareKind::Ordered {
            float: false,
            signed,
            or_equal,
        } => match (signed, or_equal) {
            (true, false) => Expr::slt(b, a),
            (true, true) => Expr::sle(b, a),
            (false, false) => Expr::ult(b, a),
            (false, true) => Expr::ule(b, a),
        },
        CompareKind::TestBits => Expr::ne(Expr::bv_and(a, b), Expr::konst(0, lane_bits)),
    };
    Some(Expr::Ite {
        cond: Box::new(predicate),
        then_expr: Box::new(all_ones(lane_bits)?),
        else_expr: Box::new(Expr::konst(0, lane_bits)),
    })
}

/// An all-ones bit-vector of `bits`.
fn all_ones(bits: u16) -> Option<Expr> {
    Some(Expr::konst(unsigned_max(bits)?, bits))
}

/// The IEEE `(ebits, sbits)` pair for a float lane width.
fn fp_sort(lane_bits: u16) -> Option<(u16, u16)> {
    match lane_bits {
        16 => Some((5, 11)),
        32 => Some((8, 24)),
        64 => Some((11, 53)),
        _ => None,
    }
}

/// The IEEE bit pattern of `2^exponent` in the `(ebits, sbits)` sort,
/// or `None` when that power is not representable there.
///
/// A subnormal result is built rather than declined: the fixed-point
/// conversions scale by `2^-fbits`, and for a 16-bit element with
/// `fbits = 16` that lands one exponent below the normal range — where
/// the value is still exact, since a power of two is a single set bit
/// of the stored significand.
fn power_of_two(exponent: i32, ebits: u16, sbits: u16) -> Option<u128> {
    let bias = (1i32 << (i32::from(ebits) - 1)) - 1;
    let stored = i32::from(sbits) - 1;
    let field = exponent.checked_add(bias)?;
    if field >= 1 {
        // Normal: the significand is implicit, so the pattern is the
        // biased exponent alone. The all-ones field is infinity / NaN.
        if field >= (1i32 << i32::from(ebits)) - 1 {
            return None;
        }
        return Some(u128::try_from(field).ok()? << u32::try_from(stored).ok()?);
    }
    // Subnormal: the value is `significand * 2^(1 - bias - stored)`, so
    // the set bit sits `exponent - (1 - bias - stored)` places up.
    let bit = exponent.checked_sub(1 - bias - stored)?;
    if bit < 0 || bit >= stored {
        return None;
    }
    Some(1u128 << u32::try_from(bit).ok()?)
}

/// Scale `value` by `2^-fbits`, the fixed-point conversions' fraction
/// width, in whichever direction `divide` names.
///
/// The constant is always built as `2^-fbits` and never as `2^fbits`,
/// which is the only spelling representable at every encodable width:
/// `2^16` is infinity in binary16, while `2^-16` is a perfectly good
/// subnormal. Scaling by a power of two only shifts the exponent, so it
/// introduces no rounding of its own — the result is the same one the
/// architecture's single correctly-rounded conversion produces, except
/// in a subnormal corner where the integer is small enough to be exact
/// anyway.
fn scale_by_fraction(
    value: Expr,
    fbits: u16,
    ebits: u16,
    sbits: u16,
    divide: bool,
) -> Option<Expr> {
    if fbits == 0 {
        return Some(value);
    }
    let scale = Expr::FpConst {
        bits: power_of_two(-i32::from(fbits), ebits, sbits)?,
        ebits,
        sbits,
    };
    let round = r2smt_ir::expr::RoundingMode::NearestTiesEven;
    Some(if divide {
        Expr::fdiv(value, scale, round)
    } else {
        Expr::fmul(value, scale, round)
    })
}

/// One destination lane of a conversion.
///
/// The IR carries no unsigned integer conversion, so the unsigned forms
/// go through the signed node with one extra bit of range — which covers
/// the unsigned range exactly rather than approximately, the same trick
/// the scalar `ucvtf` / `fcvtzu` already use.
///
/// `fbits` is the fixed-point fraction width, zero for the plain
/// register forms. It reads the integer side as `Int(lane) / 2^fbits`,
/// so the conversion into float multiplies by that factor and the
/// conversion out of float divides by it.
fn convert_lane(
    kind: ConvertKind,
    element: Expr,
    source_bits: u16,
    lane_bits: u16,
    fbits: u16,
) -> Option<Expr> {
    let round = r2smt_ir::expr::RoundingMode::NearestTiesEven;
    match kind {
        ConvertKind::IntToFloat { signed } => {
            let (ebits, sbits) = fp_sort(lane_bits)?;
            let source = if signed {
                element
            } else {
                Expr::zero_ext(element, source_bits.checked_add(1)?)
            };
            let converted = Expr::sbv_to_fp(source, round, ebits, sbits);
            Some(Expr::fp_to_ieee_bv(scale_by_fraction(
                converted, fbits, ebits, sbits, false,
            )?))
        }
        ConvertKind::FloatToInt { signed } => {
            let (ebits, sbits) = fp_sort(source_bits)?;
            let value = scale_by_fraction(
                Expr::bv_to_fp(element, ebits, sbits),
                fbits,
                ebits,
                sbits,
                true,
            )?;
            // The `z` in the mnemonic is round-toward-zero, so no
            // control register is assumed.
            let toward_zero = r2smt_ir::expr::RoundingMode::TowardZero;
            Some(if signed {
                Expr::fp_to_sbv(value, toward_zero, lane_bits)
            } else {
                Expr::extract(
                    Expr::fp_to_sbv(value, toward_zero, lane_bits.checked_add(1)?),
                    lane_bits - 1,
                    0,
                )
            })
        }
        ConvertKind::FloatToFloat { .. } => {
            // No fixed-point form: `convert_shape` rejects a third
            // operand here, so `fbits` is zero and nothing scales.
            let (from_e, from_s) = fp_sort(source_bits)?;
            let (to_e, to_s) = fp_sort(lane_bits)?;
            Some(Expr::fp_to_ieee_bv(Expr::fp_to_fp(
                Expr::bv_to_fp(element, from_e, from_s),
                round,
                to_e,
                to_s,
            )))
        }
    }
}

/// The replicated lane value of a `movi` / `mvni`.
fn immediate_lanes(shape: &NeonShape, value: u64, invert: bool) -> Expr {
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
            if index % 2 == 0 {
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
            if index % 2 == 0 {
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
