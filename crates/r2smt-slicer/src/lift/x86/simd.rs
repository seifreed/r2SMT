//! The x86 SIMD and floating-point per-mnemonic handlers.
//!
//! Split out of the parent module when the packed-integer families took
//! it past the decomposition limit. These are methods on [`LiftCtx`]
//! like every other handler; what separates them is that they reason in
//! the vector model of [`crate::lift::simd`] — parents, views, lanes —
//! rather than in whole registers.
//!
//! The mnemonic tables and shape predicates deliberately stay in the
//! parent (`x86_simd_shape`, `is_fp_compare_mnemonic`,
//! `sse_scalar_move_lane`, `pins_rounding_mode`): they are the x86
//! *vocabulary*, consulted by the effect table and the dispatch gate as
//! well as by these handlers, and a child module can read its parent's
//! private items without any of them being widened.

use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::lift::simd::{CompareKind, FpArithOp, clamp_to_element, compare_lane};
use crate::lift::{
    LiftCtx, PackedIntOp, ShiftOp, fp_lane_result, fp_sort_bits_checked, nonzero_width,
};

use super::{
    ABSOLUTE, ADD, AVERAGE, BITS_PER_BYTE, F16C_HALF_BITS, F16C_IMM_USE_MXCSR, F16C_SINGLE_BITS,
    FpCompare, MAX_SIGNED, MAX_UNSIGNED, MIN_SIGNED, MIN_UNSIGNED, MUL, SAT_ADD_SIGNED,
    SAT_ADD_UNSIGNED, SAT_SUB_SIGNED, SAT_SUB_UNSIGNED, SUB, SimdBitOp, VECTOR_TEST_CLEARED_FLAGS,
    fp_mask_lane, is_vex, parse_immediate, same_xmm_register,
};

impl LiftCtx {
    /// `movaps`/`movups`/`movdqa`/`movdqu dst, src` (and their VEX
    /// `v*` forms) — a full-view copy of `src` into `dst`. The view
    /// width (128 / 256 / 512) comes from the operands; a legacy move
    /// preserves the parent bits above the view, a VEX move zeroes them.
    pub(super) fn lift_simd_move(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let zero_upper = is_vex(insn);
        let Some(value) = self.read_simd_operand(src) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, value, zero_upper) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `movss`/`movsd` (and their VEX forms) — a *scalar* move, which
    /// merges one lane instead of copying a whole view.
    ///
    /// Legacy `movss xmm0, xmm1` writes the low lane and preserves every
    /// bit above it. The VEX 3-operand form `vmovss xmm0, xmm1, xmm2`
    /// takes the low lane from `src2`, the rest of the view from `src1`,
    /// and zeroes the register above the view.
    ///
    /// A load form (`movss xmm0, dword [rbp - 8]`) zeroes everything
    /// above the lane instead of merging, per the SDM — there is no
    /// prior register value to merge with. A store form writes just the
    /// lane. Both go through the byte-granular memory model.
    ///
    /// The 2-operand VEX shape with a register source is declined: it
    /// does not exist in the ISA, so there is nothing to model.
    pub(super) fn lift_sse_scalar_move(&mut self, insn: &Instruction, lane: u16) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            return;
        };
        match (ops.len(), is_vex(insn)) {
            (2, vex) => {
                let src = &ops[1];
                if vex && !self.is_modellable_simd_memory(src) {
                    self.push_simd_unsupported(insn);
                    return;
                }
                let Some(value) = self.read_simd_lane_bits(src, lane, 0) else {
                    self.push_simd_unsupported(insn);
                    return;
                };
                // Loading from memory zeroes above the lane; a
                // register-to-register move merges.
                let ok = if self.is_modellable_simd_memory(src) && self.is_simd_register(dst) {
                    self.write_simd_dst(dst, value, true)
                } else {
                    self.write_simd_lane(dst, value, lane, 0)
                };
                if !ok {
                    self.push_simd_unsupported(insn);
                }
            }
            (3, true) => {
                let Some(merged) = self.vex_scalar_move_value(dst, &ops[1], &ops[2], lane) else {
                    self.push_simd_unsupported(insn);
                    return;
                };
                if !self.write_simd_dst(dst, merged, true) {
                    self.push_simd_unsupported(insn);
                }
            }
            _ => self.push_simd_unsupported(insn),
        }
    }

    /// The full-view value written by a VEX 3-operand scalar move: the
    /// low lane from `src2`, the lanes above it from `src1`.
    fn vex_scalar_move_value(
        &mut self,
        dst: &Operand,
        src1: &Operand,
        src2: &Operand,
        lane: u16,
    ) -> Option<Expr> {
        let view = self.simd_view_bits(dst)?;
        if view <= lane || self.simd_view_bits(src1)? != view {
            return None;
        }
        let low = self.read_simd_lane_bits(src2, lane, 0)?;
        let upper = Expr::extract(self.read_simd_operand(src1)?, view - 1, lane);
        Some(Expr::concat(upper, low))
    }

    /// `pxor`/`pand`/`por`/`pandn` (2-operand RMW) and their `v`-prefixed
    /// 3-operand VEX forms, modelled as 128-bit bit-vector ops. A
    /// `pxor`/`vpxor` of a register with itself is the zero idiom —
    /// the result is the 128-bit constant 0, independent of inputs.
    pub(super) fn lift_simd_bitwise(&mut self, insn: &Instruction, op: SimdBitOp) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            return;
        };
        let zero_upper = is_vex(insn);
        let operand_refs: Vec<&Operand> = ops.iter().collect();
        let Some(width) = self.simd_instruction_view_bits(&operand_refs) else {
            self.push_simd_unsupported(insn);
            return;
        };
        // Operand roles: 2-operand `OP dst, src` is RMW (a = dst,
        // b = src); 3-operand VEX `OP dst, src1, src2` writes dst from
        // (a = src1, b = src2).
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        if matches!(op, SimdBitOp::Xor) && same_xmm_register(a_op, b_op) {
            if !self.write_simd_dst(dst, Expr::konst(0, width), zero_upper) {
                self.push_simd_unsupported(insn);
            }
            return;
        }
        let Some(a) = self.simd_operand_value(a_op, width) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(b) = self.simd_operand_value(b_op, width) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let result = match op {
            SimdBitOp::Xor => Expr::bv_xor(a, b),
            SimdBitOp::And => Expr::bv_and(a, b),
            SimdBitOp::Or => Expr::bv_or(a, b),
            // `pandn`/`vpandn` compute `(~a) & b`. The IR has no bitwise
            // NOT, so `~a` is `a XOR all-ones` at the vector width.
            SimdBitOp::AndNot => {
                Expr::bv_and(Expr::bv_xor(a, crate::lift::simd::all_ones(width)), b)
            }
        };
        if !self.write_simd_dst(dst, result, zero_upper) {
            self.push_simd_unsupported(insn);
        }
    }

    /// The packed-integer half of the x86 dispatch: arithmetic, the
    /// whole-view byte slides and the lane shifts.
    ///
    /// Returns whether it claimed the mnemonic. Split out of
    /// [`Self::lift_instruction_x86`] only to keep each under the line
    /// limit; both are pure dispatch tables.
    pub(super) fn lift_x86_packed_int_by_mnemonic(
        &mut self,
        insn: &Instruction,
        mnem: &str,
    ) -> bool {
        match mnem {
            // Packed integer arithmetic. The lane width is spelled by
            // the mnemonic suffix, as with the FP families.
            "paddb" | "vpaddb" => self.lift_simd_packed_int(insn, ADD, 8),
            "paddw" | "vpaddw" => self.lift_simd_packed_int(insn, ADD, 16),
            "paddd" | "vpaddd" => self.lift_simd_packed_int(insn, ADD, 32),
            "paddq" | "vpaddq" => self.lift_simd_packed_int(insn, ADD, 64),
            "psubb" | "vpsubb" => self.lift_simd_packed_int(insn, SUB, 8),
            "psubw" | "vpsubw" => self.lift_simd_packed_int(insn, SUB, 16),
            "psubd" | "vpsubd" => self.lift_simd_packed_int(insn, SUB, 32),
            "psubq" | "vpsubq" => self.lift_simd_packed_int(insn, SUB, 64),
            "pmullw" | "vpmullw" => self.lift_simd_packed_int(insn, MUL, 16),
            "pmulld" | "vpmulld" => self.lift_simd_packed_int(insn, MUL, 32),
            // The saturating forms. Only byte and word lanes exist —
            // x86 spells no saturating dword add.
            "paddsb" | "vpaddsb" => self.lift_simd_packed_int(insn, SAT_ADD_SIGNED, 8),
            "paddsw" | "vpaddsw" => self.lift_simd_packed_int(insn, SAT_ADD_SIGNED, 16),
            "paddusb" | "vpaddusb" => self.lift_simd_packed_int(insn, SAT_ADD_UNSIGNED, 8),
            "paddusw" | "vpaddusw" => self.lift_simd_packed_int(insn, SAT_ADD_UNSIGNED, 16),
            "psubsb" | "vpsubsb" => self.lift_simd_packed_int(insn, SAT_SUB_SIGNED, 8),
            "psubsw" | "vpsubsw" => self.lift_simd_packed_int(insn, SAT_SUB_SIGNED, 16),
            "psubusb" | "vpsubusb" => self.lift_simd_packed_int(insn, SAT_SUB_UNSIGNED, 8),
            "psubusw" | "vpsubusw" => self.lift_simd_packed_int(insn, SAT_SUB_UNSIGNED, 16),
            // The rounded unsigned average.
            "pavgb" | "vpavgb" => self.lift_simd_packed_int(insn, AVERAGE, 8),
            "pavgw" | "vpavgw" => self.lift_simd_packed_int(insn, AVERAGE, 16),
            // Lane-wise min / max. The `s` / `u` infix is the whole
            // difference between two otherwise identical instructions.
            "pmaxsb" | "vpmaxsb" => self.lift_simd_packed_int(insn, MAX_SIGNED, 8),
            "pmaxsw" | "vpmaxsw" => self.lift_simd_packed_int(insn, MAX_SIGNED, 16),
            "pmaxsd" | "vpmaxsd" => self.lift_simd_packed_int(insn, MAX_SIGNED, 32),
            "pmaxub" | "vpmaxub" => self.lift_simd_packed_int(insn, MAX_UNSIGNED, 8),
            "pmaxuw" | "vpmaxuw" => self.lift_simd_packed_int(insn, MAX_UNSIGNED, 16),
            "pmaxud" | "vpmaxud" => self.lift_simd_packed_int(insn, MAX_UNSIGNED, 32),
            "pminsb" | "vpminsb" => self.lift_simd_packed_int(insn, MIN_SIGNED, 8),
            "pminsw" | "vpminsw" => self.lift_simd_packed_int(insn, MIN_SIGNED, 16),
            "pminsd" | "vpminsd" => self.lift_simd_packed_int(insn, MIN_SIGNED, 32),
            "pminub" | "vpminub" => self.lift_simd_packed_int(insn, MIN_UNSIGNED, 8),
            "pminuw" | "vpminuw" => self.lift_simd_packed_int(insn, MIN_UNSIGNED, 16),
            "pminud" | "vpminud" => self.lift_simd_packed_int(insn, MIN_UNSIGNED, 32),
            // The one single-source member of the family.
            "pabsb" | "vpabsb" => self.lift_simd_packed_int_unary(insn, ABSOLUTE, 8),
            "pabsw" | "vpabsw" => self.lift_simd_packed_int_unary(insn, ABSOLUTE, 16),
            "pabsd" | "vpabsd" => self.lift_simd_packed_int_unary(insn, ABSOLUTE, 32),
            // Constant permutations selected by an immediate.
            "palignr" | "vpalignr" => self.lift_simd_align_right(insn),
            "pshufd" | "vpshufd" => self.lift_simd_shuffle(insn, ShuffleForm::Doubleword),
            "pshuflw" | "vpshuflw" => self.lift_simd_shuffle(insn, ShuffleForm::LowWords),
            "pshufhw" | "vpshufhw" => self.lift_simd_shuffle(insn, ShuffleForm::HighWords),
            // Interleaves. The suffix names the *source* element and the
            // destination element is twice it, which is why `punpcklbw`
            // reads bytes and writes words.
            "punpcklbw" | "vpunpcklbw" => self.lift_simd_unpack(insn, false, 8),
            "punpcklwd" | "vpunpcklwd" => self.lift_simd_unpack(insn, false, 16),
            "punpckldq" | "vpunpckldq" => self.lift_simd_unpack(insn, false, 32),
            "punpcklqdq" | "vpunpcklqdq" => self.lift_simd_unpack(insn, false, 64),
            "punpckhbw" | "vpunpckhbw" => self.lift_simd_unpack(insn, true, 8),
            "punpckhwd" | "vpunpckhwd" => self.lift_simd_unpack(insn, true, 16),
            "punpckhdq" | "vpunpckhdq" => self.lift_simd_unpack(insn, true, 32),
            "punpckhqdq" | "vpunpckhqdq" => self.lift_simd_unpack(insn, true, 64),
            // The variable byte shuffle, whose table is a register.
            "pshufb" | "vpshufb" => self.lift_simd_shuffle_bytes(insn),
            // Saturating narrows. The `s`/`u` infix names the
            // *destination* range; both read their sources signed.
            "packsswb" | "vpacksswb" => self.lift_simd_pack(insn, 16, true),
            "packssdw" | "vpackssdw" => self.lift_simd_pack(insn, 32, true),
            "packuswb" | "vpackuswb" => self.lift_simd_pack(insn, 16, false),
            "packusdw" | "vpackusdw" => self.lift_simd_pack(insn, 32, false),
            // Whole-view byte slides, not lane shifts.
            "pslldq" | "vpslldq" => self.lift_simd_byte_shift(insn, true),
            "psrldq" | "vpsrldq" => self.lift_simd_byte_shift(insn, false),
            // Lane shifts, immediate count only.
            "psllw" | "vpsllw" => self.lift_simd_lane_shift(insn, ShiftOp::Shl, 16),
            "pslld" | "vpslld" => self.lift_simd_lane_shift(insn, ShiftOp::Shl, 32),
            "psllq" | "vpsllq" => self.lift_simd_lane_shift(insn, ShiftOp::Shl, 64),
            "psrlw" | "vpsrlw" => self.lift_simd_lane_shift(insn, ShiftOp::Shr, 16),
            "psrld" | "vpsrld" => self.lift_simd_lane_shift(insn, ShiftOp::Shr, 32),
            "psrlq" | "vpsrlq" => self.lift_simd_lane_shift(insn, ShiftOp::Shr, 64),
            "psraw" | "vpsraw" => self.lift_simd_lane_shift(insn, ShiftOp::Sar, 16),
            "psrad" | "vpsrad" => self.lift_simd_lane_shift(insn, ShiftOp::Sar, 32),
            _ => return false,
        }
        true
    }

    /// `pshufd` / `pshuflw` / `pshufhw` — a constant permutation of one
    /// source, selected by an immediate.
    ///
    /// Never read-modify-write despite having three operands: the
    /// immediate occupies the third slot, and every destination lane
    /// comes from the source.
    fn lift_simd_shuffle(&mut self, insn: &Instruction, form: ShuffleForm) {
        let ops = &insn.operands;
        let (Some(dst), Some(src), Some(imm)) = (ops.first(), ops.get(1), ops.get(2)) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (dst, src, imm) = (dst.clone(), src.clone(), imm.clone());
        if imm.kind != OperandKind::Immediate {
            self.push_simd_unsupported(insn);
            return;
        }
        let (Some(view), Some(selector)) = (
            self.simd_instruction_view_bits(&[&dst, &src]),
            parse_immediate(&imm.raw),
        ) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(picks) = shuffle_picks(form, selector, view) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(result) = self.permuted_value(&src, None, view, form.lane_bits(), &picks) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `punpck{l,h}{bw,wd,dq,qdq}` — interleave the matching half of two
    /// sources, lane by lane.
    ///
    /// Operand roles follow the arithmetic family: 2-operand is
    /// read-modify-write, 3-operand VEX reads its two explicit sources.
    fn lift_simd_unpack(&mut self, insn: &Instruction, high: bool, lane_bits: u16) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let (dst, a_op, b_op) = (dst.clone(), a_op.clone(), b_op.clone());
        let Some(view) = self.simd_instruction_view_bits(&[&dst, &a_op, &b_op]) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(picks) = unpack_picks(high, lane_bits, view) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(result) = self.permuted_value(&a_op, Some(&b_op), view, lane_bits, &picks) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `palignr` — concatenate the destination above the source and
    /// take the 16-byte window starting `imm8` bytes up.
    ///
    /// The first family here whose legacy form is 3-operand and still
    /// reads its destination: it is the *high* half of the pair. The
    /// VEX form takes four operands and the pair from its two explicit
    /// sources instead.
    ///
    /// Per 128-bit block like every other x86 permutation, so a `ymm`
    /// form aligns each half against its own half and never across.
    fn lift_simd_align_right(&mut self, insn: &Instruction) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            self.push_simd_unsupported(insn);
            return;
        };
        // The pair is (high, low) = (first source, second source), and
        // the immediate is always last.
        let (high_op, low_op, selector) = match ops.len() {
            3 => (dst, &ops[1], &ops[2]),
            4 => (&ops[1], &ops[2], &ops[3]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        if selector.kind != OperandKind::Immediate {
            self.push_simd_unsupported(insn);
            return;
        }
        let Some(offset) = parse_immediate(&selector.raw) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (dst, high_op, low_op) = (dst.clone(), high_op.clone(), low_op.clone());
        let Some(view) = self.simd_instruction_view_bits(&[&dst, &high_op, &low_op]) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(picks) = align_right_picks(offset, view) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(result) =
            self.permuted_value(&low_op, Some(&high_op), view, BITS_PER_BYTE, &picks)
        else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `pshufb` — a byte shuffle whose indices come from a *register*
    /// rather than an immediate, so the table cannot be resolved at lift
    /// time and each destination byte becomes a select over every
    /// candidate.
    ///
    /// Two rules separate it from the constant shuffles. A source byte
    /// whose index has bit 7 set writes zero rather than selecting
    /// anything, and only the low four index bits are used — which is
    /// also what confines the lookup to its own 128-bit block on a
    /// 256-bit view.
    fn lift_simd_shuffle_bytes(&mut self, insn: &Instruction) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let (dst, a_op, b_op) = (dst.clone(), a_op.clone(), b_op.clone());
        let Some(view) = self.simd_instruction_view_bits(&[&dst, &a_op, &b_op]) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(result) = self.shuffled_bytes(&a_op, &b_op, view) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// The `Ite` chain `pshufb` lowers to: one select ladder per
    /// destination byte over the sixteen candidates in its block.
    ///
    /// Capped like the `AArch64` table lookup it mirrors, and for the
    /// same reason — the formula is quadratic in the block size, so a
    /// wide view would hand the solver a term nobody wants to see.
    fn shuffled_bytes(&mut self, a_op: &Operand, b_op: &Operand, view: u16) -> Option<Expr> {
        let bytes = view.checked_div(BITS_PER_BYTE)?;
        let per_block = permute_blocks(view).and(Some(PERMUTE_BLOCK_BITS / BITS_PER_BYTE))?;
        if u32::from(bytes).checked_mul(u32::from(per_block))? > SHUFFLE_BYTE_SELECT_CAP {
            tracing::debug!(
                view,
                cap = SHUFFLE_BYTE_SELECT_CAP,
                "declining pshufb: select count over cap"
            );
            return None;
        }
        let source = self.simd_operand_value(a_op, view)?;
        let indices = self.simd_operand_value(b_op, view)?;
        let mut lanes = Vec::with_capacity(usize::from(bytes));
        for byte in 0..bytes {
            let index = Self::extract_lane(indices.clone(), BITS_PER_BYTE, byte)?;
            let base = (byte / per_block).checked_mul(per_block)?;
            // Only the low four bits select, so the ladder is built over
            // the masked index and every candidate is reachable — the
            // value it bottoms out on is unreachable rather than
            // meaningful.
            let selector = Expr::bv_and(
                index.clone(),
                Expr::konst(u128::from(per_block - 1), BITS_PER_BYTE),
            );
            let mut chosen = Expr::konst(0, BITS_PER_BYTE);
            for candidate in 0..per_block {
                let selected = Self::extract_lane(
                    source.clone(),
                    BITS_PER_BYTE,
                    base.checked_add(candidate)?,
                )?;
                chosen = Expr::Ite {
                    cond: Box::new(Expr::eq(
                        selector.clone(),
                        Expr::konst(u128::from(candidate), BITS_PER_BYTE),
                    )),
                    then_expr: Box::new(selected),
                    else_expr: Box::new(chosen),
                };
            }
            // A set bit 7 clears the byte outright, whatever the low
            // bits would have selected.
            lanes.push(Expr::Ite {
                cond: Box::new(Expr::eq(
                    Expr::extract(index, BITS_PER_BYTE - 1, BITS_PER_BYTE - 1),
                    Expr::konst(1, 1),
                )),
                then_expr: Box::new(Expr::konst(0, BITS_PER_BYTE)),
                else_expr: Box::new(chosen),
            });
        }
        Self::concat_lanes(lanes)
    }

    /// `pack{ss,us}{wb,dw}` — narrow two sources into one destination,
    /// saturating each element into the destination's range.
    ///
    /// The first source fills the low half of every 128-bit block and
    /// the second the high half, so on a 256-bit view the halves
    /// interleave per block rather than the first source filling the
    /// whole low 128 bits.
    fn lift_simd_pack(&mut self, insn: &Instruction, source_bits: u16, signed_dst: bool) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let (dst, a_op, b_op) = (dst.clone(), a_op.clone(), b_op.clone());
        let Some(view) = self.simd_instruction_view_bits(&[&dst, &a_op, &b_op]) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(result) = self.packed_narrow(&a_op, &b_op, view, source_bits, signed_dst) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// The saturating narrow itself, block by block.
    fn packed_narrow(
        &mut self,
        a_op: &Operand,
        b_op: &Operand,
        view: u16,
        source_bits: u16,
        signed_dst: bool,
    ) -> Option<Expr> {
        let dest_bits = source_bits.checked_div(2).filter(|b| *b > 0)?;
        let blocks = permute_blocks(view)?;
        let per_block = PERMUTE_BLOCK_BITS
            .checked_div(source_bits)
            .filter(|n| *n > 0)?;
        let a = self.simd_operand_value(a_op, view)?;
        let b = self.simd_operand_value(b_op, view)?;
        let mut lanes = Vec::with_capacity(usize::from(blocks) * 2 * usize::from(per_block));
        for block in 0..blocks {
            let base = block.checked_mul(per_block)?;
            for source in [&a, &b] {
                for lane in 0..per_block {
                    let element =
                        Self::extract_lane(source.clone(), source_bits, base.checked_add(lane)?)?;
                    lanes.push(clamp_to_element(
                        element,
                        source_bits,
                        dest_bits,
                        signed_dst,
                    )?);
                }
            }
        }
        Self::concat_lanes(lanes)
    }

    /// Build a whole view from a constant table of lane sources.
    ///
    /// Each operand is materialised **once** and the lanes are extracted
    /// from that value, so a memory source costs one load rather than
    /// one per lane. The picks carry absolute lane indices, which is what
    /// lets the per-128-bit-block tables above be written once and
    /// repeated across a 256-bit view.
    fn permuted_value(
        &mut self,
        a_op: &Operand,
        b_op: Option<&Operand>,
        view: u16,
        lane_bits: u16,
        picks: &[LanePick],
    ) -> Option<Expr> {
        let a = self.simd_operand_value(a_op, view)?;
        let b = match b_op {
            Some(op) => Some(self.simd_operand_value(op, view)?),
            None => None,
        };
        let mut lanes = Vec::with_capacity(picks.len());
        for pick in picks {
            let (source, index) = match *pick {
                LanePick::First(index) => (a.clone(), index),
                LanePick::Second(index) => (b.clone()?, index),
                LanePick::Zero => {
                    lanes.push(Expr::konst(0, lane_bits));
                    continue;
                }
            };
            lanes.push(Self::extract_lane(source, lane_bits, index)?);
        }
        Self::concat_lanes(lanes)
    }

    /// Packed integer arithmetic: the same lane operation over every
    /// `lane_bits` lane of the destination's view.
    ///
    /// Shares its operand roles and upper-bits rule with the packed FP
    /// twin — 2-operand is read-modify-write, 3-operand VEX reads its
    /// two explicit sources, and a VEX write zeroes above the view.
    fn lift_simd_packed_int(&mut self, insn: &Instruction, op: PackedIntOp, lane_bits: u16) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            return;
        };
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let (dst_c, a_c, b_c) = (dst.clone(), a_op.clone(), b_op.clone());
        let Some(result) = self.packed_int_result(&dst_c, &a_c, Some(&b_c), op, lane_bits) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst_c, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// The single-source packed integer form (`pabs*`).
    ///
    /// Separate from [`Self::lift_simd_packed_int`] rather than a flag on
    /// it, because the two disagree about what a 2-operand instruction
    /// means: there it is read-modify-write and the destination is also
    /// the first source, here the destination is written from the one
    /// source and never read.
    fn lift_simd_packed_int_unary(&mut self, insn: &Instruction, op: PackedIntOp, lane_bits: u16) {
        let ops = &insn.operands;
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (dst, src) = (dst.clone(), src.clone());
        let Some(result) = self.packed_int_result(&dst, &src, None, op, lane_bits) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `pslldq` / `psrldq` — shift the *whole view* by a byte count.
    ///
    /// Despite sitting among the lane shifts these are not lane
    /// operations at all: they slide the entire 128-bit (or 256-bit)
    /// value, so treating them as a per-lane shift would be silently
    /// wrong. A count of 16 or more clears the register.
    fn lift_simd_byte_shift(&mut self, insn: &Instruction, left: bool) {
        let ops = &insn.operands;
        let (Some(dst), Some(count)) = (ops.first(), ops.last()) else {
            return;
        };
        let (dst, count) = (dst.clone(), count.clone());
        let source = if ops.len() == 3 {
            ops[1].clone()
        } else {
            dst.clone()
        };
        let (Some(view), Some(bytes)) = (self.simd_view_bits(&dst), parse_immediate(&count.raw))
        else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(value) = self.simd_operand_value(&source, view) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let shift = bytes.saturating_mul(u64::from(BITS_PER_BYTE));
        let result = if shift >= u64::from(view) {
            Expr::konst(0, view)
        } else {
            let amount = Expr::konst(u128::from(shift), view);
            if left {
                Expr::shl(value, amount)
            } else {
                Expr::lshr(value, amount)
            }
        };
        if !self.write_simd_dst(&dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `psll` / `psrl` / `psra` with an immediate count, applied to
    /// every lane.
    ///
    /// The register-count form is deliberately absent: it takes the
    /// amount from the low 64 bits of a vector register, which is a
    /// different shape, and declining it is sound.
    fn lift_simd_lane_shift(&mut self, insn: &Instruction, op: ShiftOp, lane_bits: u16) {
        let ops = &insn.operands;
        let (Some(dst), Some(count)) = (ops.first(), ops.last()) else {
            return;
        };
        let (dst, count) = (dst.clone(), count.clone());
        let source = if ops.len() == 3 {
            ops[1].clone()
        } else {
            dst.clone()
        };
        if count.kind != OperandKind::Immediate {
            self.push_simd_unsupported(insn);
            return;
        }
        let (Some(view), Some(amount)) = (self.simd_view_bits(&dst), parse_immediate(&count.raw))
        else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (Some(value), Some(lanes)) = (
            self.simd_operand_value(&source, view),
            Self::packed_lane_count(view, lane_bits),
        ) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let mut result = Vec::with_capacity(usize::from(lanes));
        for index in 0..lanes {
            let Some(lane) = Self::extract_lane(value.clone(), lane_bits, index) else {
                self.push_simd_unsupported(insn);
                return;
            };
            // x86 saturates the shift rather than masking it: a count
            // at or beyond the lane width zeroes the lane (or fills it
            // with the sign for an arithmetic shift), where a bare
            // `Shl` at that width is undefined in the IR.
            result.push(if amount >= u64::from(lane_bits) {
                match op {
                    ShiftOp::Sar => {
                        Expr::ashr(lane, Expr::konst(u128::from(lane_bits - 1), lane_bits))
                    }
                    _ => Expr::konst(0, lane_bits),
                }
            } else {
                let by = Expr::konst(u128::from(amount), lane_bits);
                match op {
                    ShiftOp::Shl => Expr::shl(lane, by),
                    ShiftOp::Shr => Expr::lshr(lane, by),
                    ShiftOp::Sar => Expr::ashr(lane, by),
                }
            });
        }
        let Some(packed) = Self::concat_lanes(result) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(&dst, packed, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `pmovmskb r32, xmm` — the sign bit of each source byte,
    /// concatenated into the low bits of a general register with the
    /// rest zeroed (16 bits from an `xmm` source, 32 from a `ymm`).
    ///
    /// This is the instruction that carries a vector compare's result
    /// into the integer world, so it is the hinge of the SSE
    /// strlen/memcmp idiom: without it the mask never reaches a flag
    /// and the branch stays unresolved.
    pub(super) fn lift_simd_move_mask(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let (dst, src) = (dst.clone(), src.clone());
        let (Some(view), Some(width)) = (
            self.simd_view_bits(&src),
            nonzero_width(self.operand_width(&dst)),
        ) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(value) = self.simd_operand_value(&src, view) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(lanes) = Self::packed_lane_count(view, BITS_PER_BYTE) else {
            self.push_simd_unsupported(insn);
            return;
        };
        // Most significant byte first: `concat_lanes` takes its input
        // least-significant first, and byte `i`'s sign bit lands at bit
        // `i` of the result.
        let mut bits = Vec::with_capacity(usize::from(lanes));
        for index in 0..lanes {
            let Some(byte) = Self::extract_lane(value.clone(), BITS_PER_BYTE, index) else {
                self.push_simd_unsupported(insn);
                return;
            };
            bits.push(Expr::extract(byte, BITS_PER_BYTE - 1, BITS_PER_BYTE - 1));
        }
        let Some(mask) = Self::concat_lanes(bits) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_register_to(&dst, Expr::zero_ext(mask, width)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `movd` / `movq` between a vector register and a general register
    /// or memory, at `width` bits.
    ///
    /// Both directions **zero** everything above the transferred value
    /// rather than merging: `movd xmm0, eax` clears bits 127:32 of the
    /// destination, and the general-register direction zero-extends
    /// into the full register. Merging instead would leave whatever the
    /// vector held before, which is a wrong value rather than a
    /// decline.
    pub(super) fn lift_simd_gpr_transfer(&mut self, insn: &Instruction, width: u16) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let (dst, src) = (dst.clone(), src.clone());
        let to_vector = self.simd_layout(&dst).is_some();
        let value = if self.simd_layout(&src).is_some() {
            let Some(view) = self.simd_view_bits(&src) else {
                self.push_simd_unsupported(insn);
                return;
            };
            let Some(whole) = self.simd_operand_value(&src, view) else {
                self.push_simd_unsupported(insn);
                return;
            };
            if view <= width {
                whole
            } else {
                Expr::extract(whole, width - 1, 0)
            }
        } else {
            self.read_operand_at(&src, width)
        };
        if to_vector {
            // `zero_upper = true` regardless of VEX: the legacy forms
            // zero the vector register too.
            if !self.write_simd_dst(&dst, value, true) {
                self.push_simd_unsupported(insn);
            }
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(&dst)) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let widened = if dst_width > width {
            Expr::zero_ext(value, dst_width)
        } else {
            value
        };
        if !self.write_register_to(&dst, widened) {
            self.push_simd_unsupported(insn);
        }
    }

    /// The packed integer compares: `pcmpeq{b,w,d,q}` and
    /// `pcmpgt{b,w,d,q}`.
    ///
    /// Each writes an all-ones mask into a lane where the predicate
    /// holds and zeros where it does not — a *value*, not a flag, which
    /// is why it cannot reuse the scalar compare path that writes
    /// EFLAGS. `pcmpgt*` is signed on x86; there is no unsigned form.
    ///
    /// The mask is what feeds `pmovmskb` in the SSE string idiom, so
    /// this family is the one the corpus measurement puts in a
    /// conditional-branch block essentially every time it appears.
    pub(super) fn lift_simd_packed_compare(
        &mut self,
        insn: &Instruction,
        kind: CompareKind,
        lane_bits: u16,
    ) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            return;
        };
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let (a_op, b_op) = (a_op.clone(), b_op.clone());
        let Some(result) = self.packed_compare_result(dst, &a_op, &b_op, kind, lane_bits) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, is_vex(insn)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// Lane-wise compare of two SIMD operands, materialising each once.
    fn packed_compare_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: &Operand,
        kind: CompareKind,
        lane_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[dst, a_op, b_op])?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = self.simd_operand_value(b_op, view)?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let b = Self::extract_lane(b_val.clone(), lane_bits, index)?;
            lanes.push(compare_lane(kind, a, b, lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `ptest` / `vptest` — a whole-view bitwise test that writes flags
    /// and no register.
    ///
    /// `ZF` is set when `src AND dst` is zero and `CF` when
    /// `src AND NOT dst` is; the SDM calls the second one the
    /// "`ANDN`" form. The asymmetry is the trap: it is the *first*
    /// operand that gets complemented, so reversing the roles produces a
    /// wrong flag rather than a decline. `OF`, `AF`, `PF` and `SF` are
    /// architecturally cleared.
    ///
    /// The AND needs no lane loop — no carry crosses a lane boundary, so
    /// the whole view at once is the same bits and a far smaller term.
    pub(super) fn lift_simd_vector_test(&mut self, insn: &Instruction) {
        let ops = &insn.operands;
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (dst, src) = (dst.clone(), src.clone());
        let Some(view) = self.simd_instruction_view_bits(&[&dst, &src]) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let (Some(dst_val), Some(src_val)) = (
            self.simd_operand_value(&dst, view),
            self.simd_operand_value(&src, view),
        ) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let zero = Expr::konst(0, view);
        let both = Expr::bv_and(src_val.clone(), dst_val.clone());
        let complemented = Expr::bv_and(
            src_val,
            Expr::bv_xor(dst_val, crate::lift::simd::all_ones(view)),
        );
        self.set_flag("ZF", Expr::eq(both, zero.clone()));
        self.set_flag("CF", Expr::eq(complemented, zero));
        for flag in VECTOR_TEST_CLEARED_FLAGS {
            self.set_flag(flag, Expr::konst(0, 1));
        }
    }

    /// Packed floating-point arithmetic: the same lane operation applied
    /// independently to every `lane_bits`-wide lane of the destination's
    /// vector view (`addps` → 4 single lanes over 128 bits, `vaddps ymm`
    /// → 8 over 256).
    ///
    /// Operand roles and the upper-bits rule are shared with
    /// [`Self::lift_simd_bitwise`]: 2-operand is RMW, 3-operand VEX reads
    /// its two explicit sources, and a VEX write zeroes above the view.
    pub(super) fn lift_simd_packed_fp(
        &mut self,
        insn: &Instruction,
        op: FpArithOp,
        lane_bits: u16,
    ) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            return;
        };
        let zero_upper = is_vex(insn);
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let Some(result) = self.packed_fp_result(dst, a_op, b_op, op, lane_bits) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, zero_upper) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `vcvtph2ps` — widen packed half-precision lanes to single.
    ///
    /// The lane count comes from the *destination* view: each 32-bit
    /// result lane consumes one 16-bit source lane, so a 128-bit
    /// destination reads the low 64 bits of the source.
    pub(super) fn lift_f16c_widen(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(result) = self.f16c_lanes(dst, src, F16C_HALF_BITS, F16C_SINGLE_BITS) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `vcvtps2ph` — narrow packed single lanes to half-precision.
    ///
    /// The lane count comes from the *source* view. The result is half
    /// as wide as that view, and the VEX write zeroes everything above
    /// it, so the narrower value is handed to the write as-is.
    ///
    /// The immediate selects the rounding mode, and bit 2 means "use
    /// MXCSR" — which this lifter cannot pin, so that encoding declines
    /// rather than assuming a mode.
    pub(super) fn lift_f16c_narrow(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let uses_mxcsr = insn
            .operands
            .get(2)
            .and_then(|o| parse_immediate(&o.raw))
            .is_none_or(|imm| imm & F16C_IMM_USE_MXCSR != 0);
        if uses_mxcsr {
            self.push_simd_unsupported(insn);
            return;
        }
        let Some(result) = self.f16c_lanes(src, src, F16C_SINGLE_BITS, F16C_HALF_BITS) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_simd_unsupported(insn);
        }
    }

    /// Convert every lane of `src` from `from_bits` to `to_bits`, with
    /// the lane count taken from `count_from`'s view divided by the
    /// wider of the two lane widths.
    fn f16c_lanes(
        &mut self,
        count_from: &Operand,
        src: &Operand,
        from_bits: u16,
        to_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[count_from, src])?;
        let count = Self::packed_lane_count(view, from_bits.max(to_bits))?;
        let (from_e, from_s) = fp_sort_bits_checked(from_bits)?;
        let (to_e, to_s) = fp_sort_bits_checked(to_bits)?;
        let src_val = self.simd_operand_value(src, view)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let raw = Self::extract_lane(src_val.clone(), from_bits, index)?;
            lanes.push(Expr::fp_to_ieee_bv(Expr::fp_to_fp(
                Expr::bv_to_fp(raw, from_e, from_s),
                RoundingMode::NearestTiesEven,
                to_e,
                to_s,
            )));
        }
        Self::concat_lanes(lanes)
    }

    /// The SSE/AVX compares, which write a per-lane mask of all-ones
    /// (predicate true) or all-zeros rather than a float.
    pub(super) fn lift_simd_fp_mask_compare(&mut self, insn: &Instruction, cmp: &FpCompare) {
        let ops = &insn.operands;
        let Some(dst) = ops.first() else {
            return;
        };
        let (a_op, b_op) = match ops.len() {
            2 => (dst, &ops[1]),
            3 => (&ops[1], &ops[2]),
            _ => {
                self.push_simd_unsupported(insn);
                return;
            }
        };
        let Some(result) = self.fp_mask_result(dst, a_op, b_op, cmp) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let written = if cmp.packed {
            self.write_simd_dst(dst, result, is_vex(insn))
        } else {
            self.write_simd_lane(dst, result, cmp.lane_bits, 0)
        };
        if !written {
            self.push_simd_unsupported(insn);
        }
    }

    fn fp_mask_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: &Operand,
        cmp: &FpCompare,
    ) -> Option<Expr> {
        if !cmp.packed {
            let a = self.read_simd_lane_bits(a_op, cmp.lane_bits, 0)?;
            let b = self.read_simd_lane_bits(b_op, cmp.lane_bits, 0)?;
            return fp_mask_lane(cmp, a, b);
        }
        let view = self.simd_instruction_view_bits(&[dst, a_op, b_op])?;
        let count = Self::packed_lane_count(view, cmp.lane_bits)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = self.simd_operand_value(b_op, view)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), cmp.lane_bits, index)?;
            let b = Self::extract_lane(b_val.clone(), cmp.lane_bits, index)?;
            lanes.push(fp_mask_lane(cmp, a, b)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `sqrtps`/`sqrtpd` (every lane) and `sqrtss`/`sqrtsd` (low lane
    /// only, upper bits of the destination preserved).
    ///
    /// Unary, so the source is the *second* operand in both the SSE and
    /// the VEX form — unlike the binary handlers there is no RMW read of
    /// the destination.
    pub(super) fn lift_simd_sqrt(&mut self, insn: &Instruction, lane_bits: u16, packed: bool) {
        let ops = &insn.operands;
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return;
        };
        let Some(result) = self.sqrt_result(dst, src, lane_bits, packed) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let written = if packed {
            self.write_simd_dst(dst, result, is_vex(insn))
        } else {
            self.write_simd_lane(dst, result, lane_bits, 0)
        };
        if !written {
            self.push_simd_unsupported(insn);
        }
    }

    fn sqrt_result(
        &mut self,
        dst: &Operand,
        src: &Operand,
        lane_bits: u16,
        packed: bool,
    ) -> Option<Expr> {
        let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
        let root = |bits: Expr| {
            Expr::fp_to_ieee_bv(Expr::fsqrt(
                Expr::bv_to_fp(bits, ebits, sbits),
                RoundingMode::NearestTiesEven,
            ))
        };
        if !packed {
            let low = self.read_simd_lane_bits(src, lane_bits, 0)?;
            return Some(root(low));
        }
        let view = self.simd_instruction_view_bits(&[dst, src])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let src_val = self.simd_operand_value(src, view)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            lanes.push(root(Self::extract_lane(src_val.clone(), lane_bits, index)?));
        }
        Self::concat_lanes(lanes)
    }

    /// `addss`/`subss`/`mulss`/`divss` (32-bit lane) and their `sd`
    /// double-precision (64-bit lane) forms — a scalar FP op on the low
    /// lane of `dst`, with the upper parent bits preserved (legacy SSE
    /// scalar semantics). `dst := fp_op(lane(dst), lane(src))`.
    pub(super) fn lift_simd_scalar_fp(&mut self, insn: &Instruction, op: FpArithOp, lane: u16) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let (Some(a), Some(b)) = (
            self.read_simd_lane_bits(dst, lane, 0),
            self.read_simd_lane_bits(src, lane, 0),
        ) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some(result) = fp_lane_result(op, a, b, lane) else {
            self.push_simd_unsupported(insn);
            return;
        };
        if !self.write_simd_lane(dst, result, lane, 0) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `comiss`/`ucomiss` (32-bit lane) and `comisd`/`ucomisd` (64-bit)
    /// — compare the low lanes and write the result into EFLAGS.
    ///
    /// Per the SDM: unordered sets ZF, PF and CF; greater-than clears
    /// all three; less-than sets CF alone; equal sets ZF alone. OF, SF
    /// and AF are always cleared. The ordered and unordered forms differ
    /// only in which NaN raises the invalid-operation exception, which
    /// the value model does not track, so both lift identically.
    ///
    /// PF is exact here, unlike the integer path where it degrades to
    /// `Unknown`: it *is* the unordered predicate, which makes the
    /// `ucomiss` + `jp`/`jnp` NaN check compilers emit resolve
    /// precisely.
    pub(super) fn lift_simd_fp_compare(&mut self, insn: &Instruction, lane: u16) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let (Some(a), Some(b)) = (
            self.read_simd_lane_fp(dst, lane, 0),
            self.read_simd_lane_fp(src, lane, 0),
        ) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let unordered = Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone()));
        self.set_flag("PF", unordered.clone());
        self.set_flag(
            "ZF",
            Expr::bool_or(unordered.clone(), Expr::feq(a.clone(), b.clone())),
        );
        self.set_flag("CF", Expr::bool_or(unordered, Expr::flt(a, b)));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("SF", Expr::konst(0, 1));
    }

    /// `cvtsi2ss`/`cvtsi2sd` — convert a signed integer register to a
    /// float and write it to the low lane of `dst`.
    ///
    /// The rounding mode is the architectural MXCSR default, pinned the
    /// same way the SSE arithmetic handlers pin it; a program that
    /// reprograms MXCSR is outside the value model either way.
    pub(super) fn lift_int_to_fp(&mut self, insn: &Instruction, lane: u16) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(int) = self.read_register(src) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let converted = Expr::sbv_to_fp(int, RoundingMode::NearestTiesEven, ebits, sbits);
        if !self.write_simd_lane(dst, Expr::fp_to_ieee_bv(converted), lane, 0) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `cvtss2si`/`cvtsd2si` and their truncating `cvtt…` forms —
    /// convert the low lane of `src` to a signed integer register.
    ///
    /// The truncating forms carry the rounding mode in the opcode; the
    /// others take the MXCSR default, pinned as in [`Self::lift_int_to_fp`].
    pub(super) fn lift_fp_to_int(&mut self, insn: &Instruction, lane: u16, rm: RoundingMode) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(f) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let width = self.operand_width(dst);
        if !self.write_register_to(dst, Expr::fp_to_sbv(f, rm, width)) {
            self.push_simd_unsupported(insn);
        }
    }

    /// `cvtss2sd`/`cvtsd2ss` — convert the low lane of `src` to the
    /// other float sort and write it to the low lane of `dst`.
    ///
    /// `src_lane` is the source sort's width, `dst_lane` the target's.
    /// Widening is exact; narrowing rounds under the MXCSR default,
    /// pinned as in [`Self::lift_int_to_fp`].
    pub(super) fn lift_fp_to_fp(&mut self, insn: &Instruction, src_lane: u16, dst_lane: u16) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(f) = self.read_simd_lane_fp(src, src_lane, 0) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let Some((ebits, sbits)) = fp_sort_bits_checked(dst_lane) else {
            self.push_simd_unsupported(insn);
            return;
        };
        let converted = Expr::fp_to_fp(f, RoundingMode::NearestTiesEven, ebits, sbits);
        if !self.write_simd_lane(dst, Expr::fp_to_ieee_bv(converted), dst_lane, 0) {
            self.push_simd_unsupported(insn);
        }
    }

    pub(super) fn push_simd_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable SIMD operand at {addr}", addr = insn.address),
        });
    }
}

/// Where one destination lane comes from. Indices are absolute within
/// the whole view, not within the 128-bit block.
///
/// [`LanePick::Zero`] is not a source at all, and exists because
/// `palignr` fills with architectural zeros once its byte offset passes
/// the width of the source pair. Spelling that as an out-of-range index
/// would not work: [`LiftCtx::extract_lane`] answers `None` past the
/// view, so the whole instruction would decline where the correct
/// answer is a known constant.
#[derive(Clone, Copy)]
enum LanePick {
    /// Lane `index` of the first source.
    First(u16),
    /// Lane `index` of the second source.
    Second(u16),
    /// A lane the architecture defines as zero.
    Zero,
}

/// Bits in the 128-bit block every x86 lane permutation is confined to.
///
/// This is the load-bearing constant of the whole family. AVX widened
/// `pshufd` and `punpck*` to 256 bits by running the *same* 128-bit
/// permutation in each half rather than by widening the permutation, so
/// an index never crosses the halfway line. Treating a `ymm` operand as
/// one flat vector of lanes would move bytes between halves — a wrong
/// value, not a decline.
const PERMUTE_BLOCK_BITS: u16 = 128;

/// Ceiling on the selects a `pshufb` may expand to.
///
/// The lowering is one `Ite` per destination byte per candidate, so the
/// term is quadratic in the block size: 256 selects for an `xmm`, 512
/// for a `ymm`, 1024 for a `zmm`. Past the cap it declines and says so,
/// rather than handing the solver a formula nobody wants to see. Mirrors
/// the `AArch64` table-lookup cap this construction is borrowed from.
const SHUFFLE_BYTE_SELECT_CAP: u32 = 1024;

/// Which constant shuffle a `pshuf*` mnemonic names.
#[derive(Clone, Copy)]
enum ShuffleForm {
    /// `pshufd` — the immediate selects all four doublewords.
    Doubleword,
    /// `pshuflw` — the immediate selects the low four words; the high
    /// four are copied.
    LowWords,
    /// `pshufhw` — the mirror image: the high four are selected, the low
    /// four copied.
    HighWords,
}

impl ShuffleForm {
    /// The element width the form permutes.
    const fn lane_bits(self) -> u16 {
        match self {
            Self::Doubleword => 32,
            Self::LowWords | Self::HighWords => 16,
        }
    }

    /// The lane the immediate's four selectors start at within a block.
    const fn first_selected_lane(self) -> u16 {
        match self {
            Self::Doubleword | Self::LowWords => 0,
            Self::HighWords => SHUFFLE_SELECTORS,
        }
    }
}

/// Selectors an `imm8` carries: four 2-bit fields.
const SHUFFLE_SELECTORS: u16 = 4;
/// Width of one of those fields.
const SHUFFLE_SELECTOR_BITS: u16 = 2;
/// Mask of one field.
const SHUFFLE_SELECTOR_MASK: u64 = 0b11;

/// How many whole 128-bit blocks a view divides into, or `None` if it is
/// not a multiple of one.
fn permute_blocks(view: u16) -> Option<u16> {
    (view % PERMUTE_BLOCK_BITS == 0).then_some(view / PERMUTE_BLOCK_BITS)
}

/// The destination lane table of a `pshuf*`.
///
/// Within each block the four selected lanes come from the immediate and
/// every other lane is copied straight through, which is what makes
/// `pshuflw` leave the high quadword standing.
fn shuffle_picks(form: ShuffleForm, selector: u64, view: u16) -> Option<Vec<LanePick>> {
    let lane_bits = form.lane_bits();
    let blocks = permute_blocks(view)?;
    let per_block = PERMUTE_BLOCK_BITS / lane_bits;
    let first = form.first_selected_lane();
    let mut picks = Vec::with_capacity(usize::from(blocks) * usize::from(per_block));
    for block in 0..blocks {
        let base = block.checked_mul(per_block)?;
        for lane in 0..per_block {
            let within = match lane.checked_sub(first) {
                Some(field) if field < SHUFFLE_SELECTORS => {
                    let shift = u32::from(field.checked_mul(SHUFFLE_SELECTOR_BITS)?);
                    let chosen = (selector >> shift) & SHUFFLE_SELECTOR_MASK;
                    first.checked_add(u16::try_from(chosen).ok()?)?
                }
                // Outside the immediate's reach, so copied verbatim.
                _ => lane,
            };
            picks.push(LanePick::First(base.checked_add(within)?));
        }
    }
    Some(picks)
}

/// The destination byte table of a `palignr`, per 128-bit block.
///
/// Destination byte `i` takes byte `i + offset` of the concatenation
/// `high : low`, so indices below 16 come from the low source and
/// 16..31 from the high one. Past 31 the architecture defines the byte
/// as zero, which is why [`LanePick`] carries a variant for it: the
/// whole result is zero once `offset >= 32`, and the top bytes are zero
/// for any `offset` above 16.
fn align_right_picks(offset: u64, view: u16) -> Option<Vec<LanePick>> {
    let blocks = permute_blocks(view)?;
    let per_block = PERMUTE_BLOCK_BITS / BITS_PER_BYTE;
    let pair = per_block.checked_mul(2)?;
    // The immediate is a byte; a wider value is not an encoding.
    let offset = u16::from(u8::try_from(offset).ok()?);
    let mut picks = Vec::with_capacity(usize::from(blocks) * usize::from(per_block));
    for block in 0..blocks {
        let base = block.checked_mul(per_block)?;
        for byte in 0..per_block {
            let Some(index) = byte.checked_add(offset).filter(|i| *i < pair) else {
                picks.push(LanePick::Zero);
                continue;
            };
            picks.push(match index.checked_sub(per_block) {
                Some(high) => LanePick::Second(base.checked_add(high)?),
                None => LanePick::First(base.checked_add(index)?),
            });
        }
    }
    Some(picks)
}

/// The destination lane table of a `punpck{l,h}*`.
///
/// Alternating lanes come from the two sources, both read from the same
/// half of their block: the low form takes the bottom half of each,
/// the high form the top.
fn unpack_picks(high: bool, lane_bits: u16, view: u16) -> Option<Vec<LanePick>> {
    let blocks = permute_blocks(view)?;
    let per_block = PERMUTE_BLOCK_BITS
        .checked_div(lane_bits)
        .filter(|n| *n > 1)?;
    let half = per_block / 2;
    let source_base = if high { half } else { 0 };
    let mut picks = Vec::with_capacity(usize::from(blocks) * usize::from(per_block));
    for block in 0..blocks {
        let base = block.checked_mul(per_block)?;
        for pair in 0..half {
            let index = base.checked_add(source_base)?.checked_add(pair)?;
            picks.push(LanePick::First(index));
            picks.push(LanePick::Second(index));
        }
    }
    Some(picks)
}
