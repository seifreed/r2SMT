//! `AArch64` per-mnemonic lifter handlers, extracted from `lift.rs`.
//! Methods on [`LiftCtx`]; shared infrastructure stays in the parent.

use r2smt_ir::expr::{Expr, RoundingMode, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::register_layout;

use super::aarch32::shift::stored_carry;
use super::shifted_operand::{OPERAND2_INDEX_ARITH3, OPERAND2_INDEX_COMPARE};
use super::{
    BinOp, CsArithOp, FpArithOp, LiftCtx, MemAccess, VectorShape, Writeback,
    aarch64_cond_suffix_to_predicate, constant_delta, fp_lane_result, fp_propagating_max_min,
    fp_sort_bits_checked, nonzero_width, vector_shape, width_mask,
};
use r2smt_common::Arch;

pub(super) mod neon;

impl LiftCtx {
    pub(super) fn lift_instruction_aarch64(&mut self, insn: &Instruction) {
        // NEON reuses the integer and scalar-FP mnemonics, so the shape
        // dispatch has to sit above the match rather than in arms of its
        // own — `"add"` and `"fadd"` are already claimed below, and the
        // register table resolves `v0.4s` to the same `v0` parent a
        // scalar handler would happily add as one 128-bit value.
        match vector_shape(insn, Arch::Aarch64) {
            VectorShape::None => {}
            VectorShape::Lifted { .. } => {
                if let Some(shape) = neon::shape(insn) {
                    self.lift_neon(insn, shape);
                }
                return;
            }
            // The structured loads and stores: resolved by the same
            // seam the effect table consulted, so the two cannot
            // disagree about which of them the slicer may retain.
            VectorShape::LiftedMemory(_) => {
                if let Some(access) = neon::structured::resolve(insn) {
                    self.lift_structured(insn, &access);
                }
                return;
            }
            VectorShape::Declined => {
                self.stmts.push(IrStmt::Unsupported {
                    mnemonic: insn.mnemonic.clone(),
                    comment: format!(
                        "unmodelled vector shape at {addr} (aarch64)",
                        addr = insn.address
                    ),
                });
                return;
            }
        }
        let mnem = insn.mnemonic.trim().to_ascii_lowercase();
        match mnem.as_str() {
            // Data movement: `mov Rd, Rn/imm`, `movz Rd, #imm`. AArch64
            // `mov` already zero-extends the destination per ISA rules,
            // so it shares the 2-operand `mov`-style handler with x86.
            "mov" | "movz" => self.lift_aarch64_mov(insn),
            // 3-operand arithmetic / logical: `Rd, Rs1, Rs2`. The `s`
            // suffix toggles flag-setting (`adds`, `subs`, `ands`).
            "add" => self.lift_aarch64_arith3(insn, BinOp::Add, false),
            "adds" => self.lift_aarch64_arith3(insn, BinOp::Add, true),
            "sub" => self.lift_aarch64_arith3(insn, BinOp::Sub, false),
            "subs" => self.lift_aarch64_arith3(insn, BinOp::Sub, true),
            "and" => self.lift_aarch64_arith3(insn, BinOp::And, false),
            "ands" => self.lift_aarch64_arith3(insn, BinOp::And, true),
            "orr" => self.lift_aarch64_arith3(insn, BinOp::Or, false),
            "eor" => self.lift_aarch64_arith3(insn, BinOp::Xor, false),
            "mul" => self.lift_aarch64_arith3(insn, BinOp::Mul, false),
            // Integer divide. AArch64 `udiv` / `sdiv` never set NZCV
            // (no `s`-suffixed sibling), so flag emission stays off.
            // SMT-LIB bit-vector division-by-zero gives an all-ones
            // result, which matches what the encoder forwards via
            // `bvudiv` / `bvsdiv`.
            "udiv" => self.lift_aarch64_arith3(insn, BinOp::UDiv, false),
            "sdiv" => self.lift_aarch64_arith3(insn, BinOp::SDiv, false),
            "lsl" => self.lift_aarch64_arith3(insn, BinOp::Shl, false),
            "lsr" => self.lift_aarch64_arith3(insn, BinOp::Shr, false),
            "asr" => self.lift_aarch64_arith3(insn, BinOp::Sar, false),
            // The complemented-Operand2 logicals. A64 spells them as
            // their own mnemonics where A32 has only `bic`; the shape is
            // the plain 3-operand one with `Rm` inverted first.
            "bic" => self.lift_aarch64_logical_not(insn, BinOp::And, false),
            "bics" => self.lift_aarch64_logical_not(insn, BinOp::And, true),
            "orn" => self.lift_aarch64_logical_not(insn, BinOp::Or, false),
            "eon" => self.lift_aarch64_logical_not(insn, BinOp::Xor, false),
            // `mvn Rd, Rm` is `orn Rd, xzr, Rm`, and `neg Rd, Rm` is
            // `sub Rd, xzr, Rm` — both the 2-operand spelling of a
            // 3-operand instruction with the zero register on the left.
            "mvn" => self.lift_aarch64_mvn(insn),
            "neg" => self.lift_aarch64_neg(insn, false),
            "negs" => self.lift_aarch64_neg(insn, true),
            // Compare / test: 2-operand, no destination.
            "cmp" => self.lift_aarch64_cmp(insn),
            // `cmn Rn, Operand` is `adds xzr, Rn, Operand`.
            "cmn" => self.lift_aarch64_cmn(insn),
            "tst" => self.lift_aarch64_tst(insn),
            // Conditional select: `csel Rd, Rn, Rm, cond` → Ite.
            "csel" => self.lift_aarch64_csel(insn),
            // `cset Rd, cond` → Rd = Ite(cond, 1, 0). 2-operand
            // shortcut for `csinc Rd, xzr, xzr, !cond`.
            "cset" => self.lift_aarch64_cset(insn, false),
            // `csetm Rd, cond` → Rd = Ite(cond, -1, 0) (all-ones).
            "csetm" => self.lift_aarch64_cset(insn, true),
            // csel siblings: `csinc Rd, Rn, Rm, cond` → Ite(cond, Rn,
            // Rm+1); `csinv` → ~Rm in the else branch; `csneg` → -Rm.
            "csinc" => self.lift_aarch64_cs_arith(insn, CsArithOp::Inc, false),
            "csinv" => self.lift_aarch64_cs_arith(insn, CsArithOp::Inv, false),
            "csneg" => self.lift_aarch64_cs_arith(insn, CsArithOp::Neg, false),
            // 3-operand aliases: `cinc Rd, Rn, cond` ≡ `csinc Rd, Rn,
            // Rn, !cond`; `cinv` / `cneg` mirror that pattern.
            "cinc" => self.lift_aarch64_cs_arith(insn, CsArithOp::Inc, true),
            "cinv" => self.lift_aarch64_cs_arith(insn, CsArithOp::Inv, true),
            "cneg" => self.lift_aarch64_cs_arith(insn, CsArithOp::Neg, true),
            // Memory loads / stores in offset form `[Xn]` /
            // `[Xn, #imm]` and in the pre / post-index writeback forms
            // (`[Xn, #imm]!` / `[Xn], #imm` / `[Xn], Xm`), which
            // `aarch64_mem_access` resolves and `emit_writeback` then
            // applies to the base. Register-*offset* addressing
            // (`[Xn, Xm]`) still declines to `Unsupported`, so the
            // slice's confidence path picks it up — soundness in
            // lowering, not detection.
            "ldr" => self.lift_aarch64_load(insn, None, false),
            // Sub-word loads zero-extend (`ldrb`/`ldrh`) or sign-extend
            // (`ldrsb`/`ldrsh`/`ldrsw`) into the destination register.
            "ldrb" => self.lift_aarch64_load(insn, Some(8), false),
            "ldrh" => self.lift_aarch64_load(insn, Some(16), false),
            "ldrsb" => self.lift_aarch64_load(insn, Some(8), true),
            "ldrsh" => self.lift_aarch64_load(insn, Some(16), true),
            "ldrsw" => self.lift_aarch64_load(insn, Some(32), true),
            "str" => self.lift_aarch64_store(insn, None),
            "strb" => self.lift_aarch64_store(insn, Some(8)),
            "strh" => self.lift_aarch64_store(insn, Some(16)),
            // Paired load / store `ldp`/`stp Rt, Rt2, [Xn{, #imm}]`:
            // the second element sits one register-width above the
            // first. Unlike the single-register forms above, the pair
            // goes through `aarch64_pair_addresses`, which builds two
            // addresses and no writeback — so the pre / post-index
            // pair forms still decline.
            "ldp" => self.lift_aarch64_ldp(insn),
            "stp" => self.lift_aarch64_stp(insn),
            // Scalar floating point. The lane width comes from the
            // register letter (`s0` → 32, `d0` → 64, `h0` → 16), which
            // `simd_view_bits` already reports, so one handler covers
            // every precision.
            "fadd" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Add),
            "fsub" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Sub),
            "fmul" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Mul),
            // `fmulx` differs from `fmul` on one input class, so it goes
            // through the packed core's lane helper rather than the
            // scalar arithmetic one.
            "fmulx" => self.lift_aarch64_fmulx(insn),
            "fdiv" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Div),
            "fmax" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Max),
            "fmin" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Min),
            "fsqrt" => self.lift_aarch64_fp_sqrt(insn),
            "fabs" => self.lift_aarch64_fp_unary(insn, FpUnaryOp::Abs),
            "fneg" => self.lift_aarch64_fp_unary(insn, FpUnaryOp::Neg),
            "fmov" => self.lift_aarch64_fmov(insn),
            "fcmp" | "fcmpe" => self.lift_aarch64_fcmp(insn),
            "fcvt" => self.lift_aarch64_fcvt(insn),
            "scvtf" => self.lift_aarch64_int_to_fp(insn, true),
            "ucvtf" => self.lift_aarch64_int_to_fp(insn, false),
            "fcvtzs" => self.lift_aarch64_fp_to_int(insn, true),
            "fcvtzu" => self.lift_aarch64_fp_to_int(insn, false),
            // Round to an integral value *keeping the float sort*. Five
            // of the seven name their mode in the opcode, so they are
            // exact whatever FPCR holds. `frinti` reads FPCR, and
            // `frintx` computes the same value while additionally being
            // able to signal the inexact exception — a flag the value
            // model does not carry, so the two lower identically.
            //
            // The `FEAT_FRINTTS` members (`frint32z` / `frint32x` /
            // `frint64z` / `frint64x`) come through the same arm: the
            // table they all consult is the packed resolver's, so the
            // scalar and packed spellings cannot drift about either the
            // mode or the saturation width.
            mnemonic if let Some((rounding, saturate)) = neon::round_to_integral_kind(mnemonic) => {
                self.lift_aarch64_fp_round(insn, rounding, saturate);
            }
            "frecpx" => self.lift_aarch64_frecpx(insn),
            _ => self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("at {addr} (aarch64)", addr = insn.address),
            }),
        }
    }

    pub(super) fn lift_aarch64_mov(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (aarch64 mov)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (aarch64 mov)".into(),
            });
            return;
        };
        let value = self.read_operand_at(src, dst_width);
        if !self.write_register_to(dst, value) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (aarch64 mov)".into(),
            });
        }
    }

    /// `bic` / `bics` / `orn` / `eon` — the 3-operand logicals whose
    /// second source is complemented: `Rd := Rn <op> NOT Op`.
    ///
    /// Only the `Rm` read differs from [`Self::lift_aarch64_arith3`], so
    /// the flag policy, the destination write and the temp that orders
    /// them all come from `emit_arith3` — which is also what keeps
    /// `bics` on the same C-and-V answer as `ands` rather than a second
    /// copy of it.
    ///
    /// The complement is applied **after** any trailing shift, which is
    /// the order the architecture defines: `bic x0, x1, x2, lsl 4`
    /// clears the shifted bits, not the shifted complement.
    fn lift_aarch64_logical_not(&mut self, insn: &Instruction, op: BinOp, sets_flags: bool) {
        let (Some(dst), Some(src1), Some(_)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "fewer than 3 operands (aarch64 complemented logical)".into(),
            });
            return;
        };
        let Some(width) = self.aarch64_destination_width(insn, dst) else {
            return;
        };
        let lhs = self.read_operand_at(src1, width);
        let rhs = self.read_source_with_shift(insn, OPERAND2_INDEX_ARITH3, width);
        let rhs = Expr::bv_xor(rhs, Expr::konst(u128::from(width_mask(width)), width));
        self.emit_arith3(insn, op, &lhs, &rhs, width, sets_flags);
    }

    /// `mvn Rd, Op` — `orn Rd, xzr, Op`, i.e. the bitwise complement.
    /// Never sets flags: A64 spells no `mvns`.
    fn lift_aarch64_mvn(&mut self, insn: &Instruction) {
        let Some(dst) = insn.operands.first() else {
            return;
        };
        let Some(width) = self.aarch64_destination_width(insn, dst) else {
            return;
        };
        let src = self.read_source_with_shift(insn, OPERAND2_INDEX_COMPARE, width);
        let value = Expr::bv_xor(src, Expr::konst(u128::from(width_mask(width)), width));
        if !self.write_register_to(dst, value) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (aarch64 mvn)".into(),
            });
        }
    }

    /// `neg` / `negs Rd, Op` — `sub` / `subs Rd, xzr, Op`.
    ///
    /// Routed through `emit_arith3` with a literal zero as the left
    /// source so `negs` gets the `Sub` arm's precise carry rather than
    /// the catch-all's Unknown.
    fn lift_aarch64_neg(&mut self, insn: &Instruction, sets_flags: bool) {
        let Some(dst) = insn.operands.first() else {
            return;
        };
        let Some(width) = self.aarch64_destination_width(insn, dst) else {
            return;
        };
        let rhs = self.read_source_with_shift(insn, OPERAND2_INDEX_COMPARE, width);
        let lhs = Expr::konst(0, width);
        self.emit_arith3(insn, BinOp::Sub, &lhs, &rhs, width, sets_flags);
    }

    /// `cmn Rn, Op` — `adds xzr, Rn, Op`: flags from the sum, no
    /// register destination.
    fn lift_aarch64_cmn(&mut self, insn: &Instruction) {
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, width);
        let rhs = self.read_source_with_shift(insn, OPERAND2_INDEX_COMPARE, width);
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), Expr::add(lhs.clone(), rhs.clone()));
        let tmp_expr = Expr::Var(tmp);
        // `Add` reaches the catch-all arm, which leaves C and V Unknown
        // — a carry-out needs an extension bit the flag helper does not
        // plumb. Sharing the helper is what keeps that answer identical
        // to `adds`, which is the instruction `cmn` actually is.
        self.aarch64_set_arith_flags(insn, BinOp::Add, &lhs, &rhs, &tmp_expr, width);
    }

    /// The destination width every handler above needs, with the two
    /// shapes that have no answer declined before anything is emitted —
    /// a decline that has already pushed half an instruction is not a
    /// decline. Mirrors the guards at the head of
    /// [`Self::lift_aarch64_arith3`].
    fn aarch64_destination_width(&mut self, insn: &Instruction, dst: &Operand) -> Option<u16> {
        if dst.kind != OperandKind::Register || self.is_simd_register(dst) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-GPR destination (aarch64)".into(),
            });
            return None;
        }
        let width = nonzero_width(self.operand_width(dst));
        if width.is_none() {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (aarch64)".into(),
            });
        }
        width
    }

    pub(super) fn lift_aarch64_arith3(&mut self, insn: &Instruction, op: BinOp, sets_flags: bool) {
        let (Some(dst), Some(src1), Some(_)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "fewer than 3 operands (aarch64)".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (aarch64)".into(),
            });
            return;
        }
        // The scalar-SIMD spellings (`add d0, d1, d2`) reach here: they
        // carry no arrangement, so the vector gate does not claim them,
        // and this helper has no vector lowering for them. Declining up
        // front rather than at the destination write is what keeps the
        // reads of `d1` / `d2` out of `stmts` — a decline that has
        // already emitted half an instruction is not a decline.
        if self.is_simd_register(dst) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "scalar-SIMD destination (aarch64)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (aarch64)".into(),
            });
            return;
        };
        let lhs = self.read_operand_at(src1, dst_width);
        // The last source can carry a shift or extend specifier as one
        // more operand (`eor r0, r1, r2, lsl 2`); reading it alone drops
        // the shift and computes a wrong value.
        let rhs = self.read_source_with_shift(insn, 2, dst_width);
        self.emit_arith3(insn, op, &lhs, &rhs, dst_width, sets_flags);
    }

    /// Commit a three-operand arithmetic result: temp, flags, then the
    /// destination write.
    ///
    /// Separate from [`Self::lift_aarch64_arith3`] because `rsb` needs
    /// the same tail with its sources the other way round, and cloning
    /// the instruction to swap its operands — which is what it used to
    /// do — cannot survive a trailing shift specifier: the specifier
    /// belongs to `Operand2`, and after the swap that operand is no
    /// longer where the reader looks for it.
    ///
    /// The temp is what makes the flag order right. `adds x0, x0, x1` is
    /// legal, and without stashing the result first the `x0` reads
    /// inside CF would be renamed by SSA to the post-write version. See
    /// `lift_add_sub` for the x86 analogue and the recorded regression
    /// in `r2smt_lifter_sub_flag_bug.md`.
    pub(super) fn emit_arith3(
        &mut self,
        insn: &Instruction,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        width: u16,
        sets_flags: bool,
    ) {
        let Some(dst) = insn.operands.first() else {
            return;
        };
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), op.apply(lhs.clone(), rhs.clone()));
        let tmp_expr = Expr::Var(tmp);
        if sets_flags {
            self.aarch64_set_arith_flags(insn, op, lhs, rhs, &tmp_expr, width);
        }
        if !self.write_register_to(dst, tmp_expr) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (aarch64)".into(),
            });
        }
    }

    /// Set NZCV-equivalent flags (using the x86 polarity convention)
    /// after a flag-setting `AArch64` arithmetic / logical instruction.
    ///
    /// The condition-code mapping in [`crate::condition`] expects:
    /// - ZF = (result == 0).
    /// - SF (= N) = msb(result).
    /// - CF = (lhs < rhs) unsigned (x86 borrow polarity — opposite of
    ///   ARM's architectural C). Modelled precisely for `sub` /
    ///   `subs` / `cmp`; left Unknown for `adds` (carry-out needs an
    ///   extension bit we don't yet plumb).
    /// - OF (= V) = signed overflow — left Unknown for now;
    ///   downstream confidence machinery already downgrades
    ///   signed-comparison verdicts when OF is Unknown.
    /// - PF — irrelevant on `AArch64`; left Unknown.
    ///
    /// The logical and multiply families are **not** the same on the two
    /// ISAs, and one shared arm would fabricate a flag on whichever ISA
    /// lost. A64 `ANDS` clears C and V — and *clearing* C means the
    /// pipeline's `CF`, which holds ARM's `C` inverted, must be written
    /// **one**; V carries no such inversion and is written zero. Writing
    /// both zero, as this did, makes `b.hs` after an `ands` resolve to
    /// the opposite arm. A32 `ANDS` / `ORRS` / `EORS`
    /// takes C from the shifter carry-out of Operand2 and leaves V alone,
    /// and `MULS` from ARM v6 on leaves both alone. A32 has no `ORRS` / `EORS`
    /// counterpart on A64 at all, so those two arms used to be reachable
    /// only from `AArch32` and were unconditionally wrong there.
    fn aarch64_set_arith_flags(
        &mut self,
        insn: &Instruction,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        result: &Expr,
        width: u16,
    ) {
        self.set_flag("ZF", Expr::eq(result.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(result.clone(), Expr::konst(0, width)));
        let arm32 = self.arch == Arch::Arm;
        match op {
            BinOp::Sub => {
                self.set_flag("CF", Expr::ult(lhs.clone(), rhs.clone()));
                self.set_flag("OF", Expr::Unknown(String::new()));
            }
            BinOp::And | BinOp::Or | BinOp::Xor if arm32 => {
                self.set_aarch32_logical_carry(insn, OPERAND2_INDEX_ARITH3);
            }
            BinOp::And | BinOp::Or | BinOp::Xor => {
                self.set_flag("CF", stored_carry(Expr::konst(0, 1)));
                self.set_flag("OF", Expr::konst(0, 1));
            }
            BinOp::Mul if arm32 => {}
            // `adds` and the rest need a full extension to compute carry
            // precisely; mark Unknown rather than fabricate a value.
            _ => {
                self.set_flag("CF", Expr::Unknown(String::new()));
                self.set_flag("OF", Expr::Unknown(String::new()));
            }
        }
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    pub(super) fn lift_aarch64_cmp(&mut self, insn: &Instruction) {
        // AArch64 `cmp Rn, Operand` = `subs xzr, Rn, Operand` — sets
        // flags from Rn - Operand, no register destination.
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, width);
        // `cmp r0, r1, lsl 2` is three operands, one more than this
        // reads, so the shift has to be folded in here too.
        let rhs = self.read_source_with_shift(insn, 1, width);
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), Expr::sub(lhs.clone(), rhs.clone()));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, width)));
        self.set_flag("CF", Expr::ult(lhs, rhs));
        self.set_flag("OF", Expr::Unknown(String::new()));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    fn lift_aarch64_csel(&mut self, insn: &Instruction) {
        // `csel Rd, Rn, Rm, cond` → Rd = Ite(cond, Rn, Rm).
        let (Some(dst), Some(rn), Some(rm), Some(cond_op)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
            insn.operands.get(3),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "csel needs 4 operands".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (csel)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (csel)".into(),
            });
            return;
        };
        let Some(cond_expr) = aarch64_cond_suffix_to_predicate(&cond_op.raw) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "unrecognised csel cond".into(),
            });
            return;
        };
        let then_value = self.read_operand_at(rn, dst_width);
        let else_value = self.read_operand_at(rm, dst_width);
        let ite = Expr::Ite {
            cond: Box::new(cond_expr),
            then_expr: Box::new(then_value),
            else_expr: Box::new(else_value),
        };
        if !self.write_register_to(dst, ite) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (csel)".into(),
            });
        }
    }

    fn lift_aarch64_cs_arith(&mut self, insn: &Instruction, op: CsArithOp, aliased: bool) {
        // Conditional-select arithmetic family. Layout depends on
        // whether this is a primary mnemonic or a short alias:
        //
        //   `csinc Rd, Rn, Rm, cond`  (op count = 4)
        //   `cinc  Rd, Rn, cond`      (op count = 3, Rm := Rn, cond
        //                              negated)
        //
        // The else branch's expression varies by `op`:
        //   Inc → Rm + 1
        //   Inv → ~Rm (bitwise NOT, encoded as Xor with all-ones)
        //   Neg → -Rm (encoded as 0 - Rm)
        let dst_op = insn.operands.first();
        let lhs_operand = insn.operands.get(1);
        let (rhs_operand, cond_operand) = if aliased {
            (insn.operands.get(1), insn.operands.get(2))
        } else {
            (insn.operands.get(2), insn.operands.get(3))
        };
        let (Some(dst), Some(rn), Some(rm), Some(cond_raw)) =
            (dst_op, lhs_operand, rhs_operand, cond_operand)
        else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("missing operands ({})", insn.mnemonic),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (cs* family)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (cs* family)".into(),
            });
            return;
        };
        let Some(mut cond_expr) = aarch64_cond_suffix_to_predicate(&cond_raw.raw) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "unrecognised cs* cond".into(),
            });
            return;
        };
        if aliased {
            cond_expr = Expr::bool_not(cond_expr);
        }
        let then_value = self.read_operand_at(rn, dst_width);
        let rm_value = self.read_operand_at(rm, dst_width);
        let else_value = match op {
            CsArithOp::Inc => Expr::add(rm_value, Expr::konst(1, dst_width)),
            CsArithOp::Inv => Expr::bv_xor(
                rm_value,
                Expr::konst(u128::from(width_mask(dst_width)), dst_width),
            ),
            CsArithOp::Neg => Expr::sub(Expr::konst(0, dst_width), rm_value),
        };
        let ite = Expr::Ite {
            cond: Box::new(cond_expr),
            then_expr: Box::new(then_value),
            else_expr: Box::new(else_value),
        };
        if !self.write_register_to(dst, ite) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (cs* family)".into(),
            });
        }
    }

    fn lift_aarch64_cset(&mut self, insn: &Instruction, all_ones: bool) {
        // `cset Rd, cond` → Rd = Ite(cond, 1, 0); `csetm` uses
        // all-ones in the true branch.
        let (Some(dst), Some(cond_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "cset/csetm needs 2 operands".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (cset)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (cset)".into(),
            });
            return;
        };
        let Some(cond_expr) = aarch64_cond_suffix_to_predicate(&cond_op.raw) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "unrecognised cset cond".into(),
            });
            return;
        };
        let true_val = if all_ones {
            // `csetm` writes all-ones — represent as 0 - 1 of dst_width
            // (a single Const at the right width).
            Expr::konst(u128::from(width_mask(dst_width)), dst_width)
        } else {
            Expr::konst(1, dst_width)
        };
        let ite = Expr::Ite {
            cond: Box::new(cond_expr),
            then_expr: Box::new(true_val),
            else_expr: Box::new(Expr::konst(0, dst_width)),
        };
        if !self.write_register_to(dst, ite) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (cset)".into(),
            });
        }
    }

    /// `tst Rn, Operand` = `ands xzr, Rn, Operand` — flags from
    /// `Rn AND Operand`, no register destination. `AArch32` `tst` routes
    /// here too, and the two ISAs part company on C and V exactly as they
    /// do for `ands`; see [`Self::aarch64_set_arith_flags`].
    pub(super) fn lift_aarch64_tst(&mut self, insn: &Instruction) {
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, width);
        let rhs = self.read_source_with_shift(insn, 1, width);
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), Expr::bv_and(lhs, rhs));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, width)));
        if self.arch == Arch::Arm {
            self.set_aarch32_logical_carry(insn, OPERAND2_INDEX_COMPARE);
        } else {
            self.set_flag("CF", stored_carry(Expr::konst(0, 1)));
            self.set_flag("OF", Expr::konst(0, 1));
        }
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    /// `ldr Rd, [Xn{, #imm}]` — read `Rd`-width bytes from memory and
    /// write them to the destination register. W-form (`ldr Wd, …`)
    /// zero-extends to the parent X per the `AArch64` ABI via
    /// [`LiftCtx::write_register_to`]. Writeback (`[Xn, …]!`) and
    /// register-offset addressing decline to `Unsupported` so the
    /// confidence path picks them up rather than silently widening.
    /// Lift an `AArch64` load. `width_override` is `Some(8/16/32)` for
    /// the sub-word forms (`ldrb`/`ldrh`/`ldrsw` …); `None` for the
    /// natural-width `ldr`. `signed` selects sign- vs zero-extension
    /// into the destination register.
    pub(super) fn lift_aarch64_load(
        &mut self,
        insn: &Instruction,
        width_override: Option<u16>,
        signed: bool,
    ) {
        let (Some(dst), Some(mem)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if dst.kind != OperandKind::Register || mem.kind != OperandKind::Memory {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldr operand shape (non-Register/non-Memory)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldr zero-width destination".into(),
            });
            return;
        };
        let load_width = width_override.unwrap_or(dst_width);
        let Some(access) = aarch64_mem_access(mem, insn.operands.get(2), self.bits) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("ldr addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        let MemAccess { address, writeback } = access;
        // Two-statement lower: load into a fresh temp at the load
        // width, then write that temp into the destination register
        // so `write_register_to` zero-extends to the parent X for
        // the W-form (mirrors the `add` / `sub` flag-ordering
        // precedent: stash-then-write). The base-register writeback
        // (pre/post-index) is emitted last so its `Xn` read is the
        // pre-write value under SSA.
        let tmp = self.new_temp(insn.address, load_width);
        self.stmts.push(IrStmt::LoadMem {
            dst: tmp.clone(),
            address,
            bits: load_width,
        });
        let value = if load_width < dst_width {
            if signed {
                Expr::sign_ext(Expr::Var(tmp), dst_width)
            } else {
                Expr::zero_ext(Expr::Var(tmp), dst_width)
            }
        } else {
            Expr::Var(tmp)
        };
        // A vector destination takes the vector writer: `ldr d8` clears
        // the rest of `v8`, and the scalar path would size that parent
        // at the pointer width.
        let written = if self.is_simd_register(dst) {
            self.write_simd_dst(dst, value, true)
        } else {
            self.write_register_to(dst, value)
        };
        if !written {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldr destination not a supported register".into(),
            });
        }
        self.emit_writeback(writeback);
    }

    /// `str Rs, [Xn{, #imm}]` — write the source register (optionally
    /// truncated for `strb`/`strh`) to memory. See
    /// [`Self::lift_aarch64_load`] for the addressing-mode restrictions.
    pub(super) fn lift_aarch64_store(&mut self, insn: &Instruction, width_override: Option<u16>) {
        let (Some(src), Some(mem)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if src.kind != OperandKind::Register || mem.kind != OperandKind::Memory {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "str operand shape (non-Register/non-Memory)".into(),
            });
            return;
        }
        let Some(src_width) = nonzero_width(self.operand_width(src)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "str zero-width source".into(),
            });
            return;
        };
        let store_width = width_override.unwrap_or(src_width);
        let Some(access) = aarch64_mem_access(mem, insn.operands.get(2), self.bits) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("str addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        let value = self.read_transfer_operand(src, store_width);
        self.stmts.push(IrStmt::StoreMem {
            address: access.address,
            value,
            bits: store_width,
        });
        self.emit_writeback(access.writeback);
    }

    /// Read the source of a memory transfer, which may name either
    /// register file: `str x0, [sp]` and `str d8, [sp]` differ only in
    /// where the source lives, and both are ordinary in a prologue.
    ///
    /// The vector one has to go through the lane reader. Not for
    /// precision — [`LiftCtx::read_operand_at`] would refuse it and the
    /// store would keep an opaque value, which is sound — but because
    /// that refusal costs a real instruction: saving `d8`–`d15` across a
    /// call is what the ABI asks for, so a slice that reads the slot
    /// back would lose the value on the way through.
    fn read_transfer_operand(&mut self, op: &Operand, width: u16) -> Expr {
        if self.is_simd_register(op)
            && let Some(bits) = self.read_simd_lane_bits(op, width, 0)
        {
            return bits;
        }
        self.read_operand_at(op, width)
    }

    /// Emit the base-register mutation for a pre/post-index memory
    /// access: `Xn := Xn + delta` at pointer width. A no-op when the
    /// access has no writeback. Emitted after the load/store so the
    /// `Xn` reads inside the access address stay the pre-write value
    /// under SSA rename.
    pub(super) fn emit_writeback(&mut self, writeback: Option<Writeback>) {
        let Some(Writeback { base, delta }) = writeback else {
            return;
        };
        let base_var = Expr::Var(Var::new(&base, self.bits));
        self.assign(Var::new(&base, self.bits), Expr::add(base_var, delta));
    }

    pub(super) fn lift_aarch64_ldp(&mut self, insn: &Instruction) {
        let (Some(d0), Some(d1), Some(mem)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldp expects Rt, Rt2, [mem]".into(),
            });
            return;
        };
        if d0.kind != OperandKind::Register || d1.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldp operand shape (non-Register pair)".into(),
            });
            return;
        }
        let (Some(w0), Some(w1)) = (
            nonzero_width(self.operand_width(d0)),
            nonzero_width(self.operand_width(d1)),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldp zero-width destination".into(),
            });
            return;
        };
        let Some((first, second)) = self.aarch64_pair_addresses(mem, w0) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("ldp addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        self.load_into_register(insn, d0, first, w0);
        self.load_into_register(insn, d1, second, w1);
    }

    pub(super) fn lift_aarch64_stp(&mut self, insn: &Instruction) {
        let (Some(s0), Some(s1), Some(mem)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "stp expects Rt, Rt2, [mem]".into(),
            });
            return;
        };
        if s0.kind != OperandKind::Register || s1.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "stp operand shape (non-Register pair)".into(),
            });
            return;
        }
        let (Some(w0), Some(w1)) = (
            nonzero_width(self.operand_width(s0)),
            nonzero_width(self.operand_width(s1)),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "stp zero-width source".into(),
            });
            return;
        };
        let Some((first, second)) = self.aarch64_pair_addresses(mem, w0) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("stp addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        let (v0, v1) = (
            self.read_transfer_operand(s0, w0),
            self.read_transfer_operand(s1, w1),
        );
        self.stmts.push(IrStmt::StoreMem {
            address: first,
            value: v0,
            bits: w0,
        });
        self.stmts.push(IrStmt::StoreMem {
            address: second,
            value: v1,
            bits: w1,
        });
    }

    /// Two-statement load: into a fresh temp at `width`, then written to
    /// `dst` so `write_register_to` zero-extends the parent X for the
    /// W-form (the P26 `ldr` stash-then-write precedent).
    ///
    /// A vector destination (`ldr d8, [sp, #0x10]`) takes the vector
    /// writer instead, for the same two reasons every scalar SIMD write
    /// here does: the parent is 128 bits wide and the write clears
    /// everything above the element.
    fn load_into_register(&mut self, insn: &Instruction, dst: &Operand, address: Expr, width: u16) {
        let tmp = self.new_temp(insn.address, width);
        self.stmts.push(IrStmt::LoadMem {
            dst: tmp.clone(),
            address,
            bits: width,
        });
        let written = if self.is_simd_register(dst) {
            self.write_simd_dst(dst, Expr::Var(tmp), true)
        } else {
            self.write_register_to(dst, Expr::Var(tmp))
        };
        if !written {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldp destination not a supported register".into(),
            });
        }
    }

    /// Address of the two paired elements: the base address and the base
    /// plus one element (`first_width / 8` bytes). Returns `None` for the
    /// same writeback / register-offset forms the single `ldr` declines.
    fn aarch64_pair_addresses(&self, mem: &Operand, first_width: u16) -> Option<(Expr, Expr)> {
        if mem.kind != OperandKind::Memory {
            return None;
        }
        let first = aarch64_address_expr(mem, self.bits)?;
        let stride = u64::from(first_width / 8) & width_mask(self.bits);
        let second = Expr::add(first.clone(), Expr::konst(u128::from(stride), self.bits));
        Some((first, second))
    }
}

/// Parse an `AArch64` memory operand in the supported offset forms
/// (`[Xn]` / `[Xn, #imm]` / `[Xn, imm]`) into a symbolic address
/// expression `base ± offset` at the pointer width.
///
/// Returns `None` for writeback (`[Xn, …]!` / `[Xn], …`) and
/// register-offset (`[Xn, Xm{, lsl #k}]`) forms — those addressing
/// modes also mutate the base register and need an extra ordered
/// `Assign`, which a later P26 follow-up will add. Rejecting cleanly
/// here keeps the lifter sound (the caller emits `Unsupported` and
/// the confidence path widens) rather than silently dropping the
/// writeback effect.
fn aarch64_address_expr(mem: &Operand, ptr_bits: u16) -> Option<Expr> {
    let (base, offset) = parse_aarch64_memory(&mem.raw)?;
    let parent =
        register_layout(&base, r2smt_common::Arch::Aarch64).map_or(base.as_str(), |l| l.parent);
    let base_var = Expr::Var(Var::new(parent, ptr_bits));
    if offset == 0 {
        return Some(base_var);
    }
    // Bit-pattern reinterpretation of `offset` as `u64`: negative
    // offsets carry their two's-complement representation, which
    // `bvadd` then folds correctly at `ptr_bits` (the encoder reads
    // the constant as the unsigned representation of a negative
    // integer at that width). Going through `to_le_bytes` keeps the
    // conversion explicit (no `as` sign loss) and platform-stable.
    let masked = u64::from_le_bytes(offset.to_le_bytes()) & width_mask(ptr_bits);
    let off_const = Expr::konst(u128::from(masked), ptr_bits);
    Some(Expr::add(base_var, off_const))
}

/// Resolve an `AArch64` memory operand, including the pre-index
/// (`[Xn, #imm]!`) and post-index (`[Xn], #imm`) writeback forms.
/// `post` is the instruction's trailing operand, which for a post-index
/// access is the immediate offset. Register-offset and other
/// unrecognised shapes still return `None` (sound decline).
fn aarch64_mem_access(mem: &Operand, post: Option<&Operand>, ptr_bits: u16) -> Option<MemAccess> {
    if mem.kind != OperandKind::Memory {
        return None;
    }
    let raw = mem.raw.trim();
    // Pre-index: `[Xn, #imm]!` — address is Xn+imm, then Xn := Xn+imm.
    if let Some(body) = raw.strip_suffix('!') {
        let (base, offset) = parse_aarch64_memory(body.trim())?;
        let parent = aarch64_base_parent(&base);
        return Some(MemAccess {
            address: aarch64_addr_from(parent, offset, ptr_bits),
            writeback: Some(Writeback::by_constant(parent, offset, ptr_bits)),
        });
    }
    let (base, offset) = parse_aarch64_memory(raw)?;
    let parent = aarch64_base_parent(&base);
    // Post-index: `[Xn], #imm` or `[Xn], Xm` — a bare base plus a
    // trailing operand; the address is Xn, and Xn is then advanced by
    // that operand. Returning `None` for a trailing operand we cannot
    // read is what keeps the base update from being silently dropped.
    if let Some(op) = post
        && offset == 0
    {
        return Some(MemAccess {
            address: Expr::Var(Var::new(parent, ptr_bits)),
            writeback: Some(Writeback {
                base: parent.to_string(),
                delta: aarch64_post_index_delta(op, ptr_bits)?,
            }),
        });
    }
    Some(MemAccess {
        address: aarch64_addr_from(parent, offset, ptr_bits),
        writeback: None,
    })
}

/// The amount a post-index operand advances the base register by.
///
/// An immediate is the scalar forms' fixed stride. A register is the
/// structured accesses' variable one (`ld1 {v0.16b}, [x0], x3`), which
/// no constant can express — it is the reason a writeback delta is an
/// expression at all. Only the 64-bit spelling is accepted: the
/// architecture encodes no narrower post-index register, and widening
/// one would invent a zero extension the instruction never performs.
///
/// The zero register declines rather than reading free bits: `Rm = 31`
/// selects the *immediate* form in the encoding, so the spelling cannot
/// arise, and a free `xzr` would model an unknown stride where the
/// architecture has none.
fn aarch64_post_index_delta(op: &Operand, ptr_bits: u16) -> Option<Expr> {
    match op.kind {
        OperandKind::Immediate => {
            let raw = op.raw.strip_prefix('#').unwrap_or(&op.raw).trim();
            let delta = parse_signed_immediate(raw)?;
            Some(constant_delta(delta, ptr_bits))
        }
        OperandKind::Register => {
            let layout = register_layout(&op.raw, r2smt_common::Arch::Aarch64)?;
            let usable =
                layout.parent != ZERO_REGISTER && layout.lo == 0 && layout.width() == ptr_bits;
            usable.then(|| Expr::Var(Var::new(layout.parent, ptr_bits)))
        }
        _ => None,
    }
}

/// Canonical parent name of the `AArch64` zero register.
const ZERO_REGISTER: &str = "xzr";

/// The canonical parent register name for an addressing base.
fn aarch64_base_parent(base: &str) -> &str {
    register_layout(base, r2smt_common::Arch::Aarch64).map_or(base, |l| l.parent)
}

/// Build `base ± offset` at the pointer width (base alone if `offset`
/// is zero). Shares the two's-complement offset handling with
/// [`aarch64_address_expr`].
fn aarch64_addr_from(parent: &str, offset: i64, ptr_bits: u16) -> Expr {
    let base_var = Expr::Var(Var::new(parent, ptr_bits));
    if offset == 0 {
        return base_var;
    }
    let masked = u64::from_le_bytes(offset.to_le_bytes()) & width_mask(ptr_bits);
    Expr::add(base_var, Expr::konst(u128::from(masked), ptr_bits))
}

/// Parse `[base{, #?offset}]` into `(base, offset)`. Returns `None`
/// for any shape outside the supported subset (writeback, register
/// offset, shift modifiers, malformed input).
fn parse_aarch64_memory(raw: &str) -> Option<(String, i64)> {
    let trimmed = raw.trim();
    let body = trimmed.strip_prefix('[')?.strip_suffix(']')?;
    // Writeback suffix `]!` was stripped by `strip_suffix(']')`, so a
    // remaining `!` (e.g. inside the brackets — unusual) is still
    // rejected via the comma-split below; the post-index form
    // `[base], #imm` keeps the `, #imm` *outside* the brackets and
    // therefore fails the `strip_suffix(']')` check above.
    let parts: Vec<&str> = body.split(',').map(str::trim).collect();
    match parts.as_slice() {
        [base] => {
            if !is_valid_aarch64_base(base) {
                return None;
            }
            Some((base.to_ascii_lowercase(), 0))
        }
        [base, offset] => {
            if !is_valid_aarch64_base(base) {
                return None;
            }
            let off_str = offset.strip_prefix('#').unwrap_or(offset).trim();
            let value = parse_signed_immediate(off_str)?;
            Some((base.to_ascii_lowercase(), value))
        }
        // Three+ comma-separated parts implies a register-offset
        // with shift (`[x0, x1, lsl #3]`) or some other unsupported
        // shape — decline.
        _ => None,
    }
}

fn is_valid_aarch64_base(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();
    // Only the architectural addressing registers — `x0..x30`, `sp`,
    // and analyst aliases (`lr`, `fp`) that resolve through
    // `register_layout`. `wN` reads are rejected: AArch64 addressing
    // is 64-bit, a `Wn` base would be a malformed disassembly.
    register_layout(&lower, r2smt_common::Arch::Aarch64).is_some_and(|l| l.parent != "xzr")
        && !lower.starts_with('w')
}

fn parse_signed_immediate(raw: &str) -> Option<i64> {
    let s = raw.trim();
    let (negative, body) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest.trim())
    } else if let Some(rest) = s.strip_prefix('+') {
        (false, rest.trim())
    } else {
        (false, s)
    };
    let magnitude = if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()?
    } else {
        body.parse::<i64>().ok()?
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// The rounding mode FPCR holds out of reset (`RMode == 0b00`), which
/// the forms that read the control register are lifted against.
///
/// Pinning it is the same bargain the arithmetic already takes, and
/// [`crate::lift::pins_rounding_mode`] lists those forms so a function
/// that reprograms FPCR truncates the slice instead of trusting this.
const FPCR_DEFAULT_ROUNDING: RoundingMode = RoundingMode::NearestTiesEven;

/// Unary scalar floating-point operation.
#[derive(Clone, Copy)]
pub(super) enum FpUnaryOp {
    /// `fabs` — clear the sign bit.
    Abs,
    /// `fneg` — flip the sign bit.
    Neg,
}

impl LiftCtx {
    /// `fadd`/`fsub`/`fmul`/`fdiv`/`fmax`/`fmin Rd, Rn, Rm` — scalar
    /// floating point at the precision the register letter names.
    ///
    /// A write through a scalar view zeroes the rest of the vector
    /// register per `AArch64` SIMD&FP semantics, unlike legacy SSE,
    /// which merges — hence `zero_upper`.
    fn lift_aarch64_fp_arith3(&mut self, insn: &Instruction, op: FpArithOp) {
        let ops = &insn.operands;
        let (Some(dst), Some(lhs), Some(rhs)) = (ops.first(), ops.get(1), ops.get(2)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(a) = self.read_simd_lane_bits(lhs, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(b) = self.read_simd_lane_bits(rhs, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // `fmax` / `fmin` are `FPMax` / `FPMin`, which propagate NaN
        // and combine the signs of a zero tie. [`fp_lane_result`] is
        // Intel's `MAXPS` and does neither.
        let result = match op {
            FpArithOp::Max | FpArithOp::Min => {
                fp_propagating_max_min(a, b, lane, matches!(op, FpArithOp::Max), false)
            }
            FpArithOp::Add | FpArithOp::Sub | FpArithOp::Mul | FpArithOp::Div => {
                fp_lane_result(op, a, b, lane)
            }
        };
        let Some(result) = result else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fsqrt Rd, Rn`.
    fn lift_aarch64_fp_sqrt(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let root = Expr::fp_to_ieee_bv(Expr::fsqrt(value, RoundingMode::NearestTiesEven));
        if !self.write_simd_dst(dst, root, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `frint<mode> Rd, Rn` — round to an integral value without
    /// leaving the float sort.
    ///
    /// Not the integer round trip `fcvtzs` + `scvtf`: that agrees only
    /// on the values an integer register can hold, and turns an
    /// infinity, a NaN or an out-of-range magnitude into some in-range
    /// number — a wrong value rather than a lost one.
    ///
    /// `saturate` is the `FEAT_FRINTTS` clamp (`frint32z` / `frint32x` /
    /// `frint64z` / `frint64x`), which keeps the float sort but bounds
    /// the value to what a signed integer of that width holds. It shares
    /// [`neon::lower::round_to_integral_lane`] with the packed spelling
    /// so the two cannot disagree about the bound.
    fn lift_aarch64_fp_round(
        &mut self,
        insn: &Instruction,
        rounding: RoundingMode,
        saturate: Option<u16>,
    ) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(rounded) = neon::lower::round_to_integral_lane(value, lane, rounding, saturate)
        else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, rounded, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fmulx Rd, Rn, Rm` — the scalar spelling of the extended
    /// multiply.
    ///
    /// Shares [`crate::lift::simd::fp_multiply_extended_lane`] with the
    /// packed and by-element forms, so the `inf * 0` answer and its sign
    /// cannot differ between spellings of one instruction.
    fn lift_aarch64_fmulx(&mut self, insn: &Instruction) {
        let (Some(dst), Some(a), Some(b)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let (Some(a_bits), Some(b_bits)) = (
            self.read_simd_lane_bits(a, lane, 0),
            self.read_simd_lane_bits(b, lane, 0),
        ) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(result) = crate::lift::simd::fp_multiply_extended_lane(&a_bits, &b_bits, lane)
        else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `frecpx Rd, Rn` — the reciprocal *exponent*.
    ///
    /// Shares a prefix with `frecpe` and almost nothing else, which is
    /// the hazard worth naming. The estimate's value is
    /// implementation-defined inside an error bound, so it lowers to a
    /// free value; `FPRecpX` is defined exactly — the exponent field
    /// complemented, the significand cleared, the sign kept — so lowering
    /// it as a free value would throw away a fact the architecture
    /// guarantees, and lowering `frecpe` this way would invent one.
    ///
    /// Built on the bit pattern rather than on a float sort, which keeps
    /// the binary16 form renderable by the text backends as well as by
    /// Z3. The NaN arm reproduces `FPProcessNaN` under the reset FPCR
    /// (`DN == 0`): the operand with its quiet bit forced on, which
    /// leaves a quiet NaN alone and quietens a signalling one.
    ///
    /// There is no vector form of this instruction, so no arrangement
    /// ever reaches here.
    fn lift_aarch64_frecpx(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(result) = reciprocal_exponent(bits, lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fabs`/`fneg Rd, Rn` — sign-bit manipulation, modelled on the
    /// bit pattern rather than as arithmetic so no rounding is implied.
    fn lift_aarch64_fp_unary(&mut self, insn: &Instruction, op: FpUnaryOp) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(sign) = sign_bit_mask(lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let result = match op {
            // `abs` clears the sign bit: AND with its complement.
            FpUnaryOp::Abs => Expr::bv_and(bits, Expr::bv_xor(sign, super::simd::all_ones(lane))),
            FpUnaryOp::Neg => Expr::bv_xor(bits, sign),
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fmov` in its three shapes: vector-to-vector, general-to-vector
    /// and vector-to-general. All three move a bit pattern without
    /// interpreting it, so none of them rounds.
    fn lift_aarch64_fmov(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        match (self.is_simd_register(dst), self.is_simd_register(src)) {
            (true, true) => {
                let Some(lane) = self.simd_view_bits(dst) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                if !self.write_simd_dst(dst, bits, true) {
                    self.push_aarch64_fp_unsupported(insn);
                }
            }
            // `fmov s0, w0` — the general register supplies the bits.
            (true, false) => {
                let Some(lane) = self.simd_view_bits(dst) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let Some(value) = self.read_register(src) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let sized = Expr::extract(value, lane - 1, 0);
                if !self.write_simd_dst(dst, sized, true) {
                    self.push_aarch64_fp_unsupported(insn);
                }
            }
            // `fmov w0, s0` — the vector lane supplies the bits.
            (false, true) => {
                let Some(lane) = self.simd_view_bits(src) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                if !self.write_register_to(dst, bits) {
                    self.push_aarch64_fp_unsupported(insn);
                }
            }
            (false, false) => self.push_aarch64_fp_unsupported(insn),
        }
    }

    /// `fcmp`/`fcmpe Rn, Rm` — the floating-point compare, written into
    /// the same x86-polarity flags the rest of the `AArch64` lifter
    /// uses so the existing condition-code table keeps working.
    ///
    /// The mapping is chosen so that every `AArch64` condition after an
    /// `fcmp` comes out with its architectural meaning:
    ///
    /// - `ZF` is ordered equality, so `b.eq` is equality and unordered
    ///   does not satisfy it.
    /// - `OF` is the unordered predicate, matching `V`, so `b.vs` after
    ///   a compare means "unordered".
    /// - `SF` is ordered less-than, matching `N`, so `b.mi` means "less
    ///   than".
    /// - `b.lt` lowers to `SF != OF`, which is therefore true for
    ///   less-than *and* for unordered — exactly what `LT` means after
    ///   an `AArch64` floating-point compare, and the reason the sign
    ///   and overflow flags cannot simply be zeroed here.
    ///
    /// `fcmp` and `fcmpe` differ only in which NaN raises the invalid-
    /// operation exception, which the value model does not track.
    fn lift_aarch64_fcmp(&mut self, insn: &Instruction) {
        let (Some(lhs), Some(rhs)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(lhs) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(a) = self.read_simd_lane_fp(lhs, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // `fcmp Rn, #0.0` is the ISA's only immediate form.
        let b = if self.is_simd_register(rhs) {
            let Some(value) = self.read_simd_lane_fp(rhs, lane, 0) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            value
        } else if is_fp_zero_immediate(rhs) {
            let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            Expr::bv_to_fp(Expr::konst(0, lane), ebits, sbits)
        } else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let unordered = Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone()));
        self.set_flag("ZF", Expr::feq(a.clone(), b.clone()));
        self.set_flag("SF", Expr::flt(a.clone(), b.clone()));
        self.set_flag("CF", Expr::flt(a, b));
        self.set_flag("OF", unordered.clone());
        // `AArch64` has no parity flag; carrying the unordered
        // predicate keeps it informative and costs nothing, since no
        // `AArch64` condition reads it.
        self.set_flag("PF", unordered);
    }

    /// `fcvt Rd, Rn` — convert between floating-point precisions. The
    /// source and destination letters give both sorts.
    fn lift_aarch64_fcvt(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let (Some(dst_lane), Some(src_lane)) = (self.simd_view_bits(dst), self.simd_view_bits(src))
        else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, src_lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some((ebits, sbits)) = fp_sort_bits_checked(dst_lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let converted = Expr::fp_to_fp(value, RoundingMode::NearestTiesEven, ebits, sbits);
        if !self.write_simd_dst(dst, Expr::fp_to_ieee_bv(converted), true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `scvtf`/`ucvtf Rd, Rn` — integer register to floating point.
    fn lift_aarch64_int_to_fp(&mut self, insn: &Instruction, signed: bool) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // The source may be a general register (`scvtf s0, w0`) or a
        // vector one (`scvtf s0, s1`, the integer held *in* the vector
        // file). They are two register files, not two widths: reading
        // the vector one through `read_register` would size `v1` at the
        // pointer width, which is the mirror image of the write-side
        // defect `lift_aarch64_fp_to_int` branches around.
        let source_bits = self.operand_width(src);
        let int = if self.is_simd_register(src) {
            self.read_simd_lane_bits(src, source_bits, 0)
        } else {
            self.read_register(src)
        };
        let Some(int) = int else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // The IR carries only the signed conversion. An unsigned value
        // zero-extended by one bit is the same number read as signed,
        // so `ucvtf` goes through the signed node exactly rather than
        // approximately.
        let source = if signed {
            int
        } else {
            let Some(wider) = source_bits.checked_add(1) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            Expr::zero_ext(int, wider)
        };
        let converted = Expr::sbv_to_fp(source, RoundingMode::NearestTiesEven, ebits, sbits);
        if !self.write_simd_dst(dst, Expr::fp_to_ieee_bv(converted), true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fcvtzs`/`fcvtzu Rd, Rn` — floating point to integer, into
    /// either a general register or a SIMD one. The `z` in the mnemonic
    /// is round-toward-zero, so the rounding mode is carried by the
    /// opcode and no control register is assumed.
    ///
    /// The destination's *class* decides how the result is written, and
    /// this used to ignore it — the doc said "integer register", which
    /// is the form it was written for, and the SIMD-destination form
    /// arrived at the same code without a branch. Two things then go
    /// wrong at once, and both are wrong values rather than declines.
    ///
    /// `write_register_to` sizes the parent at [`LiftCtx::bits`], 64 on
    /// this ISA, where the vector register is 128 — and the SSA pass
    /// versions by name alone, so a later vector read binds to that
    /// 64-bit definition. And its partial-write branch *preserves* the
    /// bits above the element, where an `AArch64` scalar SIMD write
    /// clears them. `write_simd_dst` answers both, which is why every
    /// other scalar-FP handler here already uses it and why
    /// `lift_aarch64_fmov` branches the same way.
    fn lift_aarch64_fp_to_int(&mut self, insn: &Instruction, signed: bool) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(src) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let width = self.operand_width(dst);
        // Same trick in reverse: converting into one extra bit of
        // signed range covers the whole unsigned range exactly, and the
        // destination keeps the low `width` bits.
        let converted = if signed {
            Expr::fp_to_sbv(value, RoundingMode::TowardZero, width)
        } else {
            let Some(wider) = width.checked_add(1) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            Expr::extract(
                Expr::fp_to_sbv(value, RoundingMode::TowardZero, wider),
                width - 1,
                0,
            )
        };
        let written = if self.is_simd_register(dst) {
            self.write_simd_dst(dst, converted, true)
        } else {
            self.write_register_to(dst, converted)
        };
        if !written {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    fn push_aarch64_fp_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable operand at {addr}", addr = insn.address),
        });
    }
}

/// `FPRecpX` over the stored bit pattern of a `lane`-bit float.
///
/// The architecture's three cases, in the order it spells them. A NaN is
/// quietened and returned. A zero or subnormal — every encoding whose
/// exponent field is zero — yields `Ones(E) - 1`, which is one below the
/// infinity / NaN field and so the largest *finite* exponent; the
/// complement would give the all-ones field and turn a zero into an
/// infinity. Everything else takes the complemented field, which sends an
/// infinity to a zero of the same sign because the significand is cleared
/// either way.
fn reciprocal_exponent(bits: Expr, lane: u16) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(lane)?;
    let frac_bits = sbits.checked_sub(1)?;
    let all_ones = 1u128.checked_shl(u32::from(ebits))?.checked_sub(1)?;
    let sign = Expr::extract(bits.clone(), lane - 1, lane - 1);
    let exponent = Expr::extract(bits.clone(), lane - 2, frac_bits);
    let fraction = Expr::extract(bits.clone(), frac_bits - 1, 0);
    let reciprocal = Expr::Ite {
        cond: Box::new(Expr::eq(exponent.clone(), Expr::konst(0, ebits))),
        then_expr: Box::new(Expr::konst(all_ones.checked_sub(1)?, ebits)),
        else_expr: Box::new(Expr::bv_xor(exponent.clone(), Expr::konst(all_ones, ebits))),
    };
    let value = Expr::concat(Expr::concat(sign, reciprocal), Expr::konst(0, frac_bits));
    let is_nan = Expr::bool_and(
        Expr::eq(exponent, Expr::konst(all_ones, ebits)),
        Expr::ne(fraction, Expr::konst(0, frac_bits)),
    );
    let quiet = Expr::bv_or(
        bits,
        Expr::konst(1u128.checked_shl(u32::from(frac_bits - 1))?, lane),
    );
    Some(Expr::Ite {
        cond: Box::new(is_nan),
        then_expr: Box::new(quiet),
        else_expr: Box::new(value),
    })
}

/// A bit-vector of `bits` with only the sign bit set.
pub(super) fn sign_bit_mask(bits: u16) -> Option<Expr> {
    let shift = u32::from(bits.checked_sub(1)?);
    let value = 1u128.checked_shl(shift)?;
    Some(Expr::konst(value, bits))
}

/// Whether `op` is the `#0.0` immediate that `fcmp`'s compare-with-zero
/// form takes.
fn is_fp_zero_immediate(op: &Operand) -> bool {
    let raw = op.raw.trim().trim_start_matches('#');
    matches!(raw, "0" | "0.0" | "#0.0" | "0.00000")
}
