//! `AArch64` per-mnemonic lifter handlers, extracted from `lift.rs`.
//! Methods on [`LiftCtx`]; shared infrastructure stays in the parent.

use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::register_layout;

use super::{
    BinOp, CsArithOp, LiftCtx, aarch64_cond_suffix_to_predicate, nonzero_width, width_mask,
};

impl LiftCtx {
    pub(super) fn lift_instruction_aarch64(&mut self, insn: &Instruction) {
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
            // Compare / test: 2-operand, no destination.
            "cmp" => self.lift_aarch64_cmp(insn),
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
            // P26 — memory loads / stores in offset form `[Xn]` /
            // `[Xn, #imm]`. Pre / post-index writeback (`[Xn, #imm]!`
            // / `[Xn], #imm`) and register-offset addressing
            // (`[Xn, Xm]`) decline to `Unsupported` so the slice's
            // confidence path picks them up — soundness in lowering,
            // not detection.
            "ldr" => self.lift_aarch64_ldr(insn),
            "str" => self.lift_aarch64_str(insn),
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

    pub(super) fn lift_aarch64_arith3(&mut self, insn: &Instruction, op: BinOp, sets_flags: bool) {
        let (Some(dst), Some(src1), Some(src2)) = (
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
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (aarch64)".into(),
            });
            return;
        };
        let lhs = self.read_operand_at(src1, dst_width);
        let rhs = self.read_operand_at(src2, dst_width);
        // Stash the computed result in a temp and emit the flag updates
        // *before* writing the destination. AArch64 `adds Rd, Rn, Rm`
        // is normally 3-operand with `Rd` distinct from `Rn`/`Rm`, but
        // the architecture allows `adds x0, x0, x1`. Without the
        // pre-write flag emission the `x0` reads inside CF (and any
        // other lhs/rhs-derived flag) would be renamed by SSA to the
        // post-write version, breaking the flag value. See
        // `lift_add_sub` for the x86 analogue and the recorded
        // regression in `r2smt_lifter_sub_flag_bug.md`.
        let tmp = self.new_temp(insn.address, dst_width);
        self.assign(tmp.clone(), op.apply(lhs.clone(), rhs.clone()));
        let tmp_expr = Expr::Var(tmp);
        if sets_flags {
            self.aarch64_set_arith_flags(op, &lhs, &rhs, &tmp_expr, dst_width);
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
    fn aarch64_set_arith_flags(
        &mut self,
        op: BinOp,
        lhs: &Expr,
        rhs: &Expr,
        result: &Expr,
        width: u8,
    ) {
        self.set_flag("ZF", Expr::eq(result.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(result.clone(), Expr::konst(0, width)));
        let cf = match op {
            BinOp::Sub => Expr::ult(lhs.clone(), rhs.clone()),
            // Logical ops clear C/V on AArch64. `adds` / `mul` etc.
            // need a full extension to compute carry precisely; mark
            // Unknown rather than fabricate a value.
            BinOp::And | BinOp::Or | BinOp::Xor => Expr::konst(0, 1),
            _ => Expr::Unknown(String::new()),
        };
        self.set_flag("CF", cf);
        // OF clears for logical ops, Unknown otherwise (until we add
        // signed-overflow modelling).
        let of = match op {
            BinOp::And | BinOp::Or | BinOp::Xor => Expr::konst(0, 1),
            _ => Expr::Unknown(String::new()),
        };
        self.set_flag("OF", of);
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
        let rhs = self.read_operand_at(rhs_op, width);
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
            CsArithOp::Inv => Expr::bv_xor(rm_value, Expr::konst(width_mask(dst_width), dst_width)),
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
            Expr::konst(width_mask(dst_width), dst_width)
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

    pub(super) fn lift_aarch64_tst(&mut self, insn: &Instruction) {
        // AArch64 `tst Rn, Operand` = `ands xzr, Rn, Operand` — sets
        // flags from Rn AND Operand, no register destination.
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, width);
        let rhs = self.read_operand_at(rhs_op, width);
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), Expr::bv_and(lhs, rhs));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, width)));
        self.set_flag("CF", Expr::konst(0, 1));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    /// `ldr Rd, [Xn{, #imm}]` — read `Rd`-width bytes from memory and
    /// write them to the destination register. W-form (`ldr Wd, …`)
    /// zero-extends to the parent X per the `AArch64` ABI via
    /// [`LiftCtx::write_register_to`]. Writeback (`[Xn, …]!`) and
    /// register-offset addressing decline to `Unsupported` so the
    /// confidence path picks them up rather than silently widening.
    pub(super) fn lift_aarch64_ldr(&mut self, insn: &Instruction) {
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
        let Some(load_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldr zero-width destination".into(),
            });
            return;
        };
        let Some(address) = aarch64_address_expr(mem, self.bits) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("ldr addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        // Two-statement lower: load into a fresh temp at the load
        // width, then write that temp into the destination register
        // so `write_register_to` zero-extends to the parent X for
        // the W-form (mirrors the `add` / `sub` flag-ordering
        // precedent: stash-then-write).
        let tmp = self.new_temp(insn.address, load_width);
        self.stmts.push(IrStmt::LoadMem {
            dst: tmp.clone(),
            address,
            bits: load_width,
        });
        if !self.write_register_to(dst, Expr::Var(tmp)) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldr destination not a supported register".into(),
            });
        }
    }

    /// `str Rs, [Xn{, #imm}]` — write the source register's natural
    /// width to memory. See [`Self::lift_aarch64_ldr`] for the
    /// addressing-mode restrictions.
    pub(super) fn lift_aarch64_str(&mut self, insn: &Instruction) {
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
        let Some(store_width) = nonzero_width(self.operand_width(src)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "str zero-width source".into(),
            });
            return;
        };
        let Some(address) = aarch64_address_expr(mem, self.bits) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("str addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        let value = self.read_operand_at(src, store_width);
        self.stmts.push(IrStmt::StoreMem {
            address,
            value,
            bits: store_width,
        });
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
fn aarch64_address_expr(mem: &Operand, ptr_bits: u8) -> Option<Expr> {
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
    let off_const = Expr::konst(masked, ptr_bits);
    Some(Expr::add(base_var, off_const))
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
    register_layout(&lower, r2smt_common::Arch::Aarch64)
        .map(|l| l.parent != "xzr")
        .filter(|valid| *valid)
        .is_some()
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
