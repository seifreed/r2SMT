//! x86 / `x86_64` per-mnemonic lifter handlers, extracted from
//! `lift.rs`. Methods on [`LiftCtx`]; shared infrastructure stays in
//! the parent module.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::register_layout;

use super::{BitwiseOp, ExtendKind, LiftCtx, ShiftOp, nonzero_width};

impl LiftCtx {
    pub(super) fn lift_instruction_x86(&mut self, insn: &Instruction) {
        let mnem = insn.mnemonic.trim().to_ascii_lowercase();
        match mnem.as_str() {
            "mov" => self.lift_mov(insn),
            "movzx" => self.lift_mov_extending(insn, ExtendKind::Zero),
            "movsx" | "movsxd" => self.lift_mov_extending(insn, ExtendKind::Sign),
            "lea" => self.lift_lea(insn),
            "xor" => self.lift_xor(insn),
            "and" => self.lift_bitwise(insn, BitwiseOp::And),
            "or" => self.lift_bitwise(insn, BitwiseOp::Or),
            "add" => self.lift_add_sub(insn, true),
            "sub" => self.lift_add_sub(insn, false),
            "imul" => self.lift_imul(insn),
            "cmp" => self.lift_cmp(insn),
            "test" => self.lift_test(insn),
            "shl" | "sal" => self.lift_shift(insn, ShiftOp::Shl),
            "shr" => self.lift_shift(insn, ShiftOp::Shr),
            "sar" => self.lift_shift(insn, ShiftOp::Sar),
            _ => self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("at {addr}", addr = insn.address),
            }),
        }
    }

    fn lift_mov(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if !matches!(dst.kind, OperandKind::Register | OperandKind::Memory) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-writable destination".into(),
            });
            return;
        }
        let dst_width = self.operand_width(dst);
        let value = self.read_operand_at(src, dst_width);
        if !self.write_dst(dst, value, dst_width) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "memory destination".into(),
            });
        }
    }

    fn lift_mov_extending(&mut self, insn: &Instruction, kind: ExtendKind) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination".into(),
            });
            return;
        };
        let src_width = self.operand_width(src);
        let raw = self.read_operand_at(src, src_width);
        let extended = if src_width >= dst_width {
            raw
        } else {
            match kind {
                ExtendKind::Zero => Expr::zero_ext(raw, dst_width),
                ExtendKind::Sign => Expr::sign_ext(raw, dst_width),
            }
        };
        if !self.write_dst(dst, extended, dst_width) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "unsupported destination".into(),
            });
        }
    }

    fn lift_lea(&mut self, insn: &Instruction) {
        let (Some(dst), Some(_src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(dst_var) = self.dst_var(dst) else {
            return;
        };
        // Modelling the exact memory expression is messy and rarely
        // needed for opaque-predicate detection — we treat the result
        // as an opaque symbolic value.
        self.assign(
            dst_var,
            Expr::Unknown(format!("lea {raw}", raw = insn.operands[1].raw)),
        );
    }

    fn lift_xor(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let dst_raw = dst.raw.trim().to_ascii_lowercase();
        let src_raw = src.raw.trim().to_ascii_lowercase();
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination".into(),
            });
            return;
        };
        if dst_raw == src_raw && register_layout(&dst_raw, self.arch).is_some() {
            // True zero idiom: `xor eax, eax`. The textual match
            // guarantees both operands address the same physical
            // sub-register, so the result is 0.
            if self.write_register_to(dst, Expr::konst(0, dst_width)) {
                self.set_flag("ZF", Expr::konst(1, 1));
                self.set_flag("CF", Expr::konst(0, 1));
                self.set_flag("SF", Expr::konst(0, 1));
                self.set_flag("OF", Expr::konst(0, 1));
                self.set_flag("PF", Expr::konst(1, 1));
                return;
            }
        }
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        let lhs = self.read_operand_at(dst, dst_width);
        let rhs = self.read_operand_at(src, dst_width);
        // Stash the computed result in a temporary before writing the
        // destination so flag updates that follow reference the value
        // the instruction actually produced — without the temp, SSA
        // would rename their `rax` reads to the *post-op* version.
        let tmp = self.new_temp(insn.address, dst_width);
        self.assign(tmp.clone(), Expr::bv_xor(lhs, rhs));
        let tmp_expr = Expr::Var(tmp);
        if !self.write_register_to(dst, tmp_expr.clone()) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        self.update_logic_flags(&tmp_expr, dst_width);
    }

    fn lift_bitwise(&mut self, insn: &Instruction, kind: BitwiseOp) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        let lhs = self.read_operand_at(dst, dst_width);
        let rhs = self.read_operand_at(src, dst_width);
        let result_expr = match kind {
            BitwiseOp::And => Expr::bv_and(lhs, rhs),
            BitwiseOp::Or => Expr::bv_or(lhs, rhs),
        };
        let tmp = self.new_temp(insn.address, dst_width);
        self.assign(tmp.clone(), result_expr);
        let tmp_expr = Expr::Var(tmp);
        if !self.write_register_to(dst, tmp_expr.clone()) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        self.update_logic_flags(&tmp_expr, dst_width);
    }

    fn lift_add_sub(&mut self, insn: &Instruction, is_add: bool) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        let lhs_before = self.read_operand_at(dst, dst_width);
        let rhs = self.read_operand_at(src, dst_width);
        // Stash the computed delta in a temporary before the destination
        // write so the flag updates that follow reference the value the
        // instruction actually produced. Without the temp, SSA would
        // rename the operand reads inside the flag expressions to the
        // *post-op* register version and the flags would compute against
        // the just-written destination instead of the operation result.
        let tmp = self.new_temp(insn.address, dst_width);
        let computed = if is_add {
            Expr::add(lhs_before, rhs)
        } else {
            Expr::sub(lhs_before, rhs)
        };
        self.assign(tmp.clone(), computed);
        let tmp_expr = Expr::Var(tmp);
        if !self.write_register_to(dst, tmp_expr.clone()) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, dst_width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, dst_width)));
        // CF for `add` is `result <u lhs_before` (carry out); for `sub`
        // it is `lhs_before <u rhs` (borrow). The slicer cannot witness
        // either precisely without a 1-bit extension, so we leave the
        // bit unmodelled for now.
        self.set_flag("CF", Expr::Unknown(String::new()));
        self.set_flag("OF", Expr::Unknown(String::new()));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    fn lift_imul(&mut self, insn: &Instruction) {
        // Only the two-operand and three-operand forms appear in slices
        // we care about; the one-operand form writes rdx:rax which we
        // do not model. Mark unsupported for that case.
        match insn.operands.len() {
            2 => {
                let dst = &insn.operands[0];
                let src = &insn.operands[1];
                let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
                    self.stmts.push(IrStmt::Unsupported {
                        mnemonic: insn.mnemonic.clone(),
                        comment: "zero-width destination".into(),
                    });
                    return;
                };
                if dst.kind != OperandKind::Register {
                    self.stmts.push(IrStmt::Unsupported {
                        mnemonic: insn.mnemonic.clone(),
                        comment: "non-register destination".into(),
                    });
                    return;
                }
                let lhs = self.read_operand_at(dst, dst_width);
                let rhs = self.read_operand_at(src, dst_width);
                let result = Expr::mul(lhs, rhs);
                if !self.write_register_to(dst, result) {
                    self.stmts.push(IrStmt::Unsupported {
                        mnemonic: insn.mnemonic.clone(),
                        comment: "non-register destination".into(),
                    });
                    return;
                }
                self.clear_all_flags();
            }
            3 => {
                let dst = &insn.operands[0];
                let src1 = &insn.operands[1];
                let src2 = &insn.operands[2];
                let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
                    self.stmts.push(IrStmt::Unsupported {
                        mnemonic: insn.mnemonic.clone(),
                        comment: "zero-width destination".into(),
                    });
                    return;
                };
                if dst.kind != OperandKind::Register {
                    self.stmts.push(IrStmt::Unsupported {
                        mnemonic: insn.mnemonic.clone(),
                        comment: "non-register destination".into(),
                    });
                    return;
                }
                let lhs = self.read_operand_at(src1, dst_width);
                let rhs = self.read_operand_at(src2, dst_width);
                let result = Expr::mul(lhs, rhs);
                if !self.write_register_to(dst, result) {
                    self.stmts.push(IrStmt::Unsupported {
                        mnemonic: insn.mnemonic.clone(),
                        comment: "non-register destination".into(),
                    });
                    return;
                }
                self.clear_all_flags();
            }
            _ => self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "1-operand imul writes rdx:rax".into(),
            }),
        }
    }

    fn lift_cmp(&mut self, insn: &Instruction) {
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let cmp_width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, cmp_width);
        let rhs = self.read_operand_at(rhs_op, cmp_width);
        let tmp = self.new_temp(insn.address, cmp_width);
        self.assign(tmp.clone(), Expr::sub(lhs.clone(), rhs.clone()));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, cmp_width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, cmp_width)));
        // For `cmp lhs, rhs`, CF is the unsigned borrow `lhs <u rhs`.
        self.set_flag("CF", Expr::ult(lhs, rhs));
        self.set_flag("OF", Expr::Unknown(String::new()));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    fn lift_test(&mut self, insn: &Instruction) {
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let cmp_width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, cmp_width);
        let rhs = self.read_operand_at(rhs_op, cmp_width);
        let tmp = self.new_temp(insn.address, cmp_width);
        self.assign(tmp.clone(), Expr::bv_and(lhs, rhs));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, cmp_width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, cmp_width)));
        // `test` always clears CF and OF, leaves AF undefined, and
        // sets PF/SF/ZF from the result.
        self.set_flag("CF", Expr::konst(0, 1));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    fn lift_shift(&mut self, insn: &Instruction, op: ShiftOp) {
        let (Some(dst), Some(count)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        let lhs = self.read_operand_at(dst, dst_width);
        let shift = self.read_operand_at(count, dst_width);
        let computed = match op {
            ShiftOp::Shl => Expr::shl(lhs, shift),
            ShiftOp::Shr => Expr::lshr(lhs, shift),
            ShiftOp::Sar => Expr::ashr(lhs, shift),
        };
        // Temp the result so flag reads survive the destination write
        // under SSA rename — see `lift_add_sub` for the full rationale.
        let tmp = self.new_temp(insn.address, dst_width);
        self.assign(tmp.clone(), computed);
        let tmp_expr = Expr::Var(tmp);
        if !self.write_register_to(dst, tmp_expr.clone()) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination".into(),
            });
            return;
        }
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, dst_width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, dst_width)));
        // CF/OF for shifts depend on shift count and direction; not yet
        // modelled.
        self.set_flag("CF", Expr::Unknown(String::new()));
        self.set_flag("OF", Expr::Unknown(String::new()));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }
}
