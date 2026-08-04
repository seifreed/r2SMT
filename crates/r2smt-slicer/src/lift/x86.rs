//! x86 / `x86_64` per-mnemonic lifter handlers, extracted from
//! `lift.rs`. Methods on [`LiftCtx`]; shared infrastructure stays in
//! the parent module.

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::{is_simd_parent, register_layout};

use super::simd::{CompareKind, compare_lane};
use super::{BinOp, PackedIntOp};
use super::{
    BitwiseOp, ExtendKind, FpArithOp, LiftCtx, ShiftOp, fp_lane_result, fp_sort_bits_checked,
    nonzero_width,
};

/// x86 SHL/SHR/SAR/SAL mask the shift count before shifting: 5 bits
/// for 8/16/32-bit operands, 6 bits for 64-bit operands (Intel SDM
/// Vol. 2, "SAL/SAR/SHL/SHR"). Without this, an oversized count
/// (e.g. `shl eax, 32`) makes Z3's `bvshl` zero the result while the
/// CPU treats it as a no-op / small shift — flipping ZF/SF-derived
/// verdicts.
const X86_SHIFT_COUNT_MASK_NARROW: u64 = 0x1F;
const X86_SHIFT_COUNT_MASK_64: u64 = 0x3F;
const X86_WIDTH_64: u16 = 64;

/// Which bit of AH each flag `sahf` transfers comes from (Intel SDM
/// Vol. 2, "SAHF"). AF is omitted because the flag model does not carry
/// it, and OF because `sahf` does not write it.
const SAHF_FLAG_BITS: [(&str, u16); 4] = [("CF", 0), ("PF", 2), ("ZF", 6), ("SF", 7)];

/// Width of the x87 double-extended memory image. A float store to
/// anything narrower rounds; a store of this width does not.
const X87_EXTENDED_MEMORY_BITS: u16 = 80;

impl LiftCtx {
    pub(super) fn lift_instruction_x86(&mut self, insn: &Instruction) {
        let mnem = insn.mnemonic.trim().to_ascii_lowercase();
        if self.lift_x86_packed_int_by_mnemonic(insn, mnem.as_str()) {
            return;
        }
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
            "movaps" | "movups" | "movapd" | "movupd" | "movdqa" | "movdqu" | "vmovaps"
            | "vmovups" | "vmovapd" | "vmovupd" | "vmovdqa" | "vmovdqu" => {
                self.lift_simd_move(insn);
            }
            "pxor" | "vpxor" => self.lift_simd_bitwise(insn, SimdBitOp::Xor),
            "pand" | "vpand" => self.lift_simd_bitwise(insn, SimdBitOp::And),
            "por" | "vpor" => self.lift_simd_bitwise(insn, SimdBitOp::Or),
            "pandn" | "vpandn" => self.lift_simd_bitwise(insn, SimdBitOp::AndNot),
            "addss" => self.lift_simd_scalar_fp(insn, FpArithOp::Add, 32),
            "subss" => self.lift_simd_scalar_fp(insn, FpArithOp::Sub, 32),
            "mulss" => self.lift_simd_scalar_fp(insn, FpArithOp::Mul, 32),
            "divss" => self.lift_simd_scalar_fp(insn, FpArithOp::Div, 32),
            "addsd" => self.lift_simd_scalar_fp(insn, FpArithOp::Add, 64),
            "subsd" => self.lift_simd_scalar_fp(insn, FpArithOp::Sub, 64),
            "mulsd" => self.lift_simd_scalar_fp(insn, FpArithOp::Mul, 64),
            "divsd" => self.lift_simd_scalar_fp(insn, FpArithOp::Div, 64),
            // Packed integer compares. `pcmpgt*` is signed; x86 has no
            // unsigned form.
            "pmovmskb" | "vpmovmskb" => self.lift_simd_move_mask(insn),
            "movd" | "vmovd" => self.lift_simd_gpr_transfer(insn, 32),
            "movq" | "vmovq" => self.lift_simd_gpr_transfer(insn, 64),
            "pcmpeqb" | "vpcmpeqb" => self.lift_simd_packed_compare(insn, EQUAL, 8),
            "pcmpeqw" | "vpcmpeqw" => self.lift_simd_packed_compare(insn, EQUAL, 16),
            "pcmpeqd" | "vpcmpeqd" => self.lift_simd_packed_compare(insn, EQUAL, 32),
            "pcmpeqq" | "vpcmpeqq" => self.lift_simd_packed_compare(insn, EQUAL, 64),
            "pcmpgtb" | "vpcmpgtb" => self.lift_simd_packed_compare(insn, GREATER, 8),
            "pcmpgtw" | "vpcmpgtw" => self.lift_simd_packed_compare(insn, GREATER, 16),
            "pcmpgtd" | "vpcmpgtd" => self.lift_simd_packed_compare(insn, GREATER, 32),
            "pcmpgtq" | "vpcmpgtq" => self.lift_simd_packed_compare(insn, GREATER, 64),
            "addps" | "vaddps" => self.lift_simd_packed_fp(insn, FpArithOp::Add, 32),
            "subps" | "vsubps" => self.lift_simd_packed_fp(insn, FpArithOp::Sub, 32),
            "mulps" | "vmulps" => self.lift_simd_packed_fp(insn, FpArithOp::Mul, 32),
            "divps" | "vdivps" => self.lift_simd_packed_fp(insn, FpArithOp::Div, 32),
            "addpd" | "vaddpd" => self.lift_simd_packed_fp(insn, FpArithOp::Add, 64),
            "subpd" | "vsubpd" => self.lift_simd_packed_fp(insn, FpArithOp::Sub, 64),
            "mulpd" | "vmulpd" => self.lift_simd_packed_fp(insn, FpArithOp::Mul, 64),
            "divpd" | "vdivpd" => self.lift_simd_packed_fp(insn, FpArithOp::Div, 64),
            "maxps" | "vmaxps" => self.lift_simd_packed_fp(insn, FpArithOp::Max, 32),
            "minps" | "vminps" => self.lift_simd_packed_fp(insn, FpArithOp::Min, 32),
            "maxpd" | "vmaxpd" => self.lift_simd_packed_fp(insn, FpArithOp::Max, 64),
            "minpd" | "vminpd" => self.lift_simd_packed_fp(insn, FpArithOp::Min, 64),
            "maxss" => self.lift_simd_scalar_fp(insn, FpArithOp::Max, 32),
            "minss" => self.lift_simd_scalar_fp(insn, FpArithOp::Min, 32),
            "maxsd" => self.lift_simd_scalar_fp(insn, FpArithOp::Max, 64),
            "minsd" => self.lift_simd_scalar_fp(insn, FpArithOp::Min, 64),
            "sqrtps" | "vsqrtps" => self.lift_simd_sqrt(insn, 32, true),
            "sqrtpd" | "vsqrtpd" => self.lift_simd_sqrt(insn, 64, true),
            "sqrtss" => self.lift_simd_sqrt(insn, 32, false),
            "sqrtsd" => self.lift_simd_sqrt(insn, 64, false),
            m if parse_fp_compare(m).is_some() => {
                if let Some(cmp) = parse_fp_compare(m) {
                    self.lift_simd_fp_mask_compare(insn, &cmp);
                }
            }
            _ if sse_scalar_move_lane(insn).is_some() => {
                if let Some(lane) = sse_scalar_move_lane(insn) {
                    self.lift_sse_scalar_move(insn, lane);
                }
            }
            "vcvtph2ps" => self.lift_f16c_widen(insn),
            "vcvtps2ph" => self.lift_f16c_narrow(insn),
            "comiss" | "ucomiss" => self.lift_simd_fp_compare(insn, 32),
            "comisd" | "ucomisd" => self.lift_simd_fp_compare(insn, 64),
            "cvtsi2ss" => self.lift_int_to_fp(insn, 32),
            "cvtsi2sd" => self.lift_int_to_fp(insn, 64),
            "cvtss2si" => self.lift_fp_to_int(insn, 32, RoundingMode::NearestTiesEven),
            "cvtsd2si" => self.lift_fp_to_int(insn, 64, RoundingMode::NearestTiesEven),
            "cvttss2si" => self.lift_fp_to_int(insn, 32, RoundingMode::TowardZero),
            "cvttsd2si" => self.lift_fp_to_int(insn, 64, RoundingMode::TowardZero),
            "cvtss2sd" => self.lift_fp_to_fp(insn, 32, 64),
            "cvtsd2ss" => self.lift_fp_to_fp(insn, 64, 32),
            "sahf" => self.lift_sahf(),
            // x87 keeps its own slice-scoped stack rather than a
            // register model, so it is recognised by shape rather than
            // by mnemonic alone — see `lift/x87.rs`.
            _ if crate::lift::is_modelled_x87(insn) => self.lift_instruction_x87(insn),
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
        let value = self.read_operand_lowered(src, dst_width);
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
        let raw = self.read_operand_lowered(src, src_width);
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
        let lhs = self.read_operand_lowered(dst, dst_width);
        let rhs = self.read_operand_lowered(src, dst_width);
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
        let lhs = self.read_operand_lowered(dst, dst_width);
        let rhs = self.read_operand_lowered(src, dst_width);
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
        let lhs_before = self.read_operand_lowered(dst, dst_width);
        let rhs = self.read_operand_lowered(src, dst_width);
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
                let lhs = self.read_operand_lowered(dst, dst_width);
                let rhs = self.read_operand_lowered(src, dst_width);
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
                let lhs = self.read_operand_lowered(src1, dst_width);
                let rhs = self.read_operand_lowered(src2, dst_width);
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
        let lhs = self.read_operand_lowered(lhs_op, cmp_width);
        let rhs = self.read_operand_lowered(rhs_op, cmp_width);
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
        let lhs = self.read_operand_lowered(lhs_op, cmp_width);
        let rhs = self.read_operand_lowered(rhs_op, cmp_width);
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
        let lhs = self.read_operand_lowered(dst, dst_width);
        let raw_shift = self.read_operand_lowered(count, dst_width);
        let count_mask = if dst_width == X86_WIDTH_64 {
            X86_SHIFT_COUNT_MASK_64
        } else {
            X86_SHIFT_COUNT_MASK_NARROW
        };
        let shift = Expr::bv_and(raw_shift, Expr::konst(u128::from(count_mask), dst_width));
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

    /// `movaps`/`movups`/`movdqa`/`movdqu dst, src` (and their VEX
    /// `v*` forms) — a full-view copy of `src` into `dst`. The view
    /// width (128 / 256 / 512) comes from the operands; a legacy move
    /// preserves the parent bits above the view, a VEX move zeroes them.
    fn lift_simd_move(&mut self, insn: &Instruction) {
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
    fn lift_sse_scalar_move(&mut self, insn: &Instruction, lane: u16) {
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
    fn lift_simd_bitwise(&mut self, insn: &Instruction, op: SimdBitOp) {
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
            SimdBitOp::AndNot => Expr::bv_and(Expr::bv_xor(a, super::simd::all_ones(width)), b),
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
    fn lift_x86_packed_int_by_mnemonic(&mut self, insn: &Instruction, mnem: &str) -> bool {
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
    fn lift_simd_move_mask(&mut self, insn: &Instruction) {
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
    fn lift_simd_gpr_transfer(&mut self, insn: &Instruction, width: u16) {
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
    fn lift_simd_packed_compare(&mut self, insn: &Instruction, kind: CompareKind, lane_bits: u16) {
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

    /// Packed floating-point arithmetic: the same lane operation applied
    /// independently to every `lane_bits`-wide lane of the destination's
    /// vector view (`addps` → 4 single lanes over 128 bits, `vaddps ymm`
    /// → 8 over 256).
    ///
    /// Operand roles and the upper-bits rule are shared with
    /// [`Self::lift_simd_bitwise`]: 2-operand is RMW, 3-operand VEX reads
    /// its two explicit sources, and a VEX write zeroes above the view.
    fn lift_simd_packed_fp(&mut self, insn: &Instruction, op: FpArithOp, lane_bits: u16) {
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
    fn lift_f16c_widen(&mut self, insn: &Instruction) {
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
    fn lift_f16c_narrow(&mut self, insn: &Instruction) {
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
    fn lift_simd_fp_mask_compare(&mut self, insn: &Instruction, cmp: &FpCompare) {
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
    fn lift_simd_sqrt(&mut self, insn: &Instruction, lane_bits: u16, packed: bool) {
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
    fn lift_simd_scalar_fp(&mut self, insn: &Instruction, op: FpArithOp, lane: u16) {
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
    fn lift_simd_fp_compare(&mut self, insn: &Instruction, lane: u16) {
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
    fn lift_int_to_fp(&mut self, insn: &Instruction, lane: u16) {
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
    fn lift_fp_to_int(&mut self, insn: &Instruction, lane: u16, rm: RoundingMode) {
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
    fn lift_fp_to_fp(&mut self, insn: &Instruction, src_lane: u16, dst_lane: u16) {
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

    /// `sahf` — load SF, ZF, AF and PF and CF from the bits of AH.
    ///
    /// The bit positions are the point rather than an implementation
    /// detail: they are why `fnstsw ax ; sahf` transfers an x87 compare
    /// into the integer flags at all. `fnstsw` puts status-word bits
    /// 15..8 into AH, which lands C0 at bit 0, C2 at bit 2 and C3 at
    /// bit 6 — the CF, PF and ZF positions this reads.
    ///
    /// OF is deliberately absent: `sahf` does not write it, so whatever
    /// defined it upstream still holds. AF is not modelled at all.
    fn lift_sahf(&mut self) {
        let Some(ah) = self.read_register(&Operand {
            raw: "ah".into(),
            kind: OperandKind::Register,
        }) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: "sahf".into(),
                comment: "no ah in this register model".into(),
            });
            return;
        };
        for (flag, bit) in SAHF_FLAG_BITS {
            self.set_flag(flag, Expr::extract(ah.clone(), bit, bit));
        }
    }

    fn push_simd_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable SIMD operand at {addr}", addr = insn.address),
        });
    }
}

/// Bit-vector operation of an integer-SIMD bitwise instruction.
#[derive(Clone, Copy)]
enum SimdBitOp {
    Xor,
    And,
    Or,
    /// `pandn`/`vpandn`: `(~a) & b`.
    AndNot,
}

/// Whether two operands name the same SIMD register view (same vector
/// parent and same width) — the condition for a `pxor`/`vpxor` zero
/// idiom.
fn same_xmm_register(a: &Operand, b: &Operand) -> bool {
    match (
        register_layout(&a.raw, Arch::X86_64),
        register_layout(&b.raw, Arch::X86_64),
    ) {
        (Some(la), Some(lb)) => {
            is_simd_parent(la.parent, Arch::X86_64)
                && la.parent == lb.parent
                && la.width() == lb.width()
        }
        _ => false,
    }
}

/// Bits in a byte — the lane width `pmovmskb` samples.
const BITS_PER_BYTE: u16 = 8;

/// The lane-wise arithmetic `PackedIntOp`s x86 spells.
const ADD: PackedIntOp = PackedIntOp::Bin(BinOp::Add);
const SUB: PackedIntOp = PackedIntOp::Bin(BinOp::Sub);
const MUL: PackedIntOp = PackedIntOp::Bin(BinOp::Mul);

/// `pcmpeq*` — lane equality.
const EQUAL: CompareKind = CompareKind::Equal { float: false };

/// `pcmpgt*` — signed lane greater-than. x86 spells no unsigned form.
const GREATER: CompareKind = CompareKind::Ordered {
    float: false,
    signed: true,
    or_equal: false,
};

/// How an x86 SIMD instruction's destination relates to its sources —
/// the one fact the effect table needs that the lifter's own dispatch
/// does not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X86SimdShape {
    /// The destination is fully overwritten (`movaps`, `sqrtps`,
    /// `cvtss2si`), so it is a def and not also a use.
    Move,
    /// The 2-operand form is a read-modify-write, so the destination is
    /// a source too; the 3-operand VEX form reads its two explicit
    /// sources instead, which the effect table tells apart by operand
    /// count.
    ReadModifyWrite,
}

/// The x86 SIMD mnemonics this crate models, with each one's operand
/// shape.
///
/// Single source of truth for *membership*. `is_x86_simd_instruction`
/// — which pre-empts the ESIL / P-code ladder — and the effect table
/// both consult this, so a mnemonic cannot be retained by the slicer
/// while the lifter drops it. That asymmetry is the historical `pandn`
/// bug, and the same shape as the ARM def/use gaps that fabricated
/// verdicts: an instruction the effect table keeps but no handler
/// models leaves its destination undefined, and a later read binds to a
/// stale value.
///
/// Two families are recognised structurally instead of listed, by
/// `is_fp_compare_mnemonic` and `sse_scalar_move_lane`: the packed FP
/// compares are 64 mnemonics once eight predicates cross `ps`/`pd`/
/// `ss`/`sd` and VEX, and `movsd` needs its operands inspected because
/// the name is also the string instruction.
pub(crate) fn x86_simd_shape(mnemonic: &str) -> Option<X86SimdShape> {
    let shape = match mnemonic {
        // Whole-destination writes.
        "movaps" | "movups" | "movapd" | "movupd" | "movdqa" | "movdqu" | "vmovaps" | "vmovups"
        | "vmovapd" | "vmovupd" | "vmovdqa" | "vmovdqu" | "cvtss2si" | "cvtsd2si" | "cvttss2si"
        | "cvttsd2si" | "sqrtps" | "sqrtpd" | "vsqrtps" | "vsqrtpd" | "vcvtph2ps" | "vcvtps2ph"
        // `pmovmskb` writes a general register; `movd`/`movq` go either
        // way. All three overwrite the whole destination — the vector
        // direction zeroes above the transferred value rather than
        // merging.
        | "pmovmskb" | "vpmovmskb" | "movd" | "vmovd" | "movq" | "vmovq" => X86SimdShape::Move,
        // 2-operand read-modify-write (or its 3-operand VEX form).
        "pxor" | "vpxor" | "pand" | "vpand" | "por" | "vpor" | "pandn" | "vpandn" | "addss"
        | "subss" | "mulss" | "divss" | "addsd" | "subsd" | "mulsd" | "divsd" | "addps"
        | "subps" | "mulps" | "divps" | "vaddps" | "vsubps" | "vmulps" | "vdivps" | "addpd"
        | "subpd" | "mulpd" | "divpd" | "vaddpd" | "vsubpd" | "vmulpd" | "vdivpd" | "maxps"
        | "minps" | "maxpd" | "minpd" | "vmaxps" | "vminps" | "vmaxpd" | "vminpd" | "maxss"
        | "minss" | "maxsd" | "minsd" | "cvtsi2ss" | "cvtsi2sd" | "cvtss2sd" | "cvtsd2ss"
        | "sqrtss" | "sqrtsd"
        // The packed integer compares overwrite every lane, but the
        // 2-operand form still reads the destination as its first
        // source, so they share the arithmetic's shape.
        | "pcmpeqb" | "pcmpeqw" | "pcmpeqd" | "pcmpeqq" | "pcmpgtb" | "pcmpgtw" | "pcmpgtd"
        | "pcmpgtq" | "vpcmpeqb" | "vpcmpeqw" | "vpcmpeqd" | "vpcmpeqq" | "vpcmpgtb"
        | "vpcmpgtw" | "vpcmpgtd" | "vpcmpgtq"
        // Packed integer arithmetic and the shifts, same shape again.
        | "paddb" | "paddw" | "paddd" | "paddq" | "psubb" | "psubw" | "psubd" | "psubq"
        | "pmullw" | "pmulld" | "pslldq" | "psrldq" | "psllw" | "pslld" | "psllq" | "psrlw"
        | "psrld" | "psrlq" | "psraw" | "psrad" | "vpaddb" | "vpaddw" | "vpaddd" | "vpaddq"
        | "vpsubb" | "vpsubw" | "vpsubd" | "vpsubq" | "vpmullw" | "vpmulld" | "vpslldq"
        | "vpsrldq" | "vpsllw" | "vpslld" | "vpsllq" | "vpsrlw" | "vpsrld" | "vpsrlq"
        | "vpsraw" | "vpsrad" => X86SimdShape::ReadModifyWrite,
        _ => return None,
    };
    Some(shape)
}

/// Whether `insn` is a VEX/EVEX-encoded (`v`-prefixed) SIMD form, whose
/// destination write zeroes the vector-register bits above the view.
/// Legacy SSE forms preserve those bits.
fn is_vex(insn: &Instruction) -> bool {
    insn.mnemonic.trim().to_ascii_lowercase().starts_with('v')
}

/// The lane width of an SSE **scalar move** (`movss` → 32, `movsd` → 64,
/// and their VEX forms), or `None` if `insn` is not one.
///
/// `movsd` names two unrelated instructions: the scalar double move
/// (SDM Vol. 2, "MOVSD—Move or Merge Scalar Double Precision
/// Floating-Point Value") and the string move (opcode `A5`,
/// "MOVS/MOVSB/MOVSW/MOVSD/MOVSQ—Move Data from String to String").
/// They are told apart by operand shape, not by the mnemonic: the SSE
/// form always names an XMM register, while the string form's operands
/// are only the implicit `[rdi]` / `[rsi]` pair. Claiming the string
/// form here would route it away from the ESIL ladder that models it
/// correctly, so the discriminator has to be exact in that direction.
pub(crate) fn sse_scalar_move_lane(insn: &Instruction) -> Option<u16> {
    let lower = insn.mnemonic.trim().to_ascii_lowercase();
    let body = lower.strip_prefix('v').unwrap_or(&lower);
    let lane = match body {
        "movss" => 32,
        "movsd" => 64,
        _ => return None,
    };
    insn.operands.iter().any(is_xmm_register).then_some(lane)
}

/// Whether `op` is a register operand naming a view of an x86 vector
/// register (the synthetic `zmm<n>` parent).
fn is_xmm_register(op: &Operand) -> bool {
    op.kind == OperandKind::Register
        && register_layout(&op.raw, Arch::X86_64)
            .is_some_and(|layout| is_simd_parent(layout.parent, Arch::X86_64))
}

/// Predicate of an SSE/AVX compare, as radare2 spells it: the immediate
/// is baked into the mnemonic (`cmpeqps`, `cmpltsd`, …) rather than
/// appearing as an operand.
#[derive(Clone, Copy)]
enum FpCmpPred {
    Eq,
    Lt,
    Le,
    Unord,
}

/// A parsed compare mnemonic: predicate, lane width, whether it is
/// packed, and whether the result is the negation of the base
/// predicate.
struct FpCompare {
    pred: FpCmpPred,
    lane_bits: u16,
    packed: bool,
    negated: bool,
}

/// Whether `mnemonic` is one of the SSE/AVX compare pseudo-mnemonics.
///
/// Parse `cmp<pred><ps|pd|ss|sd>` and its VEX `v`-prefixed form.
///
/// The four negated predicates (`neq`, `nlt`, `nle`, `ord`) are exactly
/// the boolean negations of the four base ones — which is also why they
/// are the "unordered" variants: negating a comparison that is false on
/// NaN yields one that is true on NaN.
/// Mnemonics whose lifting pins a floating-point control field to its
/// architectural default: the rounding mode in both control words, and
/// on x87 the precision-control field as well.
///
/// Deliberately narrow on the SSE side: `max`/`min` select an operand
/// and the `cvtt` forms carry round-toward-zero in the opcode, so
/// neither depends on MXCSR. Only the operations that actually round
/// are listed.
///
/// The x87 side is broader, because its control word carries a second
/// field the SSE one does not. Precision control sets the significand
/// width of every `FADD` / `FSUB` / `FMUL` / `FDIV` / `FSQRT` result,
/// so those depend on it *whatever* they round to — a fixed sort cannot
/// express a narrowed significand at all. `FLD` and `FILD` are absent
/// because neither rounds: both convert exactly into the extended
/// format, and neither is in the SDM's list of instructions precision
/// control affects.
///
/// `fst` / `fstp` are the one shape that has to consult an operand:
/// they round only when the destination is narrower than the extended
/// sort, so the `m80fp` and register forms are exact and stay out.
pub(crate) fn pins_rounding_mode(insn: &Instruction) -> bool {
    let lower = insn.mnemonic.trim().to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "fadd"
            | "faddp"
            | "fsub"
            | "fsubp"
            | "fsubr"
            | "fsubrp"
            | "fmul"
            | "fmulp"
            | "fdiv"
            | "fdivp"
            | "fdivr"
            | "fdivrp"
            | "fsqrt"
            // The integer-operand arithmetic converts its operand
            // exactly and then rounds the result, exactly as the
            // float-operand forms do.
            | "fiadd"
            | "fisub"
            | "fisubr"
            | "fimul"
            | "fidiv"
            | "fidivr"
            // The integer stores always round: no integer format holds
            // an arbitrary extended value. `fisttp` is deliberately
            // absent — it carries round-toward-zero in the opcode, so
            // nothing it computes depends on the control word.
            | "fist"
            | "fistp"
    ) {
        return true;
    }
    // A float store rounds only when the destination is narrower than
    // the extended sort — `fstp st(i)` copies and `fstp tbyte` writes
    // the value it already holds.
    if matches!(lower.as_str(), "fst" | "fstp") {
        return insn
            .operands
            .first()
            .and_then(|op| crate::effect::memory_operand_width(&op.raw))
            .is_some_and(|width| width < X87_EXTENDED_MEMORY_BITS);
    }
    let body = lower.strip_prefix('v').unwrap_or(&lower);
    matches!(
        body,
        "addss"
            | "subss"
            | "mulss"
            | "divss"
            | "addsd"
            | "subsd"
            | "mulsd"
            | "divsd"
            | "addps"
            | "subps"
            | "mulps"
            | "divps"
            | "addpd"
            | "subpd"
            | "mulpd"
            | "divpd"
            | "sqrtss"
            | "sqrtsd"
            | "sqrtps"
            | "sqrtpd"
            | "cvtsi2ss"
            | "cvtsi2sd"
            | "cvtss2si"
            | "cvtsd2si"
            | "cvtss2sd"
            | "cvtsd2ss"
    )
}

/// Instructions that load a new floating-point control word, and so
/// invalidate what [`pins_rounding_mode`] assumes.
///
/// x86 has two such words and this covers both. `ldmxcsr` loads the SSE
/// one; `fldcw` loads the x87 one, and `fldenv` / `frstor` reload it as
/// part of the environment they restore. `fxrstor` / `xrstor` restore
/// *both* wholesale as part of the extended state.
///
/// `fldcw` is the stronger hazard of the two, because the x87 control
/// word carries a precision-control field the SSE one does not:
/// narrowing it changes the significand width of every subsequent
/// arithmetic result, which no fixed sort can reflect.
pub(crate) fn writes_fp_control(mnemonic: &str) -> bool {
    let lower = mnemonic.trim().to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "ldmxcsr"
            | "vldmxcsr"
            | "fldcw"
            | "fldenv"
            | "frstor"
            | "fxrstor"
            | "fxrstor64"
            | "xrstor"
            | "xrstor64"
            | "xrstors"
    )
}

pub(crate) fn is_fp_compare_mnemonic(mnemonic: &str) -> bool {
    parse_fp_compare(mnemonic).is_some()
}

fn parse_fp_compare(mnemonic: &str) -> Option<FpCompare> {
    let lower = mnemonic.trim().to_ascii_lowercase();
    let body = lower.strip_prefix('v').unwrap_or(&lower);
    let rest = body.strip_prefix("cmp")?;
    let (pred_part, suffix) = rest.split_at(rest.len().checked_sub(2)?);
    let (lane_bits, packed) = match suffix {
        "ps" => (32, true),
        "pd" => (64, true),
        "ss" => (32, false),
        "sd" => (64, false),
        _ => return None,
    };
    let (pred, negated) = match pred_part {
        "eq" => (FpCmpPred::Eq, false),
        "lt" => (FpCmpPred::Lt, false),
        "le" => (FpCmpPred::Le, false),
        "unord" => (FpCmpPred::Unord, false),
        "neq" => (FpCmpPred::Eq, true),
        "nlt" => (FpCmpPred::Lt, true),
        "nle" => (FpCmpPred::Le, true),
        "ord" => (FpCmpPred::Unord, true),
        _ => return None,
    };
    Some(FpCompare {
        pred,
        lane_bits,
        packed,
        negated,
    })
}

/// One lane of a compare: the predicate as a 1-bit value, widened to a
/// full-lane mask of all-ones or all-zeros.
///
/// `None` for a lane width with no IEEE sort, so an unrecognised width
/// declines instead of being reinterpreted as a double.
fn fp_mask_lane(cmp: &FpCompare, a_bits: Expr, b_bits: Expr) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(cmp.lane_bits)?;
    let a = Expr::bv_to_fp(a_bits, ebits, sbits);
    let b = Expr::bv_to_fp(b_bits, ebits, sbits);
    let base = match cmp.pred {
        FpCmpPred::Eq => Expr::feq(a, b),
        FpCmpPred::Lt => Expr::flt(a, b),
        FpCmpPred::Le => Expr::fle(a, b),
        FpCmpPred::Unord => Expr::bool_or(Expr::fisnan(a), Expr::fisnan(b)),
    };
    let cond = if cmp.negated {
        Expr::BoolNot(Box::new(base))
    } else {
        base
    };
    Some(Expr::Ite {
        cond: Box::new(cond),
        then_expr: Box::new(super::simd::all_ones(cmp.lane_bits)),
        else_expr: Box::new(Expr::konst(0, cmp.lane_bits)),
    })
}

/// IEEE binary16 lane width used by the F16C conversions.
const F16C_HALF_BITS: u16 = 16;
/// IEEE binary32 lane width used by the F16C conversions.
const F16C_SINGLE_BITS: u16 = 32;
/// `vcvtps2ph` immediate bit selecting "round per MXCSR" instead of an
/// explicit mode.
const F16C_IMM_USE_MXCSR: u64 = 0b100;

/// Parse an immediate operand as radare2 renders it (`0`, `0x10`).
fn parse_immediate(raw: &str) -> Option<u64> {
    let text = raw.trim();
    text.strip_prefix("0x").map_or_else(
        || text.parse::<u64>().ok(),
        |hex| u64::from_str_radix(hex, 16).ok(),
    )
}
