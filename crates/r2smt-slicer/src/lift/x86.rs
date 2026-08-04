//! x86 / `x86_64` per-mnemonic lifter handlers, extracted from
//! `lift.rs`. Methods on [`LiftCtx`]; shared infrastructure stays in
//! the parent module.

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

mod simd;

use crate::registers::{is_simd_parent, register_layout};

use super::simd::CompareKind;
use super::{BinOp, PackedIntOp};
use super::{
    BitwiseOp, ExtendKind, FpArithOp, LiftCtx, ShiftOp, fp_sort_bits_checked, nonzero_width,
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
            "ptest" | "vptest" => self.lift_simd_vector_test(insn),
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

/// Flags `ptest` architecturally clears. `ZF` and `CF` carry the result
/// and are written separately.
const VECTOR_TEST_CLEARED_FLAGS: [&str; 3] = ["OF", "SF", "PF"];

/// Whether `mnemonic` reads vector operands, writes flags, and defines
/// no register — the vector bitwise test `ptest` and the scalar
/// floating-point compares.
///
/// These are deliberately **not** in [`x86_simd_shape`]. That table's
/// effect-side arm runs before the compare arms and would make operand 0
/// a definition, so the slicer would treat `ptest xmm0, xmm0` as
/// redefining `xmm0` and drop whatever actually produced it — the
/// def/use asymmetry that fabricates verdicts. Membership therefore has
/// to live here, and the dispatch gate and the effect table consult this
/// one predicate rather than each keeping a list.
///
/// The compares are load-bearing in the *gate*, and were lost there when
/// the two hand-maintained lists were folded into `x86_simd_shape`: they
/// have no register destination, so nothing else in
/// `is_x86_simd_instruction` claims them, and an unclaimed mnemonic
/// falls to the ESIL ladder — which models no floating point at all and
/// so drops the flags entirely. `ucomiss xmm0, xmm0 ; jne` then lifts to
/// an *empty* slice with a free `ZF`, turning a correct `always_false`
/// into `both_possible`. Sound, since a free flag only widens, but it
/// silently discards the analysis.
pub(crate) fn is_x86_vector_flag_compare(mnemonic: &str) -> bool {
    let lower = mnemonic.trim().to_ascii_lowercase();
    X86_VECTOR_FLAG_COMPARES.contains(&lower.as_str())
}

/// The mnemonics [`is_x86_vector_flag_compare`] claims.
///
/// A slice rather than a `matches!` so the parity guard can iterate it.
/// The `ucomiss` regression these instructions carry was invisible
/// precisely because the guard's membership list was retyped by hand
/// and this family was never on it.
pub(crate) const X86_VECTOR_FLAG_COMPARES: &[&str] =
    &["ptest", "vptest", "comiss", "ucomiss", "comisd", "ucomisd"];

/// The lane-wise arithmetic `PackedIntOp`s x86 spells.
const ADD: PackedIntOp = PackedIntOp::Bin(BinOp::Add);
const SUB: PackedIntOp = PackedIntOp::Bin(BinOp::Sub);
const MUL: PackedIntOp = PackedIntOp::Bin(BinOp::Mul);

/// `padds*` / `paddus*` / `psubs*` / `psubus*` — the saturating forms.
/// The `signed` flag picks both the extension of the operands and the
/// bounds they clamp to, so `paddusb` stops at `0xff` where `paddsb`
/// stops at `0x7f`.
const SAT_ADD_SIGNED: PackedIntOp = PackedIntOp::Saturating {
    subtract: false,
    signed: true,
};
const SAT_ADD_UNSIGNED: PackedIntOp = PackedIntOp::Saturating {
    subtract: false,
    signed: false,
};
const SAT_SUB_SIGNED: PackedIntOp = PackedIntOp::Saturating {
    subtract: true,
    signed: true,
};
const SAT_SUB_UNSIGNED: PackedIntOp = PackedIntOp::Saturating {
    subtract: true,
    signed: false,
};

/// `pavgb` / `pavgw` — the unsigned rounded average `(a + b + 1) >> 1`.
/// `rounding` is the half added before the shift, and dropping it would
/// be a wrong value rather than a decline: `pavgb` of `3` and `4` is
/// `4`, not `3`.
const AVERAGE: PackedIntOp = PackedIntOp::Halving {
    subtract: false,
    signed: false,
    rounding: true,
};

/// `pmax*` / `pmin*`. The comparison *is* the operation, so signedness
/// is carried rather than inferred — `pmaxub` of `0xff` and `0x01` is
/// `0xff`, `pmaxsb` of the same lanes is `0x01`.
const MAX_SIGNED: PackedIntOp = PackedIntOp::MinMax {
    max: true,
    signed: true,
};
const MAX_UNSIGNED: PackedIntOp = PackedIntOp::MinMax {
    max: true,
    signed: false,
};
const MIN_SIGNED: PackedIntOp = PackedIntOp::MinMax {
    max: false,
    signed: true,
};
const MIN_UNSIGNED: PackedIntOp = PackedIntOp::MinMax {
    max: false,
    signed: false,
};

/// `pabs{b,w,d}` — the lane-wise absolute value, and the one x86 packed
/// integer arithmetic form with a single source. Non-saturating, so
/// `pabsb` of `0x80` is `0x80`; the two's-complement negation at a fixed
/// width already gives that with no special case.
const ABSOLUTE: PackedIntOp = PackedIntOp::Abs;

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
    if X86_SIMD_MOVES.contains(&mnemonic) {
        return Some(X86SimdShape::Move);
    }
    X86_SIMD_READ_MODIFY_WRITES
        .contains(&mnemonic)
        .then_some(X86SimdShape::ReadModifyWrite)
}

/// The [`X86SimdShape::Move`] half of the membership table: the
/// destination is fully overwritten, so it is a def and not also a use.
pub(crate) const X86_SIMD_MOVES: &[&str] = &[
    "movaps",
    "movups",
    "movapd",
    "movupd",
    "movdqa",
    "movdqu",
    "vmovaps",
    "vmovups",
    "vmovapd",
    "vmovupd",
    "vmovdqa",
    "vmovdqu",
    "cvtss2si",
    "cvtsd2si",
    "cvttss2si",
    "cvttsd2si",
    "sqrtps",
    "sqrtpd",
    "vsqrtps",
    "vsqrtpd",
    "vcvtph2ps",
    "vcvtps2ph",
    // `pmovmskb` writes a general register; `movd`/`movq` go either
    // way. All three overwrite the whole destination — the vector
    // direction zeroes above the transferred value rather than merging.
    "pmovmskb",
    "vpmovmskb",
    "movd",
    "vmovd",
    "movq",
    "vmovq",
    // `pabs*` is the one packed integer arithmetic form whose
    // 2-operand shape is not read-modify-write: it writes the
    // destination from its single source.
    "pabsb",
    "pabsw",
    "pabsd",
    "vpabsb",
    "vpabsw",
    "vpabsd",
    // The `pshuf*` family has three operands, but the third is the
    // immediate selector rather than a second source, and every
    // destination lane comes from the one source.
    "pshufd",
    "pshuflw",
    "pshufhw",
    "vpshufd",
    "vpshuflw",
    "vpshufhw",
];

/// The [`X86SimdShape::ReadModifyWrite`] half of the membership table:
/// the 2-operand form reads its destination as a source, while the
/// 3-operand VEX form reads its two explicit sources instead.
pub(crate) const X86_SIMD_READ_MODIFY_WRITES: &[&str] = &[
    "pxor",
    "vpxor",
    "pand",
    "vpand",
    "por",
    "vpor",
    "pandn",
    "vpandn",
    "addss",
    "subss",
    "mulss",
    "divss",
    "addsd",
    "subsd",
    "mulsd",
    "divsd",
    "addps",
    "subps",
    "mulps",
    "divps",
    "vaddps",
    "vsubps",
    "vmulps",
    "vdivps",
    "addpd",
    "subpd",
    "mulpd",
    "divpd",
    "vaddpd",
    "vsubpd",
    "vmulpd",
    "vdivpd",
    "maxps",
    "minps",
    "maxpd",
    "minpd",
    "vmaxps",
    "vminps",
    "vmaxpd",
    "vminpd",
    "maxss",
    "minss",
    "maxsd",
    "minsd",
    "cvtsi2ss",
    "cvtsi2sd",
    "cvtss2sd",
    "cvtsd2ss",
    "sqrtss",
    "sqrtsd",
    // The packed integer compares overwrite every lane, but the
    // 2-operand form still reads the destination as its first source,
    // so they share the arithmetic's shape.
    "pcmpeqb",
    "pcmpeqw",
    "pcmpeqd",
    "pcmpeqq",
    "pcmpgtb",
    "pcmpgtw",
    "pcmpgtd",
    "pcmpgtq",
    "vpcmpeqb",
    "vpcmpeqw",
    "vpcmpeqd",
    "vpcmpeqq",
    "vpcmpgtb",
    "vpcmpgtw",
    "vpcmpgtd",
    "vpcmpgtq",
    // Packed integer arithmetic and the shifts, same shape again.
    "paddb",
    "paddw",
    "paddd",
    "paddq",
    "psubb",
    "psubw",
    "psubd",
    "psubq",
    "pmullw",
    "pmulld",
    "pslldq",
    "psrldq",
    "psllw",
    "pslld",
    "psllq",
    "psrlw",
    "psrld",
    "psrlq",
    "psraw",
    "psrad",
    "vpaddb",
    "vpaddw",
    "vpaddd",
    "vpaddq",
    "vpsubb",
    "vpsubw",
    "vpsubd",
    "vpsubq",
    "vpmullw",
    "vpmulld",
    "vpslldq",
    "vpsrldq",
    "vpsllw",
    "vpslld",
    "vpsllq",
    "vpsrlw",
    "vpsrld",
    "vpsrlq",
    "vpsraw",
    "vpsrad",
    // The saturating, averaging and min/max families. Same
    // read-modify-write shape as the wrapping arithmetic above — only
    // the lane operation differs.
    "paddsb",
    "paddsw",
    "paddusb",
    "paddusw",
    "psubsb",
    "psubsw",
    "psubusb",
    "psubusw",
    "pavgb",
    "pavgw",
    "pmaxsb",
    "pmaxsw",
    "pmaxsd",
    "pmaxub",
    "pmaxuw",
    "pmaxud",
    "pminsb",
    "pminsw",
    "pminsd",
    "pminub",
    "pminuw",
    "pminud",
    "vpaddsb",
    "vpaddsw",
    "vpaddusb",
    "vpaddusw",
    "vpsubsb",
    "vpsubsw",
    "vpsubusb",
    "vpsubusw",
    "vpavgb",
    "vpavgw",
    "vpmaxsb",
    "vpmaxsw",
    "vpmaxsd",
    "vpmaxub",
    "vpmaxuw",
    "vpmaxud",
    "vpminsb",
    "vpminsw",
    "vpminsd",
    "vpminub",
    "vpminuw",
    "vpminud",
    // The interleaves read the destination as their first source in the
    // 2-operand form, like the arithmetic.
    "punpcklbw",
    "punpcklwd",
    "punpckldq",
    "punpcklqdq",
    "punpckhbw",
    "punpckhwd",
    "punpckhdq",
    "punpckhqdq",
    "vpunpcklbw",
    "vpunpcklwd",
    "vpunpckldq",
    "vpunpcklqdq",
    "vpunpckhbw",
    "vpunpckhwd",
    "vpunpckhdq",
    "vpunpckhqdq",
    // `pshufb` shuffles the destination itself, indexed by the source,
    // so the destination is a use as well; the packs fill their
    // destination from both operands.
    "pshufb",
    "vpshufb",
    "packsswb",
    "packssdw",
    "packuswb",
    "packusdw",
    "vpacksswb",
    "vpackssdw",
    "vpackuswb",
    "vpackusdw",
];

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
