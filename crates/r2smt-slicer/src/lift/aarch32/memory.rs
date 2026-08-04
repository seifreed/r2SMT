//! `AArch32` memory and register-list transfer handlers.
//!
//! Split out of `lift/aarch32.rs` when the addressing work took that
//! file past the 2 000-line hard limit. The addressing *model* — the
//! operand parsers, `MemOffset`, the PC value, the `MultipleMode`
//! table — stays in the parent, which is also where the mnemonic
//! dispatch that reaches these handlers lives; only the per-mnemonic
//! lowerings moved.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use super::super::{LiftCtx, MemAccess, Writeback, constant_delta, nonzero_width};
use super::{
    AARCH32_WORD_BYTES, MultipleMode, aarch32_addr_from, aarch32_base_writeback,
    aarch32_mem_access, aarch32_pc_value, parse_reglist, parse_vfp_reglist,
};

impl LiftCtx {
    /// Lift an `AArch32` load. `width_override` is `Some(8)` for
    /// `ldrb`/`ldrsb` and `Some(16)` for `ldrh`/`ldrsh`; `None` for the
    /// word-sized `ldr`. `signed` selects the sign-extending `ldrs*`
    /// family, which is the only difference between the two — the
    /// address, the width and the writeback are identical.
    pub(super) fn lift_aarch32_load(
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
        let Some(access) = aarch32_mem_access(
            mem,
            insn.operands.get(2..).unwrap_or_default(),
            self.bits,
            aarch32_pc_value(insn),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("ldr addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        let MemAccess { address, writeback } = access;
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
        if !self.write_register_to(dst, value) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "ldr destination not a supported register".into(),
            });
        }
        self.emit_writeback(writeback);
    }

    /// Resolve the memory operand of an `AArch32` doubleword transfer,
    /// which radare2 writes third (`ldrd r0, r1, [r2, 8]`).
    pub(super) fn aarch32_pair_access(&mut self, insn: &Instruction) -> Option<MemAccess> {
        let mem = insn.operands.get(2)?;
        if mem.kind != OperandKind::Memory {
            return None;
        }
        aarch32_mem_access(
            mem,
            insn.operands.get(3..).unwrap_or_default(),
            self.bits,
            aarch32_pc_value(insn),
        )
    }

    /// `ldrd Rt, Rt2, [mem]` — two consecutive words, `Rt` at the
    /// address and `Rt2` one word above it.
    pub(super) fn lift_aarch32_load_pair(&mut self, insn: &Instruction) {
        let (Some(first), Some(second)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.unsupported_aarch32(insn, "ldrd expects Rt, Rt2, [mem]");
            return;
        };
        if first.kind != OperandKind::Register || second.kind != OperandKind::Register {
            self.unsupported_aarch32(insn, "ldrd operand shape (non-Register pair)");
            return;
        }
        let Some(access) = self.aarch32_pair_access(insn) else {
            self.unsupported_aarch32(insn, "ldrd addressing mode not yet modelled");
            return;
        };
        let high = Expr::add(
            access.address.clone(),
            constant_delta(AARCH32_WORD_BYTES, self.bits),
        );
        for (dst, address) in [(first, access.address), (second, high)] {
            let tmp = self.new_temp(insn.address, self.bits);
            self.stmts.push(IrStmt::LoadMem {
                dst: tmp.clone(),
                address,
                bits: self.bits,
            });
            if !self.write_register_to(dst, Expr::Var(tmp)) {
                self.unsupported_aarch32(insn, "ldrd destination not a supported register");
                return;
            }
        }
        self.emit_writeback(access.writeback);
    }

    /// `strd Rt, Rt2, [mem]` — the store direction of
    /// [`Self::lift_aarch32_load_pair`].
    pub(super) fn lift_aarch32_store_pair(&mut self, insn: &Instruction) {
        let (Some(first), Some(second)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.unsupported_aarch32(insn, "strd expects Rt, Rt2, [mem]");
            return;
        };
        if first.kind != OperandKind::Register || second.kind != OperandKind::Register {
            self.unsupported_aarch32(insn, "strd operand shape (non-Register pair)");
            return;
        }
        let Some(access) = self.aarch32_pair_access(insn) else {
            self.unsupported_aarch32(insn, "strd addressing mode not yet modelled");
            return;
        };
        let high = Expr::add(
            access.address.clone(),
            constant_delta(AARCH32_WORD_BYTES, self.bits),
        );
        for (src, address) in [(first, access.address), (second, high)] {
            let value = self.read_operand_at(src, self.bits);
            self.stmts.push(IrStmt::StoreMem {
                address,
                value,
                bits: self.bits,
            });
        }
        self.emit_writeback(access.writeback);
    }

    /// Lift an `AArch32` store. `width_override` is `Some(8)` for `strb`
    /// and `Some(16)` for `strh` (the low bits of the source register);
    /// `None` for the word-sized `str`.
    pub(super) fn lift_aarch32_store(&mut self, insn: &Instruction, width_override: Option<u16>) {
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
        let Some(access) = aarch32_mem_access(
            mem,
            insn.operands.get(2..).unwrap_or_default(),
            self.bits,
            aarch32_pc_value(insn),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("str addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
        let value = self.read_operand_at(src, store_width);
        self.stmts.push(IrStmt::StoreMem {
            address: access.address,
            value,
            bits: store_width,
        });
        self.emit_writeback(access.writeback);
    }

    /// `push {regs}` ≡ `stmdb sp!, {regs}` — store each register to a
    /// descending stack slot (lowest register number at the lowest
    /// address) and decrement `sp` by `4 * n`.
    pub(super) fn lift_aarch32_push(&mut self, insn: &Instruction) {
        let Some(regs) = insn.operands.first().and_then(|o| parse_reglist(&o.raw)) else {
            self.unsupported_aarch32(insn, "push expects a register list");
            return;
        };
        let n = i64::try_from(regs.len()).unwrap_or(0);
        self.emit_store_multiple("sp", &regs, -4 * n);
        self.emit_writeback(Some(Writeback::by_constant("sp", -4 * n, self.bits)));
    }

    /// `pop {regs}` ≡ `ldmia sp!, {regs}` — load each register from an
    /// ascending stack slot and increment `sp` by `4 * n`.
    pub(super) fn lift_aarch32_pop(&mut self, insn: &Instruction) {
        let Some(regs) = insn.operands.first().and_then(|o| parse_reglist(&o.raw)) else {
            self.unsupported_aarch32(insn, "pop expects a register list");
            return;
        };
        let n = i64::try_from(regs.len()).unwrap_or(0);
        self.emit_load_multiple(insn, "sp", &regs, 0);
        self.emit_writeback(Some(Writeback::by_constant("sp", 4 * n, self.bits)));
    }

    /// `vpush {regs}` — store each VFP register to a descending stack
    /// slot (lowest register at the lowest address) and decrement `sp`
    /// by the total byte count. `s` registers are 4 bytes, `d` are 8.
    pub(super) fn lift_aarch32_vpush(&mut self, insn: &Instruction) {
        let Some((regs, width)) = insn
            .operands
            .first()
            .and_then(|o| parse_vfp_reglist(&o.raw))
        else {
            self.unsupported_aarch32(insn, "vpush expects a VFP register list");
            return;
        };
        let stride = i64::from(width / 8);
        let total = stride * i64::try_from(regs.len()).unwrap_or(0);
        for (i, reg) in regs.iter().enumerate() {
            let off = -total + stride * i64::try_from(i).unwrap_or(0);
            let address = aarch32_addr_from("sp", off, self.bits);
            let Some(value) = self.read_simd_lane_bits(reg, width, 0) else {
                self.unsupported_aarch32(insn, "vpush source not modelled");
                return;
            };
            self.stmts.push(IrStmt::StoreMem {
                address,
                value,
                bits: width,
            });
        }
        self.emit_writeback(Some(Writeback::by_constant("sp", -total, self.bits)));
    }

    /// `vpop {regs}` — load each VFP register from an ascending stack
    /// slot and increment `sp` by the total byte count.
    pub(super) fn lift_aarch32_vpop(&mut self, insn: &Instruction) {
        let Some((regs, width)) = insn
            .operands
            .first()
            .and_then(|o| parse_vfp_reglist(&o.raw))
        else {
            self.unsupported_aarch32(insn, "vpop expects a VFP register list");
            return;
        };
        let stride = i64::from(width / 8);
        let total = stride * i64::try_from(regs.len()).unwrap_or(0);
        for (i, reg) in regs.iter().enumerate() {
            let off = stride * i64::try_from(i).unwrap_or(0);
            let address = aarch32_addr_from("sp", off, self.bits);
            let tmp = self.new_temp(insn.address, width);
            self.stmts.push(IrStmt::LoadMem {
                dst: tmp.clone(),
                address,
                bits: width,
            });
            if !self.write_simd_lane(reg, Expr::Var(tmp), width, 0) {
                self.unsupported_aarch32(insn, "vpop destination not modelled");
                return;
            }
        }
        self.emit_writeback(Some(Writeback::by_constant("sp", total, self.bits)));
    }

    /// `ldm<mode> Rn{!}, {regs}` — load multiple from the explicit base
    /// register in any of the four addressing modes, with optional
    /// writeback.
    pub(super) fn lift_aarch32_ldm(&mut self, insn: &Instruction, mode: MultipleMode) {
        let (Some(base_op), Some(list_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.unsupported_aarch32(insn, "ldm expects base and register list");
            return;
        };
        let (Some((base, writeback)), Some(regs)) = (
            aarch32_base_writeback(&base_op.raw),
            parse_reglist(&list_op.raw),
        ) else {
            self.unsupported_aarch32(insn, "ldm base or list not modelled");
            return;
        };
        let (start, delta) = mode.span(regs.len());
        self.emit_load_multiple(insn, &base, &regs, start);
        if writeback {
            self.emit_writeback(Some(Writeback::by_constant(&base, delta, self.bits)));
        }
    }

    /// `stm<mode> Rn{!}, {regs}` — store multiple to the explicit base
    /// register in any of the four addressing modes, with optional
    /// writeback.
    pub(super) fn lift_aarch32_stm(&mut self, insn: &Instruction, mode: MultipleMode) {
        let (Some(base_op), Some(list_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.unsupported_aarch32(insn, "stm expects base and register list");
            return;
        };
        let (Some((base, writeback)), Some(regs)) = (
            aarch32_base_writeback(&base_op.raw),
            parse_reglist(&list_op.raw),
        ) else {
            self.unsupported_aarch32(insn, "stm base or list not modelled");
            return;
        };
        let (start, delta) = mode.span(regs.len());
        self.emit_store_multiple(&base, &regs, start);
        if writeback {
            self.emit_writeback(Some(Writeback::by_constant(&base, delta, self.bits)));
        }
    }

    /// Store `regs` at `base + start_off + 4*i` (ascending register
    /// number → ascending address), one word each.
    pub(super) fn emit_store_multiple(&mut self, base: &str, regs: &[String], start_off: i64) {
        for (i, reg) in regs.iter().enumerate() {
            let off = start_off + 4 * i64::try_from(i).unwrap_or(0);
            let address = aarch32_addr_from(base, off, self.bits);
            let value = self.read_named_register(reg, self.bits);
            self.stmts.push(IrStmt::StoreMem {
                address,
                value,
                bits: self.bits,
            });
        }
    }

    /// Load `regs` from `base + start_off + 4*i` into each register.
    pub(super) fn emit_load_multiple(
        &mut self,
        insn: &Instruction,
        base: &str,
        regs: &[String],
        start_off: i64,
    ) {
        for (i, reg) in regs.iter().enumerate() {
            let off = start_off + 4 * i64::try_from(i).unwrap_or(0);
            let address = aarch32_addr_from(base, off, self.bits);
            let tmp = self.new_temp(insn.address, self.bits);
            self.stmts.push(IrStmt::LoadMem {
                dst: tmp.clone(),
                address,
                bits: self.bits,
            });
            if !self.write_named_register(reg, Expr::Var(tmp)) {
                self.unsupported_aarch32(insn, "ldm/pop destination not a register");
            }
        }
    }

    fn read_named_register(&self, name: &str, width: u16) -> Expr {
        let op = Operand {
            raw: name.to_string(),
            kind: OperandKind::Register,
        };
        self.read_operand_at(&op, width)
    }

    fn write_named_register(&mut self, name: &str, value: Expr) -> bool {
        let op = Operand {
            raw: name.to_string(),
            kind: OperandKind::Register,
        };
        self.write_register_to(&op, value)
    }
}
