//! `AArch32` per-mnemonic lifter handlers, extracted from `lift.rs`.
//! Methods on [`LiftCtx`]; reuses the `AArch64` 3-operand family and
//! shared infrastructure from the parent module.

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::register_layout;

use super::{
    BinOp, LiftCtx, MemAccess, aarch64_cond_suffix_to_predicate, is_aarch32_base_supported,
    nonzero_width, strip_aarch32_cond_suffix, width_mask,
};

impl LiftCtx {
    pub(super) fn lift_instruction_aarch32(&mut self, insn: &Instruction) {
        // AArch32 instruction shapes mirror AArch64 (3-operand
        // arithmetic / 2-operand compare). The lifter reuses the
        // AArch64 handler family — register reads / writes flow
        // through `register_layout(name, self.arch)` which respects
        // `Arch::Arm` and produces `r0..r15` parents.
        let mnem = insn.mnemonic.trim().to_ascii_lowercase();
        // Conditional execution suffix: `<base><cond>` such as `addeq`
        // or `subne`. Strip the recognised tail, look up the cond
        // predicate, and wrap every assignment the base handler emits
        // in `Ite(cond, new, old)` so flags and destination writes
        // become predicated. `al` (always) is the unmodified base;
        // `nv` (never) is reserved and treated as predicated with a
        // constant-false condition for soundness.
        if let Some((base, cond_suffix)) = strip_aarch32_cond_suffix(&mnem)
            && is_aarch32_base_supported(base)
            && let Some(cond_expr) = aarch64_cond_suffix_to_predicate(cond_suffix)
        {
            self.lift_aarch32_predicated(insn, base, &cond_expr);
            return;
        }
        match mnem.as_str() {
            "mov" => self.lift_aarch64_mov(insn),
            "mvn" => self.lift_aarch32_mvn(insn),
            "add" => self.lift_aarch64_arith3(insn, BinOp::Add, false),
            "adds" => self.lift_aarch64_arith3(insn, BinOp::Add, true),
            "sub" => self.lift_aarch64_arith3(insn, BinOp::Sub, false),
            "subs" => self.lift_aarch64_arith3(insn, BinOp::Sub, true),
            // `rsb Rd, Rn, Op` ≡ `sub Rd, Op, Rn` (reverse subtract).
            "rsb" => self.lift_aarch32_rsb(insn, false),
            "rsbs" => self.lift_aarch32_rsb(insn, true),
            "and" => self.lift_aarch64_arith3(insn, BinOp::And, false),
            "ands" => self.lift_aarch64_arith3(insn, BinOp::And, true),
            // `bic Rd, Rn, Op` = `and Rd, Rn, ~Op`. Bit-clear.
            "bic" => self.lift_aarch32_bic(insn, false),
            "bics" => self.lift_aarch32_bic(insn, true),
            "orr" => self.lift_aarch64_arith3(insn, BinOp::Or, false),
            "orrs" => self.lift_aarch64_arith3(insn, BinOp::Or, true),
            "eor" => self.lift_aarch64_arith3(insn, BinOp::Xor, false),
            "eors" => self.lift_aarch64_arith3(insn, BinOp::Xor, true),
            "mul" => self.lift_aarch64_arith3(insn, BinOp::Mul, false),
            "muls" => self.lift_aarch64_arith3(insn, BinOp::Mul, true),
            // AArch32 integer divide (`udiv` / `sdiv`) — ARMv7-A
            // optional, ARMv8 mandatory. Same 3-operand shape as the
            // arithmetic family; never set flags.
            "udiv" => self.lift_aarch64_arith3(insn, BinOp::UDiv, false),
            "sdiv" => self.lift_aarch64_arith3(insn, BinOp::SDiv, false),
            "lsl" => self.lift_aarch64_arith3(insn, BinOp::Shl, false),
            "lsls" => self.lift_aarch64_arith3(insn, BinOp::Shl, true),
            "lsr" => self.lift_aarch64_arith3(insn, BinOp::Shr, false),
            "lsrs" => self.lift_aarch64_arith3(insn, BinOp::Shr, true),
            "asr" => self.lift_aarch64_arith3(insn, BinOp::Sar, false),
            "asrs" => self.lift_aarch64_arith3(insn, BinOp::Sar, true),
            "cmp" => self.lift_aarch64_cmp(insn),
            // `cmn Rn, Op` = compare-negative, sets flags from Rn + Op.
            "cmn" => self.lift_aarch32_cmn(insn),
            "tst" => self.lift_aarch64_tst(insn),
            // `teq Rn, Op` = test-equivalence, sets flags from Rn ^ Op.
            "teq" => self.lift_aarch32_teq(insn),
            // Memory in offset form `[Rn]` / `[Rn, #imm]`. Word (`ldr`/
            // `str`), byte (`ldrb`/`strb`), and halfword (`ldrh`/`strh`)
            // widths — byte/halfword loads zero-extend to the 32-bit
            // register. Register-offset, writeback, predicated
            // (`ldreq`), and the sign-extending `ldrsb`/`ldrsh` forms
            // are not in this match and decline to `Unsupported` —
            // sound, the confidence path widens rather than mis-lifting.
            "ldr" => self.lift_aarch32_load(insn, None),
            "ldrb" => self.lift_aarch32_load(insn, Some(8)),
            "ldrh" => self.lift_aarch32_load(insn, Some(16)),
            "str" => self.lift_aarch32_store(insn, None),
            "strb" => self.lift_aarch32_store(insn, Some(8)),
            "strh" => self.lift_aarch32_store(insn, Some(16)),
            // Register-list load/store multiple. `push`/`pop` are the
            // stack idioms (`stmdb sp!` / `ldmia sp!`); `ldm`/`stm`
            // (increment-after) carry an explicit base. Other
            // addressing suffixes (`stmdb`/`ldmdb`/`stmib`/`ldmda`) are
            // not in this match and decline soundly.
            "push" => self.lift_aarch32_push(insn),
            "pop" => self.lift_aarch32_pop(insn),
            "ldm" | "ldmia" => self.lift_aarch32_ldm(insn),
            "stm" | "stmia" => self.lift_aarch32_stm(insn),
            _ => self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("at {addr} (aarch32)", addr = insn.address),
            }),
        }
    }

    /// `rsb Rd, Rn, Op` — reverse subtract: `Rd := Op - Rn`. Delegates
    /// to the 3-operand handler with `Rn`/`Op` swapped so the flag-
    /// ordering fix and operand-validation invariants stay in one
    /// place.
    fn lift_aarch32_rsb(&mut self, insn: &Instruction, sets_flags: bool) {
        let (Some(dst), Some(src1), Some(src2)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "rsb needs 3 operands".into(),
            });
            return;
        };
        let mut swapped = insn.clone();
        swapped.operands = vec![dst.clone(), src2.clone(), src1.clone()];
        self.lift_aarch64_arith3(&swapped, BinOp::Sub, sets_flags);
    }

    /// `bic Rd, Rn, Op` — bit-clear: `Rd := Rn & ~Op`.
    fn lift_aarch32_bic(&mut self, insn: &Instruction, sets_flags: bool) {
        let (Some(dst), Some(src1), Some(src2)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "bic needs 3 operands".into(),
            });
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (bic)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (bic)".into(),
            });
            return;
        };
        let lhs = self.read_operand_at(src1, dst_width);
        let rhs = self.read_operand_at(src2, dst_width);
        // ~Op = Op XOR all-ones.
        let ones = Expr::konst(width_mask(dst_width), dst_width);
        let not_rhs = Expr::bv_xor(rhs.clone(), ones);
        let computed = Expr::bv_and(lhs.clone(), not_rhs);
        let tmp = self.new_temp(insn.address, dst_width);
        self.assign(tmp.clone(), computed);
        let tmp_expr = Expr::Var(tmp);
        if sets_flags {
            // Logical-op flag policy mirrors AArch64 `ands` (CF/OF clear,
            // ZF/SF from the result). Emit before the destination write
            // so any dst/src overlap doesn't rename the lhs/rhs reads
            // — see `lift_aarch64_arith3`.
            self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, dst_width)));
            self.set_flag("SF", Expr::slt(tmp_expr.clone(), Expr::konst(0, dst_width)));
            self.set_flag("CF", Expr::konst(0, 1));
            self.set_flag("OF", Expr::konst(0, 1));
            self.set_flag("PF", Expr::Unknown(String::new()));
        }
        if !self.write_register_to(dst, tmp_expr) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (bic)".into(),
            });
        }
    }

    /// `cmn Rn, Op` — compare-negative: sets flags from `Rn + Op`,
    /// no register destination. Mirrors [`Self::lift_aarch64_cmp`].
    fn lift_aarch32_cmn(&mut self, insn: &Instruction) {
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, width);
        let rhs = self.read_operand_at(rhs_op, width);
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), Expr::add(lhs, rhs));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, width)));
        // CF/OF on `cmn` need a full extension to compute precisely;
        // mark Unknown rather than fabricate a value.
        self.set_flag("CF", Expr::Unknown(String::new()));
        self.set_flag("OF", Expr::Unknown(String::new()));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    /// `teq Rn, Op` — test-equivalence: sets flags from `Rn ^ Op`,
    /// no register destination. Mirrors [`Self::lift_aarch64_tst`] but
    /// with XOR instead of AND.
    fn lift_aarch32_teq(&mut self, insn: &Instruction) {
        let (Some(lhs_op), Some(rhs_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        let width = self.binop_width(lhs_op, rhs_op);
        let lhs = self.read_operand_at(lhs_op, width);
        let rhs = self.read_operand_at(rhs_op, width);
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), Expr::bv_xor(lhs, rhs));
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(tmp_expr, Expr::konst(0, width)));
        // `teq` clears C and V on AArch32 (architectural behaviour).
        self.set_flag("CF", Expr::konst(0, 1));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    fn lift_aarch32_predicated(&mut self, insn: &Instruction, base: &str, cond_expr: &Expr) {
        // Re-enter the AArch32 dispatcher with the cond suffix peeled
        // off, then wrap every `Assign` it emitted in
        // `Ite(cond, new_src, Var(dst))`. The SSA pass downstream
        // turns `Var(dst)` into the previous version of the
        // destination, so on the `cond == 0` path the assignment
        // becomes a no-op — the value that flowed in from before the
        // predicated body persists.
        let mut base_insn = insn.clone();
        base_insn.mnemonic = base.to_string();
        let start_idx = self.stmts.len();
        // Reentrant call: at this point `mnemonic` no longer carries
        // a cond suffix, so `strip_aarch32_cond_suffix` returns `None`
        // and the `match` body executes normally.
        self.lift_instruction_aarch32(&base_insn);
        for stmt in self.stmts.iter_mut().skip(start_idx) {
            if let IrStmt::Assign { dst, src } = stmt {
                let old_value = Expr::Var(dst.clone());
                let placeholder = Expr::unknown();
                let new_src = std::mem::replace(src, placeholder);
                *src = Expr::Ite {
                    cond: Box::new(cond_expr.clone()),
                    then_expr: Box::new(new_src),
                    else_expr: Box::new(old_value),
                };
            }
        }
    }

    fn lift_aarch32_mvn(&mut self, insn: &Instruction) {
        // `mvn Rd, Op` = bitwise NOT. Encoded as Xor with -1 of the
        // destination width.
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (mvn)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            return;
        };
        let value = self.read_operand_at(src, dst_width);
        let result = Expr::bv_xor(value, Expr::konst(width_mask(dst_width), dst_width));
        if !self.write_register_to(dst, result) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (mvn)".into(),
            });
        }
    }

    /// Lift an `AArch32` load. `width_override` is `Some(8)` for `ldrb`
    /// and `Some(16)` for `ldrh` (zero-extended into the 32-bit
    /// register); `None` for the word-sized `ldr`.
    fn lift_aarch32_load(&mut self, insn: &Instruction, width_override: Option<u8>) {
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
        let Some(access) = aarch32_mem_access(mem, insn.operands.get(2), self.bits) else {
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
            Expr::zero_ext(Expr::Var(tmp), dst_width)
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

    /// Lift an `AArch32` store. `width_override` is `Some(8)` for `strb`
    /// and `Some(16)` for `strh` (the low bits of the source register);
    /// `None` for the word-sized `str`.
    fn lift_aarch32_store(&mut self, insn: &Instruction, width_override: Option<u8>) {
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
        let Some(access) = aarch32_mem_access(mem, insn.operands.get(2), self.bits) else {
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
    fn lift_aarch32_push(&mut self, insn: &Instruction) {
        let Some(regs) = insn.operands.first().and_then(|o| parse_reglist(&o.raw)) else {
            self.unsupported_aarch32(insn, "push expects a register list");
            return;
        };
        let n = i64::try_from(regs.len()).unwrap_or(0);
        self.emit_store_multiple("sp", &regs, -4 * n);
        self.emit_writeback(Some(("sp".to_string(), -4 * n)));
    }

    /// `pop {regs}` ≡ `ldmia sp!, {regs}` — load each register from an
    /// ascending stack slot and increment `sp` by `4 * n`.
    fn lift_aarch32_pop(&mut self, insn: &Instruction) {
        let Some(regs) = insn.operands.first().and_then(|o| parse_reglist(&o.raw)) else {
            self.unsupported_aarch32(insn, "pop expects a register list");
            return;
        };
        let n = i64::try_from(regs.len()).unwrap_or(0);
        self.emit_load_multiple(insn, "sp", &regs, 0);
        self.emit_writeback(Some(("sp".to_string(), 4 * n)));
    }

    /// `ldm{ia} Rn{!}, {regs}` — increment-after load multiple from the
    /// explicit base register, with optional writeback.
    fn lift_aarch32_ldm(&mut self, insn: &Instruction) {
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
        let n = i64::try_from(regs.len()).unwrap_or(0);
        self.emit_load_multiple(insn, &base, &regs, 0);
        if writeback {
            self.emit_writeback(Some((base, 4 * n)));
        }
    }

    /// `stm{ia} Rn{!}, {regs}` — increment-after store multiple to the
    /// explicit base register, with optional writeback.
    fn lift_aarch32_stm(&mut self, insn: &Instruction) {
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
        let n = i64::try_from(regs.len()).unwrap_or(0);
        self.emit_store_multiple(&base, &regs, 0);
        if writeback {
            self.emit_writeback(Some((base, 4 * n)));
        }
    }

    /// Store `regs` at `base + start_off + 4*i` (ascending register
    /// number → ascending address), one word each.
    fn emit_store_multiple(&mut self, base: &str, regs: &[String], start_off: i64) {
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
    fn emit_load_multiple(
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

    fn read_named_register(&self, name: &str, width: u8) -> Expr {
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

    fn unsupported_aarch32(&mut self, insn: &Instruction, reason: &str) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: reason.to_string(),
        });
    }
}

/// Parse a `{r4, r5, lr}` register list into canonical register names
/// sorted by architectural register number (lowest number first, which
/// is the lowest stack address). `None` if any entry is not a
/// recognised `AArch32` GPR.
fn parse_reglist(raw: &str) -> Option<Vec<String>> {
    let body = raw.trim().strip_prefix('{')?.strip_suffix('}')?;
    let mut regs: Vec<String> = body
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if regs.is_empty() || regs.iter().any(|r| aarch32_reg_number(r).is_none()) {
        return None;
    }
    regs.sort_by_key(|r| aarch32_reg_number(r).unwrap_or(u8::MAX));
    Some(regs)
}

/// Architectural register number for an `AArch32` GPR name (`r0..r15`
/// plus the AAPCS aliases). `None` for anything else.
fn aarch32_reg_number(name: &str) -> Option<u8> {
    match name.trim().to_ascii_lowercase().as_str() {
        "sp" => Some(13),
        "lr" => Some(14),
        "pc" => Some(15),
        "fp" => Some(11),
        "ip" => Some(12),
        "sb" => Some(9),
        "sl" => Some(10),
        other => other
            .strip_prefix('r')
            .and_then(|d| d.parse::<u8>().ok())
            .filter(|&n| n <= 15),
    }
}

/// Split an `ldm`/`stm` base operand `Rn` or `Rn!` into
/// `(base_name, writeback)`. `None` if `Rn` is not a recognised base.
fn aarch32_base_writeback(raw: &str) -> Option<(String, bool)> {
    let trimmed = raw.trim();
    let (name, writeback) = match trimmed.strip_suffix('!') {
        Some(base) => (base.trim(), true),
        None => (trimmed, false),
    };
    let parent = register_layout(name, Arch::Arm).map(|l| l.parent)?;
    Some((parent.to_string(), writeback))
}

/// Resolve an `AArch32` memory operand, including the pre-index
/// (`[Rn, #imm]!`) and post-index (`[Rn], #imm`) writeback forms.
/// Mirrors the `AArch64` resolver but validates the base through the
/// `Arch::Arm` register table. Unrecognised shapes still return `None`.
fn aarch32_mem_access(mem: &Operand, post: Option<&Operand>, ptr_bits: u8) -> Option<MemAccess> {
    if mem.kind != OperandKind::Memory {
        return None;
    }
    let raw = mem.raw.trim();
    if let Some(body) = raw.strip_suffix('!') {
        let (base, offset) = parse_aarch32_memory(body.trim())?;
        let parent = register_layout(&base, Arch::Arm).map(|l| l.parent)?;
        return Some(MemAccess {
            address: aarch32_addr_from(parent, offset, ptr_bits),
            writeback: Some((parent.to_string(), offset)),
        });
    }
    let (base, offset) = parse_aarch32_memory(raw)?;
    let parent = register_layout(&base, Arch::Arm).map(|l| l.parent)?;
    if let Some(op) = post
        && op.kind == OperandKind::Immediate
        && offset == 0
    {
        let delta = parse_aarch32_immediate(op.raw.strip_prefix('#').unwrap_or(&op.raw).trim())?;
        return Some(MemAccess {
            address: Expr::Var(Var::new(parent, ptr_bits)),
            writeback: Some((parent.to_string(), delta)),
        });
    }
    Some(MemAccess {
        address: aarch32_addr_from(parent, offset, ptr_bits),
        writeback: None,
    })
}

/// Build `base ± offset` at the pointer width (base alone if zero).
fn aarch32_addr_from(parent: &str, offset: i64, ptr_bits: u8) -> Expr {
    let base_var = Expr::Var(Var::new(parent, ptr_bits));
    if offset == 0 {
        return base_var;
    }
    let masked = u64::from_le_bytes(offset.to_le_bytes()) & width_mask(ptr_bits);
    Expr::add(base_var, Expr::konst(masked, ptr_bits))
}

/// Split `[base{, #?imm}]` into `(base, offset)`; `None` for any shape
/// outside the supported offset subset (writeback keeps the `!` or a
/// post-index `, #imm` outside the brackets; register-offset yields a
/// non-numeric second part).
fn parse_aarch32_memory(raw: &str) -> Option<(String, i64)> {
    let body = raw.trim().strip_prefix('[')?.strip_suffix(']')?;
    let parts: Vec<&str> = body.split(',').map(str::trim).collect();
    match parts.as_slice() {
        [base] => Some((base.to_ascii_lowercase(), 0)),
        [base, offset] => {
            let off = offset.strip_prefix('#').unwrap_or(offset).trim();
            Some((base.to_ascii_lowercase(), parse_aarch32_immediate(off)?))
        }
        _ => None,
    }
}

/// Parse a signed immediate in decimal or `0x` hex (with optional `-`).
fn parse_aarch32_immediate(raw: &str) -> Option<i64> {
    let s = raw.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return i64::from_str_radix(hex, 16).ok();
    }
    if let Some(hex) = s.strip_prefix("-0x").or_else(|| s.strip_prefix("-0X")) {
        return i64::from_str_radix(hex, 16).ok().map(|v| -v);
    }
    s.parse::<i64>().ok()
}
