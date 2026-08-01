//! `AArch32` per-mnemonic lifter handlers, extracted from `lift.rs`.
//! Methods on [`LiftCtx`]; reuses the `AArch64` 3-operand family and
//! shared infrastructure from the parent module.

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::register_layout;

use super::{
    BinOp, LiftCtx, aarch64_cond_suffix_to_predicate, is_aarch32_base_supported, nonzero_width,
    strip_aarch32_cond_suffix, width_mask,
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
            // Memory in offset form `[Rn]` / `[Rn, #imm]`. Register-
            // offset, writeback, and predicated forms (`ldreq`) are not
            // in this match and decline to `Unsupported` — sound, the
            // slice's confidence path widens rather than mis-lifting.
            "ldr" => self.lift_aarch32_ldr(insn),
            "str" => self.lift_aarch32_str(insn),
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

    fn lift_aarch32_ldr(&mut self, insn: &Instruction) {
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
        let Some(address) = aarch32_address_expr(mem, self.bits) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("ldr addressing mode not yet modelled: {}", mem.raw),
            });
            return;
        };
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

    fn lift_aarch32_str(&mut self, insn: &Instruction) {
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
        let Some(address) = aarch32_address_expr(mem, self.bits) else {
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

/// Parse an `AArch32` memory operand in the supported offset forms
/// (`[Rn]` / `[Rn, #imm]`) into a symbolic address `base ± offset` at
/// the pointer width. Returns `None` for register-offset, writeback,
/// shifted, or otherwise unrecognised shapes so the caller declines
/// soundly (the ARM `r0..r15` base is resolved through
/// [`register_layout`] under [`Arch::Arm`]).
fn aarch32_address_expr(mem: &Operand, ptr_bits: u8) -> Option<Expr> {
    let (base, offset) = parse_aarch32_memory(&mem.raw)?;
    let parent = register_layout(&base, Arch::Arm).map(|l| l.parent)?;
    let base_var = Expr::Var(Var::new(parent, ptr_bits));
    if offset == 0 {
        return Some(base_var);
    }
    let masked = u64::from_le_bytes(offset.to_le_bytes()) & width_mask(ptr_bits);
    Some(Expr::add(base_var, Expr::konst(masked, ptr_bits)))
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
