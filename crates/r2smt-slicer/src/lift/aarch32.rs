//! `AArch32` per-mnemonic lifter handlers, extracted from `lift.rs`.
//! Methods on [`LiftCtx`]; reuses the `AArch64` 3-operand family and
//! shared infrastructure from the parent module.

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::registers::register_layout;

pub(super) mod neon;

use super::{
    BinOp, FpArithOp, LiftCtx, MemAccess, PackedIntOp, PackedOp, VectorShape, Writeback,
    aarch64_cond_suffix_to_predicate, constant_delta, fp_lane_result, fp_propagating_max_min,
    fp_sort_bits_checked, is_aarch32_arith_short_form, is_aarch32_base_supported, nonzero_width,
    strip_aarch32_cond_suffix, strip_thumb_width_suffix, vector_shape, width_mask,
};

impl LiftCtx {
    pub(super) fn lift_instruction_aarch32(&mut self, insn: &Instruction) {
        // AArch32 instruction shapes mirror AArch64 (3-operand
        // arithmetic / 2-operand compare). The lifter reuses the
        // AArch64 handler family — register reads / writes flow
        // through `register_layout(name, self.arch)` which respects
        // `Arch::Arm` and produces `r0..r15` parents.
        // `AArch32` NEON is spelled with a typed mnemonic and bare
        // register operands, so the operand-shape collision is narrower
        // than on `AArch64` — but the indexed form (`vmov r0, d0[1]`)
        // reaches the integer arms by exactly the same route.
        //
        // `vpush` / `vpop` carry a VFP register list that reads as a
        // vector arrangement, so they must be dispatched before the
        // vector-shape gate that would otherwise decline them — they are
        // plain stack transfers, not an unmodelled vector shape.
        match insn.mnemonic.trim().to_ascii_lowercase().as_str() {
            "vpush" => {
                self.lift_aarch32_vpush(insn);
                return;
            }
            "vpop" => {
                self.lift_aarch32_vpop(insn);
                return;
            }
            _ => {}
        }
        if vector_shape(insn, Arch::Arm) == VectorShape::Declined {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!(
                    "unmodelled vector shape at {addr} (aarch32)",
                    addr = insn.address
                ),
            });
            return;
        }
        // Peel a Thumb-2 `.w` / `.n` encoding-width suffix first: it is
        // an assembler hint, so `add.w` dispatches as `add`. Done before
        // the cond peel so a wide predicated form (`addne.w`) composes.
        let mnem_full = insn.mnemonic.trim().to_ascii_lowercase();
        let mnem = strip_thumb_width_suffix(&mnem_full).to_string();
        // Thumb 2-operand narrow form: `add r0, r1` is `add r0, r0, r1`.
        // Duplicate the destination into the first-source slot so the
        // shared 3-operand handler (and its flag-ordering temp) applies
        // unchanged. The predicated re-entry sees the expanded operands,
        // so it does not re-normalize.
        let expanded;
        let insn = if insn.operands.len() == 2 && is_aarch32_arith_short_form(&mnem) {
            let mut clone = insn.clone();
            if let Some(dst) = clone.operands.first().cloned() {
                clone.operands.insert(1, dst);
            }
            expanded = clone;
            &expanded
        } else {
            insn
        };
        // Conditional execution suffix: `<base><cond>` such as `addeq`
        // or `subne`. Strip the recognised tail, look up the cond
        // predicate, and wrap every assignment the base handler emits
        // in `Ite(cond, new, old)` so flags and destination writes
        // become predicated. `al` (always) is the unmodified base;
        // `nv` (never) is reserved and treated as predicated with a
        // constant-false condition for soundness.
        //
        // Only peel when the full mnemonic is not itself a supported
        // base. Several flag-setting `s`-forms end in a cond spelling —
        // `lsls`/`lsrs`/`asrs` end in `ls`/`rs`... `ls`, `bics` in `cs`
        // — so peeling first would mis-lift `lsls` as a conditional
        // `lsl` and shadow its own exact match arm.
        if !is_aarch32_base_supported(mnem.as_str())
            && let Some((base, cond_suffix)) = strip_aarch32_cond_suffix(&mnem)
            && is_aarch32_base_supported(base)
            && let Some(cond_expr) = aarch64_cond_suffix_to_predicate(cond_suffix)
        {
            self.lift_aarch32_predicated(insn, base, &cond_expr);
            return;
        }
        self.lift_aarch32_by_mnemonic(insn, &mnem);
    }

    /// The per-mnemonic `AArch32` dispatch, reached after the vector-shape
    /// gate, the `.w` / `.n` and 2-operand normalisations, and the
    /// cond-suffix peel. Split out only to keep the entry point under the
    /// line limit; it is a pure dispatch table.
    fn lift_aarch32_by_mnemonic(&mut self, insn: &Instruction, mnem: &str) {
        match mnem {
            "mov" => self.lift_aarch64_mov(insn),
            "movs" => self.lift_aarch32_movs(insn),
            "vmrs" => self.lift_aarch32_vmrs(insn),
            "vldr" => self.lift_aarch32_vfp_mem(insn, true),
            "vstr" => self.lift_aarch32_vfp_mem(insn, false),
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
            "ldr" => self.lift_aarch32_load(insn, None, false),
            "ldrb" => self.lift_aarch32_load(insn, Some(8), false),
            "ldrh" => self.lift_aarch32_load(insn, Some(16), false),
            // The sign-extending forms differ from `ldrb`/`ldrh` only in
            // how the loaded bits reach the 32-bit register.
            "ldrsb" => self.lift_aarch32_load(insn, Some(8), true),
            "ldrsh" => self.lift_aarch32_load(insn, Some(16), true),
            // `ldrd`/`strd` name two registers before the memory
            // operand, so they carry their own operand shape.
            "ldrd" => self.lift_aarch32_load_pair(insn),
            "strd" => self.lift_aarch32_store_pair(insn),
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
            _ if let Some(mode) = aarch32_multiple_mode(mnem, true) => {
                self.lift_aarch32_ldm(insn, mode);
            }
            _ if let Some(mode) = aarch32_multiple_mode(mnem, false) => {
                self.lift_aarch32_stm(insn, mode);
            }
            // The NEON families the element-typed mnemonic dispatch
            // cannot resolve. Tried first because they constrain their
            // operands most tightly — the shape resolver checks operand
            // count, register class and geometry, where the packed arm
            // below asks only what the mnemonic spells.
            _ if let Some(access) = neon::structured::resolve(insn) => {
                self.lift_aarch32_structured(insn, &access);
            }
            _ if let Some(shape) = neon::resolve(insn) => self.lift_aarch32_neon(insn, shape),
            // NEON packed data processing. Recognised ahead of the VFP
            // arm because the two families share mnemonics: `vadd.f32`
            // is scalar when its destination is an `s` register and
            // packed when it is a `d` (two lanes) or a `q` (four).
            // `neon_packed_shape` answers `None` for the single-lane
            // case, so everything the scalar handler lifts today still
            // reaches it.
            _ if self.neon_packed_shape(insn).is_some() => {
                if let Some((op, lane)) = self.neon_packed_shape(insn) {
                    // A NEON write preserves the vector register above
                    // the destination's view — `d1` survives a write to
                    // `d0`, both being halves of `q0`.
                    self.lift_packed_vector(insn, op, lane, false);
                }
            }
            // VFP scalar floating point. Unlike AArch64, the lane
            // width is spelled in the mnemonic (`vadd.f32` /
            // `vadd.f64`), not in the operand, and every mnemonic is
            // `v`-prefixed so none of them can collide with the
            // integer handlers above.
            _ if vfp_scalar(mnem).is_some() => {
                if let Some((op, lane)) = vfp_scalar(mnem) {
                    self.lift_aarch32_vfp(insn, op, lane);
                }
            }
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
        let ones = Expr::konst(u128::from(width_mask(dst_width)), dst_width);
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
        let result = Expr::bv_xor(
            value,
            Expr::konst(u128::from(width_mask(dst_width)), dst_width),
        );
        if !self.write_register_to(dst, result) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (mvn)".into(),
            });
        }
    }

    /// `vmrs APSR_nzcv, FPSCR` — transfer the FP compare flags into the
    /// integer condition flags, the ARM analogue of x86 `fnstsw` +
    /// `sahf`. Our `vcmp` writes NZCV directly, so in this model the
    /// transfer is an identity: it emits a `Nop` rather than declining,
    /// which keeps it out of `InstructionKind::Other` so the slice walks
    /// past it to the `vcmp` that actually defined the flags. Any other
    /// `vmrs` form (`vmrs r0, FPSCR`, a GPR read of the status word) is
    /// not modelled and declines.
    fn lift_aarch32_vmrs(&mut self, insn: &Instruction) {
        if aarch32_vmrs_transfers_flags(insn) {
            self.stmts.push(IrStmt::Nop);
            return;
        }
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("at {addr} (vmrs)", addr = insn.address),
        });
    }

    /// `movs Rd, Op` — a move that also sets N/Z from the moved value.
    /// C and V are unchanged for the register form and non-rotated
    /// immediates (the modified-immediate carry is not modelled), so
    /// they are left untouched rather than fabricated. The value is
    /// stashed in a temp before the destination write so the flag terms
    /// read the moved value, not the post-write register (the
    /// flag-ordering invariant, load-bearing when `Rd` overlaps `Op`).
    fn lift_aarch32_movs(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if dst.kind != OperandKind::Register {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (movs)".into(),
            });
            return;
        }
        let Some(dst_width) = nonzero_width(self.operand_width(dst)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "zero-width destination (movs)".into(),
            });
            return;
        };
        let value = self.read_operand_at(src, dst_width);
        let tmp = self.new_temp(insn.address, dst_width);
        self.assign(tmp.clone(), value);
        let tmp_expr = Expr::Var(tmp);
        self.set_flag("ZF", Expr::eq(tmp_expr.clone(), Expr::konst(0, dst_width)));
        self.set_flag("SF", Expr::slt(tmp_expr.clone(), Expr::konst(0, dst_width)));
        if !self.write_register_to(dst, tmp_expr) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "non-register destination (movs)".into(),
            });
        }
    }

    /// `vldr` / `vstr` — scalar VFP load / store of a single `s` (32) or
    /// `d` (64) register through the byte-granular memory model, reusing
    /// the integer address resolver. The write merges (VFP preserves the
    /// rest of the register file), like every other `AArch32` vector
    /// write. A PC-relative literal pool declines: this model carries no
    /// live PC value.
    fn lift_aarch32_vfp_mem(&mut self, insn: &Instruction, is_load: bool) {
        let (Some(reg), Some(mem)) = (insn.operands.first(), insn.operands.get(1)) else {
            return;
        };
        if reg.kind != OperandKind::Register || mem.kind != OperandKind::Memory {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "vldr/vstr operand shape".into(),
            });
            return;
        }
        if aarch32_mem_base_is_pc(mem) {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "vldr/vstr pc-relative literal pool not modelled".into(),
            });
            return;
        }
        let Some(bits) = nonzero_width(self.operand_width(reg)) else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: "vldr/vstr zero-width register".into(),
            });
            return;
        };
        let Some(access) =
            aarch32_mem_access(mem, insn.operands.get(2..).unwrap_or_default(), self.bits)
        else {
            self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!("vldr/vstr addressing mode not modelled: {}", mem.raw),
            });
            return;
        };
        let MemAccess { address, writeback } = access;
        if is_load {
            let tmp = self.new_temp(insn.address, bits);
            self.stmts.push(IrStmt::LoadMem {
                dst: tmp.clone(),
                address,
                bits,
            });
            if !self.write_simd_lane(reg, Expr::Var(tmp), bits, 0) {
                self.stmts.push(IrStmt::Unsupported {
                    mnemonic: insn.mnemonic.clone(),
                    comment: "vldr destination not modelled".into(),
                });
                return;
            }
        } else {
            let Some(value) = self.read_simd_lane_bits(reg, bits, 0) else {
                self.stmts.push(IrStmt::Unsupported {
                    mnemonic: insn.mnemonic.clone(),
                    comment: "vstr source not modelled".into(),
                });
                return;
            };
            self.stmts.push(IrStmt::StoreMem {
                address,
                value,
                bits,
            });
        }
        self.emit_writeback(writeback);
    }

    /// Lift an `AArch32` load. `width_override` is `Some(8)` for
    /// `ldrb`/`ldrsb` and `Some(16)` for `ldrh`/`ldrsh`; `None` for the
    /// word-sized `ldr`. `signed` selects the sign-extending `ldrs*`
    /// family, which is the only difference between the two — the
    /// address, the width and the writeback are identical.
    fn lift_aarch32_load(&mut self, insn: &Instruction, width_override: Option<u16>, signed: bool) {
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
        let Some(access) =
            aarch32_mem_access(mem, insn.operands.get(2..).unwrap_or_default(), self.bits)
        else {
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
    fn aarch32_pair_access(&mut self, insn: &Instruction) -> Option<MemAccess> {
        let mem = insn.operands.get(2)?;
        if mem.kind != OperandKind::Memory {
            return None;
        }
        aarch32_mem_access(mem, insn.operands.get(3..).unwrap_or_default(), self.bits)
    }

    /// `ldrd Rt, Rt2, [mem]` — two consecutive words, `Rt` at the
    /// address and `Rt2` one word above it.
    fn lift_aarch32_load_pair(&mut self, insn: &Instruction) {
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
    fn lift_aarch32_store_pair(&mut self, insn: &Instruction) {
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
    fn lift_aarch32_store(&mut self, insn: &Instruction, width_override: Option<u16>) {
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
        let Some(access) =
            aarch32_mem_access(mem, insn.operands.get(2..).unwrap_or_default(), self.bits)
        else {
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
        self.emit_writeback(Some(Writeback::by_constant("sp", -4 * n, self.bits)));
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
        self.emit_writeback(Some(Writeback::by_constant("sp", 4 * n, self.bits)));
    }

    /// `vpush {regs}` — store each VFP register to a descending stack
    /// slot (lowest register at the lowest address) and decrement `sp`
    /// by the total byte count. `s` registers are 4 bytes, `d` are 8.
    fn lift_aarch32_vpush(&mut self, insn: &Instruction) {
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
    fn lift_aarch32_vpop(&mut self, insn: &Instruction) {
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
    fn lift_aarch32_ldm(&mut self, insn: &Instruction, mode: MultipleMode) {
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
    fn lift_aarch32_stm(&mut self, insn: &Instruction, mode: MultipleMode) {
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

/// Parse a VFP register list `{s0, s1}` / `{d8-d15}` into its member
/// operands (in list order) and the element width (32 for `s`, 64 for
/// `d`). Handles the dash range form and requires a single register
/// class. Brace lists render as `OperandKind::Unknown` under real
/// radare2, so the caller must not gate on operand kind.
pub(crate) fn parse_vfp_reglist(raw: &str) -> Option<(Vec<Operand>, u16)> {
    let body = raw.trim().strip_prefix('{')?.strip_suffix('}')?;
    let mut members: Vec<(char, u16)> = Vec::new();
    for part in body.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (class, first) = parse_vfp_reg(lo.trim())?;
                let (class_hi, last) = parse_vfp_reg(hi.trim())?;
                if class != class_hi || last < first {
                    return None;
                }
                for n in first..=last {
                    members.push((class, n));
                }
            }
            None => members.push(parse_vfp_reg(part)?),
        }
    }
    let class = members.first()?.0;
    if members.iter().any(|(c, _)| *c != class) {
        return None;
    }
    let width = if class == 's' { 32 } else { 64 };
    let operands = members
        .into_iter()
        .map(|(c, n)| Operand {
            raw: format!("{c}{n}"),
            kind: OperandKind::Register,
        })
        .collect();
    Some((operands, width))
}

/// Parse a single VFP register name into `(class, number)` — `s0..s31`
/// or `d0..d31`. `None` for anything else.
fn parse_vfp_reg(name: &str) -> Option<(char, u16)> {
    let name = name.trim().to_ascii_lowercase();
    let mut chars = name.chars();
    let class = chars.next()?;
    if class != 's' && class != 'd' {
        return None;
    }
    let number: u16 = chars.as_str().parse().ok()?;
    if number > 31 {
        return None;
    }
    Some((class, number))
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
fn aarch32_mem_access(mem: &Operand, post: &[Operand], ptr_bits: u16) -> Option<MemAccess> {
    if mem.kind != OperandKind::Memory {
        return None;
    }
    let raw = mem.raw.trim();
    if let Some(body) = raw.strip_suffix('!') {
        let (base, offset) = parse_aarch32_memory(body.trim())?;
        let parent = register_layout(&base, Arch::Arm).map(|l| l.parent)?;
        let delta = aarch32_offset_expr(&offset, ptr_bits)?;
        return Some(MemAccess {
            address: aarch32_addr_from_offset(parent, &offset, ptr_bits)?,
            writeback: Some(Writeback {
                base: parent.to_string(),
                delta,
            }),
        });
    }
    let (base, offset) = parse_aarch32_memory(raw)?;
    let parent = register_layout(&base, Arch::Arm).map(|l| l.parent)?;
    if !post.is_empty() {
        // A post-index delta is written outside the brackets, so the
        // bracketed part must be the bare base: `ldr r0, [r1], r2`.
        if offset != MemOffset::Immediate(0) {
            return None;
        }
        let delta = parse_aarch32_offset(post)?;
        return Some(MemAccess {
            address: Expr::Var(Var::new(parent, ptr_bits)),
            writeback: Some(Writeback {
                base: parent.to_string(),
                delta: aarch32_offset_expr(&delta, ptr_bits)?,
            }),
        });
    }
    Some(MemAccess {
        address: aarch32_addr_from_offset(parent, &offset, ptr_bits)?,
        writeback: None,
    })
}

/// Byte distance between the two words of a doubleword transfer, and
/// the stride of a register-list transfer.
const AARCH32_WORD_BYTES: i64 = 4;

/// Which way a multi-register transfer walks memory.
///
/// The lowest-numbered register always takes the lowest address; the
/// mode decides only where the run of addresses starts relative to the
/// base and which way the base moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MultipleMode {
    increment: bool,
    before: bool,
}

impl MultipleMode {
    /// `(first offset from the base, writeback delta)` for `count`
    /// registers.
    fn span(self, count: usize) -> (i64, i64) {
        let total = AARCH32_WORD_BYTES * i64::try_from(count).unwrap_or(0);
        if self.increment {
            let start = if self.before { AARCH32_WORD_BYTES } else { 0 };
            return (start, total);
        }
        let start = if self.before {
            -total
        } else {
            -total + AARCH32_WORD_BYTES
        };
        (start, -total)
    }
}

/// `Some(true)` for a load-multiple, `Some(false)` for a store-multiple,
/// `None` for anything else — in any addressing mode or stack spelling.
///
/// Every one of them shares a single def/use shape, so the effect table
/// needs only the direction; which way the addresses run is the
/// lifter's question.
pub(crate) fn aarch32_multiple_is_load(mnemonic: &str) -> Option<bool> {
    [true, false]
        .into_iter()
        .find(|&is_load| aarch32_multiple_mode(mnemonic, is_load).is_some())
}

/// The addressing mode a load/store-multiple mnemonic spells, or `None`
/// when it is not one.
///
/// The stack spellings are direction-relative, so the same suffix means
/// different things on a load and a store: `stmfd` (full descending
/// push) is `stmdb`, while `ldmfd` (the matching pop) is `ldmia`.
/// Resolving them without `is_load` would silently walk memory the
/// wrong way.
fn aarch32_multiple_mode(mnemonic: &str, is_load: bool) -> Option<MultipleMode> {
    let suffix = mnemonic.strip_prefix(if is_load { "ldm" } else { "stm" })?;
    let (increment, before) = match suffix {
        "" | "ia" => (true, false),
        "ib" => (true, true),
        "da" => (false, false),
        "db" => (false, true),
        "fd" => (is_load, !is_load),
        "ea" => (!is_load, is_load),
        "fa" => (!is_load, !is_load),
        "ed" => (is_load, is_load),
        _ => return None,
    };
    Some(MultipleMode { increment, before })
}

/// The offset part of an `AArch32` memory operand.
///
/// `AArch32` scales an index register inside the addressing mode
/// (`[r0, r1, lsl 2]`), which `AArch64` spells with an extend operator
/// and x86 with a scale field. Modelling it as an offset *expression*
/// rather than a constant is what lets one resolver answer all three
/// spellings.
#[derive(Debug, Clone, PartialEq, Eq)]
enum MemOffset {
    /// `[Rn]` (zero) or `[Rn, #imm]`.
    Immediate(i64),
    /// `[Rn, Rm]`, `[Rn, -Rm]`, `[Rn, Rm, lsl 2]`.
    Register {
        name: String,
        subtract: bool,
        shift: Option<(BinOp, u32)>,
    },
}

/// The shift applied to an index register, or `None` for a spelling
/// this model does not carry.
///
/// `ror` and `rrx` are declined rather than lowered: the IR has no
/// rotate, and composing one from two shifts plus an `Or` would be a
/// new lowering to validate for a form that is rare in an address.
fn aarch32_shift_op(name: &str) -> Option<BinOp> {
    match name {
        "lsl" => Some(BinOp::Shl),
        "lsr" => Some(BinOp::Shr),
        "asr" => Some(BinOp::Sar),
        _ => None,
    }
}

/// The offset as an expression to add to the base, at the pointer
/// width. A subtracted index becomes `0 - Rm` so the caller adds
/// unconditionally.
fn aarch32_offset_expr(offset: &MemOffset, ptr_bits: u16) -> Option<Expr> {
    match offset {
        MemOffset::Immediate(value) => Some(constant_delta(*value, ptr_bits)),
        MemOffset::Register {
            name,
            subtract,
            shift,
        } => {
            let parent = register_layout(name, Arch::Arm).map(|l| l.parent)?;
            let mut value = Expr::Var(Var::new(parent, ptr_bits));
            if let Some((op, amount)) = shift {
                value = op.apply(value, Expr::konst(u128::from(*amount), ptr_bits));
            }
            if *subtract {
                value = Expr::sub(Expr::konst(0, ptr_bits), value);
            }
            Some(value)
        }
    }
}

/// Build `base + offset` at the pointer width, keeping a zero
/// immediate as the bare base so the common shape is unchanged.
fn aarch32_addr_from_offset(parent: &str, offset: &MemOffset, ptr_bits: u16) -> Option<Expr> {
    let base_var = Expr::Var(Var::new(parent, ptr_bits));
    if *offset == MemOffset::Immediate(0) {
        return Some(base_var);
    }
    Some(Expr::add(base_var, aarch32_offset_expr(offset, ptr_bits)?))
}

/// Parse the operands that follow a bracketed memory operand into the
/// post-index delta: `, r2` or `, r2, lsl 2` or `, #4`.
fn parse_aarch32_offset(operands: &[Operand]) -> Option<MemOffset> {
    let parts: Vec<&str> = operands.iter().map(|o| o.raw.trim()).collect();
    parse_aarch32_offset_parts(&parts)
}

/// Parse the comma-separated tail of an addressing mode, shared by the
/// in-bracket and post-index spellings.
fn parse_aarch32_offset_parts(parts: &[&str]) -> Option<MemOffset> {
    let (index, shift) = match parts {
        [index] => (*index, None),
        [index, shift] => (*index, Some(*shift)),
        _ => return None,
    };
    let bare = index.strip_prefix('#').unwrap_or(index).trim();
    if let Some(value) = parse_aarch32_immediate(bare) {
        // An immediate takes no shift; `[r0, #4, lsl 2]` is not a form.
        return shift.is_none().then_some(MemOffset::Immediate(value));
    }
    let (subtract, name) = bare
        .strip_prefix('-')
        .map_or((false, bare), |rest| (true, rest.trim()));
    let name = name.to_ascii_lowercase();
    register_layout(&name, Arch::Arm)?;
    let shift = match shift {
        None => None,
        Some(spec) => {
            let mut words = spec.split_whitespace();
            let op = aarch32_shift_op(&words.next()?.to_ascii_lowercase())?;
            let amount = words.next()?;
            let amount = parse_aarch32_immediate(amount.strip_prefix('#').unwrap_or(amount))?;
            if words.next().is_some() {
                return None;
            }
            Some((op, u32::try_from(amount).ok()?))
        }
    };
    Some(MemOffset::Register {
        name,
        subtract,
        shift,
    })
}

/// Whether a memory operand's base register is the program counter — a
/// PC-relative literal pool this model cannot resolve (no live PC value).
fn aarch32_mem_base_is_pc(mem: &Operand) -> bool {
    let raw = mem.raw.trim();
    let body = raw.strip_suffix('!').unwrap_or(raw);
    parse_aarch32_memory(body).is_some_and(|(base, _)| matches!(base.as_str(), "pc" | "r15"))
}

/// Build `base ± offset` at the pointer width (base alone if zero).
pub(super) fn aarch32_addr_from(parent: &str, offset: i64, ptr_bits: u16) -> Expr {
    let base_var = Expr::Var(Var::new(parent, ptr_bits));
    if offset == 0 {
        return base_var;
    }
    let masked = u64::from_le_bytes(offset.to_le_bytes()) & width_mask(ptr_bits);
    Expr::add(base_var, Expr::konst(u128::from(masked), ptr_bits))
}

/// Split `[base{, offset}]` into `(base, offset)`, where the offset is
/// an immediate, an index register, or a shifted index register.
/// `None` for any shape outside that subset (writeback keeps the `!` or
/// a post-index delta outside the brackets).
fn parse_aarch32_memory(raw: &str) -> Option<(String, MemOffset)> {
    let body = raw.trim().strip_prefix('[')?.strip_suffix(']')?;
    let mut parts = body.split(',').map(str::trim);
    let base = parts.next()?.to_ascii_lowercase();
    let tail: Vec<&str> = parts.collect();
    if tail.is_empty() {
        return Some((base, MemOffset::Immediate(0)));
    }
    Some((base, parse_aarch32_offset_parts(&tail)?))
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

/// The signedness class an `AArch32` NEON data type spells.
///
/// It is part of the mnemonic rather than the operands, and for some of
/// the family it is part of the *operation*: two's-complement add,
/// subtract and multiply give the same bits either way, but `vmax` and
/// `vabs` do not, and the assembler will not accept an untyped `i` form
/// for those at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ElementKind {
    /// `s8` / `s16` / `s32` / `s64`.
    Signed,
    /// `u8` / `u16` / `u32` / `u64`.
    Unsigned,
    /// `i8` / `i16` / `i32` / `i64` — the sign-agnostic spelling.
    Untyped,
    /// `f16` / `f32` / `f64`.
    Float,
}

/// Element type named by an `AArch32` NEON data type (`i32`, `s16`,
/// `u8`, `f32`), as a signedness class and a width.
fn neon_element_type(ty: &str) -> Option<(ElementKind, u16)> {
    // Checked rather than `split_at`, which panics on a suffix shorter
    // than one character: `split_once('.')` hands this an empty string
    // for a mnemonic spelled `vadd.`.
    let (kind, width) = ty.split_at_checked(1)?;
    let kind = match kind {
        "i" => ElementKind::Untyped,
        "s" => ElementKind::Signed,
        "u" => ElementKind::Unsigned,
        "f" => ElementKind::Float,
        _ => return None,
    };
    let bits = match width {
        "8" if kind != ElementKind::Float => 8,
        "16" => 16,
        "32" => 32,
        "64" => 64,
        _ => return None,
    };
    Some((kind, bits))
}

/// The packed operation and element width an `AArch32` NEON
/// data-processing mnemonic names.
///
/// Two spellings exist. The arithmetic family carries an element type
/// (`vadd.i32`, `vmul.f32`); the bitwise family operates on raw bits and
/// the assembler makes the type optional, so disassemblers emit it bare
/// (`vand q0, q1, q2`). A bitwise operation needs no element width to be
/// exact — it is lane-independent — so the bare forms report the byte,
/// the smallest element the register file admits.
///
/// Recognising a mnemonic here is load-bearing beyond adding it: the
/// dispatcher tries this **before** the scalar VFP handler, and several
/// of these mnemonics — `vmax`, `vmin`, `vabs`, `vneg` — are spelled
/// identically in both families. A packed form this function does not
/// know therefore does not decline; it falls through to the scalar
/// handler, which computes lane 0 and leaves every other lane holding
/// whatever the destination held before.
pub(crate) fn neon_packed_op(mnemonic: &str) -> Option<(PackedOp, u16)> {
    const NEON_BITWISE_ELEMENT_BITS: u16 = 8;
    if let Some(op) = neon_bitwise_op(mnemonic) {
        return Some((PackedOp::Int(op), NEON_BITWISE_ELEMENT_BITS));
    }
    let (base, ty) = mnemonic.split_once('.')?;
    let (kind, lane_bits) = neon_element_type(ty)?;
    if let Some(op) = neon_accumulate_op(base, kind) {
        return Some((op, lane_bits));
    }
    if let Some(op) = neon_shift_op(base, kind) {
        return Some((op, lane_bits));
    }
    let op = match kind {
        ElementKind::Float => neon_float_op(base)?,
        _ => PackedOp::Int(neon_integer_op(base, kind)?),
    };
    Some((op, lane_bits))
}

/// The multiply-accumulate pair, which is the one family reading its
/// destination as an input and so is resolved above the split by
/// element class.
///
/// Every integer class is accepted for the same reason `vadd` accepts
/// them: a two's-complement multiply and add give the same bits signed
/// or unsigned, which is why the disassembler spells these `.i8` /
/// `.i16` / `.i32` in the first place.
fn neon_accumulate_op(base: &str, kind: ElementKind) -> Option<PackedOp> {
    let subtract = match base {
        "vmla" => false,
        "vmls" => true,
        _ => return None,
    };
    Some(PackedOp::Accumulate {
        float: kind == ElementKind::Float,
        subtract,
    })
}

/// The shift family, whose *element class* selects the encoding rather
/// than merely describing the lanes.
///
/// `VSHL` has two encodings and the type tells them apart: the untyped
/// `vshl.i32 q0, q1, 3` takes an immediate, and the signed or unsigned
/// `vshl.s32 q0, q1, q2` takes a per-lane amount from a register whose
/// sign chooses the direction. The right-shifting `vshr` and `vsra`
/// have only the immediate encoding, and only in the signed and
/// unsigned classes — arithmetic against logical is the whole
/// distinction, so there is nothing for an untyped form to mean.
fn neon_shift_op(base: &str, kind: ElementKind) -> Option<PackedOp> {
    let signed = match kind {
        ElementKind::Signed => true,
        ElementKind::Unsigned => false,
        ElementKind::Untyped => {
            return (base == "vshl").then_some(PackedOp::ShiftImmediate {
                left: true,
                signed: false,
                accumulate: false,
                rounding: false,
            });
        }
        ElementKind::Float => return None,
    };
    let right = |accumulate, rounding| PackedOp::ShiftImmediate {
        left: false,
        signed,
        accumulate,
        rounding,
    };
    Some(match base {
        "vshl" => PackedOp::ShiftRegister { signed },
        "vshr" => right(false, false),
        "vsra" => right(true, false),
        "vrshr" => right(false, true),
        "vrsra" => right(true, true),
        "vqshl" => PackedOp::SaturatingShiftLeftImmediate { signed },
        _ => return None,
    })
}

/// The float-typed packed forms.
///
/// `vabs` and `vneg` come back as *integer* lane operations on purpose:
/// both are sign-bit manipulations, exact at every value including the
/// NaNs, and routing them through a float sort would gain nothing and
/// decline at the widths that have none.
fn neon_float_op(base: &str) -> Option<PackedOp> {
    Some(match base {
        "vadd" => PackedOp::Fp(FpArithOp::Add),
        "vsub" => PackedOp::Fp(FpArithOp::Sub),
        "vmul" => PackedOp::Fp(FpArithOp::Mul),
        "vdiv" => PackedOp::Fp(FpArithOp::Div),
        "vmax" => PackedOp::Fp(FpArithOp::Max),
        "vmin" => PackedOp::Fp(FpArithOp::Min),
        "vabs" => PackedOp::Int(PackedIntOp::SignBit { negate: false }),
        "vneg" => PackedOp::Int(PackedIntOp::SignBit { negate: true }),
        _ => return None,
    })
}

/// The integer-typed packed forms.
///
/// The arithmetic family accepts every signedness class because
/// two's-complement add, subtract and multiply give the same bits either
/// way — which is exactly why the assembler offers the untyped `i`
/// spelling for them. `vmax` / `vmin` need the comparison's signedness
/// and `vabs` / `vneg` are meaningless on an unsigned element, so both
/// reject the classes their encodings do not have.
fn neon_integer_op(base: &str, kind: ElementKind) -> Option<PackedIntOp> {
    let signed = match kind {
        ElementKind::Signed => true,
        ElementKind::Unsigned => false,
        // Reached only by the arithmetic arms below, which ignore it.
        _ => return neon_untyped_op(base),
    };
    Some(match base {
        "vmax" => PackedIntOp::MinMax { max: true, signed },
        "vmin" => PackedIntOp::MinMax { max: false, signed },
        "vabs" if signed => PackedIntOp::Abs,
        "vneg" if signed => PackedIntOp::Neg,
        "vqadd" => PackedIntOp::Saturating {
            subtract: false,
            signed,
        },
        "vqsub" => PackedIntOp::Saturating {
            subtract: true,
            signed,
        },
        "vhadd" => PackedIntOp::Halving {
            subtract: false,
            signed,
            rounding: false,
        },
        "vhsub" => PackedIntOp::Halving {
            subtract: true,
            signed,
            rounding: false,
        },
        // There is no `vrhsub`: the architecture gives the rounding
        // form to the add alone.
        "vrhadd" => PackedIntOp::Halving {
            subtract: false,
            signed,
            rounding: true,
        },
        other => neon_untyped_op(other)?,
    })
}

/// The forms whose result does not depend on the element's signedness.
fn neon_untyped_op(base: &str) -> Option<PackedIntOp> {
    Some(PackedIntOp::Bin(match base {
        "vadd" => BinOp::Add,
        "vsub" => BinOp::Sub,
        "vmul" => BinOp::Mul,
        _ => return None,
    }))
}

/// The bitwise NEON mnemonics, which carry no element type.
///
/// `vmov` is deliberately absent: untyped `vmov` also spells the
/// general-register transfers (`vmov r0, s0`), which move between
/// register files rather than within the vector one.
fn neon_bitwise_op(mnemonic: &str) -> Option<PackedIntOp> {
    Some(match mnemonic {
        "vand" => PackedIntOp::Bin(BinOp::And),
        "vorr" => PackedIntOp::Bin(BinOp::Or),
        "veor" => PackedIntOp::Bin(BinOp::Xor),
        "vbic" => PackedIntOp::BitClear,
        "vmvn" => PackedIntOp::Not,
        _ => return None,
    })
}

impl LiftCtx {
    /// The packed operation and element width an `AArch32` instruction
    /// lowers to, or `None` when the scalar VFP handler owns it.
    ///
    /// The lane *count* comes from the destination register, because
    /// `AArch32` puts only the element type in the mnemonic: `q0` holds
    /// four `i32` elements, `d0` two. A destination holding exactly one
    /// element is the scalar VFP form — `vadd.f32 s0, s1, s2` and
    /// `vadd.f64 d0, d1, d2` — which the scalar handler already lifts,
    /// so this declines and leaves that path byte-identical.
    ///
    /// `vadd.f32 d0, d1, d2` is the case the distinction exists for: it
    /// holds *two* single-precision elements, and the scalar handler
    /// would compute only the low one while leaving the high one at
    /// whatever it held before.
    fn neon_packed_shape(&self, insn: &Instruction) -> Option<(PackedOp, u16)> {
        let mnem = insn.mnemonic.trim().to_ascii_lowercase();
        let (op, lane_bits) = neon_packed_op(&mnem)?;
        if insn.operands.len() != op.operand_count() || !neon_last_operand_fits(insn, op) {
            return None;
        }
        let view = self.simd_view_bits(insn.operands.first()?)?;
        (Self::packed_lane_count(view, lane_bits)? > 1).then_some((op, lane_bits))
    }
}

/// Whether `insn` is one of the `AArch32` NEON forms [`neon::resolve`]
/// models.
///
/// The effect table and the lifter both go through this, so they cannot
/// disagree about which instructions the slicer may retain.
pub(crate) fn is_aarch32_neon_instruction(insn: &Instruction) -> bool {
    neon::resolve(insn).is_some()
}

/// Whether `insn` is the `vmrs APSR_nzcv, FPSCR` flag-transfer form —
/// the destination names the application status register's condition
/// flags and the source is `FPSCR`. The GPR-destination form
/// (`vmrs r0, FPSCR`) is not this and is not modelled.
pub(crate) fn aarch32_vmrs_transfers_flags(insn: &Instruction) -> bool {
    if !insn.mnemonic.trim().eq_ignore_ascii_case("vmrs") {
        return false;
    }
    let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
        return false;
    };
    let dst = dst.raw.trim().to_ascii_lowercase();
    let src = src.raw.trim().to_ascii_lowercase();
    dst.starts_with("apsr") && src == "fpscr"
}

/// Whether `insn` is a NEON form that writes both of its named
/// registers — `vzip` / `vuzp` / `vtrn`.
///
/// The effect table needs this separately from
/// [`is_aarch32_neon_instruction`]: a second destination recorded only
/// as a use would let the slicer drop whatever defined it, leaving a
/// later read bound to a stale value.
pub(crate) fn aarch32_neon_writes_operand_pair(insn: &Instruction) -> bool {
    neon::writes_operand_pair(insn)
}

/// How the slicer should read a structured access's operands, or `None`
/// when `insn` is not one this lifter models.
pub(crate) fn aarch32_structured_effect(
    insn: &Instruction,
) -> Option<crate::lift::StructuredEffect> {
    neon::structured::resolve(insn).map(|access| access.effect())
}

/// Whether `insn` carries a packed NEON form in an operand shape the
/// lifter accepts.
///
/// Instruction-taking rather than mnemonic-taking because the shift
/// family's two encodings differ in operand *kind* rather than in
/// spelling, and the effect table has to answer this exactly as the
/// lifter does. It deliberately does **not** ask how many lanes the
/// destination holds: a single-lane destination means the scalar
/// handler owns the instruction, which is a different answer from "this
/// is not an instruction".
pub(crate) fn is_aarch32_packed_instruction(insn: &Instruction) -> bool {
    let mnem = insn.mnemonic.trim().to_ascii_lowercase();
    let Some((op, _)) = neon_packed_op(&mnem) else {
        return false;
    };
    insn.operands.len() == op.operand_count() && neon_last_operand_fits(insn, op)
}

/// Whether the instruction's final operand is the kind its resolved
/// shape reads.
///
/// Only the shift family makes this question interesting, and there it
/// is load-bearing rather than defensive: the immediate and register
/// encodings of `vshl` are told apart by the element class alone, so a
/// mnemonic whose class says "immediate" beside a register operand is
/// not an instruction, and lifting it would read a vector register as a
/// shift count.
fn neon_last_operand_fits(insn: &Instruction, op: PackedOp) -> bool {
    let Some(last) = insn.operands.last() else {
        return false;
    };
    match op {
        PackedOp::ShiftImmediate { .. } | PackedOp::SaturatingShiftLeftImmediate { .. } => {
            last.kind == OperandKind::Immediate
        }
        PackedOp::ShiftRegister { .. } => last.kind == OperandKind::Register,
        _ => true,
    }
}

/// What a VFP scalar mnemonic does, and at what precision.
#[derive(Clone, Copy)]
pub(crate) enum VfpOp {
    /// Binary arithmetic on two source registers.
    Arith(FpArithOp),
    /// `vcmp` / `vcmpe` — compare into FPSCR.
    Compare,
    /// `vmov` — move a bit pattern without interpreting it.
    Move,
    /// `vneg` / `vabs` — sign-bit manipulation.
    Sign { negate: bool },
    /// `vsqrt`.
    Sqrt,
}

/// Parse `v<op>.f<width>` — the `AArch32` spelling that carries the
/// precision in the mnemonic. Integer-typed vector forms (`vadd.i32`)
/// are not scalar floating point and decline here.
pub(crate) fn vfp_scalar(mnemonic: &str) -> Option<(VfpOp, u16)> {
    let (base, ty) = mnemonic.split_once('.')?;
    let lane = match ty {
        "f32" => 32,
        "f64" => 64,
        "f16" => 16,
        _ => return None,
    };
    let op = match base {
        "vadd" => VfpOp::Arith(FpArithOp::Add),
        "vsub" => VfpOp::Arith(FpArithOp::Sub),
        "vmul" => VfpOp::Arith(FpArithOp::Mul),
        "vdiv" => VfpOp::Arith(FpArithOp::Div),
        "vmax" => VfpOp::Arith(FpArithOp::Max),
        "vmin" => VfpOp::Arith(FpArithOp::Min),
        "vcmp" | "vcmpe" => VfpOp::Compare,
        "vmov" => VfpOp::Move,
        "vneg" => VfpOp::Sign { negate: true },
        "vabs" => VfpOp::Sign { negate: false },
        "vsqrt" => VfpOp::Sqrt,
        _ => return None,
    };
    Some((op, lane))
}

impl LiftCtx {
    /// The VFP scalar family. Operand shapes mirror `AArch64`, so the
    /// lane helpers are shared; only the flag write differs, because
    /// `AArch32` compares land in FPSCR and are copied to the
    /// condition flags by a subsequent `vmrs`.
    fn lift_aarch32_vfp(&mut self, insn: &Instruction, op: VfpOp, lane: u16) {
        let Some(dst) = insn.operands.first().cloned() else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        let result = match op {
            VfpOp::Arith(arith) => self.vfp_arith_value(insn, arith, lane),
            VfpOp::Sqrt => self.vfp_sqrt_value(insn, lane),
            VfpOp::Sign { negate } => self.vfp_sign_value(insn, lane, negate),
            VfpOp::Move => insn
                .operands
                .get(1)
                .cloned()
                .and_then(|src| self.read_simd_lane_bits(&src, lane, 0)),
            VfpOp::Compare => {
                self.lift_aarch32_vcmp(insn, lane);
                return;
            }
        };
        let Some(value) = result else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        // `AArch32` VFP writes the addressed slice and preserves the
        // rest of the register file — the opposite of `AArch64`, and
        // the reason `write_simd_lane` has to honour the view's offset.
        if !self.write_simd_lane(&dst, value, lane, 0) {
            self.push_aarch32_vfp_unsupported(insn);
        }
    }

    fn vfp_arith_value(&mut self, insn: &Instruction, arith: FpArithOp, lane: u16) -> Option<Expr> {
        let lhs = insn.operands.get(1)?.clone();
        let rhs = insn.operands.get(2)?.clone();
        let a = self.read_simd_lane_bits(&lhs, lane, 0)?;
        let b = self.read_simd_lane_bits(&rhs, lane, 0)?;
        // `vmax` / `vmin` are `FPMax` / `FPMin`, which propagate NaN and
        // combine the signs of a zero tie. [`fp_lane_result`] is Intel's
        // `MAXPS` and does neither.
        match arith {
            FpArithOp::Max | FpArithOp::Min => {
                fp_propagating_max_min(a, b, lane, matches!(arith, FpArithOp::Max), false)
            }
            FpArithOp::Add | FpArithOp::Sub | FpArithOp::Mul | FpArithOp::Div => {
                fp_lane_result(arith, a, b, lane)
            }
        }
    }

    fn vfp_sqrt_value(&mut self, insn: &Instruction, lane: u16) -> Option<Expr> {
        let src = insn.operands.get(1)?.clone();
        let value = self.read_simd_lane_fp(&src, lane, 0)?;
        Some(Expr::fp_to_ieee_bv(Expr::fsqrt(
            value,
            r2smt_ir::expr::RoundingMode::NearestTiesEven,
        )))
    }

    fn vfp_sign_value(&mut self, insn: &Instruction, lane: u16, negate: bool) -> Option<Expr> {
        let src = insn.operands.get(1)?.clone();
        let bits = self.read_simd_lane_bits(&src, lane, 0)?;
        let sign = super::aarch64::sign_bit_mask(lane)?;
        Some(if negate {
            Expr::bv_xor(bits, sign)
        } else {
            Expr::bv_and(bits, Expr::bv_xor(sign, super::simd::all_ones(lane)))
        })
    }

    /// `vcmp` / `vcmpe` — same flag mapping as the `AArch64` compare:
    /// ordered equality in ZF, ordered less-than in SF, the unordered
    /// predicate in OF so `lt` covers less-than and unordered exactly
    /// as the architecture defines after a floating-point compare.
    fn lift_aarch32_vcmp(&mut self, insn: &Instruction, lane: u16) {
        let (Some(lhs), Some(rhs)) = (
            insn.operands.first().cloned(),
            insn.operands.get(1).cloned(),
        ) else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        let Some(a) = self.read_simd_lane_fp(&lhs, lane, 0) else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        let b = if self.is_simd_register(&rhs) {
            let Some(value) = self.read_simd_lane_fp(&rhs, lane, 0) else {
                self.push_aarch32_vfp_unsupported(insn);
                return;
            };
            value
        } else {
            let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
                self.push_aarch32_vfp_unsupported(insn);
                return;
            };
            Expr::bv_to_fp(Expr::konst(0, lane), ebits, sbits)
        };
        let unordered = Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone()));
        self.set_flag("ZF", Expr::feq(a.clone(), b.clone()));
        self.set_flag("SF", Expr::flt(a.clone(), b.clone()));
        self.set_flag("CF", Expr::flt(a, b));
        self.set_flag("OF", unordered.clone());
        self.set_flag("PF", unordered);
    }

    fn push_aarch32_vfp_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable VFP operand at {addr}", addr = insn.address),
        });
    }
}
