//! Lift a [`Slice`] of x86 / `x86_64` instructions into a list of
//! [`IrStmt`]s plus a symbolic [`Expr`] for the branch's condition.
//!
//! The lifter is intentionally narrow: it handles the mnemonic set
//! produced by the Phase 3 slicer (`mov`, `xor` (zero idiom),
//! `and / or / add / sub / imul`, `cmp`, `test`, `shl / sal / shr / sar`,
//! `lea`). For each instruction it emits one or more [`IrStmt::Assign`]
//! statements that capture the data-flow effect plus, when relevant, the
//! flag updates that the SMT backend reads through
//! [`BranchCandidate`].
//!
//! Flags are modelled as 1-bit [`r2smt_ir::Var`]s named `ZF`, `CF`,
//! `SF`, `OF`, `PF`. Flags we cannot model precisely yet (`OF`, `PF`,
//! `AF`) are set to [`Expr::Unknown`] so the SMT translator can treat
//! them as free symbolic inputs without producing wrong answers.
//!
//! Sub-register precision: register reads and writes consult
//! [`crate::registers::register_layout`] so that operands like `al`,
//! `ah`, `ax`, `eax` are modelled as bit-slices of the canonical
//! 64-bit parent (`rax`) via [`Expr::Extract`] / [`Expr::Concat`] /
//! [`Expr::ZeroExtend`]. The lifter still tracks the parent as the
//! single SSA-renamed variable; bit-precise rewrites live inside the
//! right-hand side of the assignment.

use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::collector::BranchCandidate;
use crate::condition::BranchCondition;
use crate::effect::stack_slot;
use crate::registers::register_layout;
use crate::slice::{Slice, SliceStatus};

mod aarch32;
mod aarch64;
mod merge;
mod x86;
use merge::lower_merge;

/// IR representation of a [`Slice`] plus the branch's symbolic
/// condition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiftedSlice {
    /// The branch the slice belongs to.
    pub branch: BranchCandidate,
    /// IR statements produced by lifting `slice.instructions`, in
    /// execution order.
    pub statements: Vec<IrStmt>,
    /// Symbolic condition the branch evaluates over the flags emitted
    /// by `statements`.
    pub condition: Expr,
    /// Status forwarded from the slice.
    pub status: SliceStatus,
    /// Forwarded from [`Slice::treat_truncation_as_inputs`]: when set,
    /// downstream stages treat a truncated slice as if it were
    /// complete, with the remaining roots becoming free symbolic
    /// inputs at the SSA layer.
    #[serde(default)]
    pub treat_truncation_as_inputs: bool,
    /// Architecture the slice was lifted under. Surfaced downstream
    /// (SSA, pretty-printer, report) so register-name resolution can
    /// pick the correct ISA table without re-deriving it.
    #[serde(default = "default_arch")]
    pub arch: Arch,
}

fn default_arch() -> Arch {
    Arch::X86_64
}

const FLAGS: &[&str] = &["ZF", "CF", "SF", "OF", "PF"];

/// Lift `slice` under `arch`. The lifter still only handles x86
/// mnemonics; passing a non-x86 arch results in every instruction
/// falling through to `IrStmt::Unsupported`. The arch flows into
/// `register_layout` lookups so register naming is unambiguous if a
/// future caller hand-builds non-x86 IR.
#[must_use]
pub fn lift_slice(slice: &Slice, arch: Arch) -> LiftedSlice {
    let mut ctx = LiftCtx::new(arch);
    // Lower any bounded simple-diamond Φ-merges *first*: the head
    // condition definitions and the resulting `Ite` assignments
    // execute before the join-block body, so they must precede it in
    // the statement stream. A merge that turns out not to be soundly
    // foldable at lift time emits nothing — the merged register then
    // surfaces as a free SSA input through its use in the join body,
    // which is exactly the sound free-input boundary (widen-only,
    // never a fabricated verdict).
    for merge in &slice.merges {
        lower_merge(&mut ctx, merge, arch);
    }
    for insn in &slice.instructions {
        ctx.lift_instruction(insn);
    }
    let condition = lift_branch_condition(&slice.branch, arch);
    debug!(
        target: "r2smt::lift",
        at = %slice.branch.address,
        statements = ctx.stmts.len(),
        "slice lifted"
    );
    LiftedSlice {
        branch: slice.branch.clone(),
        statements: ctx.stmts,
        condition,
        status: slice.status.clone(),
        treat_truncation_as_inputs: slice.treat_truncation_as_inputs,
        arch,
    }
}

/// Lift a branch's [`BranchCondition`] into an [`Expr`] over the flag
/// variables. The arch's pointer width drives the `CxZero` family.
#[must_use]
pub fn lift_branch_condition(candidate: &BranchCandidate, arch: Arch) -> Expr {
    let bits = arch.pointer_bits();
    let zf = Expr::flag("ZF");
    let cf = Expr::flag("CF");
    let sf = Expr::flag("SF");
    let of = Expr::flag("OF");
    let pf = Expr::flag("PF");
    let one = || Expr::konst(1, 1);
    let zero = || Expr::konst(0, 1);
    match candidate.condition {
        BranchCondition::Equal => Expr::eq(zf, one()),
        BranchCondition::NotEqual => Expr::eq(zf, zero()),
        BranchCondition::Above => Expr::bool_and(Expr::eq(cf, zero()), Expr::eq(zf, zero())),
        BranchCondition::AboveOrEqual => Expr::eq(cf, zero()),
        BranchCondition::Below => Expr::eq(cf, one()),
        BranchCondition::BelowOrEqual => Expr::bool_or(Expr::eq(cf, one()), Expr::eq(zf, one())),
        BranchCondition::Greater => Expr::bool_and(Expr::eq(zf, zero()), Expr::eq(sf, of)),
        BranchCondition::GreaterOrEqual => Expr::eq(sf, of),
        BranchCondition::Less => Expr::ne(sf, of),
        BranchCondition::LessOrEqual => Expr::bool_or(Expr::eq(zf, one()), Expr::ne(sf, of)),
        BranchCondition::Sign => Expr::eq(sf, one()),
        BranchCondition::NotSign => Expr::eq(sf, zero()),
        BranchCondition::Overflow => Expr::eq(of, one()),
        BranchCondition::NotOverflow => Expr::eq(of, zero()),
        BranchCondition::ParityEven => Expr::eq(pf, one()),
        BranchCondition::ParityOdd => Expr::eq(pf, zero()),
        BranchCondition::CxZero => {
            // Modelled at the program's pointer width; the actual
            // register width depends on the mnemonic (`jcxz` ≡ cx,
            // `jecxz` ≡ ecx, `jrcxz` ≡ rcx) but our canonical naming
            // collapses them.
            Expr::eq(Expr::var("rcx", bits), Expr::konst(0, bits))
        }
        BranchCondition::RegisterZero | BranchCondition::RegisterNotZero => {
            // `cbz Rn` / `cbnz Rn` — read the register the collector
            // parsed out of operand[0], canonicalise via the
            // arch-aware layout table, and emit `Rn == 0` / `Rn != 0`.
            let (var, vbits) = aarch64_branch_var(candidate, arch);
            let cmp = Expr::eq(var, Expr::konst(0, vbits));
            match candidate.condition {
                BranchCondition::RegisterZero => cmp,
                _ => Expr::bool_not(cmp),
            }
        }
        BranchCondition::BitZero | BranchCondition::BitNotZero => {
            // `tbz Rn, #bit` / `tbnz Rn, #bit` — extract a single bit
            // and compare against zero.
            let (var, vbits) = aarch64_branch_var(candidate, arch);
            match candidate.bit_index {
                Some(bit) if bit < vbits => {
                    let slice = Expr::extract(var, bit, bit);
                    let cmp = Expr::eq(slice, Expr::konst(0, 1));
                    match candidate.condition {
                        BranchCondition::BitZero => cmp,
                        _ => Expr::bool_not(cmp),
                    }
                }
                // Unparsed or out-of-range bit index: substituting a
                // concrete bit would fabricate a different predicate
                // (an unsound `AlwaysX`). Surface as a free symbolic
                // value instead — the solver can only widen it to
                // `BothPossible`, never fabricate a verdict.
                _ => Expr::Unknown(format!(
                    "tbz/tbnz bit-index unresolved for `{reg}`",
                    reg = candidate.compare_register.as_deref().unwrap_or("")
                )),
            }
        }
    }
}

/// Resolve a `cbz`/`cbnz`/`tbz`/`tbnz` candidate's compare register
/// against the arch's layout table. Falls back to a parent-width
/// `Unknown(name)` for unrecognised tokens so the SMT backend
/// produces a free input rather than silently dropping the operand.
fn aarch64_branch_var(candidate: &BranchCandidate, arch: Arch) -> (Expr, u8) {
    let raw = candidate.compare_register.as_deref().unwrap_or("");
    if let Some(layout) = register_layout(raw, arch) {
        // `xzr`/`wzr` always read 0. Mirrors `read_register` so
        // `cbz xzr` / `tbz xzr, #n` resolve precisely instead of
        // getting stuck on a free input named `xzr`.
        if layout.parent == "xzr" {
            return (Expr::konst(0, layout.width()), layout.width());
        }
        let parent_bits = arch.pointer_bits();
        if u16::from(layout.hi) < u16::from(parent_bits) {
            let parent = Expr::var(layout.parent, parent_bits);
            let width = layout.width();
            let var = if layout.lo == 0 && layout.hi + 1 == parent_bits {
                parent
            } else {
                Expr::extract(parent, layout.hi, layout.lo)
            };
            return (var, width);
        }
    }
    let bits = arch.pointer_bits();
    (Expr::Unknown(raw.to_string()), bits)
}

/// Lift a single instruction through **only** the per-mnemonic
/// handler dispatch, bypassing the P-code-first / ESIL-first ladder
/// that [`lift_slice`] runs.
///
/// This is the differential-lifting seam consumed by the
/// `r2smt-difflift` harness: it exposes the per-mnemonic lowering in
/// isolation so it can be cross-checked against the independent ESIL
/// ([`r2smt_esil::lift_esil`]) and P-code ([`r2smt_pcode::lift_pcode`])
/// lowerings of the same instruction. The production dispatch ladder
/// in [`lift_slice`] is deliberately **not** routed through this
/// function — its observable behaviour is unchanged.
#[must_use]
pub fn lift_per_mnemonic(insn: &Instruction, arch: Arch) -> Vec<IrStmt> {
    let mut ctx = LiftCtx::new(arch);
    // Closed structural dispatch over the supported ISAs (the
    // documented exhaustive-dispatch-table exception). Mirrors the
    // tail of `LiftCtx::lift_instruction` *without* the ESIL / P-code
    // short-circuits, which is the entire purpose of this seam.
    match arch {
        Arch::X86 | Arch::X86_64 => ctx.lift_instruction_x86(insn),
        Arch::Aarch64 => ctx.lift_instruction_aarch64(insn),
        Arch::Arm => ctx.lift_instruction_aarch32(insn),
        _ => ctx.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!(
                "at {addr} (arch {arch:?})",
                addr = insn.address,
                arch = arch
            ),
        }),
    }
    ctx.stmts
}

struct LiftCtx {
    stmts: Vec<IrStmt>,
    bits: u8,
    arch: Arch,
    temp_counter: u32,
}

impl LiftCtx {
    fn new(arch: Arch) -> Self {
        Self {
            stmts: Vec::new(),
            bits: arch.pointer_bits(),
            arch,
            temp_counter: 0,
        }
    }

    fn new_temp(&mut self, address: Address, width: u8) -> Var {
        let name = format!(
            "t_{addr:x}_{n}",
            addr = address.get(),
            n = self.temp_counter
        );
        self.temp_counter += 1;
        Var::new(name, width)
    }

    fn lift_instruction(&mut self, insn: &Instruction) {
        // ESIL-first path: when radare2 has attached an ESIL string
        // to the instruction and the mini stack machine can evaluate
        // it, splice the resulting IrStmts straight into the buffer.
        // This covers every opcode r2 knows how to disassemble
        // without writing a per-mnemonic handler. Failures (unknown
        // tokens, control-flow markers, …) fall through to the
        // arch-specific dispatcher below — which keeps Fase C's
        // bespoke handlers as overrides for mnemonics ESIL describes
        // imprecisely.
        // P-code-first path: when the r2ghidra adapter attached SLEIGH
        // P-code for this instruction (opt-in `--ir pcode|auto`),
        // prefer it — decompiler-grade IR with explicit varnodes and
        // flag derivation. Any [`r2smt_pcode::PcodeError`] (opcode
        // outside the sound subset, …) falls through to ESIL exactly
        // as before; a declined P-code lift never emits output, so
        // this is sound and the default (no `pcode` attached) path is
        // byte-identical to before.
        if let Some(pcode) = insn.pcode.as_deref()
            && let Ok(lift) = r2smt_pcode::lift_pcode(pcode, self.arch)
        {
            debug!(
                target: "r2smt::lift",
                at = %insn.address,
                stmts = lift.statements.len(),
                "pcode-hit"
            );
            self.stmts.extend(lift.statements);
            return;
        }

        if let Some(esil) = insn.esil.as_deref()
            && let Ok(lift) = r2smt_esil::lift_esil(esil, self.arch)
        {
            debug!(
                target: "r2smt::lift",
                at = %insn.address,
                stmts = lift.statements.len(),
                "esil-hit"
            );
            self.stmts.extend(lift.statements);
            return;
        }
        debug!(
            target: "r2smt::lift",
            at = %insn.address,
            "esil-miss"
        );
        match self.arch {
            Arch::X86 | Arch::X86_64 => self.lift_instruction_x86(insn),
            Arch::Aarch64 => self.lift_instruction_aarch64(insn),
            Arch::Arm => self.lift_instruction_aarch32(insn),
            _ => self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!(
                    "at {addr} (arch {arch:?})",
                    addr = insn.address,
                    arch = self.arch
                ),
            }),
        }
    }

    /// Width of the operand at its natural granularity (sub-register
    /// width, stack-slot width, or pointer width for immediates).
    fn operand_width(&self, op: &Operand) -> u8 {
        match op.kind {
            OperandKind::Register => {
                register_layout(&op.raw, self.arch).map_or(self.bits, |layout| layout.width())
            }
            OperandKind::Memory => stack_slot(op).map_or(self.bits, |(_, w)| w),
            _ => self.bits,
        }
    }

    /// Read a register operand, returning the matching parent slice
    /// (full register or [`Expr::Extract`] of the parent). Returns
    /// `None` if the operand is not a recognised GPR.
    fn read_register(&self, op: &Operand) -> Option<Expr> {
        let layout = register_layout(&op.raw, self.arch)?;
        // `AArch64` zero registers always read as 0 regardless of
        // which alias is named. Modelling this lets opaque-predicate
        // patterns like `mov x0, xzr; cbz x0, …` resolve to
        // `AlwaysTrue` instead of getting stuck on a free input.
        if layout.parent == "xzr" {
            return Some(Expr::konst(0, layout.width()));
        }
        let parent_bits = self.bits;
        if u16::from(layout.hi) >= u16::from(parent_bits) {
            // Caller is running at a pointer width that cannot fit the
            // register (`rax` in a 32-bit lift). Surface as Unsound at
            // the lifter level rather than fabricate a slice.
            return None;
        }
        let parent = Expr::var(layout.parent, parent_bits);
        if layout.lo == 0 && layout.hi + 1 == parent_bits {
            Some(parent)
        } else {
            Some(Expr::extract(parent, layout.hi, layout.lo))
        }
    }

    /// Read an operand and coerce the result to `width` bits.
    ///
    /// Immediates are constructed directly at `width` (masked to the
    /// requested width), so a small constant alongside a sub-register
    /// read does not need to round-trip through a wider type.
    /// Sub-register and stack-slot reads are produced at their natural
    /// width and zero-extended / truncated to match `width`.
    fn read_operand_at(&self, op: &Operand, width: u8) -> Expr {
        match op.kind {
            OperandKind::Register => match self.read_register(op) {
                Some(natural) => {
                    let nw = register_layout(&op.raw, self.arch)
                        .map_or(self.bits, |layout| layout.width());
                    coerce_to_width(natural, width, nw)
                }
                None => Expr::Unknown(op.raw.clone()),
            },
            OperandKind::Immediate => match parse_immediate(&op.raw) {
                Some(value) => Expr::konst(value & width_mask(width), width),
                None => Expr::Unknown(op.raw.clone()),
            },
            OperandKind::Memory => match stack_slot(op) {
                Some((slot, slot_width)) => {
                    let var = Expr::var(slot, slot_width);
                    coerce_to_width(var, width, slot_width)
                }
                None => Expr::Unknown(format!("mem {raw}", raw = op.raw)),
            },
            _ => Expr::Unknown(op.raw.clone()),
        }
    }

    /// Build the right-hand-side expression that, when assigned to the
    /// parent register, captures writing `value` (already at the
    /// destination's natural width) to the operand `op`.
    fn build_parent_write(&self, op: &Operand, value: Expr) -> Option<(Var, Expr)> {
        let layout = register_layout(&op.raw, self.arch)?;
        let parent_bits = self.bits;
        if u16::from(layout.hi) >= u16::from(parent_bits) {
            return None;
        }
        let parent_var = Var::new(layout.parent, parent_bits);
        let rhs = if layout.lo == 0 && layout.hi + 1 == parent_bits {
            // Full-width write: replaces the parent entirely.
            value
        } else if layout.zero_extends_parent_64 && parent_bits == 64 {
            // x86_64 32-bit write: zero-extends to 64 bits.
            Expr::zero_ext(value, parent_bits)
        } else {
            // Partial write: preserve surrounding bits.
            let parent_read = Expr::var(layout.parent, parent_bits);
            let mut acc = value;
            if layout.lo > 0 {
                let low_preserve = Expr::extract(parent_read.clone(), layout.lo - 1, 0);
                acc = Expr::concat(acc, low_preserve);
            }
            if layout.hi + 1 < parent_bits {
                let high_preserve = Expr::extract(parent_read, parent_bits - 1, layout.hi + 1);
                acc = Expr::concat(high_preserve, acc);
            }
            acc
        };
        Some((parent_var, rhs))
    }

    /// Compute the destination [`Var`] (parent register or stack-slot)
    /// the operand `op` writes to. Used by paths that build their own
    /// right-hand side (`lea`, the unsupported-destination fall-throughs).
    fn dst_var(&self, op: &Operand) -> Option<Var> {
        if let Some(layout) = register_layout(&op.raw, self.arch)
            && u16::from(layout.hi) < u16::from(self.bits)
        {
            return Some(Var::new(layout.parent, self.bits));
        }
        if let Some((slot, width)) = stack_slot(op) {
            return Some(Var::new(slot, width));
        }
        None
    }

    fn assign(&mut self, dst: Var, src: Expr) {
        self.stmts.push(IrStmt::Assign { dst, src });
    }

    fn set_flag(&mut self, name: &str, src: Expr) {
        self.assign(Var::new(name, 1), src);
    }

    fn clear_all_flags(&mut self) {
        for f in FLAGS {
            self.set_flag(f, Expr::Unknown(String::new()));
        }
    }

    fn update_logic_flags(&mut self, result: &Expr, width: u8) {
        // After `and / or / xor / test`:
        //   ZF = result == 0
        //   SF = msb(result)
        //   CF = 0, OF = 0
        //   PF = parity (unmodelled).
        self.set_flag("ZF", Expr::eq(result.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(result.clone(), Expr::konst(0, width)));
        self.set_flag("CF", Expr::konst(0, 1));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    /// Emit an assignment that writes `value` to register operand
    /// `dst_op`, accounting for sub-register semantics. Returns
    /// `false` if the operand is not a recognised register (caller
    /// should fall back).
    fn write_register_to(&mut self, dst_op: &Operand, value: Expr) -> bool {
        if let Some((dst_var, rhs)) = self.build_parent_write(dst_op, value) {
            self.assign(dst_var, rhs);
            return true;
        }
        false
    }

    /// Emit an assignment for any supported destination (register or
    /// stack slot). Memory destinations that resolve to a stack slot
    /// preserve the slot's natural width.
    fn write_dst(&mut self, dst_op: &Operand, value: Expr, dst_width: u8) -> bool {
        if dst_op.kind == OperandKind::Register {
            return self.write_register_to(dst_op, value);
        }
        if let Some((slot, width)) = stack_slot(dst_op) {
            let coerced = coerce_to_width(value, width, dst_width);
            self.assign(Var::new(slot, width), coerced);
            return true;
        }
        false
    }

    fn binop_width(&self, lhs: &Operand, rhs: &Operand) -> u8 {
        // For the compare/test family the operation width is the width
        // of the register/memory operand; `read_operand_at` then masks
        // the immediate down to it. An immediate's `operand_width` is
        // the pointer-width pseudo-size (`self.bits`), so taking the
        // max here would inflate `cmp al, 1` to 64 bits and corrupt the
        // sign-dependent flags (SF) for every sub-register compare.
        let lhs_imm = matches!(lhs.kind, OperandKind::Immediate);
        let rhs_imm = matches!(rhs.kind, OperandKind::Immediate);
        match (lhs_imm, rhs_imm) {
            (false, true) => self.operand_width(lhs),
            (true, false) => self.operand_width(rhs),
            _ => {
                let lw = self.operand_width(lhs);
                let rw = self.operand_width(rhs);
                if lw >= rw { lw } else { rw }
            }
        }
    }

    // ---------- AArch64 handlers ------------------------------------
    //
    // AArch64 instructions are 3-operand (`Rd, Rs1, Rs2`) where Rd is
    // write-only — unlike x86's 2-operand RMW shape. Reads / writes
    // route through the same `read_operand_at` / `write_register_to`
    // helpers as the x86 path; the arch-aware `register_layout`
    // resolves `x0` / `w0` correctly because `LiftCtx.arch` is
    // forwarded into every layout query.
    //
    // Flag polarity is x86-style: `cmp_aarch64` emits `CF = (a < b)
    // unsigned`. This is the opposite of ARM's architectural carry
    // (`C = (a >= b) unsigned`), but it lets the existing
    // `lift_branch_condition` stay arch-agnostic — AArch64 condition
    // codes are mapped to x86-equivalent `BranchCondition` variants
    // by `condition::classify`.
}

#[derive(Debug, Clone, Copy)]
enum BitwiseOp {
    And,
    Or,
}

#[derive(Debug, Clone, Copy)]
enum ShiftOp {
    Shl,
    Shr,
    Sar,
}

#[derive(Debug, Clone, Copy)]
enum ExtendKind {
    Zero,
    Sign,
}

/// Generic `AArch64` binary operator used by `lift_aarch64_arith3`.
///
/// `AArch64` instructions are uniformly 3-operand (`Rd, Rs1, Rs2`)
/// without the x86 RMW shape, so a single handler parameterised by
/// this enum covers the whole arithmetic / logical / shift family.
#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
}

/// Else-branch shape selected by an `AArch64` conditional-select
/// arithmetic instruction (`csinc` / `csinv` / `csneg` and their
/// `cinc` / `cinv` / `cneg` 3-operand aliases).
#[derive(Debug, Clone, Copy)]
enum CsArithOp {
    /// `Rm + 1` (`csinc`, `cinc`).
    Inc,
    /// `~Rm` (`csinv`, `cinv`).
    Inv,
    /// `-Rm` (`csneg`, `cneg`).
    Neg,
}

impl BinOp {
    fn apply(self, lhs: Expr, rhs: Expr) -> Expr {
        match self {
            Self::Add => Expr::add(lhs, rhs),
            Self::Sub => Expr::sub(lhs, rhs),
            Self::Mul => Expr::mul(lhs, rhs),
            Self::UDiv => Expr::udiv(lhs, rhs),
            Self::SDiv => Expr::sdiv(lhs, rhs),
            Self::And => Expr::bv_and(lhs, rhs),
            Self::Or => Expr::bv_or(lhs, rhs),
            Self::Xor => Expr::bv_xor(lhs, rhs),
            Self::Shl => Expr::shl(lhs, rhs),
            Self::Shr => Expr::lshr(lhs, rhs),
            Self::Sar => Expr::ashr(lhs, rhs),
        }
    }
}

/// Recognised `AArch64` / `AArch32` condition-code suffixes, longest
/// first so a greedy `ends_with` walk picks the proper boundary
/// (e.g. `hi` before `s` would split `cmphi` correctly).
const AARCH_COND_SUFFIXES: &[&str] = &[
    "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le",
    "al", "nv",
];

/// `AArch32` base mnemonics that the dispatcher knows how to lift.
/// Used by the predicated-execution wrapper to decide whether to
/// peel the cond suffix or fall through to `Unsupported`.
const AARCH32_BASE_MNEMONICS: &[&str] = &[
    "mov", "mvn", "add", "adds", "sub", "subs", "rsb", "rsbs", "and", "ands", "bic", "bics", "orr",
    "orrs", "eor", "eors", "mul", "muls", "udiv", "sdiv", "lsl", "lsls", "lsr", "lsrs", "asr",
    "asrs", "cmp", "cmn", "tst", "teq",
];

pub(crate) fn is_aarch32_base_supported(base: &str) -> bool {
    AARCH32_BASE_MNEMONICS.contains(&base)
}

/// If `mnem` ends with a recognised condition-code suffix and the
/// remaining prefix is non-empty, return `(base, cond_suffix)`.
/// Otherwise return `None` so the caller dispatches the mnem as-is.
pub(crate) fn strip_aarch32_cond_suffix(mnem: &str) -> Option<(&str, &str)> {
    for cond in AARCH_COND_SUFFIXES {
        let Some(base) = mnem.strip_suffix(cond) else {
            continue;
        };
        if base.is_empty() {
            // Pure conditional mnemonics like `eq` make no sense.
            continue;
        }
        return Some((base, cond));
    }
    None
}

/// Translate an `AArch64` condition suffix (`eq`, `ne`, `cs`, …)
/// into a 1-bit boolean `Expr` over the lifter's flag variables.
/// Mirrors the predicates [`lift_branch_condition`] emits for the
/// equivalent [`BranchCondition`] variants. Used by `csel` / `cset`
/// lifting and any future predicated-execution path.
fn aarch64_cond_suffix_to_predicate(raw: &str) -> Option<Expr> {
    let suffix = raw.trim().to_ascii_lowercase();
    let zf = || Expr::flag("ZF");
    let cf = || Expr::flag("CF");
    let sf = || Expr::flag("SF");
    let of = || Expr::flag("OF");
    let one = || Expr::konst(1, 1);
    let zero = || Expr::konst(0, 1);
    Some(match suffix.as_str() {
        "eq" => Expr::eq(zf(), one()),
        "ne" => Expr::eq(zf(), zero()),
        "cs" | "hs" => Expr::eq(cf(), zero()),
        "cc" | "lo" => Expr::eq(cf(), one()),
        "mi" => Expr::eq(sf(), one()),
        "pl" => Expr::eq(sf(), zero()),
        "vs" => Expr::eq(of(), one()),
        "vc" => Expr::eq(of(), zero()),
        "hi" => Expr::bool_and(Expr::eq(cf(), zero()), Expr::eq(zf(), zero())),
        "ls" => Expr::bool_or(Expr::eq(cf(), one()), Expr::eq(zf(), one())),
        "ge" => Expr::eq(sf(), of()),
        "lt" => Expr::ne(sf(), of()),
        "gt" => Expr::bool_and(Expr::eq(zf(), zero()), Expr::eq(sf(), of())),
        "le" => Expr::bool_or(Expr::eq(zf(), one()), Expr::ne(sf(), of())),
        _ => return None,
    })
}

fn nonzero_width(width: u8) -> Option<u8> {
    if width == 0 { None } else { Some(width) }
}

fn coerce_to_width(value: Expr, target: u8, source: u8) -> Expr {
    match source.cmp(&target) {
        std::cmp::Ordering::Equal => value,
        std::cmp::Ordering::Less => Expr::zero_ext(value, target),
        std::cmp::Ordering::Greater => Expr::extract(value, target - 1, 0),
    }
}

/// Mask covering the low `width` bits of a `u64`. `width == 64` keeps
/// every bit; smaller widths zero the upper bits.
const fn width_mask(width: u8) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

// `parse_immediate` parses signed ARM immediates (`#-1`, `#0x10`) but
// returns the bit-pattern as `u64` so it can flow straight into the IR's
// fixed-width bit-vector constants — the slicer never re-interprets the
// returned value as a signed integer, so the `i64 as u64` reinterpret on
// the negation path is intentional.
#[allow(clippy::cast_sign_loss)]
fn parse_immediate(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    // AArch64 / AArch32 assembly emits immediates with a leading `#`
    // (`#0x10`, `#-1`). Strip it so the rest of the parser only deals
    // with the numeric body.
    let trimmed = trimmed.strip_prefix('#').unwrap_or(trimmed).trim_start();
    let (negative, body) = if let Some(rest) = trimmed.strip_prefix('-') {
        (true, rest.trim_start())
    } else {
        (false, trimmed)
    };
    let value = if let Some(rest) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).ok()?
    } else {
        body.parse::<u64>().ok()?
    };
    if negative {
        Some(value.wrapping_neg())
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests;
