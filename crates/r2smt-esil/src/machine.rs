//! ESIL stack machine.
//!
//! Consumes an ESIL token stream (`crate::parse::tokenize`) and emits
//! `Vec<r2smt_ir::IrStmt>` ready to feed the slicer's SSA → SMT
//! pipeline. The machine is intentionally a strict subset:
//!
//! - **Supported**: arithmetic (`+ - * & | ^ / %`), shifts (`<< >>`),
//!   compares (`== != < <= > >=`), assignment (`=`), compound
//!   assignment (`+= -= … >>=`), memory load / store (`[N]`, `=[N]`),
//!   the named flags `$z` / `$c` / `$s` / `$o` / `$p`, parametric
//!   carry/borrow tokens (`$cN` / `$bN`), and predicated blocks
//!   `cond,?{,...,}` (mapped to per-statement `Ite` wraps).
//! - **Rejected**: predicated blocks that contain memory loads or
//!   stores (the IR has no conditional memory effect), `GOTO`,
//!   `BREAK`, parametric overflow digits (`$0..$15`, radare2-specific
//!   NZCV-bit indices whose semantics vary by build), and any
//!   operator the tokenizer surfaced as `EsilToken::Unknown`.
//!
//! Callers are expected to fall back to per-mnemonic lifter when the
//! machine returns `Err(EsilError)`. Soundness is preserved: a failed
//! ESIL pass never partially populates the slicer's statement buffer
//! — the machine emits into a private vector and only hands it back
//! on success.

use r2smt_common::Arch;
use r2smt_common::registers::{is_simd_parent, is_zero_parent, register_layout};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;

use crate::flags;
use crate::parse::{EsilToken, tokenize};

/// Why an ESIL parse / evaluation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EsilError {
    /// The tokenizer surfaced an `Unknown` token. Carries the
    /// original text for diagnostics.
    UnknownToken(String),
    /// Operator popped from an empty / under-filled stack.
    StackUnderflow(&'static str),
    /// The token stream contained `?{` / `}` / a `GOTO`-style flow
    /// marker the MVP does not support.
    UnsupportedControlFlow,
    /// `=` (or compound `=`) targeted a non-register operand.
    InvalidAssignmentTarget,
    /// Parametric flag (`$cN`, `$bN`, `$0`) — out of MVP scope.
    UnsupportedFlag(String),
}

/// Result of a successful ESIL lift.
///
/// Mirrors the shape of the per-mnemonic lifter's `Vec<IrStmt>`
/// output so callers can splice it into their own statement list
/// directly.
#[derive(Debug, Clone)]
pub struct EsilLift {
    /// Statements produced by the machine, in execution order.
    pub statements: Vec<IrStmt>,
}

/// Lift the supplied ESIL string under `arch` (pointer width is read
/// off the architecture for default register widths).
///
/// # Errors
///
/// Returns [`EsilError`] when the token stream contains unsupported
/// constructs (see [`EsilError`]). Callers should treat any error
/// as "fall back to the per-mnemonic handler".
pub fn lift_esil(esil: &str, arch: Arch) -> Result<EsilLift, EsilError> {
    let tokens = tokenize(esil);
    let mut machine = Machine::new(arch);
    for token in tokens {
        machine.step(token)?;
    }
    // An unclosed `?{` would otherwise commit unwrapped statements
    // as if they were unconditional. Reject so the slicer falls back
    // to the per-mnemonic handler.
    if !machine.block_stack.is_empty() {
        return Err(EsilError::UnsupportedControlFlow);
    }
    Ok(EsilLift {
        statements: machine.statements,
    })
}

struct Machine {
    /// Default bit width when the lifter cannot recover a more
    /// precise size. Set to the architecture pointer width — `Var`s
    /// for unknown registers default to this, matching the slicer's
    /// canonical register layout.
    default_bits: u16,
    /// The architecture whose register names this ESIL describes.
    /// Load-bearing: several names collide across ISAs (`sp`, `lr`,
    /// `fp`, `r8`–`r15`) and answering with the wrong family's width is
    /// a wrong *value*, not a lost one.
    arch: Arch,
    stack: Vec<StackValue>,
    statements: Vec<IrStmt>,
    /// Last arithmetic / logical result. ESIL's `$z`, `$s`, `$cN`,
    /// `$bN`, … flag tokens derive their value from this — they
    /// describe "the flag the latest math operation would have set"
    /// rather than the content of a register named ZF/SF/...
    flag_ctx: Option<FlagCtx>,
    /// Active predicated-block frames. A `?{` token pushes a frame
    /// holding the condition and the index where the block body
    /// started; the matching `}` token pops the frame and wraps every
    /// `IrStmt::Assign` emitted between `stmt_start` and the close
    /// in `Ite(cond, new_src, Var(dst))`. Memory loads / stores inside
    /// a block flip `contains_mem_op` so the close errors out — the
    /// IR has no conditional memory effect we can model soundly.
    block_stack: Vec<BlockFrame>,
    /// A seeded flag context named but not yet emitted. See
    /// [`Machine::seed_flag_ctx`].
    pending: Option<PendingCtx>,
}

/// The two halves of a flag context, each deferred until a flag reads
/// it. Both are spliced in at `at`, which is the position **before** the
/// destination write — see [`Machine::seed_flag_ctx`] for why that is
/// load-bearing rather than tidy.
#[derive(Debug, Clone)]
struct PendingCtx {
    at: usize,
    /// The destination's value before the write. Note it snapshots the
    /// *slice* the write targets, not the whole parent, so a narrow
    /// write's flags stay narrow.
    old: Option<(Var, Expr)>,
    /// The value written.
    cur: Option<(Var, Expr)>,
}

/// The flag context radare2 keeps: the value a destination held before
/// the seeding operation and the value it holds after, at that
/// destination's width.
///
/// Measured, not inferred — see the module doc of [`crate::flags`]. Two
/// things about it replaced an entire `ArithKind` enum. The context is
/// seeded by **writes and compares**, never by bare arithmetic; and every
/// flag is a function of this pair alone, so whether the seeding
/// operation added or subtracted is a distinction radare2 does not make.
#[derive(Debug, Clone)]
pub(crate) struct FlagCtx {
    /// The destination's value before the operation.
    pub(crate) old: Expr,
    /// Its value after.
    pub(crate) cur: Expr,
    /// The destination's width, which is what `$z` masks to.
    pub(crate) bits: u16,
}

/// One active predicated `?{ ... }` block. Pushed on `?{`, popped on
/// `}` — the close handler wraps every `Assign` emitted within the
/// block in an `Ite` that gates on `cond`.
#[derive(Debug, Clone)]
struct BlockFrame {
    /// 1-bit guard expression (popped from the stack at `?{`).
    cond: Expr,
    /// Index into `Machine::statements` where the block body started.
    stmt_start: usize,
    /// Sticky bit: any `LoadMem` / `StoreMem` emitted inside the
    /// block flips this. The block close then rejects the lift —
    /// the IR has no conditional memory model.
    contains_mem_op: bool,
}

#[derive(Debug, Clone)]
enum StackValue {
    /// A register / variable reference. Kept distinct from
    /// `Expression` so the `=` operator can recover the destination
    /// without re-parsing the expression.
    Register(RegRef),
    /// A computed value.
    Expression { expr: Expr, bits: u16 },
}

impl StackValue {
    fn bits(&self) -> u16 {
        match self {
            StackValue::Register(reg) => reg.bits(),
            StackValue::Expression { bits, .. } => *bits,
        }
    }

    fn into_expr(self) -> Expr {
        match self {
            StackValue::Register(reg) => reg.read(),
            StackValue::Expression { expr, .. } => expr,
        }
    }

    /// The width this operand carries *architecturally*, as opposed to
    /// the width it happens to be spelled at.
    ///
    /// A register read and a memory read carry one; a bare constant does
    /// not, because radare2 spells every immediate as a `ut64` whatever
    /// the operand size is — `cmp dword [rbx], 0xffffffff` comes back as
    /// `18446744073709551615,rbx,[4],==`. Letting that spelling decide
    /// the operation width is what made `$z` compare a 32-bit value
    /// against a 64-bit `-1` and answer *never equal*.
    fn architectural_bits(&self) -> Option<u16> {
        match self {
            StackValue::Register(reg) => Some(reg.bits()),
            StackValue::Expression {
                expr: Expr::Const { .. },
                ..
            } => None,
            StackValue::Expression { bits, .. } => Some(*bits),
        }
    }
}

/// A register operand: the variable a read and a write actually touch,
/// plus the bit range of it this name addresses.
///
/// The indirection exists because `eax` is not a variable — it is bits
/// 31..0 of `rax`. Spelling it as its own `Var("eax", 32)`, which this
/// machine did until 2026-08-06, makes it a *different* data-flow node
/// from the `Extract(Var("rax", 64), 31, 0)` the per-mnemonic handlers
/// read, so a slice that mixes the two lifters — which the ladder in
/// `r2smt_slicer::lift` produces every time one instruction declines and
/// the next does not — silently loses the flow between them.
#[derive(Debug, Clone)]
struct RegRef {
    /// The variable read and written: the architectural parent when
    /// [`register_layout`] places the name, the name itself when it
    /// does not.
    var: Var,
    /// Inclusive bit range of `var` this name addresses.
    lo: u16,
    hi: u16,
    /// Whether a write through this name zero-extends the whole parent
    /// (the `x86_64` / `AArch64` 32-bit-alias rule) rather than
    /// preserving the bits around the slice.
    zero_extends: bool,
    /// Whether this names the ISA's hardwired zero register, which reads
    /// as the constant zero and discards its writes.
    zero: bool,
    /// Whether the pipeline stores the complement of what radare2 puts in
    /// this register, so every read and every write flips a bit.
    invert: bool,
}

impl RegRef {
    /// The degenerate case: the name *is* the variable.
    fn whole(var: Var) -> Self {
        let hi = var.bits.saturating_sub(1);
        Self {
            var,
            lo: 0,
            hi,
            zero_extends: false,
            zero: false,
            invert: false,
        }
    }

    fn bits(&self) -> u16 {
        self.hi.saturating_sub(self.lo).saturating_add(1)
    }

    fn covers_whole_var(&self) -> bool {
        self.lo == 0 && self.hi.saturating_add(1) == self.var.bits
    }

    fn read(&self) -> Expr {
        if self.zero {
            return Expr::konst(0, self.bits());
        }
        let parent = Expr::Var(self.var.clone());
        let value = if self.covers_whole_var() {
            parent
        } else {
            Expr::extract(parent, self.hi, self.lo)
        };
        if self.invert {
            return Expr::bv_xor(value, Expr::konst(1, self.bits()));
        }
        value
    }

    /// The whole-variable expression a write of `value` — already at
    /// [`RegRef::bits`] — leaves behind.
    fn write(&self, value: Expr) -> Expr {
        let value = if self.invert {
            Expr::bv_xor(value, Expr::konst(1, self.bits()))
        } else {
            value
        };
        if self.covers_whole_var() {
            return value;
        }
        if self.zero_extends {
            return Expr::zero_ext(value, self.var.bits);
        }
        let mut result = value;
        if self.lo > 0 {
            result = Expr::concat(
                result,
                Expr::extract(Expr::Var(self.var.clone()), self.lo.saturating_sub(1), 0),
            );
        }
        if self.hi.saturating_add(1) < self.var.bits {
            result = Expr::concat(
                Expr::extract(
                    Expr::Var(self.var.clone()),
                    self.var.bits.saturating_sub(1),
                    self.hi.saturating_add(1),
                ),
                result,
            );
        }
        result
    }
}

impl Machine {
    fn new(arch: Arch) -> Self {
        Self {
            default_bits: arch.pointer_bits(),
            arch,
            stack: Vec::new(),
            statements: Vec::new(),
            flag_ctx: None,
            block_stack: Vec::new(),
            pending: None,
        }
    }

    fn step(&mut self, token: EsilToken) -> Result<(), EsilError> {
        match token {
            EsilToken::Integer(value) => {
                self.push_const(value, self.default_bits);
                Ok(())
            }
            EsilToken::Register(name) => {
                // Normalise the lowercase flag register names ESIL
                // uses (`zf`, `cf`, …) to the canonical uppercase form
                // (`ZF`, `CF`, …) the per-mnemonic lifter, slicer,
                // and branch-condition predicate share. Everything
                // else passes through unchanged.
                //
                // `nf` and `vf` are ARM's spelling of the same two
                // flags this pipeline calls `SF` and `OF`, and both
                // ARM ISAs write all four (verified against r2 6.1.8:
                // `cmp` yields `…,$z,zf,:=,31,$s,nf,:=,32,$b,!,cf,:=,
                // 31,$o,vf,:=`). Without them every ESIL-lifted ARM
                // instruction left `SF` and `OF` undefined while every
                // signed condition — `lt`, `ge`, `gt`, `le`, `mi`,
                // `vs` — reads exactly those two, so the predicate got
                // a free input. Sound, but blind. The polarity matches
                // directly: `SF` is `msb(result)`, which is ARM's `N`,
                // and `OF` is signed overflow, which is its `V`.
                let canonical = match name.as_str() {
                    "zf" => "ZF",
                    "cf" => "CF",
                    "sf" | "nf" => "SF",
                    "of" | "vf" => "OF",
                    "pf" => "PF",
                    _ => name.as_str(),
                }
                .to_string();
                let mut reg = self.register_ref(&canonical);
                // …and `CF` is where it does *not* match directly. This
                // pipeline deliberately stores the **complement** of
                // ARM's `C`, so that one `lift_branch_condition` serves
                // both ISAs (`cs`/`hs` lowers to `CF == 0`), and the
                // convention binds every producer. radare2 writes the
                // architectural `C` — `cmp` emits `32,$b,!,cf,:=` and
                // a64 `ands`/`tst` emit `0,cf,:=` — so this machine has
                // to flip it on the way in and back on the way out.
                //
                // Without the flip a `b.hs` after an ESIL-lifted `cmp`
                // resolves to the arm the machine does not take: an
                // inverted branch, not a lost one. It is why the
                // `--esil-flags` gate excludes ARM, and it is exactly
                // what the gate has to stop being true of.
                reg.invert = canonical == "CF" && matches!(self.arch, Arch::Aarch64 | Arch::Arm);
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }
            EsilToken::Flag(suffix) => self.apply_flag(&suffix),
            EsilToken::Binary(op) => self.apply_binary(op),
            EsilToken::Compare => self.apply_compare(),
            EsilToken::NegAssign => self.apply_neg_assign(),
            EsilToken::Unary(op) => self.apply_unary(op),
            EsilToken::Assign => self.apply_assign(None, true),
            EsilToken::AssignNoFlags => self.apply_assign(None, false),
            EsilToken::CompoundAssign(op) => self.apply_assign(Some(op), true),
            EsilToken::Load(size) => self.apply_load(size),
            EsilToken::Store(size) => self.apply_store(size),
            EsilToken::BlockOpen => self.open_block(),
            EsilToken::BlockClose => self.close_block(),
            EsilToken::Unknown(text) => Err(EsilError::UnknownToken(text)),
        }
    }

    fn open_block(&mut self) -> Result<(), EsilError> {
        let cond_value = self.pop("?{ condition")?;
        let cond_1bit = cast_to_1bit(cond_value);
        self.block_stack.push(BlockFrame {
            cond: cond_1bit,
            stmt_start: self.statements.len(),
            contains_mem_op: false,
        });
        Ok(())
    }

    fn close_block(&mut self) -> Result<(), EsilError> {
        let frame = self
            .block_stack
            .pop()
            .ok_or(EsilError::UnsupportedControlFlow)?;
        if frame.contains_mem_op {
            return Err(EsilError::UnsupportedControlFlow);
        }
        // Wrap every Assign emitted between `?{` and `}` in
        // `Ite(cond, new_src, Var(dst))` so the SSA pass downstream
        // turns it into "previous version of dst" on the cond==0 path
        // — same recipe used by
        // `r2smt_slicer::lift::lift_aarch32_predicated`.
        for stmt in self.statements.iter_mut().skip(frame.stmt_start) {
            if let IrStmt::Assign { dst, src } = stmt {
                let old_value = Expr::Var(dst.clone());
                let placeholder = Expr::unknown();
                let new_src = std::mem::replace(src, placeholder);
                *src = Expr::Ite {
                    cond: Box::new(frame.cond.clone()),
                    then_expr: Box::new(new_src),
                    else_expr: Box::new(old_value),
                };
            }
        }
        Ok(())
    }

    fn push_const(&mut self, value: u64, bits: u16) {
        self.stack.push(StackValue::Expression {
            expr: Expr::konst(u128::from(value), bits),
            bits,
        });
    }

    fn apply_binary(&mut self, op: &'static str) -> Result<(), EsilError> {
        // ESIL is postfix and `a,b,OP` means `b OP a`: the *second*
        // token is the left-hand side, so it is the one on top of the
        // stack and therefore the one popped first.
        let lhs = self.pop("binary lhs")?;
        let rhs = self.pop("binary rhs")?;
        let bits = lhs.bits().max(rhs.bits());
        let result_bits = match op {
            "==" | "<" | "<=" | ">" | ">=" => 1,
            _ => bits,
        };
        let expr = match op {
            "+" => Expr::add(widen(lhs, bits), widen(rhs, bits)),
            "-" => Expr::sub(widen(lhs, bits), widen(rhs, bits)),
            "*" => Expr::mul(widen(lhs, bits), widen(rhs, bits)),
            "/" => Expr::udiv(widen(lhs, bits), widen(rhs, bits)),
            "%" => Expr::urem(widen(lhs, bits), widen(rhs, bits)),
            "&" => Expr::bv_and(widen(lhs, bits), widen(rhs, bits)),
            "|" => Expr::bv_or(widen(lhs, bits), widen(rhs, bits)),
            "^" => Expr::bv_xor(widen(lhs, bits), widen(rhs, bits)),
            "<<" => Expr::shl(widen(lhs, bits), widen(rhs, bits)),
            ">>" => Expr::lshr(widen(lhs, bits), widen(rhs, bits)),
            "==" => Expr::eq(widen(lhs, bits), widen(rhs, bits)),
            // Signed, measured against r2 6.1.8 and not assumed:
            // `ae 1,0x8000000000000000,<` answers 1, so the machine reads
            // those bits as `INT64_MIN` and not as the largest unsigned
            // value. Read unsigned, every comparison with the high bit
            // set flips — half the number line, silently. The narrower
            // operand is *sign*-extended for the same reason: a signed
            // compare of a zero-extended value would strip its sign.
            "<" => Expr::slt(widen_signed(lhs, bits), widen_signed(rhs, bits)),
            "<=" => Expr::sle(widen_signed(lhs, bits), widen_signed(rhs, bits)),
            ">" => Expr::slt(widen_signed(rhs, bits), widen_signed(lhs, bits)),
            ">=" => Expr::sle(widen_signed(rhs, bits), widen_signed(lhs, bits)),
            _ => return Err(EsilError::UnknownToken(op.to_string())),
        };
        // Deliberately seeds nothing. `ae 0x81,1,-,7,$s` leaves the
        // difference on the stack and still answers `$s(7) = 0`, and
        // `ae 5,3,+,$z` answers the *unseeded* default — bare arithmetic
        // does not touch the flag context. The model used to seed here
        // and nowhere else, which is the exact inverse of what radare2
        // does; the seeding sites are `apply_assign` and `apply_compare`.
        self.stack.push(StackValue::Expression {
            expr,
            bits: result_bits,
        });
        Ok(())
    }

    /// Evaluate one `$<flag>` token against the current context.
    ///
    /// The split is by **measured stack effect**: `$z` and `$p` pop
    /// nothing, while `$s` / `$c` / `$b` / `$o` pop their bit index —
    /// `ae 5,0x80,rax,=,7,$s` leaves `[5, 1]`, taking exactly one. The
    /// index must be a literal, since a mask cannot be built from a
    /// symbolic width.
    ///
    /// With no context there is no sound answer, and this declines
    /// rather than handing back a free `ZF`. radare2's context survives
    /// across instructions and ours cannot, so a free variable would let
    /// a branch predicate silently take an input; declining sends the
    /// instruction to a per-mnemonic handler that has contracts.
    fn apply_flag(&mut self, suffix: &str) -> Result<(), EsilError> {
        let unsupported = || EsilError::UnsupportedFlag(suffix.to_string());
        let ctx = self.flag_ctx.clone().ok_or_else(unsupported)?;
        // Every flag is a function of `cur`, so the snapshot of it goes
        // in ahead of the destination write for all of them.
        self.materialise_cur();
        let expr = match suffix {
            "z" => flags::zero_flag(&ctx),
            "p" => flags::parity_flag(&ctx).ok_or_else(unsupported)?,
            "s" | "c" | "b" | "o" => {
                let index = self.pop_flag_index()?;
                // Only the carry family reads `old`; the sign bit is a
                // function of the result alone.
                if matches!(suffix, "c" | "b" | "o") {
                    self.materialise_old();
                }
                match suffix {
                    "s" => flags::sign_flag(&ctx, index),
                    "c" => flags::carry_flag(&ctx, index),
                    "b" => flags::borrow_flag(&ctx, index),
                    _ => flags::overflow_flag(&ctx, index),
                }
                .ok_or_else(unsupported)?
            }
            _ => return Err(unsupported()),
        };
        self.stack.push(StackValue::Expression { expr, bits: 1 });
        Ok(())
    }

    /// Pop the bit index a parametric flag token consumes, requiring a
    /// literal.
    fn pop_flag_index(&mut self) -> Result<u16, EsilError> {
        match self.pop("flag bit index")? {
            StackValue::Expression {
                expr: Expr::Const { value, .. },
                ..
            } => u16::try_from(value).map_err(|_| EsilError::UnsupportedFlag(value.to_string())),
            _ => Err(EsilError::UnsupportedFlag("non-literal bit index".into())),
        }
    }
    fn apply_unary(&mut self, op: &'static str) -> Result<(), EsilError> {
        let operand = self.pop("unary operand")?;
        let bits = operand.bits();
        let expr = operand.into_expr();
        let result = match op {
            // Logical NOT in ESIL: 1 if value == 0, else 0.
            "!" => Expr::Ite {
                cond: Box::new(Expr::eq(expr, Expr::konst(0, bits))),
                then_expr: Box::new(Expr::konst(1, 1)),
                else_expr: Box::new(Expr::konst(0, 1)),
            },
            _ => return Err(EsilError::UnknownToken(op.to_string())),
        };
        self.stack.push(StackValue::Expression {
            expr: result,
            bits: 1,
        });
        Ok(())
    }

    /// `seeds_flags` is what separates `=` from `:=`. radare2's own help
    /// spells the difference — "assign updating internal flags" versus
    /// "assign without updating internal flags" — and a probe confirms
    /// it: after a `==`, a following `:=` leaves `$z` reading the
    /// compare's context while `=` overwrites it with the assigned
    /// value. Every flag write inside a flag-setting instruction has to
    /// be `:=`, or it would clobber the context the next flag token
    /// still needs to read.
    fn apply_assign(
        &mut self,
        compound: Option<&'static str>,
        seeds_flags: bool,
    ) -> Result<(), EsilError> {
        // ESIL is postfix: for `value,target,=` the stack at this
        // point is `[value, target]`, so the *target* is popped first
        // and the *value* second. Same convention for the compound
        // forms (`value,target,+=`).
        let target = self.pop("assign target")?;
        let value = self.pop("assign value")?;
        let StackValue::Register(target_reg) = target else {
            return Err(EsilError::InvalidAssignmentTarget);
        };
        let dst_bits = target_reg.bits();
        let value_expr = widen(value, dst_bits);
        let final_expr = match compound {
            None => value_expr,
            Some(op) => {
                let lhs = target_reg.read();
                match op {
                    "+" => Expr::add(lhs, value_expr),
                    "-" => Expr::sub(lhs, value_expr),
                    "*" => Expr::mul(lhs, value_expr),
                    "&" => Expr::bv_and(lhs, value_expr),
                    "|" => Expr::bv_or(lhs, value_expr),
                    "^" => Expr::bv_xor(lhs, value_expr),
                    "<<" => Expr::shl(lhs, value_expr),
                    ">>" => Expr::lshr(lhs, value_expr),
                    _ => return Err(EsilError::UnknownToken(op.to_string())),
                }
            }
        };
        // The flag context is the *slice*'s before / after pair, not the
        // parent's: `$z` masks to the width of the register written, so
        // `0x100,al,=,$z` is 1 where `0x100,rax,=,$z` is 0. Seeding with
        // the parent-width write would answer the second for both.
        if seeds_flags {
            self.seed_flag_ctx(&target_reg, &final_expr);
        }
        // A write to the zero register is discarded, and the flags it
        // seeds are not: `subs xzr, x0, x1` *is* `cmp`, so dropping the
        // seed would lose the comparison and dropping the discard would
        // invent a variable the machine does not have.
        if !target_reg.zero {
            self.statements.push(IrStmt::Assign {
                dst: target_reg.var.clone(),
                src: target_reg.write(final_expr),
            });
        }
        Ok(())
    }

    /// Record the flag context a write leaves behind.
    ///
    /// **Both halves are captured into temps emitted before the
    /// destination write**, and neither is optional. `ssa_convert`
    /// renames reads by statement *position*, and every flag statement
    /// is emitted after the write, so an inlined expression there has
    /// its references to the destination rewritten to the version the
    /// write just created. For `sub rsp, 0x20` that made `$z` read
    /// `((rsp - 32) - 32) == 0` — the destination subtracted twice.
    ///
    /// This is the flag-ordering invariant the per-mnemonic handlers
    /// have carried since Fase C, arriving here. It was found by the
    /// differential harness and not by a test, because each output is
    /// individually correct: the rebinding only shows when the flag and
    /// the destination are compared *together*.
    ///
    /// Both temps are named now and emitted only if a flag actually
    /// reads them, so a lift with no flag token stays byte-identical.
    fn seed_flag_ctx(&mut self, target: &RegRef, value: &Expr) {
        let bits = target.bits();
        let at = self.statements.len();
        let old_temp = Var::new(self.fresh_tmp_name("old"), bits);
        let cur_temp = Var::new(self.fresh_tmp_name("cur"), bits);
        self.pending = Some(PendingCtx {
            at,
            old: Some((old_temp.clone(), target.read())),
            cur: Some((cur_temp.clone(), value.clone())),
        });
        self.flag_ctx = Some(FlagCtx {
            old: Expr::Var(old_temp),
            cur: Expr::Var(cur_temp),
            bits,
        });
    }

    /// Emit the deferred `cur` snapshot — every flag reads it.
    fn materialise_cur(&mut self) {
        let Some(pending) = self.pending.as_mut() else {
            return;
        };
        let at = pending.at;
        let Some((temp, src)) = pending.cur.take() else {
            return;
        };
        self.insert_snapshot(at, temp, src);
    }

    /// Emit the deferred `old` snapshot — only `$c`, `$b` and `$o` read it.
    fn materialise_old(&mut self) {
        let Some(pending) = self.pending.as_mut() else {
            return;
        };
        let at = pending.at;
        let Some((temp, src)) = pending.old.take() else {
            return;
        };
        self.insert_snapshot(at, temp, src);
    }

    /// Splice a snapshot in ahead of the write it precedes, keeping the
    /// open predicated-block frames pointing at the same statements.
    fn insert_snapshot(&mut self, at: usize, temp: Var, src: Expr) {
        self.statements
            .insert(at, IrStmt::Assign { dst: temp, src });
        for frame in &mut self.block_stack {
            if frame.stmt_start > at {
                frame.stmt_start += 1;
            }
        }
    }

    /// `==` — pop both operands, seed the context with the subtraction,
    /// and push **nothing**.
    ///
    /// `ae 5,3,==` leaves the stack empty and `ae 5,5,==,$z` answers 1,
    /// so the comparison is a pure seeding operation. Modelling it as a
    /// binary operator that pushes a boolean left a residue the next
    /// token would consume as an operand — a wrong value, not a decline.
    ///
    /// The context's width is the **destination's**, falling back to the
    /// source's and only then to the spelled widths. `cmp` compares at
    /// its destination's width and sign-extends the source into it, and
    /// radare2 agrees: `esil_cmp` takes `lastsz` from whichever operand
    /// resolves to a register, preferring the first popped. Taking
    /// `max(lhs, rhs)` instead let a 64-bit immediate widen a 32-bit
    /// comparison, so `cmp ecx, 0xffffffff` asked whether a zero-extended
    /// `ecx` equalled `2^64 - 1` and answered *never* — an unsatisfiable
    /// `ZF` where the architecture sets it for `ecx == 0xffffffff`, and a
    /// following `jz` resolving to a branch the machine does take.
    fn apply_compare(&mut self) -> Result<(), EsilError> {
        let lhs = self.pop("compare lhs")?;
        let rhs = self.pop("compare rhs")?;
        let bits = lhs
            .architectural_bits()
            .or_else(|| rhs.architectural_bits())
            .unwrap_or_else(|| lhs.bits().max(rhs.bits()));
        let lhs_e = widen(lhs, bits);
        let rhs_e = widen(rhs, bits);
        self.flag_ctx = Some(FlagCtx {
            old: lhs_e.clone(),
            cur: Expr::sub(lhs_e, rhs_e),
            bits,
        });
        Ok(())
    }

    /// `!=` — logical NOT, written back to the popped register.
    ///
    /// radare2's help calls this "negate all bits" and that is wrong:
    /// `ar rax=0x0f; ae rax,!=` leaves `0`, not `0xfffffffffffffff0`.
    /// Measured across 0, 2, 0x0f and all-ones, it is
    /// `dst = (dst == 0) ? 1 : 0`, stored rather than pushed, and it
    /// does not seed. The operator was withdrawn from the model in
    /// `f45b3cdd` rather than fixed, because two tests pinned the wrong
    /// reading; this is the measured one.
    fn apply_neg_assign(&mut self) -> Result<(), EsilError> {
        let target = self.pop("negate target")?;
        let StackValue::Register(target_reg) = target else {
            return Err(EsilError::InvalidAssignmentTarget);
        };
        let bits = target_reg.bits();
        let is_zero = Expr::eq(target_reg.read(), Expr::konst(0, bits));
        let src = Expr::Ite {
            cond: Box::new(is_zero),
            then_expr: Box::new(Expr::konst(1, bits)),
            else_expr: Box::new(Expr::konst(0, bits)),
        };
        self.statements.push(IrStmt::Assign {
            dst: target_reg.var.clone(),
            src: target_reg.write(src),
        });
        Ok(())
    }

    fn apply_load(&mut self, size: u8) -> Result<(), EsilError> {
        let address = self.pop("load address")?;
        let bits = u16::try_from(usize::from(size) * 8).unwrap_or(self.default_bits);
        let dst = Var::new(self.fresh_tmp_name("ld"), bits);
        self.statements.push(IrStmt::LoadMem {
            dst: dst.clone(),
            address: address.into_expr(),
            bits,
        });
        self.mark_active_block_mem_op();
        self.stack.push(StackValue::Register(RegRef::whole(dst)));
        Ok(())
    }

    fn apply_store(&mut self, size: u8) -> Result<(), EsilError> {
        // ESIL store `value,address,=[N]`: stack is [value, address]
        // when the operator runs. Pop the *address* first, then the
        // value being stored.
        let address = self.pop("store address")?;
        let value = self.pop("store value")?;
        let bits = u16::try_from(usize::from(size) * 8).unwrap_or(self.default_bits);
        let value_expr = widen(value, bits);
        self.statements.push(IrStmt::StoreMem {
            address: address.into_expr(),
            value: value_expr,
            bits,
        });
        self.mark_active_block_mem_op();
        Ok(())
    }

    /// Mark the innermost active `?{` frame (if any) as containing a
    /// memory op. Block close will abort the lift on `contains_mem_op`
    /// because the IR has no conditional-memory semantics.
    fn mark_active_block_mem_op(&mut self) {
        if let Some(top) = self.block_stack.last_mut() {
            top.contains_mem_op = true;
        }
    }

    fn pop(&mut self, ctx: &'static str) -> Result<StackValue, EsilError> {
        self.stack.pop().ok_or(EsilError::StackUnderflow(ctx))
    }

    fn fresh_tmp_name(&self, prefix: &str) -> String {
        format!("__esil_{prefix}_{idx}", idx = self.statements.len())
    }

    /// Place `name` against its architectural parent.
    ///
    /// Two shapes decline to the historical narrow spelling, and both
    /// have to: a **SIMD** name, whose parent is 128 bits wide where
    /// this machine sizes variables at the pointer width — two `Var`s of
    /// one name and different widths do not collide loudly, since SSA
    /// versions by name and the encoder keeps the first width it sees —
    /// and a slice reaching **past the pointer width**, which is `rax`
    /// read under a 32-bit `Arch`. Same two guards, for the same
    /// reasons, as `r2smt_slicer::lift::LiftCtx::read_register`.
    fn register_ref(&self, name: &str) -> RegRef {
        let parent_bits = self.default_bits;
        if let Some(layout) = register_layout(name, self.arch)
            && !is_simd_parent(layout.parent, self.arch)
            && layout.hi < parent_bits
        {
            return RegRef {
                var: Var::new(layout.parent, parent_bits),
                lo: layout.lo,
                hi: layout.hi,
                zero_extends: layout.zero_extends_parent_64 && parent_bits == 64,
                zero: is_zero_parent(layout.parent, self.arch),
                invert: false,
            };
        }
        let bits = register_width(name, self.arch).unwrap_or(parent_bits);
        RegRef::whole(Var::new(name, bits))
    }
}

/// Width lookup for common register / flag names emitted by radare2's
/// ESIL strings. Returns `None` for names the table does not know;
/// callers fall back to `default_bits` (architecture pointer width).
///
/// The table is intentionally a sub-set of the full register layout
/// that lives in `r2smt-slicer` — duplicated here to avoid an inverse
/// `r2smt-slicer → r2smt-esil` dependency. The downstream slicer's
/// canonical-name resolution still owns the parent-register mapping;
/// here we only care about width so the lifter does not zero-extend
/// flag bits to the pointer width by accident.
fn register_width(name: &str, arch: Arch) -> Option<u16> {
    let lower = name.to_ascii_lowercase();
    // Flags are one bit on every ISA, and both spellings reach here —
    // `zf` from radare2 and `ZF` from this crate's canonicalisation.
    if matches!(
        lower.as_str(),
        "zf" | "cf" | "sf" | "of" | "pf" | "af" | "df" | "if" | "tf"
    ) {
        return Some(1);
    }
    match arch {
        Arch::X86 | Arch::X86_64 => x86_register_width(&lower),
        Arch::Aarch64 => aarch64_register_width(&lower),
        Arch::Arm => aarch32_register_width(&lower),
        _ => None,
    }
}

fn x86_register_width(lower: &str) -> Option<u16> {
    match lower {
        "al" | "ah" | "bl" | "bh" | "cl" | "ch" | "dl" | "dh" | "sil" | "dil" | "bpl" | "spl"
        | "r8b" | "r9b" | "r10b" | "r11b" | "r12b" | "r13b" | "r14b" | "r15b" => Some(8),
        "ax" | "bx" | "cx" | "dx" | "si" | "di" | "bp" | "sp" | "r8w" | "r9w" | "r10w" | "r11w"
        | "r12w" | "r13w" | "r14w" | "r15w" => Some(16),
        "eax" | "ebx" | "ecx" | "edx" | "esi" | "edi" | "ebp" | "esp" | "r8d" | "r9d" | "r10d"
        | "r11d" | "r12d" | "r13d" | "r14d" | "r15d" | "eip" => Some(32),
        "rax" | "rbx" | "rcx" | "rdx" | "rsi" | "rdi" | "rbp" | "rsp" | "r8" | "r9" | "r10"
        | "r11" | "r12" | "r13" | "r14" | "r15" | "rip" => Some(64),
        _ => None,
    }
}

fn aarch64_register_width(lower: &str) -> Option<u16> {
    match lower {
        n if n.starts_with('w') && parse_arm_reg_index(&n[1..]).is_some_and(|i| i <= 30) => {
            Some(32)
        }
        "wsp" | "wzr" => Some(32),
        n if n.starts_with('x') && parse_arm_reg_index(&n[1..]).is_some_and(|i| i <= 30) => {
            Some(64)
        }
        // `sp` is the whole reason this table is dispatched on the
        // architecture. It used to fall into x86's 16-bit arm, so every
        // stack-relative computation radare2 spells with a bare `sp`
        // modelled `(sp & 0xffff)` — `add x2, sp, 8` became
        // `(sp & 0xffff) + 8`, a wrong address rather than a lost one.
        // 1 557 instructions of one sample carry a bare `sp`.
        "sp" | "xzr" | "lr" | "fp" => Some(64),
        _ => None,
    }
}

fn aarch32_register_width(lower: &str) -> Option<u16> {
    match lower {
        // `r8`–`r15` matched x86-64's arm first and came back 64-bit,
        // and `lr` / `fp` were hardcoded to `AArch64`'s width.
        n if n.starts_with('r') && parse_arm_reg_index(&n[1..]).is_some_and(|i| i <= 15) => {
            Some(32)
        }
        "sp" | "lr" | "fp" | "ip" | "pc" => Some(32),
        _ => None,
    }
}

fn parse_arm_reg_index(s: &str) -> Option<u8> {
    if s.is_empty() {
        return None;
    }
    s.parse::<u8>().ok()
}

fn widen(value: StackValue, target_bits: u16) -> Expr {
    let cur_bits = value.bits();
    let expr = value.into_expr();
    if cur_bits == target_bits {
        return expr;
    }
    if cur_bits < target_bits {
        return Expr::zero_ext(expr, target_bits);
    }
    Expr::extract(expr, target_bits - 1, 0)
}

/// Like [`widen`] but sign-extends a narrower value. Used by the signed
/// comparators, where zero-extending a narrower operand would strip its
/// sign — a wrong result for any negative value. A no-op when widths
/// already match, which is the common case (r2 emits same-width operands).
fn widen_signed(value: StackValue, target_bits: u16) -> Expr {
    let cur_bits = value.bits();
    let expr = value.into_expr();
    if cur_bits == target_bits {
        return expr;
    }
    if cur_bits < target_bits {
        return Expr::sign_ext(expr, target_bits);
    }
    Expr::extract(expr, target_bits - 1, 0)
}

/// Reduce any value to a 1-bit truthiness expression: `value != 0` →
/// 1, `value == 0` → 0. Already-1-bit values pass through.
fn cast_to_1bit(value: StackValue) -> Expr {
    let bits = value.bits();
    let expr = value.into_expr();
    if bits == 1 {
        return expr;
    }
    Expr::Ite {
        cond: Box::new(Expr::eq(expr, Expr::konst(0, bits))),
        then_expr: Box::new(Expr::konst(0, 1)),
        else_expr: Box::new(Expr::konst(1, 1)),
    }
}

#[cfg(test)]
mod tests;
