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
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;

use crate::flags::flag_token_to_expr_in_ctx;
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
    default_bits: u8,
    stack: Vec<StackValue>,
    statements: Vec<IrStmt>,
    /// Last arithmetic / logical result. ESIL's `$z`, `$s`, `$cN`,
    /// `$bN`, … flag tokens derive their value from this — they
    /// describe "the flag the latest math operation would have set"
    /// rather than the content of a register named ZF/SF/...
    last_arith: Option<LastArith>,
    /// Active predicated-block frames. A `?{` token pushes a frame
    /// holding the condition and the index where the block body
    /// started; the matching `}` token pops the frame and wraps every
    /// `IrStmt::Assign` emitted between `stmt_start` and the close
    /// in `Ite(cond, new_src, Var(dst))`. Memory loads / stores inside
    /// a block flip `contains_mem_op` so the close errors out — the
    /// IR has no conditional memory effect we can model soundly.
    block_stack: Vec<BlockFrame>,
}

/// Family the last arithmetic / logical operation belonged to.
/// Carries enough information for the parametric flag tokens
/// (`$cN`, `$bN`) to know which operand-aware formula to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArithKind {
    /// `+` — `lhs + rhs`. Drives `$cN` (carry into bit N+1).
    Add,
    /// `-` — `lhs - rhs`. Drives `$bN` (borrow into bit N+1).
    Sub,
    /// `*`. Flags are not modelled for multiplication.
    Mul,
    /// `/` (unsigned divide).
    Div,
    /// `%` (unsigned remainder).
    Rem,
    /// `&`, `|`, `^`.
    Logic,
    /// `<<`, `>>` — logical shifts.
    Shift,
}

/// Snapshot of an arithmetic / logical operation kept around so the
/// flag-token evaluator can derive `$z` / `$s` / `$c` / `$o` / `$p`
/// and the parametric `$cN` / `$bN` bit-precise carry/borrow tokens.
#[derive(Debug, Clone)]
pub(crate) struct LastArith {
    pub(crate) kind: ArithKind,
    pub(crate) lhs: Expr,
    pub(crate) rhs: Expr,
    pub(crate) result: Expr,
    pub(crate) bits: u8,
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
    /// name without re-parsing the expression.
    Register(Var),
    /// A computed value.
    Expression { expr: Expr, bits: u8 },
}

impl StackValue {
    fn bits(&self) -> u8 {
        match self {
            StackValue::Register(var) => var.bits,
            StackValue::Expression { bits, .. } => *bits,
        }
    }

    fn into_expr(self) -> Expr {
        match self {
            StackValue::Register(var) => Expr::Var(var),
            StackValue::Expression { expr, .. } => expr,
        }
    }
}

impl Machine {
    fn new(arch: Arch) -> Self {
        Self {
            default_bits: arch.pointer_bits(),
            stack: Vec::new(),
            statements: Vec::new(),
            last_arith: None,
            block_stack: Vec::new(),
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
                let canonical = match name.as_str() {
                    "zf" => "ZF",
                    "cf" => "CF",
                    "sf" => "SF",
                    "of" => "OF",
                    "pf" => "PF",
                    _ => name.as_str(),
                }
                .to_string();
                let bits = register_width(&canonical).unwrap_or(self.default_bits);
                self.stack
                    .push(StackValue::Register(Var::new(canonical, bits)));
                Ok(())
            }
            EsilToken::Flag(suffix) => {
                // ESIL `$z` / `$s` are derived from the *latest math
                // operation*, not from a register named ZF / SF. When
                // the machine has seen an arithmetic op recently, we
                // synthesise the bit from that result. The parametric
                // `$cN` / `$bN` tokens emit a bit-precise carry/borrow
                // expression over the operands of the same snapshot;
                // see `flag_token_to_expr_in_ctx` for the formula.
                // Anything we cannot model surfaces as
                // `UnsupportedFlag` so the slicer falls back to the
                // per-mnemonic handler.
                let expr = match suffix.as_str() {
                    "z" => self.derive_zero_flag(),
                    "s" => self.derive_sign_flag(),
                    other => flag_token_to_expr_in_ctx(other, self.last_arith.as_ref())
                        .ok_or_else(|| EsilError::UnsupportedFlag(suffix.clone()))?,
                };
                self.stack.push(StackValue::Expression { expr, bits: 1 });
                Ok(())
            }
            EsilToken::Binary(op) => self.apply_binary(op),
            EsilToken::Unary(op) => self.apply_unary(op),
            EsilToken::Assign => self.apply_assign(None),
            EsilToken::CompoundAssign(op) => self.apply_assign(Some(op)),
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

    fn push_const(&mut self, value: u64, bits: u8) {
        self.stack.push(StackValue::Expression {
            expr: Expr::konst(value, bits),
            bits,
        });
    }

    fn apply_binary(&mut self, op: &'static str) -> Result<(), EsilError> {
        let rhs = self.pop("binary rhs")?;
        let lhs = self.pop("binary lhs")?;
        let bits = lhs.bits().max(rhs.bits());
        let lhs_e = widen(lhs, bits);
        let rhs_e = widen(rhs, bits);
        let result_bits = match op {
            "==" | "!=" | "<" | "<=" | ">" | ">=" => 1,
            _ => bits,
        };
        let expr = match op {
            "+" => Expr::add(lhs_e.clone(), rhs_e.clone()),
            "-" => Expr::sub(lhs_e.clone(), rhs_e.clone()),
            "*" => Expr::mul(lhs_e.clone(), rhs_e.clone()),
            "/" => Expr::udiv(lhs_e.clone(), rhs_e.clone()),
            "%" => Expr::urem(lhs_e.clone(), rhs_e.clone()),
            "&" => Expr::bv_and(lhs_e.clone(), rhs_e.clone()),
            "|" => Expr::bv_or(lhs_e.clone(), rhs_e.clone()),
            "^" => Expr::bv_xor(lhs_e.clone(), rhs_e.clone()),
            "<<" => Expr::shl(lhs_e.clone(), rhs_e.clone()),
            ">>" => Expr::lshr(lhs_e.clone(), rhs_e.clone()),
            "==" => Expr::eq(lhs_e, rhs_e),
            "!=" => Expr::ne(lhs_e, rhs_e),
            "<" => Expr::ult(lhs_e, rhs_e),
            "<=" => Expr::ule(lhs_e, rhs_e),
            ">" => Expr::ult(rhs_e, lhs_e),
            ">=" => Expr::ule(rhs_e, lhs_e),
            _ => return Err(EsilError::UnknownToken(op.to_string())),
        };
        // Snapshot arithmetic / logical results — the comparison
        // operators do not seed flag-token derivation because their
        // 1-bit output is not the "math result" ESIL's `$z` /
        // `$s` reference. The lhs/rhs operands are retained so the
        // parametric `$cN` / `$bN` tokens can emit a bit-precise
        // carry/borrow formula over the same pre-widened pair.
        let arith_kind = match op {
            "+" => Some(ArithKind::Add),
            "-" => Some(ArithKind::Sub),
            "*" => Some(ArithKind::Mul),
            "/" => Some(ArithKind::Div),
            "%" => Some(ArithKind::Rem),
            "&" | "|" | "^" => Some(ArithKind::Logic),
            "<<" | ">>" => Some(ArithKind::Shift),
            _ => None,
        };
        if let Some(kind) = arith_kind {
            // Recover the pre-widened operands from the IR — `expr`
            // already carries them as its direct children for the
            // arith ops listed above, so we re-extract instead of
            // double-cloning the stack values.
            let (lhs_snap, rhs_snap) = arith_operands(&expr);
            self.last_arith = Some(LastArith {
                kind,
                lhs: lhs_snap,
                rhs: rhs_snap,
                result: expr.clone(),
                bits: result_bits,
            });
        }
        self.stack.push(StackValue::Expression {
            expr,
            bits: result_bits,
        });
        Ok(())
    }

    fn derive_zero_flag(&self) -> Expr {
        match &self.last_arith {
            Some(arith) => Expr::Ite {
                cond: Box::new(Expr::eq(arith.result.clone(), Expr::konst(0, arith.bits))),
                then_expr: Box::new(Expr::konst(1, 1)),
                else_expr: Box::new(Expr::konst(0, 1)),
            },
            None => Expr::Var(Var::new("ZF", 1)),
        }
    }

    fn derive_sign_flag(&self) -> Expr {
        match &self.last_arith {
            Some(arith) if arith.bits > 0 => {
                let hi = arith.bits - 1;
                Expr::extract(arith.result.clone(), hi, hi)
            }
            _ => Expr::Var(Var::new("SF", 1)),
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

    fn apply_assign(&mut self, compound: Option<&'static str>) -> Result<(), EsilError> {
        // ESIL is postfix: for `value,target,=` the stack at this
        // point is `[value, target]`, so the *target* is popped first
        // and the *value* second. Same convention for the compound
        // forms (`value,target,+=`).
        let target = self.pop("assign target")?;
        let value = self.pop("assign value")?;
        let StackValue::Register(target_var) = target else {
            return Err(EsilError::InvalidAssignmentTarget);
        };
        let dst_bits = target_var.bits;
        let value_expr = widen(value, dst_bits);
        let final_expr = match compound {
            None => value_expr,
            Some(op) => {
                let lhs = Expr::Var(target_var.clone());
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
        self.statements.push(IrStmt::Assign {
            dst: target_var,
            src: final_expr,
        });
        Ok(())
    }

    fn apply_load(&mut self, size: u8) -> Result<(), EsilError> {
        let address = self.pop("load address")?;
        let bits = u8::try_from(usize::from(size) * 8).unwrap_or(self.default_bits);
        let dst = Var::new(self.fresh_tmp_name("ld"), bits);
        self.statements.push(IrStmt::LoadMem {
            dst: dst.clone(),
            address: address.into_expr(),
            bits,
        });
        self.mark_active_block_mem_op();
        self.stack.push(StackValue::Register(dst));
        Ok(())
    }

    fn apply_store(&mut self, size: u8) -> Result<(), EsilError> {
        // ESIL store `value,address,=[N]`: stack is [value, address]
        // when the operator runs. Pop the *address* first, then the
        // value being stored.
        let address = self.pop("store address")?;
        let value = self.pop("store value")?;
        let bits = u8::try_from(usize::from(size) * 8).unwrap_or(self.default_bits);
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
fn register_width(name: &str) -> Option<u8> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        // x86 / x86_64 flags. Both `zf` (ESIL convention) and `ZF`
        // (lifter canonical form) resolve here.
        "zf" | "cf" | "sf" | "of" | "pf" | "af" | "df" | "if" | "tf" => Some(1),
        // x86 8-bit sub-registers.
        "al" | "ah" | "bl" | "bh" | "cl" | "ch" | "dl" | "dh" | "sil" | "dil" | "bpl" | "spl"
        | "r8b" | "r9b" | "r10b" | "r11b" | "r12b" | "r13b" | "r14b" | "r15b" => Some(8),
        // x86 16-bit sub-registers.
        "ax" | "bx" | "cx" | "dx" | "si" | "di" | "bp" | "sp" | "r8w" | "r9w" | "r10w" | "r11w"
        | "r12w" | "r13w" | "r14w" | "r15w" => Some(16),
        // x86 32-bit sub-registers.
        "eax" | "ebx" | "ecx" | "edx" | "esi" | "edi" | "ebp" | "esp" | "r8d" | "r9d" | "r10d"
        | "r11d" | "r12d" | "r13d" | "r14d" | "r15d" | "eip" => Some(32),
        // x86 64-bit registers.
        "rax" | "rbx" | "rcx" | "rdx" | "rsi" | "rdi" | "rbp" | "rsp" | "r8" | "r9" | "r10"
        | "r11" | "r12" | "r13" | "r14" | "r15" | "rip" => Some(64),
        // AArch64 32-bit `w` views (general purpose r0..r30 + wsp/wzr).
        n if n.starts_with('w') && parse_arm_reg_index(&n[1..]).is_some_and(|i| i <= 30) => {
            Some(32)
        }
        "wsp" | "wzr" => Some(32),
        // AArch64 64-bit `x` views.
        n if n.starts_with('x') && parse_arm_reg_index(&n[1..]).is_some_and(|i| i <= 30) => {
            Some(64)
        }
        "xzr" | "lr" | "fp" => Some(64),
        // AArch32 general purpose r0..r15.
        n if n.starts_with('r') && parse_arm_reg_index(&n[1..]).is_some_and(|i| i <= 15) => {
            Some(32)
        }
        _ => None,
    }
}

fn parse_arm_reg_index(s: &str) -> Option<u8> {
    if s.is_empty() {
        return None;
    }
    s.parse::<u8>().ok()
}

fn widen(value: StackValue, target_bits: u8) -> Expr {
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

/// Recover the immediate operands of a binary arith expression as
/// owned clones. Used to snapshot the operands behind the latest math
/// operation for the parametric `$cN` / `$bN` flag tokens. The caller
/// guarantees `expr` is one of the supported arith / logic / shift
/// kinds — anything else returns conservative `Unknown` placeholders
/// rather than panicking.
fn arith_operands(expr: &Expr) -> (Expr, Expr) {
    match expr {
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::UDiv(a, b)
        | Expr::URem(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Shl(a, b)
        | Expr::LShr(a, b) => ((**a).clone(), (**b).clone()),
        _ => (Expr::unknown(), Expr::unknown()),
    }
}

#[cfg(test)]
mod tests;
