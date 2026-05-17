//! Translate a [`SsaLiftedSlice`] into Z3 bit-vector ASTs.
//!
//! Uses the thread-local Z3 context exposed by the `z3` 0.20 crate.

use std::collections::HashMap;

use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_ssa::SsaLiftedSlice;
use z3::Solver;
use z3::ast::{BV, Bool};

/// Either a multi-bit bit-vector or a boolean produced by the encoder.
#[derive(Debug, Clone)]
enum Encoded {
    Bv(BV),
    Bool(Bool),
}

/// Encodes IR expressions into Z3 ASTs and feeds the resulting
/// assertions into a [`Solver`].
pub struct Encoder {
    vars: HashMap<String, BV>,
    unknown_counter: u32,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Build a fresh encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            unknown_counter: 0,
        }
    }

    /// Declare and constrain every statement of `slice` into `solver`.
    ///
    /// Returns the Z3 boolean expression representing the branch
    /// condition's *truth value*.
    pub fn encode(&mut self, slice: &SsaLiftedSlice, solver: &Solver) -> Bool {
        for stmt in &slice.statements {
            self.encode_stmt(stmt, solver);
        }
        self.encode_as_bool(&slice.condition)
    }

    fn encode_stmt(&mut self, stmt: &IrStmt, solver: &Solver) {
        match stmt {
            IrStmt::Assign { dst, src } => {
                let rhs = self.encode_as_bv_with_width(src, dst.bits);
                let dst_bv = self.declare(&dst.name, dst.bits);
                let assertion = dst_bv.eq(&rhs);
                solver.assert(&assertion);
            }
            IrStmt::LoadMem { dst, bits, .. } => {
                // Phase 6 has no memory model; declare the destination
                // as a fresh free symbolic value.
                let _ = self.declare(&dst.name, *bits);
            }
            IrStmt::StoreMem { .. } | IrStmt::Unsupported { .. } | IrStmt::Nop => {}
        }
    }

    fn declare(&mut self, name: &str, bits: u8) -> BV {
        if let Some(existing) = self.vars.get(name) {
            return existing.clone();
        }
        let bv = BV::new_const(name, u32::from(bits));
        self.vars.insert(name.to_string(), bv.clone());
        bv
    }

    fn encode_as_bv(&mut self, expr: &Expr) -> BV {
        match self.encode_expr(expr) {
            Encoded::Bv(bv) => bv,
            Encoded::Bool(b) => Self::bool_to_bv(&b),
        }
    }

    fn encode_as_bv_with_width(&mut self, expr: &Expr, target_bits: u8) -> BV {
        let bv = self.encode_as_bv(expr);
        let actual = bv.get_size();
        let target = u32::from(target_bits);
        match actual.cmp(&target) {
            std::cmp::Ordering::Equal => bv,
            std::cmp::Ordering::Less => bv.zero_ext(target - actual),
            std::cmp::Ordering::Greater => bv.extract(target - 1, 0),
        }
    }

    fn encode_as_bool(&mut self, expr: &Expr) -> Bool {
        match self.encode_expr(expr) {
            Encoded::Bool(b) => b,
            Encoded::Bv(bv) => Self::bv_is_true(&bv),
        }
    }

    fn encode_expr(&mut self, expr: &Expr) -> Encoded {
        match expr {
            Expr::Var(v) => Encoded::Bv(self.encode_var(v)),
            Expr::Const { value, bits } => Encoded::Bv(BV::from_u64(*value, u32::from(*bits))),
            Expr::Add(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvadd(&y)),
            Expr::Sub(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvsub(&y)),
            Expr::Mul(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvmul(&y)),
            Expr::UDiv(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvudiv(&y)),
            Expr::URem(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvurem(&y)),
            Expr::SDiv(a, b) => self.bv_bin(a, b, Signedness::Signed, |x, y| x.bvsdiv(&y)),
            Expr::SRem(a, b) => self.bv_bin(a, b, Signedness::Signed, |x, y| x.bvsrem(&y)),
            Expr::And(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvand(&y)),
            Expr::Or(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvor(&y)),
            Expr::Xor(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvxor(&y)),
            Expr::Shl(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvshl(&y)),
            Expr::LShr(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvlshr(&y)),
            Expr::AShr(a, b) => self.bv_bin(a, b, Signedness::Unsigned, |x, y| x.bvashr(&y)),
            Expr::Eq(a, b) => {
                let lhs = self.encode_as_bv(a);
                let rhs = self.encode_as_bv(b);
                let (lhs, rhs) = match_widths(lhs, rhs, Signedness::Unsigned);
                Encoded::Bool(lhs.eq(&rhs))
            }
            Expr::Ne(a, b) => {
                let lhs = self.encode_as_bv(a);
                let rhs = self.encode_as_bv(b);
                let (lhs, rhs) = match_widths(lhs, rhs, Signedness::Unsigned);
                Encoded::Bool(lhs.eq(&rhs).not())
            }
            Expr::Ult(a, b) => self.bool_cmp(a, b, Signedness::Unsigned, |x, y| x.bvult(&y)),
            Expr::Ule(a, b) => self.bool_cmp(a, b, Signedness::Unsigned, |x, y| x.bvule(&y)),
            Expr::Slt(a, b) => self.bool_cmp(a, b, Signedness::Signed, |x, y| x.bvslt(&y)),
            Expr::Sle(a, b) => self.bool_cmp(a, b, Signedness::Signed, |x, y| x.bvsle(&y)),
            Expr::BoolAnd(a, b) => {
                let pa = self.encode_as_bool(a);
                let pb = self.encode_as_bool(b);
                Encoded::Bool(Bool::and(&[&pa, &pb]))
            }
            Expr::BoolOr(a, b) => {
                let pa = self.encode_as_bool(a);
                let pb = self.encode_as_bool(b);
                Encoded::Bool(Bool::or(&[&pa, &pb]))
            }
            Expr::BoolNot(inner) => {
                let p = self.encode_as_bool(inner);
                Encoded::Bool(p.not())
            }
            Expr::Ite {
                cond,
                then_expr,
                else_expr,
            } => {
                let c = self.encode_as_bool(cond);
                let t = self.encode_as_bv(then_expr);
                let e = self.encode_as_bv(else_expr);
                // Ite branches carry no signedness label; widen with
                // zero-extension as the unsigned default. Lifter call
                // sites that need a signed `Ite` should produce
                // matched-width branches.
                let (t, e) = match_widths(t, e, Signedness::Unsigned);
                Encoded::Bv(c.ite(&t, &e))
            }
            Expr::Extract { src, hi, lo } => {
                let bv = self.encode_as_bv(src);
                Encoded::Bv(bv.extract(u32::from(*hi), u32::from(*lo)))
            }
            Expr::Concat { high, low } => {
                let h = self.encode_as_bv(high);
                let l = self.encode_as_bv(low);
                Encoded::Bv(h.concat(&l))
            }
            Expr::ZeroExtend { src, to_bits } => {
                let bv = self.encode_as_bv(src);
                let cur = bv.get_size();
                let target = u32::from(*to_bits);
                let result = match cur.cmp(&target) {
                    std::cmp::Ordering::Equal => bv,
                    std::cmp::Ordering::Less => bv.zero_ext(target - cur),
                    std::cmp::Ordering::Greater => bv.extract(target - 1, 0),
                };
                Encoded::Bv(result)
            }
            Expr::SignExtend { src, to_bits } => {
                let bv = self.encode_as_bv(src);
                let cur = bv.get_size();
                let target = u32::from(*to_bits);
                let result = match cur.cmp(&target) {
                    std::cmp::Ordering::Equal => bv,
                    std::cmp::Ordering::Less => bv.sign_ext(target - cur),
                    std::cmp::Ordering::Greater => bv.extract(target - 1, 0),
                };
                Encoded::Bv(result)
            }
            Expr::Unknown(_) => Encoded::Bv(self.fresh_unknown(32)),
        }
    }

    fn encode_var(&mut self, var: &Var) -> BV {
        self.declare(&var.name, var.bits)
    }

    fn bv_bin<F>(&mut self, a: &Expr, b: &Expr, sign: Signedness, op: F) -> Encoded
    where
        F: FnOnce(BV, BV) -> BV,
    {
        let lhs = self.encode_as_bv(a);
        let rhs = self.encode_as_bv(b);
        let (lhs, rhs) = match_widths(lhs, rhs, sign);
        Encoded::Bv(op(lhs, rhs))
    }

    fn bool_cmp<F>(&mut self, a: &Expr, b: &Expr, sign: Signedness, op: F) -> Encoded
    where
        F: FnOnce(BV, BV) -> Bool,
    {
        let lhs = self.encode_as_bv(a);
        let rhs = self.encode_as_bv(b);
        let (lhs, rhs) = match_widths(lhs, rhs, sign);
        Encoded::Bool(op(lhs, rhs))
    }

    fn bv_is_true(bv: &BV) -> Bool {
        let one = BV::from_u64(1, bv.get_size());
        bv.eq(&one)
    }

    fn bool_to_bv(b: &Bool) -> BV {
        let one = BV::from_u64(1, 1);
        let zero = BV::from_u64(0, 1);
        b.ite(&one, &zero)
    }

    fn fresh_unknown(&mut self, bits: u32) -> BV {
        let name = format!("__unk_{}", self.unknown_counter);
        self.unknown_counter += 1;
        BV::new_const(name.as_str(), bits)
    }
}

/// Whether a binary operation interprets its operands as signed
/// or unsigned bit-vectors when the encoder needs to widen one
/// operand to match the other.
#[derive(Debug, Clone, Copy)]
enum Signedness {
    Signed,
    Unsigned,
}

fn match_widths(lhs: BV, rhs: BV, sign: Signedness) -> (BV, BV) {
    let lw = lhs.get_size();
    let rw = rhs.get_size();
    match lw.cmp(&rw) {
        std::cmp::Ordering::Equal => (lhs, rhs),
        std::cmp::Ordering::Greater => {
            let widened = widen(&rhs, lw - rw, sign);
            (lhs, widened)
        }
        std::cmp::Ordering::Less => {
            let widened = widen(&lhs, rw - lw, sign);
            (widened, rhs)
        }
    }
}

fn widen(bv: &BV, extra: u32, sign: Signedness) -> BV {
    match sign {
        Signedness::Signed => bv.sign_ext(extra),
        Signedness::Unsigned => bv.zero_ext(extra),
    }
}
