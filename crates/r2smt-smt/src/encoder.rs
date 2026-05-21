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

/// One byte recorded by an [`IrStmt::StoreMem`]: byte address (at the
/// slice's pointer width) and its 8-bit value. A `LoadMem` walks the
/// store list in reverse, building an [`Ite`] chain over the
/// `byte_addr == store.addr` predicate and falling back to a fresh
/// free byte — aliasing is resolved by the solver, no oracle decides.
#[derive(Debug, Clone)]
struct ByteStore {
    addr: BV,
    byte: BV,
}

/// Soft cap on the number of bytes the byte-granular memory model
/// tracks before it havocs the store list and starts answering every
/// subsequent `LoadMem` with a fresh free value. Generous — one full
/// stack frame is well under this — but bounded so a pathological
/// slice cannot blow up Z3's `Ite`-chain depth (Host-Side Safety:
/// solver-bound resources must maintain explicit budgets).
const MEM_BYTE_STORE_CAP: usize = 4096;

/// Encodes IR expressions into Z3 ASTs and feeds the resulting
/// assertions into a [`Solver`].
pub struct Encoder {
    vars: HashMap<String, BV>,
    unknown_counter: u32,
    /// P26 memory model. Byte-granular store list consulted by every
    /// [`IrStmt::LoadMem`]; an empty list and a `false` `mem_havoced`
    /// reproduces the pre-P26 "every load is fresh" behaviour exactly.
    byte_stores: Vec<ByteStore>,
    /// `true` once the byte-store cap fired — all subsequent loads
    /// see a fresh free value (sound widen-only).
    mem_havoced: bool,
    /// Pointer width for the slice currently being encoded. Set at
    /// the top of [`Encoder::encode`]; defaults to 64 for the rare
    /// caller that drives `encode_*` directly without going through
    /// `encode` (no slice → no memory state).
    ptr_bits: u8,
    /// Disambiguates the fresh per-byte free variables minted by
    /// repeated `LoadMem` operations.
    load_counter: u32,
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
            byte_stores: Vec::new(),
            mem_havoced: false,
            ptr_bits: 64,
            load_counter: 0,
        }
    }

    /// Declare and constrain every statement of `slice` into `solver`.
    ///
    /// Returns the Z3 boolean expression representing the branch
    /// condition's *truth value*.
    pub fn encode(&mut self, slice: &SsaLiftedSlice, solver: &Solver) -> Bool {
        self.ptr_bits = slice.arch.pointer_bits();
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
            IrStmt::LoadMem { dst, address, bits } => {
                self.encode_load_mem(dst, address, *bits, solver);
            }
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => {
                self.encode_store_mem(address, value, *bits);
            }
            IrStmt::Unsupported { .. } | IrStmt::Nop => {}
        }
    }

    /// Lower an [`IrStmt::LoadMem`] into a byte-granular value built
    /// by walking the store list in reverse. Each output byte is an
    /// [`Ite`] chain `byte_addr == store.addr ? store.byte : older`
    /// folded from the most recent store backward, with a fresh free
    /// byte as the base case. Aliasing is decided by the solver:
    /// equal addresses pick the stored byte, unequal addresses fall
    /// through to older stores or the fresh free value.
    fn encode_load_mem(&mut self, dst: &Var, address: &Expr, bits: u8, solver: &Solver) {
        let dst_bv = self.declare(&dst.name, bits);
        if bits == 0 {
            return;
        }
        let addr_bv = self.encode_address(address);
        let nbytes = bits.div_ceil(8);
        let load_id = self.load_counter;
        self.load_counter = self.load_counter.wrapping_add(1);
        let mut acc: Option<BV> = None;
        for i in 0..nbytes {
            let byte_addr = bv_add_offset(&addr_bv, i, self.ptr_bits);
            let mut byte_val = mint_free_byte(load_id, i);
            if !self.mem_havoced {
                // Walk stores OLDEST → LATEST so the latest write
                // ends up as the *outermost* `Ite`: the resulting
                // value reads `latest.alias ? latest.byte : (…older
                // chain…)`. With the order reversed, the *oldest*
                // matching write would shadow every subsequent
                // overwrite — silently unsound.
                for store in &self.byte_stores {
                    let alias = byte_addr.eq(&store.addr);
                    byte_val = alias.ite(&store.byte, &byte_val);
                }
            }
            acc = Some(match acc.take() {
                None => byte_val,
                Some(prev) => byte_val.concat(&prev),
            });
        }
        let Some(loaded) = acc else { return };
        let value = coerce_bv(loaded, u32::from(bits));
        let assertion = dst_bv.eq(&value);
        solver.assert(&assertion);
    }

    /// Record an [`IrStmt::StoreMem`] in the byte-granular store
    /// list. Each byte is enqueued at `address + i`, little-endian.
    /// Overflowing [`MEM_BYTE_STORE_CAP`] triggers a sound havoc:
    /// the list is cleared and every subsequent load reads a fresh
    /// free value (precision lost, soundness preserved).
    fn encode_store_mem(&mut self, address: &Expr, value: &Expr, bits: u8) {
        if bits == 0 {
            return;
        }
        let nbytes = usize::from(bits.div_ceil(8));
        if self.byte_stores.len().saturating_add(nbytes) > MEM_BYTE_STORE_CAP {
            self.byte_stores.clear();
            self.mem_havoced = true;
            return;
        }
        let addr_bv = self.encode_address(address);
        let value_bv = self.encode_as_bv_with_width(value, bits);
        for i in 0..u8::try_from(nbytes).unwrap_or(u8::MAX) {
            let byte_addr = bv_add_offset(&addr_bv, i, self.ptr_bits);
            let lo = u32::from(i) * 8;
            let hi = lo + 7;
            let byte = value_bv.extract(hi, lo);
            self.byte_stores.push(ByteStore {
                addr: byte_addr,
                byte,
            });
        }
    }

    /// Encode a memory address expression at the slice's pointer
    /// width, the canonical type for both store and load indices.
    fn encode_address(&mut self, address: &Expr) -> BV {
        self.encode_as_bv_with_width(address, self.ptr_bits)
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
            // Mint at the maximum modelled width (64). A free var
            // narrower than its consumer is *zero-extended* by
            // `match_widths` / `encode_as_bv_with_width`, which caps
            // the unmodelled value's range (e.g. a 64-bit operand
            // would only range over `[0, 2^32)`), fabricating a
            // confident `AlwaysX` and breaking the "Unknowns only
            // weaken, never fabricate" invariant. A 64-bit free var is
            // only ever *truncated* downstream, which stays sound.
            Expr::Unknown(_) => Encoded::Bv(self.fresh_unknown(64)),
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

/// Build `base + offset` at `ptr_bits` width. `offset` is small
/// enough to fit in a `u64`, so no truncation risk: it's the byte
/// index inside a single memory access (≤ 16 for an `ldp` / `stp`).
fn bv_add_offset(base: &BV, offset: u8, ptr_bits: u8) -> BV {
    if offset == 0 {
        return base.clone();
    }
    let off = BV::from_u64(u64::from(offset), u32::from(ptr_bits));
    base.bvadd(&off)
}

/// Mint a fresh, anonymous 8-bit value for byte `i` of load `id` —
/// the base case of a `LoadMem`'s `Ite` chain when no prior store
/// aliases the byte's address.
fn mint_free_byte(id: u32, i: u8) -> BV {
    BV::new_const(format!("__load_{id}_b{i}").as_str(), 8)
}

/// Truncate or zero-extend `bv` to exactly `target` bits. Used by
/// the memory-load reconstruction to coerce the byte-concat (a
/// multiple of 8 bits) to the load's exact width.
fn coerce_bv(bv: BV, target: u32) -> BV {
    let cur = bv.get_size();
    match cur.cmp(&target) {
        std::cmp::Ordering::Equal => bv,
        std::cmp::Ordering::Greater => bv.extract(target - 1, 0),
        std::cmp::Ordering::Less => bv.zero_ext(target - cur),
    }
}
