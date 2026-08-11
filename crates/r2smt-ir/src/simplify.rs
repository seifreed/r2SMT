//! Peephole simplifier for [`Expr`].
//!
//! Walks an [`Expr`] tree bottom-up and applies a small set of
//! algebraic identities (constant folding, neutral / absorbing
//! elements, idempotents, redundant width ops). The result is
//! semantically equivalent and never wider than the input — designed
//! so the pretty-printer (`r2smt-ssa::pretty::pretty_condition`) can
//! consume it directly and produce shorter analyst-facing formulas.
//!
//! Out of scope:
//!
//! - flow-sensitive reasoning (`Ite(cond, x, x) → x` requires
//!   structural equality, which we do not yet have);
//! - SMT-tactic-level simplification (`r2smt-smt::pretty` covers that
//!   path against the Z3 binding).
//!
//! Every fold is a pure rewrite — no global state, no panic, no
//! `unwrap`. Unknown / unsupported shapes pass through verbatim.

use crate::expr::Expr;

/// Apply the peephole rule set to `expr` and return the simplified
/// form. Idempotent: `simplify_expr(simplify_expr(e)) ==
/// simplify_expr(e)` on every supported shape (regression-tested).
#[must_use]
pub fn simplify_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Var(_) | Expr::Const { .. } | Expr::Unknown(_) | Expr::FpConst { .. } => expr.clone(),
        Expr::Add(a, b) => fold_add(simplify_expr(a), simplify_expr(b)),
        Expr::Sub(a, b) => fold_sub(simplify_expr(a), simplify_expr(b)),
        Expr::Mul(a, b) => fold_mul(simplify_expr(a), simplify_expr(b)),
        Expr::UDiv(a, b) => Expr::udiv(simplify_expr(a), simplify_expr(b)),
        Expr::URem(a, b) => Expr::urem(simplify_expr(a), simplify_expr(b)),
        Expr::SDiv(a, b) => Expr::sdiv(simplify_expr(a), simplify_expr(b)),
        Expr::SRem(a, b) => Expr::srem(simplify_expr(a), simplify_expr(b)),
        Expr::And(a, b) => fold_and(simplify_expr(a), simplify_expr(b)),
        Expr::Or(a, b) => fold_or(simplify_expr(a), simplify_expr(b)),
        Expr::Xor(a, b) => fold_xor(simplify_expr(a), simplify_expr(b)),
        Expr::Shl(a, b) => Expr::shl(simplify_expr(a), simplify_expr(b)),
        Expr::Ror(a, b) => Expr::ror(simplify_expr(a), simplify_expr(b)),
        Expr::LShr(a, b) => Expr::lshr(simplify_expr(a), simplify_expr(b)),
        Expr::AShr(a, b) => Expr::ashr(simplify_expr(a), simplify_expr(b)),
        Expr::Eq(a, b) => fold_eq(simplify_expr(a), simplify_expr(b)),
        Expr::Ne(a, b) => fold_ne(simplify_expr(a), simplify_expr(b)),
        Expr::Ult(a, b) => Expr::ult(simplify_expr(a), simplify_expr(b)),
        Expr::Ule(a, b) => Expr::ule(simplify_expr(a), simplify_expr(b)),
        Expr::Slt(a, b) => Expr::slt(simplify_expr(a), simplify_expr(b)),
        Expr::Sle(a, b) => Expr::sle(simplify_expr(a), simplify_expr(b)),
        Expr::BoolAnd(a, b) => Expr::bool_and(simplify_expr(a), simplify_expr(b)),
        Expr::BoolOr(a, b) => Expr::bool_or(simplify_expr(a), simplify_expr(b)),
        Expr::BoolNot(inner) => fold_bool_not(simplify_expr(inner)),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => fold_ite(
            simplify_expr(cond),
            simplify_expr(then_expr),
            simplify_expr(else_expr),
        ),
        Expr::Extract { src, hi, lo } => fold_extract(simplify_expr(src), *hi, *lo),
        Expr::Concat { high, low } => Expr::concat(simplify_expr(high), simplify_expr(low)),
        Expr::ZeroExtend { src, to_bits } => fold_zero_ext(simplify_expr(src), *to_bits),
        Expr::SignExtend { src, to_bits } => fold_sign_ext(simplify_expr(src), *to_bits),
        // Floating-point nodes carry no bit-vector folding rules — the
        // simplifier only recurses structurally into their operands.
        Expr::FAdd(a, b, rm) => {
            Expr::FAdd(Box::new(simplify_expr(a)), Box::new(simplify_expr(b)), *rm)
        }
        Expr::FSub(a, b, rm) => {
            Expr::FSub(Box::new(simplify_expr(a)), Box::new(simplify_expr(b)), *rm)
        }
        Expr::FMul(a, b, rm) => {
            Expr::FMul(Box::new(simplify_expr(a)), Box::new(simplify_expr(b)), *rm)
        }
        Expr::FDiv(a, b, rm) => {
            Expr::FDiv(Box::new(simplify_expr(a)), Box::new(simplify_expr(b)), *rm)
        }
        Expr::FEq(a, b) => Expr::FEq(Box::new(simplify_expr(a)), Box::new(simplify_expr(b))),
        Expr::FLt(a, b) => Expr::FLt(Box::new(simplify_expr(a)), Box::new(simplify_expr(b))),
        Expr::FLe(a, b) => Expr::FLe(Box::new(simplify_expr(a)), Box::new(simplify_expr(b))),
        Expr::FIsNaN(a) => Expr::FIsNaN(Box::new(simplify_expr(a))),
        Expr::FSqrt(a, rm) => Expr::FSqrt(Box::new(simplify_expr(a)), *rm),
        Expr::FRoundToIntegral(a, rm) => Expr::FRoundToIntegral(Box::new(simplify_expr(a)), *rm),
        Expr::BvToFp { src, ebits, sbits } => Expr::BvToFp {
            src: Box::new(simplify_expr(src)),
            ebits: *ebits,
            sbits: *sbits,
        },
        Expr::FpToIeeeBv(src) => Expr::FpToIeeeBv(Box::new(simplify_expr(src))),
        Expr::FpToSbv { src, rm, bits } => Expr::FpToSbv {
            src: Box::new(simplify_expr(src)),
            rm: *rm,
            bits: *bits,
        },
        Expr::SbvToFp {
            src,
            rm,
            ebits,
            sbits,
        } => Expr::SbvToFp {
            src: Box::new(simplify_expr(src)),
            rm: *rm,
            ebits: *ebits,
            sbits: *sbits,
        },
        Expr::FpToFp {
            src,
            rm,
            ebits,
            sbits,
        } => Expr::FpToFp {
            src: Box::new(simplify_expr(src)),
            rm: *rm,
            ebits: *ebits,
            sbits: *sbits,
        },
    }
}

/// Static bit width of `expr`, when it can be recovered from the
/// node's structure alone. Returns `None` for shapes whose width
/// depends on data we cannot inspect (free `Unknown`, free `Var`s
/// whose declared width does not match — never happens in practice).
fn expr_bits(expr: &Expr) -> Option<u16> {
    match expr {
        Expr::Var(v) => Some(v.bits),
        Expr::Const { bits, .. } | Expr::FpToSbv { bits, .. } => Some(*bits),
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::UDiv(a, b)
        | Expr::URem(a, b)
        | Expr::SDiv(a, b)
        | Expr::SRem(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Shl(a, b)
        | Expr::LShr(a, b)
        | Expr::AShr(a, b) => {
            // The encoder widens mismatched operands to the wider of the
            // two (`match_widths`), so the node's width is the max, not
            // the left operand's. Underestimating it lets `fold_or`'s
            // all-ones absorb and `fold_extract`'s identity fire on a
            // value whose true high bits they would then drop. Fall back
            // to the left width when the right operand's is unrecoverable
            // — never a *lower* estimate than before, so this only ever
            // withholds an unsound fold, never enables one.
            let wa = expr_bits(a)?;
            Some(expr_bits(b).map_or(wa, |wb| wa.max(wb)))
        }
        // Rotate keeps the value's width, and floating point keeps its
        // sort — both operands share it, so the left width is exact.
        Expr::Ror(a, _)
        | Expr::FAdd(a, ..)
        | Expr::FSub(a, ..)
        | Expr::FMul(a, ..)
        | Expr::FDiv(a, ..) => expr_bits(a),
        Expr::Eq(_, _)
        | Expr::Ne(_, _)
        | Expr::Ult(_, _)
        | Expr::Ule(_, _)
        | Expr::Slt(_, _)
        | Expr::Sle(_, _)
        | Expr::BoolAnd(_, _)
        | Expr::BoolOr(_, _)
        | Expr::BoolNot(_)
        | Expr::FEq(..)
        | Expr::FLt(..)
        | Expr::FLe(..)
        | Expr::FIsNaN(_) => Some(1),
        Expr::Ite {
            then_expr,
            else_expr,
            ..
        } => expr_bits(then_expr).or_else(|| expr_bits(else_expr)),
        Expr::Extract { hi, lo, .. } => Some(hi.saturating_sub(*lo).saturating_add(1)),
        Expr::Concat { high, low } => {
            let h = expr_bits(high)?;
            let l = expr_bits(low)?;
            h.checked_add(l)
        }
        Expr::ZeroExtend { to_bits, .. } | Expr::SignExtend { to_bits, .. } => Some(*to_bits),
        Expr::Unknown(_) => None,
        Expr::FpConst { ebits, sbits, .. }
        | Expr::BvToFp { ebits, sbits, .. }
        | Expr::SbvToFp { ebits, sbits, .. }
        | Expr::FpToFp { ebits, sbits, .. } => ebits.checked_add(*sbits),
        // Square root and rounding to integral stay in their operand's
        // sort.
        Expr::FpToIeeeBv(src) | Expr::FSqrt(src, _) | Expr::FRoundToIntegral(src, _) => {
            expr_bits(src)
        }
    }
}

fn width_mask(bits: u16) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// `true` if `expr` contains an [`Expr::Unknown`] anywhere in its tree.
///
/// A self-referential fold (`x == x → 1`, `x - x → 0`, `x ^ x → 0`) is
/// sound only when `x` is deterministic. Each `Unknown` encodes to a
/// *fresh* free variable in the solver backends, so two structurally
/// identical `Unknown`-bearing operands are not the same runtime value —
/// and folding them away also strips the `slice_contains_unknown` decline
/// signal those backends rely on, turning a safe decline into a
/// fabricated definite verdict. `expr_bits(&e).is_some()` cannot stand in
/// for this: an `Unknown` under an `Extract` / `ZeroExtend` / comparison
/// still yields a width.
fn contains_unknown(expr: &Expr) -> bool {
    match expr {
        Expr::Unknown(_) => true,
        Expr::Var(_) | Expr::Const { .. } | Expr::FpConst { .. } => false,
        Expr::Add(a, b)
        | Expr::Ror(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::UDiv(a, b)
        | Expr::URem(a, b)
        | Expr::SDiv(a, b)
        | Expr::SRem(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Shl(a, b)
        | Expr::LShr(a, b)
        | Expr::AShr(a, b)
        | Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Ult(a, b)
        | Expr::Ule(a, b)
        | Expr::Slt(a, b)
        | Expr::Sle(a, b)
        | Expr::BoolAnd(a, b)
        | Expr::BoolOr(a, b)
        | Expr::FAdd(a, b, _)
        | Expr::FSub(a, b, _)
        | Expr::FMul(a, b, _)
        | Expr::FDiv(a, b, _)
        | Expr::FEq(a, b)
        | Expr::FLt(a, b)
        | Expr::FLe(a, b) => contains_unknown(a) || contains_unknown(b),
        Expr::BoolNot(e)
        | Expr::Extract { src: e, .. }
        | Expr::ZeroExtend { src: e, .. }
        | Expr::SignExtend { src: e, .. }
        | Expr::FIsNaN(e)
        | Expr::FSqrt(e, _)
        | Expr::FRoundToIntegral(e, _)
        | Expr::FpToIeeeBv(e)
        | Expr::BvToFp { src: e, .. }
        | Expr::FpToSbv { src: e, .. }
        | Expr::SbvToFp { src: e, .. }
        | Expr::FpToFp { src: e, .. } => contains_unknown(e),
        Expr::Concat { high, low } => contains_unknown(high) || contains_unknown(low),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => contains_unknown(cond) || contains_unknown(then_expr) || contains_unknown(else_expr),
    }
}

fn as_const(expr: &Expr) -> Option<(u128, u16)> {
    match expr {
        Expr::Const { value, bits } => Some((*value, *bits)),
        _ => None,
    }
}

fn structurally_equal(a: &Expr, b: &Expr) -> bool {
    a == b
}

fn fold_add(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        // A `Const` value is a `u128`, so beyond 128 bits the carry out
        // of bit 127 has nowhere to land and the true result is
        // unrepresentable. Leave it symbolic rather than fold to a
        // truncated (wrong) constant — the encoder computes it at full
        // width. (`width_mask` saturates to `u128::MAX` there, the same
        // hazard `fold_or` already guards with `bits <= 128`.)
        if w <= 128 {
            let r = (va & width_mask(ba)).wrapping_add(vb & width_mask(bb)) & width_mask(w);
            return Expr::konst(r, w);
        }
    }
    if matches!(&a, Expr::Const { value: 0, .. }) {
        return b;
    }
    if matches!(&b, Expr::Const { value: 0, .. }) {
        return a;
    }
    Expr::add(a, b)
}

fn fold_sub(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        // See `fold_add`: a borrow past bit 127 is unrepresentable in a
        // `u128` constant, so only fold up to 128 bits.
        if w <= 128 {
            let r = (va & width_mask(ba)).wrapping_sub(vb & width_mask(bb)) & width_mask(w);
            return Expr::konst(r, w);
        }
    }
    if matches!(&b, Expr::Const { value: 0, .. }) {
        return a;
    }
    if structurally_equal(&a, &b)
        && !contains_unknown(&a)
        && let Some(bits) = expr_bits(&a)
    {
        return Expr::konst(0, bits);
    }
    Expr::sub(a, b)
}

fn fold_mul(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        // See `fold_add`: a product overflows 128 bits well before its
        // declared width, and the high bits are unrepresentable in a
        // `u128` constant, so only fold up to 128 bits.
        if w <= 128 {
            let r = (va & width_mask(ba)).wrapping_mul(vb & width_mask(bb)) & width_mask(w);
            return Expr::konst(r, w);
        }
    }
    if let Some((0, bits)) = as_const(&a) {
        return Expr::konst(0, bits);
    }
    if let Some((0, bits)) = as_const(&b) {
        return Expr::konst(0, bits);
    }
    if matches!(&a, Expr::Const { value: 1, .. }) {
        return b;
    }
    if matches!(&b, Expr::Const { value: 1, .. }) {
        return a;
    }
    Expr::mul(a, b)
}

fn fold_and(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        let r = (va & width_mask(ba)) & (vb & width_mask(bb));
        return Expr::konst(r & width_mask(w), w);
    }
    if let Some((0, bits)) = as_const(&a) {
        return Expr::konst(0, bits);
    }
    if let Some((0, bits)) = as_const(&b) {
        return Expr::konst(0, bits);
    }
    // `x & x → x` is sound only when `x` is deterministic: two
    // structurally identical `Unknown`-bearing operands encode to
    // *distinct* fresh free variables, so collapsing them fabricates a
    // definite value (and strips the `slice_contains_unknown` decline
    // signal). Same guard as `fold_sub` / `fold_xor` / `fold_eq`.
    if structurally_equal(&a, &b) && !contains_unknown(&a) {
        return a;
    }
    Expr::bv_and(a, b)
}

fn fold_or(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        let r = (va & width_mask(ba)) | (vb & width_mask(bb));
        return Expr::konst(r & width_mask(w), w);
    }
    if matches!(&a, Expr::Const { value: 0, .. }) {
        return b;
    }
    if matches!(&b, Expr::Const { value: 0, .. }) {
        return a;
    }
    // `x | all_ones_W = all_ones_W` only when the all-ones constant is
    // at least as wide as the other operand: otherwise the result's
    // high bits come from `x`, not from the constant, and absorbing
    // would fabricate a definitive value (and a confident verdict).
    // The `bits <= 128` guard is load-bearing above 128-bit widths:
    // `width_mask` saturates to `u128::MAX` there, so without it a
    // low-128-ones constant at (say) 256 bits would be misread as
    // all-ones@256. A genuine all-ones constant wider than 128 bits is
    // unrepresentable as a single `Const` (`value: u128`), so the rule
    // simply cannot apply and the guard loses nothing.
    if let Some((va, bits)) = as_const(&a)
        && bits <= 128
        && va == width_mask(bits)
        && expr_bits(&b).is_some_and(|other| bits >= other)
    {
        return Expr::konst(va, bits);
    }
    if let Some((vb, bits)) = as_const(&b)
        && bits <= 128
        && vb == width_mask(bits)
        && expr_bits(&a).is_some_and(|other| bits >= other)
    {
        return Expr::konst(vb, bits);
    }
    // `x | x → x`: sound only when `x` is deterministic, for the same
    // reason as `fold_and` — two identical `Unknown`-bearing operands are
    // distinct fresh free variables.
    if structurally_equal(&a, &b) && !contains_unknown(&a) {
        return a;
    }
    Expr::bv_or(a, b)
}

fn fold_xor(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        let r = (va & width_mask(ba)) ^ (vb & width_mask(bb));
        return Expr::konst(r & width_mask(w), w);
    }
    if matches!(&a, Expr::Const { value: 0, .. }) {
        return b;
    }
    if matches!(&b, Expr::Const { value: 0, .. }) {
        return a;
    }
    if structurally_equal(&a, &b)
        && !contains_unknown(&a)
        && let Some(bits) = expr_bits(&a)
    {
        return Expr::konst(0, bits);
    }
    Expr::bv_xor(a, b)
}

fn fold_eq(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let eq = (va & width_mask(ba)) == (vb & width_mask(bb));
        return Expr::konst(u128::from(eq), 1);
    }
    if structurally_equal(&a, &b) && !contains_unknown(&a) {
        return Expr::konst(1, 1);
    }
    Expr::eq(a, b)
}

fn fold_ne(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let ne = (va & width_mask(ba)) != (vb & width_mask(bb));
        return Expr::konst(u128::from(ne), 1);
    }
    if structurally_equal(&a, &b) && !contains_unknown(&a) {
        return Expr::konst(0, 1);
    }
    Expr::ne(a, b)
}

fn fold_bool_not(inner: Expr) -> Expr {
    if let Expr::BoolNot(inner_inner) = inner {
        return *inner_inner;
    }
    if let Some((value, _)) = as_const(&inner) {
        return Expr::konst(u128::from(value == 0), 1);
    }
    Expr::bool_not(inner)
}

fn fold_ite(cond: Expr, then_expr: Expr, else_expr: Expr) -> Expr {
    if let Some((value, _)) = as_const(&cond) {
        return if value == 0 { else_expr } else { then_expr };
    }
    Expr::Ite {
        cond: Box::new(cond),
        then_expr: Box::new(then_expr),
        else_expr: Box::new(else_expr),
    }
}

fn fold_extract(src: Expr, hi: u16, lo: u16) -> Expr {
    if let Some(bits) = expr_bits(&src)
        && lo == 0
        && hi.saturating_add(1) == bits
    {
        return src;
    }
    // Extract past a zero-extend (when the slice lives entirely in the
    // original payload): drop the zero-extend.
    if let Expr::ZeroExtend { src: inner, .. } = &src
        && let Some(inner_bits) = expr_bits(inner)
        && hi < inner_bits
    {
        return Expr::extract((**inner).clone(), hi, lo);
    }
    Expr::extract(src, hi, lo)
}

fn fold_zero_ext(src: Expr, to_bits: u16) -> Expr {
    if let Some(bits) = expr_bits(&src)
        && bits == to_bits
    {
        return src;
    }
    Expr::zero_ext(src, to_bits)
}

fn fold_sign_ext(src: Expr, to_bits: u16) -> Expr {
    if let Some(bits) = expr_bits(&src)
        && bits == to_bits
    {
        return src;
    }
    Expr::sign_ext(src, to_bits)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use crate::expr::Var;

    fn v(name: &str, bits: u16) -> Expr {
        Expr::Var(Var::new(name, bits))
    }

    #[test]
    fn const_fold_add() {
        let e = Expr::add(Expr::konst(2, 32), Expr::konst(3, 32));
        assert_eq!(simplify_expr(&e), Expr::konst(5, 32));
    }

    #[test]
    fn add_zero_eliminates_on_both_sides() {
        let lhs = Expr::add(v("x", 32), Expr::konst(0, 32));
        let rhs = Expr::add(Expr::konst(0, 32), v("x", 32));
        assert_eq!(simplify_expr(&lhs), v("x", 32));
        assert_eq!(simplify_expr(&rhs), v("x", 32));
    }

    #[test]
    fn sub_self_yields_zero() {
        let e = Expr::sub(v("x", 32), v("x", 32));
        assert_eq!(simplify_expr(&e), Expr::konst(0, 32));
    }

    #[test]
    fn mul_zero_collapses_to_zero() {
        let e = Expr::mul(v("x", 32), Expr::konst(0, 32));
        assert_eq!(simplify_expr(&e), Expr::konst(0, 32));
    }

    #[test]
    fn eq_of_two_unknowns_does_not_fold_to_true() {
        // Each `Unknown` encodes to a fresh free variable, so two
        // structurally identical unknowns are not the same value.
        // Folding to a constant true would both assert a false equality
        // and strip the decline signal the solver backends rely on.
        let e = Expr::eq(Expr::unknown(), Expr::unknown());
        let s = simplify_expr(&e);
        assert_eq!(s, e, "must stay a symbolic Eq carrying the Unknowns");
        assert!(contains_unknown(&s));
    }

    #[test]
    fn ne_of_two_unknowns_does_not_fold_to_false() {
        let e = Expr::ne(Expr::unknown(), Expr::unknown());
        let s = simplify_expr(&e);
        assert_eq!(s, e);
        assert!(contains_unknown(&s));
    }

    #[test]
    fn sub_self_does_not_fold_when_the_operand_hides_an_unknown() {
        // `expr_bits(Extract(Unknown, ..))` is `Some`, so the width guard
        // alone would let `x - x → 0` fire and erase the Unknown. The
        // extract's source is a fresh var per side, so the difference is
        // not provably zero.
        let x = Expr::extract(Expr::unknown(), 7, 0);
        let e = Expr::sub(x.clone(), x);
        let s = simplify_expr(&e);
        assert!(contains_unknown(&s), "the Unknown must survive: {s:?}");
        assert_ne!(s, Expr::konst(0, 8));
    }

    #[test]
    fn xor_self_does_not_fold_when_the_operand_hides_an_unknown() {
        let x = Expr::extract(Expr::unknown(), 7, 0);
        let e = Expr::bv_xor(x.clone(), x);
        let s = simplify_expr(&e);
        assert!(contains_unknown(&s), "the Unknown must survive: {s:?}");
    }

    #[test]
    fn and_self_does_not_fold_when_the_operand_hides_an_unknown() {
        // `x & x → x` would collapse two distinct fresh free variables
        // into one and strip the decline signal.
        let x = Expr::extract(Expr::unknown(), 7, 0);
        let e = Expr::bv_and(x.clone(), x);
        let s = simplify_expr(&e);
        assert!(contains_unknown(&s), "the Unknown must survive: {s:?}");
    }

    #[test]
    fn or_self_does_not_fold_when_the_operand_hides_an_unknown() {
        let x = Expr::extract(Expr::unknown(), 7, 0);
        let e = Expr::bv_or(x.clone(), x);
        let s = simplify_expr(&e);
        assert!(contains_unknown(&s), "the Unknown must survive: {s:?}");
    }

    #[test]
    fn wide_constant_add_that_overflows_128_bits_is_not_folded_wrong() {
        // At 256 bits the true sum has bit 128 set, which a `u128`
        // constant cannot hold; folding would truncate it to a wrong
        // value, so it stays symbolic for the encoder to compute.
        let e = Expr::add(Expr::konst(u128::MAX, 256), Expr::konst(1, 256));
        let s = simplify_expr(&e);
        assert_ne!(
            s,
            Expr::konst(0, 256),
            "must not truncate to a wrong constant"
        );
        assert!(matches!(s, Expr::Add(..)), "left symbolic: {s:?}");
    }

    #[test]
    fn mul_one_eliminates() {
        let e = Expr::mul(v("x", 32), Expr::konst(1, 32));
        assert_eq!(simplify_expr(&e), v("x", 32));
    }

    #[test]
    fn xor_self_yields_zero() {
        let e = Expr::bv_xor(v("x", 32), v("x", 32));
        assert_eq!(simplify_expr(&e), Expr::konst(0, 32));
    }

    #[test]
    fn and_self_is_idempotent() {
        let e = Expr::bv_and(v("x", 32), v("x", 32));
        assert_eq!(simplify_expr(&e), v("x", 32));
    }

    #[test]
    fn or_with_all_ones_returns_ones() {
        let ones = Expr::konst(width_mask(32), 32);
        let e = Expr::bv_or(v("x", 32), ones.clone());
        assert_eq!(simplify_expr(&e), ones);
    }

    #[test]
    fn ite_true_collapses_to_then() {
        let e = Expr::Ite {
            cond: Box::new(Expr::konst(1, 1)),
            then_expr: Box::new(v("a", 32)),
            else_expr: Box::new(v("b", 32)),
        };
        assert_eq!(simplify_expr(&e), v("a", 32));
    }

    #[test]
    fn ite_false_collapses_to_else() {
        let e = Expr::Ite {
            cond: Box::new(Expr::konst(0, 1)),
            then_expr: Box::new(v("a", 32)),
            else_expr: Box::new(v("b", 32)),
        };
        assert_eq!(simplify_expr(&e), v("b", 32));
    }

    #[test]
    fn zext_to_same_width_is_identity() {
        let e = Expr::zero_ext(v("x", 32), 32);
        assert_eq!(simplify_expr(&e), v("x", 32));
    }

    #[test]
    fn extract_of_zext_through_when_in_range() {
        // Extract bits 15:0 of zext(x:8 → 32) === Extract(x, 15, 0) is
        // out of range (x is 8 bits). The in-range case: extract 7:0 of
        // zext(x:8 → 32) should drop the zext.
        let e = Expr::extract(Expr::zero_ext(v("x", 8), 32), 7, 0);
        match simplify_expr(&e) {
            Expr::Extract { src, hi, lo } => {
                assert_eq!(*src, v("x", 8));
                assert_eq!(hi, 7);
                assert_eq!(lo, 0);
            }
            other => panic!("expected Extract over the raw var, got {other:?}"),
        }
    }

    #[test]
    fn extract_full_width_is_identity() {
        let e = Expr::extract(v("x", 32), 31, 0);
        assert_eq!(simplify_expr(&e), v("x", 32));
    }

    #[test]
    fn boolnot_boolnot_cancels() {
        let e = Expr::bool_not(Expr::bool_not(Expr::eq(v("x", 32), Expr::konst(0, 32))));
        let s = simplify_expr(&e);
        assert!(matches!(s, Expr::Eq(_, _)));
    }

    #[test]
    fn eq_of_equal_consts_folds_to_one() {
        let e = Expr::eq(Expr::konst(7, 32), Expr::konst(7, 32));
        assert_eq!(simplify_expr(&e), Expr::konst(1, 1));
    }

    #[test]
    fn ne_of_distinct_consts_folds_to_one() {
        let e = Expr::ne(Expr::konst(1, 32), Expr::konst(2, 32));
        assert_eq!(simplify_expr(&e), Expr::konst(1, 1));
    }

    #[test]
    fn eq_of_out_of_width_const_matches_in_width_twin() {
        // P-code emits negative constants as full 64-bit two's complement
        // even when the varnode size is 4 (e.g. `(const,0xff..fc,4)` == -4).
        // At the declared 32-bit width both encode 0xfffffffc, so `==` is true.
        let e = Expr::eq(
            Expr::konst(0xffff_ffff_ffff_fffc, 32),
            Expr::konst(0x0000_0000_ffff_fffc, 32),
        );
        assert_eq!(simplify_expr(&e), Expr::konst(1, 1));
    }

    #[test]
    fn add_of_mixed_width_consts_uses_max_width_not_first_operand() {
        // BV `+` result width is max(ba,bb), narrower operand
        // zero-extended. `0:32 + 0x1_0000_0000:64 == 0x1_0000_0000:64`
        // is TRUE; masking to the first operand's 32 bits would fold
        // it to AlwaysFalse — a True↔False flip.
        let e = Expr::eq(
            Expr::add(Expr::konst(0, 32), Expr::konst(0x1_0000_0000, 64)),
            Expr::konst(0x1_0000_0000, 64),
        );
        assert_eq!(simplify_expr(&e), Expr::konst(1, 1));
    }

    #[test]
    fn mul_of_mixed_width_consts_does_not_truncate_to_narrow_operand() {
        // 2:8 * 0x80:32 = 0x100 at width 32; masking to 8 bits would
        // wrongly yield 0.
        let e = Expr::mul(Expr::konst(2, 8), Expr::konst(0x80, 32));
        assert_eq!(simplify_expr(&e), Expr::konst(0x100, 32));
    }

    #[test]
    fn or_with_narrow_all_ones_does_not_absorb_wider_operand() {
        // `0xFF:8 | rax:32` is NOT 0xFF — the high 24 bits come from
        // `rax`. The all-ones absorbing rule must not fire here.
        let e = Expr::bv_or(Expr::konst(0xFF, 8), Expr::Var(Var::new("rax", 32)));
        assert!(
            !matches!(simplify_expr(&e), Expr::Const { .. }),
            "narrow all-ones must not absorb a wider operand"
        );
    }

    #[test]
    fn or_with_narrow_all_ones_does_not_absorb_wider_arithmetic_operand() {
        // `0xFF:8 | (x:8 + y:64)`. The add's true width is 64 (the encoder
        // widens to the max operand), so its high 56 bits are not all-ones
        // and the absorbing rule must not fire. It only fired before
        // `expr_bits` took the max of both operands rather than the left.
        let e = Expr::bv_or(
            Expr::konst(0xFF, 8),
            Expr::add(Expr::Var(Var::new("x", 8)), Expr::Var(Var::new("y", 64))),
        );
        assert!(
            !matches!(simplify_expr(&e), Expr::Const { .. }),
            "narrow all-ones must not absorb a wider arithmetic operand"
        );
    }

    #[test]
    fn or_absorb_does_not_fire_above_128_bits() {
        // `u128::MAX:256 | zmm0:256` — at 256 bits `u128::MAX` is only
        // the low 128 ones, NOT all-ones@256, so absorbing would zero
        // out `zmm0`'s real high 128 bits. `width_mask` saturates above
        // 128, so the rule must be gated off there.
        let e = Expr::bv_or(
            Expr::konst(u128::MAX, 256),
            Expr::Var(Var::new("zmm0", 256)),
        );
        assert!(
            !matches!(simplify_expr(&e), Expr::Const { .. }),
            "all-ones absorbing must not fire for >128-bit widths"
        );
    }

    #[test]
    fn ne_of_out_of_width_const_matches_in_width_twin() {
        let e = Expr::ne(
            Expr::konst(0xffff_ffff_ffff_fffc, 32),
            Expr::konst(0x0000_0000_ffff_fffc, 32),
        );
        assert_eq!(simplify_expr(&e), Expr::konst(0, 1));
    }

    #[test]
    fn simplify_is_idempotent_on_combined_rule_set() {
        // (x + 0) - (x + 0) — sub_self_yields_zero after add fold.
        let e = Expr::sub(
            Expr::add(v("x", 32), Expr::konst(0, 32)),
            Expr::add(Expr::konst(0, 32), v("x", 32)),
        );
        let once = simplify_expr(&e);
        let twice = simplify_expr(&once);
        assert_eq!(once, twice);
        assert_eq!(once, Expr::konst(0, 32));
    }

    #[test]
    fn unknown_passes_through() {
        let e = Expr::Unknown("foo".into());
        assert_eq!(simplify_expr(&e), e);
    }
}
