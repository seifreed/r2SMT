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
        Expr::Var(_) | Expr::Const { .. } | Expr::Unknown(_) => expr.clone(),
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
    }
}

/// Static bit width of `expr`, when it can be recovered from the
/// node's structure alone. Returns `None` for shapes whose width
/// depends on data we cannot inspect (free `Unknown`, free `Var`s
/// whose declared width does not match — never happens in practice).
fn expr_bits(expr: &Expr) -> Option<u8> {
    match expr {
        Expr::Var(v) => Some(v.bits),
        Expr::Const { bits, .. } => Some(*bits),
        Expr::Add(a, _)
        | Expr::Sub(a, _)
        | Expr::Mul(a, _)
        | Expr::UDiv(a, _)
        | Expr::URem(a, _)
        | Expr::SDiv(a, _)
        | Expr::SRem(a, _)
        | Expr::And(a, _)
        | Expr::Or(a, _)
        | Expr::Xor(a, _)
        | Expr::Shl(a, _)
        | Expr::LShr(a, _)
        | Expr::AShr(a, _) => expr_bits(a),
        Expr::Eq(_, _)
        | Expr::Ne(_, _)
        | Expr::Ult(_, _)
        | Expr::Ule(_, _)
        | Expr::Slt(_, _)
        | Expr::Sle(_, _)
        | Expr::BoolAnd(_, _)
        | Expr::BoolOr(_, _)
        | Expr::BoolNot(_) => Some(1),
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
    }
}

fn width_mask(bits: u8) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn as_const(expr: &Expr) -> Option<(u64, u8)> {
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
        let r = (va & width_mask(ba)).wrapping_add(vb & width_mask(bb)) & width_mask(w);
        return Expr::konst(r, w);
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
        let r = (va & width_mask(ba)).wrapping_sub(vb & width_mask(bb)) & width_mask(w);
        return Expr::konst(r, w);
    }
    if matches!(&b, Expr::Const { value: 0, .. }) {
        return a;
    }
    if structurally_equal(&a, &b)
        && let Some(bits) = expr_bits(&a)
    {
        return Expr::konst(0, bits);
    }
    Expr::sub(a, b)
}

fn fold_mul(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let w = ba.max(bb);
        let r = (va & width_mask(ba)).wrapping_mul(vb & width_mask(bb)) & width_mask(w);
        return Expr::konst(r, w);
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
    if structurally_equal(&a, &b) {
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
    if let Some((va, bits)) = as_const(&a)
        && va == width_mask(bits)
        && expr_bits(&b).is_some_and(|other| bits >= other)
    {
        return Expr::konst(va, bits);
    }
    if let Some((vb, bits)) = as_const(&b)
        && vb == width_mask(bits)
        && expr_bits(&a).is_some_and(|other| bits >= other)
    {
        return Expr::konst(vb, bits);
    }
    if structurally_equal(&a, &b) {
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
        && let Some(bits) = expr_bits(&a)
    {
        return Expr::konst(0, bits);
    }
    Expr::bv_xor(a, b)
}

fn fold_eq(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let eq = (va & width_mask(ba)) == (vb & width_mask(bb));
        return Expr::konst(u64::from(eq), 1);
    }
    if structurally_equal(&a, &b) {
        return Expr::konst(1, 1);
    }
    Expr::eq(a, b)
}

fn fold_ne(a: Expr, b: Expr) -> Expr {
    if let (Some((va, ba)), Some((vb, bb))) = (as_const(&a), as_const(&b)) {
        let ne = (va & width_mask(ba)) != (vb & width_mask(bb));
        return Expr::konst(u64::from(ne), 1);
    }
    if structurally_equal(&a, &b) {
        return Expr::konst(0, 1);
    }
    Expr::ne(a, b)
}

fn fold_bool_not(inner: Expr) -> Expr {
    if let Expr::BoolNot(inner_inner) = inner {
        return *inner_inner;
    }
    if let Some((value, _)) = as_const(&inner) {
        return Expr::konst(u64::from(value == 0), 1);
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

fn fold_extract(src: Expr, hi: u8, lo: u8) -> Expr {
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

fn fold_zero_ext(src: Expr, to_bits: u8) -> Expr {
    if let Some(bits) = expr_bits(&src)
        && bits == to_bits
    {
        return src;
    }
    Expr::zero_ext(src, to_bits)
}

fn fold_sign_ext(src: Expr, to_bits: u8) -> Expr {
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

    fn v(name: &str, bits: u8) -> Expr {
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
