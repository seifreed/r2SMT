//! ESIL pseudo-flag tokens.
//!
//! ESIL emits `$z`, `$c`, `$s`, `$o`, `$p`, `$b`, plus parametric
//! `$cN` / `$bN` variants. They evaluate to a 1-bit result derived
//! from the *last* operation that affected NZCV — the stack machine
//! tracks that operation and answers the flag query at expression
//! time.
//!
//! For r2SMT the mapping is straightforward: each named flag becomes
//! a 1-bit [`Var`] with the canonical ASCII name (`ZF` / `CF` /
//! `SF` / `OF` / `PF`) used everywhere else in the lifter. The
//! parametric `$cN` / `$bN` tokens emit bit-precise carry/borrow
//! expressions over the operands snapshot in the machine's
//! `LastArith`; the formula is the XOR-of-MSB-bits derivation
//! documented inline in `flag_token_to_expr_in_ctx`.

use r2smt_ir::expr::{Expr, Var};

use crate::machine::{ArithKind, LastArith};

/// Translate a flag suffix (`"z"`, `"c"`, `"s"`, `"o"`, `"p"`) into
/// the canonical [`Var`] used by the rest of the lifter. Returns
/// `None` for suffixes the named-flag table does not cover —
/// parametric `$cN` / `$bN` go through `flag_token_to_expr_in_ctx`,
/// and lone `$b` / `$<digit>` overflow indices are intentionally
/// surfaced as `None` so the slicer falls back to the per-mnemonic
/// handler.
#[must_use]
pub fn flag_token_to_var(suffix: &str) -> Option<Var> {
    let name = match suffix {
        "z" => "ZF",
        "c" => "CF",
        "s" => "SF",
        "o" => "OF",
        "p" => "PF",
        _ => return None,
    };
    Some(Var::new(name, 1))
}

/// Convenience wrapper: build an `Expr::Var` over the flag's 1-bit
/// representation. Same coverage as [`flag_token_to_var`] — no
/// parametric forms.
#[must_use]
pub fn flag_token_to_expr(suffix: &str) -> Option<Expr> {
    flag_token_to_var(suffix).map(Expr::Var)
}

/// Translate any ESIL flag suffix to a 1-bit `Expr`, consulting the
/// machine's `LastArith` snapshot when the token is parametric
/// (`$cN` / `$bN`). Returns `None` for suffixes outside the modelled
/// set so the caller can surface `UnsupportedFlag` cleanly.
///
/// ## Formula references
///
/// - `$cN` — carry *into* bit `N+1` after the last `Add`. Standard
///   XOR-of-carries derivation: `lhs XOR rhs` is the "sum with no
///   carries". The bit at position `N+1` of the actual sum differs
///   from `(lhs XOR rhs)[N+1]` exactly when a carry propagated into
///   `N+1`. See Intel SDM Vol. 1 §4.5, ARM ARM §C5.1 (PSTATE.C).
/// - `$bN` — borrow *out of* bit `N` after the last `Sub`. Dual
///   derivation: `(lhs - rhs)[N+1]` differs from `(lhs XOR rhs)[N+1]`
///   when a borrow propagated into `N+1`. The "x86 borrow polarity"
///   convention matches what the per-mnemonic lifter emits for
///   `cmp` / `sub`.
/// - `$0..$15` — radare2-specific NZCV bit indices whose semantics
///   are not stable across builds. Return `None` so the machine
///   surfaces `UnsupportedFlag` and the slicer falls back.
#[must_use]
pub(crate) fn flag_token_to_expr_in_ctx(
    suffix: &str,
    last_arith: Option<&LastArith>,
) -> Option<Expr> {
    // Fast path: legacy named flags ($z, $c, $s, $o, $p) keep going
    // through the canonical-name table even when an arith snapshot
    // is available — the stack machine's $z/$s `derive_*_flag`
    // helpers already cover the "synthesise from last arith" case,
    // and the slicer's SSA pass treats a free `ZF` / `CF` as an input
    // when the flag wasn't defined inside the slice, which is sound.
    if let Some(var) = flag_token_to_var(suffix) {
        return Some(Expr::Var(var));
    }
    if let Some(n) = parse_parametric_suffix(suffix, 'c')
        && let Some(arith) = last_arith
        && matches!(arith.kind, ArithKind::Add)
    {
        return carry_bit_expr(arith, n);
    }
    if let Some(n) = parse_parametric_suffix(suffix, 'b')
        && let Some(arith) = last_arith
        && matches!(arith.kind, ArithKind::Sub)
    {
        return borrow_bit_expr(arith, n);
    }
    None
}

/// Parse `cN` / `bN` (where `prefix` is the leading char) into the
/// numeric bit index. Returns `None` for unmatched prefixes, empty
/// digits, or numeric overflow.
fn parse_parametric_suffix(suffix: &str, prefix: char) -> Option<u8> {
    let rest = suffix.strip_prefix(prefix)?;
    if rest.is_empty() {
        return None;
    }
    rest.parse::<u8>().ok()
}

/// `$cN` — carry into bit `N+1` after the last `Add`. Returns `None`
/// when the bit index would land outside the result width (cannot
/// extract bit `N+1` from a `bits`-wide value).
fn carry_bit_expr(arith: &LastArith, n: u8) -> Option<Expr> {
    let bit = n.checked_add(1)?;
    if bit >= arith.bits {
        return None;
    }
    Some(Expr::ne(
        Expr::extract(arith.result.clone(), bit, bit),
        Expr::extract(Expr::bv_xor(arith.lhs.clone(), arith.rhs.clone()), bit, bit),
    ))
}

/// `$bN` — borrow into bit `N+1` after the last `Sub`. Same XOR-of-
/// carries derivation as `$cN`, but the "result" is the difference
/// `lhs - rhs` rather than `lhs + rhs`.
fn borrow_bit_expr(arith: &LastArith, n: u8) -> Option<Expr> {
    let bit = n.checked_add(1)?;
    if bit >= arith.bits {
        return None;
    }
    Some(Expr::ne(
        Expr::extract(arith.result.clone(), bit, bit),
        Expr::extract(Expr::bv_xor(arith.lhs.clone(), arith.rhs.clone()), bit, bit),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn dummy_add(bits: u8) -> LastArith {
        LastArith {
            kind: ArithKind::Add,
            lhs: Expr::Var(Var::new("a", bits)),
            rhs: Expr::Var(Var::new("b", bits)),
            result: Expr::add(
                Expr::Var(Var::new("a", bits)),
                Expr::Var(Var::new("b", bits)),
            ),
            bits,
        }
    }

    fn dummy_sub(bits: u8) -> LastArith {
        LastArith {
            kind: ArithKind::Sub,
            lhs: Expr::Var(Var::new("a", bits)),
            rhs: Expr::Var(Var::new("b", bits)),
            result: Expr::sub(
                Expr::Var(Var::new("a", bits)),
                Expr::Var(Var::new("b", bits)),
            ),
            bits,
        }
    }

    #[test]
    fn canonical_flag_names_round_trip() {
        for (suffix, name) in [
            ("z", "ZF"),
            ("c", "CF"),
            ("s", "SF"),
            ("o", "OF"),
            ("p", "PF"),
        ] {
            let var = flag_token_to_var(suffix).expect("known flag");
            assert_eq!(var.name, name);
            assert_eq!(var.bits, 1);
        }
    }

    #[test]
    fn unknown_suffix_returns_none() {
        assert!(flag_token_to_var("xyz").is_none());
        assert!(flag_token_to_var("b").is_none());
    }

    #[test]
    fn carry_bit_after_add_emits_extract_difference() {
        let arith = dummy_add(32);
        let expr = flag_token_to_expr_in_ctx("c5", Some(&arith)).expect("c5 after add");
        // The expression must compare two 1-bit extracts (bit 6 of
        // the sum vs bit 6 of `lhs ^ rhs`).
        match expr {
            Expr::Ne(_, _) => {}
            other => panic!("expected Ne for $cN, got {other:?}"),
        }
    }

    #[test]
    fn borrow_bit_after_sub_emits_extract_difference() {
        let arith = dummy_sub(32);
        let expr = flag_token_to_expr_in_ctx("b3", Some(&arith)).expect("b3 after sub");
        match expr {
            Expr::Ne(_, _) => {}
            other => panic!("expected Ne for $bN, got {other:?}"),
        }
    }

    #[test]
    fn parametric_overflow_digit_returns_none() {
        // `$0..$15` are radare2-specific NZCV-bit indices; not
        // modelled. Without a leading `c`/`b` prefix the parser
        // bails out and the machine surfaces UnsupportedFlag.
        let arith = dummy_add(32);
        assert!(flag_token_to_expr_in_ctx("0", Some(&arith)).is_none());
        assert!(flag_token_to_expr_in_ctx("15", Some(&arith)).is_none());
    }

    #[test]
    fn parametric_without_context_returns_none() {
        // No last_arith snapshot ⇒ no formula to emit.
        assert!(flag_token_to_expr_in_ctx("c5", None).is_none());
        assert!(flag_token_to_expr_in_ctx("b3", None).is_none());
    }

    #[test]
    fn carry_bit_after_sub_does_not_fire() {
        // `$cN` requires an Add snapshot; after a Sub it must return
        // None so the machine surfaces UnsupportedFlag rather than
        // emit a nonsensical formula.
        let arith = dummy_sub(32);
        assert!(flag_token_to_expr_in_ctx("c5", Some(&arith)).is_none());
    }

    #[test]
    fn carry_bit_out_of_width_returns_none() {
        let arith = dummy_add(8);
        // Bit index 7 wants to read bit 8 of the result — out of width.
        assert!(flag_token_to_expr_in_ctx("c7", Some(&arith)).is_none());
    }
}
