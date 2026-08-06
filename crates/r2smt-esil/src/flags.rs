//! ESIL flag tokens, derived from the flag *context* radare2 keeps.
//!
//! ## Everything here was measured against r2 6.1.8, not inferred
//!
//! The model this file carried until 2026-08-06 was wrong on four
//! independent counts, and none of them could fire: `==` and the whole
//! `$z`/`$s`/`$o`/`$b`/`$c`/`$p` family appear **zero** times among the
//! ESIL strings that lift today, because every real use sits in a string
//! that also carries `:=`, which the parser does not know. So the errors
//! were invisible rather than harmless, and they are exactly what would
//! become reachable the moment `:=` lands.
//!
//! What `ae` actually reports:
//!
//! - **The context is seeded by writes and compares, never by bare
//!   arithmetic.** `ae 0x81,1,-,7,$s` leaves `0xff…80` on the stack and
//!   still answers `$s(7) = 0`; `ae 0x80,rax,=,7,$s` answers `1`, and so
//!   does `ae 1,rax,==,7,$s`. The old model seeded on `apply_binary` —
//!   precisely the operations that do *not* seed — and seeded nothing on
//!   assignment, which is the one that does.
//! - **`$s`/`$c`/`$b`/`$o` take their bit index from the stack**, not
//!   from the token text: `ae 5,0x80,rax,=,7,$s` leaves `[5, 1]`, popping
//!   exactly one. And `$s7` / `$b4` are **not radare2 tokens at all** —
//!   they come back unresolved, verbatim. The old `parse_parametric_suffix`
//!   parsed a syntax radare2 never emits.
//! - **`$c` and `$b` are old-versus-current comparisons and nothing
//!   else**, independent of whether the seeding operation added or
//!   subtracted. That is what deletes the old `ArithKind` and the
//!   `matches!(kind, Add)` gates: they guarded a distinction radare2 does
//!   not make. The previous XOR-of-carries formula is a *different
//!   function*, not a rounding error.
//! - **`$z` masks to the written register's width**: `0x100,al,=,$z` is
//!   `1` where `0x100,rax,=,$z` is `0`.
//!
//! The measured formulas, with `m = ones(min(N+1, w))`:
//!
//! | token | value |
//! |---|---|
//! | `$z` | `(cur & ones(w)) == 0` |
//! | `$s(N)` | `cur[N]` |
//! | `$c(N)` | `(cur & m) <ᵤ (old & m)` |
//! | `$b(N)` | `(old & m) <ᵤ (cur & m)` |
//! | `$o(N)` | `$c(N-1) ⊕ $c(N)` |
//! | `$p` | even parity of the low 8 bits of `cur` |

use r2smt_ir::expr::{Expr, Var};

use crate::machine::FlagCtx;

/// Bits of the low byte `$p` folds over.
const PARITY_BITS: u16 = 8;

/// Canonical `Var` for a named flag token (`$z` → `ZF`, …).
///
/// Kept because these are the names the rest of the pipeline stores
/// flags under; the *values* now come from the context helpers below.
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

/// [`flag_token_to_var`] as a free expression, for callers with no flag
/// context to derive from.
#[must_use]
pub fn flag_token_to_expr(suffix: &str) -> Option<Expr> {
    flag_token_to_var(suffix).map(Expr::Var)
}

/// A condition as a 1-bit value, which is what the ESIL stack carries.
fn as_bit(cond: Expr) -> Expr {
    Expr::Ite {
        cond: Box::new(cond),
        then_expr: Box::new(Expr::konst(1, 1)),
        else_expr: Box::new(Expr::konst(0, 1)),
    }
}

/// `ones(min(n + 1, bits))` at `bits` width.
fn low_mask(n: u16, bits: u16) -> Option<Expr> {
    if bits == 0 || bits > 128 {
        return None;
    }
    let width = u32::from(n.checked_add(1)?.min(bits));
    let value = if width >= 128 {
        u128::MAX
    } else {
        1u128.checked_shl(width)?.checked_sub(1)?
    };
    Some(Expr::konst(value, bits))
}

/// `$z` — the seeding operation's result is zero at its own width.
#[must_use]
pub(crate) fn zero_flag(ctx: &FlagCtx) -> Expr {
    as_bit(Expr::eq(ctx.cur.clone(), Expr::konst(0, ctx.bits)))
}

/// `$s` — bit `n` of the result.
#[must_use]
pub(crate) fn sign_flag(ctx: &FlagCtx, n: u16) -> Option<Expr> {
    if n >= ctx.bits {
        return None;
    }
    Some(Expr::extract(ctx.cur.clone(), n, n))
}

/// `$c` — carry out of bit `n`, as the unsigned wrap of the masked
/// result below the masked original.
#[must_use]
pub(crate) fn carry_flag(ctx: &FlagCtx, n: u16) -> Option<Expr> {
    let mask = low_mask(n, ctx.bits)?;
    Some(as_bit(Expr::ult(
        Expr::bv_and(ctx.cur.clone(), mask.clone()),
        Expr::bv_and(ctx.old.clone(), mask),
    )))
}

/// `$b` — borrow out of bit `n`: the same comparison, the other way
/// round. The two are duals and neither depends on whether the seeding
/// operation added or subtracted.
#[must_use]
pub(crate) fn borrow_flag(ctx: &FlagCtx, n: u16) -> Option<Expr> {
    let mask = low_mask(n, ctx.bits)?;
    Some(as_bit(Expr::ult(
        Expr::bv_and(ctx.old.clone(), mask.clone()),
        Expr::bv_and(ctx.cur.clone(), mask),
    )))
}

/// `$o` — signed overflow at bit `n`: the carry into the sign bit
/// disagreeing with the carry out of it.
#[must_use]
pub(crate) fn overflow_flag(ctx: &FlagCtx, n: u16) -> Option<Expr> {
    let into = carry_flag(ctx, n.checked_sub(1)?)?;
    let out_of = carry_flag(ctx, n)?;
    Some(Expr::bv_xor(into, out_of))
}

/// `$p` — even parity of the result's low byte.
#[must_use]
pub(crate) fn parity_flag(ctx: &FlagCtx) -> Option<Expr> {
    if ctx.bits == 0 {
        return None;
    }
    let top = PARITY_BITS.min(ctx.bits);
    let mut acc: Option<Expr> = None;
    for i in 0..top {
        let bit = Expr::extract(ctx.cur.clone(), i, i);
        acc = Some(match acc.take() {
            None => bit,
            Some(prev) => Expr::bv_xor(prev, bit),
        });
    }
    // Even parity: the flag is set when the count of set bits is even,
    // which is the *complement* of their exclusive or.
    acc.map(|xor| Expr::bv_xor(xor, Expr::konst(1, 1)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn ctx(old: u128, cur: u128, bits: u16) -> FlagCtx {
        FlagCtx {
            old: Expr::konst(old, bits),
            cur: Expr::konst(cur, bits),
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
            assert_eq!(flag_token_to_var(suffix).map(|v| v.name), Some(name.into()));
        }
    }

    #[test]
    fn unknown_suffix_returns_none() {
        assert!(flag_token_to_var("q").is_none());
    }

    #[test]
    fn the_bit_index_is_not_read_from_the_token_text() {
        // `$s7` / `$b4` are not radare2 tokens: `ae 0x80,rax,=,$s7`
        // leaves the literal `$s7` on the stack unresolved. Parsing them
        // modelled a syntax that is never emitted.
        assert!(flag_token_to_var("s7").is_none());
        assert!(flag_token_to_var("b4").is_none());
        assert!(flag_token_to_var("c5").is_none());
    }

    #[test]
    fn sign_flag_past_the_width_declines() {
        assert!(sign_flag(&ctx(0, 0, 8), 8).is_none());
    }

    #[test]
    fn overflow_at_bit_zero_declines() {
        // `$o(0)` would need the carry out of bit -1.
        assert!(overflow_flag(&ctx(0, 0, 8), 0).is_none());
    }
}
