//! Z3 AST → C-style infix pretty-printer.
//!
//! Renders the boolean formula that the solver actually saw, after
//! `crate::solver::solve_branch_with_pretty` ran the Z3 tactic chain
//! over it. Inspired by `MicroSMT/microSMT.py:expr_to_text`, but
//! written against the Z3 0.20 Rust binding's
//! [`z3::ast::Ast`] / [`z3::DeclKind`] APIs.
//!
//! ## Soundness
//!
//! The renderer is a pure tree walk. Any operator we don't recognise
//! falls through to `<unsupported: {decl_name}>` — never a panic,
//! never an `unwrap`. The caller treats the output as a best-effort
//! analyst-facing artifact, not as a normative formula.
//!
//! ## Layout
//!
//! * Booleans / comparisons are wrapped in parentheses so precedence
//!   stays unambiguous when several are composed.
//! * Constants emit `0x<hex>:<bits>` for bit-vectors and `true`/`false`
//!   for booleans.
//! * `Ite(cond, 1, 0)` collapses to just `cond` (same shortcut
//!   `MicroSMT` applies).
//! * `Extract` / `ZeroExt` / `SignExt` / `Concat` delegate to Z3's
//!   built-in `Display` for the subtree because the 0.20 binding
//!   doesn't expose `Z3_get_decl_int_parameter`. The text is correct
//!   SMT-LIB, just less ergonomic — better than fabricating numbers.

use z3::DeclKind;
use z3::ast::{Ast, Bool, Dynamic};

/// Render a `Bool` AST as a C-style infix expression.
#[must_use]
pub fn z3_bool_to_infix(b: &Bool) -> String {
    render_dyn(&Dynamic::from(b.clone()))
}

/// Render any Z3 dynamic node (bool or bit-vector) as infix.
fn render_dyn(node: &Dynamic) -> String {
    // Booleans first — they own the `true` / `false` leaves.
    if let Some(b) = node.as_bool() {
        if let Some(v) = b.as_bool() {
            return if v { "true".into() } else { "false".into() };
        }
    }
    // Bit-vector numeric constant: emit as hex with bit-width suffix.
    if let Some(bv) = node.as_bv() {
        if let Some(value) = bv.as_u64() {
            return format!("0x{value:x}:{bits}", bits = bv.get_size());
        }
        // Symbolic constants (`rax#0`, …) — decl().name() carries the
        // SSA-versioned identifier the encoder declared.
        if bv.is_const() {
            return bv.decl().name();
        }
    }

    // Function applications dispatch by DeclKind. The 0.20 binding
    // panics if we call `decl()` on a non-app, so guard.
    let Ok(decl) = node.safe_decl() else {
        return format!("{node}");
    };
    let kind = decl.kind();
    let kids = node.children();

    match kind {
        DeclKind::True => "true".into(),
        DeclKind::False => "false".into(),
        DeclKind::Eq => bin_infix(&kids, "=="),
        DeclKind::Distinct => bin_infix(&kids, "!="),
        DeclKind::Not => format!("(!{})", render_or_one(&kids, 0)),
        DeclKind::And => join(&kids, "&&"),
        DeclKind::Or => join(&kids, "||"),
        DeclKind::Iff => bin_infix(&kids, "<==>"),
        DeclKind::Implies => bin_infix(&kids, "=>"),
        DeclKind::Xor => bin_infix(&kids, "^^"),
        // BV comparisons.
        DeclKind::Ult => bin_infix(&kids, "<"),
        DeclKind::Uleq => bin_infix(&kids, "<="),
        DeclKind::Ugt => bin_infix(&kids, ">"),
        DeclKind::Ugeq => bin_infix(&kids, ">="),
        DeclKind::Slt => bin_infix(&kids, "<s"),
        DeclKind::Sleq => bin_infix(&kids, "<=s"),
        DeclKind::Sgt => bin_infix(&kids, ">s"),
        DeclKind::Sgeq => bin_infix(&kids, ">=s"),
        // BV arithmetic.
        DeclKind::Badd => bin_infix(&kids, "+"),
        DeclKind::Bsub => bin_infix(&kids, "-"),
        DeclKind::Bmul => bin_infix(&kids, "*"),
        DeclKind::Bsdiv | DeclKind::BsdivI => bin_infix(&kids, "/s"),
        DeclKind::Budiv => bin_infix(&kids, "/"),
        DeclKind::Bsrem | DeclKind::BsremI => bin_infix(&kids, "%s"),
        DeclKind::Burem => bin_infix(&kids, "%"),
        DeclKind::Bsmod | DeclKind::BsmodI => bin_infix(&kids, "mod"),
        DeclKind::Bneg => format!("(-{})", render_or_one(&kids, 0)),
        // BV bitwise.
        DeclKind::Band => bin_infix(&kids, "&"),
        DeclKind::Bor => bin_infix(&kids, "|"),
        DeclKind::Bxor => bin_infix(&kids, "^"),
        DeclKind::Bnot => format!("(~{})", render_or_one(&kids, 0)),
        DeclKind::Bshl => bin_infix(&kids, "<<"),
        DeclKind::Blshr => bin_infix(&kids, ">>"),
        DeclKind::Bashr => bin_infix(&kids, ">>s"),
        // Conditional. Collapse `Ite(cond, 1, 0)` to just `cond`
        // (matches MicroSMT's `expr_to_text` rule for booleans
        // round-tripped through a bv8).
        DeclKind::Ite => render_ite(&kids),
        // Bit-precise ops: the 0.20 binding doesn't expose the
        // integer parameters (hi/lo for extract, n for sign/zero
        // extend), so we delegate to Z3's built-in SMT-LIB printer
        // for those subtrees. The output is still correct, just
        // less compact.
        DeclKind::Extract | DeclKind::SignExt | DeclKind::ZeroExt | DeclKind::Concat => {
            format!("{node}")
        }
        // BV constant (covered above by the `as_u64` path, but a
        // wide constant lands here).
        DeclKind::Bnum => format!("{node}"),
        // Anything else surfaces verbatim — never panic. The
        // `<unsupported>` marker makes it obvious in reports.
        _ => format!("<unsupported: {name}>", name = decl.name()),
    }
}

fn bin_infix(kids: &[Dynamic], op: &str) -> String {
    if kids.len() == 2 {
        format!(
            "({lhs} {op} {rhs})",
            lhs = render_dyn(&kids[0]),
            rhs = render_dyn(&kids[1]),
        )
    } else {
        // Z3 sometimes hands us n-ary applications for `And`/`Or` —
        // we hit that case via `join`. For the strict-binary table
        // above an arity mismatch is an unexpected encoding; surface
        // raw text instead of panicking.
        join(kids, op)
    }
}

fn join(kids: &[Dynamic], op: &str) -> String {
    if kids.is_empty() {
        return op.into();
    }
    let mut out = String::from("(");
    let parts: Vec<String> = kids.iter().map(render_dyn).collect();
    out.push_str(&parts.join(&format!(" {op} ")));
    out.push(')');
    out
}

fn render_or_one(kids: &[Dynamic], idx: usize) -> String {
    kids.get(idx).map_or_else(|| "?".into(), render_dyn)
}

fn render_ite(kids: &[Dynamic]) -> String {
    if kids.len() != 3 {
        // Defensive — Z3 ITE is ternary by construction. Falling
        // back to the SMT-LIB form preserves the information.
        return kids.iter().map(render_dyn).collect::<Vec<_>>().join(" ");
    }
    // `Ite(cond, 1, 0)` over a 1-bit BV collapses to just `cond`.
    if let (Some(then_bv), Some(else_bv)) = (kids[1].as_bv(), kids[2].as_bv())
        && let (Some(t), Some(e)) = (then_bv.as_u64(), else_bv.as_u64())
        && t == 1
        && e == 0
        && then_bv.get_size() == 1
    {
        return render_dyn(&kids[0]);
    }
    format!(
        "(ite {cond} {t} {e})",
        cond = render_dyn(&kids[0]),
        t = render_dyn(&kids[1]),
        e = render_dyn(&kids[2]),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use z3::Solver;
    use z3::ast::{BV as Z3BV, Bool as Z3Bool};

    fn bv(name: &str, size: u32) -> Z3BV {
        Z3BV::new_const(name, size)
    }

    #[test]
    fn bool_true_and_false_render_directly() {
        let solver = Solver::new();
        let _ = solver; // keep ctx alive
        let t = Z3Bool::from_bool(true);
        let f = Z3Bool::from_bool(false);
        assert_eq!(z3_bool_to_infix(&t), "true");
        assert_eq!(z3_bool_to_infix(&f), "false");
    }

    #[test]
    fn bool_eq_renders_double_equal() {
        let _s = Solver::new();
        let a = bv("a", 32);
        let b = bv("b", 32);
        let formula = a.eq(&b);
        assert_eq!(z3_bool_to_infix(&formula), "(a == b)");
    }

    #[test]
    fn bool_and_renders_double_amp() {
        let _s = Solver::new();
        let a = bv("a", 32);
        let b = bv("b", 32);
        let zero = Z3BV::from_u64(0, 32);
        let lhs = a.eq(&zero);
        let rhs = b.eq(&zero);
        let formula = Z3Bool::and(&[&lhs, &rhs]);
        let out = z3_bool_to_infix(&formula);
        assert!(out.contains("&&"), "got {out}");
        assert!(out.contains("(a == 0x0:32)"), "got {out}");
    }

    #[test]
    fn bv_unsigned_lt_uses_bare_op() {
        let _s = Solver::new();
        let a = bv("a", 32);
        let b = bv("b", 32);
        let formula = a.bvult(&b);
        assert_eq!(z3_bool_to_infix(&formula), "(a < b)");
    }

    #[test]
    fn bv_signed_gt_renders_gt_s_suffix() {
        let _s = Solver::new();
        let a = bv("a", 32);
        let b = bv("b", 32);
        let formula = a.bvsgt(&b);
        assert_eq!(z3_bool_to_infix(&formula), "(a >s b)");
    }

    #[test]
    fn bv_arith_add_renders_plus() {
        let _s = Solver::new();
        let a = bv("a", 32);
        let b = bv("b", 32);
        let sum = (&a) + (&b);
        let zero = Z3BV::from_u64(0, 32);
        let formula = sum.eq(&zero);
        let out = z3_bool_to_infix(&formula);
        assert!(out.contains("(a + b)"), "got {out}");
    }

    #[test]
    fn ite_const_const_collapses_to_cond() {
        let _s = Solver::new();
        let a = bv("a", 32);
        let zero = Z3BV::from_u64(0, 32);
        let cond = a.eq(&zero);
        // Build ite(cond, 1:1, 0:1) and compare against 0:1 — the
        // outer Eq must render the cond directly (sans `(ite …)`).
        let one1 = Z3BV::from_u64(1, 1);
        let zero1 = Z3BV::from_u64(0, 1);
        let bool_to_bv = cond.ite(&one1, &zero1);
        // Take that 1-bit value and == it against 1 — the renderer
        // sees Eq(Ite(cond, 1, 0), 1) and collapses the inner Ite.
        let formula = bool_to_bv.eq(&one1);
        let out = z3_bool_to_infix(&formula);
        assert!(
            !out.contains("ite"),
            "ite(cond, 1, 0) must collapse, got: {out}"
        );
        assert!(out.contains("(a == 0x0:32)"), "got {out}");
    }

    #[test]
    fn unsupported_op_renders_marker_not_panic() {
        // Construct an Array select — we don't model arrays so the
        // dispatch falls to the `_` arm. Must not panic and must
        // surface the `<unsupported>` marker.
        use z3::Sort;
        use z3::ast::Array;
        let _s = Solver::new();
        let idx_sort = Sort::bitvector(32);
        let val_sort = Sort::bitvector(8);
        let arr = Array::new_const("mem", &idx_sort, &val_sort);
        let idx = Z3BV::from_u64(0, 32);
        let sel = arr.select(&idx);
        let zero8 = Z3BV::from_u64(0, 8);
        let formula = sel.as_bv().unwrap().eq(&zero8);
        let out = z3_bool_to_infix(&formula);
        assert!(
            out.contains("<unsupported:") || out.contains("select"),
            "expected unsupported or raw SMT-LIB select, got: {out}",
        );
    }
}
