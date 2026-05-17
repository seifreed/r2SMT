//! Pre-solver optimization pass over an [`SsaLiftedSlice`].
//!
//! r2SMT lifts raw radare2 disassembly / ESIL, so the SSA form carries
//! the instruction-level noise that a decompiler IR (e.g. Hex-Rays
//! microcode) would already have cleaned up: register-shuffle copy
//! chains, constant moves, and dead flag computations the branch never
//! reads. This pass collapses that noise *before* the formula reaches
//! the SMT backend.
//!
//! It deliberately does **not** duplicate Z3's algebraic normalisation
//! (`som` / `blast_eq_value` / `propagate-values` / `ctx-simplify`,
//! applied in `r2smt-smt`). Its value is structural and lands *before*
//! encoding: it shrinks `condition`, drops dead `Expr::Unknown`-bearing
//! computations, and reduces the free-input set — all of which feed the
//! confidence ladder in `r2smt-core` (which is computed pre-Z3).
//!
//! Every rewrite is semantics-preserving on the branch condition:
//! constant folding and the algebraic identities come from
//! [`r2smt_ir::simplify_expr`]; copy / constant propagation is sound
//! because SSA gives each name exactly one definition; dead-code
//! elimination only removes definitions the (transitively reduced)
//! condition and the slice's memory side effects never reference.

use std::collections::{BTreeMap, HashMap, HashSet};

use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::simplify_expr;
use r2smt_ir::stmt::IrStmt;

use crate::convert::SsaLiftedSlice;

/// Upper bound on optimize/propagate/DCE passes. The transform is
/// monotone-shrinking (each pass only inlines or removes), so it
/// reaches a fixed point quickly; the cap is a host-side guardrail
/// against a pathological slice, never expected to bind in practice.
const OPTIMIZE_MAX_PASSES: usize = 16;

/// Recursion guard for [`substitute`]. SSA definitions are acyclic so
/// substitution cannot loop, but a deeply nested expression should not
/// be able to blow the stack.
const SUBST_MAX_DEPTH: usize = 256;

/// Optimize a slice for the solver: constant-fold, copy/constant
/// propagate, and eliminate dead definitions, then recompute the
/// `defs` / `inputs` book-keeping so downstream confidence and
/// pretty-printing stay correct.
///
/// Pure: the input is not mutated. Idempotent:
/// `optimize_slice(&optimize_slice(&s)) == optimize_slice(&s)`.
#[must_use]
pub fn optimize_slice(slice: &SsaLiftedSlice) -> SsaLiftedSlice {
    let mut statements = slice.statements.clone();
    let mut condition = slice.condition.clone();

    for _ in 0..OPTIMIZE_MAX_PASSES {
        let before_stmts = statements.clone();
        let before_cond = condition.clone();

        // 1. Constant folding / algebraic identities on every expr.
        for stmt in &mut statements {
            simplify_stmt_exprs(stmt);
        }
        condition = simplify_expr(&condition);

        // 2 + 3. Copy / constant propagation: inline definitions whose
        // RHS is a bare variable or a constant. Bounded blow-up (the
        // substituted node is a leaf), and the dominant obfuscation
        // shape (register-shuffle `mov` chains) is exactly this.
        let inline = build_inline_map(&statements);
        if !inline.is_empty() {
            for stmt in &mut statements {
                substitute_stmt(stmt, &inline);
            }
            condition = substitute(&condition, &inline, 0);
        }

        // 4. Dead-code / dead-flag elimination.
        let live = live_names(&statements, &condition);
        statements.retain(|stmt| stmt_is_live(stmt, &live));

        if statements == before_stmts && condition == before_cond {
            break;
        }
    }

    // 5. Recompute defs / inputs to match the reduced statement list,
    // mirroring `ssa_convert`'s final assembly (names are already
    // versioned, so no re-versioning — only re-collection).
    let (defs, inputs) = recompute_defs_inputs(&statements, &condition);

    SsaLiftedSlice {
        branch: slice.branch.clone(),
        statements,
        condition,
        status: slice.status.clone(),
        treat_truncation_as_inputs: slice.treat_truncation_as_inputs,
        inputs,
        defs,
        arch: slice.arch,
    }
}

/// Apply [`simplify_expr`] to every expression position of `stmt`.
fn simplify_stmt_exprs(stmt: &mut IrStmt) {
    match stmt {
        IrStmt::Assign { src, .. } => *src = simplify_expr(src),
        IrStmt::LoadMem { address, .. } => *address = simplify_expr(address),
        IrStmt::StoreMem { address, value, .. } => {
            *address = simplify_expr(address);
            *value = simplify_expr(value);
        }
        IrStmt::Unsupported { .. } | IrStmt::Nop => {}
    }
}

/// Build the propagation map: SSA name → inlinable RHS. Only
/// definitions whose RHS is a leaf (`Var` or `Const`) are inlined so
/// the substituted formula cannot grow.
fn build_inline_map(statements: &[IrStmt]) -> HashMap<String, Expr> {
    let mut map = HashMap::new();
    for stmt in statements {
        if let IrStmt::Assign { dst, src } = stmt
            && matches!(src, Expr::Var(_) | Expr::Const { .. })
        {
            map.insert(dst.name.clone(), src.clone());
        }
    }
    map
}

fn substitute_stmt(stmt: &mut IrStmt, map: &HashMap<String, Expr>) {
    match stmt {
        IrStmt::Assign { src, .. } => *src = substitute(src, map, 0),
        IrStmt::LoadMem { address, .. } => *address = substitute(address, map, 0),
        IrStmt::StoreMem { address, value, .. } => {
            *address = substitute(address, map, 0);
            *value = substitute(value, map, 0);
        }
        IrStmt::Unsupported { .. } | IrStmt::Nop => {}
    }
}

/// Replace every `Var` whose name is a key of `map` with the mapped
/// expression. One level per call; chained copies resolve across the
/// outer fixed-point loop in [`optimize_slice`].
fn substitute(expr: &Expr, map: &HashMap<String, Expr>, depth: usize) -> Expr {
    if depth >= SUBST_MAX_DEPTH {
        return expr.clone();
    }
    let d = depth + 1;
    match expr {
        Expr::Var(v) => map.get(&v.name).cloned().unwrap_or_else(|| expr.clone()),
        Expr::Const { .. } | Expr::Unknown(_) => expr.clone(),
        Expr::Add(a, b) => Expr::add(substitute(a, map, d), substitute(b, map, d)),
        Expr::Sub(a, b) => Expr::sub(substitute(a, map, d), substitute(b, map, d)),
        Expr::Mul(a, b) => Expr::mul(substitute(a, map, d), substitute(b, map, d)),
        Expr::UDiv(a, b) => Expr::udiv(substitute(a, map, d), substitute(b, map, d)),
        Expr::URem(a, b) => Expr::urem(substitute(a, map, d), substitute(b, map, d)),
        Expr::SDiv(a, b) => Expr::sdiv(substitute(a, map, d), substitute(b, map, d)),
        Expr::SRem(a, b) => Expr::srem(substitute(a, map, d), substitute(b, map, d)),
        Expr::And(a, b) => Expr::bv_and(substitute(a, map, d), substitute(b, map, d)),
        Expr::Or(a, b) => Expr::bv_or(substitute(a, map, d), substitute(b, map, d)),
        Expr::Xor(a, b) => Expr::bv_xor(substitute(a, map, d), substitute(b, map, d)),
        Expr::Shl(a, b) => Expr::shl(substitute(a, map, d), substitute(b, map, d)),
        Expr::LShr(a, b) => Expr::lshr(substitute(a, map, d), substitute(b, map, d)),
        Expr::AShr(a, b) => Expr::ashr(substitute(a, map, d), substitute(b, map, d)),
        Expr::Eq(a, b) => Expr::eq(substitute(a, map, d), substitute(b, map, d)),
        Expr::Ne(a, b) => Expr::ne(substitute(a, map, d), substitute(b, map, d)),
        Expr::Ult(a, b) => Expr::ult(substitute(a, map, d), substitute(b, map, d)),
        Expr::Ule(a, b) => Expr::ule(substitute(a, map, d), substitute(b, map, d)),
        Expr::Slt(a, b) => Expr::slt(substitute(a, map, d), substitute(b, map, d)),
        Expr::Sle(a, b) => Expr::sle(substitute(a, map, d), substitute(b, map, d)),
        Expr::BoolAnd(a, b) => Expr::bool_and(substitute(a, map, d), substitute(b, map, d)),
        Expr::BoolOr(a, b) => Expr::bool_or(substitute(a, map, d), substitute(b, map, d)),
        Expr::BoolNot(e) => Expr::bool_not(substitute(e, map, d)),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ite {
            cond: Box::new(substitute(cond, map, d)),
            then_expr: Box::new(substitute(then_expr, map, d)),
            else_expr: Box::new(substitute(else_expr, map, d)),
        },
        Expr::Extract { src, hi, lo } => Expr::extract(substitute(src, map, d), *hi, *lo),
        Expr::Concat { high, low } => {
            Expr::concat(substitute(high, map, d), substitute(low, map, d))
        }
        Expr::ZeroExtend { src, to_bits } => Expr::zero_ext(substitute(src, map, d), *to_bits),
        Expr::SignExtend { src, to_bits } => Expr::sign_ext(substitute(src, map, d), *to_bits),
    }
}

/// Names transitively required to evaluate the branch condition or the
/// slice's memory side effects. Computed by backward fixed point.
fn live_names(statements: &[IrStmt], condition: &Expr) -> HashSet<String> {
    let mut live: HashSet<String> = HashSet::new();
    collect_var_names(condition, &mut live);
    // StoreMem is an observable side effect — it and the values it
    // reads must stay live regardless of the condition.
    for stmt in statements {
        if let IrStmt::StoreMem { address, value, .. } = stmt {
            collect_var_names(address, &mut live);
            collect_var_names(value, &mut live);
        }
    }
    loop {
        let mut changed = false;
        for stmt in statements {
            let (dst, exprs): (Option<&Var>, Vec<&Expr>) = match stmt {
                IrStmt::Assign { dst, src } => (Some(dst), vec![src]),
                IrStmt::LoadMem { dst, address, .. } => (Some(dst), vec![address]),
                IrStmt::StoreMem { .. } | IrStmt::Unsupported { .. } | IrStmt::Nop => {
                    (None, Vec::new())
                }
            };
            if let Some(dst) = dst
                && live.contains(&dst.name)
            {
                for e in exprs {
                    let mut found = HashSet::new();
                    collect_var_names(e, &mut found);
                    for name in found {
                        if live.insert(name) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    live
}

/// `true` when `stmt` must be kept after dead-code elimination.
fn stmt_is_live(stmt: &IrStmt, live: &HashSet<String>) -> bool {
    match stmt {
        IrStmt::Assign { dst, .. } | IrStmt::LoadMem { dst, .. } => live.contains(&dst.name),
        // StoreMem is a side effect; Unsupported is a conservative
        // marker we must not silently drop. Nop carries no information.
        IrStmt::StoreMem { .. } | IrStmt::Unsupported { .. } => true,
        IrStmt::Nop => false,
    }
}

fn collect_var_names(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Var(v) => {
            out.insert(v.name.clone());
        }
        Expr::Const { .. } | Expr::Unknown(_) => {}
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
        | Expr::AShr(a, b)
        | Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Ult(a, b)
        | Expr::Ule(a, b)
        | Expr::Slt(a, b)
        | Expr::Sle(a, b)
        | Expr::BoolAnd(a, b)
        | Expr::BoolOr(a, b) => {
            collect_var_names(a, out);
            collect_var_names(b, out);
        }
        Expr::BoolNot(e) => collect_var_names(e, out),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_var_names(cond, out);
            collect_var_names(then_expr, out);
            collect_var_names(else_expr, out);
        }
        Expr::Extract { src, .. } | Expr::ZeroExtend { src, .. } | Expr::SignExtend { src, .. } => {
            collect_var_names(src, out);
        }
        Expr::Concat { high, low } => {
            collect_var_names(high, out);
            collect_var_names(low, out);
        }
    }
}

/// Rebuild `defs` (assigned vars in order) and `inputs` (vars read but
/// never defined in the slice, deduped by name). Mirrors the final
/// assembly in [`crate::convert::ssa_convert`] without re-versioning.
fn recompute_defs_inputs(statements: &[IrStmt], condition: &Expr) -> (Vec<Var>, Vec<Var>) {
    let mut defs: Vec<Var> = Vec::new();
    let mut def_names: HashSet<String> = HashSet::new();
    for stmt in statements {
        if let IrStmt::Assign { dst, .. } | IrStmt::LoadMem { dst, .. } = stmt {
            def_names.insert(dst.name.clone());
            defs.push(dst.clone());
        }
    }
    let mut inputs: BTreeMap<String, Var> = BTreeMap::new();
    let mut visit = |expr: &Expr| {
        let mut seen: Vec<Var> = Vec::new();
        collect_vars(expr, &mut seen);
        for v in seen {
            if !def_names.contains(&v.name) {
                inputs.entry(v.name.clone()).or_insert(v);
            }
        }
    };
    for stmt in statements {
        match stmt {
            IrStmt::Assign { src, .. } => visit(src),
            IrStmt::LoadMem { address, .. } => visit(address),
            IrStmt::StoreMem { address, value, .. } => {
                visit(address);
                visit(value);
            }
            IrStmt::Unsupported { .. } | IrStmt::Nop => {}
        }
    }
    visit(condition);
    (defs, inputs.into_values().collect())
}

fn collect_vars(expr: &Expr, out: &mut Vec<Var>) {
    match expr {
        Expr::Var(v) => out.push(v.clone()),
        Expr::Const { .. } | Expr::Unknown(_) => {}
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
        | Expr::AShr(a, b)
        | Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Ult(a, b)
        | Expr::Ule(a, b)
        | Expr::Slt(a, b)
        | Expr::Sle(a, b)
        | Expr::BoolAnd(a, b)
        | Expr::BoolOr(a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::BoolNot(e) => collect_vars(e, out),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_vars(cond, out);
            collect_vars(then_expr, out);
            collect_vars(else_expr, out);
        }
        Expr::Extract { src, .. } | Expr::ZeroExtend { src, .. } | Expr::SignExtend { src, .. } => {
            collect_vars(src, out);
        }
        Expr::Concat { high, low } => {
            collect_vars(high, out);
            collect_vars(low, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use r2smt_common::{Address, Arch};
    use r2smt_ir::expr::{Expr, Var};
    use r2smt_ir::stmt::IrStmt;
    use r2smt_slicer::condition::{BranchCondition, BranchKind};
    use r2smt_slicer::{BranchCandidate, SliceStatus};

    use super::optimize_slice;
    use crate::convert::SsaLiftedSlice;

    fn slice(statements: Vec<IrStmt>, condition: Expr) -> SsaLiftedSlice {
        SsaLiftedSlice {
            branch: BranchCandidate {
                address: Address(0x40_1000),
                function: Address(0x40_1000),
                block: Address(0x40_1000),
                kind: BranchKind::Jcc,
                mnemonic: "jne".into(),
                condition: BranchCondition::NotEqual,
                formula: String::new(),
                taken_target: None,
                fallthrough_target: None,
                compare_register: None,
                bit_index: None,
                upstream_resolved: None,
                operand_raws: Vec::new(),
                is_thumb: false,
            },
            statements,
            condition,
            status: SliceStatus::Complete,
            treat_truncation_as_inputs: false,
            inputs: Vec::new(),
            defs: Vec::new(),
            arch: Arch::X86_64,
        }
    }

    fn assign(name: &str, bits: u8, src: Expr) -> IrStmt {
        IrStmt::Assign {
            dst: Var::new(name, bits),
            src,
        }
    }

    #[test]
    fn copy_prop_inlines_register_alias() {
        // rax#0 := rsi (free input);  condition: (rax#0 == 0)
        // After copy-prop the def is dead and removed; the condition
        // references the input `rsi` directly.
        let s = slice(
            vec![assign("rax#0", 64, Expr::var("rsi", 64))],
            Expr::eq(Expr::var("rax#0", 64), Expr::konst(0, 64)),
        );
        let o = optimize_slice(&s);
        assert!(o.statements.is_empty(), "dead copy must be removed");
        assert_eq!(
            o.condition,
            Expr::eq(Expr::var("rsi", 64), Expr::konst(0, 64))
        );
        assert_eq!(o.inputs, vec![Var::new("rsi", 64)]);
        assert!(o.defs.is_empty());
    }

    #[test]
    fn const_prop_folds_into_condition() {
        // eax#0 := 5;  condition: (eax#0 == 5)  →  (5 == 5)  →  const 1.
        let s = slice(
            vec![assign("eax#0", 32, Expr::konst(5, 32))],
            Expr::eq(Expr::var("eax#0", 32), Expr::konst(5, 32)),
        );
        let o = optimize_slice(&s);
        assert!(o.statements.is_empty());
        assert_eq!(o.condition, Expr::konst(1, 1));
        assert!(o.inputs.is_empty());
    }

    #[test]
    fn dead_flag_def_is_eliminated() {
        // ZF#0 used by the branch; OF#0 computed but never read.
        let s = slice(
            vec![
                assign(
                    "ZF#0",
                    1,
                    Expr::eq(Expr::var("rax", 64), Expr::konst(0, 64)),
                ),
                assign("OF#0", 1, Expr::var("rcx", 1)),
            ],
            Expr::eq(Expr::var("ZF#0", 1), Expr::konst(0, 1)),
        );
        let o = optimize_slice(&s);
        let def_names: Vec<&str> = o.defs.iter().map(|v| v.name.as_str()).collect();
        assert!(def_names.contains(&"ZF#0"), "live flag kept");
        assert!(!def_names.contains(&"OF#0"), "dead flag removed");
        // `rcx` fed only the dead OF def, so it must not be an input.
        assert!(o.inputs.iter().all(|v| v.name != "rcx"));
        assert!(o.inputs.iter().any(|v| v.name == "rax"));
    }

    #[test]
    fn dead_unknown_stmt_removed_raises_confidence_inputs() {
        // A dead def whose RHS is Unknown (e.g. an unmodelled flag).
        // It is never read by the condition, so eliminating it is
        // sound and removes the Unknown the classifier would have
        // counted (Medium → High).
        let s = slice(
            vec![
                assign("PF#0", 1, Expr::unknown()),
                assign(
                    "ZF#0",
                    1,
                    Expr::eq(Expr::var("rdi", 64), Expr::konst(1, 64)),
                ),
            ],
            Expr::eq(Expr::var("ZF#0", 1), Expr::konst(1, 1)),
        );
        let o = optimize_slice(&s);
        let has_unknown = o.statements.iter().any(|st| {
            matches!(
                st,
                IrStmt::Assign {
                    src: Expr::Unknown(_),
                    ..
                }
            )
        });
        assert!(!has_unknown, "dead Unknown computation must be gone");
    }

    #[test]
    fn optimize_is_idempotent() {
        let s = slice(
            vec![
                assign("t#0", 64, Expr::var("rdx", 64)),
                assign("rax#0", 64, Expr::var("t#0", 64)),
            ],
            Expr::eq(Expr::var("rax#0", 64), Expr::konst(7, 64)),
        );
        let once = optimize_slice(&s);
        let twice = optimize_slice(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn optimize_preserves_condition_semantics_via_known_reduction() {
        // x#0 := rcx; y#0 := x#0; condition: (y#0 - rcx == 0).
        // Copy-prop collapses y#0 → rcx, then simplify folds
        // (rcx - rcx) → 0 and (0 == 0) → const 1.
        let s = slice(
            vec![
                assign("x#0", 64, Expr::var("rcx", 64)),
                assign("y#0", 64, Expr::var("x#0", 64)),
            ],
            Expr::eq(
                Expr::sub(Expr::var("y#0", 64), Expr::var("rcx", 64)),
                Expr::konst(0, 64),
            ),
        );
        let o = optimize_slice(&s);
        assert_eq!(o.condition, Expr::konst(1, 1));
        assert!(o.statements.is_empty());
        assert!(o.inputs.is_empty(), "rcx folded away entirely");
    }

    #[test]
    fn defs_and_inputs_recomputed_after_elimination() {
        // Two defs: one live (feeds condition), one dead. Inputs must
        // list only variables the surviving form actually reads.
        let s = slice(
            vec![
                assign(
                    "live#0",
                    64,
                    Expr::add(Expr::var("arg1", 64), Expr::konst(3, 64)),
                ),
                assign("dead#0", 64, Expr::var("arg2", 64)),
            ],
            Expr::eq(Expr::var("live#0", 64), Expr::konst(9, 64)),
        );
        let o = optimize_slice(&s);
        assert_eq!(
            o.defs.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
            vec!["live#0".to_string()]
        );
        assert_eq!(
            o.inputs.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
            vec!["arg1".to_string()]
        );
    }

    #[test]
    fn store_mem_side_effect_is_never_dropped() {
        // A StoreMem has an observable side effect; DCE must keep it
        // and the values it reads, even though the branch condition
        // does not reference them.
        let s = slice(
            vec![IrStmt::StoreMem {
                address: Expr::var("rbp", 64),
                value: Expr::var("rsi", 64),
                bits: 64,
            }],
            Expr::eq(Expr::var("rdi", 64), Expr::konst(0, 64)),
        );
        let o = optimize_slice(&s);
        assert_eq!(o.statements.len(), 1);
        assert!(matches!(o.statements[0], IrStmt::StoreMem { .. }));
    }
}
