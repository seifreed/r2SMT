//! Bounded simple-diamond Φ-merge lowering (`lower_merge` /
//! `fold_arm` / `subst_expr`), extracted from `lift.rs`.

use std::collections::HashMap;

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::Instruction;
use r2smt_ir::stmt::IrStmt;

use crate::slice::SliceMerge;

use super::{LiftCtx, lift_branch_condition};

/// Lower one bounded simple-diamond Φ-merge into the context.
///
/// On success the emitted shape is, in execution order:
/// 1. the head-condition definitions (so the selector's flag /
///    register reads resolve);
/// 2. one `dst := Ite(head_condition, taken_value, fallthrough_value)`
///    per merged register.
///
/// Both arms are folded to closed expressions *before* anything is
/// pushed, so a non-foldable arm (memory / call / unsupported survived
/// detection) emits nothing at all and the merged register degrades to
/// a sound free SSA input via its later use. Polarity comes verbatim
/// from the head [`BranchCandidate`]: `taken_arm` feeds the `then`
/// branch (selector true), `fallthrough_arm` the `else` branch.
pub(super) fn lower_merge(ctx: &mut LiftCtx, merge: &SliceMerge, arch: Arch) {
    let Some(taken_env) = fold_arm(&merge.taken_arm, arch) else {
        return;
    };
    let Some(fallthrough_env) = fold_arm(&merge.fallthrough_arm, arch) else {
        return;
    };
    for insn in &merge.head_instructions {
        ctx.lift_instruction(insn);
    }
    let cond = lift_branch_condition(&merge.head, arch);
    for mv in &merge.merged {
        let then_expr = taken_env
            .get(&mv.name)
            .cloned()
            .unwrap_or_else(|| Expr::var(mv.name.clone(), mv.bits));
        let else_expr = fallthrough_env
            .get(&mv.name)
            .cloned()
            .unwrap_or_else(|| Expr::var(mv.name.clone(), mv.bits));
        ctx.stmts.push(IrStmt::Assign {
            dst: Var::new(mv.name.clone(), mv.bits),
            src: Expr::Ite {
                cond: Box::new(cond.clone()),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            },
        });
    }
}

/// Symbolically fold a straight-line arm into a `register → value`
/// environment by lifting it in isolation and inlining each
/// definition into subsequent reads.
///
/// Returns `None` when the arm lifts to anything other than
/// [`IrStmt::Assign`] / [`IrStmt::Nop`] (memory, call, or unsupported
/// that slipped past detection) — the caller then emits nothing,
/// keeping the result sound.
fn fold_arm(arm: &[Instruction], arch: Arch) -> Option<HashMap<String, Expr>> {
    let mut actx = LiftCtx::new(arch);
    for insn in arm {
        actx.lift_instruction(insn);
    }
    let mut env: HashMap<String, Expr> = HashMap::new();
    for stmt in &actx.stmts {
        match stmt {
            IrStmt::Assign { dst, src } => {
                let folded = subst_expr(src, &env);
                env.insert(dst.name.clone(), folded);
            }
            IrStmt::Nop => {}
            IrStmt::LoadMem { .. } | IrStmt::StoreMem { .. } | IrStmt::Unsupported { .. } => {
                return None;
            }
        }
    }
    Some(env)
}

// Exhaustive structural recursion over every `Expr` variant —
// substitutes each `Var` read with its folded arm definition. The
// body carries no domain logic, only the closed-enum mapping the
// CLAUDE.md exhaustive-dispatch exception covers; it grows linearly
// with `Expr`'s arms, not with behaviour.
#[allow(clippy::too_many_lines)]
fn subst_expr(expr: &Expr, env: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Var(v) => env
            .get(&v.name)
            .cloned()
            .unwrap_or_else(|| Expr::Var(v.clone())),
        Expr::Const { value, bits } => Expr::Const {
            value: *value,
            bits: *bits,
        },
        Expr::Add(a, b) => Expr::add(subst_expr(a, env), subst_expr(b, env)),
        Expr::Sub(a, b) => Expr::sub(subst_expr(a, env), subst_expr(b, env)),
        Expr::Mul(a, b) => Expr::mul(subst_expr(a, env), subst_expr(b, env)),
        Expr::UDiv(a, b) => Expr::udiv(subst_expr(a, env), subst_expr(b, env)),
        Expr::URem(a, b) => Expr::urem(subst_expr(a, env), subst_expr(b, env)),
        Expr::SDiv(a, b) => Expr::sdiv(subst_expr(a, env), subst_expr(b, env)),
        Expr::SRem(a, b) => Expr::srem(subst_expr(a, env), subst_expr(b, env)),
        Expr::And(a, b) => Expr::bv_and(subst_expr(a, env), subst_expr(b, env)),
        Expr::Or(a, b) => Expr::bv_or(subst_expr(a, env), subst_expr(b, env)),
        Expr::Xor(a, b) => Expr::bv_xor(subst_expr(a, env), subst_expr(b, env)),
        Expr::Shl(a, b) => Expr::shl(subst_expr(a, env), subst_expr(b, env)),
        Expr::LShr(a, b) => Expr::lshr(subst_expr(a, env), subst_expr(b, env)),
        Expr::AShr(a, b) => Expr::ashr(subst_expr(a, env), subst_expr(b, env)),
        Expr::Eq(a, b) => Expr::eq(subst_expr(a, env), subst_expr(b, env)),
        Expr::Ne(a, b) => Expr::ne(subst_expr(a, env), subst_expr(b, env)),
        Expr::Ult(a, b) => Expr::ult(subst_expr(a, env), subst_expr(b, env)),
        Expr::Ule(a, b) => Expr::ule(subst_expr(a, env), subst_expr(b, env)),
        Expr::Slt(a, b) => Expr::slt(subst_expr(a, env), subst_expr(b, env)),
        Expr::Sle(a, b) => Expr::sle(subst_expr(a, env), subst_expr(b, env)),
        Expr::BoolAnd(a, b) => Expr::bool_and(subst_expr(a, env), subst_expr(b, env)),
        Expr::BoolOr(a, b) => Expr::bool_or(subst_expr(a, env), subst_expr(b, env)),
        Expr::BoolNot(inner) => Expr::bool_not(subst_expr(inner, env)),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ite {
            cond: Box::new(subst_expr(cond, env)),
            then_expr: Box::new(subst_expr(then_expr, env)),
            else_expr: Box::new(subst_expr(else_expr, env)),
        },
        Expr::Extract { src, hi, lo } => Expr::extract(subst_expr(src, env), *hi, *lo),
        Expr::Concat { high, low } => Expr::concat(subst_expr(high, env), subst_expr(low, env)),
        Expr::ZeroExtend { src, to_bits } => Expr::zero_ext(subst_expr(src, env), *to_bits),
        Expr::SignExtend { src, to_bits } => Expr::sign_ext(subst_expr(src, env), *to_bits),
        Expr::Unknown(hint) => Expr::Unknown(hint.clone()),
    }
}
