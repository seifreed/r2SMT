//! Bounded simple-diamond Φ-merge lowering (`lower_merge` /
//! `fold_arm` / `subst_expr`), extracted from `lift.rs`.

use std::collections::HashMap;

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::Instruction;
use r2smt_ir::stmt::IrStmt;

use crate::slice::SliceMerge;

use super::{LiftCtx, lift_branch_condition};

/// Folded representation of a straight-line arm: the register
/// definitions it produces and the memory stores it performs, each
/// resolved against the arm's own earlier definitions.
struct ArmFold {
    regs: HashMap<String, Expr>,
    stores: Vec<ArmStore>,
}

/// One memory store performed by an arm, folded to closed expressions.
struct ArmStore {
    address: Expr,
    value: Expr,
    bits: u16,
}

/// Lower one bounded simple-diamond Φ-merge into the context.
///
/// On success the emitted shape is, in execution order:
/// 1. the head-condition definitions (so the selector's flag /
///    register reads resolve);
/// 2. one `dst := Ite(head_condition, taken_value, fallthrough_value)`
///    per merged register;
/// 3. one conditional `store addr := Ite(head_condition, taken_value,
///    fallthrough_value)` per address written by either arm (the value
///    on the non-storing side is the pre-diamond byte, loaded into a
///    temp, so the store is a no-op on that path).
///
/// Both arms are folded to closed expressions *before* anything is
/// pushed, so a non-foldable arm (an arm-local load, a call, or an
/// unsupported lowering that slipped past detection) emits nothing at
/// all and the merged register/memory degrades to a sound free SSA
/// input / free byte via its later use. Polarity comes verbatim from
/// the head [`BranchCandidate`]: `taken_arm` feeds the `then` branch
/// (selector true), `fallthrough_arm` the `else` branch.
pub(super) fn lower_merge(ctx: &mut LiftCtx, merge: &SliceMerge, arch: Arch) {
    let Some(taken) = fold_arm(&merge.taken_arm, arch) else {
        return;
    };
    let Some(fallthrough) = fold_arm(&merge.fallthrough_arm, arch) else {
        return;
    };
    for insn in &merge.head_instructions {
        ctx.lift_instruction(insn);
    }
    let cond = lift_branch_condition(&merge.head, arch);
    for mv in &merge.merged {
        let then_expr = taken
            .regs
            .get(&mv.name)
            .cloned()
            .unwrap_or_else(|| Expr::var(mv.name.clone(), mv.bits));
        let else_expr = fallthrough
            .regs
            .get(&mv.name)
            .cloned()
            .unwrap_or_else(|| Expr::var(mv.name.clone(), mv.bits));
        ctx.stmts.push(IrStmt::Assign {
            dst: Var::new(mv.name.clone(), mv.bits),
            src: ite(&cond, then_expr, else_expr),
        });
    }
    lower_merge_memory(ctx, merge, &cond, &taken.stores, &fallthrough.stores);
}

/// Emit one conditional store per address written by either arm. The
/// byte-granular encoder resolves address equality inside its `Ite`
/// chain, so this is sound even when two arms write structurally
/// distinct addresses that alias at runtime.
fn lower_merge_memory(
    ctx: &mut LiftCtx,
    merge: &SliceMerge,
    cond: &Expr,
    taken_stores: &[ArmStore],
    fallthrough_stores: &[ArmStore],
) {
    // Union of stored addresses, deterministic order (by rendering).
    let mut addrs: Vec<(Expr, u16)> = Vec::new();
    for s in taken_stores.iter().chain(fallthrough_stores.iter()) {
        if !addrs.iter().any(|(a, _)| *a == s.address) {
            addrs.push((s.address.clone(), s.bits));
        }
    }
    addrs.sort_by_key(|(a, _)| a.to_string());

    let last_val = |stores: &[ArmStore], addr: &Expr| {
        stores
            .iter()
            .rev()
            .find(|s| s.address == *addr)
            .map(|s| s.value.clone())
    };

    // Pass 1: for an address written by only one arm, load the
    // pre-diamond byte into a temp so the other path stores it back
    // unchanged. All temps read memory before any conditional store, so
    // they observe the pre-diamond state regardless of aliasing.
    let mut keep: HashMap<String, Expr> = HashMap::new();
    for (addr, bits) in &addrs {
        let in_taken = last_val(taken_stores, addr).is_some();
        let in_fallthrough = last_val(fallthrough_stores, addr).is_some();
        if in_taken != in_fallthrough {
            let temp = ctx.new_temp(merge.head.address, *bits);
            ctx.stmts.push(IrStmt::LoadMem {
                dst: temp.clone(),
                address: addr.clone(),
                bits: *bits,
            });
            keep.insert(addr.to_string(), Expr::Var(temp));
        }
    }

    // Pass 2: one conditional store per address.
    for (addr, bits) in &addrs {
        let preserved = keep.get(&addr.to_string()).cloned();
        let then_v = last_val(taken_stores, addr).or_else(|| preserved.clone());
        let else_v = last_val(fallthrough_stores, addr).or_else(|| preserved.clone());
        let (Some(then_v), Some(else_v)) = (then_v, else_v) else {
            continue;
        };
        ctx.stmts.push(IrStmt::StoreMem {
            address: addr.clone(),
            value: ite(cond, then_v, else_v),
            bits: *bits,
        });
    }
}

/// Build `Ite(cond, then_expr, else_expr)`.
fn ite(cond: &Expr, then_expr: Expr, else_expr: Expr) -> Expr {
    Expr::Ite {
        cond: Box::new(cond.clone()),
        then_expr: Box::new(then_expr),
        else_expr: Box::new(else_expr),
    }
}

/// Symbolically fold a straight-line arm into its register definitions
/// and memory stores by lifting it in isolation and inlining each
/// definition into subsequent reads.
///
/// Returns `None` when the arm performs an arm-local load, stores to an
/// unresolved (`Expr::Unknown`) address, or lifts to a call / unsupported
/// construct that slipped past detection — the caller then emits nothing,
/// keeping the result sound.
fn fold_arm(arm: &[Instruction], arch: Arch) -> Option<ArmFold> {
    let mut actx = LiftCtx::new(arch);
    for insn in arm {
        actx.lift_instruction(insn);
    }
    let mut regs: HashMap<String, Expr> = HashMap::new();
    let mut stores: Vec<ArmStore> = Vec::new();
    for stmt in &actx.stmts {
        match stmt {
            IrStmt::Assign { dst, src } => {
                let folded = subst_expr(src, &regs);
                regs.insert(dst.name.clone(), folded);
            }
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => {
                let address = subst_expr(address, &regs);
                if expr_has_unknown(&address) {
                    return None;
                }
                let value = subst_expr(value, &regs);
                stores.retain(|s| s.address != address);
                stores.push(ArmStore {
                    address,
                    value,
                    bits: *bits,
                });
            }
            IrStmt::Nop => {}
            IrStmt::LoadMem { .. } | IrStmt::Unsupported { .. } => {
                return None;
            }
        }
    }
    Some(ArmFold { regs, stores })
}

/// Whether an expression contains an [`Expr::Unknown`] anywhere — an
/// unresolved store address must not be merged, since the byte model
/// would let it alias arbitrary memory.
// `match_same_arms`: the floating-point arms deliberately mirror the
// bit-vector arms' recursion but are kept structurally separate so the
// FP / BV split stays legible; merging them into the 20-variant BV
// groups would obscure that intent for no behavioural gain.
#[allow(clippy::match_same_arms)]
fn expr_has_unknown(expr: &Expr) -> bool {
    match expr {
        Expr::Unknown(_) => true,
        Expr::Var(_) | Expr::Const { .. } => false,
        Expr::BoolNot(a)
        | Expr::Extract { src: a, .. }
        | Expr::ZeroExtend { src: a, .. }
        | Expr::SignExtend { src: a, .. } => expr_has_unknown(a),
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
        | Expr::Concat { high: a, low: b } => expr_has_unknown(a) || expr_has_unknown(b),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => expr_has_unknown(cond) || expr_has_unknown(then_expr) || expr_has_unknown(else_expr),
        Expr::FAdd(a, b, _) | Expr::FSub(a, b, _) | Expr::FMul(a, b, _) | Expr::FDiv(a, b, _) => {
            expr_has_unknown(a) || expr_has_unknown(b)
        }
        Expr::FEq(a, b) | Expr::FLt(a, b) | Expr::FLe(a, b) => {
            expr_has_unknown(a) || expr_has_unknown(b)
        }
        Expr::FIsNaN(a) | Expr::FpToIeeeBv(a) | Expr::FSqrt(a, _) => expr_has_unknown(a),
        Expr::FpConst { .. } => false,
        Expr::BvToFp { src, .. }
        | Expr::FpToSbv { src, .. }
        | Expr::SbvToFp { src, .. }
        | Expr::FpToFp { src, .. } => expr_has_unknown(src),
    }
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
        Expr::Ror(a, b) => Expr::ror(subst_expr(a, env), subst_expr(b, env)),
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
        Expr::FAdd(a, b, rm) => Expr::FAdd(
            Box::new(subst_expr(a, env)),
            Box::new(subst_expr(b, env)),
            *rm,
        ),
        Expr::FSub(a, b, rm) => Expr::FSub(
            Box::new(subst_expr(a, env)),
            Box::new(subst_expr(b, env)),
            *rm,
        ),
        Expr::FMul(a, b, rm) => Expr::FMul(
            Box::new(subst_expr(a, env)),
            Box::new(subst_expr(b, env)),
            *rm,
        ),
        Expr::FDiv(a, b, rm) => Expr::FDiv(
            Box::new(subst_expr(a, env)),
            Box::new(subst_expr(b, env)),
            *rm,
        ),
        Expr::FEq(a, b) => Expr::FEq(Box::new(subst_expr(a, env)), Box::new(subst_expr(b, env))),
        Expr::FLt(a, b) => Expr::FLt(Box::new(subst_expr(a, env)), Box::new(subst_expr(b, env))),
        Expr::FLe(a, b) => Expr::FLe(Box::new(subst_expr(a, env)), Box::new(subst_expr(b, env))),
        Expr::FIsNaN(a) => Expr::FIsNaN(Box::new(subst_expr(a, env))),
        Expr::FSqrt(a, rm) => Expr::FSqrt(Box::new(subst_expr(a, env)), *rm),
        Expr::FpConst { .. } => expr.clone(),
        Expr::BvToFp { src, ebits, sbits } => Expr::BvToFp {
            src: Box::new(subst_expr(src, env)),
            ebits: *ebits,
            sbits: *sbits,
        },
        Expr::FpToIeeeBv(src) => Expr::FpToIeeeBv(Box::new(subst_expr(src, env))),
        Expr::FpToSbv { src, rm, bits } => Expr::FpToSbv {
            src: Box::new(subst_expr(src, env)),
            rm: *rm,
            bits: *bits,
        },
        Expr::SbvToFp {
            src,
            rm,
            ebits,
            sbits,
        } => Expr::SbvToFp {
            src: Box::new(subst_expr(src, env)),
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
            src: Box::new(subst_expr(src, env)),
            rm: *rm,
            ebits: *ebits,
            sbits: *sbits,
        },
    }
}
