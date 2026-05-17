//! Semantic equivalence oracle for two independent lowerings of the
//! same instruction.
//!
//! The check is **semantic, not syntactic**: two lowerings agree when
//! no assignment to the (shared) free machine state makes any
//! jointly-defined, identically-named architectural output differ.
//! That question is decided by the existing, P20-contract-gated SMT
//! path — this module only *builds* the query and *classifies* the
//! solver's verdict, so the harness stays a pure-domain crate with no
//! solver dependency (the solve is delegated to the wiring layer,
//! exactly like the production pipeline delegates to an adapter).
//!
//! ## Soundness posture (mission invariant)
//!
//! This harness may flag a disagreement or stay silent; it must never
//! *fabricate* one. Three guards enforce that:
//!
//! 1. If either lowering touches memory ([`IrStmt::LoadMem`] /
//!    [`IrStmt::StoreMem`]) or declines an instruction
//!    ([`IrStmt::Unsupported`]), the lowerings are *not comparable* —
//!    the query is `None` and the caller reports `Inconclusive`. An
//!    [`Expr::Unknown`] is *not* disqualifying on its own: it
//!    forward-taints only the outputs whose value depends on it (see
//!    [`tainted_defs`]), and a tainted output is excluded from the
//!    comparison set so it can never become an independently-free
//!    value that fabricates a difference.
//! 2. Free inputs are tied to one shared machine state through the
//!    project's own register-layout contract
//!    ([`r2smt_slicer::register_layout`]), so two lowerings that read
//!    the same architectural register observe the *same* symbolic
//!    value. Without this, distinct input spellings (`eax` vs the
//!    parent `rax`) would let the solver pick them independently and
//!    fabricate a difference.
//! 3. Only jointly-defined, identically-named, modelled outputs are
//!    compared (the `ZF`/`CF`/`SF` flags — `OF`/`PF` are excluded for
//!    the same reason `r2smt_core` treats them as unmodelled — plus
//!    registers defined under the same name and width on both sides).
//!    Comparing fewer things can only *miss* a disagreement (a recall
//!    gap, acceptable for a corroboration harness), never invent one.
//!
//! Anything indecisive (solver timeout / unknown / unsound, or nothing
//! comparable) is `Inconclusive`, never `Agree`.

use std::collections::{BTreeMap, BTreeSet};

use r2smt_common::{Address, Arch, SmtResult};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, SliceStatus, register_layout,
};
use r2smt_ssa::{SsaLiftedSlice, ssa_convert};

/// Flags compared by the equivalence oracle. `OF` / `PF` are
/// deliberately excluded: `r2smt_core::downgrade_for_unmodeled_flags`
/// already treats them as unmodelled, so a difference there reflects
/// modelling depth, not a lifter bug — comparing them would fabricate
/// disagreements.
const COMPARABLE_FLAGS: &[&str] = &["ZF", "CF", "SF"];

/// Global recursion budget for the expression walkers (Host-Side
/// Safety guardrail). Per-instruction expression trees are shallow;
/// exceeding this means the IR is pathological and is treated as
/// not comparable rather than risking host-stack exhaustion.
const EXPR_DEPTH_BUDGET: u32 = 256;

/// Verdict of a single pairwise lowering comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffVerdict {
    /// The two lowerings are provably equal over every jointly-defined
    /// modelled output (the differ-condition is unsatisfiable).
    Agree,
    /// A concrete machine state exists under which a jointly-defined
    /// modelled output differs — an engine-integrity finding.
    Disagree,
    /// Not decided: solver timeout / unknown / unsound, or the
    /// lowerings share nothing comparable. Fail-closed — never
    /// reported as [`DiffVerdict::Agree`].
    Inconclusive,
}

/// Map an [`SmtResult`] over the differ-condition into a
/// [`DiffVerdict`].
///
/// The query's condition asserts "some jointly-defined output
/// differs", so an `AlwaysFalse` verdict means the outputs can never
/// differ → the lowerings agree. Any satisfiable-difference verdict is
/// a disagreement; anything the solver could not settle is
/// `Inconclusive`.
#[must_use]
pub fn classify_equivalence(verdict: SmtResult) -> DiffVerdict {
    match verdict {
        SmtResult::AlwaysFalse => DiffVerdict::Agree,
        SmtResult::AlwaysTrue | SmtResult::BothPossible => DiffVerdict::Disagree,
        // `Unsound` / `Timeout` / `Unknown` and any future
        // `#[non_exhaustive]` variant fail closed: never `Agree`.
        _ => DiffVerdict::Inconclusive,
    }
}

/// Running tally of pairwise comparisons, used to emit the
/// lifter-agreement-rate metric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AgreementStats {
    /// Comparisons that proved equivalence.
    pub agree: u64,
    /// Comparisons that proved a disagreement.
    pub disagree: u64,
    /// Comparisons the harness could not decide.
    pub inconclusive: u64,
}

impl AgreementStats {
    /// Fold one verdict into the tally.
    pub fn record(&mut self, verdict: DiffVerdict) {
        match verdict {
            DiffVerdict::Agree => self.agree += 1,
            DiffVerdict::Disagree => self.disagree += 1,
            DiffVerdict::Inconclusive => self.inconclusive += 1,
        }
    }

    /// Agreement rate over the *decided* comparisons
    /// (`agree / (agree + disagree)`). `None` when nothing was
    /// decided, so an all-inconclusive run is never reported as
    /// "100% agreement".
    #[must_use]
    pub fn agreement_rate(&self) -> Option<f64> {
        let decided = self.agree + self.disagree;
        if decided == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(self.agree as f64 / decided as f64)
    }
}

/// Build the SMT query whose `AlwaysFalse` verdict proves lowerings
/// `a` and `b` (for the *same* instruction, under `arch`) produce an
/// equivalent post-state.
///
/// Returns `None` when the lowerings are not comparable (see the
/// module-level soundness posture); the caller treats `None` as
/// [`DiffVerdict::Inconclusive`].
#[must_use]
pub fn build_equivalence_query(a: &[IrStmt], b: &[IrStmt], arch: Arch) -> Option<SsaLiftedSlice> {
    if !comparable(a) || !comparable(b) {
        return None;
    }

    let ssa_a = ssa_of(a, arch);
    let ssa_b = ssa_of(b, arch);

    let (parents, none_inputs) = tie_inputs(&ssa_a, &ssa_b, arch)?;

    let finals_a = final_defs(&ssa_a);
    let finals_b = final_defs(&ssa_b);
    let tainted_a = tainted_defs(&ssa_a)?;
    let tainted_b = tainted_defs(&ssa_b)?;
    let condition = differ_condition(&finals_a, &finals_b, &tainted_a, &tainted_b, arch)?;

    let mut statements: Vec<IrStmt> = Vec::new();
    statements.extend(tie_statements(&ssa_a, &ssa_b, &parents, arch)?);
    for stmt in &ssa_a.statements {
        statements.push(namespace_stmt(stmt, "A·")?);
    }
    for stmt in &ssa_b.statements {
        statements.push(namespace_stmt(stmt, "B·")?);
    }

    let mut inputs: Vec<Var> = parents.into_values().collect();
    inputs.extend(none_inputs);
    let defs: Vec<Var> = statements
        .iter()
        .filter_map(|s| match s {
            IrStmt::Assign { dst, .. } => Some(dst.clone()),
            _ => None,
        })
        .collect();

    Some(SsaLiftedSlice {
        branch: synthetic_branch(),
        statements,
        condition,
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        inputs,
        defs,
        arch,
    })
}

/// `true` when a lowering is structurally comparable. Memory effects
/// ([`IrStmt::LoadMem`] / [`IrStmt::StoreMem`]) and declined
/// instructions ([`IrStmt::Unsupported`]) have no sound model here, so
/// the pair is not comparable. An [`Expr::Unknown`] is *not*
/// disqualifying — it only taints the specific outputs it feeds (see
/// [`tainted_defs`]); fully-modelled outputs are still compared.
fn comparable(stmts: &[IrStmt]) -> bool {
    !stmts.iter().any(|stmt| {
        matches!(
            stmt,
            IrStmt::LoadMem { .. } | IrStmt::StoreMem { .. } | IrStmt::Unsupported { .. }
        )
    })
}

/// Forward-propagate "this value is not fully modelled" taint over the
/// SSA statement list. A def is tainted when its RHS contains an
/// [`Expr::Unknown`] or reads an already-tainted SSA variable. SSA
/// guarantees defs precede their uses, so a single forward pass
/// reaches the fixed point. `None` if the recursion budget is
/// exceeded (pathological IR — caller treats it as not comparable).
fn tainted_defs(ssa: &SsaLiftedSlice) -> Option<BTreeSet<String>> {
    let mut tainted: BTreeSet<String> = BTreeSet::new();
    for stmt in &ssa.statements {
        if let IrStmt::Assign { dst, src } = stmt
            && expr_is_tainted(src, &tainted, 0)?
        {
            tainted.insert(dst.name.clone());
        }
    }
    Some(tainted)
}

fn expr_is_tainted(expr: &Expr, tainted: &BTreeSet<String>, depth: u32) -> Option<bool> {
    if depth > EXPR_DEPTH_BUDGET {
        return None;
    }
    let next = depth + 1;
    let any = match expr {
        Expr::Unknown(_) => return Some(true),
        Expr::Var(v) => tainted.contains(&v.name),
        Expr::Const { .. } => false,
        Expr::BoolNot(a) => expr_is_tainted(a, tainted, next)?,
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
        | Expr::BoolOr(a, b)
        | Expr::Concat { high: a, low: b } => {
            expr_is_tainted(a, tainted, next)? || expr_is_tainted(b, tainted, next)?
        }
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            expr_is_tainted(cond, tainted, next)?
                || expr_is_tainted(then_expr, tainted, next)?
                || expr_is_tainted(else_expr, tainted, next)?
        }
        Expr::Extract { src, .. } | Expr::ZeroExtend { src, .. } | Expr::SignExtend { src, .. } => {
            expr_is_tainted(src, tainted, next)?
        }
    };
    Some(any)
}

fn ssa_of(stmts: &[IrStmt], arch: Arch) -> SsaLiftedSlice {
    let lifted = LiftedSlice {
        branch: synthetic_branch(),
        statements: stmts.to_vec(),
        condition: Expr::konst(0, 1),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch,
    };
    ssa_convert(&lifted)
}

fn synthetic_branch() -> BranchCandidate {
    let zero = Address::new(0);
    BranchCandidate {
        address: zero,
        function: zero,
        block: zero,
        kind: BranchKind::Jcc,
        mnemonic: "difflift".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "lifter-equivalence".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Namespace every *defined* variable (SSA gives every def a `#N`
/// suffix; free inputs never carry one) so the two lowering bodies can
/// coexist in one slice while still *sharing* their free inputs.
fn namespace_stmt(stmt: &IrStmt, pfx: &str) -> Option<IrStmt> {
    match stmt {
        IrStmt::Assign { dst, src } => Some(IrStmt::Assign {
            dst: namespace_var(dst, pfx),
            src: namespace_expr(src, pfx, 0)?,
        }),
        IrStmt::Nop => Some(IrStmt::Nop),
        // Rejected upstream by `comparable`; defensively pass through.
        other => Some(other.clone()),
    }
}

fn namespace_var(var: &Var, pfx: &str) -> Var {
    if var.name.contains('#') {
        Var::new(format!("{pfx}{}", var.name), var.bits)
    } else {
        var.clone()
    }
}

fn namespace_expr(expr: &Expr, pfx: &str, depth: u32) -> Option<Expr> {
    if depth > EXPR_DEPTH_BUDGET {
        return None;
    }
    let next = depth + 1;
    let bin = |ctor: fn(Box<Expr>, Box<Expr>) -> Expr, a: &Expr, b: &Expr| -> Option<Expr> {
        Some(ctor(
            Box::new(namespace_expr(a, pfx, next)?),
            Box::new(namespace_expr(b, pfx, next)?),
        ))
    };
    let out = match expr {
        Expr::Var(v) => Expr::Var(namespace_var(v, pfx)),
        Expr::Const { .. } | Expr::Unknown(_) => expr.clone(),
        Expr::Add(a, b) => bin(Expr::Add, a, b)?,
        Expr::Sub(a, b) => bin(Expr::Sub, a, b)?,
        Expr::Mul(a, b) => bin(Expr::Mul, a, b)?,
        Expr::UDiv(a, b) => bin(Expr::UDiv, a, b)?,
        Expr::URem(a, b) => bin(Expr::URem, a, b)?,
        Expr::SDiv(a, b) => bin(Expr::SDiv, a, b)?,
        Expr::SRem(a, b) => bin(Expr::SRem, a, b)?,
        Expr::And(a, b) => bin(Expr::And, a, b)?,
        Expr::Or(a, b) => bin(Expr::Or, a, b)?,
        Expr::Xor(a, b) => bin(Expr::Xor, a, b)?,
        Expr::Shl(a, b) => bin(Expr::Shl, a, b)?,
        Expr::LShr(a, b) => bin(Expr::LShr, a, b)?,
        Expr::AShr(a, b) => bin(Expr::AShr, a, b)?,
        Expr::Eq(a, b) => bin(Expr::Eq, a, b)?,
        Expr::Ne(a, b) => bin(Expr::Ne, a, b)?,
        Expr::Ult(a, b) => bin(Expr::Ult, a, b)?,
        Expr::Ule(a, b) => bin(Expr::Ule, a, b)?,
        Expr::Slt(a, b) => bin(Expr::Slt, a, b)?,
        Expr::Sle(a, b) => bin(Expr::Sle, a, b)?,
        Expr::BoolAnd(a, b) => bin(Expr::BoolAnd, a, b)?,
        Expr::BoolOr(a, b) => bin(Expr::BoolOr, a, b)?,
        Expr::BoolNot(a) => Expr::BoolNot(Box::new(namespace_expr(a, pfx, next)?)),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ite {
            cond: Box::new(namespace_expr(cond, pfx, next)?),
            then_expr: Box::new(namespace_expr(then_expr, pfx, next)?),
            else_expr: Box::new(namespace_expr(else_expr, pfx, next)?),
        },
        Expr::Extract { src, hi, lo } => Expr::Extract {
            src: Box::new(namespace_expr(src, pfx, next)?),
            hi: *hi,
            lo: *lo,
        },
        Expr::Concat { high, low } => Expr::Concat {
            high: Box::new(namespace_expr(high, pfx, next)?),
            low: Box::new(namespace_expr(low, pfx, next)?),
        },
        Expr::ZeroExtend { src, to_bits } => Expr::ZeroExtend {
            src: Box::new(namespace_expr(src, pfx, next)?),
            to_bits: *to_bits,
        },
        Expr::SignExtend { src, to_bits } => Expr::SignExtend {
            src: Box::new(namespace_expr(src, pfx, next)?),
            to_bits: *to_bits,
        },
    };
    Some(out)
}

/// Collect the final SSA version of every defined base name (the last
/// `base#N` in definition order maps `base → Var{"base#N", bits}`).
fn final_defs(ssa: &SsaLiftedSlice) -> BTreeMap<String, Var> {
    let mut finals: BTreeMap<String, Var> = BTreeMap::new();
    for def in &ssa.defs {
        if let Some((base, _)) = def.name.rsplit_once('#') {
            finals.insert(base.to_string(), def.clone());
        }
    }
    finals
}

/// Build the disjunction "some jointly-defined modelled output
/// differs". `None` when nothing is comparable.
fn differ_condition(
    finals_a: &BTreeMap<String, Var>,
    finals_b: &BTreeMap<String, Var>,
    tainted_a: &BTreeSet<String>,
    tainted_b: &BTreeSet<String>,
    arch: Arch,
) -> Option<Expr> {
    let mut terms: Vec<Expr> = Vec::new();
    for (base, var_a) in finals_a {
        let Some(var_b) = finals_b.get(base) else {
            continue;
        };
        let is_flag = COMPARABLE_FLAGS.contains(&base.as_str());
        let is_reg = register_layout(base, arch).is_some();
        if !(is_flag || is_reg) || var_a.bits != var_b.bits {
            continue;
        }
        // Only compare outputs that are fully modelled on *both*
        // sides. An output fed by an `Expr::Unknown` would otherwise
        // become an independent free value per side and fabricate a
        // disagreement — forbidden by the mission invariant.
        if tainted_a.contains(&var_a.name) || tainted_b.contains(&var_b.name) {
            continue;
        }
        terms.push(Expr::ne(
            Expr::Var(namespace_var(var_a, "A·")),
            Expr::Var(namespace_var(var_b, "B·")),
        ));
    }
    let mut iter = terms.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, Expr::bool_or))
}

/// Tie each side's free inputs to one shared machine state. Returns
/// the shared parent registers and the shared non-register free
/// inputs (flags read before being defined). `None` when the two
/// lowerings observe a different set of non-register free inputs
/// (not comparable).
#[allow(clippy::type_complexity)]
fn tie_inputs(
    ssa_a: &SsaLiftedSlice,
    ssa_b: &SsaLiftedSlice,
    arch: Arch,
) -> Option<(BTreeMap<String, Var>, Vec<Var>)> {
    let ptr = arch.pointer_bits();
    let mut parents: BTreeMap<String, Var> = BTreeMap::new();
    let mut none_a: BTreeMap<String, u8> = BTreeMap::new();
    let mut none_b: BTreeMap<String, u8> = BTreeMap::new();

    for (side, store) in [(&ssa_a.inputs, true), (&ssa_b.inputs, false)] {
        for input in side {
            if let Some(parent) = register_layout(&input.name, arch).map(|l| l.parent) {
                parents
                    .entry(parent.to_string())
                    .or_insert_with(|| Var::new(parent, ptr));
            } else {
                let bucket = if store { &mut none_a } else { &mut none_b };
                bucket.insert(input.name.clone(), input.bits);
            }
        }
    }

    if none_a != none_b {
        return None;
    }
    let none_inputs: Vec<Var> = none_a
        .into_iter()
        .map(|(name, bits)| Var::new(name, bits))
        .collect();
    Some((parents, none_inputs))
}

/// Emit the binding `side_input == slice_of(parent)` for every
/// register-typed free input that is not already the shared parent.
fn tie_statements(
    ssa_a: &SsaLiftedSlice,
    ssa_b: &SsaLiftedSlice,
    parents: &BTreeMap<String, Var>,
    arch: Arch,
) -> Option<Vec<IrStmt>> {
    let ptr = arch.pointer_bits();
    let mut ties: Vec<IrStmt> = Vec::new();
    let mut bound: BTreeSet<String> = BTreeSet::new();
    for input in ssa_a.inputs.iter().chain(ssa_b.inputs.iter()) {
        let Some(layout) = register_layout(&input.name, arch) else {
            continue;
        };
        if input.name == layout.parent && input.bits == ptr {
            continue;
        }
        if !bound.insert(input.name.clone()) {
            continue;
        }
        let _ = parents.get(layout.parent)?;
        let parent_var = Expr::var(layout.parent, ptr);
        let (slice, slice_bits) = if layout.lo == 0 && layout.hi + 1 == ptr {
            (parent_var, ptr)
        } else {
            (
                Expr::extract(parent_var, layout.hi, layout.lo),
                layout.width(),
            )
        };
        ties.push(IrStmt::Assign {
            dst: input.clone(),
            src: coerce(slice, slice_bits, input.bits),
        });
    }
    Some(ties)
}

fn coerce(expr: Expr, from_bits: u8, to_bits: u8) -> Expr {
    use std::cmp::Ordering::{Equal, Greater, Less};
    match to_bits.cmp(&from_bits) {
        Equal => expr,
        Less => Expr::extract(expr, to_bits.saturating_sub(1), 0),
        Greater => Expr::ZeroExtend {
            src: Box::new(expr),
            to_bits,
        },
    }
}
