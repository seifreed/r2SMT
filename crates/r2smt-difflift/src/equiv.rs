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
//!    `tainted_defs`), and a tainted output is excluded from the
//!    comparison set so it can never become an independently-free
//!    value that fabricates a difference.
//! 2. Free inputs are tied to one shared machine state through the
//!    project's own register-layout contract
//!    ([`r2smt_slicer::register_layout`]), so two lowerings that read
//!    the same architectural register observe the *same* symbolic
//!    value. Without this, distinct input spellings (`eax` vs the
//!    parent `rax`) would let the solver pick them independently and
//!    fabricate a difference.
//! 3. Only jointly-defined, modelled outputs are compared (the
//!    `ZF`/`CF`/`SF` flags — `OF`/`PF` are excluded for the same reason
//!    `r2smt_core` treats them as unmodelled — plus registers, projected
//!    into their *parent's* coordinates by `canonical_finals` so `fp` and
//!    `x29` pair instead of both dropping out). Comparing fewer things
//!    can only *miss* a disagreement (a recall gap, acceptable for a
//!    corroboration harness), never invent one.
//!
//! Two consequences of that projection are worth stating, because both
//! used to be silent recall gaps rather than deliberate policy. A
//! definition whose width disagrees with its architectural view is
//! **compared, not skipped** — that is the shape a width defect takes in
//! a destination, so the old equal-widths guard suppressed exactly the
//! bug it should have caught. And the one exception is the vector files,
//! where such a mismatch means the side has no SIMD model at all; that is
//! modelling depth, like `OF`/`PF`, and reporting it would bury every
//! real finding under one entry per vector instruction.
//!
//! Anything indecisive (solver timeout / unknown / unsound, or nothing
//! comparable) is `Inconclusive`, never `Agree`.

use std::collections::{BTreeMap, BTreeSet};

use r2smt_common::{Address, Arch, SmtResult};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::registers::{is_simd_parent, simd_parent_bits};
use r2smt_slicer::{
    BranchCandidate, BranchCondition, BranchKind, LiftedSlice, RegisterLayout, SliceStatus,
    register_layout,
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

    let tainted_a = tainted_defs(&ssa_a)?;
    let tainted_b = tainted_defs(&ssa_b)?;
    let finals_a = canonical_finals(&ssa_a, "A·", &tainted_a, arch);
    let finals_b = canonical_finals(&ssa_b, "B·", &tainted_b, arch);
    let condition = differ_condition(&finals_a, &finals_b)?;

    // No tie statements: `namespace_stmt` substitutes each register-typed
    // free input with its slice of the shared parent, so the two sides
    // already read one machine state and no name is ever declared twice.
    let mut statements: Vec<IrStmt> = Vec::new();
    for stmt in &ssa_a.statements {
        statements.push(namespace_stmt(stmt, "A·", arch)?);
    }
    for stmt in &ssa_b.statements {
        statements.push(namespace_stmt(stmt, "B·", arch)?);
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

// Exhaustive `Expr` dispatch: FP arms mirror the BV arms' shape but stay separate for legibility (CLAUDE.md exhaustive-dispatch exception).
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
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
        Expr::FAdd(a, b, _)
        | Expr::FSub(a, b, _)
        | Expr::FMul(a, b, _)
        | Expr::FDiv(a, b, _)
        | Expr::FEq(a, b)
        | Expr::FLt(a, b)
        | Expr::FLe(a, b) => {
            expr_is_tainted(a, tainted, next)? || expr_is_tainted(b, tainted, next)?
        }
        Expr::FIsNaN(s)
        | Expr::FSqrt(s, _)
        | Expr::FRoundToIntegral(s, _)
        | Expr::FpToIeeeBv(s)
        | Expr::BvToFp { src: s, .. }
        | Expr::FpToSbv { src: s, .. }
        | Expr::SbvToFp { src: s, .. }
        | Expr::FpToFp { src: s, .. } => expr_is_tainted(s, tainted, next)?,
        Expr::FpConst { .. } => false,
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
fn namespace_stmt(stmt: &IrStmt, pfx: &str, arch: Arch) -> Option<IrStmt> {
    match stmt {
        IrStmt::Assign { dst, src } => Some(IrStmt::Assign {
            dst: namespace_var(dst, pfx),
            src: namespace_expr(src, pfx, arch, 0)?,
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

/// Rewrite one variable occurrence: a definition is namespaced, a
/// register-typed free input is *substituted* by an exact slice of the
/// shared parent.
///
/// Substituting rather than asserting `input == slice_of(parent)` is what
/// keeps the query single-sorted. The assertion form emitted a second
/// symbol with the same name at another width — for a 16-bit `sp` it was
/// even degenerate, `sp[15:0] == sp[15:0]` — and the only reason that
/// side still modelled a truncation was the Z3 encoder answering a
/// narrower request with a sub-view. That made a Layer-3 correctness
/// property depend on a Layer-4 implementation detail, and the text
/// backends, which have no such behaviour, emitted an ill-sorted script
/// instead.
fn substitute_or_namespace(var: &Var, pfx: &str, arch: Arch) -> Expr {
    if var.name.contains('#') {
        return Expr::Var(namespace_var(var, pfx));
    }
    let Some(layout) = register_layout(&var.name, arch) else {
        return Expr::Var(var.clone());
    };
    let bits = parent_width(layout.parent, arch);
    if layout.hi >= bits {
        return Expr::Var(var.clone());
    }
    let parent = Expr::var(layout.parent, bits);
    let (slice, slice_bits) = if layout.lo == 0 && layout.hi.saturating_add(1) == bits {
        (parent, bits)
    } else {
        (Expr::extract(parent, layout.hi, layout.lo), layout.width())
    };
    coerce(slice, slice_bits, var.bits)
}

// Exhaustive `Expr` dispatch: FP arms mirror the BV arms' shape but stay separate for legibility (CLAUDE.md exhaustive-dispatch exception).
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
fn namespace_expr(expr: &Expr, pfx: &str, arch: Arch, depth: u32) -> Option<Expr> {
    if depth > EXPR_DEPTH_BUDGET {
        return None;
    }
    let next = depth + 1;
    let bin = |ctor: fn(Box<Expr>, Box<Expr>) -> Expr, a: &Expr, b: &Expr| -> Option<Expr> {
        Some(ctor(
            Box::new(namespace_expr(a, pfx, arch, next)?),
            Box::new(namespace_expr(b, pfx, arch, next)?),
        ))
    };
    let out = match expr {
        Expr::Var(v) => substitute_or_namespace(v, pfx, arch),
        Expr::Const { .. } | Expr::Unknown(_) => expr.clone(),
        Expr::Add(a, b) => bin(Expr::Add, a, b)?,
        Expr::Ror(a, b) => bin(Expr::Ror, a, b)?,
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
        Expr::BoolNot(a) => Expr::BoolNot(Box::new(namespace_expr(a, pfx, arch, next)?)),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ite {
            cond: Box::new(namespace_expr(cond, pfx, arch, next)?),
            then_expr: Box::new(namespace_expr(then_expr, pfx, arch, next)?),
            else_expr: Box::new(namespace_expr(else_expr, pfx, arch, next)?),
        },
        Expr::Extract { src, hi, lo } => Expr::Extract {
            src: Box::new(namespace_expr(src, pfx, arch, next)?),
            hi: *hi,
            lo: *lo,
        },
        Expr::Concat { high, low } => Expr::Concat {
            high: Box::new(namespace_expr(high, pfx, arch, next)?),
            low: Box::new(namespace_expr(low, pfx, arch, next)?),
        },
        Expr::ZeroExtend { src, to_bits } => Expr::ZeroExtend {
            src: Box::new(namespace_expr(src, pfx, arch, next)?),
            to_bits: *to_bits,
        },
        Expr::SignExtend { src, to_bits } => Expr::SignExtend {
            src: Box::new(namespace_expr(src, pfx, arch, next)?),
            to_bits: *to_bits,
        },
        Expr::FAdd(a, b, rm) => Expr::fadd(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
            *rm,
        ),
        Expr::FSub(a, b, rm) => Expr::fsub(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
            *rm,
        ),
        Expr::FMul(a, b, rm) => Expr::fmul(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
            *rm,
        ),
        Expr::FDiv(a, b, rm) => Expr::fdiv(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
            *rm,
        ),
        Expr::FEq(a, b) => Expr::feq(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
        ),
        Expr::FLt(a, b) => Expr::flt(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
        ),
        Expr::FLe(a, b) => Expr::fle(
            namespace_expr(a, pfx, arch, next)?,
            namespace_expr(b, pfx, arch, next)?,
        ),
        Expr::FIsNaN(a) => Expr::fisnan(namespace_expr(a, pfx, arch, next)?),
        Expr::FSqrt(a, rm) => Expr::fsqrt(namespace_expr(a, pfx, arch, next)?, *rm),
        Expr::FRoundToIntegral(a, rm) => {
            Expr::fround_to_integral(namespace_expr(a, pfx, arch, next)?, *rm)
        }
        Expr::FpConst { .. } => expr.clone(),
        Expr::BvToFp { src, ebits, sbits } => {
            Expr::bv_to_fp(namespace_expr(src, pfx, arch, next)?, *ebits, *sbits)
        }
        Expr::FpToIeeeBv(src) => Expr::fp_to_ieee_bv(namespace_expr(src, pfx, arch, next)?),
        Expr::FpToSbv { src, rm, bits } => {
            Expr::fp_to_sbv(namespace_expr(src, pfx, arch, next)?, *rm, *bits)
        }
        Expr::SbvToFp {
            src,
            rm,
            ebits,
            sbits,
        } => Expr::sbv_to_fp(namespace_expr(src, pfx, arch, next)?, *rm, *ebits, *sbits),
        Expr::FpToFp {
            src,
            rm,
            ebits,
            sbits,
        } => Expr::fp_to_fp(namespace_expr(src, pfx, arch, next)?, *rm, *ebits, *sbits),
    };
    Some(out)
}

/// Architectural width of a register parent under `arch`.
///
/// A vector parent is **not** the pointer width, and assuming it is was
/// the same class of defect as the `sp`-at-16-bits bug this harness was
/// built to find: `register_layout("xmm0", X86_64)` resolves to the
/// synthetic `zmm0` parent at 512 bits and both ARM ISAs carry `v<n>` at
/// 128, so sizing either at 64 declares a name at a width nothing else
/// uses and slices it out of range.
fn parent_width(parent: &str, arch: Arch) -> u16 {
    if is_simd_parent(parent, arch) {
        simd_parent_bits(arch).unwrap_or_else(|| arch.pointer_bits())
    } else {
        arch.pointer_bits()
    }
}

/// Rebuild what a definition of one register *view* says about the whole
/// parent, so two lowerings that name the same physical register through
/// different spellings are compared instead of dropped.
///
/// Reading the def in its own coordinates and matching on the base name
/// is what lost 181 instances of the `sp` bug: a side defining `fp` and a
/// side defining `x29` never paired, because `differ_condition` matched
/// base names exactly while `tie_inputs` had been canonicalising the
/// *input* side through [`register_layout`] all along.
///
/// `parent_in` is the shared free input for the parent, so the bits this
/// write does not touch come from the one machine state both sides read.
/// A 32-bit alias write zero-extends its parent on both `x86_64` and
/// `AArch64`, so that form is the whole parent and reads nothing.
fn parent_space_value(
    def: &Var,
    prefix: &str,
    layout: RegisterLayout,
    parent_bits: u16,
) -> Option<Expr> {
    let piece = Expr::Var(namespace_var(def, prefix));
    if layout.zero_extends_parent_64 || (layout.lo == 0 && layout.hi.checked_add(1)? >= parent_bits)
    {
        return Some(coerce(piece, def.bits, parent_bits));
    }
    // Deliberately *not* guarded on `def.bits == layout.width()`: a
    // definition whose width disagrees with the architectural view is
    // precisely the bug shape this harness must report, so it is coerced
    // and compared rather than skipped.
    let mut value = coerce(piece, def.bits, layout.width());
    let parent_in = Expr::var(layout.parent, parent_bits);
    if layout.lo > 0 {
        value = Expr::concat(
            value,
            Expr::extract(parent_in.clone(), layout.lo.checked_sub(1)?, 0),
        );
    }
    if layout.hi.checked_add(1)? < parent_bits {
        value = Expr::concat(
            Expr::extract(
                parent_in,
                parent_bits.checked_sub(1)?,
                layout.hi.checked_add(1)?,
            ),
            value,
        );
    }
    Some(value)
}

/// Project one side's final definitions into parent coordinates, keyed
/// by parent name (registers) or flag name.
///
/// Two definitions landing on one parent — a lowering that writes both
/// `w0` and `x0` — drop that parent entirely: [`final_defs`] keys by base
/// name and so does not order them against each other, and the narrower
/// reconstruction would splice against a `parent_in` that the wider write
/// has already replaced. Fails toward missing a disagreement.
fn canonical_finals(
    ssa: &SsaLiftedSlice,
    prefix: &str,
    tainted: &BTreeSet<String>,
    arch: Arch,
) -> BTreeMap<String, Expr> {
    let mut out: BTreeMap<String, Expr> = BTreeMap::new();
    let mut collided: BTreeSet<String> = BTreeSet::new();
    for (base, def) in final_defs(ssa) {
        if tainted.contains(&def.name) {
            continue;
        }
        if COMPARABLE_FLAGS.contains(&base.as_str()) {
            out.insert(base, Expr::Var(namespace_var(&def, prefix)));
            continue;
        }
        let Some(layout) = register_layout(&base, arch) else {
            continue;
        };
        // A definition whose width contradicts its own architectural
        // view means the side has no model of that register file at all,
        // which is modelling depth rather than a lifter defect — the same
        // reason `OF` / `PF` sit outside [`COMPARABLE_FLAGS`]. This is
        // scoped to the **vector** files on purpose: r2's ESIL carries no
        // SIMD model and names `xmm0` at the pointer width, so comparing
        // it against a real 128-bit merge reports every vector
        // instruction and buries the findings that matter. The same
        // mismatch on a general register is the `sp`-at-16-bits shape and
        // must still be reported, so it deliberately falls through.
        if def.bits != layout.width() && is_simd_parent(layout.parent, arch) {
            continue;
        }
        let parent = layout.parent.to_string();
        let bits = parent_width(layout.parent, arch);
        let Some(value) = parent_space_value(&def, prefix, layout, bits) else {
            collided.insert(parent);
            continue;
        };
        if out.insert(parent.clone(), value).is_some() {
            collided.insert(parent);
        }
    }
    for parent in &collided {
        out.remove(parent);
    }
    out
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
///
/// Both sides arrive already projected into parent coordinates by
/// [`canonical_finals`], which is what removed the old
/// `var_a.bits != var_b.bits` guard. That guard skipped exactly the shape
/// a width bug produces in a destination — `sub sp, sp, #N` defining `sp`
/// at 16 bits on one side and 64 on the other — so a width defect landing
/// on the destination suppressed its own detection.
fn differ_condition(
    finals_a: &BTreeMap<String, Expr>,
    finals_b: &BTreeMap<String, Expr>,
) -> Option<Expr> {
    let mut terms: Vec<Expr> = Vec::new();
    for (key, value_a) in finals_a {
        let Some(value_b) = finals_b.get(key) else {
            continue;
        };
        terms.push(Expr::ne(value_a.clone(), value_b.clone()));
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
    let mut parents: BTreeMap<String, Var> = BTreeMap::new();
    let mut none_a: BTreeMap<String, u16> = BTreeMap::new();
    let mut none_b: BTreeMap<String, u16> = BTreeMap::new();

    for (side, store) in [(&ssa_a.inputs, true), (&ssa_b.inputs, false)] {
        for input in side {
            if let Some(parent) = register_layout(&input.name, arch).map(|l| l.parent) {
                parents
                    .entry(parent.to_string())
                    .or_insert_with(|| Var::new(parent, parent_width(parent, arch)));
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

fn coerce(expr: Expr, from_bits: u16, to_bits: u16) -> Expr {
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
