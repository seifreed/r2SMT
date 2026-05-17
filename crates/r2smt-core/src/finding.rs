//! Decision engine: turn an SSA-lifted slice and an SMT verdict into
//! a [`Finding`].
//!
//! A `Finding` captures everything the downstream report / patch
//! pipeline needs to act on a single branch:
//!
//! - where the branch lives ([`Finding::address`], [`Finding::function`]);
//! - what the branch *does* ([`Finding::mnemonic`],
//!   [`Finding::condition`], [`Finding::formula`]);
//! - what the solver said ([`Finding::verdict`]);
//! - r2SMT's interpretation ([`Finding::kind`], [`Finding::confidence`]);
//! - structural evidence ([`Finding::evidence`]).

use r2smt_common::Address;
use r2smt_common::smt::SmtResult;
use r2smt_ir::NameHints;
use r2smt_ir::expr::Expr;
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::BranchCandidate;
use r2smt_slicer::condition::BranchCondition;
use r2smt_slicer::slice::SliceStatus;
use r2smt_ssa::{SsaLiftedSlice, pretty_condition, pretty_condition_with_hints};
use serde::{Deserialize, Serialize};

/// A classified observation about a single conditional branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    /// Address of the conditional instruction.
    pub address: Address,
    /// Address of the owning function.
    pub function: Address,
    /// Mnemonic of the conditional instruction (`jne`, `setz`, …).
    pub mnemonic: String,
    /// Symbolic flag-predicate family.
    pub condition: BranchCondition,
    /// Human-readable formula at the flag level (e.g. `"ZF == 0"`).
    pub formula: String,
    /// Human-readable formula with SSA definitions substituted back
    /// into the condition (e.g. `"(((rcx * rcx) & 0x1:32) == 0x2:32)"`).
    /// Surfaces the arithmetic that actually drives the branch.
    #[serde(default)]
    pub formula_pretty: String,
    /// C-style infix rendering of the post-`aggressive_simplify` Z3
    /// formula. `None` when no solver verdict was produced (truncated
    /// slice, CFG shortcut), or when the SMT-LIB backend was used
    /// (no Z3 AST available). Surfaces the formula the solver
    /// actually decided on, after its tactic chain folded identities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formula_z3_pretty: Option<String>,
    /// Solver verdict.
    pub verdict: SmtResult,
    /// r2SMT's classification of the verdict.
    pub kind: FindingKind,
    /// How much to trust the verdict.
    pub confidence: Confidence,
    /// Address control transfers to when the branch is taken
    /// (`Some` only for `jcc` with an immediate operand).
    pub taken_target: Option<Address>,
    /// Address of the instruction immediately after the conditional.
    pub fallthrough_target: Option<Address>,
    /// Raw textual operand strings as reported by the disassembler, in
    /// source order. Empty for findings produced from legacy fixtures
    /// or for instructions whose operand list could not be parsed.
    /// Surfaces to the patcher so register-aware rewrites (e.g.
    /// `AArch64` `cset Xd` → `mov Xd, #imm`) can locate the destination
    /// register from the textual mnemonic alone.
    #[serde(default)]
    pub operands: Vec<String>,
    /// `true` when the originating instruction is encoded in Thumb mode
    /// (`AArch32` only). Drives the patcher's choice between the 4-byte
    /// ARM-mode NOP / branch encodings and the 2-byte Thumb forms.
    /// Defaults to `false` so legacy fixtures and non-ARM targets
    /// require no migration.
    #[serde(default)]
    pub is_thumb: bool,
    /// Structural information that fed the classification.
    pub evidence: FindingEvidence,
    /// Decompiler pseudocode for the owning function, attached by the
    /// composition root only when `--with-decompiler` is set. Purely
    /// analyst-facing context: it never influences the verdict,
    /// confidence, or classification. `None` by default so the domain
    /// stays decompiler-agnostic and legacy JSON needs no migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pseudocode: Option<String>,
}

/// What r2SMT thinks the verdict means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FindingKind {
    /// `AlwaysTrue` with at least one register input — the condition
    /// is a tautology over a value the program does not actually
    /// inspect. Classic obfuscation pattern.
    OpaquePredicate,
    /// `AlwaysFalse` with at least one register input — the
    /// conditional jump's taken target is unreachable (dead code).
    DeadBranch,
    /// `AlwaysTrue` / `AlwaysFalse` whose result depends only on
    /// constants embedded in the slice.
    ConstantCondition,
    /// `BothPossible` — the branch is a genuine choice. Not a
    /// "finding" in the deobfuscation sense; included so the report
    /// can show a full census.
    RealBranch,
    /// Solver returned `Unknown` / `Timeout`, or the slice was
    /// truncated. The verdict cannot be trusted yet.
    SuspiciousButUnknown,
    /// Two independent lowerings (P-code / ESIL / per-mnemonic) of the
    /// same instruction were proven *not* semantically equivalent by
    /// the differential harness. This is an engine-integrity defect —
    /// one of r2SMT's own lifters is unsound for that instruction —
    /// not a property of the analysed sample. Produced solely by the
    /// opt-in `--differential-lift` path, never by [`kind_for`].
    LifterDisagreement,
}

/// How much weight to give a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Confidence {
    /// Solver returned a definitive verdict and the IR is fully
    /// modelled (no `Expr::Unknown` nodes).
    High,
    /// Solver returned a definitive verdict but the IR contains some
    /// `Expr::Unknown` (typically `OF` / `PF` we do not yet model).
    /// Verdict is still sound because Unknowns are treated as free
    /// symbolic inputs — they can only *weaken* the verdict from
    /// always-X to `BothPossible`, never fabricate one.
    Medium,
    /// Reserved for future phases (partial memory model, multi-block
    /// slicing). Unused today.
    Low,
    /// Slice was truncated, or the solver returned `Unknown` /
    /// `Timeout` / `Unsound`.
    Unknown,
}

/// Structural data extracted from the slice / SSA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FindingEvidence {
    /// Slicer status forwarded for transparency.
    pub slice_status: SliceStatus,
    /// Number of IR statements in the SSA form.
    pub statement_count: usize,
    /// Number of free symbolic inputs to the slice.
    pub input_count: usize,
    /// Canonical names of the free inputs (in `BTreeMap` order).
    pub inputs: Vec<String>,
    /// Total number of `Expr::Unknown` nodes across all statements
    /// and the branch condition.
    pub unknown_count: usize,
    /// `Some(target)` when the finding came from a "branch lowered
    /// upstream" detection rather than the SMT pipeline — the
    /// containing block had a single successor, so the CFG analyser
    /// had already collapsed the cjmp before r2SMT saw it. The
    /// recorded target is the only successor the block exits to.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub upstream_resolved_to: Option<Address>,
}

impl Finding {
    /// `true` if the kind is one a deobfuscator would act on.
    #[must_use]
    pub const fn is_actionable(&self) -> bool {
        matches!(
            self.kind,
            FindingKind::OpaquePredicate | FindingKind::DeadBranch | FindingKind::ConstantCondition
        )
    }
}

/// Build a [`FindingKind::LifterDisagreement`] finding for an
/// instruction whose independent lowerings the differential harness
/// proved non-equivalent.
///
/// `detail` is the open-domain diagnostic naming the disagreeing
/// lowering pair. The verdict is [`SmtResult::Unsound`] and confidence
/// [`Confidence::Unknown`]: the disagreement says nothing trustworthy
/// about the branch — it flags r2SMT's own engine. This is never
/// produced by the normal classify path; only the opt-in
/// `--differential-lift` wiring constructs it.
#[must_use]
pub fn lifter_disagreement_finding(
    address: Address,
    function: Address,
    mnemonic: String,
    detail: String,
) -> Finding {
    Finding {
        address,
        function,
        mnemonic,
        condition: BranchCondition::NotEqual,
        formula: detail.clone(),
        formula_pretty: detail,
        formula_z3_pretty: None,
        verdict: SmtResult::Unsound,
        kind: FindingKind::LifterDisagreement,
        confidence: Confidence::Unknown,
        taken_target: None,
        fallthrough_target: None,
        operands: Vec::new(),
        is_thumb: false,
        evidence: FindingEvidence {
            slice_status: SliceStatus::Complete,
            statement_count: 0,
            input_count: 0,
            inputs: Vec::new(),
            unknown_count: 0,
            upstream_resolved_to: None,
        },
        pseudocode: None,
    }
}

/// Emit a [`Finding`] directly from a [`BranchCandidate`] when the
/// CFG analyser already proved the branch unconditional (the
/// containing block has a single successor). Returns `None` for
/// branches with the usual two-way successor set so callers can fall
/// through to the regular slice → lift → SSA → solve → classify flow.
///
/// The verdict is always [`SmtResult::AlwaysTrue`] when the resolved
/// target equals [`BranchCandidate::taken_target`],
/// [`SmtResult::AlwaysFalse`] when it equals
/// [`BranchCandidate::fallthrough_target`], and falls back to
/// [`SmtResult::AlwaysTrue`] when neither matches (the CFG resolved
/// to a target the operand parser did not recover — still a
/// definitive verdict, just one r2SMT cannot localise to taken /
/// fallthrough). [`FindingKind::ConstantCondition`] and
/// [`Confidence::High`] reflect that the CFG analyser is authoritative
/// here.
#[must_use]
pub fn classify_lowered_upstream(branch: &BranchCandidate) -> Option<Finding> {
    let resolved_to = branch.upstream_resolved?;
    let verdict = if branch.taken_target == Some(resolved_to) {
        SmtResult::AlwaysTrue
    } else if branch.fallthrough_target == Some(resolved_to) {
        SmtResult::AlwaysFalse
    } else {
        // CFG analyser proved the block has a single successor but the
        // address it picked matches neither the branch's textual jump
        // target nor its fallthrough. This is a disagreement between
        // r2's CFG and the operand decoder — surface it to the SMT
        // pipeline instead of guessing.
        return None;
    };
    let evidence = FindingEvidence {
        slice_status: SliceStatus::Complete,
        statement_count: 0,
        input_count: 0,
        inputs: Vec::new(),
        unknown_count: 0,
        upstream_resolved_to: Some(resolved_to),
    };
    Some(Finding {
        address: branch.address,
        function: branch.function,
        mnemonic: branch.mnemonic.clone(),
        condition: branch.condition,
        formula: branch.formula.clone(),
        formula_pretty: branch.formula.clone(),
        formula_z3_pretty: None,
        verdict,
        kind: FindingKind::ConstantCondition,
        confidence: Confidence::High,
        taken_target: branch.taken_target,
        fallthrough_target: branch.fallthrough_target,
        operands: branch.operand_raws.clone(),
        is_thumb: branch.is_thumb,
        evidence,
        pseudocode: None,
    })
}

/// Reconcile a branch radare2's `aaa` already folded (single-successor
/// block) between an independent SMT re-derivation and the CFG
/// shortcut ([`classify_lowered_upstream`]).
///
/// Policy: a re-derived finding the solver decided on a *sound* slice
/// (confidence anything but [`Confidence::Unknown`]) is authoritative —
/// it carries a real SMT proof, not just r2's CFG opinion, and it can
/// surface a folded branch that is actually two-way. If the
/// re-derivation was inconclusive (truncated / timeout / unsound →
/// `Confidence::Unknown`), the CFG shortcut is the safety net. With
/// neither, the inconclusive re-derivation is still returned (better
/// than dropping the branch); `None` only when there is nothing at all.
#[must_use]
pub fn reconcile_folded(
    rederived: Option<Finding>,
    cfg_shortcut: Option<Finding>,
) -> Option<Finding> {
    match rederived {
        Some(f) if f.confidence != Confidence::Unknown => Some(f),
        rederived => cfg_shortcut.or(rederived),
    }
}

/// Classify a single `(slice, verdict)` pair using the canonical names
/// the lifter emitted (no hints). Convenience wrapper for callers that
/// have no symbol info — equivalent to
/// [`classify_finding_with_hints`] with an empty [`NameHints`].
#[must_use]
pub fn classify_finding(slice: &SsaLiftedSlice, verdict: SmtResult) -> Finding {
    classify_finding_with_hints(slice, verdict, &NameHints::default())
}

/// Like [`classify_finding_with_hints`] but also accepts a C-style
/// infix rendering of the post-Z3-simplify formula. Used by callers
/// that already obtained the rendering from
/// `r2smt_smt::solve_branch_with_pretty`.
#[must_use]
pub fn classify_finding_with_pretty(
    slice: &SsaLiftedSlice,
    verdict: SmtResult,
    formula_z3_pretty: Option<String>,
    hints: &NameHints,
) -> Finding {
    let mut finding = classify_finding_with_hints(slice, verdict, hints);
    finding.formula_z3_pretty = formula_z3_pretty;
    finding
}

/// Classify a single `(slice, verdict)` pair, swapping canonical
/// stack-slot names (`stk_rbp_-4`) for analyst-facing aliases
/// (`var_4h`) recorded in `hints` when rendering `formula_pretty`.
#[must_use]
pub fn classify_finding_with_hints(
    slice: &SsaLiftedSlice,
    verdict: SmtResult,
    hints: &NameHints,
) -> Finding {
    let unknown_count: usize = slice
        .statements
        .iter()
        .map(count_unknowns_in_stmt)
        .sum::<usize>()
        + count_unknowns(&slice.condition);
    let inputs: Vec<String> = slice
        .inputs
        .iter()
        .map(|v| strip_ssa_suffix(&v.name).to_string())
        .collect();
    let evidence = FindingEvidence {
        slice_status: slice.status.clone(),
        statement_count: slice.statements.len(),
        input_count: slice.inputs.len(),
        inputs,
        unknown_count,
        upstream_resolved_to: None,
    };
    let is_complete = matches!(slice.status, SliceStatus::Complete);
    let kind = kind_for(verdict, slice);
    let raw_confidence = confidence_for(
        verdict,
        is_complete,
        unknown_count,
        slice.treat_truncation_as_inputs,
    );
    let confidence = downgrade_for_unmodeled_flags(raw_confidence, slice.branch.condition);
    let formula_pretty = if hints.is_empty() {
        pretty_condition(slice)
    } else {
        pretty_condition_with_hints(slice, hints)
    };
    Finding {
        address: slice.branch.address,
        function: slice.branch.function,
        mnemonic: slice.branch.mnemonic.clone(),
        condition: slice.branch.condition,
        formula: slice.branch.formula.clone(),
        formula_pretty,
        formula_z3_pretty: None,
        verdict,
        kind,
        confidence,
        taken_target: slice.branch.taken_target,
        fallthrough_target: slice.branch.fallthrough_target,
        operands: slice.branch.operand_raws.clone(),
        is_thumb: slice.branch.is_thumb,
        evidence,
        pseudocode: None,
    }
}

fn kind_for(verdict: SmtResult, slice: &SsaLiftedSlice) -> FindingKind {
    match verdict {
        SmtResult::AlwaysTrue | SmtResult::AlwaysFalse if slice.inputs.is_empty() => {
            FindingKind::ConstantCondition
        }
        SmtResult::AlwaysTrue => FindingKind::OpaquePredicate,
        SmtResult::AlwaysFalse => FindingKind::DeadBranch,
        SmtResult::BothPossible => FindingKind::RealBranch,
        _ => FindingKind::SuspiciousButUnknown,
    }
}

fn confidence_for(
    verdict: SmtResult,
    is_complete: bool,
    unknown_count: usize,
    treat_truncation_as_inputs: bool,
) -> Confidence {
    match verdict {
        SmtResult::AlwaysTrue | SmtResult::AlwaysFalse | SmtResult::BothPossible => {
            if is_complete {
                if unknown_count == 0 {
                    Confidence::High
                } else {
                    Confidence::Medium
                }
            } else if treat_truncation_as_inputs {
                // Truncated slice driven through the SMT pipeline
                // because the caller opted into
                // `unknowns_on_truncation`. SSA already widened the
                // unresolved roots into free symbolic inputs, so a
                // definitive verdict is sound but lives one notch
                // below a Complete slice on the trust ladder.
                Confidence::Medium
            } else {
                Confidence::Unknown
            }
        }
        _ => Confidence::Unknown,
    }
}

/// Downgrade the confidence to at most [`Confidence::Low`] when the
/// branch's flag predicate references flags the lifter cannot model
/// (currently `OF` / `PF`). Signed comparisons (`jg`, `jl`, …) and
/// parity / overflow branches fall into this bucket: even if the
/// solver returns `AlwaysTrue` / `AlwaysFalse`, the verdict could
/// only be coincidental with the Unknowns picked by the solver, so
/// it does not deserve `High` or `Medium` trust.
fn downgrade_for_unmodeled_flags(confidence: Confidence, condition: BranchCondition) -> Confidence {
    if condition.depends_on_unmodeled_flag()
        && matches!(confidence, Confidence::High | Confidence::Medium)
    {
        Confidence::Low
    } else {
        confidence
    }
}

/// Strip the trailing SSA `#N` suffix from a variable name. Free
/// inputs are at version 0 by construction, so the suffix carries no
/// information for the user-facing inputs list — but the IR keeps it
/// for correctness during rename and substitution.
fn strip_ssa_suffix(name: &str) -> &str {
    name.split_once('#').map_or(name, |(base, _)| base)
}

fn count_unknowns(expr: &Expr) -> usize {
    match expr {
        Expr::Unknown(_) => 1,
        Expr::Var(_) | Expr::Const { .. } => 0,
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
        | Expr::BoolOr(a, b) => count_unknowns(a) + count_unknowns(b),
        Expr::BoolNot(e) => count_unknowns(e),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => count_unknowns(cond) + count_unknowns(then_expr) + count_unknowns(else_expr),
        Expr::Extract { src, .. } | Expr::ZeroExtend { src, .. } | Expr::SignExtend { src, .. } => {
            count_unknowns(src)
        }
        Expr::Concat { high, low } => count_unknowns(high) + count_unknowns(low),
    }
}

fn count_unknowns_in_stmt(stmt: &IrStmt) -> usize {
    match stmt {
        IrStmt::Assign { src, .. } => count_unknowns(src),
        IrStmt::LoadMem { address, .. } => count_unknowns(address),
        IrStmt::StoreMem { address, value, .. } => count_unknowns(address) + count_unknowns(value),
        IrStmt::Unsupported { .. } | IrStmt::Nop => 0,
    }
}

#[cfg(test)]
mod tests;
