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
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
    use r2smt_ssa::ssa_convert;

    use super::*;

    fn op(raw: &str, kind: OperandKind) -> Operand {
        Operand {
            raw: raw.into(),
            kind,
        }
    }

    fn insn(addr: u64, size: u8, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            address: Address(addr),
            size,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands,
            esil: None,
            pcode: None,
            is_thumb: false,
        }
    }

    fn one_block(insns: Vec<Instruction>) -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: insns,
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    fn ssa_first(program: &Program) -> SsaLiftedSlice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        let slice = slice_branch(
            cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        );
        let lifted = lift_slice(&slice, program.arch);
        ssa_convert(&lifted)
    }

    #[test]
    fn always_true_with_inputs_classifies_as_opaque_predicate() {
        // test eax, eax ; jne — but we'll synthetically force the
        // verdict to AlwaysTrue to exercise the classifier alone.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::AlwaysTrue);
        assert_eq!(finding.kind, FindingKind::OpaquePredicate);
        assert!(finding.is_actionable());
    }

    #[test]
    fn always_false_with_inputs_classifies_as_dead_branch() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
        assert_eq!(finding.kind, FindingKind::DeadBranch);
        assert!(finding.is_actionable());
    }

    #[test]
    fn constant_condition_has_no_inputs() {
        // mov eax, 1 ; cmp eax, 1 ; jne — no register inputs.
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
        assert_eq!(finding.kind, FindingKind::ConstantCondition);
        assert!(finding.evidence.inputs.is_empty());
    }

    #[test]
    fn both_possible_is_real_branch_not_actionable() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::BothPossible);
        assert_eq!(finding.kind, FindingKind::RealBranch);
        assert!(!finding.is_actionable());
    }

    #[test]
    fn unknown_verdict_becomes_suspicious() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::Timeout);
        assert_eq!(finding.kind, FindingKind::SuspiciousButUnknown);
        assert_eq!(finding.confidence, Confidence::Unknown);
    }

    #[test]
    fn complete_slice_with_unknowns_yields_medium_confidence() {
        // cmp emits OF/PF as Unknown — at least 2 Unknown exprs.
        let program = one_block(vec![
            insn(
                0x40_1000,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0x10", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1003,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::BothPossible);
        assert!(finding.evidence.unknown_count >= 1);
        assert_eq!(finding.confidence, Confidence::Medium);
    }

    #[test]
    fn truncated_slice_gives_unknown_confidence() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        // SmtResult is Unsound for truncated, simulate the pipeline.
        let finding = classify_finding(&ssa, SmtResult::Unsound);
        assert_eq!(finding.kind, FindingKind::SuspiciousButUnknown);
        assert_eq!(finding.confidence, Confidence::Unknown);
    }

    #[test]
    fn evidence_inputs_strip_ssa_suffix() {
        // `test eax, eax ; jne` produces a single free input (rax). The
        // SSA layer surfaces it as `rax#0` so substitution stays
        // unambiguous; user-facing evidence must drop the suffix.
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::BothPossible);
        assert_eq!(finding.evidence.inputs, vec!["rax".to_string()]);
        // The pretty form must also drop the suffix.
        assert!(
            !finding.formula_pretty.contains('#'),
            "formula_pretty must not surface SSA suffixes; got: {}",
            finding.formula_pretty
        );
    }

    #[test]
    fn finding_round_trips_with_operands_through_json() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let mut finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
        finding.operands = vec!["x0".into(), "x1".into(), "x2".into(), "eq".into()];
        let json = serde_json::to_string(&finding).unwrap();
        let back: Finding = serde_json::from_str(&json).unwrap();
        assert_eq!(back.operands, finding.operands);
    }

    #[test]
    fn finding_round_trips_through_json() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
        let json = serde_json::to_string(&finding).unwrap();
        let back: Finding = serde_json::from_str(&json).unwrap();
        assert_eq!(back, finding);
    }

    #[test]
    fn signed_jge_with_unknown_flags_downgrades_to_low_confidence() {
        // cmp eax, 0; jge dest — `jge` depends on `SF == OF`, but our
        // lifter leaves `OF` as Unknown after a cmp. The solver may
        // resolve the formula either way; the verdict should land at
        // `Low` confidence per the Phase E flag-helper refusal.
        let program = one_block(vec![
            insn(
                0x40_1000,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1003,
                6,
                "jge",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let finding = classify_finding(&ssa, SmtResult::BothPossible);
        assert_eq!(finding.confidence, Confidence::Low);
    }

    #[test]
    fn truncated_slice_with_unknowns_on_truncation_yields_medium_confidence() {
        // call f ; cmp eax, eax ; jne junk — same fixture used by the
        // smt crate's regression. The slicer truncates because of the
        // `call`; with the policy the SSA layer surfaces eax as a free
        // input and the solver returns `AlwaysFalse`. The classifier
        // must keep the verdict but downgrade confidence to Medium
        // (not Unknown, not High).
        let program = one_block(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                2,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let candidates = collect_branches(&program);
        let cand = candidates.first().expect("at least one branch");
        let limits = SliceLimits {
            unknowns_on_truncation: true,
            ..SliceLimits::default()
        };
        let slice = slice_branch(cand, &program.functions[0], &limits, program.arch);
        assert!(
            matches!(slice.status, SliceStatus::Truncated { .. }),
            "slice must be truncated to exercise the policy"
        );
        assert!(slice.treat_truncation_as_inputs);
        let lifted = lift_slice(&slice, program.arch);
        let ssa = ssa_convert(&lifted);
        let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
        assert_eq!(finding.kind, FindingKind::DeadBranch);
        assert_eq!(finding.confidence, Confidence::Medium);
    }

    #[test]
    fn classify_lowered_upstream_emits_constant_condition_when_resolved_to_taken_target() {
        // Block whose only successor is the cjmp's taken target. The
        // CFG analyser already lowered the branch to unconditional;
        // r2SMT must report `ConstantCondition` with `High` confidence
        // and stamp the resolved target in `evidence.upstream_resolved_to`.
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![insn(
                        0x40_1000,
                        6,
                        "jne",
                        vec![op("0x401080", OperandKind::Immediate)],
                    )],
                    successors: vec![Address(0x40_1080)],
                }],
                is_thumb: false,
            }],
        };
        let candidates = collect_branches(&program);
        let cand = candidates.first().expect("at least one branch");
        let finding =
            classify_lowered_upstream(cand).expect("upstream_resolved must trigger shortcut");
        assert_eq!(finding.kind, FindingKind::ConstantCondition);
        assert_eq!(finding.confidence, Confidence::High);
        assert_eq!(finding.verdict, SmtResult::AlwaysTrue);
        assert_eq!(
            finding.evidence.upstream_resolved_to,
            Some(Address(0x40_1080))
        );
    }

    #[test]
    fn classify_lowered_upstream_returns_none_for_two_way_branch() {
        // Block with both successors recorded — the analyser left the
        // cjmp two-way, so the shortcut must not fire and the SMT
        // pipeline takes over.
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![insn(
                        0x40_1000,
                        6,
                        "jne",
                        vec![op("0x401080", OperandKind::Immediate)],
                    )],
                    successors: vec![Address(0x40_1080), Address(0x40_1006)],
                }],
                is_thumb: false,
            }],
        };
        let candidates = collect_branches(&program);
        let cand = candidates.first().expect("at least one branch");
        assert!(classify_lowered_upstream(cand).is_none());
    }

    #[test]
    fn classify_lowered_upstream_returns_always_false_when_resolved_to_fallthrough() {
        // Block whose single successor is the fallthrough address —
        // the CFG analyser proved the cjmp is never taken.
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![insn(
                        0x40_1000,
                        6,
                        "jne",
                        vec![op("0x401080", OperandKind::Immediate)],
                    )],
                    successors: vec![Address(0x40_1006)],
                }],
                is_thumb: false,
            }],
        };
        let candidates = collect_branches(&program);
        let cand = candidates.first().expect("at least one branch");
        let finding =
            classify_lowered_upstream(cand).expect("upstream_resolved must trigger shortcut");
        assert_eq!(finding.verdict, SmtResult::AlwaysFalse);
    }

    #[test]
    fn classify_lowered_upstream_returns_none_when_cfg_disagrees_with_targets() {
        // The successor recorded in the CFG (here `0x40_1234`) matches
        // neither `taken_target` (0x40_1080 from the operand) nor the
        // fallthrough (0x40_1006). Rather than guessing AlwaysTrue, the
        // classifier must hand the case off to the SMT pipeline.
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![insn(
                        0x40_1000,
                        6,
                        "jne",
                        vec![op("0x401080", OperandKind::Immediate)],
                    )],
                    successors: vec![Address(0x40_1234)],
                }],
                is_thumb: false,
            }],
        };
        let candidates = collect_branches(&program);
        let cand = candidates.first().expect("at least one branch");
        assert!(classify_lowered_upstream(cand).is_none());
    }

    #[test]
    fn confidence_ordering_high_below_medium_below_low_below_unknown() {
        // Lint: PartialOrd derive gives lexical (declaration) order:
        // High < Medium < Low < Unknown. Verify.
        assert!(Confidence::High < Confidence::Medium);
        assert!(Confidence::Medium < Confidence::Low);
        assert!(Confidence::Low < Confidence::Unknown);
    }

    #[test]
    fn reconcile_folded_prefers_sound_smt_over_cfg_shortcut() {
        let program = one_block(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        // Sound SMT verdict (complete slice → not Unknown confidence).
        let smt_sound = classify_finding(&ssa, SmtResult::AlwaysFalse);
        assert_ne!(smt_sound.confidence, Confidence::Unknown);
        // Inconclusive SMT (unsound/truncated → Unknown confidence).
        let smt_unknown = classify_finding(&ssa, SmtResult::Unsound);
        assert_eq!(smt_unknown.confidence, Confidence::Unknown);
        // A stand-in CFG-shortcut finding.
        let cfg = classify_finding(&ssa, SmtResult::BothPossible);

        // Sound SMT wins over the CFG shortcut.
        let r = reconcile_folded(Some(smt_sound.clone()), Some(cfg.clone())).expect("some");
        assert_eq!(r.verdict, SmtResult::AlwaysFalse);
        // Inconclusive SMT falls back to the CFG shortcut.
        let r = reconcile_folded(Some(smt_unknown.clone()), Some(cfg.clone())).expect("some");
        assert_eq!(r.verdict, cfg.verdict);
        // No CFG safety net → the inconclusive SMT is still returned.
        let r = reconcile_folded(Some(smt_unknown), None).expect("some");
        assert_eq!(r.verdict, SmtResult::Unsound);
        // Nothing at all → None.
        assert!(reconcile_folded(None, None).is_none());
    }
}
