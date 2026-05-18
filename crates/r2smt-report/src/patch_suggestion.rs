//! Translate a [`Finding`] into a suggested patch.
//!
//! Phase 8 only emits suggestions (text and r2 commands); Phase 10
//! will apply them under explicit caller authorisation.

use r2smt_common::Address;
use r2smt_common::smt::SmtResult;
use r2smt_core::{Finding, FindingKind};

/// A patch the analyst could apply to make the binary's static
/// behaviour reflect the solver's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSuggestion {
    /// Address of the conditional instruction being patched.
    pub address: Address,
    /// Strategy name (matches `SPEC.md` §5.7 list).
    pub strategy: PatchStrategy,
    /// Human-readable rationale.
    pub rationale: String,
    /// radare2 `wa` command (or `wx`) suitable for the r2 script
    /// output. Always emitted commented-out in scripts; Phase 10 will
    /// apply it directly.
    pub r2_command: String,
}

/// Patch strategies supported in the SPEC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatchStrategy {
    /// Replace a `jcc` whose taken-target is reachable with an
    /// unconditional `jmp`.
    ReplaceJccWithJmp,
    /// NOP-out a `jcc` whose taken-target is dead.
    NopJcc,
    /// `setcc` with constant result → `mov r, imm`. Phase 8 emits
    /// only the comment; the byte sequence depends on the destination
    /// register and is left for Phase 10.
    ReplaceSetCcWithMovConst,
    /// `cmovcc` with constant condition → `mov` or `nop`. Same caveat.
    ReplaceCMovCcWithMovOrNop,
    /// `AArch64` `cset` / `csetm` with constant condition →
    /// `mov Rd, #0|#1|#-1`. The destination register is recovered from
    /// the operand list; `csetm`'s "all-ones" form maps to `#-1` when
    /// the predicate is true.
    ReplaceCsetWithMovConst,
    /// `AArch64` `csel` with constant condition → `mov Rd, Rn` (true) or
    /// `mov Rd, Rm` (false).
    ReplaceCselWithMov,
    /// `AArch64` `csinc` (and its `cinc` 2-operand alias) with constant
    /// condition → `mov Rd, Rn` (true) or `add Rd, Rm, #1` (false).
    ReplaceCsincWithMovOrAdd1,
    /// `AArch64` `csinv` (and its `cinv` 2-operand alias) with constant
    /// condition → `mov Rd, Rn` (true) or `mvn Rd, Rm` (false).
    ReplaceCsinvWithMovOrMvn,
    /// `AArch64` `csneg` (and its `cneg` 2-operand alias) with constant
    /// condition → `mov Rd, Rn` (true) or `neg Rd, Rm` (false).
    ReplaceCsnegWithMovOrNeg,
    /// No safe transformation, only an annotation.
    CommentOnly,
}

impl PatchStrategy {
    /// String name used in JSON / Markdown / r2 script outputs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReplaceJccWithJmp => "replace_jcc_with_jmp",
            Self::NopJcc => "nop_jcc",
            Self::ReplaceSetCcWithMovConst => "replace_setcc_with_mov_const",
            Self::ReplaceCMovCcWithMovOrNop => "replace_cmovcc_with_mov_or_nop",
            Self::ReplaceCsetWithMovConst => "replace_cset_with_mov_const",
            Self::ReplaceCselWithMov => "replace_csel_with_mov",
            Self::ReplaceCsincWithMovOrAdd1 => "replace_csinc_with_mov_or_add1",
            Self::ReplaceCsinvWithMovOrMvn => "replace_csinv_with_mov_or_mvn",
            Self::ReplaceCsnegWithMovOrNeg => "replace_csneg_with_mov_or_neg",
            Self::CommentOnly => "comment_only",
        }
    }
}

/// Pick a strategy for the given finding, if any.
///
/// Returns `None` for [`FindingKind::RealBranch`] and
/// [`FindingKind::SuspiciousButUnknown`].
// `clippy::too_many_lines` / `unnested_or_patterns`: the body is an
// exhaustive dispatch table over the closed pair (`FindingKind`,
// `mnemonic-family`); flattening the or-patterns or splitting the
// function would just scatter the table without removing any branch.
#[must_use]
#[allow(clippy::too_many_lines, clippy::unnested_or_patterns)]
pub fn suggest_patch(finding: &Finding) -> Option<PatchSuggestion> {
    let mnemonic = finding.mnemonic.to_ascii_lowercase();
    let is_jcc = mnemonic.starts_with('j') && mnemonic != "jmp";
    let is_setcc = mnemonic.starts_with("set");
    let is_cmovcc = mnemonic.starts_with("cmov");

    match (finding.verdict, &finding.kind) {
        // jcc whose taken-target is dead — NOP it.
        (SmtResult::AlwaysFalse, FindingKind::OpaquePredicate)
        | (SmtResult::AlwaysFalse, FindingKind::DeadBranch)
        | (SmtResult::AlwaysFalse, FindingKind::ConstantCondition)
            if is_jcc =>
        {
            Some(PatchSuggestion {
                address: finding.address,
                strategy: PatchStrategy::NopJcc,
                rationale: format!(
                    "`{mnemonic}` is never taken ({formula} is always false); NOP-out the conditional jump.",
                    mnemonic = finding.mnemonic,
                    formula = finding.formula,
                ),
                r2_command: format!("wa nop @ {addr}", addr = finding.address),
            })
        }
        // jcc whose taken-target is always reached — replace with jmp.
        (SmtResult::AlwaysTrue, FindingKind::OpaquePredicate)
        | (SmtResult::AlwaysTrue, FindingKind::DeadBranch)
        | (SmtResult::AlwaysTrue, FindingKind::ConstantCondition)
            if is_jcc =>
        {
            if let Some(target) = finding.taken_target {
                Some(PatchSuggestion {
                    address: finding.address,
                    strategy: PatchStrategy::ReplaceJccWithJmp,
                    rationale: format!(
                        "`{mnemonic}` is always taken ({formula} is always true); replace with an unconditional jump to {target}.",
                        mnemonic = finding.mnemonic,
                        formula = finding.formula,
                    ),
                    r2_command: format!("wa jmp {target} @ {addr}", addr = finding.address),
                })
            } else {
                Some(PatchSuggestion {
                    address: finding.address,
                    strategy: PatchStrategy::CommentOnly,
                    rationale: format!(
                        "`{mnemonic}` is always taken but its target is not statically resolved.",
                        mnemonic = finding.mnemonic,
                    ),
                    r2_command: format!(
                        "# manual: replace {mnemonic} with jmp <target> @ {addr}",
                        mnemonic = finding.mnemonic,
                        addr = finding.address,
                    ),
                })
            }
        }
        // setcc with constant result.
        (SmtResult::AlwaysTrue, _) if is_setcc => Some(PatchSuggestion {
            address: finding.address,
            strategy: PatchStrategy::ReplaceSetCcWithMovConst,
            rationale: format!(
                "`{mnemonic}` always sets its destination to 1; can be rewritten as `mov r, 1`.",
                mnemonic = finding.mnemonic,
            ),
            r2_command: format!(
                "# manual: replace {mnemonic} with mov <reg>, 1 @ {addr}",
                mnemonic = finding.mnemonic,
                addr = finding.address,
            ),
        }),
        (SmtResult::AlwaysFalse, _) if is_setcc => Some(PatchSuggestion {
            address: finding.address,
            strategy: PatchStrategy::ReplaceSetCcWithMovConst,
            rationale: format!(
                "`{mnemonic}` always sets its destination to 0; can be rewritten as `mov r, 0` (or `xor r, r`).",
                mnemonic = finding.mnemonic,
            ),
            r2_command: format!(
                "# manual: replace {mnemonic} with mov <reg>, 0 @ {addr}",
                mnemonic = finding.mnemonic,
                addr = finding.address,
            ),
        }),
        // cmovcc with constant condition.
        (SmtResult::AlwaysTrue, _) if is_cmovcc => Some(PatchSuggestion {
            address: finding.address,
            strategy: PatchStrategy::ReplaceCMovCcWithMovOrNop,
            rationale: format!(
                "`{mnemonic}` always moves the source; can be rewritten as an unconditional `mov`.",
                mnemonic = finding.mnemonic,
            ),
            r2_command: format!(
                "# manual: replace {mnemonic} with mov <dst>, <src> @ {addr}",
                mnemonic = finding.mnemonic,
                addr = finding.address,
            ),
        }),
        (SmtResult::AlwaysFalse, _) if is_cmovcc => Some(PatchSuggestion {
            address: finding.address,
            strategy: PatchStrategy::ReplaceCMovCcWithMovOrNop,
            rationale: format!(
                "`{mnemonic}` never moves; can be rewritten as `nop`.",
                mnemonic = finding.mnemonic,
            ),
            r2_command: format!("wa nop @ {addr}", addr = finding.address),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::Address;
    use r2smt_common::smt::SmtResult;
    use r2smt_core::{Confidence, Finding, FindingEvidence, FindingKind};

    use r2smt_slicer::condition::BranchCondition;
    use r2smt_slicer::slice::SliceStatus;

    use super::*;

    fn make_finding(verdict: SmtResult, kind: FindingKind, mnem: &str) -> Finding {
        Finding {
            address: Address(0x40_1050),
            function: Address(0x40_1000),
            mnemonic: mnem.into(),
            condition: BranchCondition::NotEqual,
            formula: "ZF == 0".into(),
            formula_pretty: "(ZF == 0)".into(),
            formula_z3_pretty: None,
            verdict,
            kind,
            confidence: Confidence::High,
            taken_target: Some(Address(0x40_1080)),
            fallthrough_target: Some(Address(0x40_1056)),
            operands: Vec::new(),
            is_thumb: false,
            evidence: FindingEvidence {
                slice_status: SliceStatus::Complete,
                statement_count: 0,
                input_count: 0,
                inputs: vec![],
                unknown_count: 0,
                upstream_resolved_to: None,
                oracle_agreement: None,
            },
            pseudocode: None,
        }
    }

    #[test]
    fn dead_branch_jcc_suggests_nop() {
        let f = make_finding(SmtResult::AlwaysFalse, FindingKind::DeadBranch, "jne");
        let s = suggest_patch(&f).unwrap();
        assert_eq!(s.strategy, PatchStrategy::NopJcc);
        assert!(s.r2_command.starts_with("wa nop"));
    }

    #[test]
    fn always_true_jcc_with_target_suggests_jmp() {
        let f = make_finding(SmtResult::AlwaysTrue, FindingKind::OpaquePredicate, "jne");
        let s = suggest_patch(&f).unwrap();
        assert_eq!(s.strategy, PatchStrategy::ReplaceJccWithJmp);
        assert!(s.r2_command.starts_with("wa jmp"));
    }

    #[test]
    fn always_true_jcc_without_target_falls_back_to_comment() {
        let mut f = make_finding(SmtResult::AlwaysTrue, FindingKind::OpaquePredicate, "jne");
        f.taken_target = None;
        let s = suggest_patch(&f).unwrap();
        assert_eq!(s.strategy, PatchStrategy::CommentOnly);
        assert!(s.r2_command.starts_with("# manual"));
    }

    #[test]
    fn setcc_constant_yields_setcc_strategy() {
        let f = make_finding(
            SmtResult::AlwaysFalse,
            FindingKind::ConstantCondition,
            "sete",
        );
        let s = suggest_patch(&f).unwrap();
        assert_eq!(s.strategy, PatchStrategy::ReplaceSetCcWithMovConst);
    }

    #[test]
    fn cmovcc_always_false_yields_nop_strategy() {
        let f = make_finding(
            SmtResult::AlwaysFalse,
            FindingKind::ConstantCondition,
            "cmovne",
        );
        let s = suggest_patch(&f).unwrap();
        assert_eq!(s.strategy, PatchStrategy::ReplaceCMovCcWithMovOrNop);
        assert!(s.r2_command.starts_with("wa nop"));
    }

    #[test]
    fn real_branch_yields_no_suggestion() {
        let f = make_finding(SmtResult::BothPossible, FindingKind::RealBranch, "jne");
        assert!(suggest_patch(&f).is_none());
    }

    #[test]
    fn suspicious_yields_no_suggestion() {
        let f = make_finding(SmtResult::Timeout, FindingKind::SuspiciousButUnknown, "jne");
        assert!(suggest_patch(&f).is_none());
    }
}
