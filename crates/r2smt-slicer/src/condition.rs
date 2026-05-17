//! Mapping from x86 / `x86_64` and `AArch64` conditional mnemonics
//! to symbolic flag predicates.
//!
//! Used by the branch collector to label each candidate with a
//! solver-ready condition. The same enum drives the SSA→SMT
//! translation: the [`BranchCondition`] enum is arch-agnostic
//! (semantic: "above or equal", "less than", …) while
//! [`crate::lift::lift_branch_condition`] picks the right flag
//! predicate per architecture.
//!
//! **`AArch64` coverage**: the `b.<cond>` family (`b.eq`, `b.ne`,
//! `b.cs`/`b.hs`, `b.cc`/`b.lo`, `b.mi`, `b.pl`, `b.vs`, `b.vc`,
//! `b.hi`, `b.ls`, `b.ge`, `b.lt`, `b.gt`, `b.le`, `b.al`, `b.nv`)
//! plus the compare-and-branch family (`cbz`/`cbnz`/`tbz`/`tbnz`),
//! which surface via the parameterised [`BranchCondition::RegisterZero`]
//! / [`BranchCondition::RegisterNotZero`] / [`BranchCondition::BitZero`]
//! / [`BranchCondition::BitNotZero`] variants (the register name and
//! bit index ride on [`crate::collector::BranchCandidate`]).
//!
//! **`AArch32` (ARM 32-bit) coverage**: the suffix form `b<cond>`
//! with the AAPCS condition suffixes (`eq`/`ne`/`cs`/`hs`/`cc`/`lo`/
//! `mi`/`pl`/`vs`/`vc`/`hi`/`ls`/`ge`/`lt`/`gt`/`le`). Unconditional
//! `b`, link forms `bl`/`blx`, and indirect `bx` are excluded.

use r2smt_common::Arch;
use serde::{Deserialize, Serialize};

/// Symbolic interpretation of an x86 condition code.
///
/// Each variant corresponds to the family of mnemonics listed in
/// [`Self::from_suffix`] and exposes its flag predicate via
/// [`Self::formula`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BranchCondition {
    /// `ZF == 1` — operands compared equal.
    Equal,
    /// `ZF == 0`.
    NotEqual,
    /// `CF == 0 && ZF == 0` (unsigned).
    Above,
    /// `CF == 0` (unsigned).
    AboveOrEqual,
    /// `CF == 1` (unsigned).
    Below,
    /// `CF == 1 || ZF == 1` (unsigned).
    BelowOrEqual,
    /// `ZF == 0 && SF == OF` (signed).
    Greater,
    /// `SF == OF` (signed).
    GreaterOrEqual,
    /// `SF != OF` (signed).
    Less,
    /// `ZF == 1 || SF != OF` (signed).
    LessOrEqual,
    /// `SF == 1`.
    Sign,
    /// `SF == 0`.
    NotSign,
    /// `OF == 1`.
    Overflow,
    /// `OF == 0`.
    NotOverflow,
    /// `PF == 1`.
    ParityEven,
    /// `PF == 0`.
    ParityOdd,
    /// `(R)CX == 0` — not a flag predicate; used by `jcxz` / `jecxz` /
    /// `jrcxz` (no `setcc` / `cmovcc` counterpart).
    CxZero,
    /// `AArch64` `cbz Rn, label` — branch when `Rn == 0`. The register
    /// name lives on [`crate::collector::BranchCandidate::compare_register`]
    /// so the enum stays `Copy`.
    RegisterZero,
    /// `AArch64` `cbnz Rn, label` — branch when `Rn != 0`.
    RegisterNotZero,
    /// `AArch64` `tbz Rn, #bit, label` — branch when bit `#bit` of
    /// `Rn` is zero. Register and bit live on
    /// [`crate::collector::BranchCandidate::compare_register`] /
    /// [`crate::collector::BranchCandidate::bit_index`].
    BitZero,
    /// `AArch64` `tbnz Rn, #bit, label` — branch when bit `#bit` of
    /// `Rn` is set.
    BitNotZero,
}

/// Family of conditional instruction the candidate belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BranchKind {
    /// Conditional jump (`jcc`).
    Jcc,
    /// Conditional byte set (`setcc`).
    SetCc,
    /// Conditional move (`cmovcc`).
    CMovCc,
}

impl BranchCondition {
    /// `true` if this condition's flag predicate references a flag the
    /// lifter cannot soundly model (currently `OF` or `PF`). Used by
    /// the decision engine to downgrade confidence for signed
    /// comparisons and parity checks rather than silently letting the
    /// Unknowns propagate into a `BothPossible` verdict.
    #[must_use]
    pub const fn depends_on_unmodeled_flag(self) -> bool {
        matches!(
            self,
            Self::Greater
                | Self::GreaterOrEqual
                | Self::Less
                | Self::LessOrEqual
                | Self::Overflow
                | Self::NotOverflow
                | Self::ParityEven
                | Self::ParityOdd
        )
    }

    /// Return the symbolic flag predicate as a human-readable formula.
    #[must_use]
    pub const fn formula(self) -> &'static str {
        match self {
            Self::Equal => "ZF == 1",
            Self::NotEqual => "ZF == 0",
            Self::Above => "CF == 0 && ZF == 0",
            Self::AboveOrEqual => "CF == 0",
            Self::Below => "CF == 1",
            Self::BelowOrEqual => "CF == 1 || ZF == 1",
            Self::Greater => "ZF == 0 && SF == OF",
            Self::GreaterOrEqual => "SF == OF",
            Self::Less => "SF != OF",
            Self::LessOrEqual => "ZF == 1 || SF != OF",
            Self::Sign => "SF == 1",
            Self::NotSign => "SF == 0",
            Self::Overflow => "OF == 1",
            Self::NotOverflow => "OF == 0",
            Self::ParityEven => "PF == 1",
            Self::ParityOdd => "PF == 0",
            Self::CxZero => "(R)CX == 0",
            Self::RegisterZero => "Rn == 0",
            Self::RegisterNotZero => "Rn != 0",
            Self::BitZero => "bit(Rn, #bit) == 0",
            Self::BitNotZero => "bit(Rn, #bit) == 1",
        }
    }

    /// Parse a mnemonic suffix (the part after the family prefix:
    /// `j`, `set`, or `cmov`) and return the matching condition.
    ///
    /// Returns `None` if the suffix is not a recognised condition.
    #[must_use]
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        Some(match suffix {
            "e" | "z" => Self::Equal,
            "ne" | "nz" => Self::NotEqual,
            "a" | "nbe" => Self::Above,
            "ae" | "nb" | "nc" => Self::AboveOrEqual,
            "b" | "c" | "nae" => Self::Below,
            "be" | "na" => Self::BelowOrEqual,
            "g" | "nle" => Self::Greater,
            "ge" | "nl" => Self::GreaterOrEqual,
            "l" | "nge" => Self::Less,
            "le" | "ng" => Self::LessOrEqual,
            "s" => Self::Sign,
            "ns" => Self::NotSign,
            "o" => Self::Overflow,
            "no" => Self::NotOverflow,
            "p" | "pe" => Self::ParityEven,
            "np" | "po" => Self::ParityOdd,
            _ => return None,
        })
    }
}

/// Classify a mnemonic under `arch` as one of the supported
/// conditional families.
///
/// Returns `(kind, condition)` for `jcc` / `setcc` / `cmovcc` (x86)
/// or `b.<cond>` (`AArch64`) mnemonics, including their synonym
/// suffixes (`nz` ≡ `ne`, `cs` ≡ `hs`, `cc` ≡ `lo`, …).
/// Returns `None` for:
///
/// - unconditional `jmp` / `b` (no condition to model),
/// - `(j|e|r)cxz` (modelled separately via [`BranchCondition::CxZero`]
///   only when the caller wants to surface them),
/// - any non-conditional mnemonic or mnemonic outside `arch`.
#[must_use]
pub fn classify(mnemonic: &str, arch: Arch) -> Option<(BranchKind, BranchCondition)> {
    let normalized = mnemonic.trim().to_ascii_lowercase();
    match arch {
        Arch::X86 | Arch::X86_64 => classify_x86(&normalized),
        Arch::Aarch64 => classify_aarch64(&normalized),
        Arch::Arm => classify_aarch32(&normalized),
        _ => None,
    }
}

fn classify_aarch32(normalized: &str) -> Option<(BranchKind, BranchCondition)> {
    // AArch32 conditional branches are `b<cond>` (no dot). Plain `b`
    // is unconditional; `bl` / `bx` / `blx` are calls. Conditional
    // execution on non-branch mnemonics (`addeq` …) is deliberately
    // unsupported here — predicated data flow is out of scope.
    match normalized {
        "b" | "bl" | "blx" | "bx" => return None,
        _ => {}
    }
    let suffix = normalized.strip_prefix('b')?;
    aarch64_condition(suffix).map(|c| (BranchKind::Jcc, c))
}

fn classify_x86(normalized: &str) -> Option<(BranchKind, BranchCondition)> {
    if normalized == "jmp" {
        return None;
    }
    if let Some(suffix) = normalized.strip_prefix('j') {
        // jcxz / jecxz / jrcxz are CX-zero jumps without a flag predicate
        // and without setcc/cmov counterparts; surface them explicitly.
        if matches!(suffix, "cxz" | "ecxz" | "rcxz") {
            return Some((BranchKind::Jcc, BranchCondition::CxZero));
        }
        return BranchCondition::from_suffix(suffix).map(|c| (BranchKind::Jcc, c));
    }
    if let Some(suffix) = normalized.strip_prefix("set") {
        return BranchCondition::from_suffix(suffix).map(|c| (BranchKind::SetCc, c));
    }
    if let Some(suffix) = normalized.strip_prefix("cmov") {
        return BranchCondition::from_suffix(suffix).map(|c| (BranchKind::CMovCc, c));
    }
    None
}

fn classify_aarch64(normalized: &str) -> Option<(BranchKind, BranchCondition)> {
    // Compare-and-branch family. These do not read NZCV; the operand
    // register (and bit index for `tbz`/`tbnz`) is parsed by the
    // collector and stored on the `BranchCandidate`.
    match normalized {
        "cbz" => return Some((BranchKind::Jcc, BranchCondition::RegisterZero)),
        "cbnz" => return Some((BranchKind::Jcc, BranchCondition::RegisterNotZero)),
        "tbz" => return Some((BranchKind::Jcc, BranchCondition::BitZero)),
        "tbnz" => return Some((BranchKind::Jcc, BranchCondition::BitNotZero)),
        _ => {}
    }
    classify_aarch64_bcond(normalized)
}

fn classify_aarch64_bcond(normalized: &str) -> Option<(BranchKind, BranchCondition)> {
    // AArch64 conditional branches are spelled `b.<cond>` (with the
    // dot). r2 emits the dot verbatim. Unconditional `b` returns None;
    // `bl` / `blr` (calls), `ret` and the compare-and-branch family
    // (`cbz`/`cbnz`/`tbz`/`tbnz`) are also outside this classifier.
    let suffix = normalized.strip_prefix("b.")?;
    aarch64_condition(suffix).map(|c| (BranchKind::Jcc, c))
}

/// Map an `AArch64` condition suffix (`eq`, `ne`, `cs`, …) to the
/// semantically equivalent [`BranchCondition`].
///
/// The Arm Arm encodes the condition codes such that they translate
/// directly to the x86 semantic family — `b.cs` is "carry set",
/// which on `cmp` means "unsigned ≥", matching
/// [`BranchCondition::AboveOrEqual`]; `b.lt` is "signed <", matching
/// [`BranchCondition::Less`]; etc. The actual flag polarity
/// differences (Arm `C` vs x86 `CF`) are absorbed by the lifter:
/// `cmp_aarch64` emits flags with x86-style polarity so
/// [`crate::lift::lift_branch_condition`] needs no per-arch dispatch.
fn aarch64_condition(suffix: &str) -> Option<BranchCondition> {
    Some(match suffix {
        "eq" => BranchCondition::Equal,
        "ne" => BranchCondition::NotEqual,
        "cs" | "hs" => BranchCondition::AboveOrEqual,
        "cc" | "lo" => BranchCondition::Below,
        "mi" => BranchCondition::Sign,
        "pl" => BranchCondition::NotSign,
        "vs" => BranchCondition::Overflow,
        "vc" => BranchCondition::NotOverflow,
        "hi" => BranchCondition::Above,
        "ls" => BranchCondition::BelowOrEqual,
        "ge" => BranchCondition::GreaterOrEqual,
        "lt" => BranchCondition::Less,
        "gt" => BranchCondition::Greater,
        "le" => BranchCondition::LessOrEqual,
        // `b.al` is "always" and `b.nv` is reserved-encoded "never";
        // neither is a true conditional, so the classifier rejects
        // them. (`b.al` in practice is the unconditional branch.)
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only adapter that fixes `Arch::X86_64` so the existing
    /// x86-only assertions stay terse. The arch-aware test cases use
    /// `classify` directly.
    fn cx86(mnemonic: &str) -> Option<(BranchKind, BranchCondition)> {
        classify(mnemonic, Arch::X86_64)
    }

    #[test]
    fn jcc_synonyms_resolve_to_same_condition() {
        assert_eq!(
            classify("je", Arch::X86_64),
            Some((BranchKind::Jcc, BranchCondition::Equal))
        );
        assert_eq!(
            classify("jz", Arch::X86_64),
            Some((BranchKind::Jcc, BranchCondition::Equal))
        );
        assert_eq!(
            classify("JNE", Arch::X86_64),
            Some((BranchKind::Jcc, BranchCondition::NotEqual))
        );
        assert_eq!(
            classify("jnz", Arch::X86_64),
            Some((BranchKind::Jcc, BranchCondition::NotEqual))
        );
    }

    #[test]
    fn unsigned_family_classified() {
        assert_eq!(cx86("ja").map(|x| x.1), Some(BranchCondition::Above));
        assert_eq!(cx86("jnbe").map(|x| x.1), Some(BranchCondition::Above));
        assert_eq!(
            cx86("jae").map(|x| x.1),
            Some(BranchCondition::AboveOrEqual)
        );
        assert_eq!(
            cx86("jnc").map(|x| x.1),
            Some(BranchCondition::AboveOrEqual)
        );
        assert_eq!(cx86("jb").map(|x| x.1), Some(BranchCondition::Below));
        assert_eq!(
            cx86("jbe").map(|x| x.1),
            Some(BranchCondition::BelowOrEqual)
        );
    }

    #[test]
    fn signed_family_classified() {
        assert_eq!(cx86("jg").map(|x| x.1), Some(BranchCondition::Greater));
        assert_eq!(
            cx86("jge").map(|x| x.1),
            Some(BranchCondition::GreaterOrEqual)
        );
        assert_eq!(cx86("jl").map(|x| x.1), Some(BranchCondition::Less));
        assert_eq!(cx86("jle").map(|x| x.1), Some(BranchCondition::LessOrEqual));
    }

    #[test]
    fn miscellaneous_flag_branches() {
        assert_eq!(cx86("js").map(|x| x.1), Some(BranchCondition::Sign));
        assert_eq!(cx86("jno").map(|x| x.1), Some(BranchCondition::NotOverflow));
        assert_eq!(cx86("jp").map(|x| x.1), Some(BranchCondition::ParityEven));
        assert_eq!(cx86("jpo").map(|x| x.1), Some(BranchCondition::ParityOdd));
    }

    #[test]
    fn cx_zero_family() {
        assert_eq!(
            cx86("jcxz"),
            Some((BranchKind::Jcc, BranchCondition::CxZero))
        );
        assert_eq!(
            cx86("jecxz"),
            Some((BranchKind::Jcc, BranchCondition::CxZero))
        );
        assert_eq!(
            cx86("jrcxz"),
            Some((BranchKind::Jcc, BranchCondition::CxZero))
        );
    }

    #[test]
    fn setcc_family_classified() {
        assert_eq!(
            cx86("sete"),
            Some((BranchKind::SetCc, BranchCondition::Equal))
        );
        assert_eq!(
            cx86("setnz"),
            Some((BranchKind::SetCc, BranchCondition::NotEqual))
        );
        assert_eq!(
            cx86("setle"),
            Some((BranchKind::SetCc, BranchCondition::LessOrEqual))
        );
    }

    #[test]
    fn cmovcc_family_classified() {
        assert_eq!(
            cx86("cmovne"),
            Some((BranchKind::CMovCc, BranchCondition::NotEqual))
        );
        assert_eq!(
            cx86("cmovg"),
            Some((BranchKind::CMovCc, BranchCondition::Greater))
        );
    }

    #[test]
    fn unconditional_and_unknown_return_none() {
        assert!(cx86("jmp").is_none());
        assert!(cx86("mov").is_none());
        assert!(cx86("call").is_none());
        assert!(cx86("ret").is_none());
        // `jx` is not a valid suffix.
        assert!(cx86("jx").is_none());
    }

    // --- AArch64 ---

    #[test]
    fn aarch64_b_cond_family_resolves_to_branch_conditions() {
        let cases: &[(&str, BranchCondition)] = &[
            ("b.eq", BranchCondition::Equal),
            ("b.ne", BranchCondition::NotEqual),
            ("b.cs", BranchCondition::AboveOrEqual),
            ("b.hs", BranchCondition::AboveOrEqual),
            ("b.cc", BranchCondition::Below),
            ("b.lo", BranchCondition::Below),
            ("b.mi", BranchCondition::Sign),
            ("b.pl", BranchCondition::NotSign),
            ("b.vs", BranchCondition::Overflow),
            ("b.vc", BranchCondition::NotOverflow),
            ("b.hi", BranchCondition::Above),
            ("b.ls", BranchCondition::BelowOrEqual),
            ("b.ge", BranchCondition::GreaterOrEqual),
            ("b.lt", BranchCondition::Less),
            ("b.gt", BranchCondition::Greater),
            ("b.le", BranchCondition::LessOrEqual),
        ];
        for (mnem, expected) in cases {
            let got = classify(mnem, Arch::Aarch64);
            assert_eq!(
                got,
                Some((BranchKind::Jcc, *expected)),
                "AArch64 mnemonic {mnem} should classify as {expected:?}"
            );
        }
    }

    #[test]
    fn aarch64_unconditional_and_unsupported_return_none() {
        // Plain `b` is unconditional, `bl` / `blr` are calls, `ret` is
        // a return. Now-supported `cbz`/`cbnz`/`tbz`/`tbnz` are
        // exercised in their own test.
        for mnem in ["b", "bl", "blr", "ret"] {
            assert!(
                classify(mnem, Arch::Aarch64).is_none(),
                "{mnem} should not classify as a condition on AArch64"
            );
        }
        // `b.al` (always) and `b.nv` (reserved) are not real conditions.
        assert!(classify("b.al", Arch::Aarch64).is_none());
        assert!(classify("b.nv", Arch::Aarch64).is_none());
    }

    #[test]
    fn aarch64_compare_and_branch_family_resolves() {
        let cases: &[(&str, BranchCondition)] = &[
            ("cbz", BranchCondition::RegisterZero),
            ("cbnz", BranchCondition::RegisterNotZero),
            ("tbz", BranchCondition::BitZero),
            ("tbnz", BranchCondition::BitNotZero),
        ];
        for (mnem, expected) in cases {
            assert_eq!(
                classify(mnem, Arch::Aarch64),
                Some((BranchKind::Jcc, *expected)),
                "{mnem} classify"
            );
        }
    }

    #[test]
    fn aarch64_classify_uppercases_safely() {
        assert_eq!(
            classify("B.EQ", Arch::Aarch64),
            Some((BranchKind::Jcc, BranchCondition::Equal))
        );
    }

    #[test]
    fn x86_mnemonics_do_not_classify_under_aarch64() {
        // `je` is an x86 mnemonic; under `Arch::Aarch64` the classifier
        // must reject it to keep the slicer from picking up cross-arch
        // false positives.
        assert!(classify("je", Arch::Aarch64).is_none());
        assert!(classify("jne", Arch::Aarch64).is_none());
    }

    #[test]
    fn aarch32_b_cond_family_classifies() {
        // AArch32 conditional branches share suffixes with AArch64
        // but lack the dot.
        for (mnem, expected) in [
            ("beq", BranchCondition::Equal),
            ("bne", BranchCondition::NotEqual),
            ("blt", BranchCondition::Less),
            ("bge", BranchCondition::GreaterOrEqual),
        ] {
            assert_eq!(
                classify(mnem, Arch::Arm),
                Some((BranchKind::Jcc, expected)),
                "{mnem} classify"
            );
        }
        // Plain `b` / `bl` / `bx` / `blx` are unconditional or call.
        assert!(classify("b", Arch::Arm).is_none());
        assert!(classify("bl", Arch::Arm).is_none());
        assert!(classify("bx", Arch::Arm).is_none());
    }

    #[test]
    fn formula_is_non_empty_for_every_condition() {
        for cond in [
            BranchCondition::Equal,
            BranchCondition::NotEqual,
            BranchCondition::Above,
            BranchCondition::AboveOrEqual,
            BranchCondition::Below,
            BranchCondition::BelowOrEqual,
            BranchCondition::Greater,
            BranchCondition::GreaterOrEqual,
            BranchCondition::Less,
            BranchCondition::LessOrEqual,
            BranchCondition::Sign,
            BranchCondition::NotSign,
            BranchCondition::Overflow,
            BranchCondition::NotOverflow,
            BranchCondition::ParityEven,
            BranchCondition::ParityOdd,
            BranchCondition::CxZero,
        ] {
            assert!(!cond.formula().is_empty(), "missing formula for {cond:?}");
        }
    }
}
