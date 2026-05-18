//! P25 consensus-gate contract.
//!
//! The differential oracle may corroborate a sound verdict or flag a
//! disagreement; it must never upgrade confidence, flip polarity, or
//! fabricate a verdict, and with no oracle the pipeline is
//! byte-identical. These are proven structurally here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::Address;
use r2smt_common::smt::SmtResult;
use r2smt_core::{
    Confidence, Finding, FindingKind, OracleAgreement, lifter_disagreement_finding,
    reconcile_finding_with_oracle, reconcile_with_oracle,
};

const DEFINITIVE: [SmtResult; 3] = [
    SmtResult::AlwaysTrue,
    SmtResult::AlwaysFalse,
    SmtResult::BothPossible,
];
const NON_DEFINITIVE: [SmtResult; 3] = [SmtResult::Timeout, SmtResult::Unknown, SmtResult::Unsound];
const ALL_CONF: [Confidence; 4] = [
    Confidence::High,
    Confidence::Medium,
    Confidence::Low,
    Confidence::Unknown,
];

/// A definitive sound finding (`AlwaysTrue` / `High`) to exercise the
/// applier without pulling the slicer/SSA stack into the test.
fn high_confidence_finding() -> Finding {
    let mut f = lifter_disagreement_finding(
        Address::new(0x1000),
        Address::new(0x1000),
        "jne".to_string(),
        "fixture".to_string(),
    );
    f.verdict = SmtResult::AlwaysTrue;
    f.kind = FindingKind::OpaquePredicate;
    f.confidence = Confidence::High;
    f.evidence.oracle_agreement = None;
    f
}

#[test]
fn test_absent_oracle_is_a_no_op() {
    let r = reconcile_with_oracle(SmtResult::AlwaysTrue, Confidence::High, None);
    assert_eq!(
        (r.confidence, r.agreement),
        (Confidence::High, None),
        "absent oracle must not touch confidence or set metadata"
    );
}

#[test]
fn test_inconclusive_oracle_is_a_no_op() {
    for oracle in NON_DEFINITIVE {
        let r = reconcile_with_oracle(SmtResult::AlwaysFalse, Confidence::High, Some(oracle));
        assert_eq!(
            (r.confidence, r.agreement),
            (Confidence::High, None),
            "non-definitive oracle {oracle:?} must be a no-op"
        );
    }
}

#[test]
fn test_oracle_cannot_promote_an_inconclusive_sound_verdict() {
    // Sound side undecided, oracle definitive: the oracle must NOT
    // manufacture a corroboration — it can confirm a proof, not
    // create one.
    for sound in NON_DEFINITIVE {
        let r = reconcile_with_oracle(sound, Confidence::Unknown, Some(SmtResult::AlwaysTrue));
        assert_eq!(
            (r.confidence, r.agreement),
            (Confidence::Unknown, None),
            "oracle promoted an inconclusive sound verdict ({sound:?})"
        );
    }
}

#[test]
fn test_agreement_corroborates_without_changing_confidence() {
    for verdict in DEFINITIVE {
        let r = reconcile_with_oracle(verdict, Confidence::High, Some(verdict));
        assert_eq!(
            (r.confidence, r.agreement),
            (Confidence::High, Some(OracleAgreement::Corroborated)),
            "matching oracle for {verdict:?} must corroborate, not alter confidence"
        );
    }
}

#[test]
fn test_disagreement_caps_confidence_at_low() {
    let r = reconcile_with_oracle(
        SmtResult::AlwaysTrue,
        Confidence::High,
        Some(SmtResult::AlwaysFalse),
    );
    assert_eq!(
        (r.confidence, r.agreement),
        (Confidence::Low, Some(OracleAgreement::Disagreed)),
        "a definitive disagreement must cap confidence at Low and flag it"
    );
}

#[test]
fn test_oracle_never_upgrades_confidence() {
    // Exhaustive: across every definitive sound verdict, every
    // confidence, and every definitive oracle verdict, the resulting
    // confidence is never *more trusted* than the input. The
    // `Confidence` order runs most→least trusted, so "never more
    // trusted" is `result >= input`.
    for sound in DEFINITIVE {
        for sound_conf in ALL_CONF {
            for oracle in DEFINITIVE {
                let r = reconcile_with_oracle(sound, sound_conf, Some(oracle));
                assert!(
                    r.confidence >= sound_conf,
                    "oracle upgraded {sound_conf:?} -> {:?} (sound={sound:?}, oracle={oracle:?})",
                    r.confidence
                );
            }
        }
    }
}

#[test]
fn test_disagreement_preserves_verdict_and_kind() {
    // The structural anti-fabrication guarantee: a disagreeing oracle
    // only downgrades confidence and sets the flag — `verdict` and
    // `kind` (hence polarity) are untouched, so a `BothPossible`
    // sound finding can never become an `AlwaysX`.
    let mut finding = high_confidence_finding();
    finding.verdict = SmtResult::BothPossible;
    finding.kind = FindingKind::RealBranch;
    finding.confidence = Confidence::High;

    let out = reconcile_finding_with_oracle(finding, Some(SmtResult::AlwaysTrue));

    assert_eq!(
        (
            out.verdict,
            out.kind,
            out.confidence,
            out.evidence.oracle_agreement
        ),
        (
            SmtResult::BothPossible,
            FindingKind::RealBranch,
            Confidence::Low,
            Some(OracleAgreement::Disagreed)
        ),
        "oracle disagreement must keep verdict/kind and only cap confidence"
    );
}

#[test]
fn test_finding_applier_with_no_oracle_is_byte_identical() {
    let before = high_confidence_finding();
    let after = reconcile_finding_with_oracle(before.clone(), None);
    assert_eq!(
        serde_json::to_string(&before).expect("serialize before"),
        serde_json::to_string(&after).expect("serialize after"),
        "no-oracle reconciliation must be byte-identical"
    );
}

#[test]
fn test_none_oracle_agreement_is_omitted_from_json() {
    let json =
        serde_json::to_string(&high_confidence_finding().evidence).expect("serialize evidence");
    assert!(
        !json.contains("oracle_agreement"),
        "a None oracle_agreement must be skipped, not emitted: {json}"
    );
}

#[test]
fn test_corroborated_oracle_agreement_serializes_snake_case() {
    let mut finding = high_confidence_finding();
    finding = reconcile_finding_with_oracle(finding, Some(SmtResult::AlwaysTrue));
    let json = serde_json::to_string(&finding.evidence).expect("serialize evidence");
    assert!(
        json.contains("\"oracle_agreement\":\"corroborated\""),
        "expected snake_case oracle_agreement in {json}"
    );
}
