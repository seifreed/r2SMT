//! Differential-oracle consensus: corroboration plumbing that is
//! *structurally incapable of upgrading a verdict*.
//!
//! r2SMT owns the sound stack. An optional second opinion from an
//! independent differential oracle (a separate lifter + solver,
//! wired in a later phase) may **corroborate** a sound verdict or
//! **flag a disagreement** — it may never flip a verdict's polarity,
//! fabricate one, or raise its confidence. Those guarantees are
//! enforced by *construction* here, not by convention:
//!
//! - [`reconcile_with_oracle`] returns only a (possibly lowered)
//!   [`Confidence`] plus optional metadata. It is handed the
//!   already-decided sound verdict and **cannot return a different
//!   one** — the type signature has no verdict output, so an oracle
//!   physically cannot change `Finding::verdict` / `Finding::kind`.
//! - Every return path's confidence is either the input `sound_conf`
//!   unchanged or `sound_conf.max(Confidence::Low)`. Because the
//!   [`Confidence`] ordering runs most-trusted → least-trusted
//!   (`High < Medium < Low < Unknown`), `max` can only move *down*
//!   the trust ladder. There is no path that yields a more-trusted
//!   confidence than the input.
//! - An oracle that is absent, or itself inconclusive, is a no-op
//!   (returns the sound confidence unchanged and no metadata), so
//!   output is byte-identical to the no-oracle pipeline.
//! - An oracle is never allowed to *promote* an inconclusive sound
//!   verdict: if the sound side was not definitive, the oracle is
//!   ignored entirely (it can corroborate a proof, not manufacture
//!   one).

use r2smt_common::smt::SmtResult;

use crate::finding::{Confidence, Finding, OracleAgreement};

/// Result of reconciling a sound verdict with an optional oracle
/// verdict. Carries no `SmtResult` — by design the oracle cannot
/// change which verdict the finding reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleReconciliation {
    /// Confidence after consulting the oracle. Guaranteed to be **no
    /// more trusted** than the `sound_conf` passed in.
    pub confidence: Confidence,
    /// Metadata to record on the finding's evidence. `None` when the
    /// oracle was absent or inconclusive — serialization skips it, so
    /// the output is byte-identical to the no-oracle path.
    pub agreement: Option<OracleAgreement>,
}

/// `true` for the three definitive verdicts the consensus gate is
/// allowed to act on. `Unsound` / `Timeout` / `Unknown` (and any
/// future `#[non_exhaustive]` variant) are *not* definitive.
fn is_definitive(verdict: SmtResult) -> bool {
    matches!(
        verdict,
        SmtResult::AlwaysTrue | SmtResult::AlwaysFalse | SmtResult::BothPossible
    )
}

/// Reconcile the authoritative `sound` verdict (already classified at
/// `sound_conf`) with an optional independent `oracle` verdict.
///
/// The policy, in order:
///
/// 1. No oracle, or a non-definitive oracle → no effect.
/// 2. Sound side itself non-definitive → no effect (an oracle may
///    corroborate a proof, never manufacture one).
/// 3. Oracle agrees with the definitive sound verdict →
///    [`OracleAgreement::Corroborated`], confidence unchanged.
/// 4. Oracle reaches a *different* definitive verdict →
///    [`OracleAgreement::Disagreed`]: the sound verdict is kept
///    verbatim, its confidence capped at [`Confidence::Low`].
#[must_use]
pub fn reconcile_with_oracle(
    sound: SmtResult,
    sound_conf: Confidence,
    oracle: Option<SmtResult>,
) -> OracleReconciliation {
    let no_effect = OracleReconciliation {
        confidence: sound_conf,
        agreement: None,
    };

    let Some(oracle) = oracle else {
        return no_effect;
    };
    if !is_definitive(oracle) || !is_definitive(sound) {
        return no_effect;
    }
    if oracle == sound {
        return OracleReconciliation {
            confidence: sound_conf,
            agreement: Some(OracleAgreement::Corroborated),
        };
    }
    OracleReconciliation {
        // `max` over `High < Medium < Low < Unknown` moves *down* the
        // trust ladder only: High/Medium → Low, Low → Low, Unknown →
        // Unknown. A disagreement can never make a finding more
        // trusted, and never better than `Low`.
        confidence: sound_conf.max(Confidence::Low),
        agreement: Some(OracleAgreement::Disagreed),
    }
}

/// Apply [`reconcile_with_oracle`] to a finding that was **already
/// built from the sound verdict**. Only `confidence` and the
/// `oracle_agreement` evidence flag are touched; `verdict`, `kind`,
/// polarity and every other field are preserved exactly. With
/// `oracle == None` this is a no-op (the finding is returned
/// unchanged and `oracle_agreement` stays `None`, which serialization
/// skips), so the no-oracle pipeline is byte-identical.
#[must_use]
pub fn reconcile_finding_with_oracle(mut finding: Finding, oracle: Option<SmtResult>) -> Finding {
    let reconciled = reconcile_with_oracle(finding.verdict, finding.confidence, oracle);
    finding.confidence = reconciled.confidence;
    finding.evidence.oracle_agreement = reconciled.agreement;
    finding
}
