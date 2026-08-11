//! Non-suppressible provenance banner for output produced by the
//! unsound `explore` engine.
//!
//! The exploration engine ([`r2smt-explore`](https://docs.rs)) searches
//! for concrete inputs and is best-effort, not a proof. Any explore
//! output shown to a human must be wrapped through [`wrap_unsound`] so
//! its provenance is unmistakable. There is deliberately no flag to turn
//! the banner off: it is prepended unconditionally.
//!
//! This helper takes a plain `&str` and holds no dependency on
//! `r2smt-explore`, preserving the dependency fence (report must never
//! reach the engine crate).

/// The banner text prepended to every rendered explore result.
pub const UNSOUND_BANNER: &str = "\u{26a0} UNSOUND \u{2014} produced by the exploration engine \
(radius2), a best-effort search. This is NOT a verified result and must NOT be used for \
verify or patch decisions.";

/// Prepend the [`UNSOUND_BANNER`] to `body`. Non-suppressible by design.
#[must_use]
pub fn wrap_unsound(body: &str) -> String {
    format!("{UNSOUND_BANNER}\n{body}")
}

#[cfg(test)]
mod tests {
    use super::{UNSOUND_BANNER, wrap_unsound};

    #[test]
    fn test_wrap_unsound_always_prepends_banner() {
        let wrapped = wrap_unsound("stdin = \"PASS\"");
        assert!(wrapped.starts_with(UNSOUND_BANNER));
        assert!(wrapped.contains("stdin = \"PASS\""));
    }

    #[test]
    fn test_banner_names_the_engine_as_unsound() {
        assert!(UNSOUND_BANNER.contains("UNSOUND"));
        assert!(
            UNSOUND_BANNER
                .to_lowercase()
                .contains("not a verified result")
        );
    }
}
