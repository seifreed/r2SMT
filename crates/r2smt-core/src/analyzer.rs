//! Top-level analyzer configuration and (forthcoming) orchestration
//! entrypoints.

/// Knobs that control how aggressive the analyzer is.
///
/// Defaults match the v0 limits documented in `SPEC.md` §5.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyzerConfig {
    /// Maximum instructions per backward slice.
    pub max_slice_instructions: u32,
    /// Maximum basic blocks a slice may cross.
    pub max_basic_blocks: u32,
    /// Whether memory loads / stores are followed during slicing.
    pub allow_memory: bool,
    /// Whether calls are followed during slicing.
    pub allow_calls: bool,
    /// Per-branch SMT solver timeout, in milliseconds.
    pub solver_timeout_ms: u32,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_slice_instructions: 32,
            max_basic_blocks: 1,
            allow_memory: false,
            allow_calls: false,
            solver_timeout_ms: 500,
        }
    }
}

/// Top-level analyzer.
///
/// Phase 0/1 exposes this as a configuration holder; the analysis
/// pipeline grows in subsequent phases (collector, slicer, SSA, SMT).
#[derive(Debug, Clone, Copy)]
pub struct Analyzer {
    config: AnalyzerConfig,
}

impl Analyzer {
    /// Build an analyzer with the given configuration.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new(AnalyzerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_spec_v0() {
        let cfg = AnalyzerConfig::default();
        assert_eq!(cfg.max_slice_instructions, 32);
        assert_eq!(cfg.max_basic_blocks, 1);
        assert!(!cfg.allow_memory);
        assert!(!cfg.allow_calls);
        assert_eq!(cfg.solver_timeout_ms, 500);
    }

    #[test]
    fn analyzer_holds_provided_config() {
        let cfg = AnalyzerConfig {
            max_slice_instructions: 64,
            ..AnalyzerConfig::default()
        };
        let analyzer = Analyzer::new(cfg);
        assert_eq!(analyzer.config().max_slice_instructions, 64);
    }
}
