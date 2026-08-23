//! Public orchestration API for the analysis pipeline.

use std::collections::BTreeMap;

use r2smt_common::smt::SolveOptions;
use r2smt_common::{Address, Arch, Error, Result};
use r2smt_ir::program::{Function, Program};
use r2smt_ir::{BinaryProvider, NameHints};
use r2smt_slicer::{BranchCandidate, SliceLimits, collect_branches, collect_function_branches};
use r2smt_solver_port::{Solver, SolverRole};
use serde::{Deserialize, Serialize};

use crate::{
    Finding, classify_finding_with_pretty, classify_lowered_upstream, dump_program, prepare_ssa,
    reconcile_folded,
};

/// Knobs used by [`Analyzer::analyze`] when no custom request is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Scope of one analysis request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "address", rename_all = "snake_case")]
pub enum AnalysisTarget {
    /// Analyze every conditional branch discovered in the program.
    #[default]
    All,
    /// Analyze conditional branches in one function.
    Function(Address),
    /// Analyze one conditional branch, synthesizing its block if needed.
    Branch(Address),
}

/// Complete request accepted by [`Analyzer::analyze_request`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisRequest {
    /// Program scope to analyze.
    pub target: AnalysisTarget,
    /// Backward-slicing policy.
    pub slicing: SliceLimits,
    /// Solver budgets and deterministic seed.
    pub solving: SolveOptions,
    /// Maximum functions accepted from the provider before analysis.
    #[serde(default = "unbounded_items")]
    pub max_functions: usize,
    /// Maximum branch candidates accepted before solving.
    #[serde(default = "unbounded_items")]
    pub max_branches: usize,
}

const fn unbounded_items() -> usize {
    usize::MAX
}

impl Default for AnalysisRequest {
    fn default() -> Self {
        Self {
            target: AnalysisTarget::All,
            slicing: SliceLimits::default(),
            solving: SolveOptions::default(),
            max_functions: usize::MAX,
            max_branches: usize::MAX,
        }
    }
}

/// Tool-independent provenance produced by the core engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisProvenance {
    /// Stable solver backend name.
    pub solver: String,
    /// Target architecture reported by the binary provider.
    pub arch: Arch,
    /// Pointer width reported by the binary provider.
    pub bits: u16,
}

/// Counts produced by a core analysis run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    /// Functions discovered by the provider.
    pub functions_analyzed: usize,
    /// Candidate branches passed through the pipeline.
    pub branches_analyzed: usize,
    /// Findings eligible for analyst action.
    pub actionable_findings: usize,
}

/// Result of one application-level analysis request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Normalized program loaded from the provider.
    pub program: Program,
    /// Synthetic functions created for explicit branch targets.
    pub extra_functions: Vec<Function>,
    /// Classified branch findings in deterministic candidate order.
    pub findings: Vec<Finding>,
    /// Tool-independent engine provenance.
    pub provenance: AnalysisProvenance,
    /// Aggregate core counts.
    pub metrics: AnalysisMetrics,
}

impl AnalysisResult {
    /// Find a loaded or synthetic function by entry address.
    #[must_use]
    pub fn find_function(&self, address: Address) -> Option<&Function> {
        find_function(&self.program, &self.extra_functions, address)
    }

    /// Iterate over loaded and synthetic functions.
    pub fn all_functions(&self) -> impl Iterator<Item = &Function> {
        self.program.functions.iter().chain(&self.extra_functions)
    }
}

/// Top-level reusable analysis engine.
#[derive(Debug, Clone, Copy)]
pub struct Analyzer {
    config: AnalyzerConfig,
}

impl Analyzer {
    /// Build an analyzer with the given default configuration.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Return the active default configuration.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Analyze all discovered branches using this analyzer's defaults.
    ///
    /// # Errors
    ///
    /// Propagates provider and solver-adapter failures. Differential
    /// oracles are rejected because they may not author findings.
    pub fn analyze(
        &self,
        provider: &mut dyn BinaryProvider,
        solver: &dyn Solver,
    ) -> Result<AnalysisResult> {
        let max_instructions = usize::try_from(self.config.max_slice_instructions)
            .map_err(|error| Error::parse("analysis_request", error.to_string()))?;
        let request = AnalysisRequest {
            target: AnalysisTarget::All,
            slicing: SliceLimits {
                max_instructions,
                max_basic_blocks: self.config.max_basic_blocks,
                allow_memory: self.config.allow_memory,
                allow_calls: self.config.allow_calls,
                ..SliceLimits::default()
            },
            solving: SolveOptions {
                timeout_ms: self.config.solver_timeout_ms,
                ..SolveOptions::default()
            },
            max_functions: usize::MAX,
            max_branches: usize::MAX,
        };
        self.analyze_request(provider, solver, &request)
    }

    /// Analyze a caller-selected target under explicit slice and solver policy.
    ///
    /// # Errors
    ///
    /// Propagates provider and solver-adapter failures. Returns
    /// [`Error::Unsupported`] when a differential oracle is supplied as
    /// the authoritative backend, or [`Error::Parse`] when the requested
    /// function/branch does not exist.
    pub fn analyze_request(
        &self,
        provider: &mut dyn BinaryProvider,
        solver: &dyn Solver,
        request: &AnalysisRequest,
    ) -> Result<AnalysisResult> {
        if solver.role() != SolverRole::Sound {
            return Err(Error::Unsupported(
                "a differential oracle cannot author analysis findings".into(),
            ));
        }
        let program = dump_program(provider)?;
        if program.functions.len() > request.max_functions {
            return Err(Error::Unsupported(format!(
                "function limit exceeded: {} > {}",
                program.functions.len(),
                request.max_functions
            )));
        }
        let mut extra_functions = Vec::new();
        let candidates =
            resolve_candidates(provider, &program, &mut extra_functions, request.target)?;
        if candidates.len() > request.max_branches {
            return Err(Error::Unsupported(format!(
                "branch limit exceeded: {} > {}",
                candidates.len(),
                request.max_branches
            )));
        }
        let mut findings = Vec::with_capacity(candidates.len());
        let mut hint_cache: BTreeMap<Address, NameHints> = BTreeMap::new();
        for candidate in &candidates {
            if let Some(finding) = resolve_folded(
                provider,
                candidate,
                program.arch,
                &request.slicing,
                solver,
                request.solving,
            )? {
                findings.push(finding);
                continue;
            }
            let Some(function) = find_function(&program, &extra_functions, candidate.function)
            else {
                continue;
            };
            let ssa = prepare_ssa(function, candidate, &request.slicing, program.arch);
            let outcome = solver
                .solve(&ssa, request.solving)
                .map_err(|error| Error::Unsupported(error.to_string()))?;
            let hints = hint_cache
                .entry(candidate.function)
                .or_insert_with(|| provider.name_hints(candidate.function).unwrap_or_default());
            findings.push(classify_finding_with_pretty(
                &ssa,
                outcome.verdict,
                outcome.formula_pretty,
                hints,
            ));
        }
        let metrics = AnalysisMetrics {
            functions_analyzed: program.functions.len(),
            branches_analyzed: candidates.len(),
            actionable_findings: findings
                .iter()
                .filter(|finding| finding.is_actionable())
                .count(),
        };
        let provenance = AnalysisProvenance {
            solver: solver.name().to_string(),
            arch: program.arch,
            bits: program.bits,
        };
        Ok(AnalysisResult {
            program,
            extra_functions,
            findings,
            provenance,
            metrics,
        })
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new(AnalyzerConfig::default())
    }
}

fn resolve_candidates(
    provider: &mut dyn BinaryProvider,
    program: &Program,
    extra_functions: &mut Vec<Function>,
    target: AnalysisTarget,
) -> Result<Vec<BranchCandidate>> {
    match target {
        AnalysisTarget::All => Ok(collect_branches(program)),
        AnalysisTarget::Function(address) => {
            let function = program
                .functions
                .iter()
                .find(|function| function.address == address)
                .ok_or_else(|| {
                    Error::parse("analysis_target", format!("no function at {address}"))
                })?;
            Ok(collect_function_branches(function, program.arch))
        }
        AnalysisTarget::Branch(address) => {
            if let Some(candidate) = collect_branches(program)
                .into_iter()
                .find(|candidate| candidate.address == address)
            {
                return Ok(vec![candidate]);
            }
            let function = provider.load_block_at(address)?;
            let candidate = collect_function_branches(&function, program.arch)
                .into_iter()
                .find(|candidate| candidate.address == address)
                .ok_or_else(|| {
                    Error::parse(
                        "analysis_target",
                        format!("no conditional branch at {address}"),
                    )
                })?;
            extra_functions.push(function);
            Ok(vec![candidate])
        }
    }
}

fn find_function<'a>(
    program: &'a Program,
    extra_functions: &'a [Function],
    address: Address,
) -> Option<&'a Function> {
    program
        .functions
        .iter()
        .find(|function| function.address == address)
        .or_else(|| {
            extra_functions
                .iter()
                .find(|function| function.address == address)
        })
}

fn resolve_folded(
    provider: &mut dyn BinaryProvider,
    candidate: &BranchCandidate,
    arch: Arch,
    limits: &SliceLimits,
    solver: &dyn Solver,
    options: SolveOptions,
) -> Result<Option<Finding>> {
    if candidate.upstream_resolved.is_none() {
        return Ok(None);
    }
    let rederived = match provider.load_block_at(candidate.address) {
        Ok(function) => collect_function_branches(&function, arch)
            .iter()
            .find(|synthetic| synthetic.address == candidate.address)
            .map(|synthetic| {
                let ssa = prepare_ssa(&function, synthetic, limits, arch);
                let outcome = solver
                    .solve(&ssa, options)
                    .map_err(|error| Error::Unsupported(error.to_string()))?;
                Ok::<_, Error>(classify_finding_with_pretty(
                    &ssa,
                    outcome.verdict,
                    outcome.formula_pretty,
                    &NameHints::default(),
                ))
            })
            .transpose()?,
        Err(_) => None,
    };
    Ok(reconcile_folded(
        rederived,
        classify_lowered_upstream(candidate),
    ))
}

#[cfg(test)]
mod tests {
    use r2smt_common::smt::SmtResult;
    use r2smt_ir::program::{BasicBlock, Instruction};
    use r2smt_ir::testing::InMemoryProvider;
    use r2smt_solver_port::{SolverError, SolverOutcome};

    use super::*;

    struct FixedSolver;

    impl Solver for FixedSolver {
        fn solve(
            &self,
            _slice: &r2smt_ssa::SsaLiftedSlice,
            _options: SolveOptions,
        ) -> core::result::Result<SolverOutcome, SolverError> {
            Ok(SolverOutcome {
                verdict: SmtResult::AlwaysFalse,
                formula_pretty: None,
            })
        }

        fn name(&self) -> &'static str {
            "fixed"
        }

        fn role(&self) -> SolverRole {
            SolverRole::Sound
        }
    }

    #[test]
    fn default_config_is_conservative() {
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

    #[test]
    fn analyze_request_routes_a_branch_through_the_public_engine() -> Result<()> {
        let branch = Address(0x1000);
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(branch),
            functions: vec![Function {
                address: branch,
                name: Some("test".into()),
                blocks: vec![BasicBlock {
                    address: branch,
                    instructions: vec![Instruction {
                        address: branch,
                        size: 2,
                        bytes: vec![0x75, 0x02],
                        mnemonic: "jne".into(),
                        operands: Vec::new(),
                        esil: None,
                        pcode: None,
                        is_thumb: false,
                    }],
                    successors: vec![Address(0x1002), Address(0x1004)],
                }],
                is_thumb: false,
            }],
        };
        let mut provider = InMemoryProvider::new(program);
        let result = Analyzer::default().analyze_request(
            &mut provider,
            &FixedSolver,
            &AnalysisRequest {
                target: AnalysisTarget::Branch(branch),
                ..AnalysisRequest::default()
            },
        )?;
        assert_eq!(result.provenance.solver, "fixed");
        assert_eq!(result.metrics.branches_analyzed, 1);
        assert_eq!(result.findings[0].address, branch);
        Ok(())
    }
}
