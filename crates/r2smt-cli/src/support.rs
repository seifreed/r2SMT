//! Shared CLI support: analysis-context plumbing and provider
//! lifecycle helpers used by every subcommand module.

use std::path::Path;

use anyhow::{Context, Result};
use r2smt_common::Address;
use r2smt_ir::BinaryProvider;
use r2smt_ir::program::{Function, Program};
use r2smt_r2pipe::{AnalysisLevel, R2PipeProvider};
use r2smt_slicer::{BranchCandidate, collect_branches, collect_function_branches};

pub(crate) fn analysis_level(deep: bool) -> AnalysisLevel {
    if deep {
        AnalysisLevel::Deep
    } else {
        AnalysisLevel::Standard
    }
}

pub(crate) fn open_provider(file: &Path, deep: bool) -> Result<R2PipeProvider> {
    R2PipeProvider::open_with_analysis(file, false, analysis_level(deep))
        .with_context(|| format!("opening {} with radare2", file.display()))
}

pub(crate) fn open_provider_writable(file: &Path, deep: bool) -> Result<R2PipeProvider> {
    R2PipeProvider::open_with_analysis(file, true, analysis_level(deep))
        .with_context(|| format!("opening {} with radare2 (-w)", file.display()))
}

/// Owns the [`Program`] returned by r2 plus any synthesised functions
/// produced by the shellcode finder. Subcommands look up branches
/// against the union, so candidates created from a synthetic block
/// still resolve.
pub(crate) struct AnalysisContext {
    pub(crate) program: Program,
    pub(crate) extra_functions: Vec<Function>,
}

impl AnalysisContext {
    pub(crate) fn new(program: Program) -> Self {
        Self {
            program,
            extra_functions: Vec::new(),
        }
    }

    pub(crate) fn find_function(&self, address: Address) -> Option<&Function> {
        self.program
            .functions
            .iter()
            .find(|f| f.address == address)
            .or_else(|| self.extra_functions.iter().find(|f| f.address == address))
    }

    pub(crate) fn all_functions(&self) -> impl Iterator<Item = &Function> {
        self.program
            .functions
            .iter()
            .chain(self.extra_functions.iter())
    }
}

/// Build the list of candidate branches the user asked about, plus
/// the augmented [`AnalysisContext`] (potentially extended with a
/// synthetic block when `--at addr` points outside any analysed
/// function).
pub(crate) fn resolve_targets(
    provider: &mut R2PipeProvider,
    file: &Path,
    program: Program,
    at: Option<&str>,
    function_filter: Option<&str>,
) -> Result<(AnalysisContext, Vec<BranchCandidate>)> {
    let mut ctx = AnalysisContext::new(program);

    let candidates: Vec<BranchCandidate> = match function_filter {
        None => collect_branches(&ctx.program),
        Some(raw) => {
            let address: Address = raw
                .parse()
                .with_context(|| format!("parsing --function value '{raw}'"))?;
            let function = ctx
                .program
                .functions
                .iter()
                .find(|f| f.address == address)
                .ok_or_else(|| anyhow::anyhow!("no function at {address} in {}", file.display()))?;
            collect_function_branches(function, ctx.program.arch)
        }
    };

    let filtered: Vec<BranchCandidate> = if let Some(at_raw) = at {
        let target: Address = at_raw
            .parse()
            .with_context(|| format!("parsing --at value '{at_raw}'"))?;
        if let Some(found) = candidates.into_iter().find(|c| c.address == target) {
            vec![found]
        } else {
            // Shellcode / unanalysed region fallback: synthesise the
            // basic block around `target` and look for the branch
            // inside it.
            let func = provider
                .load_block_at(target)
                .with_context(|| format!("synthesising block at {target}"))?;
            let synth_candidates = collect_function_branches(&func, ctx.program.arch);
            let candidate = synth_candidates
                .into_iter()
                .find(|c| c.address == target)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no candidate at {target} (program had no match; synthetic block at {} had no conditional branch at the requested address)",
                        func.address,
                    )
                })?;
            ctx.extra_functions.push(func);
            vec![candidate]
        }
    } else {
        candidates
    };

    Ok((ctx, filtered))
}
