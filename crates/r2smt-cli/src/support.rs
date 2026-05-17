//! Shared CLI support: analysis-context plumbing and provider
//! lifecycle helpers used by every subcommand module.

use std::path::Path;

use anyhow::{Context, Result};
use r2smt_common::{Address, Arch};
use r2smt_core::{
    Finding, classify_finding_with_pretty, classify_lowered_upstream, dump_program, prepare_ssa,
    reconcile_folded,
};
use r2smt_ir::BinaryProvider;
use r2smt_ir::Decompiler;
use r2smt_ir::NameHints;
use r2smt_ir::program::{Function, Program};
use r2smt_r2pipe::{AnalysisLevel, R2PipeProvider};
use r2smt_slicer::{BranchCandidate, SliceLimits, collect_branches, collect_function_branches};
use r2smt_smt::{SolveOptions, solve_branch_with_pretty};

use crate::args::SolverArg;
use crate::render::truncate_on_char_boundary;

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

/// Per-function pseudocode byte budget. Host-Side Safety: the cache
/// is keyed by function (a finite set) and every entry is truncated
/// on a UTF-8 boundary so a pathological decompilation cannot blow
/// host RAM or the `CCu` payload.
const MAX_PSEUDOCODE_BYTES: usize = 16 * 1024;

/// Attach decompiler pseudocode to every finding, one decompile per
/// function (cached). Best-effort: a missing backend or transport
/// hiccup leaves `pseudocode` as `None` and never fails the run.
pub(crate) fn attach_pseudocode(provider: &mut R2PipeProvider, findings: &mut [Finding]) {
    let mut cache: std::collections::BTreeMap<Address, Option<String>> =
        std::collections::BTreeMap::new();
    for f in findings.iter_mut() {
        let entry = cache.entry(f.function).or_insert_with(|| {
            provider
                .pseudocode(f.function)
                .ok()
                .flatten()
                .map(|s| truncate_on_char_boundary(&s, MAX_PSEUDOCODE_BYTES))
        });
        f.pseudocode = entry.clone();
    }
}

// `clippy::too_many_arguments`: same rationale as `solve` / `annotate`
// — a CLI driver threading through independent, read-at-distinct-stages
// knobs (the `with_decompiler` opt-in is the 8th). A params struct
// would only relocate the noise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_findings(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    options: SolveOptions,
    solver: SolverArg,
    with_decompiler: bool,
    ir_pcode: bool,
) -> Result<(Arch, Vec<Finding>)> {
    let mut provider = open_provider(file, deep)?;
    provider.set_attach_pcode(ir_pcode);
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;
    let arch = program.arch;

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;

    let mut hint_cache: std::collections::BTreeMap<Address, NameHints> =
        std::collections::BTreeMap::new();
    let mut findings: Vec<Finding> = Vec::with_capacity(filtered.len());
    for cand in &filtered {
        if let Some(finding) = resolve_folded_branch(
            &mut provider,
            cand,
            ctx.program.arch,
            limits,
            solver,
            options,
        )? {
            findings.push(finding);
            continue;
        }
        let Some(function) = ctx.find_function(cand.function) else {
            continue;
        };
        let ssa = prepare_ssa(function, cand, limits, ctx.program.arch);
        let (verdict, z3_pretty) = dispatch_solver(solver, &ssa, options)?;
        let hints = hint_cache
            .entry(cand.function)
            .or_insert_with(|| provider.name_hints(cand.function).unwrap_or_default());
        findings.push(classify_finding_with_pretty(
            &ssa, verdict, z3_pretty, hints,
        ));
    }
    if with_decompiler {
        attach_pseudocode(&mut provider, &mut findings);
    }
    Ok((arch, findings))
}

/// Dispatch a single solve request to the selected backend. CVC5
/// failures (subprocess missing, garbled output) are surfaced as
/// `Err(anyhow::Error)` so the CLI can return a clear message to the
/// user; the Z3 path is infallible by contract and always succeeds.
///
/// Returns the verdict plus the C-style infix rendering of the
/// post-`aggressive_simplify` Z3 formula when the Z3 backend was
/// used; CVC5 has no Z3 AST so it returns `None`.
pub(crate) fn dispatch_solver(
    solver: SolverArg,
    slice: &r2smt_ssa::SsaLiftedSlice,
    options: SolveOptions,
) -> Result<(r2smt_common::smt::SmtResult, Option<String>)> {
    match solver {
        SolverArg::Z3 => {
            let outcome = solve_branch_with_pretty(slice, options);
            Ok((outcome.verdict, outcome.formula_z3_pretty))
        }
        SolverArg::Cvc5 => r2smt_smt::solve_branch_cvc5(slice, options)
            .map(|v| (v, None))
            .map_err(|err| match err {
                r2smt_smt::Cvc5Error::NotFound(detail) => anyhow::anyhow!(
                    "cvc5 backend: cvc5 binary not found on PATH ({detail}); install it with `brew install cvc5` / `apt install cvc5`"
                ),
                r2smt_smt::Cvc5Error::SubprocessError(detail) => {
                    anyhow::anyhow!("cvc5 backend: subprocess failed: {detail}")
                }
                r2smt_smt::Cvc5Error::UnrecognisedVerdict(out) => {
                    anyhow::anyhow!("cvc5 backend: unrecognised stdout: {out}")
                }
            }),
    }
}

/// Per-branch analysis-maturity fallback (always active).
///
/// Returns `Ok(None)` for branches radare2 did **not** fold (the
/// caller's normal slice → solve pipeline handles them unchanged). For
/// a branch `aaa` already collapsed to a single successor, it
/// independently re-derives a verdict: `BinaryProvider::load_block_at`
/// performs a *raw linear decode* of the containing block (its
/// shellcode-synthesis step does not apply r2's CFG folding), so the
/// genuine two-way `jcc` reappears and goes through the full
/// slice → optimize → SMT pipeline. The result is then reconciled
/// against the CFG shortcut via [`reconcile_folded`]: a sound SMT
/// proof wins; an inconclusive one falls back to r2's CFG verdict.
///
/// Cost is bounded — at most one extra block synthesis + one SMT
/// solve per folded branch — and accepted by design (always-on).
pub(crate) fn resolve_folded_branch(
    provider: &mut R2PipeProvider,
    cand: &BranchCandidate,
    arch: r2smt_common::Arch,
    limits: &SliceLimits,
    solver: SolverArg,
    options: SolveOptions,
) -> Result<Option<Finding>> {
    if cand.upstream_resolved.is_none() {
        return Ok(None);
    }
    let rederived = match provider.load_block_at(cand.address) {
        Ok(synth) => collect_function_branches(&synth, arch)
            .iter()
            .find(|c| c.address == cand.address)
            .map(|scand| {
                let ssa = prepare_ssa(&synth, scand, limits, arch);
                let (verdict, z3) = dispatch_solver(solver, &ssa, options)?;
                Ok::<_, anyhow::Error>(classify_finding_with_pretty(
                    &ssa,
                    verdict,
                    z3,
                    &NameHints::default(),
                ))
            })
            .transpose()?,
        // Synthesis failed (unmapped / undecodable): no re-derivation,
        // the CFG shortcut below is the only evidence available.
        Err(_) => None,
    };
    Ok(reconcile_folded(rederived, classify_lowered_upstream(cand)))
}
