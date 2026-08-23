//! Shared CLI support: analysis-context plumbing and provider
//! lifecycle helpers used by every subcommand module.

use std::path::Path;

use anyhow::{Context, Result};
use r2smt_common::{Address, Arch};
use r2smt_core::{
    AnalysisRequest, AnalysisResult, AnalysisTarget, Analyzer, Finding, dump_program, prepare_ssa,
};
use r2smt_ir::BinaryProvider;
use r2smt_ir::Decompiler;
use r2smt_ir::program::{Function, Program};
use r2smt_r2pipe::{AnalysisLevel, R2PipeProvider};
use r2smt_report::{AnalysisOptions, IrPolicy, Report, ReportMetadata, SolverProvenance};
use r2smt_slicer::{BranchCandidate, SliceLimits, collect_branches, collect_function_branches};
use r2smt_smt::SolveOptions;

use crate::args::SolverArg;
use crate::doctor;
use crate::render::truncate_on_char_boundary;
use crate::worker::{
    AnalysisWorkerRequest, MAX_BRANCHES, MAX_FUNCTIONS, MEMORY_MIB, WALL_CLOCK_MS, run_isolated,
};

/// Inputs needed to stamp a stable report with the exact analysis policy.
// Mirrors independently named CLI flags into a serde DTO; replacing
// them with enums would obscure the JSON provenance without reducing state.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct ReportRun<'a> {
    pub(crate) file: &'a Path,
    pub(crate) deep: bool,
    pub(crate) at: Option<&'a str>,
    pub(crate) function_filter: Option<&'a str>,
    pub(crate) limits: &'a SliceLimits,
    pub(crate) options: SolveOptions,
    pub(crate) solver: SolverArg,
    pub(crate) with_decompiler: bool,
    pub(crate) ir_pcode: bool,
    pub(crate) differential_lift: bool,
}

pub(crate) fn report_from_findings(
    run: &ReportRun<'_>,
    arch: Arch,
    bits: u16,
    functions_analyzed: usize,
    findings: Vec<Finding>,
) -> Result<Report> {
    let branch = run
        .at
        .map(str::parse)
        .transpose()
        .with_context(|| "parsing report branch scope")?;
    let function = run
        .function_filter
        .map(str::parse)
        .transpose()
        .with_context(|| "parsing report function scope")?;
    let metadata = ReportMetadata {
        radare2_version: doctor::radare2_version().to_string(),
        r2ghidra_version: doctor::r2ghidra_version().to_string(),
        solver: SolverProvenance {
            name: solver_name(run.solver).to_string(),
            version: doctor::solver_version(run.solver).to_string(),
            timeout_ms: run.options.timeout_ms,
            rlimit: run.options.rlimit,
            random_seed: run.options.random_seed,
        },
        ir_policy: if run.ir_pcode {
            IrPolicy::PcodeEsilMnemonic
        } else {
            IrPolicy::EsilMnemonic
        },
        binary_sha256: r2smt_patch::sha256_hex(run.file)?,
        analysis_options: AnalysisOptions {
            deep: run.deep,
            branch,
            function,
            max_slice_instructions: run.limits.max_instructions,
            max_basic_blocks: run.limits.max_basic_blocks,
            allow_memory: run.limits.allow_memory,
            allow_calls: run.limits.allow_calls,
            unknowns_on_truncation: run.limits.unknowns_on_truncation,
            allow_join_merge: run.limits.allow_join_merge,
            esil_flags: run.limits.esil_flags,
            differential_lift: run.differential_lift,
            with_decompiler: run.with_decompiler,
            worker_isolated: true,
            worker_wall_clock_ms: WALL_CLOCK_MS,
            worker_memory_mib: MEMORY_MIB,
            max_functions: MAX_FUNCTIONS,
            max_branches: MAX_BRANCHES,
        },
    };
    Ok(Report::from_findings(
        env!("CARGO_PKG_VERSION"),
        run.file.display().to_string(),
        arch,
        bits,
        functions_analyzed,
        findings,
    )
    .with_metadata(metadata))
}

const fn solver_name(solver: SolverArg) -> &'static str {
    match solver {
        SolverArg::Z3 => "z3",
        SolverArg::Cvc5 => "cvc5",
        SolverArg::Bitwuzla => "bitwuzla",
    }
}

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

/// The functions the differential-lift harness should cross-check.
///
/// [`resolve_targets`] narrows the *branch candidates* and never touches
/// `ctx.program.functions`, so the harness used to walk the whole binary
/// even under `--at` / `--function`. Triaging one disagreement then cost
/// a full re-run of the program.
///
/// `--function` resolves against the program directly rather than
/// through `candidates`, so a function holding no conditional branch is
/// still cross-checked. `--at` reads the containing function off the
/// single candidate [`resolve_targets`] kept, which is also what carries
/// the synthesised block when the address lies outside any analysed
/// function.
pub(crate) fn difflift_scope(
    result: &AnalysisResult,
    at: Option<&str>,
    function_filter: Option<&str>,
) -> Vec<Function> {
    let target = function_filter
        .and_then(|raw| raw.parse::<Address>().ok())
        .or_else(|| {
            result
                .findings
                .first()
                .map(|finding| finding.function)
                .filter(|_| at.is_some())
        });
    match target.and_then(|addr| result.find_function(addr)) {
        Some(function) => vec![function.clone()],
        None => result.all_functions().cloned().collect(),
    }
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
    let result = run_isolated(&AnalysisWorkerRequest::new(
        file,
        deep,
        at,
        function_filter,
        limits,
        options,
        solver,
        with_decompiler,
        ir_pcode,
    )?)?;
    let arch = result.program.arch;
    Ok((arch, result.findings))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_provider_with_caps(
    provider: &mut dyn BinaryProvider,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    options: SolveOptions,
    solver: SolverArg,
    max_functions: usize,
    max_branches: usize,
) -> Result<AnalysisResult> {
    let target = match (at, function_filter) {
        (Some(raw), None) => AnalysisTarget::Branch(
            raw.parse()
                .with_context(|| format!("parsing --at value '{raw}'"))?,
        ),
        (None, Some(raw)) => AnalysisTarget::Function(
            raw.parse()
                .with_context(|| format!("parsing --function value '{raw}'"))?,
        ),
        (None, None) => AnalysisTarget::All,
        (Some(_), Some(_)) => anyhow::bail!("--at and --function are mutually exclusive"),
    };
    let request = AnalysisRequest {
        target,
        slicing: *limits,
        solving: options,
        max_functions,
        max_branches,
    };
    Analyzer::default()
        .analyze_request(provider, build_solver(solver).as_ref(), &request)
        .map_err(anyhow::Error::from)
}

/// Build the SSA slices for the targeted branches without solving them.
/// Shares the open → dump → resolve → `prepare_ssa` path with
/// [`compute_findings`], but returns the raw `SsaLiftedSlice`s for
/// consumers (like the taint pass) that run their own analysis.
///
/// # Errors
///
/// Propagates provider / program-load failures.
pub(crate) fn compute_slices(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    ir_pcode: bool,
) -> Result<(Arch, Vec<r2smt_ssa::SsaLiftedSlice>)> {
    let mut provider = open_provider(file, deep)?;
    provider.set_attach_pcode(ir_pcode);
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;
    let arch = program.arch;
    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;
    let mut slices = Vec::with_capacity(filtered.len());
    for cand in &filtered {
        if let Some(function) = ctx.find_function(cand.function) {
            slices.push(prepare_ssa(function, cand, limits, ctx.program.arch));
        }
    }
    Ok((arch, slices))
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
    let backend = build_solver(solver);
    match backend.solve(slice, options) {
        Ok(outcome) => Ok((outcome.verdict, outcome.formula_pretty)),
        // Byte-identical to the pre-port message: the adapter's
        // `detail()` is the original text minus the backend prefix.
        Err(err) => Err(anyhow::anyhow!(
            "{} backend: {}",
            backend.name(),
            err.detail()
        )),
    }
}

/// Composition-root factory — the only place that knows concrete
/// solver adapter types. Exhaustive structural mapping over the CLI
/// enum (no domain logic): the documented exhaustive-dispatch-table
/// exception applies.
fn build_solver(solver: SolverArg) -> Box<dyn r2smt_solver_port::Solver> {
    match solver {
        SolverArg::Z3 => Box::new(r2smt_smt::Z3Solver),
        SolverArg::Cvc5 => Box::new(r2smt_smt::Cvc5Solver),
        SolverArg::Bitwuzla => Box::new(r2smt_smt::BitwuzlaSolver),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::difflift_scope;
    use r2smt_common::{Address, Arch};
    use r2smt_core::{
        AnalysisMetrics, AnalysisProvenance, AnalysisResult, classify_lowered_upstream,
    };
    use r2smt_ir::program::{Function, Program};
    use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind};

    fn function(address: u64) -> Function {
        Function {
            address: Address::new(address),
            name: None,
            blocks: Vec::new(),
            is_thumb: false,
        }
    }

    fn result(candidates: &[BranchCandidate]) -> AnalysisResult {
        AnalysisResult {
            program: Program {
                arch: Arch::X86_64,
                bits: 64,
                entry: None,
                functions: vec![function(0x1000), function(0x2000)],
            },
            extra_functions: Vec::new(),
            findings: candidates
                .iter()
                .filter_map(classify_lowered_upstream)
                .collect(),
            provenance: AnalysisProvenance {
                solver: "test".into(),
                arch: Arch::X86_64,
                bits: 64,
            },
            metrics: AnalysisMetrics::default(),
        }
    }

    fn candidate_in(function: u64) -> BranchCandidate {
        BranchCandidate {
            address: Address::new(function + 4),
            function: Address::new(function),
            block: Address::new(function),
            kind: BranchKind::Jcc,
            mnemonic: "je".into(),
            condition: BranchCondition::Equal,
            formula: "ZF".into(),
            taken_target: Some(Address::new(function + 8)),
            fallthrough_target: Some(Address::new(function + 6)),
            compare_register: None,
            bit_index: None,
            upstream_resolved: Some(Address::new(function + 8)),
            operand_raws: Vec::new(),
            is_thumb: false,
        }
    }

    #[test]
    fn test_difflift_scope_unfiltered_covers_the_whole_program() {
        let scope = difflift_scope(&result(&[]), None, None);
        assert_eq!(scope.len(), 2);
    }

    #[test]
    fn test_difflift_scope_function_filter_keeps_only_that_function() {
        let scope = difflift_scope(&result(&[]), None, Some("0x2000"));
        assert_eq!(
            scope.iter().map(|f| f.address).collect::<Vec<_>>(),
            vec![Address::new(0x2000)]
        );
    }

    #[test]
    fn test_difflift_scope_function_filter_holds_without_any_candidate() {
        // A function with no conditional branch yields no candidate, and
        // must still be cross-checked rather than widening to the program.
        let scope = difflift_scope(&result(&[]), None, Some("0x1000"));
        assert_eq!(scope.len(), 1);
    }

    #[test]
    fn test_difflift_scope_at_keeps_the_function_containing_the_address() {
        let scope = difflift_scope(&result(&[candidate_in(0x2000)]), Some("0x2004"), None);
        assert_eq!(
            scope.iter().map(|f| f.address).collect::<Vec<_>>(),
            vec![Address::new(0x2000)]
        );
    }

    #[test]
    fn test_difflift_scope_falls_back_to_the_program_when_the_function_is_absent() {
        let scope = difflift_scope(&result(&[]), None, Some("0x9999"));
        assert_eq!(scope.len(), 2);
    }
}
