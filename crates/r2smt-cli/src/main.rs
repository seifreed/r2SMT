#![deny(missing_docs)]
// A CLI is the one layer that legitimately writes to standard streams.
// The workspace lints deny `print_stdout` / `print_stderr` everywhere
// else; we relax the rule here so user-facing output and error messages
// can flow normally.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! `r2smt` command-line entrypoint.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use r2smt_core::{
    Confidence, Finding, FindingKind, classify_finding_with_pretty, dump_program, prepare_ssa,
};
use r2smt_explore::{ExploreBudget, ExploreRequest};
use r2smt_ir::Annotator;
use r2smt_ir::NameHints;
use r2smt_ir::program::Function;
use r2smt_report::Report;
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;
use tracing::{debug, error, warn};
use tracing_subscriber::EnvFilter;

mod args;
use args::{Cli, Command, SolverArg};
mod doctor;
mod render;
use render::{print_annotation_preview, print_findings_summary};
mod support;
use support::{
    attach_pseudocode, compute_findings, difflift_scope, dispatch_solver, open_provider,
    resolve_folded_branch, resolve_targets,
};
mod commands;
use commands::batch::batch;
use commands::inspect::{analyze, branches, lift, slice, ssa};
use commands::patch::{PatchCli, patch};
use commands::taint::{TaintCli, taint};

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(e) = init_tracing(cli.verbose) {
        eprintln!("r2smt: failed to initialise tracing: {e:#}");
        return ExitCode::FAILURE;
    }

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!(target: "r2smt::cli", "{err:#}");
            eprintln!("r2smt: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn init_tracing(verbosity: u8) -> Result<()> {
    let default_level = match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;
    Ok(())
}

/// Assemble [`SliceLimits`] from the shared slicing CLI flags. Every
/// subcommand that slices builds them identically; a caller without a
/// given flag passes its default (`false` / `None`).
// fn_params_excessive_bools: these are the independent slicing toggles
// threaded verbatim from the CLI into `SliceLimits`; grouping them into
// a struct would only move the same argument list to each call site.
#[allow(clippy::fn_params_excessive_bools)]
fn build_slice_limits(
    esil_flags: bool,
    max_instructions: Option<usize>,
    allow_memory: bool,
    allow_calls: bool,
    unknowns_on_truncation: bool,
    allow_join_merge: bool,
    max_blocks: Option<u32>,
) -> SliceLimits {
    let mut limits = SliceLimits {
        esil_flags,
        ..SliceLimits::default()
    };
    if let Some(n) = max_instructions {
        limits.max_instructions = n;
    }
    limits.allow_memory = allow_memory;
    limits.allow_calls = allow_calls;
    limits.unknowns_on_truncation = unknowns_on_truncation;
    limits.allow_join_merge = allow_join_merge;
    if let Some(n) = max_blocks {
        limits.max_basic_blocks = n;
    }
    limits
}

/// Assemble [`SolveOptions`] from the shared solver CLI flags, falling
/// back to the defaults when a flag is unset.
fn build_solve_options(timeout_ms: Option<u32>, rlimit: Option<u32>) -> SolveOptions {
    SolveOptions {
        timeout_ms: timeout_ms.unwrap_or(SolveOptions::default().timeout_ms),
        rlimit: rlimit.unwrap_or(SolveOptions::default().rlimit),
        ..SolveOptions::default()
    }
}

// Each subcommand dispatch is a single short block; the function is
// long because there are many subcommands. Allow the pedantic lint
// for this dispatcher specifically.
#[allow(clippy::too_many_lines)]
fn run(cli: Cli) -> Result<()> {
    if matches!(&cli.command, Command::Version) {
        println!("r2smt {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if matches!(&cli.command, Command::Doctor) {
        return doctor::doctor();
    }
    let deep = cli.deep_analysis;
    let ir_pcode = cli.ir.wants_pcode();
    let esil_flags = !cli.no_esil_flags && doctor::esil_flags_supported();
    if !cli.no_esil_flags && !esil_flags {
        warn!(
            target: "r2smt::compat",
            radare2_version = doctor::radare2_version(),
            "disabling ESIL flag lifting; radare2 6.2.0 or newer is required"
        );
    }
    match cli.command {
        Command::Version | Command::Doctor => Ok(()),
        Command::Why {
            file,
            addr,
            timeout_ms,
            max_paths,
        } => why_command(&file, deep, &addr, timeout_ms, max_paths),
        Command::Taint {
            file,
            addr,
            sources,
            concretise,
            max_instructions,
            allow_memory,
            allow_calls,
            timeout_ms,
            max_paths,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                false,
                false,
                None,
            );
            taint(
                &file,
                deep,
                &limits,
                ir_pcode,
                &TaintCli {
                    addr: &addr,
                    sources: &sources,
                    concretise,
                    wall_clock_ms: timeout_ms.unwrap_or(30_000),
                    max_paths: max_paths.unwrap_or(10_000),
                },
            )
        }
        Command::Analyze {
            file,
            dump_program: dump_flag,
            json,
        } => analyze(&file, deep, dump_flag, json.as_deref()),
        Command::Branches {
            file,
            function,
            json,
        } => branches(&file, deep, function.as_deref(), json.as_deref()),
        Command::Slice {
            file,
            at,
            function,
            max_instructions,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver: _,
            max_blocks,
            json,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                false,
                max_blocks,
            );
            slice(
                &file,
                deep,
                at.as_deref(),
                function.as_deref(),
                &limits,
                json.as_deref(),
            )
        }
        Command::Lift {
            file,
            at,
            function,
            max_instructions,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver: _,
            max_blocks,
            json,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                false,
                max_blocks,
            );
            lift(
                &file,
                deep,
                at.as_deref(),
                function.as_deref(),
                &limits,
                json.as_deref(),
            )
        }
        Command::Annotate {
            file,
            at,
            function,
            max_instructions,
            timeout_ms,
            rlimit,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver,
            max_blocks,
            min_confidence,
            dry_run,
            r2_script,
            save_project,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                false,
                max_blocks,
            );
            let options = build_solve_options(timeout_ms, rlimit);
            let plan = AnnotatePlan {
                min_confidence: min_confidence.to_confidence(),
                dry_run,
                r2_script,
                save_project: save_project.as_deref(),
            };
            annotate(
                &file,
                deep,
                at.as_deref(),
                function.as_deref(),
                &limits,
                options,
                &plan,
                solver,
                ir_pcode,
            )
        }
        Command::Patch {
            file,
            at,
            function,
            max_instructions,
            timeout_ms,
            rlimit,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver,
            max_blocks,
            min_confidence,
            apply,
            backup,
            manifest,
            rollback,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                false,
                max_blocks,
            );
            let options = build_solve_options(timeout_ms, rlimit);
            let cfg = PatchCli {
                min_confidence: min_confidence.to_confidence(),
                apply,
                backup: backup.as_deref(),
                manifest: manifest.as_deref(),
                rollback,
                solver,
            };
            patch(
                &file,
                deep,
                at.as_deref(),
                function.as_deref(),
                &limits,
                options,
                &cfg,
                ir_pcode,
            )
        }
        Command::Solve {
            file,
            at,
            function,
            max_instructions,
            timeout_ms,
            rlimit,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver,
            max_blocks,
            min_confidence,
            include_real,
            include_suspicious,
            json,
            markdown,
            r2_script,
            with_decompiler,
            allow_join_merge,
            differential_lift,
            max_difflift_comparisons,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                allow_join_merge,
                max_blocks,
            );
            let options = build_solve_options(timeout_ms, rlimit);
            let filters = SolveFilters {
                min_confidence: min_confidence.to_confidence(),
                include_real,
                include_suspicious,
            };
            let outputs = SolveOutputs {
                json: json.as_deref(),
                markdown: markdown.as_deref(),
                r2_script: r2_script.as_deref(),
            };
            let difflift = DiffLiftOptions {
                enabled: differential_lift,
                max_comparisons: max_difflift_comparisons
                    .unwrap_or(DEFAULT_MAX_DIFFLIFT_COMPARISONS),
            };
            solve(
                &file,
                deep,
                at.as_deref(),
                function.as_deref(),
                &limits,
                options,
                &filters,
                &outputs,
                solver,
                with_decompiler,
                ir_pcode,
                &difflift,
            )
        }
        Command::Batch {
            dir,
            threads,
            max_instructions,
            timeout_ms,
            rlimit,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver,
            max_blocks,
            json,
            markdown,
            with_decompiler,
            allow_join_merge,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                allow_join_merge,
                max_blocks,
            );
            let options = build_solve_options(timeout_ms, rlimit);
            batch(
                &dir,
                deep,
                threads,
                &limits,
                options,
                solver,
                with_decompiler,
                ir_pcode,
                json.as_deref(),
                markdown.as_deref(),
            )
        }
        Command::At {
            file,
            addr,
            patch: do_patch,
            timeout_ms,
            rlimit,
            max_instructions,
            allow_memory,
            allow_calls,
            solver,
            with_decompiler,
            quiet,
            explain,
            allow_join_merge,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                false,
                allow_join_merge,
                None,
            );
            let options = build_solve_options(timeout_ms, rlimit);
            let verbosity = if quiet {
                AtVerbosity::Quiet
            } else if explain {
                AtVerbosity::Explain
            } else {
                AtVerbosity::Normal
            };
            at_command(
                &file,
                deep,
                &addr,
                &limits,
                options,
                solver,
                &AtOptions {
                    do_patch,
                    with_decompiler,
                    ir_pcode,
                    verbosity,
                },
            )
        }
        Command::Ssa {
            file,
            at,
            function,
            max_instructions,
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver: _,
            max_blocks,
            json,
        } => {
            let limits = build_slice_limits(
                esil_flags,
                max_instructions,
                allow_memory,
                allow_calls,
                unknowns_on_truncation,
                false,
                max_blocks,
            );
            ssa(
                &file,
                deep,
                at.as_deref(),
                function.as_deref(),
                &limits,
                json.as_deref(),
            )
        }
    }
}

struct SolveFilters {
    min_confidence: Confidence,
    include_real: bool,
    include_suspicious: bool,
}

struct AnnotatePlan<'a> {
    min_confidence: Confidence,
    dry_run: bool,
    r2_script: bool,
    save_project: Option<&'a str>,
}

/// How much `at_command` prints below the one-line verdict.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AtVerbosity {
    /// Verdict line only (ideal for sweeps).
    Quiet,
    /// Verdict plus decompiled context (when fetched).
    Normal,
    /// Also the solver-simplified formula and slice evidence.
    Explain,
}

/// Output / mode toggles for [`at_command`], grouped so the function
/// keeps a narrow signature.
struct AtOptions {
    /// Apply the conservative patch when the verdict is actionable at
    /// high confidence.
    do_patch: bool,
    /// Fetch and print decompiler pseudocode for the owning function.
    with_decompiler: bool,
    /// Attach r2ghidra P-code IR (prefer it; ESIL fallback).
    ir_pcode: bool,
    /// How much detail to print below the verdict.
    verbosity: AtVerbosity,
}

/// Interactive single-branch entrypoint (`r2smt at`). Reuses the full
/// solve pipeline (folded-branch re-derivation + budget retry +
/// optimizer + SMT) for exactly the branch at `addr`, prints a compact
/// one-line verdict suitable for a live r2 shell-out, and — when
/// `opts.do_patch` and the verdict is actionable at `high` confidence
/// — delegates to the existing patch path (backup + manifest next to
/// the binary). Analysis logic stays in the use-case crates; this is
/// a thin combinator.
fn why_command(
    file: &Path,
    deep: bool,
    addr: &str,
    timeout_ms: Option<u64>,
    max_paths: Option<u64>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }
    let target: r2smt_common::Address = addr
        .parse()
        .with_context(|| format!("invalid target address: {addr}"))?;
    let mut provider = open_provider(file, deep)?;
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;
    let request = ExploreRequest {
        binary_path: file.to_path_buf(),
        target,
        arch: program.arch,
    };
    let budget = ExploreBudget {
        wall_clock_ms: timeout_ms.unwrap_or(30_000),
        max_paths: max_paths.unwrap_or(10_000),
    };
    let result = r2smt_explore::explore(&request, budget);
    println!("{}", render::render_explore_result(target, &result));
    Ok(())
}

fn at_command(
    file: &Path,
    deep: bool,
    addr: &str,
    limits: &SliceLimits,
    options: SolveOptions,
    solver: SolverArg,
    opts: &AtOptions,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }
    let (_arch, findings) = compute_findings(
        file,
        deep,
        Some(addr),
        None,
        limits,
        options,
        solver,
        opts.with_decompiler,
        opts.ir_pcode,
    )?;
    let Some(finding) = findings.first() else {
        println!("r2smt: no conditional branch at {addr}");
        return Ok(());
    };
    println!(
        "@ {addr} {mnem}  {verdict:?}  {kind:?}/{conf:?}  {formula}",
        addr = finding.address,
        mnem = finding.mnemonic,
        verdict = finding.verdict,
        kind = finding.kind,
        conf = finding.confidence,
        formula = finding.formula_pretty,
    );
    if opts.verbosity != AtVerbosity::Quiet {
        if opts.verbosity == AtVerbosity::Explain {
            if let Some(z3) = &finding.formula_z3_pretty
                && !z3.is_empty()
                && z3.as_str() != finding.formula_pretty
            {
                println!("  solver-simplified: {z3}");
            }
            if !finding.evidence.inputs.is_empty() {
                println!("  free inputs: {}", finding.evidence.inputs.join(", "));
            }
            println!(
                "  IR statements: {stmt}, unknowns: {unk}",
                stmt = finding.evidence.statement_count,
                unk = finding.evidence.unknown_count,
            );
        }
        if let Some(code) = &finding.pseudocode {
            println!("--- decompiled context ---");
            println!("{code}");
        }
    }
    if !opts.do_patch {
        return Ok(());
    }
    if !(finding.is_actionable() && finding.confidence == Confidence::High) {
        println!(
            "  not patched: needs an actionable verdict at high confidence (got {kind:?}/{conf:?})",
            kind = finding.kind,
            conf = finding.confidence,
        );
        return Ok(());
    }
    let backup = file.with_extension("r2smt.bak");
    let manifest = file.with_extension("r2smt.manifest.json");
    let cfg = PatchCli {
        min_confidence: Confidence::High,
        apply: true,
        backup: Some(backup.as_path()),
        manifest: Some(manifest.as_path()),
        rollback: false,
        solver,
    };
    patch(
        file,
        deep,
        Some(addr),
        None,
        limits,
        options,
        &cfg,
        opts.ir_pcode,
    )
}

// Argument count crosses clippy's 7-arg threshold because every CLI
// subcommand orchestrates an end-to-end pipeline (file, analysis
// depth, address / function selectors, slice limits, solve options,
// per-command plan, solver backend). Grouping these into a single
// config struct would only shuffle field names without reducing
// surface area; the parameters are independent and read at distinct
// stages. Same rationale applies to `solve` below.
#[allow(clippy::too_many_arguments)]
fn annotate(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    options: SolveOptions,
    plan: &AnnotatePlan<'_>,
    solver: SolverArg,
    ir_pcode: bool,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
    // Honour the global `--ir` flag like every other analysis path: without
    // this the annotate pipeline runs the ESIL lifter regardless, and its
    // verdicts (and therefore the written comments / saved project) diverge
    // from `solve --ir pcode` on the same branches.
    provider.set_attach_pcode(ir_pcode);
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;
    let arch = program.arch;
    let bits = program.bits;
    let function_count = program.functions.len();

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;
    let merged_functions: Vec<Function> = ctx.all_functions().cloned().collect();

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
        findings.push(classify_finding_with_pretty(
            &ssa,
            verdict,
            z3_pretty,
            &NameHints::default(),
        ));
    }

    let actionable: Vec<Finding> = findings
        .into_iter()
        .filter(|f| {
            f.is_actionable()
                && matches!(
                    f.kind,
                    FindingKind::OpaquePredicate
                        | FindingKind::DeadBranch
                        | FindingKind::ConstantCondition
                )
                && f.confidence <= plan.min_confidence
        })
        .collect();

    let report = Report::from_findings(
        env!("CARGO_PKG_VERSION"),
        file.display().to_string(),
        arch,
        bits,
        function_count,
        actionable.clone(),
    )
    .with_radare2_version(doctor::radare2_version());
    let annotations = report.annotations(&merged_functions);

    if plan.r2_script {
        print!("{}", report.render_r2_script(&merged_functions));
        return Ok(());
    }

    println!(
        "annotations: {n} (from {act} actionable findings, min_confidence={mc:?})",
        n = annotations.len(),
        act = actionable.len(),
        mc = plan.min_confidence,
    );
    print_annotation_preview(&annotations, &merged_functions, &actionable);

    if plan.dry_run {
        println!();
        println!("dry-run: no comments applied");
        return Ok(());
    }

    let mut applied = 0usize;
    for ann in &annotations {
        provider
            .set_comment(ann.address, &ann.text)
            .with_context(|| format!("setting comment at {addr}", addr = ann.address))?;
        applied += 1;
    }
    println!();
    println!("applied: {applied} CCu comments");

    if let Some(name) = plan.save_project {
        provider
            .save_project(name)
            .with_context(|| format!("saving r2 project '{name}'"))?;
        println!("saved r2 project: {name}");
    }
    Ok(())
}

struct SolveOutputs<'a> {
    json: Option<&'a Path>,
    markdown: Option<&'a Path>,
    r2_script: Option<&'a Path>,
}

/// Whether to run the differential-lift harness, and the budget to run
/// it under.
///
/// Grouped rather than threaded as two loose parameters because the
/// criterion in this file is cohesion and not arity: both are read in
/// the same place, at the same stage, by the same block — unlike `deep`
/// / `with_decompiler` / `ir_pcode`, which are genuinely independent and
/// stay loose for that reason. It also *removes* a bool from the
/// signature rather than adding a knob to it.
struct DiffLiftOptions {
    enabled: bool,
    max_comparisons: usize,
}

// `clippy::too_many_arguments` / `fn_params_excessive_bools`: same
// rationale as `annotate` above — this is a CLI driver threading
// independent, read-at-distinct-stages knobs (`deep`,
// `with_decompiler`, `ir_pcode`) straight from parsed args to use-case
// calls. A params struct would only relocate the noise without adding
// cohesion; those booleans are genuinely independent toggles, not a
// missing abstraction. The difflift pair is the exception and *is*
// grouped — see [`DiffLiftOptions`].
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn solve(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    options: SolveOptions,
    filters: &SolveFilters,
    outputs: &SolveOutputs<'_>,
    solver: SolverArg,
    with_decompiler: bool,
    ir_pcode: bool,
    difflift: &DiffLiftOptions,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
    provider.set_attach_pcode(ir_pcode);
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;
    let arch = program.arch;
    let bits = program.bits;
    let function_count = program.functions.len();

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;
    let merged_functions: Vec<Function> = ctx.all_functions().cloned().collect();

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
        findings.push(classify_finding_with_pretty(
            &ssa,
            verdict,
            z3_pretty,
            &NameHints::default(),
        ));
    }

    if difflift.enabled {
        let scope = difflift_scope(&ctx, at, function_filter, &filtered);
        let dl = run_differential_lift(
            &scope,
            ctx.program.arch,
            solver,
            options,
            difflift.max_comparisons,
        );
        print_lifter_agreement(&dl);
        findings.extend(dl.findings);
    }

    if with_decompiler {
        attach_pseudocode(&mut provider, &mut findings);
    }

    let displayed: Vec<&Finding> = findings
        .iter()
        .filter(|f| keep_finding(f, filters))
        .collect();

    let any_file =
        outputs.json.is_some() || outputs.markdown.is_some() || outputs.r2_script.is_some();
    if any_file {
        let report = Report::from_findings(
            env!("CARGO_PKG_VERSION"),
            file.display().to_string(),
            arch,
            bits,
            function_count,
            findings.clone(),
        )
        .with_radare2_version(doctor::radare2_version());
        if let Some(path) = outputs.json {
            let json = report.render_json().context("serialising report to JSON")?;
            fs::write(path, json).with_context(|| format!("writing JSON to {}", path.display()))?;
        }
        if let Some(path) = outputs.markdown {
            let md = report.render_markdown(&merged_functions);
            fs::write(path, md)
                .with_context(|| format!("writing Markdown to {}", path.display()))?;
        }
        if let Some(path) = outputs.r2_script {
            let script = report.render_r2_script(&merged_functions);
            fs::write(path, script)
                .with_context(|| format!("writing r2 script to {}", path.display()))?;
        }
    } else {
        print_findings_summary(&findings, &displayed, at.is_some(), &merged_functions);
    }
    Ok(())
}

/// Default comparison budget for the opt-in `--differential-lift` pass.
/// Host-Side Safety: a whole-program run must not issue an unbounded
/// number of SMT queries. Overridable per run with
/// `--max-difflift-comparisons`, but never *un*bounded — the guardrail
/// is the bound existing, not its value.
///
/// Raised from 50 000 to 250 000 once memory instructions became
/// comparable, and to 500 000 once a corpus pass showed 250 000 binding:
/// `b6b46f67` of `../aarch64-neon-samples` has 495 479 instructions and
/// stopped early, so part of it was never cross-checked.
///
/// The sizing is measured, not rounded. Comparisons land at ~0.6 per
/// instruction — many instructions leave fewer than two lowerings to
/// pair — so the largest sample in the corpora needs ~300 000 and
/// 500 000 covers it with margin. Note the memo does **not** buy
/// headroom here at the rate one might assume: measured at ~1.45
/// comparisons per solver query, not the ~7 its speedup suggests, since
/// most of that speedup came from keeping Z3's shared context small
/// rather than from skipping queries.
const DEFAULT_MAX_DIFFLIFT_COMPARISONS: usize = 500_000;

/// Bounded memo of already-decided lowering pairs. Host-Side Safety:
/// the map grows with distinct instruction shapes, which a pathological
/// input could make unbounded, so past the cap it stops inserting
/// rather than evicting — a memo that stops helping, never one that
/// stops fitting.
const MAX_DIFFLIFT_MEMO_ENTRIES: usize = 50_000;

/// Rewrite the per-instruction temporary names out of a rendered
/// lowering so two appearances of the same instruction share a memo key.
///
/// The per-mnemonic lifter mints temporaries as `t_<addr>_<n>`
/// (`r2smt_slicer`'s `LiftCtx::new_temp`), so without this every
/// flag-setting instruction — the family the ESIL rung put into
/// production on x86 — would key uniquely and the memo would never hit.
/// Dropping the address is a consistent alpha-rename: `<n>` is already
/// unique within one instruction, and the other lowering's temporaries
/// carry the `__esil_` prefix, so the two cannot collide. The
/// replacement is spelled `t_$_<n>` because `$` appears in no register
/// or variable name either lifter produces, so a canonicalised name can
/// never alias a real one.
///
/// Constants are deliberately untouched: a pc-relative lowering carries
/// its address as a value, so the same instruction at two addresses
/// keeps two keys — which is correct, since it is two different
/// equivalence questions.
fn canonical_temp_names(rendered: &str) -> String {
    let bytes = rendered.as_bytes();
    let mut out = String::with_capacity(rendered.len());
    let (mut copied, mut i) = (0usize, 0usize);
    while i + 2 <= bytes.len() {
        let starts_name = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if !(starts_name && bytes[i] == b't' && bytes[i + 1] == b'_') {
            i += 1;
            continue;
        }
        let hex_start = i + 2;
        let hex_end = hex_start
            + bytes[hex_start..]
                .iter()
                .take_while(|b| b.is_ascii_hexdigit())
                .count();
        if hex_end > hex_start && bytes.get(hex_end) == Some(&b'_') {
            out.push_str(&rendered[copied..i]);
            out.push_str("t_$_");
            i = hex_end + 1;
            copied = i;
        } else {
            i += 1;
        }
    }
    out.push_str(&rendered[copied..]);
    out
}

/// Memo key for one pair of lowerings. `\u{1}` separates the sides so
/// no rendering can straddle the boundary and alias another pair.
fn difflift_memo_key(a: &[r2smt_ir::IrStmt], b: &[r2smt_ir::IrStmt]) -> String {
    canonical_temp_names(&format!("{a:?}\u{1}{b:?}"))
}

/// Outcome of the differential-lift pass: the engine-integrity
/// findings, the running agreement tally, how many pairwise
/// comparisons were attempted, how many of those reached the solver,
/// and whether the budget cut the scan short of the program.
struct DiffLiftRun {
    findings: Vec<Finding>,
    stats: r2smt_difflift::AgreementStats,
    compared: usize,
    /// Memo misses, i.e. the number of distinct SMT queries actually
    /// issued. Reported separately from `compared` because the memo
    /// divorced the two, and the comparison budget is one
    /// whose stated rationale is about *queries*: without this number
    /// there is no way to tell what raising that cap would cost.
    queries: usize,
    truncated: bool,
    /// The budget this run actually applied. Carried so the truncation
    /// note names the number in force rather than the default, which
    /// would misreport every run that passed the flag.
    cap: usize,
}

/// Cross-check every instruction's independent lowerings. A proven
/// disagreement yields one `lifter_disagreement` finding for that
/// instruction; the agreement tally feeds the reported metric. The
/// solve is delegated through the user-selected backend, exactly like
/// the branch pipeline.
///
/// Decided pairs are memoised: a real binary spells the same
/// instruction hundreds of times and each appearance used to pay its
/// own solver query. The tally is recorded on every comparison, hit or
/// miss, so `compared` still counts the program and not the cache.
///
/// The soundness argument is about *direction*, not purity, and the
/// difference is worth stating because the obvious claim is wrong.
/// A cached `Agree` or `Disagree` is a completed proof — the solver
/// returned unsat or produced a model — so replaying it cannot
/// fabricate. What budget pressure produces is `Unknown`, i.e.
/// [`r2smt_difflift::DiffVerdict::Inconclusive`], so the only verdict
/// the cache can pin prematurely is the fail-closed one, which loses
/// precision and never hides a disagreement.
///
/// That is not hypothetical, and the measurement is worth keeping
/// because the naive reading of it is an accusation against the memo.
/// One 314 KB `AArch64` sample, 7 906 comparisons, four runs:
///
/// | memo | `--rlimit` | agree | disagree | inconclusive | wall |
/// |---|---|---|---|---|---|
/// | on | 2 000 000 | 3 829 | **7** | 4 070 | 9 s |
/// | off | 2 000 000 | 3 796 | **7** | 4 103 | 63 s |
/// | on | unset | 4 040 | **7** | 3 859 | 75 s |
/// | off | unset | 4 041 | **7** | 3 858 | 11 min 30 s |
///
/// At a *fixed* `--rlimit` the memo appears to decide 33 more
/// comparisons, which looks like the cache changing answers. With the
/// budget unset the two differ by **one**, which is what says it is
/// not: Z3's per-check resource limit is not independent of what the
/// shared thread-local context has already accumulated, so without the
/// memo the repeats spend budget the later checks then lack. The memo
/// relieves that pressure — it makes the harness *more* sensitive, not
/// less. `disagree` is the number that must not move, and it is 7 in
/// all four.
///
/// The last row is also why a corpus pass was not viable: this is the
/// sample that "did not finish in 10 minutes".
fn run_differential_lift(
    functions: &[Function],
    arch: r2smt_common::Arch,
    solver: SolverArg,
    options: SolveOptions,
    cap: usize,
) -> DiffLiftRun {
    let mut stats = r2smt_difflift::AgreementStats::default();
    let mut findings: Vec<Finding> = Vec::new();
    let mut compared = 0usize;
    let mut queries = 0usize;
    let mut truncated = false;
    let mut memo: HashMap<String, r2smt_difflift::DiffVerdict> = HashMap::new();
    'outer: for func in functions {
        for block in &func.blocks {
            for insn in &block.instructions {
                // Checked per *instruction*, not per pair: a mid-pair
                // break abandoned the disagreements already found for
                // the instruction in flight, since the finding is only
                // pushed once every pair has been compared.
                if compared >= cap {
                    truncated = true;
                    break 'outer;
                }
                let lowerings = r2smt_difflift::lower_all(insn, arch);
                let bodies: Vec<(r2smt_difflift::Lowering, &[r2smt_ir::IrStmt])> =
                    lowerings.available().collect();
                let mut disagreeing: Vec<String> = Vec::new();
                for (i, (_, sa)) in bodies.iter().enumerate() {
                    for (lb, sb) in &bodies[i + 1..] {
                        compared += 1;
                        let key = difflift_memo_key(sa, sb);
                        let verdict = memo.get(&key).copied().unwrap_or_else(|| {
                            queries += 1;
                            let fresh = compare_lowerings(sa, sb, arch, solver, options);
                            if memo.len() < MAX_DIFFLIFT_MEMO_ENTRIES {
                                memo.insert(key, fresh);
                            }
                            fresh
                        });
                        stats.record(verdict);
                        if verdict == r2smt_difflift::DiffVerdict::Disagree {
                            dump_disagreement(insn, bodies[i].0, sa, *lb, sb);
                            disagreeing.push(format!(
                                "{a} vs {b}",
                                a = bodies[i].0.as_str(),
                                b = lb.as_str(),
                            ));
                        }
                    }
                }
                if !disagreeing.is_empty() {
                    findings.push(r2smt_core::lifter_disagreement_finding(
                        insn.address,
                        func.address,
                        insn.mnemonic.clone(),
                        format!(
                            "lifter disagreement on `{mnem}`: {pairs}",
                            mnem = insn.mnemonic,
                            pairs = disagreeing.join(", "),
                        ),
                    ));
                }
            }
        }
    }
    if truncated {
        warn!(
            target: "r2smt::difflift",
            compared,
            queries,
            cap,
            "comparison budget exhausted — the rest of the program was not cross-checked"
        );
    }
    DiffLiftRun {
        findings,
        stats,
        compared,
        queries,
        truncated,
        cap,
    }
}

/// Emit both disagreeing lowerings side by side for triage.
///
/// The finding carries only which two engines disagreed, which names
/// the symptom; deciding *whose* bug it is means reading the two IR
/// trees against the architecture manual, and every attribution of the
/// four currently-known disagreements was made that way. This is a
/// `debug!` and not a flag because the observability contract already
/// owns the question — full content lives at `debug`/`trace` — and
/// `RUST_LOG=r2smt::difflift=debug` is the switch that already exists.
fn dump_disagreement(
    insn: &r2smt_ir::program::Instruction,
    lowering_a: r2smt_difflift::Lowering,
    a: &[r2smt_ir::IrStmt],
    lowering_b: r2smt_difflift::Lowering,
    b: &[r2smt_ir::IrStmt],
) {
    debug!(
        target: "r2smt::difflift",
        address = %insn.address,
        mnemonic = %insn.mnemonic,
        thumb = insn.is_thumb,
        esil = insn.esil.as_deref().unwrap_or("<none>"),
        "{a_name}:\n{a_body}\n{b_name}:\n{b_body}",
        a_name = lowering_a.as_str(),
        a_body = render_lowering(a),
        b_name = lowering_b.as_str(),
        b_body = render_lowering(b),
    );
}

/// One IR statement per line, indented, for the triage dump.
fn render_lowering(statements: &[r2smt_ir::IrStmt]) -> String {
    statements
        .iter()
        .map(|s| format!("    {s}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Resolve one pairwise lowering comparison. A solver-backend error
/// (e.g. a missing `cvc5`) is treated as `Inconclusive` — fail-closed,
/// never `Agree`, and never aborts the surrounding run.
fn compare_lowerings(
    a: &[r2smt_ir::IrStmt],
    b: &[r2smt_ir::IrStmt],
    arch: r2smt_common::Arch,
    solver: SolverArg,
    options: SolveOptions,
) -> r2smt_difflift::DiffVerdict {
    match r2smt_difflift::build_equivalence_query(a, b, arch) {
        None => r2smt_difflift::DiffVerdict::Inconclusive,
        Some(query) => match dispatch_solver(solver, &query, options) {
            Ok((verdict, _)) => r2smt_difflift::classify_equivalence(verdict),
            Err(_) => r2smt_difflift::DiffVerdict::Inconclusive,
        },
    }
}

/// Print the lifter-agreement metric line (the P22 deliverable).
///
/// The truncation note is part of the metric, not decoration: without
/// it a budget-limited scan reports coverage it never had.
///
/// `queries` is reported beside `compared` because the memo separated
/// them: `compared` says how much of the program was covered, `queries`
/// how much solver work that cost. Only the second is comparable to the
/// budget's own stated rationale, and only the ratio between them says
/// what raising the comparison budget would buy.
///
/// The exact shape of this line is a contract with
/// `scripts/difflift-corpus.py`, whose `AGREEMENT` regex parses it. That
/// script fails closed when it cannot match — reporting the sample as
/// having produced no tally rather than as a zero — so a change here
/// without a matching change there reads as a corpus that would not
/// measure.
fn print_lifter_agreement(run: &DiffLiftRun) {
    let rate = run
        .stats
        .agreement_rate()
        .map_or_else(|| "n/a".to_string(), |r| format!("{:.2}%", r * 100.0));
    let scope = if run.truncated {
        format!(
            " — TRUNCATED at the {cap} budget, the rest of the program was not compared",
            cap = run.cap,
        )
    } else {
        String::new()
    };
    println!(
        "lifter-agreement: {rate} (agree={a} disagree={d} inconclusive={i}) over {compared} comparisons, {queries} solver queries{scope}",
        a = run.stats.agree,
        d = run.stats.disagree,
        i = run.stats.inconclusive,
        compared = run.compared,
        queries = run.queries,
    );
}

fn keep_finding(finding: &Finding, filters: &SolveFilters) -> bool {
    match finding.kind {
        FindingKind::RealBranch => filters.include_real,
        FindingKind::OpaquePredicate | FindingKind::DeadBranch | FindingKind::ConstantCondition => {
            finding.confidence <= filters.min_confidence
        }
        // `FindingKind` is `#[non_exhaustive]`; SuspiciousButUnknown and
        // any future unknown variants gate on `--include-suspicious`.
        _ => filters.include_suspicious,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::{canonical_temp_names, difflift_memo_key};
    use r2smt_ir::expr::{Expr, Var};
    use r2smt_ir::stmt::IrStmt;

    fn assign(name: &str, value: u128) -> Vec<IrStmt> {
        vec![IrStmt::Assign {
            dst: Var::new(name, 64),
            src: Expr::konst(value, 64),
        }]
    }

    #[test]
    fn test_the_same_instruction_at_two_addresses_shares_a_memo_key() {
        assert_eq!(
            difflift_memo_key(&assign("t_1000_0", 5), &assign("rax", 5)),
            difflift_memo_key(&assign("t_2000_0", 5), &assign("rax", 5)),
        );
    }

    #[test]
    fn test_a_pc_relative_lowering_keeps_one_key_per_address() {
        // The address survives as a *constant*, which the canonicaliser
        // must not touch: two addresses are two equivalence questions.
        assert_ne!(
            difflift_memo_key(&assign("t_1000_0", 0x1008), &assign("rax", 0x1008)),
            difflift_memo_key(&assign("t_2000_0", 0x2008), &assign("rax", 0x2008)),
        );
    }

    #[test]
    fn test_two_temporaries_of_one_instruction_stay_distinct() {
        assert_ne!(
            canonical_temp_names("t_1000_0"),
            canonical_temp_names("t_1000_1"),
        );
    }

    #[test]
    fn test_the_canonical_name_cannot_alias_a_real_variable() {
        // `$` is in no name either lifter mints, so the rewrite of
        // `t_<addr>_<n>` can never collide with a register spelling.
        assert_eq!(canonical_temp_names("t_1000_0"), "t_$_0");
    }

    #[test]
    fn test_a_name_without_the_temporary_shape_is_left_alone() {
        assert_eq!(canonical_temp_names("t_zz_0"), "t_zz_0");
        assert_eq!(canonical_temp_names("__esil_old_0"), "__esil_old_0");
    }

    #[test]
    fn test_the_two_sides_of_a_key_cannot_straddle_the_separator() {
        assert_ne!(
            difflift_memo_key(&assign("rax", 1), &assign("rbx", 2)),
            difflift_memo_key(&assign("rbx", 2), &assign("rax", 1)),
        );
    }
}
