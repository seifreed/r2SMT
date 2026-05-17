#![deny(missing_docs)]
// A CLI is the one layer that legitimately writes to standard streams.
// The workspace lints deny `print_stdout` / `print_stderr` everywhere
// else; we relax the rule here so user-facing output and error messages
// can flow normally.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! `r2smt` command-line entrypoint.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use r2smt_core::{
    Confidence, Finding, FindingKind, classify_finding_with_pretty, dump_program, prepare_ssa,
};
use r2smt_ir::Annotator;
use r2smt_ir::NameHints;
use r2smt_ir::program::Function;
use r2smt_report::Report;
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;
use tracing::error;
use tracing_subscriber::EnvFilter;

mod args;
use args::{Cli, Command, SolverArg};
mod render;
use render::{print_annotation_preview, print_findings_summary};
mod support;
use support::{
    attach_pseudocode, compute_findings, dispatch_solver, open_provider, resolve_folded_branch,
    resolve_targets,
};
mod commands;
use commands::batch::batch;
use commands::inspect::{analyze, branches, lift, slice, ssa};
use commands::patch::{PatchCli, patch};

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
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;
    Ok(())
}

// Each subcommand dispatch is a single short block; the function is
// long because there are many subcommands. Allow the pedantic lint
// for this dispatcher specifically.
#[allow(clippy::too_many_lines)]
fn run(cli: Cli) -> Result<()> {
    let deep = cli.deep_analysis;
    let ir_pcode = cli.ir.wants_pcode();
    match cli.command {
        Command::Version => {
            println!("r2smt {}", env!("CARGO_PKG_VERSION"));
            Ok(())
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
            let mut limits = SliceLimits::default();
            if let Some(n) = max_instructions {
                limits.max_instructions = n;
            }
            limits.allow_memory = allow_memory;
            limits.allow_calls = allow_calls;
            limits.unknowns_on_truncation = unknowns_on_truncation;
            if let Some(n) = max_blocks {
                limits.max_basic_blocks = n;
            }
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
            let mut limits = SliceLimits::default();
            if let Some(n) = max_instructions {
                limits.max_instructions = n;
            }
            limits.allow_memory = allow_memory;
            limits.allow_calls = allow_calls;
            limits.unknowns_on_truncation = unknowns_on_truncation;
            if let Some(n) = max_blocks {
                limits.max_basic_blocks = n;
            }
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
            allow_memory,
            allow_calls,
            unknowns_on_truncation,
            solver,
            max_blocks,
            min_confidence,
            dry_run,
            save_project,
        } => {
            let mut limits = SliceLimits::default();
            if let Some(n) = max_instructions {
                limits.max_instructions = n;
            }
            limits.allow_memory = allow_memory;
            limits.allow_calls = allow_calls;
            limits.unknowns_on_truncation = unknowns_on_truncation;
            if let Some(n) = max_blocks {
                limits.max_basic_blocks = n;
            }
            let options = SolveOptions {
                timeout_ms: timeout_ms.unwrap_or(SolveOptions::default().timeout_ms),
            };
            let plan = AnnotatePlan {
                min_confidence: min_confidence.to_confidence(),
                dry_run,
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
            )
        }
        Command::Patch {
            file,
            at,
            function,
            max_instructions,
            timeout_ms,
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
            let mut limits = SliceLimits::default();
            if let Some(n) = max_instructions {
                limits.max_instructions = n;
            }
            limits.allow_memory = allow_memory;
            limits.allow_calls = allow_calls;
            limits.unknowns_on_truncation = unknowns_on_truncation;
            if let Some(n) = max_blocks {
                limits.max_basic_blocks = n;
            }
            let options = SolveOptions {
                timeout_ms: timeout_ms.unwrap_or(SolveOptions::default().timeout_ms),
            };
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
        } => {
            let mut limits = SliceLimits::default();
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
            let options = SolveOptions {
                timeout_ms: timeout_ms.unwrap_or(SolveOptions::default().timeout_ms),
            };
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
                differential_lift,
            )
        }
        Command::Batch {
            dir,
            threads,
            max_instructions,
            timeout_ms,
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
            let mut limits = SliceLimits::default();
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
            let options = SolveOptions {
                timeout_ms: timeout_ms.unwrap_or(SolveOptions::default().timeout_ms),
            };
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
            max_instructions,
            allow_memory,
            allow_calls,
            solver,
            with_decompiler,
            quiet,
            explain,
            allow_join_merge,
        } => {
            let mut limits = SliceLimits::default();
            if let Some(n) = max_instructions {
                limits.max_instructions = n;
            }
            limits.allow_memory = allow_memory;
            limits.allow_calls = allow_calls;
            limits.allow_join_merge = allow_join_merge;
            let options = SolveOptions {
                timeout_ms: timeout_ms.unwrap_or(SolveOptions::default().timeout_ms),
            };
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
            let mut limits = SliceLimits::default();
            if let Some(n) = max_instructions {
                limits.max_instructions = n;
            }
            limits.allow_memory = allow_memory;
            limits.allow_calls = allow_calls;
            limits.unknowns_on_truncation = unknowns_on_truncation;
            if let Some(n) = max_blocks {
                limits.max_basic_blocks = n;
            }
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
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
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
    );
    let annotations = report.annotations(&merged_functions);

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

// `clippy::too_many_arguments` / `fn_params_excessive_bools`: same
// rationale as `annotate` above — this is a CLI driver threading
// independent, read-at-distinct-stages knobs (`deep`,
// `with_decompiler`, `ir_pcode`, `differential_lift`) straight from
// parsed args to use-case calls. A params struct would only relocate
// the noise without adding cohesion; the booleans are genuinely
// independent toggles, not a missing abstraction.
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
    differential_lift: bool,
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

    if differential_lift {
        let scope: Vec<Function> = ctx.all_functions().cloned().collect();
        let dl = run_differential_lift(&scope, ctx.program.arch, solver, options);
        print_lifter_agreement(&dl.stats, dl.compared);
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
        );
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

/// Bounded scan budget for the opt-in `--differential-lift` pass.
/// Host-Side Safety: a whole-program run must not issue an unbounded
/// number of SMT queries.
const MAX_DIFFLIFT_COMPARISONS: usize = 50_000;

/// Outcome of the differential-lift pass: the engine-integrity
/// findings, the running agreement tally, and how many pairwise
/// comparisons were attempted.
struct DiffLiftRun {
    findings: Vec<Finding>,
    stats: r2smt_difflift::AgreementStats,
    compared: usize,
}

/// Cross-check every instruction's independent lowerings. A proven
/// disagreement yields one `lifter_disagreement` finding for that
/// instruction; the agreement tally feeds the reported metric. The
/// solve is delegated through the user-selected backend, exactly like
/// the branch pipeline.
fn run_differential_lift(
    functions: &[Function],
    arch: r2smt_common::Arch,
    solver: SolverArg,
    options: SolveOptions,
) -> DiffLiftRun {
    let mut stats = r2smt_difflift::AgreementStats::default();
    let mut findings: Vec<Finding> = Vec::new();
    let mut compared = 0usize;
    'outer: for func in functions {
        for block in &func.blocks {
            for insn in &block.instructions {
                let lowerings = r2smt_difflift::lower_all(insn, arch);
                let bodies: Vec<(r2smt_difflift::Lowering, &[r2smt_ir::IrStmt])> =
                    lowerings.available().collect();
                let mut disagreeing: Vec<String> = Vec::new();
                for (i, (_, sa)) in bodies.iter().enumerate() {
                    for (lb, sb) in &bodies[i + 1..] {
                        if compared >= MAX_DIFFLIFT_COMPARISONS {
                            break 'outer;
                        }
                        compared += 1;
                        let verdict = compare_lowerings(sa, sb, arch, solver, options);
                        stats.record(verdict);
                        if verdict == r2smt_difflift::DiffVerdict::Disagree {
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
    DiffLiftRun {
        findings,
        stats,
        compared,
    }
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
fn print_lifter_agreement(stats: &r2smt_difflift::AgreementStats, compared: usize) {
    let rate = stats
        .agreement_rate()
        .map_or_else(|| "n/a".to_string(), |r| format!("{:.2}%", r * 100.0));
    println!(
        "lifter-agreement: {rate} (agree={a} disagree={d} inconclusive={i}) over {compared} comparisons",
        a = stats.agree,
        d = stats.disagree,
        i = stats.inconclusive,
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
