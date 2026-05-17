#![deny(missing_docs)]
// A CLI is the one layer that legitimately writes to standard streams.
// The workspace lints deny `print_stdout` / `print_stderr` everywhere
// else; we relax the rule here so user-facing output and error messages
// can flow normally.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! `r2smt` command-line entrypoint.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use r2smt_common::{Address, Arch};
use r2smt_core::{
    Confidence, Finding, FindingKind, classify_finding_with_pretty, dump_program, prepare_ssa,
};
use r2smt_ir::Annotator;
use r2smt_ir::BinaryProvider;
use r2smt_ir::NameHints;
use r2smt_ir::program::Function;
use r2smt_patch::{ApplyConfig, PatchManifest, apply_plan, build_plan, rollback_from_manifest};
use r2smt_report::{BatchOutcome, BatchReport, BatchSampleEntry, BatchSampleSummary, Report};
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;
use rayon::ThreadPoolBuilder;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::error;
use tracing_subscriber::EnvFilter;

mod args;
use args::{Cli, Command, SolverArg};
mod render;
use render::{hex_preview, print_annotation_preview, print_findings_summary};
mod support;
use support::{
    attach_pseudocode, compute_findings, dispatch_solver, open_provider, open_provider_writable,
    resolve_folded_branch, resolve_targets,
};
mod commands;
use commands::inspect::{analyze, branches, lift, slice, ssa};

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

struct PatchCli<'a> {
    min_confidence: Confidence,
    apply: bool,
    backup: Option<&'a Path>,
    manifest: Option<&'a Path>,
    rollback: bool,
    solver: SolverArg,
}

const DEFAULT_BACKUP_SUFFIX: &str = ".r2smt.bak";
const DEFAULT_MANIFEST_SUFFIX: &str = ".r2smt.manifest.json";

fn default_backup_path(file: &Path) -> PathBuf {
    let mut s = file.as_os_str().to_owned();
    s.push(DEFAULT_BACKUP_SUFFIX);
    PathBuf::from(s)
}

fn default_manifest_path(file: &Path) -> PathBuf {
    let mut s = file.as_os_str().to_owned();
    s.push(DEFAULT_MANIFEST_SUFFIX);
    PathBuf::from(s)
}

// `clippy::too_many_arguments`: same rationale as `solve` / `batch` —
// a CLI driver threading independent, read-at-distinct-stages knobs
// (`ir_pcode` is the 8th). A params struct would only relocate noise.
#[allow(clippy::too_many_arguments)]
fn patch(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    options: SolveOptions,
    cfg: &PatchCli<'_>,
    ir_pcode: bool,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }
    if cfg.rollback {
        return patch_rollback(file, deep, cfg);
    }

    // Read-only pipeline first: open without write access so we can
    // plan and (optionally) bail out before any disk mutation.
    let (arch, findings) = compute_findings(
        file,
        deep,
        at,
        function_filter,
        limits,
        options,
        cfg.solver,
        false,
        ir_pcode,
    )?;
    let actionable: Vec<Finding> = findings
        .into_iter()
        .filter(Finding::is_actionable)
        .collect();
    println!(
        "candidate findings: {n} (min_confidence={mc:?})",
        n = actionable.len(),
        mc = cfg.min_confidence,
    );

    if !cfg.apply {
        println!();
        patch_dry_run_plan(file, deep, arch, &actionable, cfg)?;
        return Ok(());
    }

    // Apply path: backup → open writable → plan → apply → write manifest.
    let backup = cfg
        .backup
        .map_or_else(|| default_backup_path(file), Path::to_path_buf);
    let manifest_path = cfg
        .manifest
        .map_or_else(|| default_manifest_path(file), Path::to_path_buf);

    if backup.exists() {
        anyhow::bail!(
            "refusing to overwrite existing backup at {} — move or delete it first",
            backup.display()
        );
    }
    fs::copy(file, &backup)
        .with_context(|| format!("backing up {} to {}", file.display(), backup.display()))?;
    println!("backup: {}", backup.display());

    let mut provider = open_provider_writable(file, deep)?;
    let plan = build_plan(&actionable, cfg.min_confidence, arch, &mut provider)
        .with_context(|| "building patch plan")?;
    println!("plan: {} operations", plan.operations.len());
    for skip in &plan.skipped {
        println!("  skipped {addr}: {reason}", addr = skip.0, reason = skip.1);
    }
    if plan.operations.is_empty() {
        println!("nothing to apply");
        return Ok(());
    }

    let apply_cfg = ApplyConfig {
        binary_path: file.to_path_buf(),
        backup_path: backup.clone(),
        r2smt_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let manifest =
        apply_plan(&mut provider, &plan, &apply_cfg).with_context(|| "applying patch plan")?;
    drop(provider);

    manifest
        .write_to(&manifest_path)
        .with_context(|| format!("writing manifest to {}", manifest_path.display()))?;
    println!();
    println!("applied:  {} operations", manifest.operations.len());
    println!("manifest: {}", manifest_path.display());
    println!("before SHA-256: {}", manifest.binary_sha256_before);
    println!("after  SHA-256: {}", manifest.binary_sha256_after);
    Ok(())
}

fn patch_dry_run_plan(
    file: &Path,
    deep: bool,
    arch: Arch,
    actionable: &[Finding],
    cfg: &PatchCli<'_>,
) -> Result<()> {
    let mut provider = open_provider(file, deep)?;
    let plan = build_plan(actionable, cfg.min_confidence, arch, &mut provider)
        .with_context(|| "building patch plan")?;
    println!("planned operations: {}", plan.operations.len());
    for op in &plan.operations {
        println!(
            "  {addr}  {strategy:<22}  size={size}  → {bytes}",
            addr = op.address,
            strategy = op.strategy.as_str(),
            size = op.size,
            bytes = hex_preview(&op.new_bytes),
        );
    }
    if !plan.skipped.is_empty() {
        println!();
        println!("skipped: {}", plan.skipped.len());
        for (addr, reason) in &plan.skipped {
            println!("  {addr}  {reason}");
        }
    }
    println!();
    println!("dry-run: re-run with --apply to write the changes");
    Ok(())
}

fn patch_rollback(file: &Path, deep: bool, cfg: &PatchCli<'_>) -> Result<()> {
    let manifest_path = cfg
        .manifest
        .map_or_else(|| default_manifest_path(file), Path::to_path_buf);
    let manifest = PatchManifest::read_from(&manifest_path)
        .with_context(|| format!("reading manifest at {}", manifest_path.display()))?;
    println!(
        "rolling back {n} operation(s) from {path}",
        n = manifest.operations.len(),
        path = manifest_path.display(),
    );
    let mut provider = open_provider_writable(file, deep)?;
    rollback_from_manifest(&mut provider, &manifest).with_context(|| "rolling back manifest")?;
    drop(provider);
    println!("rollback completed");
    Ok(())
}

/// Analyse one sample end-to-end and return its [`Report`]. Mirrors
/// the core of `solve()` with no stdout / file side effects, so it is
/// safe to call from a parallel batch worker: the radare2 session is
/// created and dropped entirely within this call — nothing crosses
/// the thread boundary.
fn analyze_one(
    file: &Path,
    deep: bool,
    limits: &SliceLimits,
    options: SolveOptions,
    solver: SolverArg,
    with_decompiler: bool,
    ir_pcode: bool,
) -> Result<Report> {
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

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, None, None)?;

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

    Ok(Report::from_findings(
        env!("CARGO_PKG_VERSION"),
        file.display().to_string(),
        arch,
        bits,
        function_count,
        findings,
    ))
}

// `clippy::too_many_arguments`: same rationale as `solve` / `annotate`
// — a CLI driver threading through independent, read-at-distinct-stages
// knobs. Splitting into a struct would only move the noise.
#[allow(clippy::too_many_arguments)]
fn batch(
    dir: &Path,
    deep: bool,
    threads: Option<usize>,
    limits: &SliceLimits,
    options: SolveOptions,
    solver: SolverArg,
    with_decompiler: bool,
    ir_pcode: bool,
    json_out: Option<&Path>,
    markdown_out: Option<&Path>,
) -> Result<()> {
    if !dir.is_dir() {
        anyhow::bail!("not a directory: {}", dir.display());
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("reading directory {}", dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    if files.is_empty() {
        println!("no files to analyze in {}", dir.display());
        return Ok(());
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads.unwrap_or(0))
        .build()
        .context("building rayon thread pool")?;

    // `collect()` on an indexed parallel iterator preserves input
    // order, so the aggregate is deterministic regardless of
    // `--threads`. The full per-sample `Report` is downsampled to a
    // bounded `BatchSampleSummary` and dropped inside the worker —
    // unbounded finding vectors never accumulate across the sweep.
    let entries: Vec<BatchSampleEntry> = pool.install(|| {
        files
            .par_iter()
            .map(|path| {
                let outcome = match analyze_one(
                    path,
                    deep,
                    limits,
                    options,
                    solver,
                    with_decompiler,
                    ir_pcode,
                ) {
                    Ok(report) => BatchOutcome::Analyzed {
                        summary: BatchSampleSummary::from_report(&report),
                    },
                    Err(err) => BatchOutcome::Failed {
                        error: format!("{err:#}"),
                    },
                };
                BatchSampleEntry {
                    path: path.display().to_string(),
                    outcome,
                }
            })
            .collect()
    });

    let report = BatchReport::new(
        env!("CARGO_PKG_VERSION"),
        dir.display().to_string(),
        entries,
    );

    let any_file = json_out.is_some() || markdown_out.is_some();
    if let Some(path) = json_out {
        let json = report.render_json().context("serialising batch report")?;
        fs::write(path, json).with_context(|| format!("writing JSON to {}", path.display()))?;
    }
    if let Some(path) = markdown_out {
        fs::write(path, report.render_markdown())
            .with_context(|| format!("writing Markdown to {}", path.display()))?;
    }
    if !any_file {
        print!("{}", report.render_markdown());
    }
    Ok(())
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

// `clippy::too_many_arguments`: same rationale as `annotate` above —
// CLI driver passes through independent, read-at-distinct-stages knobs.
#[allow(clippy::too_many_arguments)]
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
