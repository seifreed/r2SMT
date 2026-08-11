//! `batch` subcommand: parallel end-to-end analysis of a sample
//! directory. `analyze_one` is the per-sample worker (no stdout / file
//! side effects) so it is safe to run on a rayon thread.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use r2smt_common::Address;
use r2smt_core::{Finding, classify_finding_with_pretty, dump_program, prepare_ssa};
use r2smt_ir::BinaryProvider;
use r2smt_ir::NameHints;
use r2smt_report::{BatchOutcome, BatchReport, BatchSampleEntry, BatchSampleSummary, Report};
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;
use rayon::ThreadPoolBuilder;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::args::SolverArg;
use crate::support::{
    attach_pseudocode, dispatch_solver, open_provider, resolve_folded_branch, resolve_targets,
};

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
pub(crate) fn batch(
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
