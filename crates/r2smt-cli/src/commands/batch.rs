//! `batch` subcommand: parallel end-to-end analysis of a sample
//! directory. `analyze_one` is the per-sample worker (no stdout / file
//! side effects) so it is safe to run on a rayon thread.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use r2smt_report::{BatchOutcome, BatchReport, BatchSampleEntry, BatchSampleSummary, Report};
use r2smt_slicer::SliceLimits;
use r2smt_smt::SolveOptions;
use rayon::ThreadPoolBuilder;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tracing::{Span, debug, debug_span, info, warn};

use crate::args::SolverArg;
use crate::observability::safe_value;
use crate::support::{ReportRun, report_from_findings};
use crate::worker::{AnalysisWorkerRequest, run_isolated};

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
    let result = run_isolated(&AnalysisWorkerRequest::new(
        file,
        deep,
        None,
        None,
        limits,
        options,
        solver,
        with_decompiler,
        ir_pcode,
    )?)?;
    let arch = result.program.arch;
    let bits = result.program.bits;
    let function_count = result.metrics.functions_analyzed;
    let findings = result.findings;

    report_from_findings(
        &ReportRun {
            file,
            deep,
            at: None,
            function_filter: None,
            limits,
            options,
            solver,
            with_decompiler,
            ir_pcode,
            differential_lift: false,
        },
        arch,
        bits,
        function_count,
        findings,
    )
}

// This linear command driver owns validation, parallel dispatch, telemetry,
// and output; extracting those short phases would only hide their ordering.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    let started = Instant::now();
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
        info!(
            target: "r2smt::batch",
            event_name = "batch.completed",
            status = "ok",
            samples = 0,
            failed = 0,
            duration_ms = 0,
            "batch completed without input files"
        );
        println!("no files to analyze in {}", dir.display());
        return Ok(());
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads.unwrap_or(0))
        .build()
        .context("building rayon thread pool")?;
    info!(
        target: "r2smt::batch",
        event_name = "batch.started",
        status = "started",
        samples = files.len(),
        worker_threads = pool.current_num_threads(),
        "batch analysis started"
    );
    let parent_span = Span::current();

    // `collect()` on an indexed parallel iterator preserves input
    // order, so the aggregate is deterministic regardless of
    // `--threads`. The full per-sample `Report` is downsampled to a
    // bounded `BatchSampleSummary` and dropped inside the worker —
    // unbounded finding vectors never accumulate across the sweep.
    let entries: Vec<BatchSampleEntry> = pool.install(|| {
        files
            .par_iter()
            .enumerate()
            .map(|(sample_index, path)| {
                let span = debug_span!(
                    target: "r2smt::batch",
                    parent: &parent_span,
                    "batch_sample",
                    component = "batch",
                    operation = "analyze_sample",
                    sample_index,
                );
                span.in_scope(|| {
                    let outcome = match analyze_one(
                        path,
                        deep,
                        limits,
                        options,
                        solver,
                        with_decompiler,
                        ir_pcode,
                    ) {
                        Ok(report) => {
                            debug!(
                                target: "r2smt::batch",
                                event_name = "batch.sample_completed",
                                status = "ok",
                                functions = report.functions_analyzed,
                                branches = report.branches_analyzed,
                                "batch sample completed"
                            );
                            BatchOutcome::Analyzed {
                                summary: Box::new(BatchSampleSummary::from_report(&report)),
                            }
                        }
                        Err(err) => {
                            let error = format!("{err:#}");
                            warn!(
                                target: "r2smt::batch",
                                event_name = "batch.sample_failed",
                                status = "error",
                                error_type = "analysis_error",
                                error = %safe_value(&error),
                                "batch sample failed"
                            );
                            BatchOutcome::Failed { error }
                        }
                    };
                    BatchSampleEntry {
                        path: path.display().to_string(),
                        outcome,
                    }
                })
            })
            .collect()
    });
    let failed = entries
        .iter()
        .filter(|entry| matches!(entry.outcome, BatchOutcome::Failed { .. }))
        .count();
    info!(
        target: "r2smt::batch",
        event_name = "batch.completed",
        status = if failed == 0 { "ok" } else { "partial" },
        samples = entries.len(),
        failed,
        duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        "batch analysis completed"
    );

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
