//! `taint` subcommand — E2b.
//!
//! Runs the sound may-taint pass ([`r2smt_taint::propagate`]) over the
//! slice at a target address and reports which values derive from the
//! seeded source(s). When the taint outcome is *opaque* (a flow hidden
//! behind an `Unknown` / `Unsupported` node) and `--concretise` is set,
//! it falls through to the fenced, UNSOUND exploration engine to search
//! for a concrete witness — the taint-concretisation composition the
//! sound pass explicitly delegates.
//!
//! This is the only layer allowed to depend on both `r2smt-taint` and
//! `r2smt-explore` without breaking the dependency fence.

use std::path::Path;

use anyhow::{Context, Result};
use r2smt_common::Address;
use r2smt_explore::{ExploreBudget, ExploreRequest};
use r2smt_slicer::SliceLimits;
use r2smt_ssa::SsaLiftedSlice;
use r2smt_taint::{SourceId, TaintSeeds, TaintSet, propagate};

use crate::render::render_explore_result;
use crate::support::compute_slices;

/// Options for the `taint` command.
pub(crate) struct TaintCli<'a> {
    pub(crate) addr: &'a str,
    /// Source register names to seed (each a distinct taint source). If
    /// empty, every free input is seeded as a single source.
    pub(crate) sources: &'a [String],
    /// Concretise an opaque outcome via the exploration engine.
    pub(crate) concretise: bool,
    pub(crate) wall_clock_ms: u64,
    pub(crate) max_paths: u64,
}

/// The result of a taint pass over one slice, kept pure so it can be
/// unit-tested without a live radare2 session.
pub(crate) struct TaintReport {
    /// Human-readable names of the seeded sources, indexed by `SourceId`.
    pub(crate) source_names: Vec<String>,
    /// `(variable, source ids)` for every tainted defined variable.
    pub(crate) tainted: Vec<(String, Vec<u32>)>,
    /// Whether the outcome is best-effort (see [`r2smt_taint`]).
    pub(crate) opaque: bool,
}

pub(crate) fn taint(
    file: &Path,
    deep: bool,
    limits: &SliceLimits,
    ir_pcode: bool,
    cli: &TaintCli<'_>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }
    let (arch, slices) = compute_slices(file, deep, Some(cli.addr), None, limits, ir_pcode)?;
    let Some(ssa) = slices.first() else {
        println!("r2smt: no analysable slice at {addr}", addr = cli.addr);
        return Ok(());
    };
    let report = build_taint_report(ssa, cli.sources);
    print_taint_report(cli.addr, &report);

    if cli.concretise && report.opaque {
        let target: Address = cli
            .addr
            .parse()
            .with_context(|| format!("invalid target address: {}", cli.addr))?;
        let request = ExploreRequest {
            binary_path: file.to_path_buf(),
            target,
            arch,
        };
        let budget = ExploreBudget {
            wall_clock_ms: cli.wall_clock_ms,
            max_paths: cli.max_paths,
        };
        let result = r2smt_explore::explore(&request, budget);
        println!("{}", render_explore_result(target, &result));
    }
    Ok(())
}

/// Build the taint seeds and run the pass. Pure — no I/O.
pub(crate) fn build_taint_report(ssa: &SsaLiftedSlice, sources: &[String]) -> TaintReport {
    let (seeds, source_names) = build_seeds(ssa, sources);
    let Some(outcome) = propagate(ssa, &seeds) else {
        // Recursion budget exceeded — treat as opaque, no taint proven.
        return TaintReport {
            source_names,
            tainted: Vec::new(),
            opaque: true,
        };
    };
    let mut tainted = Vec::new();
    for def in &ssa.defs {
        let set = outcome.taint_of(&def.name);
        if !set.is_untainted() {
            tainted.push((def.name.clone(), set.sources().map(|s| s.0).collect()));
        }
    }
    TaintReport {
        source_names,
        tainted,
        opaque: outcome.is_opaque(),
    }
}

/// Seed the taint sources. With no `--source`, every free input is one
/// source; otherwise each named register is a distinct source seeded
/// onto the inputs whose name matches it.
fn build_seeds(ssa: &SsaLiftedSlice, sources: &[String]) -> (TaintSeeds, Vec<String>) {
    let mut seeds = TaintSeeds::new();
    if sources.is_empty() {
        let set = TaintSet::source(SourceId::new(0));
        for input in &ssa.inputs {
            seeds.insert(input.name.clone(), set.clone());
        }
        return (seeds, vec!["<any input>".to_string()]);
    }
    for (i, src) in sources.iter().enumerate() {
        let set = TaintSet::source(SourceId::new(u32::try_from(i).unwrap_or(u32::MAX)));
        for input in &ssa.inputs {
            if input_matches(&input.name, src) {
                seeds.insert(input.name.clone(), set.clone());
            }
        }
    }
    (seeds, sources.to_vec())
}

/// Whether an SSA input variable name refers to the source register
/// `src` (exact, or the SSA-versioned `src#N` form).
fn input_matches(input_name: &str, src: &str) -> bool {
    input_name == src
        || input_name
            .strip_prefix(src)
            .is_some_and(|rest| rest.starts_with('#'))
}

fn print_taint_report(addr: &str, report: &TaintReport) {
    println!(
        "taint @ {addr}  sources: {srcs}",
        srcs = report.source_names.join(", ")
    );
    if report.tainted.is_empty() {
        println!("  no tainted values reached this slice");
    } else {
        for (name, srcs) in &report.tainted {
            let ids: Vec<String> = srcs.iter().map(u32::to_string).collect();
            println!("  {name}  <- source(s) {}", ids.join(","));
        }
    }
    if report.opaque {
        println!(
            "  note: outcome is OPAQUE (unmodelled node) — taint may be incomplete; \
             re-run with --concretise to search for a witness"
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::{build_taint_report, input_matches};
    use r2smt_common::{Address, Arch};
    use r2smt_ir::expr::{Expr, Var};
    use r2smt_ir::stmt::IrStmt;
    use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus};
    use r2smt_ssa::SsaLiftedSlice;

    fn slice(statements: Vec<IrStmt>, inputs: Vec<Var>, defs: Vec<Var>) -> SsaLiftedSlice {
        let z = Address::new(0x1000);
        SsaLiftedSlice {
            branch: BranchCandidate {
                address: z,
                function: z,
                block: z,
                kind: BranchKind::Jcc,
                mnemonic: "t".into(),
                condition: BranchCondition::NotEqual,
                formula: "t".into(),
                taken_target: None,
                fallthrough_target: None,
                compare_register: None,
                bit_index: None,
                upstream_resolved: None,
                operand_raws: Vec::new(),
                is_thumb: false,
            },
            statements,
            condition: Expr::konst(0, 1),
            status: SliceStatus::Complete,
            treat_truncation_as_inputs: false,
            inputs,
            defs,
            arch: Arch::X86_64,
        }
    }

    #[test]
    fn test_build_taint_report_flags_flow_from_seeded_input() {
        // rsi := rdi (rdi is a free input) → rsi is tainted by source 0.
        let ssa = slice(
            vec![IrStmt::Assign {
                dst: Var::new("rsi", 64),
                src: Expr::Var(Var::new("rdi", 64)),
            }],
            vec![Var::new("rdi", 64)],
            vec![Var::new("rsi", 64)],
        );
        let report = build_taint_report(&ssa, &[]);
        assert!(report.tainted.iter().any(|(n, _)| n == "rsi"));
        assert!(!report.opaque);
    }

    #[test]
    fn test_build_taint_report_marks_unknown_opaque() {
        let ssa = slice(
            vec![IrStmt::Assign {
                dst: Var::new("rsi", 64),
                src: Expr::Unknown("x".into()),
            }],
            vec![Var::new("rdi", 64)],
            vec![Var::new("rsi", 64)],
        );
        let report = build_taint_report(&ssa, &[]);
        assert!(report.opaque);
    }

    #[test]
    fn test_input_matches_exact_and_versioned() {
        assert!(input_matches("rdi", "rdi"));
        assert!(input_matches("rdi#0", "rdi"));
        assert!(!input_matches("rdix", "rdi"));
        assert!(!input_matches("rsi", "rdi"));
    }
}
