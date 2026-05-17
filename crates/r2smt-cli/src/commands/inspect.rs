//! Read-only inspection subcommands (`analyze`, `branches`,
//! `slice`, `lift`, `ssa`). Pure pipeline drivers — no mutation,
//! no patching.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use r2smt_common::Address;
use r2smt_core::{dump_program, prepare_ssa};
use r2smt_ir::program::{Function, Program};
use r2smt_slicer::{
    LiftedSlice, Slice, SliceLimits, collect_branches, collect_function_branches, lift_slice,
    slice_branch,
};
use r2smt_ssa::SsaLiftedSlice;

use crate::render::{
    print_branch_summary, print_lift_summary, print_slice_summary, print_ssa_summary, print_summary,
};
use crate::support::{open_provider, resolve_targets};

pub(crate) fn analyze(
    file: &Path,
    deep: bool,
    dump_flag: bool,
    json_out: Option<&Path>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;

    let program = dump_program(&mut provider)
        .with_context(|| format!("dumping program from {}", file.display()))?;

    if dump_flag || json_out.is_some() {
        emit_program_json(&program, json_out)?;
    } else {
        print_summary(&program);
    }
    Ok(())
}

fn emit_program_json(program: &Program, json_out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(program).context("serialising Program to JSON")?;
    if let Some(path) = json_out {
        fs::write(path, &json).with_context(|| format!("writing JSON to {}", path.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

pub(crate) fn branches(
    file: &Path,
    deep: bool,
    function: Option<&str>,
    json_out: Option<&Path>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;

    let candidates = match function {
        None => collect_branches(&program),
        Some(raw) => {
            let address: Address = raw
                .parse()
                .with_context(|| format!("parsing --function value '{raw}'"))?;
            let function = program
                .functions
                .iter()
                .find(|f| f.address == address)
                .ok_or_else(|| anyhow::anyhow!("no function at {address} in {}", file.display()))?;
            collect_function_branches(function, program.arch)
        }
    };

    if let Some(path) = json_out {
        let json = serde_json::to_string_pretty(&candidates)
            .context("serialising branch candidates to JSON")?;
        fs::write(path, &json).with_context(|| format!("writing JSON to {}", path.display()))?;
    } else {
        print_branch_summary(&candidates);
    }
    Ok(())
}

pub(crate) fn slice(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    json_out: Option<&Path>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;

    let mut slices: Vec<Slice> = Vec::with_capacity(filtered.len());
    for cand in &filtered {
        let Some(function) = ctx.find_function(cand.function) else {
            continue;
        };
        slices.push(slice_branch(cand, function, limits, ctx.program.arch));
    }

    if let Some(path) = json_out {
        let json = serde_json::to_string_pretty(&slices).context("serialising slices to JSON")?;
        fs::write(path, &json).with_context(|| format!("writing JSON to {}", path.display()))?;
    } else {
        let functions: Vec<Function> = ctx.all_functions().cloned().collect();
        print_slice_summary(&slices, at.is_some(), &functions);
    }
    Ok(())
}

pub(crate) fn lift(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    json_out: Option<&Path>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;
    let arch = program.arch;

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;

    let mut lifts: Vec<LiftedSlice> = Vec::with_capacity(filtered.len());
    for cand in &filtered {
        let Some(function) = ctx.find_function(cand.function) else {
            continue;
        };
        let slice = slice_branch(cand, function, limits, ctx.program.arch);
        lifts.push(lift_slice(&slice, arch));
    }

    if let Some(path) = json_out {
        let json =
            serde_json::to_string_pretty(&lifts).context("serialising lifted slices to JSON")?;
        fs::write(path, &json).with_context(|| format!("writing JSON to {}", path.display()))?;
    } else {
        let functions: Vec<Function> = ctx.all_functions().cloned().collect();
        print_lift_summary(&lifts, at.is_some(), &functions);
    }
    Ok(())
}

pub(crate) fn ssa(
    file: &Path,
    deep: bool,
    at: Option<&str>,
    function_filter: Option<&str>,
    limits: &SliceLimits,
    json_out: Option<&Path>,
) -> Result<()> {
    if !file.exists() {
        anyhow::bail!("input file does not exist: {}", file.display());
    }

    let mut provider = open_provider(file, deep)?;
    let program = dump_program(&mut provider)
        .with_context(|| format!("loading program from {}", file.display()))?;

    let (ctx, filtered) = resolve_targets(&mut provider, file, program, at, function_filter)?;

    let mut ssas: Vec<SsaLiftedSlice> = Vec::with_capacity(filtered.len());
    for cand in &filtered {
        let Some(function) = ctx.find_function(cand.function) else {
            continue;
        };
        ssas.push(prepare_ssa(function, cand, limits, ctx.program.arch));
    }

    if let Some(path) = json_out {
        let json = serde_json::to_string_pretty(&ssas).context("serialising SSA slices to JSON")?;
        fs::write(path, &json).with_context(|| format!("writing JSON to {}", path.display()))?;
    } else {
        let functions: Vec<Function> = ctx.all_functions().cloned().collect();
        print_ssa_summary(&ssas, at.is_some(), &functions);
    }
    Ok(())
}
