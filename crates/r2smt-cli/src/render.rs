//! CLI presentation helpers.
//!
//! Pure formatting of domain types into the terminal output. Holds no
//! orchestration state — extracted from `main.rs` so the binary entry
//! point stays a thin composition root.

use r2smt_core::{Confidence, Finding, FindingKind};
use r2smt_ir::program::{Function, Program};
use r2smt_report::Annotation;
use r2smt_slicer::{BranchCandidate, LiftedSlice, Slice, SliceStatus};
use r2smt_ssa::SsaLiftedSlice;

pub(crate) fn print_branch_summary(candidates: &[BranchCandidate]) {
    println!("candidates: {}", candidates.len());
    let mut jcc = 0usize;
    let mut setcc = 0usize;
    let mut cmovcc = 0usize;
    for cand in candidates {
        match cand.kind {
            r2smt_slicer::BranchKind::Jcc => jcc += 1,
            r2smt_slicer::BranchKind::SetCc => setcc += 1,
            r2smt_slicer::BranchKind::CMovCc => cmovcc += 1,
            // `BranchKind` is `#[non_exhaustive]`; ignore future variants.
            _ => {}
        }
    }
    println!("  jcc:     {jcc}");
    println!("  setcc:   {setcc}");
    println!("  cmovcc:  {cmovcc}");
    let resolved = candidates
        .iter()
        .filter(|c| c.taken_target.is_some())
        .count();
    println!("  with resolved taken target: {resolved}");

    if !candidates.is_empty() {
        println!();
        println!("first {} candidates:", candidates.len().min(5));
        for cand in candidates.iter().take(5) {
            let target = cand
                .taken_target
                .map_or_else(|| "?".to_string(), |t| t.to_string());
            println!(
                "  {addr}  {kind:?} {mnem:<8} → {target}   ({formula})",
                addr = cand.address,
                kind = cand.kind,
                mnem = cand.mnemonic,
                formula = cand.formula,
            );
        }
    }
}

pub(crate) fn print_slice_summary(slices: &[Slice], explicit_at: bool, functions: &[Function]) {
    let total = slices.len();
    let complete = slices
        .iter()
        .filter(|s| matches!(s.status, SliceStatus::Complete))
        .count();
    let truncated = total - complete;

    println!("slices: {total} (complete: {complete}, truncated: {truncated})");
    if total > 0 {
        let mut len_sum = 0usize;
        let mut len_max = 0usize;
        for s in slices
            .iter()
            .filter(|s| matches!(s.status, SliceStatus::Complete))
        {
            len_sum += s.instructions.len();
            len_max = len_max.max(s.instructions.len());
        }
        if complete > 0 {
            // `cast_precision_loss`: slice counts stay well below 2^53; the
            // average is rendered at 2-decimal precision for the operator.
            #[allow(clippy::cast_precision_loss)]
            let avg = len_sum as f64 / complete as f64;
            println!("  complete slice length: avg {avg:.2}, max {len_max}");
        }
        if truncated > 0 {
            let mut by_reason: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for s in slices {
                if let SliceStatus::Truncated { reason } = &s.status {
                    let bucket = reason_bucket(reason);
                    *by_reason.entry(bucket).or_insert(0) += 1;
                }
            }
            println!("  truncation reasons:");
            for (reason, count) in by_reason {
                println!("    {count:>6}  {reason}");
            }
        }
    }

    if explicit_at && total == 1 {
        println!();
        print_slice_detail(&slices[0], functions);
    } else if total > 0 && total <= 5 {
        println!();
        for s in slices {
            print_slice_detail(s, functions);
            println!();
        }
    }
}

pub(crate) fn print_slice_detail(slice: &Slice, functions: &[Function]) {
    let fname = functions
        .iter()
        .find(|f| f.address == slice.branch.function)
        .and_then(|f| f.name.as_deref())
        .unwrap_or("<anon>");
    println!(
        "branch {addr}  {kind:?} {mnem:<6}  fn={fname}  cond={cond}",
        addr = slice.branch.address,
        kind = slice.branch.kind,
        mnem = slice.branch.mnemonic,
        cond = slice.branch.formula,
    );
    match &slice.status {
        SliceStatus::Complete => println!("  status: complete"),
        SliceStatus::Truncated { reason } => println!("  status: truncated ({reason})"),
    }
    if !slice.roots.is_empty() {
        println!("  roots:  {roots}", roots = slice.roots.join(", "));
    }
    for insn in &slice.instructions {
        let operands: String = insn
            .operands
            .iter()
            .map(|o| o.raw.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "    {addr}  {mnem:<6} {operands}",
            addr = insn.address,
            mnem = insn.mnemonic
        );
    }
}

pub(crate) fn print_lift_summary(lifts: &[LiftedSlice], explicit_at: bool, functions: &[Function]) {
    let total = lifts.len();
    let complete = lifts
        .iter()
        .filter(|l| matches!(l.status, SliceStatus::Complete))
        .count();
    let truncated = total - complete;
    println!("lifted slices: {total} (complete: {complete}, truncated: {truncated})");
    if total == 0 {
        return;
    }
    let stmt_total: usize = lifts.iter().map(|l| l.statements.len()).sum();
    let unsupported: usize = lifts
        .iter()
        .flat_map(|l| &l.statements)
        .filter(|s| matches!(s, r2smt_ir::IrStmt::Unsupported { .. }))
        .count();
    println!("  ir statements:  {stmt_total} ({unsupported} unsupported)",);

    if explicit_at && total == 1 {
        println!();
        print_lift_detail(&lifts[0], functions);
    } else if total <= 3 {
        println!();
        for l in lifts {
            print_lift_detail(l, functions);
            println!();
        }
    }
}

pub(crate) fn print_lift_detail(lifted: &LiftedSlice, functions: &[Function]) {
    let fname = functions
        .iter()
        .find(|f| f.address == lifted.branch.function)
        .and_then(|f| f.name.as_deref())
        .unwrap_or("<anon>");
    println!(
        "branch {addr}  {kind:?} {mnem:<6}  fn={fname}",
        addr = lifted.branch.address,
        kind = lifted.branch.kind,
        mnem = lifted.branch.mnemonic,
    );
    match &lifted.status {
        SliceStatus::Complete => println!("  status: complete"),
        SliceStatus::Truncated { reason } => println!("  status: truncated ({reason})"),
    }
    println!("  IR ({n} statements):", n = lifted.statements.len());
    for stmt in &lifted.statements {
        println!("    {stmt}");
    }
    println!("  branch condition: {cond}", cond = lifted.condition);
}

pub(crate) fn truncate_on_char_boundary(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = s[..end].to_string();
    out.push_str("\n… [truncated]");
    out
}

pub(crate) fn hex_preview(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn print_annotation_preview(
    annotations: &[Annotation],
    functions: &[Function],
    findings: &[Finding],
) {
    if annotations.is_empty() {
        return;
    }
    let preview = annotations.len().min(10);
    println!();
    println!("preview ({preview}):");
    for ann in annotations.iter().take(preview) {
        let kind = findings
            .iter()
            .find(|f| f.address == ann.address)
            .map_or("?", |f| match f.kind {
                FindingKind::OpaquePredicate => "opaque_predicate",
                FindingKind::DeadBranch => "dead_branch",
                FindingKind::ConstantCondition => "constant_condition",
                _ => "?",
            });
        let fname = findings
            .iter()
            .find(|f| f.address == ann.address)
            .and_then(|f| {
                functions
                    .iter()
                    .find(|fun| fun.address == f.function)
                    .and_then(|fun| fun.name.as_deref())
            })
            .unwrap_or("<anon>");
        println!(
            "  {addr}  [{kind}]  fn={fname}",
            addr = ann.address,
            kind = kind,
            fname = fname,
        );
    }
    if annotations.len() > preview {
        println!("  … and {extra} more", extra = annotations.len() - preview);
    }
}

pub(crate) fn print_findings_summary(
    all: &[Finding],
    displayed: &[&Finding],
    explicit_at: bool,
    functions: &[Function],
) {
    let total = all.len();
    let mut opaque = 0usize;
    let mut dead = 0usize;
    let mut constant = 0usize;
    let mut real = 0usize;
    let mut suspicious = 0usize;
    for f in all {
        match f.kind {
            FindingKind::OpaquePredicate => opaque += 1,
            FindingKind::DeadBranch => dead += 1,
            FindingKind::ConstantCondition => constant += 1,
            FindingKind::RealBranch => real += 1,
            // `FindingKind` is `#[non_exhaustive]`; collapse SuspiciousButUnknown
            // and any unforeseen future variants.
            _ => suspicious += 1,
        }
    }
    let actionable = opaque + dead + constant;
    println!("findings: {total}");
    println!("  opaque_predicate:    {opaque}");
    println!("  dead_branch:         {dead}");
    println!("  constant_condition:  {constant}");
    println!("  real_branch:         {real}");
    println!("  suspicious:          {suspicious}");
    if total > 0 {
        // `cast_precision_loss`: finding counts stay well below 2^53; pct
        // is rendered at 2-decimal precision for the operator.
        #[allow(clippy::cast_precision_loss)]
        let pct = (actionable as f64) * 100.0 / (total as f64);
        println!("  actionable (opaque | dead | constant): {actionable} ({pct:.2}%)");
    }

    let mut high = 0usize;
    let mut medium = 0usize;
    let mut low = 0usize;
    let mut conf_unknown = 0usize;
    for f in all {
        if !matches!(
            f.kind,
            FindingKind::OpaquePredicate | FindingKind::DeadBranch | FindingKind::ConstantCondition
        ) {
            continue;
        }
        match f.confidence {
            Confidence::High => high += 1,
            Confidence::Medium => medium += 1,
            Confidence::Low => low += 1,
            // `Confidence` is `#[non_exhaustive]`; collapse Unknown and
            // any future variants.
            _ => conf_unknown += 1,
        }
    }
    if actionable > 0 {
        println!(
            "  actionable confidence: high={high}, medium={medium}, low={low}, unknown={conf_unknown}"
        );
    }

    if explicit_at && total == 1 {
        println!();
        print_finding_detail(&all[0], functions);
        return;
    }
    if !displayed.is_empty() {
        println!();
        println!("displayed findings ({n}):", n = displayed.len());
        let max_shown = 15;
        for f in displayed.iter().take(max_shown) {
            print_finding_short(f, functions);
        }
        if displayed.len() > max_shown {
            println!("  … and {extra} more", extra = displayed.len() - max_shown);
        }
    }
}

pub(crate) fn print_finding_detail(f: &Finding, functions: &[Function]) {
    let fname = functions
        .iter()
        .find(|fun| fun.address == f.function)
        .and_then(|fun| fun.name.as_deref())
        .unwrap_or("<anon>");
    println!(
        "branch {addr}  {mnem:<6}  fn={fname}",
        addr = f.address,
        mnem = f.mnemonic,
    );
    println!("  formula:    {formula}", formula = f.formula);
    println!("  verdict:    {:?}", f.verdict);
    println!("  kind:       {:?}", f.kind);
    println!("  confidence: {:?}", f.confidence);
    if let Some(taken) = f.taken_target {
        println!("  taken:      {taken}");
    }
    if let Some(ft) = f.fallthrough_target {
        println!("  fallthrough:{ft}");
    }
    if !f.evidence.inputs.is_empty() {
        println!(
            "  inputs:     {names}",
            names = f.evidence.inputs.join(", ")
        );
    }
    println!(
        "  statements: {stmt}, unknown: {unk}",
        stmt = f.evidence.statement_count,
        unk = f.evidence.unknown_count,
    );
}

pub(crate) fn print_finding_short(f: &Finding, functions: &[Function]) {
    let fname = functions
        .iter()
        .find(|fun| fun.address == f.function)
        .and_then(|fun| fun.name.as_deref())
        .unwrap_or("<anon>");
    println!(
        "  {addr}  {mnem:<6}  {kind:?}/{conf:?}  fn={fname}  ({formula})",
        addr = f.address,
        mnem = f.mnemonic,
        kind = f.kind,
        conf = f.confidence,
        formula = f.formula,
    );
}

pub(crate) fn print_ssa_summary(
    ssas: &[SsaLiftedSlice],
    explicit_at: bool,
    functions: &[Function],
) {
    let total = ssas.len();
    let complete = ssas
        .iter()
        .filter(|s| matches!(s.status, SliceStatus::Complete))
        .count();
    let truncated = total - complete;
    println!("ssa slices: {total} (complete: {complete}, truncated: {truncated})");
    if total == 0 {
        return;
    }
    let stmt_total: usize = ssas.iter().map(|s| s.statements.len()).sum();
    let total_defs: usize = ssas.iter().map(|s| s.defs.len()).sum();
    let total_inputs: usize = ssas.iter().map(|s| s.inputs.len()).sum();
    // `cast_precision_loss`: SSA def / input counts stay well below 2^53;
    // averages are rendered at 2-decimal precision for the operator.
    #[allow(clippy::cast_precision_loss)]
    let avg_inputs = total_inputs as f64 / total as f64;
    #[allow(clippy::cast_precision_loss)]
    let avg_defs = total_defs as f64 / total as f64;
    println!("  statements: {stmt_total}");
    println!("  defs avg:   {avg_defs:.2}");
    println!("  inputs avg: {avg_inputs:.2}");

    if explicit_at && total == 1 {
        println!();
        print_ssa_detail(&ssas[0], functions);
    } else if total <= 3 {
        println!();
        for s in ssas {
            print_ssa_detail(s, functions);
            println!();
        }
    }
}

pub(crate) fn print_ssa_detail(ssa: &SsaLiftedSlice, functions: &[Function]) {
    let fname = functions
        .iter()
        .find(|f| f.address == ssa.branch.function)
        .and_then(|f| f.name.as_deref())
        .unwrap_or("<anon>");
    println!(
        "branch {addr}  {kind:?} {mnem:<6}  fn={fname}",
        addr = ssa.branch.address,
        kind = ssa.branch.kind,
        mnem = ssa.branch.mnemonic,
    );
    match &ssa.status {
        SliceStatus::Complete => println!("  status: complete"),
        SliceStatus::Truncated { reason } => println!("  status: truncated ({reason})"),
    }
    if !ssa.inputs.is_empty() {
        let names: Vec<String> = ssa.inputs.iter().map(|v| v.name.clone()).collect();
        println!("  inputs: {names}", names = names.join(", "));
    }
    println!(
        "  SSA ({n} statements, {d} defs):",
        n = ssa.statements.len(),
        d = ssa.defs.len()
    );
    for stmt in &ssa.statements {
        println!("    {stmt}");
    }
    println!("  branch condition: {cond}", cond = ssa.condition);
}

pub(crate) fn reason_bucket(reason: &str) -> String {
    // Collapse address-specific reasons into a single bucket per kind so
    // the summary is readable.
    if reason.starts_with("call at") {
        return "call encountered".into();
    }
    if reason.starts_with("memory access at") {
        return "memory access encountered".into();
    }
    if reason.starts_with("unsupported '") {
        return "unsupported instruction".into();
    }
    reason.to_string()
}

pub(crate) fn print_summary(program: &Program) {
    println!("arch:      {:?}", program.arch);
    println!("bits:      {}", program.bits);
    if let Some(entry) = program.entry {
        println!("entry:     {entry}");
    }
    println!("functions: {}", program.functions.len());
    let total_blocks: usize = program.functions.iter().map(|f| f.blocks.len()).sum();
    let total_insns: usize = program
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.instructions.len())
        .sum();
    println!("blocks:    {total_blocks}");
    println!("insns:     {total_insns}");
}
