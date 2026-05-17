//! `AArch64` patch planners (`cset`/`csetm`, `csel`, the
//! `csinc`/`csinv`/`csneg` cs-arith family) + their helpers,
//! extracted from `plan.rs`. The shared classification model
//! (`MnemonicKind`, `CsArithOp`, `FindingDecision`) and the generic
//! plan engine stay in the parent module.

use r2smt_common::smt::SmtResult;
use r2smt_common::{Address, Result};
use r2smt_core::Finding;
use r2smt_ir::byte_patcher::BytePatcher;
use r2smt_report::PatchStrategy;

use crate::aarch64_encoding;
use crate::arm_encoding::ARM_INSTRUCTION_BYTES;

use super::{CsArithOp, FindingDecision, MnemonicKind, PlanOperation};

pub(super) fn classify_aarch64_mnemonic(mnemonic: &str) -> MnemonicKind {
    // `b.<cond>` covers eq/ne/cs/hs/cc/lo/mi/pl/vs/vc/hi/ls/ge/lt/
    // gt/le; the dot distinguishes from the unconditional `b`.
    if mnemonic.starts_with("b.") {
        return MnemonicKind::Jcc;
    }
    // Compare-and-branch: cbz/cbnz/tbz/tbnz read a register or bit
    // and branch on the result. They are still branch-family and
    // the v0 rewrite strategy (NOP / unconditional `b`) applies.
    if matches!(mnemonic, "cbz" | "cbnz" | "tbz" | "tbnz") {
        return MnemonicKind::Jcc;
    }
    match mnemonic {
        "cset" => MnemonicKind::Cset { all_ones: false },
        "csetm" => MnemonicKind::Cset { all_ones: true },
        "csel" => MnemonicKind::Csel,
        "csinc" => MnemonicKind::CsArith {
            op: CsArithOp::Csinc,
            aliased: false,
        },
        "cinc" => MnemonicKind::CsArith {
            op: CsArithOp::Csinc,
            aliased: true,
        },
        "csinv" => MnemonicKind::CsArith {
            op: CsArithOp::Csinv,
            aliased: false,
        },
        "cinv" => MnemonicKind::CsArith {
            op: CsArithOp::Csinv,
            aliased: true,
        },
        "csneg" => MnemonicKind::CsArith {
            op: CsArithOp::Csneg,
            aliased: false,
        },
        "cneg" => MnemonicKind::CsArith {
            op: CsArithOp::Csneg,
            aliased: true,
        },
        _ => MnemonicKind::Other,
    }
}

/// Rewrite an `AArch64` `cset` / `csetm` whose predicate the solver
/// proved constant. The destination is read from
/// [`Finding::operands`]; the new instruction is `mov Rd, #imm` where
/// `imm` is `0`, `1`, or `-1` (`csetm` with predicate true).
pub(super) fn plan_aarch64_cset(
    finding: &Finding,
    size: usize,
    patcher: &mut dyn BytePatcher,
    all_ones: bool,
) -> Result<FindingDecision> {
    let Some(value_bit) = bit_value_for(finding) else {
        return Ok(FindingDecision::Skip(format!(
            "{:?} verdict cannot drive a cset rewrite",
            finding.verdict
        )));
    };
    let Some(dst_raw) = finding.operands.first() else {
        return Ok(FindingDecision::Skip(format!(
            "cset at {addr} has no destination operand recorded",
            addr = finding.address,
        )));
    };
    let Some(dst) = aarch64_encoding::parse_xreg(dst_raw) else {
        return Ok(FindingDecision::Skip(format!(
            "cset destination operand '{dst_raw}' is not a recognised GPR"
        )));
    };
    let imm: i64 = match (value_bit, all_ones) {
        (true, true) => -1,
        (true, false) => 1,
        (false, _) => 0,
    };
    let asm = aarch64_encoding::mov_imm(&dst, imm);
    let encoded = patcher.assemble(finding.address, &asm)?;
    if let Some(skip) = enforce_aarch64_size(&encoded, size, finding.address) {
        return Ok(FindingDecision::Skip(skip));
    }
    let strategy = PatchStrategy::ReplaceCsetWithMovConst;
    Ok(FindingDecision::Plan(PlanOperation {
        address: finding.address,
        strategy,
        kind: finding.kind,
        confidence: finding.confidence,
        size,
        new_bytes: encoded,
        rationale: cs_rationale(finding, &asm),
    }))
}

/// Rewrite an `AArch64` `csel Rd, Rn, Rm, cond`. When the predicate is
/// proved always-true, the result is `Rn`; when always-false, `Rm`.
/// Either case collapses to `mov Rd, R{n|m}`.
pub(super) fn plan_aarch64_csel(
    finding: &Finding,
    size: usize,
    patcher: &mut dyn BytePatcher,
) -> Result<FindingDecision> {
    let Some(value_bit) = bit_value_for(finding) else {
        return Ok(FindingDecision::Skip(format!(
            "{:?} verdict cannot drive a csel rewrite",
            finding.verdict
        )));
    };
    let Some((dst, rn, rm_opt)) = parse_cs_operands(finding) else {
        return Ok(FindingDecision::Skip(format!(
            "csel at {addr} operand parse failed (need Rd, Rn, Rm)",
            addr = finding.address,
        )));
    };
    let Some(rm) = rm_opt else {
        return Ok(FindingDecision::Skip(format!(
            "csel at {addr} is missing Rm operand",
            addr = finding.address,
        )));
    };
    let source = if value_bit { rn } else { rm };
    let asm = aarch64_encoding::mov_reg(&dst, &source);
    let encoded = patcher.assemble(finding.address, &asm)?;
    if let Some(skip) = enforce_aarch64_size(&encoded, size, finding.address) {
        return Ok(FindingDecision::Skip(skip));
    }
    let strategy = PatchStrategy::ReplaceCselWithMov;
    Ok(FindingDecision::Plan(PlanOperation {
        address: finding.address,
        strategy,
        kind: finding.kind,
        confidence: finding.confidence,
        size,
        new_bytes: encoded,
        rationale: cs_rationale(finding, &asm),
    }))
}

/// Rewrite an `AArch64` `csinc` / `csinv` / `csneg` (or its 2-op
/// `cinc` / `cinv` / `cneg` alias). For the alias form, the
/// disassembler reports two operands `Rd, Rn` — Armv8 defines
/// `cinc Rd, Rn, cond` ≡ `csinc Rd, Rn, Rn, !cond`, so `Rm = Rn`
/// for rewrite purposes.
pub(super) fn plan_aarch64_cs_arith(
    finding: &Finding,
    size: usize,
    patcher: &mut dyn BytePatcher,
    op: CsArithOp,
    aliased: bool,
) -> Result<FindingDecision> {
    let Some(value_bit) = bit_value_for(finding) else {
        return Ok(FindingDecision::Skip(format!(
            "{:?} verdict cannot drive a cs-arithmetic rewrite",
            finding.verdict
        )));
    };
    let Some((dst, rn, rm_opt)) = parse_cs_operands(finding) else {
        return Ok(FindingDecision::Skip(format!(
            "cs-arithmetic at {addr} operand parse failed",
            addr = finding.address,
        )));
    };
    let rm = match (aliased, rm_opt) {
        (true, _) => rn.clone(),
        (false, Some(value)) => value,
        (false, None) => {
            return Ok(FindingDecision::Skip(format!(
                "cs-arithmetic at {addr} is missing Rm operand",
                addr = finding.address,
            )));
        }
    };
    let asm = if value_bit {
        aarch64_encoding::mov_reg(&dst, &rn)
    } else {
        match op {
            CsArithOp::Csinc => aarch64_encoding::add_imm(&dst, &rm, 1),
            CsArithOp::Csinv => aarch64_encoding::mvn_reg(&dst, &rm),
            CsArithOp::Csneg => aarch64_encoding::neg_reg(&dst, &rm),
        }
    };
    let encoded = patcher.assemble(finding.address, &asm)?;
    if let Some(skip) = enforce_aarch64_size(&encoded, size, finding.address) {
        return Ok(FindingDecision::Skip(skip));
    }
    let strategy = match op {
        CsArithOp::Csinc => PatchStrategy::ReplaceCsincWithMovOrAdd1,
        CsArithOp::Csinv => PatchStrategy::ReplaceCsinvWithMovOrMvn,
        CsArithOp::Csneg => PatchStrategy::ReplaceCsnegWithMovOrNeg,
    };
    Ok(FindingDecision::Plan(PlanOperation {
        address: finding.address,
        strategy,
        kind: finding.kind,
        confidence: finding.confidence,
        size,
        new_bytes: encoded,
        rationale: cs_rationale(finding, &asm),
    }))
}

fn bit_value_for(finding: &Finding) -> Option<bool> {
    match finding.verdict {
        SmtResult::AlwaysTrue => Some(true),
        SmtResult::AlwaysFalse => Some(false),
        _ => None,
    }
}

/// Parse the leading GPR-shaped operands from a `cs*` finding. r2
/// surfaces the trailing condition (`eq`, `ne`, …) as its own operand
/// entry, so we accept up to three GPR operands and ignore anything
/// else. Returns `None` only when fewer than two GPRs can be parsed —
/// every `cs*` rewrite needs at least `Rd, Rn`.
fn parse_cs_operands(finding: &Finding) -> Option<(String, String, Option<String>)> {
    let mut regs = finding
        .operands
        .iter()
        .filter_map(|raw| aarch64_encoding::parse_xreg(raw));
    let dst = regs.next()?;
    let rn = regs.next()?;
    let rm = regs.next();
    Some((dst, rn, rm))
}

fn enforce_aarch64_size(encoded: &[u8], size: usize, address: Address) -> Option<String> {
    if encoded.len() != ARM_INSTRUCTION_BYTES {
        return Some(format!(
            "`AArch64` cs-rewrite at {address} produced {} bytes, expected {ARM_INSTRUCTION_BYTES}",
            encoded.len(),
        ));
    }
    if encoded.len() != size {
        return Some(format!(
            "`AArch64` cs-rewrite at {address} length {} does not match original instruction size {size}",
            encoded.len(),
        ));
    }
    None
}

fn cs_rationale(finding: &Finding, asm: &str) -> String {
    format!(
        "{mnem} at {addr} collapses to `{asm}` ({formula} is always {verdict:?})",
        mnem = finding.mnemonic,
        addr = finding.address,
        formula = finding.formula,
        verdict = finding.verdict,
    )
}
