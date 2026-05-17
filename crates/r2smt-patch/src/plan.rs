//! Build a [`PatchPlan`] from r2SMT findings.
//!
//! Translates each actionable [`Finding`] into a concrete byte
//! sequence the patcher will write. Strategy v0 supports
//! `nop_jcc` and `replace_jcc_with_jmp` — operand-aware
//! `setcc` / `cmovcc` synthesis stays deferred per `SPEC.md` §5.7.

use r2smt_common::smt::SmtResult;
use r2smt_common::{Address, Arch, Error, Result};
use r2smt_core::{Confidence, Finding, FindingKind};
use r2smt_ir::byte_patcher::BytePatcher;
use r2smt_report::PatchStrategy;
use tracing::{debug, warn};

use crate::aarch64_encoding;
use crate::arm_encoding::{
    ARM_INSTRUCTION_BYTES, THUMB_HALFWORD_BYTES, arm_nop_buffer, thumb_nop_buffer,
};
use crate::x86_encoding::{nop_buffer, patch_cmovcc_to_mov, patch_setcc};

/// Single-byte x86 NOP opcode used by `nop_jcc` / `nop_padding`.
const X86_NOP_BYTE: u8 = 0x90;

/// A single ready-to-execute patch operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanOperation {
    /// Address of the patched instruction.
    pub address: Address,
    /// Strategy that produced this operation.
    pub strategy: PatchStrategy,
    /// Finding kind that motivated it.
    pub kind: FindingKind,
    /// Confidence forwarded from the finding.
    pub confidence: Confidence,
    /// Size, in bytes, of the original instruction.
    pub size: usize,
    /// Bytes to write at `address`.
    pub new_bytes: Vec<u8>,
    /// Human-readable rationale forwarded for the manifest.
    pub rationale: String,
}

/// Ordered list of operations the patcher will execute.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatchPlan {
    /// Operations in execution order (preserving the order of the
    /// findings they were built from).
    pub operations: Vec<PlanOperation>,
    /// Findings that were filtered out, paired with the human reason.
    /// Surfaced so the CLI can explain what was skipped.
    pub skipped: Vec<(Address, String)>,
}

/// Maximum number of bytes r2SMT will rewrite for a single
/// instruction. Sized to fit a `jmp rel32` (5 bytes) plus a generous
/// safety margin so `x86_64` encodings still fit.
pub const MAX_INSTRUCTION_SIZE: usize = 16;

/// Construct a [`PatchPlan`] from a slice of findings.
///
/// Each finding is gated by:
///
/// 1. `is_actionable()` — only opaque / dead / constant kinds.
/// 2. `confidence <= min_confidence` (using the `Ord` semantics in
///    `r2smt-core`, which place `High` lowest).
/// 3. The presence of an instruction-size measurement (the patcher
///    needs to know how many bytes to preserve).
///
/// `arch` selects the rewrite ISA: x86 / `x86_64` use the legacy
/// `jcc` / `setcc` / `cmovcc` strategies; `AArch64` / `AArch32` use
/// the ARM `b.<cond>` / `b<cond>` / `cbz` / `cbnz` / `tbz` / `tbnz`
/// strategies (NOP-out for always-false, replace with `b <target>`
/// for always-true). ARM `setcc` / `cmovcc` analogs (`cset` /
/// `csel`) are deferred and surface as "no rewrite strategy" skips.
///
/// The function never writes to the binary. Callers use `apply_plan`
/// to commit operations.
///
/// # Errors
///
/// Returns the first failure produced by the `BytePatcher` while
/// measuring instruction size or assembling replacement bytes.
pub fn build_plan(
    findings: &[Finding],
    min_confidence: Confidence,
    arch: Arch,
    patcher: &mut dyn BytePatcher,
) -> Result<PatchPlan> {
    let mut operations: Vec<PlanOperation> = Vec::new();
    let mut skipped: Vec<(Address, String)> = Vec::new();

    for finding in findings {
        match consider_finding(finding, min_confidence, arch, patcher)? {
            FindingDecision::Plan(op) => operations.push(op),
            FindingDecision::Skip(reason) => skipped.push((finding.address, reason)),
        }
    }

    Ok(PatchPlan {
        operations,
        skipped,
    })
}

enum FindingDecision {
    Plan(PlanOperation),
    Skip(String),
}

fn consider_finding(
    finding: &Finding,
    min_confidence: Confidence,
    arch: Arch,
    patcher: &mut dyn BytePatcher,
) -> Result<FindingDecision> {
    if !finding.is_actionable() {
        return Ok(FindingDecision::Skip(format!(
            "kind {:?} is not actionable",
            finding.kind
        )));
    }
    if finding.confidence > min_confidence {
        return Ok(FindingDecision::Skip(format!(
            "confidence {:?} below threshold {:?}",
            finding.confidence, min_confidence,
        )));
    }

    let mnemonic = finding.mnemonic.to_ascii_lowercase();
    let kind = classify_mnemonic(&mnemonic, arch);
    if kind == MnemonicKind::Other {
        return Ok(FindingDecision::Skip(format!(
            "{mnemonic} not a recognised branch / setcc / cmovcc for {arch:?} — no rewrite strategy"
        )));
    }

    let size = measure_instruction_size(finding, patcher)?;
    if size == 0 || size > MAX_INSTRUCTION_SIZE {
        return Ok(FindingDecision::Skip(format!(
            "instruction at {addr} has unsupported size {size}",
            addr = finding.address,
        )));
    }
    if arch_is_arm(arch) && !finding.is_thumb && size % ARM_INSTRUCTION_BYTES != 0 {
        return Ok(FindingDecision::Skip(format!(
            "ARM instruction at {addr} has non-4-byte size {size} (Thumb mode?)",
            addr = finding.address,
        )));
    }
    if finding.is_thumb && size % THUMB_HALFWORD_BYTES != 0 {
        return Ok(FindingDecision::Skip(format!(
            "Thumb instruction at {addr} has odd size {size}",
            addr = finding.address,
        )));
    }

    match kind {
        MnemonicKind::Jcc => plan_jcc(finding, size, arch, patcher),
        MnemonicKind::SetCc => plan_setcc(finding, size, patcher),
        MnemonicKind::CMovCc => plan_cmovcc(finding, size, patcher),
        MnemonicKind::Cset { all_ones } => plan_aarch64_cset(finding, size, patcher, all_ones),
        MnemonicKind::Csel => plan_aarch64_csel(finding, size, patcher),
        MnemonicKind::CsArith { op, aliased } => {
            plan_aarch64_cs_arith(finding, size, patcher, op, aliased)
        }
        MnemonicKind::Other => unreachable!(),
    }
}

fn arch_is_arm(arch: Arch) -> bool {
    matches!(arch, Arch::Aarch64 | Arch::Arm)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MnemonicKind {
    Jcc,
    SetCc,
    CMovCc,
    /// `AArch64` `cset` (`all_ones = false`) or `csetm` (`all_ones =
    /// true`). One operand: destination GPR.
    Cset {
        all_ones: bool,
    },
    /// `AArch64` `csel` — three operands `Rd, Rn, Rm`.
    Csel,
    /// `AArch64` `csinc` / `csinv` / `csneg` (3-op) and their 2-op
    /// aliases `cinc` / `cinv` / `cneg`. `aliased = true` means the
    /// disassembler used the 2-op alias form; the rewrite recipe is
    /// identical because Armv8 defines `cinc Rd, Rn, cond` ≡
    /// `csinc Rd, Rn, Rn, !cond` (and analogously for `inv` / `neg`).
    CsArith {
        op: CsArithOp,
        aliased: bool,
    },
    Other,
}

/// Which `AArch64` arithmetic cs-instruction the planner is rewriting.
/// Selects the "false" arm of the rewrite (the true arm is always a
/// `mov Rd, Rn`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsArithOp {
    /// `csinc` / `cinc` → `add Rd, Rm, #1` when the predicate is false.
    Csinc,
    /// `csinv` / `cinv` → `mvn Rd, Rm` when the predicate is false.
    Csinv,
    /// `csneg` / `cneg` → `neg Rd, Rm` when the predicate is false.
    Csneg,
}

fn classify_mnemonic(mnemonic: &str, arch: Arch) -> MnemonicKind {
    match arch {
        Arch::X86 | Arch::X86_64 => classify_x86_mnemonic(mnemonic),
        Arch::Aarch64 => classify_aarch64_mnemonic(mnemonic),
        Arch::Arm => classify_aarch32_mnemonic(mnemonic),
        _ => MnemonicKind::Other,
    }
}

fn classify_x86_mnemonic(mnemonic: &str) -> MnemonicKind {
    if mnemonic.starts_with("cmov") {
        MnemonicKind::CMovCc
    } else if mnemonic.starts_with("set") {
        MnemonicKind::SetCc
    } else if mnemonic.starts_with('j') && mnemonic != "jmp" {
        MnemonicKind::Jcc
    } else {
        MnemonicKind::Other
    }
}

fn classify_aarch64_mnemonic(mnemonic: &str) -> MnemonicKind {
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

fn classify_aarch32_mnemonic(mnemonic: &str) -> MnemonicKind {
    // AArch32 conditional branches use the suffix form `b<cond>`.
    // Exclude unconditional `b`, link forms `bl`/`blx`, and indirect
    // `bx`. The valid suffixes are the standard AAPCS condition codes.
    const COND_SUFFIXES: &[&str] = &[
        "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt",
        "le",
    ];
    if let Some(suffix) = mnemonic.strip_prefix('b')
        && COND_SUFFIXES.contains(&suffix)
    {
        return MnemonicKind::Jcc;
    }
    MnemonicKind::Other
}

fn plan_jcc(
    finding: &Finding,
    size: usize,
    arch: Arch,
    patcher: &mut dyn BytePatcher,
) -> Result<FindingDecision> {
    let strategy = jcc_strategy(finding)?;
    let new_bytes = match strategy {
        PatchStrategy::NopJcc => nop_bytes_for(arch, size, finding.is_thumb)?,
        PatchStrategy::ReplaceJccWithJmp => {
            let Some(target) = finding.taken_target else {
                return Ok(FindingDecision::Skip(
                    "AlwaysTrue jcc has no resolved taken target".into(),
                ));
            };
            let assembled =
                patcher.assemble(finding.address, &unconditional_branch_asm(arch, target))?;
            if assembled.len() > size {
                warn!(
                    target: "r2smt::patch",
                    addr = %finding.address,
                    asm_size = assembled.len(),
                    original_size = size,
                    "assembled branch larger than original — skipping"
                );
                return Ok(FindingDecision::Skip(format!(
                    "assembled branch is {asm} bytes, original instruction is {orig}",
                    asm = assembled.len(),
                    orig = size,
                )));
            }
            if arch_is_arm(arch) && assembled.len() != size {
                // ARM instructions are fixed-width; a non-matching
                // assembled length means we'd leave partial-instruction
                // bytes in the patch slot. Refuse instead of padding
                // with x86 NOPs that the ARM CPU would decode as
                // garbage.
                return Ok(FindingDecision::Skip(format!(
                    "ARM assembled branch is {asm} bytes, original instruction is {orig} — refusing to pad",
                    asm = assembled.len(),
                    orig = size,
                )));
            }
            pad_to_size(arch, assembled, size)?
        }
        _ => {
            return Ok(FindingDecision::Skip(format!(
                "{strategy:?} not applicable to jcc"
            )));
        }
    };

    debug!(
        target: "r2smt::patch",
        addr = %finding.address,
        size,
        strategy = strategy.as_str(),
        "planned jcc operation"
    );

    Ok(FindingDecision::Plan(PlanOperation {
        address: finding.address,
        strategy,
        kind: finding.kind,
        confidence: finding.confidence,
        size,
        new_bytes,
        rationale: rationale_for(finding, strategy),
    }))
}

fn plan_setcc(
    finding: &Finding,
    size: usize,
    patcher: &mut dyn BytePatcher,
) -> Result<FindingDecision> {
    let value = match finding.verdict {
        SmtResult::AlwaysTrue => true,
        SmtResult::AlwaysFalse => false,
        _ => {
            return Err(Error::parse(
                "patch_plan",
                format!(
                    "{:?} verdict at {addr} cannot drive a setcc rewrite",
                    finding.verdict,
                    addr = finding.address,
                ),
            ));
        }
    };
    let original = patcher.read_bytes(finding.address, size)?;
    let new_bytes = match patch_setcc(&original, value) {
        Ok(bytes) => bytes,
        Err(err) => {
            return Ok(FindingDecision::Skip(format!(
                "setcc byte rewrite failed: {err}"
            )));
        }
    };
    let strategy = PatchStrategy::ReplaceSetCcWithMovConst;
    debug!(
        target: "r2smt::patch",
        addr = %finding.address,
        size,
        strategy = strategy.as_str(),
        value,
        "planned setcc operation"
    );
    Ok(FindingDecision::Plan(PlanOperation {
        address: finding.address,
        strategy,
        kind: finding.kind,
        confidence: finding.confidence,
        size,
        new_bytes,
        rationale: setcc_rationale(finding, value),
    }))
}

fn plan_cmovcc(
    finding: &Finding,
    size: usize,
    patcher: &mut dyn BytePatcher,
) -> Result<FindingDecision> {
    let always_true = match finding.verdict {
        SmtResult::AlwaysTrue => true,
        SmtResult::AlwaysFalse => false,
        _ => {
            return Err(Error::parse(
                "patch_plan",
                format!(
                    "{:?} verdict at {addr} cannot drive a cmovcc rewrite",
                    finding.verdict,
                    addr = finding.address,
                ),
            ));
        }
    };

    let new_bytes = if always_true {
        let original = patcher.read_bytes(finding.address, size)?;
        match patch_cmovcc_to_mov(&original) {
            Ok(bytes) => bytes,
            Err(err) => {
                return Ok(FindingDecision::Skip(format!(
                    "cmovcc byte rewrite failed: {err}"
                )));
            }
        }
    } else {
        // Always-false: the conditional move never fires — NOP the
        // whole instruction so the destination keeps its prior value.
        nop_buffer(size)
    };

    let strategy = PatchStrategy::ReplaceCMovCcWithMovOrNop;
    debug!(
        target: "r2smt::patch",
        addr = %finding.address,
        size,
        strategy = strategy.as_str(),
        always_true,
        "planned cmovcc operation"
    );
    Ok(FindingDecision::Plan(PlanOperation {
        address: finding.address,
        strategy,
        kind: finding.kind,
        confidence: finding.confidence,
        size,
        new_bytes,
        rationale: cmovcc_rationale(finding, always_true),
    }))
}

/// Rewrite an `AArch64` `cset` / `csetm` whose predicate the solver
/// proved constant. The destination is read from
/// [`Finding::operands`]; the new instruction is `mov Rd, #imm` where
/// `imm` is `0`, `1`, or `-1` (`csetm` with predicate true).
fn plan_aarch64_cset(
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
fn plan_aarch64_csel(
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
fn plan_aarch64_cs_arith(
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

fn setcc_rationale(finding: &Finding, value: bool) -> String {
    let target = i32::from(value);
    format!(
        "{mnem} at {addr} always sets its destination to {target} ({formula} is always {value})",
        mnem = finding.mnemonic,
        addr = finding.address,
        formula = finding.formula,
        value = if value { "true" } else { "false" },
    )
}

fn cmovcc_rationale(finding: &Finding, always_true: bool) -> String {
    if always_true {
        format!(
            "{mnem} at {addr} always moves ({formula} is always true) — rewritten as unconditional MOV",
            mnem = finding.mnemonic,
            addr = finding.address,
            formula = finding.formula,
        )
    } else {
        format!(
            "{mnem} at {addr} never moves ({formula} is always false) — NOPed",
            mnem = finding.mnemonic,
            addr = finding.address,
            formula = finding.formula,
        )
    }
}

fn jcc_strategy(finding: &Finding) -> Result<PatchStrategy> {
    match finding.verdict {
        SmtResult::AlwaysFalse => Ok(PatchStrategy::NopJcc),
        SmtResult::AlwaysTrue => Ok(PatchStrategy::ReplaceJccWithJmp),
        _ => Err(Error::parse(
            "patch_plan",
            format!(
                "{:?} verdict at {addr} cannot drive a jcc rewrite",
                finding.verdict,
                addr = finding.address,
            ),
        )),
    }
}

/// Return a `size`-byte NOP buffer encoded for `arch`.
///
/// On x86 the encoding is a string of single-byte `0x90`. On ARM
/// (`AArch64` / `AArch32`) it tiles the architectural 4-byte NOP hint.
/// `size` must be a multiple of 4 for ARM; non-ARM archs accept any
/// size.
fn nop_bytes_for(arch: Arch, size: usize, is_thumb: bool) -> Result<Vec<u8>> {
    if is_thumb {
        return thumb_nop_buffer(size);
    }
    if arch_is_arm(arch) {
        arm_nop_buffer(arch, size)
    } else {
        Ok(vec![X86_NOP_BYTE; size])
    }
}

/// Pad an assembled branch byte-string up to `size`, using NOP fill
/// that's safe to execute under `arch`. On ARM the assembled length
/// must already equal `size` (callers enforce this); the function
/// becomes a no-op pass-through. On x86 it appends `0x90` until full.
fn pad_to_size(arch: Arch, mut bytes: Vec<u8>, size: usize) -> Result<Vec<u8>> {
    if arch_is_arm(arch) {
        // ARM paths reject mismatched lengths upstream; if they get
        // here something is wrong with the caller, not the encoding.
        if bytes.len() != size {
            return Err(Error::parse(
                "patch_plan.pad",
                format!(
                    "ARM assembled length {} mismatched target size {}",
                    bytes.len(),
                    size
                ),
            ));
        }
        return Ok(bytes);
    }
    while bytes.len() < size {
        bytes.push(X86_NOP_BYTE);
    }
    Ok(bytes)
}

fn unconditional_branch_asm(arch: Arch, target: Address) -> String {
    if arch_is_arm(arch) {
        format!("b {target}")
    } else {
        format!("jmp {target}")
    }
}

fn rationale_for(finding: &Finding, strategy: PatchStrategy) -> String {
    match strategy {
        PatchStrategy::NopJcc => format!(
            "{mnem} at {addr} is never taken ({formula} is always false)",
            mnem = finding.mnemonic,
            addr = finding.address,
            formula = finding.formula,
        ),
        PatchStrategy::ReplaceJccWithJmp => format!(
            "{mnem} at {addr} is always taken ({formula} is always true)",
            mnem = finding.mnemonic,
            addr = finding.address,
            formula = finding.formula,
        ),
        _ => finding.formula.clone(),
    }
}

fn measure_instruction_size(finding: &Finding, patcher: &mut dyn BytePatcher) -> Result<usize> {
    // The patcher does not have direct access to instruction sizes;
    // fall back to reading bytes until the next instruction. The
    // simplest portable proxy: read up to `MAX_INSTRUCTION_SIZE` and
    // then probe one byte at a time would require an instruction
    // length decoder. Instead, we rely on the caller's knowledge of
    // the instruction's footprint surfaced via `taken_target` and
    // `fallthrough_target`: for a `jcc`, the fallthrough address sits
    // immediately after the instruction's last byte, so
    // `fallthrough - address` is the instruction size.
    if let Some(ft) = finding.fallthrough_target {
        let raw = ft.get().saturating_sub(finding.address.get());
        if raw > 0 {
            if let Ok(size) = usize::try_from(raw) {
                if size <= MAX_INSTRUCTION_SIZE {
                    // Sanity: verify the patcher can actually read that
                    // many bytes — surfaces unmapped addresses up front.
                    let _ = patcher.read_bytes(finding.address, size)?;
                    return Ok(size);
                }
            }
        }
    }
    Err(Error::parse(
        "patch_plan",
        format!(
            "could not determine size of instruction at {addr}",
            addr = finding.address,
        ),
    ))
}

#[cfg(test)]
mod tests;
