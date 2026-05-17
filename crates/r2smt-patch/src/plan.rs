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
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use r2smt_common::Address;
    use r2smt_common::smt::SmtResult;
    use r2smt_core::{Confidence, Finding, FindingEvidence, FindingKind};
    use r2smt_ir::testing::InMemoryBytePatcher;
    use r2smt_slicer::condition::BranchCondition;
    use r2smt_slicer::slice::SliceStatus;

    use super::*;

    fn finding(
        verdict: SmtResult,
        kind: FindingKind,
        confidence: Confidence,
        mnemonic: &str,
        address: u64,
        size: u64,
    ) -> Finding {
        Finding {
            address: Address(address),
            function: Address(0x40_1000),
            mnemonic: mnemonic.into(),
            condition: BranchCondition::NotEqual,
            formula: "ZF == 0".into(),
            formula_pretty: "(ZF == 0)".into(),
            formula_z3_pretty: None,
            verdict,
            kind,
            confidence,
            taken_target: Some(Address(0x40_1080)),
            fallthrough_target: Some(Address(address + size)),
            operands: Vec::new(),
            is_thumb: false,
            evidence: FindingEvidence {
                slice_status: SliceStatus::Complete,
                statement_count: 0,
                input_count: 0,
                inputs: vec![],
                unknown_count: 0,
                upstream_resolved_to: None,
            },
            pseudocode: None,
        }
    }

    fn make_patcher() -> InMemoryBytePatcher {
        // Set up a 256-byte buffer mapped to 0x401050 onward. The
        // first two bytes simulate a `jne rel8` (0x75 0x05); the
        // following four simulate `jne rel32` (0x0f 0x85 + 4-byte
        // displacement). Tests pick the address that matches the size
        // they need.
        let mut bytes = vec![0u8; 256];
        bytes[0] = 0x75;
        bytes[1] = 0x05;
        bytes[2] = 0x0f;
        bytes[3] = 0x85;
        bytes[4] = 0x00;
        bytes[5] = 0x00;
        bytes[6] = 0x00;
        bytes[7] = 0x00;
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        patcher.add_assemble("jmp 0x401080", vec![0xe9, 0x2b, 0x00, 0x00, 0x00]);
        patcher
    }

    #[test]
    fn always_false_jne_rel8_plans_two_nops() {
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "jne",
            0x40_1050,
            2,
        );
        let mut patcher = make_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::NopJcc);
        assert_eq!(op.size, 2);
        assert_eq!(op.new_bytes, vec![X86_NOP_BYTE, X86_NOP_BYTE]);
    }

    #[test]
    fn always_true_jne_rel32_plans_jmp_with_nop_padding() {
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            Confidence::High,
            "jne",
            0x40_1052,
            6,
        );
        let mut patcher = make_patcher();
        // Add the entry for the rel32-sized address; the in-memory
        // patcher returns the same encoding for the relative jump.
        patcher.add_assemble("jmp 0x401080", vec![0xe9, 0x29, 0x00, 0x00, 0x00]);
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceJccWithJmp);
        assert_eq!(op.size, 6);
        assert_eq!(op.new_bytes.len(), 6);
        assert_eq!(op.new_bytes[0], 0xe9);
        assert_eq!(*op.new_bytes.last().unwrap(), X86_NOP_BYTE);
    }

    #[test]
    fn low_confidence_finding_is_skipped() {
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::Low,
            "jne",
            0x40_1050,
            2,
        );
        let mut patcher = make_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert_eq!(plan.skipped.len(), 1);
        assert!(plan.skipped[0].1.contains("below threshold"));
    }

    #[test]
    fn setcc_always_false_plans_mov_imm0() {
        // sete al → 0F 94 C0 (3 bytes) at 0x40_1050
        let mut bytes = vec![0u8; 256];
        bytes[0] = 0x0F;
        bytes[1] = 0x94;
        bytes[2] = 0xC0;
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::ConstantCondition,
            Confidence::High,
            "sete",
            0x40_1050,
            3,
        );
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceSetCcWithMovConst);
        assert_eq!(op.new_bytes, vec![0xC6, 0xC0, 0x00]);
        assert_eq!(op.size, 3);
    }

    #[test]
    fn setcc_always_true_plans_mov_imm1() {
        let mut bytes = vec![0u8; 256];
        bytes[0] = 0x0F;
        bytes[1] = 0x94;
        bytes[2] = 0xC0;
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::ConstantCondition,
            Confidence::High,
            "sete",
            0x40_1050,
            3,
        );
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.new_bytes, vec![0xC6, 0xC0, 0x01]);
    }

    #[test]
    fn cmovcc_always_true_plans_unconditional_mov() {
        // cmove eax, ebx → 0F 44 C3 (3 bytes)
        let mut bytes = vec![0u8; 256];
        bytes[0] = 0x0F;
        bytes[1] = 0x44;
        bytes[2] = 0xC3;
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            Confidence::High,
            "cmove",
            0x40_1050,
            3,
        );
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCMovCcWithMovOrNop);
        // 8B C3 + NOP
        assert_eq!(op.new_bytes, vec![0x8B, 0xC3, X86_NOP_BYTE]);
    }

    #[test]
    fn cmovcc_memory_operand_always_true_plans_mov_with_nop_tail() {
        // cmove rax, [rbx + 0x10] — 48 0F 44 43 10 (5 bytes)
        // Always-true ⇒ mov rax, [rbx + 0x10] (48 8B 43 10) + 1 NOP.
        let mut bytes = vec![0u8; 256];
        bytes[..5].copy_from_slice(&[0x48, 0x0F, 0x44, 0x43, 0x10]);
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            Confidence::High,
            "cmove",
            0x40_1050,
            5,
        );
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCMovCcWithMovOrNop);
        assert_eq!(op.new_bytes, vec![0x48, 0x8B, 0x43, 0x10, X86_NOP_BYTE]);
    }

    #[test]
    fn cmovcc_operand_size_16bit_plans_mov_ax_bx() {
        // cmove ax, bx — 66 0F 44 C3 (4 bytes). Must preserve 66H.
        let mut bytes = vec![0u8; 256];
        bytes[..4].copy_from_slice(&[0x66, 0x0F, 0x44, 0xC3]);
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            Confidence::High,
            "cmove",
            0x40_1050,
            4,
        );
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCMovCcWithMovOrNop);
        assert_eq!(op.new_bytes, vec![0x66, 0x8B, 0xC3, X86_NOP_BYTE]);
    }

    #[test]
    fn cmovcc_always_false_plans_all_nops() {
        let mut bytes = vec![0u8; 256];
        bytes[0] = 0x0F;
        bytes[1] = 0x44;
        bytes[2] = 0xC3;
        let mut patcher = InMemoryBytePatcher::new(Address(0x40_1050), bytes);
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "cmove",
            0x40_1050,
            3,
        );
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCMovCcWithMovOrNop);
        assert_eq!(op.new_bytes, vec![X86_NOP_BYTE; 3]);
    }

    #[test]
    fn assembled_jmp_larger_than_original_is_skipped() {
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            Confidence::High,
            "jne",
            0x40_1050,
            2,
        );
        let mut patcher = make_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::X86_64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert_eq!(plan.skipped.len(), 1);
        assert!(plan.skipped[0].1.contains("assembled branch"));
    }

    // --- ARM patching ---

    /// 32-byte patcher fixture for ARM tests. The first 4 bytes
    /// simulate a 4-byte conditional branch instruction at the
    /// chosen address; the remainder is zero. Subsequent
    /// `add_assemble` calls register the expected ARM encodings.
    fn arm_patcher() -> InMemoryBytePatcher {
        InMemoryBytePatcher::new(Address(0x40_1050), vec![0u8; 32])
    }

    #[test]
    fn aarch64_b_eq_always_false_emits_aarch64_nop() {
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "b.eq",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1, "expected one operation");
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::NopJcc);
        assert_eq!(op.size, 4);
        // D503201F little-endian — the architectural `AArch64` NOP.
        assert_eq!(op.new_bytes, vec![0x1F, 0x20, 0x03, 0xD5]);
    }

    #[test]
    fn aarch64_b_eq_always_true_emits_unconditional_b() {
        let f = finding(
            SmtResult::AlwaysTrue,
            FindingKind::OpaquePredicate,
            Confidence::High,
            "b.eq",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        // r2's `pa` for `AArch64` returns the encoding of `b 0x401080`
        // — emulate that here. The exact bytes don't matter for this
        // test; what matters is that the planner asks for `b` syntax.
        patcher.add_assemble("b 0x401080", vec![0x0B, 0x00, 0x00, 0x14]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceJccWithJmp);
        assert_eq!(op.size, 4);
        assert_eq!(op.new_bytes, vec![0x0B, 0x00, 0x00, 0x14]);
    }

    #[test]
    fn aarch64_cbz_always_false_is_nop_jcc() {
        // cbz/cbnz/tbz/tbnz read a register but the branch is still
        // a conditional fall-through, so always-false → NOP, always-
        // true → unconditional `b`.
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "cbz",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);
        assert_eq!(plan.operations[0].strategy, PatchStrategy::NopJcc);
        assert_eq!(plan.operations[0].new_bytes, vec![0x1F, 0x20, 0x03, 0xD5]);
    }

    #[test]
    fn aarch32_bne_always_false_emits_aarch32_nop() {
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "bne",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::NopJcc);
        // E320F000 little-endian — the architectural AArch32 NOP.
        assert_eq!(op.new_bytes, vec![0x00, 0xF0, 0x20, 0xE3]);
    }

    #[test]
    fn aarch32_unconditional_b_is_skipped() {
        // Plain `b` is unconditional — same exclusion as x86 `jmp`.
        // The planner should report it as not a recognised
        // conditional branch.
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "b",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert_eq!(plan.skipped.len(), 1);
        assert!(plan.skipped[0].1.contains("no rewrite strategy"));
    }

    #[test]
    fn aarch32_bl_link_branch_is_skipped() {
        // `bl` is a call (link branch) — rewriting it would orphan
        // its return address. Excluded from the conditional set.
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "bl",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
    }

    #[test]
    fn arm_two_byte_size_is_rejected() {
        // 4-byte alignment is mandatory for ARM mode. A 2-byte
        // mnemonic would be Thumb, which the patcher cannot yet
        // emit safely.
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "b.eq",
            0x40_1050,
            2,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert_eq!(plan.skipped.len(), 1);
        assert!(plan.skipped[0].1.contains("non-4-byte"));
    }

    #[test]
    fn x86_jcc_under_aarch64_classifier_is_skipped() {
        // Cross-ISA noise: an x86 `jne` mnemonic should not be
        // accidentally classified as ARM-conditional under
        // Arch::Aarch64 (`AArch64` uses b.<cond>, never `jne`).
        let f = finding(
            SmtResult::AlwaysFalse,
            FindingKind::DeadBranch,
            Confidence::High,
            "jne",
            0x40_1050,
            4,
        );
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
    }

    // --- `AArch64` cs* family ---

    fn finding_with_operands(verdict: SmtResult, mnemonic: &str, operands: &[&str]) -> Finding {
        let mut f = finding(
            verdict,
            FindingKind::OpaquePredicate,
            Confidence::High,
            mnemonic,
            0x40_1050,
            ARM_INSTRUCTION_BYTES as u64,
        );
        f.operands = operands.iter().map(|s| (*s).into()).collect();
        f
    }

    #[test]
    fn aarch64_cset_always_true_emits_mov_one() {
        let f = finding_with_operands(SmtResult::AlwaysTrue, "cset", &["x0", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x0, #1", vec![0x20, 0x00, 0x80, 0xD2]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1, "skipped: {:?}", plan.skipped);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsetWithMovConst);
        assert_eq!(op.size, 4);
        assert_eq!(op.new_bytes, vec![0x20, 0x00, 0x80, 0xD2]);
    }

    #[test]
    fn aarch64_cset_always_false_emits_mov_zero() {
        let f = finding_with_operands(SmtResult::AlwaysFalse, "cset", &["x3", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x3, #0", vec![0x03, 0x00, 0x80, 0xD2]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1, "skipped: {:?}", plan.skipped);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsetWithMovConst);
        assert_eq!(op.new_bytes, vec![0x03, 0x00, 0x80, 0xD2]);
    }

    #[test]
    fn aarch64_csetm_always_true_emits_mov_minus_one() {
        let f = finding_with_operands(SmtResult::AlwaysTrue, "csetm", &["x5", "ne"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x5, #-1", vec![0x05, 0x00, 0x80, 0x92]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsetWithMovConst);
        assert_eq!(op.new_bytes, vec![0x05, 0x00, 0x80, 0x92]);
    }

    #[test]
    fn aarch64_csel_always_true_picks_rn() {
        let f = finding_with_operands(SmtResult::AlwaysTrue, "csel", &["x0", "x1", "x2", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x0, x1", vec![0xE0, 0x03, 0x01, 0xAA]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCselWithMov);
        assert_eq!(op.new_bytes, vec![0xE0, 0x03, 0x01, 0xAA]);
    }

    #[test]
    fn aarch64_csel_always_false_picks_rm() {
        let f = finding_with_operands(SmtResult::AlwaysFalse, "csel", &["x0", "x1", "x2", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x0, x2", vec![0xE0, 0x03, 0x02, 0xAA]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCselWithMov);
        assert_eq!(op.new_bytes, vec![0xE0, 0x03, 0x02, 0xAA]);
    }

    #[test]
    fn aarch64_csinc_always_true_picks_rn() {
        let f = finding_with_operands(SmtResult::AlwaysTrue, "csinc", &["x0", "x1", "x2", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x0, x1", vec![0xE0, 0x03, 0x01, 0xAA]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsincWithMovOrAdd1);
        assert_eq!(op.new_bytes, vec![0xE0, 0x03, 0x01, 0xAA]);
    }

    #[test]
    fn aarch64_csinc_always_false_emits_add_one() {
        let f = finding_with_operands(SmtResult::AlwaysFalse, "csinc", &["x0", "x1", "x2", "eq"]);
        let mut patcher = arm_patcher();
        // add x0, x2, #1 — encoding is family 91000000 with imm=1.
        patcher.add_assemble("add x0, x2, #1", vec![0x40, 0x04, 0x00, 0x91]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsincWithMovOrAdd1);
        assert_eq!(op.new_bytes, vec![0x40, 0x04, 0x00, 0x91]);
    }

    #[test]
    fn aarch64_csinv_always_false_emits_mvn() {
        let f = finding_with_operands(SmtResult::AlwaysFalse, "csinv", &["x0", "x1", "x2", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mvn x0, x2", vec![0xE0, 0x03, 0x22, 0xAA]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsinvWithMovOrMvn);
        assert_eq!(op.new_bytes, vec![0xE0, 0x03, 0x22, 0xAA]);
    }

    #[test]
    fn aarch64_csneg_always_false_emits_neg() {
        let f = finding_with_operands(SmtResult::AlwaysFalse, "csneg", &["x0", "x1", "x2", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("neg x0, x2", vec![0xE0, 0x03, 0x02, 0xCB]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsnegWithMovOrNeg);
        assert_eq!(op.new_bytes, vec![0xE0, 0x03, 0x02, 0xCB]);
    }

    #[test]
    fn aarch64_cinc_two_operand_aliased_form() {
        // `cinc Rd, Rn, cond` ≡ `csinc Rd, Rn, Rn, !cond`. For an
        // always-false predicate the rewrite is `add Rd, Rn, #1`.
        let f = finding_with_operands(SmtResult::AlwaysFalse, "cinc", &["x0", "x1", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("add x0, x1, #1", vec![0x20, 0x04, 0x00, 0x91]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceCsincWithMovOrAdd1);
        assert_eq!(op.new_bytes, vec![0x20, 0x04, 0x00, 0x91]);
    }

    #[test]
    fn aarch64_cset_with_unrecognised_destination_register_is_skipped() {
        // `sp` is rejected by parse_xreg (patching it would corrupt
        // the call frame). The planner must skip with a clear reason
        // rather than producing wrong bytes.
        let f = finding_with_operands(SmtResult::AlwaysTrue, "cset", &["sp", "eq"]);
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert_eq!(plan.skipped.len(), 1);
        assert!(plan.skipped[0].1.contains("not a recognised GPR"));
    }

    #[test]
    fn aarch64_cset_with_assemble_length_mismatch_is_skipped() {
        // r2 returns a 2-byte encoding (impossible for `AArch64`) — the
        // planner must refuse rather than emit garbage.
        let f = finding_with_operands(SmtResult::AlwaysTrue, "cset", &["x0", "eq"]);
        let mut patcher = arm_patcher();
        patcher.add_assemble("mov x0, #1", vec![0x00, 0x00]);
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert!(plan.skipped[0].1.contains("expected 4"));
    }

    #[test]
    fn aarch64_csel_missing_rm_operand_is_skipped() {
        // r2 produced only `Rd, Rn` for a csel — the planner cannot
        // recover Rm so the rewrite must be skipped.
        let f = finding_with_operands(SmtResult::AlwaysFalse, "csel", &["x0", "x1"]);
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert!(plan.skipped[0].1.contains("missing Rm"));
    }

    // --- AArch32 Thumb mode ---

    fn thumb_finding(verdict: SmtResult, mnemonic: &str, size: u64) -> Finding {
        let mut f = finding(
            verdict,
            FindingKind::DeadBranch,
            Confidence::High,
            mnemonic,
            0x40_1050,
            size,
        );
        f.is_thumb = true;
        f
    }

    #[test]
    fn arm_thumb_jcc_always_false_nops_two_bytes() {
        let f = thumb_finding(SmtResult::AlwaysFalse, "bne", 2);
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1, "skipped: {:?}", plan.skipped);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::NopJcc);
        assert_eq!(op.size, 2);
        // Thumb NOP hint is `BF00` LE → `[0x00, 0xBF]`.
        assert_eq!(op.new_bytes, vec![0x00, 0xBF]);
    }

    #[test]
    fn arm_thumb_jcc_always_false_nops_four_bytes() {
        // Thumb-2 32-bit conditional branch — two NOP halfwords.
        let f = thumb_finding(SmtResult::AlwaysFalse, "bne", 4);
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1, "skipped: {:?}", plan.skipped);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::NopJcc);
        assert_eq!(op.size, 4);
        assert_eq!(op.new_bytes, vec![0x00, 0xBF, 0x00, 0xBF]);
    }

    #[test]
    fn arm_thumb_jcc_odd_size_is_rejected() {
        // A 3-byte instruction cannot exist in Thumb mode (encoding is
        // always 2-byte half-words). The planner must surface this as
        // a skip rather than emitting a partial NOP.
        let f = thumb_finding(SmtResult::AlwaysFalse, "bne", 3);
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert!(plan.skipped[0].1.contains("Thumb instruction"));
    }

    #[test]
    fn arm_thumb_jcc_always_true_assembles_branch_matching_size() {
        // r2 returns a 2-byte Thumb branch encoding for `b 0x401080`.
        // The planner must accept it because length == original size.
        let f = thumb_finding(SmtResult::AlwaysTrue, "bne", 2);
        let mut patcher = arm_patcher();
        patcher.add_assemble("b 0x401080", vec![0x16, 0xE0]);
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert_eq!(plan.operations.len(), 1, "skipped: {:?}", plan.skipped);
        let op = &plan.operations[0];
        assert_eq!(op.strategy, PatchStrategy::ReplaceJccWithJmp);
        assert_eq!(op.new_bytes, vec![0x16, 0xE0]);
    }

    #[test]
    fn arm_thumb_jcc_always_true_refused_when_assembled_branch_exceeds_original() {
        // r2 returns a 4-byte Thumb-2 branch encoding when the planner
        // expected 2 bytes. The planner refuses because a larger
        // replacement would overwrite the next instruction. The early-
        // exit guard reports "assembled branch is …, original …".
        let f = thumb_finding(SmtResult::AlwaysTrue, "bne", 2);
        let mut patcher = arm_patcher();
        patcher.add_assemble("b 0x401080", vec![0x00, 0xF0, 0x16, 0xB8]);
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert!(
            plan.skipped[0].1.contains("assembled branch"),
            "unexpected skip reason: {}",
            plan.skipped[0].1
        );
    }

    #[test]
    fn arm_thumb_jcc_always_true_refused_when_assembled_branch_smaller_than_original() {
        // r2 returns a 2-byte Thumb encoding for a 4-byte Thumb-2
        // conditional branch. Padding with a NOP halfword would change
        // the instruction stream shape, so the planner refuses.
        let f = thumb_finding(SmtResult::AlwaysTrue, "bne", 4);
        let mut patcher = arm_patcher();
        patcher.add_assemble("b 0x401080", vec![0x16, 0xE0]);
        let plan = build_plan(&[f], Confidence::High, Arch::Arm, &mut patcher).unwrap();
        assert!(plan.operations.is_empty());
        assert!(
            plan.skipped[0].1.contains("refusing to pad"),
            "unexpected skip reason: {}",
            plan.skipped[0].1
        );
    }

    #[test]
    fn aarch64_cset_with_both_possible_verdict_is_skipped() {
        // BothPossible cannot drive a rewrite. The planner should not
        // panic — it should report a clear skip reason.
        let f = finding_with_operands(SmtResult::BothPossible, "cset", &["x0", "eq"]);
        let mut patcher = arm_patcher();
        let plan = build_plan(&[f], Confidence::High, Arch::Aarch64, &mut patcher).unwrap();
        // Skipped before reaching the planner because the kind is
        // `OpaquePredicate` but the verdict is BothPossible — the
        // outer gate would normally classify this as a RealBranch, so
        // we exercise the planner's own verdict check directly via
        // the lower-level call path. Either skip reason is acceptable;
        // assert no operation produced.
        assert!(plan.operations.is_empty());
    }
}
