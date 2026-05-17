//! Walk a [`Program`] and emit one [`BranchCandidate`] for every
//! conditional instruction encountered.
//!
//! No backward slicing yet (Phase 3); this pass only labels candidates
//! with their owning function / block, condition family, and — for
//! `jcc` — taken / fallthrough targets derived from operand syntax and
//! instruction size.

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Program};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::condition::{self, BranchCondition, BranchKind};

/// One conditional instruction discovered in a program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCandidate {
    /// Instruction address.
    pub address: Address,
    /// Owning function start address.
    pub function: Address,
    /// Owning basic-block start address.
    pub block: Address,
    /// Conditional family (`jcc` / `setcc` / `cmovcc`).
    pub kind: BranchKind,
    /// Mnemonic as reported by radare2 (lowercased).
    pub mnemonic: String,
    /// Symbolic interpretation of the condition.
    pub condition: BranchCondition,
    /// Predicate string, e.g. `"ZF == 0"`. Convenience for reporting.
    pub formula: String,
    /// Where control transfers when the branch is taken, if statically
    /// resolvable. Always populated for `jcc` with an immediate target.
    pub taken_target: Option<Address>,
    /// Address of the instruction immediately following this one.
    /// Populated for every candidate (kind-independent).
    pub fallthrough_target: Option<Address>,
    /// Register the condition compares against, for `AArch64`
    /// compare-and-branch (`cbz`/`cbnz`/`tbz`/`tbnz`). `None` for
    /// flag-based conditions.
    #[serde(default)]
    pub compare_register: Option<String>,
    /// Bit index inspected by `tbz` / `tbnz`. `None` for all other
    /// conditions.
    #[serde(default)]
    pub bit_index: Option<u8>,
    /// `Some(target)` when the analyser that produced the CFG already
    /// proved the branch unconditional — the containing block has
    /// exactly one successor. Downstream consumers may short-circuit
    /// the SMT pipeline and emit a `Finding` (defined in `r2smt-core`)
    /// with `High` confidence directly. `None` for branches with the
    /// usual two-way successor set, or for instructions whose CFG was
    /// not analysed.
    #[serde(default)]
    pub upstream_resolved: Option<Address>,
    /// Raw textual operand strings as reported by the disassembler, in
    /// source order. Downstream consumers (e.g. the `cset` / `csel`
    /// patcher) need the destination register name and other operand
    /// text to synthesise rewrites. Empty for legacy fixtures and for
    /// instructions whose operand list could not be parsed.
    #[serde(default)]
    pub operand_raws: Vec<String>,
    /// `true` when the conditional instruction is encoded in Thumb
    /// mode (`AArch32` only). Forwarded from
    /// [`r2smt_ir::program::Instruction::is_thumb`] so downstream
    /// consumers (patcher) can pick the right encoding family without
    /// re-reading the program model.
    #[serde(default)]
    pub is_thumb: bool,
}

/// Collect every conditional instruction in `program`.
///
/// Classification dispatches on `program.arch` so the same walker
/// works for x86 / `x86_64` (`jcc` / `setcc` / `cmovcc`) and
/// `AArch64` (`b.<cond>`) functions.
#[must_use]
pub fn collect_branches(program: &Program) -> Vec<BranchCandidate> {
    let mut out = Vec::new();
    for function in &program.functions {
        collect_into(function, program.arch, &mut out);
    }
    debug!(
        target: "r2smt::slicer",
        candidates = out.len(),
        functions = program.functions.len(),
        "branch collection complete"
    );
    out
}

/// Collect every conditional instruction belonging to `function`
/// under `arch`.
#[must_use]
pub fn collect_function_branches(function: &Function, arch: Arch) -> Vec<BranchCandidate> {
    let mut out = Vec::new();
    collect_into(function, arch, &mut out);
    out
}

fn collect_into(function: &Function, arch: Arch, out: &mut Vec<BranchCandidate>) {
    for block in &function.blocks {
        for insn in &block.instructions {
            let Some((kind, condition)) = condition::classify(&insn.mnemonic, arch) else {
                continue;
            };
            out.push(make_candidate(function, block, insn, kind, condition));
        }
    }
}

fn make_candidate(
    function: &Function,
    block: &BasicBlock,
    insn: &Instruction,
    kind: BranchKind,
    condition: BranchCondition,
) -> BranchCandidate {
    let (taken_target, compare_register, bit_index) = parse_branch_operands(insn, kind, condition);
    let fallthrough_target = fallthrough_of(insn);
    // Detect "branch already lowered upstream": the analyser that
    // produced the CFG only attached a single successor to this
    // block, so the cjmp's two-way semantics collapsed to a single
    // target long before the SMT pipeline runs. Only meaningful for
    // `Jcc` — `SetCc` / `CMovCc` write a register / move data rather
    // than redirecting control flow, so their successor set is
    // unrelated to the condition's outcome.
    let upstream_resolved = if matches!(kind, BranchKind::Jcc) && block.successors.len() == 1 {
        Some(block.successors[0])
    } else {
        None
    };
    let operand_raws = insn
        .operands
        .iter()
        .map(|operand| operand.raw.clone())
        .collect();
    BranchCandidate {
        address: insn.address,
        function: function.address,
        block: block.address,
        kind,
        mnemonic: insn.mnemonic.clone(),
        condition,
        formula: condition.formula().to_string(),
        taken_target,
        fallthrough_target,
        compare_register,
        bit_index,
        upstream_resolved,
        operand_raws,
        is_thumb: insn.is_thumb || function.is_thumb,
    }
}

/// Extract per-instruction branch metadata that varies by family:
/// the taken target (for `jcc` with an immediate label), the
/// compare-against register (for `AArch64` cbz/cbnz/tbz/tbnz), and
/// the bit index (for tbz/tbnz). Flag-based branches return `(None,
/// None, None)` for the two parameter fields.
fn parse_branch_operands(
    insn: &Instruction,
    kind: BranchKind,
    condition: BranchCondition,
) -> (Option<Address>, Option<String>, Option<u8>) {
    let taken_target = match kind {
        BranchKind::Jcc => resolve_jcc_target(insn),
        BranchKind::SetCc | BranchKind::CMovCc => None,
    };
    match condition {
        BranchCondition::RegisterZero | BranchCondition::RegisterNotZero => {
            // `cbz Rn, label` — first operand is the compared
            // register, second is the target label.
            let reg = insn
                .operands
                .first()
                .map(|o| o.raw.trim().to_ascii_lowercase());
            let target = insn
                .operands
                .get(1)
                .and_then(|o| parse_hex_or_decimal(&o.raw));
            (target.or(taken_target), reg, None)
        }
        BranchCondition::BitZero | BranchCondition::BitNotZero => {
            // `tbz Rn, #bit, label`.
            let reg = insn
                .operands
                .first()
                .map(|o| o.raw.trim().to_ascii_lowercase());
            let bit = insn.operands.get(1).and_then(|o| parse_bit_index(&o.raw));
            let target = insn
                .operands
                .get(2)
                .and_then(|o| parse_hex_or_decimal(&o.raw));
            (target.or(taken_target), reg, bit)
        }
        _ => (taken_target, None, None),
    }
}

fn parse_bit_index(raw: &str) -> Option<u8> {
    let trimmed = raw.trim();
    let body = trimmed.strip_prefix('#').unwrap_or(trimmed).trim_start();
    if let Some(rest) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).ok()?.try_into().ok()
    } else {
        body.parse::<u8>().ok()
    }
}

/// Parse the first operand of a `jcc` as an immediate address. Returns
/// `None` for indirect / register targets, which we cannot resolve
/// without further analysis.
fn resolve_jcc_target(insn: &Instruction) -> Option<Address> {
    let raw = insn.operands.first()?.raw.trim();
    parse_hex_or_decimal(raw)
}

fn parse_hex_or_decimal(raw: &str) -> Option<Address> {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        return u64::from_str_radix(rest, 16).ok().map(Address);
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return trimmed.parse::<u64>().ok().map(Address);
    }
    None
}

fn fallthrough_of(insn: &Instruction) -> Option<Address> {
    insn.address
        .get()
        .checked_add(u64::from(insn.size))
        .map(Address)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};

    use super::*;

    fn insn(addr: u64, size: u8, mnemonic: &str, operands: &[&str]) -> Instruction {
        Instruction {
            address: Address(addr),
            size,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands: operands
                .iter()
                .map(|raw| Operand {
                    raw: (*raw).into(),
                    kind: OperandKind::Unknown,
                })
                .collect(),
            esil: None,
            pcode: None,
            is_thumb: false,
        }
    }

    fn one_block_program(insns: Vec<Instruction>) -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: insns,
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    #[test]
    fn jne_with_immediate_target_resolves_both_edges() {
        let program = one_block_program(vec![
            insn(0x40_1000, 2, "xor", &["eax", "eax"]),
            insn(0x40_1002, 6, "jne", &["0x401080"]),
        ]);
        let candidates = collect_branches(&program);
        assert_eq!(candidates.len(), 1);
        let cand = &candidates[0];
        assert_eq!(cand.address, Address(0x40_1002));
        assert_eq!(cand.kind, BranchKind::Jcc);
        assert_eq!(cand.condition, BranchCondition::NotEqual);
        assert_eq!(cand.taken_target, Some(Address(0x40_1080)));
        assert_eq!(cand.fallthrough_target, Some(Address(0x40_1008)));
        assert_eq!(cand.function, Address(0x40_1000));
        assert_eq!(cand.block, Address(0x40_1000));
        assert_eq!(cand.formula, "ZF == 0");
    }

    #[test]
    fn jmp_unconditional_is_ignored() {
        let program = one_block_program(vec![insn(0x40_1000, 5, "jmp", &["0x401080"])]);
        assert!(collect_branches(&program).is_empty());
    }

    #[test]
    fn indirect_jcc_emits_candidate_without_target() {
        let program = one_block_program(vec![insn(0x40_1000, 2, "je", &["rax"])]);
        let candidates = collect_branches(&program);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].taken_target, None);
        assert_eq!(candidates[0].fallthrough_target, Some(Address(0x40_1002)));
    }

    #[test]
    fn setcc_and_cmovcc_have_no_taken_target() {
        let program = one_block_program(vec![
            insn(0x40_1000, 3, "sete", &["al"]),
            insn(0x40_1003, 3, "cmovne", &["eax", "ebx"]),
        ]);
        let candidates = collect_branches(&program);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].kind, BranchKind::SetCc);
        assert_eq!(candidates[0].condition, BranchCondition::Equal);
        assert_eq!(candidates[0].taken_target, None);
        assert_eq!(candidates[1].kind, BranchKind::CMovCc);
        assert_eq!(candidates[1].condition, BranchCondition::NotEqual);
        assert_eq!(candidates[1].taken_target, None);
    }

    #[test]
    fn collect_function_branches_respects_function_address() {
        let program = one_block_program(vec![insn(0x40_1000, 2, "je", &["0x401010"])]);
        let candidates = collect_function_branches(&program.functions[0], program.arch);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].function, Address(0x40_1000));
    }

    #[test]
    fn json_round_trip_preserves_all_fields() {
        let program = one_block_program(vec![insn(0x40_1000, 2, "je", &["0x401010"])]);
        let candidates = collect_branches(&program);
        let json = serde_json::to_string(&candidates).unwrap();
        let back: Vec<BranchCandidate> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, candidates);
    }
}
