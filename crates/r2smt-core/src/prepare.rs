//! Slice → lift → SSA → optimize, with a bounded budget auto-retry.
//!
//! r2SMT lifts raw radare2 disassembly, so a heavily-obfuscated branch
//! often produces a data-flow chain longer than the slicer's default
//! instruction budget. When that happens the slice is reported
//! [`SliceStatus::Truncated`] and the verdict downgrades to
//! `Unsound` — even though the branch *is* decidable, just bigger than
//! 32 instructions. This module re-slices such a branch **once** with
//! an escalated budget; the post-SSA optimizer
//! ([`r2smt_ssa::optimize_slice`]) then keeps the larger formula
//! tractable for the solver.
//!
//! Only *budget-exhaustion* truncations are retried. Structural
//! truncations (a `call`, a memory access, an unsupported instruction,
//! no flag producer) are not budget problems — a larger budget cannot
//! change them, so retrying would only waste work.
//!
//! This is a use-case: it depends on the slicer and SSA domain crates
//! only, never on an SMT adapter. The CLI still dispatches the solver.

use r2smt_common::Arch;
use r2smt_ir::program::Function;
use r2smt_slicer::slice::SliceStatus;
use r2smt_slicer::{BranchCandidate, SliceLimits, lift_slice, slice_branch};
use r2smt_ssa::{SsaLiftedSlice, optimize_slice, ssa_convert};
use tracing::debug;

/// Factor applied to `max_instructions` on the single re-slice retry
/// (32 → 256 at the default budget). Large enough to absorb the
/// register-shuffle / constant-move padding obfuscators emit, bounded
/// so a pathological branch cannot make the slicer walk unboundedly.
const RESLICE_BUDGET_FACTOR: usize = 8;

/// Extra basic blocks granted on retry when the original truncation
/// was block-budget exhaustion. Kept tiny (the slicer still stops on
/// joins / cycles / function entry) — a guardrail, not a free pass.
const RESLICE_EXTRA_BLOCKS: u32 = 1;

/// Slice `candidate` in `function`, lift it, SSA-rename it, and run the
/// pre-solver optimizer. If the slice truncated purely because it ran
/// out of instruction (or block) budget, re-slice **once** with an
/// escalated budget and return the optimized retry instead.
///
/// Pure with respect to the inputs; the returned slice is ready for
/// the SMT backend.
#[must_use]
pub fn prepare_ssa(
    function: &Function,
    candidate: &BranchCandidate,
    limits: &SliceLimits,
    arch: Arch,
) -> SsaLiftedSlice {
    let first = build(function, candidate, limits, arch);

    if let SliceStatus::Truncated { reason } = &first.status
        && is_budget_truncation(reason)
    {
        let retry_limits = SliceLimits {
            max_instructions: limits
                .max_instructions
                .saturating_mul(RESLICE_BUDGET_FACTOR),
            max_basic_blocks: limits.max_basic_blocks.saturating_add(RESLICE_EXTRA_BLOCKS),
            ..*limits
        };
        debug!(
            target: "r2smt::core",
            at = %candidate.address,
            reason = %reason,
            old_budget = limits.max_instructions,
            new_budget = retry_limits.max_instructions,
            "budget truncation — re-slicing once with escalated budget"
        );
        return build(function, candidate, &retry_limits, arch);
    }

    first
}

/// One slice → lift → ssa → optimize pass.
fn build(
    function: &Function,
    candidate: &BranchCandidate,
    limits: &SliceLimits,
    arch: Arch,
) -> SsaLiftedSlice {
    let slice = slice_branch(candidate, function, limits, arch);
    let lifted = lift_slice(&slice, arch);
    optimize_slice(&ssa_convert(&lifted))
}

/// `true` when `reason` is a budget-exhaustion truncation (a larger
/// budget could resolve it). Structural reasons — `call at …`,
/// `memory access at …`, `unsupported '…' touches slice`,
/// `no flag-defining instruction …` — return `false`: a bigger budget
/// cannot change them. Matched against the exact phrases
/// `r2smt-slicer` emits (`slice.rs`: `"instruction limit reached"`,
/// `"block budget {N} exhausted"`).
fn is_budget_truncation(reason: &str) -> bool {
    reason.contains("instruction limit reached") || reason.contains("block budget")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::slice::SliceStatus;
    use r2smt_slicer::{SliceLimits, collect_branches};

    use super::{is_budget_truncation, prepare_ssa};

    #[test]
    fn budget_reason_classifier_distinguishes_structural_from_budget() {
        assert!(is_budget_truncation("instruction limit reached"));
        assert!(is_budget_truncation("block budget 1 exhausted"));
        assert!(!is_budget_truncation("call at 0x401050"));
        assert!(!is_budget_truncation("memory access at 0x401060"));
        assert!(!is_budget_truncation(
            "unsupported 'vpxor' at 0x401070 touches slice"
        ));
        assert!(!is_budget_truncation(
            "no flag-defining instruction found in slice (x; pending: y)"
        ));
    }

    fn op(raw: &str, kind: OperandKind) -> Operand {
        Operand {
            raw: raw.into(),
            kind,
        }
    }

    fn insn(addr: u64, size: u8, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            address: Address(addr),
            size,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands,
            esil: None,
            pcode: None,
            is_thumb: false,
        }
    }

    /// Build a single-block program whose data-flow chain length is
    /// `chain` redundant `mov`s before the `cmp` / `jne`, so the
    /// default 32-instruction budget truncates but a larger one does
    /// not.
    fn long_chain_program(chain: usize) -> Program {
        let mut instrs = Vec::new();
        let mut addr = 0x40_1000u64;
        // eax := ecx
        instrs.push(insn(
            addr,
            2,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("ecx", OperandKind::Register),
            ],
        ));
        addr += 2;
        // `chain` redundant self-moves eax := eax (keep the chain long)
        for _ in 0..chain {
            instrs.push(insn(
                addr,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ));
            addr += 2;
        }
        instrs.push(insn(
            addr,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ));
        addr += 3;
        instrs.push(insn(
            addr,
            6,
            "jne",
            vec![op("0x401900", OperandKind::Immediate)],
        ));
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.t".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: instrs,
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    #[test]
    fn budget_truncation_triggers_reslice_and_completes() {
        // 60 redundant instructions: the default budget (32) truncates,
        // the escalated retry (256) fits the whole chain → Complete.
        let program = long_chain_program(60);
        let candidates = collect_branches(&program);
        let cand = candidates.first().unwrap();
        let ssa = prepare_ssa(
            &program.functions[0],
            cand,
            &SliceLimits::default(),
            program.arch,
        );
        assert_eq!(
            ssa.status,
            SliceStatus::Complete,
            "escalated re-slice must resolve the long chain"
        );
    }

    #[test]
    fn complete_slice_is_returned_without_reslice() {
        // Short chain fits the default budget; no retry path taken.
        let program = long_chain_program(2);
        let candidates = collect_branches(&program);
        let cand = candidates.first().unwrap();
        let ssa = prepare_ssa(
            &program.functions[0],
            cand,
            &SliceLimits::default(),
            program.arch,
        );
        assert_eq!(ssa.status, SliceStatus::Complete);
    }

    #[test]
    fn structural_truncation_call_is_not_resliced() {
        // A `call` in the data-flow chain truncates structurally;
        // re-slicing with a bigger budget cannot help, so the result
        // stays Truncated with the call reason (no panic, bounded).
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.t".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        insn(
                            0x40_1000,
                            5,
                            "call",
                            vec![op("0x402000", OperandKind::Immediate)],
                        ),
                        insn(
                            0x40_1005,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("0", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1008,
                            6,
                            "jne",
                            vec![op("0x401900", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        };
        let candidates = collect_branches(&program);
        let cand = candidates.first().unwrap();
        let ssa = prepare_ssa(
            &program.functions[0],
            cand,
            &SliceLimits::default(),
            program.arch,
        );
        match ssa.status {
            SliceStatus::Truncated { reason } => {
                assert!(
                    reason.contains("call"),
                    "structural truncation preserved, got: {reason}"
                );
            }
            SliceStatus::Complete => panic!("call must not be sliced through"),
        }
    }
}
