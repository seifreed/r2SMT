#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
use r2smt_ssa::ssa_convert;

use super::*;

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

fn one_block(insns: Vec<Instruction>) -> Program {
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

fn ssa_first(program: &Program) -> SsaLiftedSlice {
    let candidates = collect_branches(program);
    let cand = candidates.first().expect("at least one branch");
    let slice = slice_branch(
        cand,
        &program.functions[0],
        &SliceLimits::default(),
        program.arch,
    );
    let lifted = lift_slice(&slice, program.arch);
    ssa_convert(&lifted)
}

#[test]
fn always_true_with_inputs_classifies_as_opaque_predicate() {
    // test eax, eax ; jne — but we'll synthetically force the
    // verdict to AlwaysTrue to exercise the classifier alone.
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::AlwaysTrue);
    assert_eq!(finding.kind, FindingKind::OpaquePredicate);
    assert!(finding.is_actionable());
}

#[test]
fn always_false_with_inputs_classifies_as_dead_branch() {
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
    assert_eq!(finding.kind, FindingKind::DeadBranch);
    assert!(finding.is_actionable());
}

#[test]
fn constant_condition_has_no_inputs() {
    // mov eax, 1 ; cmp eax, 1 ; jne — no register inputs.
    let program = one_block(vec![
        insn(
            0x40_1000,
            5,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1005,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1008,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
    assert_eq!(finding.kind, FindingKind::ConstantCondition);
    assert!(finding.evidence.inputs.is_empty());
}

#[test]
fn both_possible_is_real_branch_not_actionable() {
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::BothPossible);
    assert_eq!(finding.kind, FindingKind::RealBranch);
    assert!(!finding.is_actionable());
}

#[test]
fn unknown_verdict_becomes_suspicious() {
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::Timeout);
    assert_eq!(finding.kind, FindingKind::SuspiciousButUnknown);
    assert_eq!(finding.confidence, Confidence::Unknown);
}

#[test]
fn complete_slice_with_unknowns_yields_medium_confidence() {
    // cmp emits OF/PF as Unknown — at least 2 Unknown exprs.
    let program = one_block(vec![
        insn(
            0x40_1000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0x10", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1003,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::BothPossible);
    assert!(finding.evidence.unknown_count >= 1);
    assert_eq!(finding.confidence, Confidence::Medium);
}

#[test]
fn truncated_slice_gives_unknown_confidence() {
    let program = one_block(vec![
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
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    // SmtResult is Unsound for truncated, simulate the pipeline.
    let finding = classify_finding(&ssa, SmtResult::Unsound);
    assert_eq!(finding.kind, FindingKind::SuspiciousButUnknown);
    assert_eq!(finding.confidence, Confidence::Unknown);
}

#[test]
fn evidence_inputs_strip_ssa_suffix() {
    // `test eax, eax ; jne` produces a single free input (rax). The
    // SSA layer surfaces it as `rax#0` so substitution stays
    // unambiguous; user-facing evidence must drop the suffix.
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::BothPossible);
    assert_eq!(finding.evidence.inputs, vec!["rax".to_string()]);
    // The pretty form must also drop the suffix.
    assert!(
        !finding.formula_pretty.contains('#'),
        "formula_pretty must not surface SSA suffixes; got: {}",
        finding.formula_pretty
    );
}

#[test]
fn finding_round_trips_with_operands_through_json() {
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let mut finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
    finding.operands = vec!["x0".into(), "x1".into(), "x2".into(), "eq".into()];
    let json = serde_json::to_string(&finding).unwrap();
    let back: Finding = serde_json::from_str(&json).unwrap();
    assert_eq!(back.operands, finding.operands);
}

#[test]
fn finding_round_trips_through_json() {
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
    let json = serde_json::to_string(&finding).unwrap();
    let back: Finding = serde_json::from_str(&json).unwrap();
    assert_eq!(back, finding);
}

#[test]
fn signed_jge_with_unknown_flags_downgrades_to_low_confidence() {
    // cmp eax, 0; jge dest — `jge` depends on `SF == OF`, but our
    // lifter leaves `OF` as Unknown after a cmp. The solver may
    // resolve the formula either way; the verdict should land at
    // `Low` confidence per the Phase E flag-helper refusal.
    let program = one_block(vec![
        insn(
            0x40_1000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1003,
            6,
            "jge",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    let finding = classify_finding(&ssa, SmtResult::BothPossible);
    assert_eq!(finding.confidence, Confidence::Low);
}

#[test]
fn truncated_slice_with_unknowns_on_truncation_yields_medium_confidence() {
    // call f ; cmp eax, eax ; jne junk — same fixture used by the
    // smt crate's regression. The slicer truncates because of the
    // `call`; with the policy the SSA layer surfaces eax as a free
    // input and the solver returns `AlwaysFalse`. The classifier
    // must keep the verdict but downgrade confidence to Medium
    // (not Unknown, not High).
    let program = one_block(vec![
        insn(
            0x40_1000,
            5,
            "call",
            vec![op("0x402000", OperandKind::Immediate)],
        ),
        insn(
            0x40_1005,
            2,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1008,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let candidates = collect_branches(&program);
    let cand = candidates.first().expect("at least one branch");
    let limits = SliceLimits {
        unknowns_on_truncation: true,
        ..SliceLimits::default()
    };
    let slice = slice_branch(cand, &program.functions[0], &limits, program.arch);
    assert!(
        matches!(slice.status, SliceStatus::Truncated { .. }),
        "slice must be truncated to exercise the policy"
    );
    assert!(slice.treat_truncation_as_inputs);
    let lifted = lift_slice(&slice, program.arch);
    let ssa = ssa_convert(&lifted);
    let finding = classify_finding(&ssa, SmtResult::AlwaysFalse);
    assert_eq!(finding.kind, FindingKind::DeadBranch);
    assert_eq!(finding.confidence, Confidence::Medium);
}

#[test]
fn classify_lowered_upstream_emits_constant_condition_when_resolved_to_taken_target() {
    // Block whose only successor is the cjmp's taken target. The
    // CFG analyser already lowered the branch to unconditional;
    // r2SMT must report `ConstantCondition` with `High` confidence
    // and stamp the resolved target in `evidence.upstream_resolved_to`.
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x40_1000),
                instructions: vec![insn(
                    0x40_1000,
                    6,
                    "jne",
                    vec![op("0x401080", OperandKind::Immediate)],
                )],
                successors: vec![Address(0x40_1080)],
            }],
            is_thumb: false,
        }],
    };
    let candidates = collect_branches(&program);
    let cand = candidates.first().expect("at least one branch");
    let finding = classify_lowered_upstream(cand).expect("upstream_resolved must trigger shortcut");
    assert_eq!(finding.kind, FindingKind::ConstantCondition);
    assert_eq!(finding.confidence, Confidence::High);
    assert_eq!(finding.verdict, SmtResult::AlwaysTrue);
    assert_eq!(
        finding.evidence.upstream_resolved_to,
        Some(Address(0x40_1080))
    );
}

#[test]
fn classify_lowered_upstream_returns_none_for_two_way_branch() {
    // Block with both successors recorded — the analyser left the
    // cjmp two-way, so the shortcut must not fire and the SMT
    // pipeline takes over.
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x40_1000),
                instructions: vec![insn(
                    0x40_1000,
                    6,
                    "jne",
                    vec![op("0x401080", OperandKind::Immediate)],
                )],
                successors: vec![Address(0x40_1080), Address(0x40_1006)],
            }],
            is_thumb: false,
        }],
    };
    let candidates = collect_branches(&program);
    let cand = candidates.first().expect("at least one branch");
    assert!(classify_lowered_upstream(cand).is_none());
}

#[test]
fn classify_lowered_upstream_returns_always_false_when_resolved_to_fallthrough() {
    // Block whose single successor is the fallthrough address —
    // the CFG analyser proved the cjmp is never taken.
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x40_1000),
                instructions: vec![insn(
                    0x40_1000,
                    6,
                    "jne",
                    vec![op("0x401080", OperandKind::Immediate)],
                )],
                successors: vec![Address(0x40_1006)],
            }],
            is_thumb: false,
        }],
    };
    let candidates = collect_branches(&program);
    let cand = candidates.first().expect("at least one branch");
    let finding = classify_lowered_upstream(cand).expect("upstream_resolved must trigger shortcut");
    assert_eq!(finding.verdict, SmtResult::AlwaysFalse);
}

#[test]
fn classify_lowered_upstream_returns_none_when_cfg_disagrees_with_targets() {
    // The successor recorded in the CFG (here `0x40_1234`) matches
    // neither `taken_target` (0x40_1080 from the operand) nor the
    // fallthrough (0x40_1006). Rather than guessing AlwaysTrue, the
    // classifier must hand the case off to the SMT pipeline.
    let program = Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(0x40_1000)),
        functions: vec![Function {
            address: Address(0x40_1000),
            name: Some("sym.main".into()),
            blocks: vec![BasicBlock {
                address: Address(0x40_1000),
                instructions: vec![insn(
                    0x40_1000,
                    6,
                    "jne",
                    vec![op("0x401080", OperandKind::Immediate)],
                )],
                successors: vec![Address(0x40_1234)],
            }],
            is_thumb: false,
        }],
    };
    let candidates = collect_branches(&program);
    let cand = candidates.first().expect("at least one branch");
    assert!(classify_lowered_upstream(cand).is_none());
}

#[test]
fn confidence_ordering_high_below_medium_below_low_below_unknown() {
    // Lint: PartialOrd derive gives lexical (declaration) order:
    // High < Medium < Low < Unknown. Verify.
    assert!(Confidence::High < Confidence::Medium);
    assert!(Confidence::Medium < Confidence::Low);
    assert!(Confidence::Low < Confidence::Unknown);
}

#[test]
fn reconcile_folded_prefers_sound_smt_over_cfg_shortcut() {
    let program = one_block(vec![
        insn(
            0x40_1000,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let ssa = ssa_first(&program);
    // Sound SMT verdict (complete slice → not Unknown confidence).
    let smt_sound = classify_finding(&ssa, SmtResult::AlwaysFalse);
    assert_ne!(smt_sound.confidence, Confidence::Unknown);
    // Inconclusive SMT (unsound/truncated → Unknown confidence).
    let smt_unknown = classify_finding(&ssa, SmtResult::Unsound);
    assert_eq!(smt_unknown.confidence, Confidence::Unknown);
    // A stand-in CFG-shortcut finding.
    let cfg = classify_finding(&ssa, SmtResult::BothPossible);

    // Sound SMT wins over the CFG shortcut.
    let r = reconcile_folded(Some(smt_sound.clone()), Some(cfg.clone())).expect("some");
    assert_eq!(r.verdict, SmtResult::AlwaysFalse);
    // Inconclusive SMT falls back to the CFG shortcut.
    let r = reconcile_folded(Some(smt_unknown.clone()), Some(cfg.clone())).expect("some");
    assert_eq!(r.verdict, cfg.verdict);
    // No CFG safety net → the inconclusive SMT is still returned.
    let r = reconcile_folded(Some(smt_unknown), None).expect("some");
    assert_eq!(r.verdict, SmtResult::Unsound);
    // Nothing at all → None.
    assert!(reconcile_folded(None, None).is_none());
}
