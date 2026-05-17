#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
