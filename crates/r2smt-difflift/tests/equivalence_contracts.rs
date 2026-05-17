//! P22 differential multi-lifter contract.
//!
//! The harness must catch a genuinely unsound lowering (teeth test)
//! while never fabricating a disagreement between two correct ones
//! (soundness-direction). The solve is delegated to the real Z3
//! backend here — exactly the wiring the CLI uses.

use r2smt_common::{Arch, SmtResult};
use r2smt_difflift::{DiffVerdict, build_equivalence_query, classify_equivalence, lower_all};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_smt::{SolveOptions, solve_branch};

/// Deterministic, deliberately generous solver budget — mirrors the
/// `r2smt-smt` solver test convention so self-induced load never
/// flips a verdict to `Timeout`.
const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;

fn op(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn insn(addr: u64, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: r2smt_common::Address::new(addr),
        size: 3,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

fn diff(a: &[IrStmt], b: &[IrStmt]) -> DiffVerdict {
    match build_equivalence_query(a, b, Arch::X86_64) {
        None => DiffVerdict::Inconclusive,
        Some(query) => classify_equivalence(solve_branch(
            &query,
            SolveOptions {
                timeout_ms: TEST_SOLVE_TIMEOUT_MS,
            },
        )),
    }
}

/// Per-mnemonic lowering of the *fixed* (count-masked) `shl eax, 32`:
/// `32 & 31 == 0`, so `eax` is unchanged and `ZF == (eax == 0)`.
fn correct_shl_eax_32() -> Vec<IrStmt> {
    r2smt_slicer::lift_per_mnemonic(
        &insn(
            0x1000,
            "shl",
            vec![
                op("eax", OperandKind::Register),
                op("32", OperandKind::Immediate),
            ],
        ),
        Arch::X86_64,
    )
}

/// The lowering you get if commit `2db55b8` is reverted: the x86
/// shift count is **not** masked, so the IR computes `shl(eax, 32)`,
/// which is `0` for every input under SMT-LIB bit-vector semantics —
/// `ZF` becomes a constant `1` instead of `eax == 0`.
fn reverted_unmasked_shl_eax_32() -> Vec<IrStmt> {
    let eax = Expr::extract(Expr::var("rax", 64), 31, 0);
    let shifted = Expr::shl(eax, Expr::konst(32, 32));
    let t = Var::new("t_diff_0", 32);
    vec![
        IrStmt::Assign {
            dst: t.clone(),
            src: shifted,
        },
        IrStmt::Assign {
            dst: Var::new("rax", 64),
            src: Expr::ZeroExtend {
                src: Box::new(Expr::Var(t.clone())),
                to_bits: 64,
            },
        },
        IrStmt::Assign {
            dst: Var::new("ZF", 1),
            src: Expr::eq(Expr::Var(t.clone()), Expr::konst(0, 32)),
        },
        IrStmt::Assign {
            dst: Var::new("SF", 1),
            src: Expr::slt(Expr::Var(t), Expr::konst(0, 32)),
        },
    ]
}

#[test]
fn test_reverted_shift_mask_lowering_disagrees_with_fixed_one() {
    // Teeth: if the 2db55b8 mask fix is reverted, the harness must
    // flag the lowering as unsound.
    assert_eq!(
        diff(&correct_shl_eax_32(), &reverted_unmasked_shl_eax_32()),
        DiffVerdict::Disagree,
    );
}

#[test]
fn test_equivalent_flag_lowerings_agree() {
    // `test eax, eax` and `cmp eax, 0` set ZF/SF/CF identically from
    // `eax` and write no register — provably equivalent.
    let test_eax = r2smt_slicer::lift_per_mnemonic(
        &insn(
            0x1000,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let cmp_eax_0 = r2smt_slicer::lift_per_mnemonic(
        &insn(
            0x1000,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(diff(&test_eax, &cmp_eax_0), DiffVerdict::Agree);
}

#[test]
fn test_memory_touching_lowering_is_not_comparable() {
    // A lowering that loads memory cannot be soundly compared (no
    // memory model) — the query must decline rather than risk a
    // fabricated verdict.
    let with_load = vec![IrStmt::LoadMem {
        dst: Var::new("rax", 64),
        address: Expr::var("rbx", 64),
        bits: 64,
    }];
    let plain = vec![IrStmt::Assign {
        dst: Var::new("rax", 64),
        src: Expr::konst(0, 64),
    }];
    assert!(build_equivalence_query(&with_load, &plain, Arch::X86_64).is_none());
}

#[test]
fn test_no_jointly_defined_output_yields_no_query() {
    // Disjoint def names → nothing comparable → `None` (caller maps
    // to `Inconclusive`, never `Agree`).
    let a = vec![IrStmt::Assign {
        dst: Var::new("rax", 64),
        src: Expr::konst(1, 64),
    }];
    let b = vec![IrStmt::Assign {
        dst: Var::new("rbx", 64),
        src: Expr::konst(1, 64),
    }];
    assert!(build_equivalence_query(&a, &b, Arch::X86_64).is_none());
}

#[test]
fn test_classify_alwaysfalse_is_agree() {
    assert_eq!(
        classify_equivalence(SmtResult::AlwaysFalse),
        DiffVerdict::Agree,
    );
}

#[test]
fn test_classify_bothpossible_is_disagree() {
    assert_eq!(
        classify_equivalence(SmtResult::BothPossible),
        DiffVerdict::Disagree,
    );
}

#[test]
fn test_classify_timeout_fails_closed_to_inconclusive() {
    assert_eq!(
        classify_equivalence(SmtResult::Timeout),
        DiffVerdict::Inconclusive,
    );
}

#[test]
fn test_lower_all_produces_per_mnemonic_and_esil_bodies() {
    let mut i = insn(
        0x1000,
        "mov",
        vec![
            op("eax", OperandKind::Register),
            op("1", OperandKind::Immediate),
        ],
    );
    i.esil = Some("1,eax,=".to_string());
    let lowerings = lower_all(&i, Arch::X86_64);
    assert!(lowerings.esil.is_some() && !lowerings.mnemonic.is_empty());
}

#[test]
fn test_agreement_rate_ignores_inconclusive() {
    let mut stats = r2smt_difflift::AgreementStats::default();
    stats.record(DiffVerdict::Agree);
    stats.record(DiffVerdict::Agree);
    stats.record(DiffVerdict::Disagree);
    stats.record(DiffVerdict::Inconclusive);
    // 2 agree / (2 agree + 1 disagree) — inconclusive excluded.
    assert_eq!(stats.agreement_rate(), Some(2.0 / 3.0));
}
