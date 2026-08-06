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
    diff_arch(a, b, Arch::X86_64)
}

fn diff_arch(a: &[IrStmt], b: &[IrStmt], arch: Arch) -> DiffVerdict {
    match build_equivalence_query(a, b, arch) {
        None => DiffVerdict::Inconclusive,
        Some(query) => classify_equivalence(solve_branch(
            &query,
            SolveOptions {
                timeout_ms: TEST_SOLVE_TIMEOUT_MS,
                ..SolveOptions::default()
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
fn test_scalar_move_against_the_esil_xmm_model_is_not_a_disagreement() {
    // The per-mnemonic lowering of `movsd xmm0, xmm1` canonicalises the
    // operands to the 512-bit `zmm0` parent; ESIL models `xmm0` as a
    // register of pointer width. The two describe different machine
    // state, not different semantics, so the harness must report them
    // as not-comparable rather than manufacture a disagreement.
    let mut i = insn(
        0x1000,
        "movsd",
        vec![
            op("xmm0", OperandKind::Register),
            op("xmm1", OperandKind::Register),
        ],
    );
    i.esil = Some("xmm1,xmm0,=".to_string());
    let lowerings = lower_all(&i, Arch::X86_64);
    let esil = lowerings.esil.unwrap_or_default();
    assert!(
        !esil.is_empty(),
        "the ESIL lowering is the point of the test"
    );
    assert_ne!(diff(&lowerings.mnemonic, &esil), DiffVerdict::Disagree);
}

/// `dst := x0 + delta`, the one shape the alias contracts need.
fn define_from_x0(dst: &str, dst_bits: u16, delta: u128) -> Vec<IrStmt> {
    vec![IrStmt::Assign {
        dst: Var::new(dst, dst_bits),
        src: Expr::add(Expr::var("x0", 64), Expr::konst(delta, 64)),
    }]
}

#[test]
fn test_alias_named_outputs_are_compared_through_the_parent() {
    // `fp` and `x29` are two spellings of one register. Matching outputs
    // by base name never paired them, which is how 181 further instances
    // of the `sp` bug went unreported while `tie_inputs` had been
    // canonicalising the input side all along.
    let a = define_from_x0("fp", 64, 8);
    let b = define_from_x0("x29", 64, 16);
    assert_eq!(diff_arch(&a, &b, Arch::Aarch64), DiffVerdict::Disagree);
}

#[test]
fn test_alias_named_outputs_agree_when_the_values_match() {
    let a = define_from_x0("fp", 64, 8);
    let b = define_from_x0("x29", 64, 8);
    assert_eq!(diff_arch(&a, &b, Arch::Aarch64), DiffVerdict::Agree);
}

#[test]
fn test_mismatched_output_widths_are_compared_not_skipped() {
    // The measured `sp`-at-16-bits shape, in a *destination*: one side
    // models the stack pointer at its architectural 64 bits, the other at
    // x86's 16. The old `var_a.bits != var_b.bits` guard skipped exactly
    // this, so a width defect landing on the destination suppressed its
    // own detection.
    let wide = vec![IrStmt::Assign {
        dst: Var::new("sp", 64),
        src: Expr::sub(Expr::var("sp", 64), Expr::konst(16, 64)),
    }];
    let narrow = vec![IrStmt::Assign {
        dst: Var::new("sp", 16),
        src: Expr::sub(Expr::var("sp", 16), Expr::konst(16, 16)),
    }];
    assert_eq!(
        diff_arch(&wide, &narrow, Arch::Aarch64),
        DiffVerdict::Disagree
    );
}

#[test]
fn test_partial_register_write_agrees_with_a_full_parent_write() {
    // Anti-fabrication counterpart: reconstructing a narrow write against
    // the shared parent input must reproduce what the explicit
    // full-parent write says, or every sub-register lowering pair becomes
    // a disagreement.
    let partial = vec![IrStmt::Assign {
        dst: Var::new("ax", 16),
        src: Expr::konst(5, 16),
    }];
    let full = vec![IrStmt::Assign {
        dst: Var::new("rax", 64),
        src: Expr::concat(
            Expr::extract(Expr::var("rax", 64), 63, 16),
            Expr::konst(5, 16),
        ),
    }];
    assert_eq!(diff(&partial, &full), DiffVerdict::Agree);
}

#[test]
fn test_vector_width_mismatch_is_not_comparable_rather_than_a_disagreement() {
    // The guard that keeps the alias canonicalisation from reporting
    // every SIMD instruction: r2's ESIL carries no vector model and names
    // `xmm0` at the pointer width, which contradicts the 128-bit
    // architectural view. That is modelling depth, not a lifter defect.
    // The same mismatch on a general register is the `sp` shape above and
    // must still be reported — which is why the guard tests the parent's
    // register file and not merely the widths.
    let narrow = vec![IrStmt::Assign {
        dst: Var::new("xmm0", 64),
        src: Expr::konst(1, 64),
    }];
    let wide = vec![IrStmt::Assign {
        dst: Var::new("xmm0", 128),
        src: Expr::konst(2, 128),
    }];
    assert_eq!(diff(&narrow, &wide), DiffVerdict::Inconclusive);
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
