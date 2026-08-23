//! Generated equivalence contracts for the pre-solver simplifier.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use r2smt_common::{Address, Arch, SmtResult, SolveOptions};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus};
use r2smt_smt::solve_branch;
use r2smt_ssa::{SsaLiftedSlice, optimize_slice};

fn slice(constant: u128) -> SsaLiftedSlice {
    let address = Address::new(0x1000);
    SsaLiftedSlice {
        branch: BranchCandidate {
            address,
            function: address,
            block: address,
            kind: BranchKind::Jcc,
            mnemonic: "jne".into(),
            condition: BranchCondition::NotEqual,
            formula: String::new(),
            taken_target: None,
            fallthrough_target: None,
            compare_register: None,
            bit_index: None,
            upstream_resolved: None,
            operand_raws: Vec::new(),
            is_thumb: false,
        },
        statements: vec![
            IrStmt::Assign {
                dst: Var::new("copy#0", 64),
                src: Expr::var("input", 64),
            },
            IrStmt::Assign {
                dst: Var::new("value#0", 64),
                src: Expr::add(Expr::var("copy#0", 64), Expr::konst(constant, 64)),
            },
        ],
        condition: Expr::eq(
            Expr::sub(Expr::var("value#0", 64), Expr::konst(constant, 64)),
            Expr::var("input", 64),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        inputs: vec![Var::new("input", 64)],
        defs: vec![Var::new("copy#0", 64), Var::new("value#0", 64)],
        arch: Arch::X86_64,
    }
}

#[test]
fn generated_simplifications_preserve_solver_verdicts() {
    let options = SolveOptions {
        timeout_ms: 10_000,
        ..SolveOptions::default()
    };
    let mut state = 0x5eed_u128;
    for _ in 0..128 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let original = slice(state);
        let optimized = optimize_slice(&original);
        assert_eq!(solve_branch(&original, options), SmtResult::AlwaysTrue);
        assert_eq!(solve_branch(&optimized, options), SmtResult::AlwaysTrue);
        assert_eq!(optimize_slice(&optimized), optimized);
    }
}
