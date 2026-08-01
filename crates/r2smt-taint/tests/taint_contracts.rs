//! Contract tests for the E2 may-taint lattice.
//!
//! The synthetic slices below stand in for the two motivating use
//! cases: a source flowing register-to-register into a sink argument
//! (an `argv` byte into a `system` argument) and a value routed through
//! memory (a key spilled to the stack and reloaded). Each test pins one
//! invariant.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus};
use r2smt_ssa::SsaLiftedSlice;
use r2smt_taint::{SourceId, TaintSeeds, TaintSet, propagate};

const SRC0: SourceId = SourceId(0);
const SRC1: SourceId = SourceId(1);

fn branch() -> BranchCandidate {
    let z = Address::new(0x1000);
    BranchCandidate {
        address: z,
        function: z,
        block: z,
        kind: BranchKind::Jcc,
        mnemonic: "tainttest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "tainttest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

fn slice(statements: Vec<IrStmt>, status: SliceStatus) -> SsaLiftedSlice {
    SsaLiftedSlice {
        branch: branch(),
        statements,
        condition: Expr::konst(0, 1),
        status,
        treat_truncation_as_inputs: false,
        inputs: Vec::new(),
        defs: Vec::new(),
        arch: Arch::X86_64,
    }
}

fn seed(pairs: &[(&str, SourceId)]) -> TaintSeeds {
    pairs
        .iter()
        .map(|(name, src)| ((*name).to_string(), TaintSet::source(*src)))
        .collect()
}

fn assign(dst: &str, src: Expr) -> IrStmt {
    IrStmt::Assign {
        dst: Var::new(dst, 64),
        src,
    }
}

fn var(name: &str) -> Expr {
    Expr::Var(Var::new(name, 64))
}

#[test]
fn test_register_flow_taints_the_sink() {
    let ssa = slice(vec![assign("rsi", var("rdi"))], SliceStatus::Complete);
    let out = propagate(&ssa, &seed(&[("rdi", SRC0)])).unwrap();
    assert!(out.reaches("rsi", SRC0));
}

#[test]
fn test_constant_assignment_stays_untainted() {
    let ssa = slice(
        vec![assign("rsi", Expr::konst(0x2a, 64))],
        SliceStatus::Complete,
    );
    let out = propagate(&ssa, &seed(&[("rdi", SRC0)])).unwrap();
    assert!(out.taint_of("rsi").is_untainted());
}

#[test]
fn test_multi_hop_flow_reaches_the_sink() {
    // rdi -> rsi -> rdx: an argv byte routed to a call argument.
    let ssa = slice(
        vec![
            assign("rsi", var("rdi")),
            assign("rdx", Expr::add(var("rsi"), Expr::konst(1, 64))),
        ],
        SliceStatus::Complete,
    );
    let out = propagate(&ssa, &seed(&[("rdi", SRC0)])).unwrap();
    assert!(out.reaches("rdx", SRC0));
}

#[test]
fn test_independent_sources_do_not_cross_contaminate() {
    let ssa = slice(vec![assign("rsi", var("rdi"))], SliceStatus::Complete);
    let out = propagate(&ssa, &seed(&[("rdi", SRC0), ("rbx", SRC1)])).unwrap();
    assert!(out.reaches("rsi", SRC0) && !out.reaches("rsi", SRC1));
}

#[test]
fn test_store_then_load_propagates_taint_through_memory() {
    let ssa = slice(
        vec![
            IrStmt::StoreMem {
                address: var("rsp"),
                value: var("rdi"),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: Var::new("rax", 64),
                address: var("rsp"),
                bits: 64,
            },
        ],
        SliceStatus::Complete,
    );
    let out = propagate(&ssa, &seed(&[("rdi", SRC0)])).unwrap();
    assert!(out.reaches("rax", SRC0));
}

#[test]
fn test_unknown_rhs_sets_opaque_without_fabricating_taint() {
    let ssa = slice(
        vec![assign("rsi", Expr::Unknown("opaque".to_string()))],
        SliceStatus::Complete,
    );
    let out = propagate(&ssa, &seed(&[("rdi", SRC0)])).unwrap();
    assert!(out.is_opaque() && out.taint_of("rsi").is_untainted());
}

#[test]
fn test_truncated_slice_is_opaque() {
    let ssa = slice(
        vec![assign("rsi", var("rdi"))],
        SliceStatus::Truncated {
            reason: "budget".to_string(),
        },
    );
    let out = propagate(&ssa, &seed(&[("rdi", SRC0)])).unwrap();
    assert!(out.is_opaque());
}
