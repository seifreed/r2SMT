//! P26 memory-model contract.
//!
//! These tests pin the four exit criteria of the sound memory model:
//!
//! 1. Precise stack roundtrip → `High` confidence — a store at a
//!    statically-resolvable address followed by a load at the same
//!    address returns the stored value, with no `Expr::Unknown`
//!    polluting the slice.
//! 2. **Gating teeth** — an unresolved-address load widens, never
//!    fabricates: with no prior store, a load reads a fresh free
//!    value, and a comparison against a concrete value is
//!    `BothPossible`, never `AlwaysX`.
//! 3. An unresolved store *havocs* possibly-aliasing slots: a prior
//!    known store followed by a store at a symbolic address means
//!    the original value is no longer guaranteed when read back —
//!    the solver may pick aliasing or not, so the verdict widens
//!    to `BothPossible`.
//! 4. The `AArch64` `ldr` / `str` lifter emits the expected
//!    `IrStmt::LoadMem` / `StoreMem` shape with the right address
//!    expression and width.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_common::{Address, Arch};
use r2smt_core::{Confidence, FindingKind, classify_finding};
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus, lift_per_mnemonic};
use r2smt_smt::solve_branch;
use r2smt_ssa::SsaLiftedSlice;

const TEST_SOLVE_TIMEOUT_MS: u32 = 10_000;

fn solve_opts() -> SolveOptions {
    SolveOptions {
        timeout_ms: TEST_SOLVE_TIMEOUT_MS,
        ..SolveOptions::default()
    }
}

fn synthetic_branch() -> BranchCandidate {
    let z = Address::new(0x1000);
    BranchCandidate {
        address: z,
        function: z,
        block: z,
        kind: BranchKind::Jcc,
        mnemonic: "memtest".to_string(),
        condition: BranchCondition::NotEqual,
        formula: "memtest".to_string(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

/// Build a synthetic SSA slice over `statements` that asserts
/// `condition` as a 1-bit branch predicate. Used by the encoder
/// contracts to bypass the lifter/slicer and validate the memory
/// model in isolation.
fn synthetic_slice(
    statements: Vec<IrStmt>,
    condition: Expr,
    inputs: Vec<Var>,
    defs: Vec<Var>,
) -> SsaLiftedSlice {
    SsaLiftedSlice {
        branch: synthetic_branch(),
        statements,
        condition,
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        inputs,
        defs,
        arch: Arch::Aarch64,
    }
}

#[test]
fn test_precise_stack_roundtrip_yields_high_confidence() {
    // Build the IR equivalent of:
    //     str x29 base-relative — value 5 at [sp - 8]
    //     ldr  — value back from [sp - 8]
    //     if (loaded == 5) ...
    // The loaded value MUST equal 5 (no aliasing, no other writes).
    // Verdict: AlwaysTrue; finding confidence: High (no Unknowns).
    let sp = Expr::Var(Var::new("sp", 64));
    let offset = Expr::konst(0xFFFF_FFFF_FFFF_FFF8, 64); // -8 as u64
    let addr = Expr::add(sp, offset);
    let loaded = Var::new("loaded#0", 64);

    let slice = synthetic_slice(
        vec![
            IrStmt::StoreMem {
                address: addr.clone(),
                value: Expr::konst(5, 64),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: loaded.clone(),
                address: addr,
                bits: 64,
            },
        ],
        Expr::eq(Expr::Var(loaded.clone()), Expr::konst(5, 64)),
        vec![Var::new("sp", 64)],
        vec![loaded],
    );

    let verdict = solve_branch(&slice, solve_opts());
    assert_eq!(verdict, SmtResult::AlwaysTrue);

    let finding = classify_finding(&slice, verdict);
    assert_eq!(
        finding.confidence,
        Confidence::High,
        "precise stack roundtrip must land at High confidence (got {:?})",
        finding.confidence,
    );
}

#[test]
fn test_unknown_address_load_widens_never_fabricates() {
    // Gating teeth: with NO prior store, a load reads a fresh free
    // byte sequence. A comparison against a concrete value
    // (`loaded == 0x1234`) must be `BothPossible` — the load could
    // be anything. An `AlwaysX` verdict here would be a fabricated
    // memory model.
    let addr = Expr::Var(Var::new("addr", 64));
    let loaded = Var::new("loaded#0", 64);
    let slice = synthetic_slice(
        vec![IrStmt::LoadMem {
            dst: loaded.clone(),
            address: addr,
            bits: 64,
        }],
        Expr::eq(Expr::Var(loaded.clone()), Expr::konst(0x1234, 64)),
        vec![Var::new("addr", 64)],
        vec![loaded],
    );
    assert_eq!(
        solve_branch(&slice, solve_opts()),
        SmtResult::BothPossible,
        "unresolved load must widen to BothPossible, never fabricate AlwaysX",
    );
}

#[test]
fn test_unknown_store_havocs_possibly_aliasing_slot() {
    // Sequence:
    //     [sp-8] := 5
    //     [x0]   := 7        ; x0 unconstrained ⇒ could alias [sp-8]
    //     loaded := load [sp-8]
    //     if (loaded == 5)
    //
    // The solver may pick `x0 == sp-8` (load returns 7) or
    // `x0 != sp-8` (load returns 5). Sound widen: BothPossible.
    let sp = Expr::Var(Var::new("sp", 64));
    let known_addr = Expr::add(sp, Expr::konst(0xFFFF_FFFF_FFFF_FFF8, 64));
    let unknown_addr = Expr::Var(Var::new("x0", 64));
    let loaded = Var::new("loaded#0", 64);
    let slice = synthetic_slice(
        vec![
            IrStmt::StoreMem {
                address: known_addr.clone(),
                value: Expr::konst(5, 64),
                bits: 64,
            },
            IrStmt::StoreMem {
                address: unknown_addr,
                value: Expr::konst(7, 64),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: loaded.clone(),
                address: known_addr,
                bits: 64,
            },
        ],
        Expr::eq(Expr::Var(loaded.clone()), Expr::konst(5, 64)),
        vec![Var::new("sp", 64), Var::new("x0", 64)],
        vec![loaded],
    );
    assert_eq!(
        solve_branch(&slice, solve_opts()),
        SmtResult::BothPossible,
        "an unresolved store must havoc possibly-aliasing slots",
    );
}

#[test]
fn test_unmodelled_loaded_value_yields_real_branch_not_dead() {
    // Slice-level corollary of the gating teeth: a finding whose
    // verdict is BothPossible from an unresolved load must be
    // classified as `RealBranch` (a genuine choice), never an
    // actionable `OpaquePredicate` / `DeadBranch`. This is what
    // saves a consumer from acting on a fabricated dead branch.
    let addr = Expr::Var(Var::new("addr", 64));
    let loaded = Var::new("loaded#0", 64);
    let slice = synthetic_slice(
        vec![IrStmt::LoadMem {
            dst: loaded.clone(),
            address: addr,
            bits: 64,
        }],
        Expr::eq(Expr::Var(loaded.clone()), Expr::konst(0x4242, 64)),
        vec![Var::new("addr", 64)],
        vec![loaded],
    );
    let verdict = solve_branch(&slice, solve_opts());
    let finding = classify_finding(&slice, verdict);
    assert_eq!(finding.kind, FindingKind::RealBranch);
}

// --- AArch64 lifter goldens for `ldr` / `str` ------------------------

fn op(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn insn(addr: u64, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: Address::new(addr),
        size: 4,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

#[test]
fn test_aarch64_ldr_emits_loadmem_with_base_plus_offset_address() {
    // `ldr x0, [x1, #8]` → load 8 bytes at `x1 + 8` into a temp,
    // then write the temp into the parent X0 register. The first
    // statement must be a `LoadMem` carrying the offset address.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("x0", OperandKind::Register),
                op("[x1, 8]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    let first_load = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { address, bits, .. } => Some((address.clone(), *bits)),
            _ => None,
        })
        .expect("ldr must produce a LoadMem");
    assert_eq!(first_load.1, 64, "ldr X destination must load 64 bits");
    // Address must reference `x1`. The exact `Add` tree shape is an
    // implementation detail; assert by structural sniff.
    let rendered = format!("{first_load:?}");
    assert!(
        rendered.contains("\"x1\""),
        "expected x1 in lifted address, got: {rendered}",
    );
}

#[test]
fn test_aarch64_str_emits_storemem_with_value_and_address() {
    // `str x2, [x3, #16]` → write x2 to memory at `x3 + 16`.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "str",
            vec![
                op("x2", OperandKind::Register),
                op("[x3, 16]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    let store = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => Some((address.clone(), value.clone(), *bits)),
            _ => None,
        })
        .expect("str must produce a StoreMem");
    assert_eq!(store.2, 64, "str X must store 64 bits");
    let addr_dbg = format!("{:?}", store.0);
    let value_dbg = format!("{:?}", store.1);
    assert!(
        addr_dbg.contains("\"x3\""),
        "expected x3 in addr: {addr_dbg}"
    );
    assert!(
        value_dbg.contains("\"x2\""),
        "expected x2 in stored value: {value_dbg}",
    );
}

#[test]
fn test_aarch64_ldp_emits_two_loadmem_at_consecutive_addresses() {
    // `ldp x0, x1, [x2, #16]` → load x0 from `x2 + 16`, x1 from
    // `x2 + 16 + 8`. Two 64-bit LoadMem, second address one register
    // width above the first.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldp",
            vec![
                op("x0", OperandKind::Register),
                op("x1", OperandKind::Register),
                op("[x2, 16]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    let loads: Vec<(Expr, u8)> = stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::LoadMem { address, bits, .. } => Some((address.clone(), *bits)),
            _ => None,
        })
        .collect();
    assert_eq!(
        loads.len(),
        2,
        "ldp must produce two LoadMem, got {loads:?}"
    );
    assert_eq!(loads[0].1, 64);
    assert_eq!(loads[1].1, 64);
    assert_ne!(
        format!("{:?}", loads[0].0),
        format!("{:?}", loads[1].0),
        "the two loaded addresses must differ by the element stride",
    );
}

#[test]
fn test_aarch64_stp_emits_two_storemem_for_the_pair() {
    // `stp x0, x1, [sp, #16]` → store x0 at `sp + 16`, x1 at
    // `sp + 16 + 8`.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "stp",
            vec![
                op("x0", OperandKind::Register),
                op("x1", OperandKind::Register),
                op("[sp, 16]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    let stores: Vec<(Expr, Expr, u8)> = stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => Some((address.clone(), value.clone(), *bits)),
            _ => None,
        })
        .collect();
    assert_eq!(
        stores.len(),
        2,
        "stp must produce two StoreMem, got {stores:?}"
    );
    assert!(format!("{:?}", stores[0].1).contains("\"x0\""));
    assert!(format!("{:?}", stores[1].1).contains("\"x1\""));
}

// --- AArch32 lifter goldens for `ldr` / `str` (P28) -------------------

#[test]
fn test_aarch32_ldr_emits_loadmem_with_base_plus_offset_address() {
    // `ldr r0, [r1, #8]` → load 32 bits at `r1 + 8`.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("r0", OperandKind::Register),
                op("[r1, 8]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    let load = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { address, bits, .. } => Some((address.clone(), *bits)),
            _ => None,
        })
        .expect("aarch32 ldr must produce a LoadMem");
    assert_eq!(load.1, 32, "aarch32 ldr must load 32 bits");
    assert!(
        format!("{:?}", load.0).contains("\"r1\""),
        "expected r1 in lifted address, got: {:?}",
        load.0
    );
}

#[test]
fn test_aarch32_str_emits_storemem_with_value_and_address() {
    // `str r2, [r3, #16]` → store r2 at `r3 + 16`.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "str",
            vec![
                op("r2", OperandKind::Register),
                op("[r3, 16]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    let store = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => Some((address.clone(), value.clone(), *bits)),
            _ => None,
        })
        .expect("aarch32 str must produce a StoreMem");
    assert_eq!(store.2, 32, "aarch32 str must store 32 bits");
    assert!(format!("{:?}", store.0).contains("\"r3\""));
    assert!(format!("{:?}", store.1).contains("\"r2\""));
}

#[test]
fn test_aarch32_ldr_register_offset_declines_soundly() {
    // `ldr r0, [r1, r2]` (register offset) is not modelled — the
    // lifter must decline to `Unsupported`, never emit a LoadMem.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("r0", OperandKind::Register),
                op("[r1, r2]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    assert!(
        !stmts.iter().any(|s| matches!(s, IrStmt::LoadMem { .. })),
        "register-offset ldr must not fabricate a LoadMem: {stmts:?}"
    );
}

#[test]
fn test_aarch32_ldrb_loads_one_byte() {
    // `ldrb r0, [r1]` → 8-bit LoadMem, zero-extended into r0.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldrb",
            vec![
                op("r0", OperandKind::Register),
                op("[r1]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    let bits = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .expect("ldrb must produce a LoadMem");
    assert_eq!(bits, 8, "ldrb must load a single byte");
}

#[test]
fn test_aarch32_strh_stores_two_bytes() {
    // `strh r2, [r3, #4]` → 16-bit StoreMem of the low halfword.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "strh",
            vec![
                op("r2", OperandKind::Register),
                op("[r3, 4]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    let bits = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::StoreMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .expect("strh must produce a StoreMem");
    assert_eq!(bits, 16, "strh must store a halfword");
}
