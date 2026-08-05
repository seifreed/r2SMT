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
    let loads: Vec<(Expr, u16)> = stmts
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
    let stores: Vec<(Expr, Expr, u16)> = stmts
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
fn test_aarch32_ldr_register_offset_addresses_base_plus_index() {
    // `ldr r0, [r1, r2]` addresses `r1 + r2`. This form used to
    // decline; modelling it is a widening, so the contract now pins
    // that *both* registers reach the address rather than that no
    // LoadMem is emitted.
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
    let address = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { address, .. } => Some(format!("{address:?}")),
            _ => None,
        })
        .expect("register-offset ldr must produce a LoadMem");
    assert!(
        address.contains("\"r1\"") && address.contains("\"r2\""),
        "{address}"
    );
}

#[test]
fn test_aarch32_ldr_shifted_register_offset_scales_only_the_index() {
    // `ldr r0, [r1, r2, lsl 2]` addresses `r1 + (r2 << 2)`. Scaling the
    // base instead would be a plausible-looking wrong address, so the
    // shift must sit under the index register.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("r0", OperandKind::Register),
                op("[r1, r2, lsl 2]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    let address = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { address, .. } => Some(format!("{address:?}")),
            _ => None,
        })
        .expect("shifted register-offset ldr must produce a LoadMem");
    assert!(
        address.contains("Shl(Var(Var { name: \"r2\""),
        "the shift must apply to the index, not the base: {address}"
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

// --- AArch64 sub-word lifter goldens (P30) ---------------------------

#[test]
fn test_aarch64_ldrb_loads_one_byte_zero_extended() {
    // `ldrb w0, [x1]` → 8-bit LoadMem, zero-extended into the register.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldrb",
            vec![
                op("w0", OperandKind::Register),
                op("[x1]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
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
fn test_aarch64_ldrsw_sign_extends_the_loaded_word() {
    // `ldrsw x0, [x1]` → load 32 bits, sign-extend into the 64-bit X0.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldrsw",
            vec![
                op("x0", OperandKind::Register),
                op("[x1]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    let load_bits = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .expect("ldrsw must produce a LoadMem");
    assert_eq!(load_bits, 32, "ldrsw loads a 32-bit word");
    let has_sign_ext = stmts.iter().any(|s| match s {
        IrStmt::Assign { src, .. } => format!("{src:?}").contains("SignExtend"),
        _ => false,
    });
    assert!(has_sign_ext, "ldrsw must sign-extend: {stmts:?}");
}

#[test]
fn test_aarch64_strb_stores_one_byte() {
    // `strb w2, [x3]` → 8-bit StoreMem of the low byte.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "strb",
            vec![
                op("w2", OperandKind::Register),
                op("[x3]", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    let bits = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::StoreMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .expect("strb must produce a StoreMem");
    assert_eq!(bits, 8, "strb must store a single byte");
}

// --- P35 pre/post-index writeback goldens ----------------------------

fn has_writeback_to(stmts: &[IrStmt], reg: &str) -> bool {
    stmts
        .iter()
        .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == reg))
}

#[test]
fn test_aarch64_preindex_ldr_emits_load_and_base_writeback() {
    // `ldr x0, [x1, 8]!` → load at x1+8, then x1 := x1+8.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("x0", OperandKind::Register),
                op("[x1, 8]!", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    assert!(
        stmts.iter().any(|s| matches!(s, IrStmt::LoadMem { .. })),
        "pre-index ldr must load: {stmts:?}"
    );
    assert!(
        has_writeback_to(&stmts, "x1"),
        "pre-index must write back the base x1: {stmts:?}"
    );
}

#[test]
fn test_aarch64_postindex_ldr_emits_load_and_base_writeback() {
    // `ldr x0, [x1], 8` → three operands; load at x1, then x1 := x1+8.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("x0", OperandKind::Register),
                op("[x1]", OperandKind::Memory),
                op("8", OperandKind::Immediate),
            ],
        ),
        Arch::Aarch64,
    );
    assert!(
        stmts.iter().any(|s| matches!(s, IrStmt::LoadMem { .. })),
        "post-index ldr must load: {stmts:?}"
    );
    assert!(
        has_writeback_to(&stmts, "x1"),
        "post-index must write back the base x1: {stmts:?}"
    );
}

#[test]
fn test_aarch64_preindex_str_emits_store_and_base_writeback() {
    // `str x0, [sp, -16]!` — the stp/str stack-frame push idiom.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "str",
            vec![
                op("x0", OperandKind::Register),
                op("[sp, -16]!", OperandKind::Memory),
            ],
        ),
        Arch::Aarch64,
    );
    assert!(
        stmts.iter().any(|s| matches!(s, IrStmt::StoreMem { .. })),
        "pre-index str must store: {stmts:?}"
    );
    assert!(
        has_writeback_to(&stmts, "sp"),
        "pre-index str must write back sp: {stmts:?}"
    );
}

#[test]
fn test_aarch32_postindex_ldr_emits_load_and_base_writeback() {
    // `ldr r0, [r1], 4` → load at r1, then r1 := r1+4.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("r0", OperandKind::Register),
                op("[r1]", OperandKind::Memory),
                op("4", OperandKind::Immediate),
            ],
        ),
        Arch::Arm,
    );
    assert!(
        stmts.iter().any(|s| matches!(s, IrStmt::LoadMem { .. })),
        "aarch32 post-index ldr must load: {stmts:?}"
    );
    assert!(
        has_writeback_to(&stmts, "r1"),
        "aarch32 post-index must write back r1: {stmts:?}"
    );
}

// --- P36 AArch32 register-list multiple goldens ----------------------

fn count_stores(stmts: &[IrStmt]) -> usize {
    stmts
        .iter()
        .filter(|s| matches!(s, IrStmt::StoreMem { .. }))
        .count()
}

fn count_loads(stmts: &[IrStmt]) -> usize {
    stmts
        .iter()
        .filter(|s| matches!(s, IrStmt::LoadMem { .. }))
        .count()
}

#[test]
fn test_aarch32_push_stores_each_register_and_decrements_sp() {
    // `push {r4, r5, lr}` → three StoreMem + sp writeback.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "push",
            vec![op("{r4, r5, lr}", OperandKind::Unknown)],
        ),
        Arch::Arm,
    );
    assert_eq!(
        count_stores(&stmts),
        3,
        "push of 3 regs → 3 stores: {stmts:?}"
    );
    assert!(
        has_writeback_to(&stmts, "sp"),
        "push must update sp: {stmts:?}"
    );
}

#[test]
fn test_aarch32_pop_loads_each_register_and_increments_sp() {
    // `pop {r4, r5, pc}` → three LoadMem + sp writeback.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "pop",
            vec![op("{r4, r5, pc}", OperandKind::Unknown)],
        ),
        Arch::Arm,
    );
    assert_eq!(count_loads(&stmts), 3, "pop of 3 regs → 3 loads: {stmts:?}");
    assert!(
        has_writeback_to(&stmts, "sp"),
        "pop must update sp: {stmts:?}"
    );
}

#[test]
fn test_aarch32_ldm_without_writeback_leaves_base_unchanged() {
    // `ldm r0, {r1, r2}` (no `!`) → two LoadMem, r0 NOT written back.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldm",
            vec![
                op("r0", OperandKind::Register),
                op("{r1, r2}", OperandKind::Unknown),
            ],
        ),
        Arch::Arm,
    );
    assert_eq!(count_loads(&stmts), 2, "ldm of 2 regs → 2 loads: {stmts:?}");
    assert!(
        !has_writeback_to(&stmts, "r0"),
        "ldm without ! must not write back r0: {stmts:?}"
    );
}

// --- P41a x86 non-stack memory → LoadMem/StoreMem --------------------

#[test]
fn test_x86_load_from_register_indirect_emits_loadmem() {
    // `mov rbx, [rax]` → LoadMem at rax (was Expr::Unknown before P41).
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "mov",
            vec![
                op("rbx", OperandKind::Register),
                op("[rax]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    let load = stmts.iter().find_map(|s| match s {
        IrStmt::LoadMem { address, .. } => Some(address.clone()),
        _ => None,
    });
    let load = load.expect("mov reg, [rax] must emit a LoadMem");
    assert!(
        format!("{load:?}").contains("\"rax\""),
        "load address must reference rax: {load:?}"
    );
}

#[test]
fn test_x86_store_to_scaled_index_emits_storemem() {
    // `mov [rbp + rax*4], rbx` → StoreMem at rbp + rax*4.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "mov",
            vec![
                op("[rbp + rax*4]", OperandKind::Memory),
                op("rbx", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let store = stmts.iter().find_map(|s| match s {
        IrStmt::StoreMem { address, .. } => Some(address.clone()),
        _ => None,
    });
    let store = store.expect("mov [rbp+rax*4], rbx must emit a StoreMem");
    let dbg = format!("{store:?}");
    assert!(dbg.contains("\"rbp\"") && dbg.contains("\"rax\""));
}

#[test]
fn test_x86_stack_slot_lowers_to_loadmem_named_by_slot() {
    // `mov rbx, [rbp - 8]` — P41b: the stack slot now lowers to the
    // byte-granular memory model like every other x86 memory operand.
    // The load temp keeps the `stk_rbp_-8` canonical name so the
    // analyst alias (`var_8h`) still resolves in the pretty-printer.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            "mov",
            vec![
                op("rbx", OperandKind::Register),
                op("[rbp - 8]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    let named = stmts
        .iter()
        .any(|s| matches!(s, IrStmt::LoadMem { dst, .. } if dst.name == "stk_rbp_-8"));
    assert!(
        named,
        "stack slot must lower to a LoadMem named by the slot: {stmts:?}"
    );
}

#[test]
fn test_repeated_load_same_address_is_read_read_consistent() {
    // Two loads of the SAME address with no intervening store must
    // yield the same value, so `a != b` is AlwaysFalse. Without the
    // load memo each load minted independent free bytes and this
    // widened to BothPossible.
    let p = Expr::Var(Var::new("p", 64));
    let a = Var::new("a#0", 64);
    let b = Var::new("b#0", 64);
    let slice = synthetic_slice(
        vec![
            IrStmt::LoadMem {
                dst: a.clone(),
                address: p.clone(),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: b.clone(),
                address: p,
                bits: 64,
            },
        ],
        Expr::ne(Expr::Var(a.clone()), Expr::Var(b.clone())),
        vec![Var::new("p", 64)],
        vec![a, b],
    );

    let verdict = solve_branch(&slice, solve_opts());
    assert_eq!(verdict, SmtResult::AlwaysFalse);
}

#[test]
fn test_store_between_loads_invalidates_read_read_memo() {
    // A store at an UNKNOWN address between two loads of address `p`
    // may alias `p`, so the second load is no longer guaranteed equal
    // to the first: `a != b` must widen to BothPossible, proving the
    // memo is dropped on every store (soundness guard on the memo).
    let p = Expr::Var(Var::new("p", 64));
    let q = Expr::Var(Var::new("q", 64));
    let a = Var::new("a#0", 64);
    let b = Var::new("b#0", 64);
    let slice = synthetic_slice(
        vec![
            IrStmt::LoadMem {
                dst: a.clone(),
                address: p.clone(),
                bits: 64,
            },
            IrStmt::StoreMem {
                address: q,
                value: Expr::konst(7, 64),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: b.clone(),
                address: p,
                bits: 64,
            },
        ],
        Expr::ne(Expr::Var(a.clone()), Expr::Var(b.clone())),
        vec![Var::new("p", 64), Var::new("q", 64)],
        vec![a, b],
    );

    let verdict = solve_branch(&slice, solve_opts());
    assert_eq!(verdict, SmtResult::BothPossible);
}

#[test]
fn test_wide_const_high_bits_survive_encoding() {
    // P40-d regression: a 128-bit all-ones constant carries bits above
    // the low 64. `Extract(0xFF..FF:128, 127, 127)` must be 1, so the
    // predicate `extracted == 1` is AlwaysTrue. If the encoder still
    // built the constant with `BV::from_u64` (truncating to 64 bits),
    // the top bit would be 0 and the verdict would flip to AlwaysFalse.
    let all_ones_128 = Expr::konst(u128::MAX, 128);
    let top_bit = Expr::extract(all_ones_128, 127, 127);
    let slice = synthetic_slice(
        Vec::new(),
        Expr::eq(top_bit, Expr::konst(1, 1)),
        Vec::new(),
        Vec::new(),
    );

    let verdict = solve_branch(&slice, solve_opts());
    assert_eq!(verdict, SmtResult::AlwaysTrue);
}

// --- `rrx`, whose address depends on the carry flag ------------------

/// Solve `ldr r0, [r1, r2, rrx]` with `r1 = 0`, `r2 = 2` and `CF` bound
/// to `carry`, and report whether the loaded address is necessarily
/// `expected`.
fn aarch32_rrx_address_is(carry: u128, expected: u128) -> SmtResult {
    use r2smt_ir::expr::Var;
    let lifted = lift_per_mnemonic(
        &insn(
            0x1000,
            "ldr",
            vec![
                op("r0", OperandKind::Register),
                op("[r1, r2, rrx]", OperandKind::Memory),
            ],
        ),
        Arch::Arm,
    );
    let address = lifted
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { address, .. } => Some(address.clone()),
            _ => None,
        })
        .expect("an rrx-indexed ldr must produce a LoadMem");
    // The address reads `r1`, `r2` and `CF` unversioned, so binding
    // them as SSA inputs of the same names is what the solver needs.
    let statements = vec![IrStmt::Assign {
        dst: Var::new("t_bind", 1),
        src: Expr::bool_and(
            Expr::eq(Expr::var("r1", 32), Expr::konst(0, 32)),
            Expr::bool_and(
                Expr::eq(Expr::var("r2", 32), Expr::konst(2, 32)),
                Expr::eq(Expr::var("CF", 1), Expr::konst(carry, 1)),
            ),
        ),
    }];
    let slice = SsaLiftedSlice {
        branch: synthetic_branch(),
        statements,
        condition: Expr::bool_or(
            Expr::eq(Expr::var("t_bind", 1), Expr::konst(0, 1)),
            Expr::eq(address, Expr::konst(expected, 32)),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        inputs: vec![Var::new("r1", 32), Var::new("r2", 32), Var::new("CF", 1)],
        defs: vec![Var::new("t_bind", 1)],
        arch: Arch::Arm,
    };
    solve_branch(&slice, solve_opts())
}

// --- the vector register file crossing memory ------------------------

#[test]
fn aarch64_spilling_a_vector_register_and_reloading_it_recovers_the_value() {
    // `str d8, [sp, #0x10]` / `ldr d9, [sp, #0x10]` is the ABI's own
    // idiom — `d8`–`d15` are callee-saved — so this pair sits in a large
    // share of real prologues.
    //
    // Both ends used to take the scalar register path, which sizes the
    // parent at the pointer width, 64, where `v8` is 128. That is a
    // register named at two widths, which nothing downstream reports.
    // Routing them through the vector reader and writer is what lets the
    // value survive the round trip: the store reads lane 0 of `v8` and
    // the load clears everything above the element of `v9`, so the
    // 128-bit comparison below pins both halves of the answer at once.
    let mut statements = vec![IrStmt::Assign {
        dst: Var::new("v8", 128),
        src: Expr::konst(0x1122_3344_5566_7788, 128),
    }];
    for (mnemonic, register) in [("str", "d8"), ("ldr", "d9")] {
        statements.extend(lift_per_mnemonic(
            &insn(
                0x1000,
                mnemonic,
                vec![
                    op(register, OperandKind::Register),
                    op("[sp, 0x10]", OperandKind::Memory),
                ],
            ),
            Arch::Aarch64,
        ));
    }
    assert!(
        statements
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
        "the vector spill pair must lift: {statements:?}"
    );
    let slice = r2smt_ssa::ssa_convert(&r2smt_slicer::LiftedSlice {
        branch: synthetic_branch(),
        statements,
        condition: Expr::eq(
            Expr::var("v9", 128),
            Expr::konst(0x1122_3344_5566_7788, 128),
        ),
        status: SliceStatus::Complete,
        treat_truncation_as_inputs: false,
        arch: Arch::Aarch64,
    });
    assert_eq!(solve_branch(&slice, solve_opts()), SmtResult::AlwaysTrue);
}

/// ARM's carry in the polarity this pipeline stores, which is its
/// inverse. See `aarch32_carry_convention_contracts.rs`; the helper
/// exists so these two read as what the architecture does rather than as
/// a flipped literal.
const fn stored(arm_carry: u128) -> u128 {
    arm_carry ^ 1
}

#[test]
fn aarch32_rrx_shifts_the_index_right_and_brings_the_carry_in_at_the_top() {
    // 2 >> 1 is 1, and with ARM's carry clear nothing enters the top.
    assert_eq!(aarch32_rrx_address_is(stored(0), 1), SmtResult::AlwaysTrue);
}

#[test]
fn aarch32_rrx_puts_a_set_carry_into_the_index_sign_bit() {
    // The whole point of the family, and what makes it a 33-bit
    // operation rather than a rotate of the register: with ARM's carry
    // set the same index gives `0x8000_0001`, not `1`. A lowering as
    // `Ror(x, 1)` would answer `1` here — the register's own low bit
    // would come back round instead of the carry — so this is the pair
    // that separates the two, and it is a wrong *address* rather than a
    // decline.
    assert_eq!(
        aarch32_rrx_address_is(stored(1), 0x8000_0001),
        SmtResult::AlwaysTrue
    );
}
