//! Contract tests for x87 floating-point support: the register-model
//! boundary, the effect table's decline conditions, the ESIL pre-empt,
//! and the lift-time stack's composition / underflow / overflow
//! behaviour.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::effect::registers_in_operand;
use r2smt_slicer::{
    InstructionKind, Slice, SliceLimits, SliceStatus, analyze, collect_branches, lift_slice,
    register_layout, slice_branch,
};

const FUNCTION_ADDRESS: u64 = 0x40_1000;
/// Every fixture instruction is spaced by this many bytes; the exact
/// encoding length is irrelevant to the slicer.
const STEP: u8 = 4;
/// IEEE binary64 `(ebits, sbits)`.
const DOUBLE: (u16, u16) = (11, 53);
/// IEEE binary64 bit pattern of `+1.0`.
const ONE_BITS: u128 = 0x3ff0_0000_0000_0000;

fn op(raw: &str, kind: OperandKind) -> Operand {
    Operand {
        raw: raw.into(),
        kind,
    }
}

fn reg(raw: &str) -> Operand {
    op(raw, OperandKind::Register)
}

fn mem(raw: &str) -> Operand {
    op(raw, OperandKind::Memory)
}

fn insn(addr: u64, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
    Instruction {
        address: Address(addr),
        size: STEP,
        bytes: vec![],
        mnemonic: mnemonic.into(),
        operands,
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

/// Lay a straight-line body out at consecutive addresses and append the
/// `cmp` / `je` tail every fixture branches on, so the slicer has a
/// candidate to walk backwards from.
fn program_with_tail(body: Vec<(&str, Vec<Operand>)>, probe: Operand) -> Program {
    let mut instructions = Vec::new();
    let mut addr = FUNCTION_ADDRESS;
    for (mnemonic, operands) in body {
        instructions.push(insn(addr, mnemonic, operands));
        addr += u64::from(STEP);
    }
    instructions.push(insn(addr, "mov", vec![reg("rax"), probe]));
    addr += u64::from(STEP);
    instructions.push(insn(
        addr,
        "cmp",
        vec![reg("rax"), op("0", OperandKind::Immediate)],
    ));
    addr += u64::from(STEP);
    instructions.push(insn(
        addr,
        "je",
        vec![op("0x402000", OperandKind::Immediate)],
    ));
    Program {
        arch: Arch::X86_64,
        bits: 64,
        entry: Some(Address(FUNCTION_ADDRESS)),
        functions: vec![Function {
            address: Address(FUNCTION_ADDRESS),
            name: Some("sym.fixture".into()),
            blocks: vec![BasicBlock {
                address: Address(FUNCTION_ADDRESS),
                instructions,
                successors: vec![],
            }],
            is_thumb: false,
        }],
    }
}

fn slice_of(program: &Program) -> Slice {
    let candidates = collect_branches(program);
    let candidate = candidates
        .first()
        .expect("fixture has a conditional branch");
    let function = program.functions.first().expect("fixture has a function");
    slice_branch(candidate, function, &SliceLimits::default(), program.arch)
}

/// The value of the first `StoreMem` in a lifted fixture.
fn first_stored_value(statements: &[IrStmt]) -> Expr {
    statements
        .iter()
        .find_map(|stmt| match stmt {
            IrStmt::StoreMem { value, .. } => Some(value.clone()),
            _ => None,
        })
        .expect("fixture stores through x87")
}

fn as_double(bits: Expr) -> Expr {
    Expr::bv_to_fp(bits, DOUBLE.0, DOUBLE.1)
}

// ---------------------------------------------------------------
// Register model
// ---------------------------------------------------------------

#[test]
fn test_x87_stack_pseudo_register_resolves_under_x86() {
    for arch in [Arch::X86, Arch::X86_64] {
        let layout = register_layout("st", arch).expect("`st` names the x87 stack");
        assert_eq!(layout.parent, "st");
    }
}

#[test]
fn test_x87_esil_register_spelling_does_not_resolve() {
    // `st0`..`st7` is radare2's *ESIL* spelling; no disassembly operand
    // carries it, and resolving it would give the ESIL ladder a way to
    // reach the register model behind the lifter's back.
    for name in ["st0", "st1", "st7"] {
        assert!(register_layout(name, Arch::X86_64).is_none(), "{name}");
    }
}

#[test]
fn test_x87_operand_tokenises_to_the_stack_pseudo_register() {
    // `registers_in_operand` splits on non-alphanumerics, so the
    // disassembly spelling `st(1)` yields the bare token `st` — and the
    // index is dropped, which is the whole point.
    assert_eq!(
        registers_in_operand(&reg("st(1)"), Arch::X86_64),
        vec!["st"]
    );
}

// ---------------------------------------------------------------
// Effect table
// ---------------------------------------------------------------

#[test]
fn test_x87_load_defines_and_uses_the_stack() {
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fld", vec![mem("qword [rbp - 0x10]")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::X87);
    assert_eq!(effect.defs, vec!["st"]);
    assert!(effect.uses.contains(&"st"));
    assert!(effect.uses.contains(&"rbp"));
}

#[test]
fn test_x87_extended_precision_operand_stays_unmodelled() {
    // 80-bit `tbyte` has no renderable IEEE sort, so the effect table
    // must fail closed to `Other` and let the slicer truncate rather
    // than let the lifter read ten bytes as eight.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fld", vec![mem("tbyte [rbp - 0x10]")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::Other);
}

#[test]
fn test_x87_status_word_family_stays_unmodelled() {
    // The compare / status-word idiom is not lifted, so it must not be
    // claimed by the effect table either.
    for mnemonic in ["fcom", "fcomp", "fnstsw", "fldcw"] {
        let effect = analyze(&insn(FUNCTION_ADDRESS, mnemonic, vec![]), Arch::X86_64);
        assert_eq!(effect.kind, InstructionKind::Other, "{mnemonic}");
    }
}

// ---------------------------------------------------------------
// Slicer
// ---------------------------------------------------------------

#[test]
fn test_x87_chain_is_kept_whole_by_the_slicer() {
    let program = program_with_tail(
        vec![
            ("fld", vec![mem("qword [rbp - 0x10]")]),
            ("fld", vec![mem("qword [rbp - 0x18]")]),
            ("faddp", vec![reg("st(1)"), reg("st(0)")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let slice = slice_of(&program);
    assert_eq!(slice.status, SliceStatus::Complete);
    let kept: Vec<&str> = slice
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    assert_eq!(kept, vec!["fld", "fld", "faddp", "fstp", "mov", "cmp"]);
}

// ---------------------------------------------------------------
// Lifter
// ---------------------------------------------------------------

#[test]
fn test_x87_chain_lifts_to_a_single_store_of_the_sum() {
    let program = program_with_tail(
        vec![
            ("fld", vec![mem("qword [rbp - 0x10]")]),
            ("fld", vec![mem("qword [rbp - 0x18]")]),
            ("faddp", vec![reg("st(1)"), reg("st(0)")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    // `fld` at -0x10 lands in ST(1) once the second push happens, so
    // `faddp st(1), st(0)` computes first + second in that order.
    let expected = Expr::fp_to_ieee_bv(Expr::fadd(
        as_double(Expr::var("stk_rbp_-16", 64)),
        as_double(Expr::var("stk_rbp_-24", 64)),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(first_stored_value(&lifted.statements), expected);
}

#[test]
fn test_x87_reverse_subtract_keeps_intel_operand_order() {
    // `fsubp st(1), st(0)` is `ST(1) := ST(1) - ST(0)`: the first
    // loaded value minus the second. Getting this backwards is the
    // classic x87 lifting bug, so it is pinned rather than assumed.
    let program = program_with_tail(
        vec![
            ("fld", vec![mem("qword [rbp - 0x10]")]),
            ("fld", vec![mem("qword [rbp - 0x18]")]),
            ("fsubp", vec![reg("st(1)"), reg("st(0)")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let expected = Expr::fp_to_ieee_bv(Expr::fsub(
        as_double(Expr::var("stk_rbp_-16", 64)),
        as_double(Expr::var("stk_rbp_-24", 64)),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(first_stored_value(&lifted.statements), expected);
}

#[test]
fn test_x87_single_precision_load_widens_to_double() {
    let program = program_with_tail(
        vec![
            ("fld", vec![mem("dword [rbp - 0x8]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let expected = Expr::fp_to_ieee_bv(Expr::fp_to_fp(
        Expr::bv_to_fp(Expr::var("stk_rbp_-8", 32), 8, 24),
        RoundingMode::NearestTiesEven,
        DOUBLE.0,
        DOUBLE.1,
    ));
    assert_eq!(first_stored_value(&lifted.statements), expected);
}

#[test]
fn test_fld1_is_not_lifted_through_esil() {
    // radare2's ESIL for `fld1` is eight register moves plus a 64-bit
    // immediate, every token of which the ESIL machine evaluates
    // happily — into `st<n>` variables at pointer width that no x87
    // handler ever reads. The per-mnemonic pre-empt has to win.
    let mut program = program_with_tail(
        vec![("fld1", vec![]), ("fstp", vec![mem("qword [rbp - 0x20]")])],
        mem("qword [rbp - 0x20]"),
    );
    let block = &mut program.functions[0].blocks[0];
    block.instructions[0].esil = Some(
        "st6,st7,=,st5,st6,=,st4,st5,=,st3,st4,=,st2,st3,=,st1,st2,=,st0,st1,=,\
         0x3ff0000000000000,st0,="
            .into(),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let esil_names: Vec<String> = (0..8).map(|n| format!("st{n}")).collect();
    for stmt in &lifted.statements {
        if let IrStmt::Assign { dst, .. } = stmt {
            assert!(
                !esil_names.contains(&dst.name),
                "ESIL lowering leaked `{name}` into the slice",
                name = dst.name
            );
        }
    }
    assert_eq!(
        first_stored_value(&lifted.statements),
        Expr::konst(ONE_BITS, 64)
    );
}

#[test]
fn test_x87_stack_underflow_becomes_a_free_input() {
    // A `fstp` with nothing pushed inside the slice reads a value that
    // existed before it. That is a free symbolic input, not an error —
    // and certainly not a panic.
    let program = program_with_tail(
        vec![("fstp", vec![mem("qword [rbp - 0x20]")])],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    assert_eq!(
        first_stored_value(&lifted.statements),
        Expr::var("x87_in_0", 64)
    );
}

#[test]
fn test_x87_stack_overflow_declines_and_havocs() {
    // Nine pushes overflow the eight data registers. The ninth declines
    // and forgets every modelled slot, so the store that follows reads
    // a free input rather than the stale `+1.0` the hardware would
    // never have left there.
    let mut body: Vec<(&str, Vec<Operand>)> = (0..9).map(|_| ("fld1", Vec::new())).collect();
    body.push(("fstp", vec![mem("qword [rbp - 0x20]")]));
    let program = program_with_tail(body, mem("qword [rbp - 0x20]"));
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    assert!(
        lifted.statements.iter().any(|stmt| matches!(
            stmt,
            IrStmt::Unsupported { mnemonic, .. } if mnemonic == "fld1"
        )),
        "the overflowing push must decline"
    );
    let stored = first_stored_value(&lifted.statements);
    assert_ne!(stored, Expr::konst(ONE_BITS, 64));
    assert!(
        matches!(&stored, Expr::Var(v) if v.name.starts_with("x87_in_")),
        "post-havoc reads must be free inputs, got {stored}"
    );
}
