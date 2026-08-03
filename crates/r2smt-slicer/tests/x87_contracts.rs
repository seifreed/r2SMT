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

/// Bit of AH the `fnstsw` / `test` idiom masks: C0 and C3, i.e. "not
/// greater than".
const NOT_GREATER_MASK: &str = "0x41";

const FUNCTION_ADDRESS: u64 = 0x40_1000;
/// Every fixture instruction is spaced by this many bytes; the exact
/// encoding length is irrelevant to the slicer.
const STEP: u8 = 4;
/// IEEE binary32 `(ebits, sbits)`.
const SINGLE: (u16, u16) = (8, 24);
/// IEEE binary64 `(ebits, sbits)`.
const DOUBLE: (u16, u16) = (11, 53);
/// x87 double-extended `(ebits, sbits)` — the sort the stack holds.
const EXTENDED: (u16, u16) = (15, 64);
/// Width of the extended sort's IEEE pattern, and so of a stack slot.
const SLOT_BITS: u16 = EXTENDED.0 + EXTENDED.1;
/// Width of the *stored* double-extended image, one bit wider because
/// x87 makes the leading significand bit explicit.
const MEMORY_BITS: u16 = SLOT_BITS + 1;
/// `+1.0` in the extended sort's 79-bit pattern.
const ONE_BITS: u128 = 0x1fff_8000_0000_0000_0000;
/// Prefix of the free symbolic values a stack underflow mints.
const FREE_INPUT_PREFIX: &str = "x87_in_";

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

/// Lay a straight-line body out at consecutive addresses. The last
/// entry must be the conditional branch the slicer walks back from.
fn program(body: Vec<(&str, Vec<Operand>)>) -> Program {
    let mut instructions = Vec::new();
    let mut addr = FUNCTION_ADDRESS;
    for (mnemonic, operands) in body {
        instructions.push(insn(addr, mnemonic, operands));
        addr += u64::from(STEP);
    }
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

/// `body`, then the `mov` / `cmp` / `je` tail that reads `probe` back
/// out of memory and gives the slicer a branch to walk from.
fn program_with_tail(mut body: Vec<(&str, Vec<Operand>)>, probe: Operand) -> Program {
    body.push(("mov", vec![reg("rax"), probe]));
    body.push(("cmp", vec![reg("rax"), op("0", OperandKind::Immediate)]));
    body.push(("je", vec![op("0x402000", OperandKind::Immediate)]));
    program(body)
}

fn slice_of(program: &Program) -> Slice {
    let candidates = collect_branches(program);
    let candidate = candidates
        .first()
        .expect("fixture has a conditional branch");
    let function = program.functions.first().expect("fixture has a function");
    slice_branch(candidate, function, &SliceLimits::default(), program.arch)
}

/// The first `StoreMem` in a lifted fixture, as `(value, width)`.
fn first_store(statements: &[IrStmt]) -> (Expr, u16) {
    statements
        .iter()
        .find_map(|stmt| match stmt {
            IrStmt::StoreMem { value, bits, .. } => Some((value.clone(), *bits)),
            _ => None,
        })
        .expect("fixture stores through x87")
}

fn first_stored_value(statements: &[IrStmt]) -> Expr {
    first_store(statements).0
}

/// Read a slot pattern as the double-extended float it denotes.
fn as_extended(pattern: Expr) -> Expr {
    Expr::bv_to_fp(pattern, EXTENDED.0, EXTENDED.1)
}

/// What `fld` of a `sort`-wide memory operand puts on the stack: an
/// exact widening into the extended sort, since both interchange
/// formats nest inside it.
fn widened(bits: Expr, sort: (u16, u16)) -> Expr {
    Expr::fp_to_ieee_bv(Expr::fp_to_fp(
        Expr::bv_to_fp(bits, sort.0, sort.1),
        RoundingMode::NearestTiesEven,
        EXTENDED.0,
        EXTENDED.1,
    ))
}

/// What an integer memory operand contributes: a two's-complement
/// conversion into the extended sort. Exact for every `m16int` and
/// `m32int`, both of which fit the 64-bit significand.
fn from_integer(bits: Expr) -> Expr {
    Expr::fp_to_ieee_bv(Expr::sbv_to_fp(
        bits,
        RoundingMode::NearestTiesEven,
        EXTENDED.0,
        EXTENDED.1,
    ))
}

/// What `fstp` to a `sort`-wide memory operand writes: the single
/// rounding, performed at the store exactly as the hardware does.
fn narrowed(slot: Expr, sort: (u16, u16)) -> Expr {
    Expr::fp_to_ieee_bv(Expr::fp_to_fp(
        as_extended(slot),
        RoundingMode::NearestTiesEven,
        sort.0,
        sort.1,
    ))
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
fn test_x87_extended_precision_load_is_modelled() {
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fld", vec![mem("xword [rbp - 0x10]")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::X87);
}

#[test]
fn test_x87_non_popping_store_has_no_extended_encoding() {
    // The SDM gives `m80fp` to `fld` and `fstp` alone: there is no way
    // to store the extended format without popping, so `fst tbyte` is
    // not an instruction and must not be claimed.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fst", vec![mem("xword [rbp - 0x10]")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::Other);
}

#[test]
fn test_x87_control_word_family_stays_unmodelled() {
    // `fldcw` reprograms the rounding mode *and* the precision control,
    // neither of which any fixed sort reflects. Until the guard lands it
    // must fail closed.
    for mnemonic in ["fldcw", "fnstcw", "fldenv"] {
        let effect = analyze(&insn(FUNCTION_ADDRESS, mnemonic, vec![]), Arch::X86_64);
        assert_eq!(effect.kind, InstructionKind::Other, "{mnemonic}");
    }
}

#[test]
fn test_x87_status_compare_defines_the_status_word_not_eflags() {
    // `fcom` reports through C0/C2/C3, so the slicer must see it as a
    // definition of `fsw` — and must *not* see it as the flag definer a
    // following `jcc` is looking for.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fcomp", vec![reg("st(1)")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::X87);
    assert!(effect.defs.contains(&"fsw"));
    assert!(!effect.defines_flags);
}

#[test]
fn test_x87_flag_compare_defines_eflags() {
    // `fcomi` writes ZF/PF/CF directly, so it *is* the flag definer.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fcomi", vec![reg("st(0)"), reg("st(1)")]),
        Arch::X86_64,
    );
    assert!(effect.defines_flags);
    assert!(!effect.defs.contains(&"fsw"));
}

#[test]
fn test_fnstsw_reads_the_status_word_without_touching_the_stack() {
    // `fnstsw` is the one x87 shape that leaves the data registers
    // alone: it reads `fsw` and writes AX.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fnstsw", vec![reg("ax")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::X87);
    assert!(effect.uses.contains(&"fsw"));
    assert!(effect.defs.contains(&"rax"));
    assert!(!effect.defs.contains(&"st"));
}

#[test]
fn test_fnstsw_only_has_a_register_form_naming_ax() {
    // The encoding stores to AX and nothing else; claiming another
    // register would model an instruction that does not exist.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fnstsw", vec![reg("bx")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::Other);
}

#[test]
fn test_sahf_is_a_flag_definer_reading_rax() {
    let effect = analyze(&insn(FUNCTION_ADDRESS, "sahf", vec![]), Arch::X86_64);
    assert_eq!(effect.kind, InstructionKind::Sahf);
    assert!(effect.defines_flags);
    assert_eq!(effect.uses, vec!["rax"]);
}

#[test]
fn test_unordered_compare_has_no_memory_form() {
    // `FUCOM` / `FUCOMP` are register-only; only the ordered compares
    // take an `m32fp` / `m64fp` source.
    let ordered = analyze(
        &insn(FUNCTION_ADDRESS, "fcom", vec![mem("qword [rbp - 0x10]")]),
        Arch::X86_64,
    );
    assert_eq!(ordered.kind, InstructionKind::X87);
    let unordered = analyze(
        &insn(FUNCTION_ADDRESS, "fucom", vec![mem("qword [rbp - 0x10]")]),
        Arch::X86_64,
    );
    assert_eq!(unordered.kind, InstructionKind::Other);
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
            ("faddp", vec![reg("st(1)")]),
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

#[test]
fn test_status_word_idiom_is_kept_whole_by_the_slicer() {
    // The chain that dominates 32-bit binaries, end to end at the
    // slicer: the branch reads ZF from `test`, which reads AH from
    // `fnstsw`, which reads the status word from `fcomp`, which reads
    // the stack the two loads filled. Every link has to be kept, and
    // the two pseudo-registers are what carry the last two.
    let fixture = program(vec![
        ("fld1", vec![]),
        ("fldz", vec![]),
        ("fcomp", vec![reg("st(1)")]),
        ("fnstsw", vec![reg("ax")]),
        (
            "test",
            vec![reg("ah"), op(NOT_GREATER_MASK, OperandKind::Immediate)],
        ),
        ("jz", vec![op("0x402000", OperandKind::Immediate)]),
    ]);
    let slice = slice_of(&fixture);
    assert_eq!(slice.status, SliceStatus::Complete);
    let kept: Vec<&str> = slice
        .instructions
        .iter()
        .map(|i| i.mnemonic.as_str())
        .collect();
    assert_eq!(kept, vec!["fld1", "fldz", "fcomp", "fnstsw", "test"]);
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
            ("faddp", vec![reg("st(1)")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    // `fld` at -0x10 lands in ST(1) once the second push happens, so
    // `faddp st(1), st(0)` computes first + second in that order. Both
    // loads widen exactly into the extended sort, the add rounds there,
    // and the store performs the one narrowing rounding.
    let sum = Expr::fp_to_ieee_bv(Expr::fadd(
        as_extended(widened(Expr::var("stk_rbp_-16", 64), DOUBLE)),
        as_extended(widened(Expr::var("stk_rbp_-24", 64), DOUBLE)),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(sum, DOUBLE)
    );
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
            ("fsubp", vec![reg("st(1)")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let difference = Expr::fp_to_ieee_bv(Expr::fsub(
        as_extended(widened(Expr::var("stk_rbp_-16", 64), DOUBLE)),
        as_extended(widened(Expr::var("stk_rbp_-24", 64), DOUBLE)),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(difference, DOUBLE)
    );
}

#[test]
fn test_x87_load_widens_exactly_and_the_store_rounds_once() {
    // The whole point of holding slots at the extended sort: the load
    // is an exact widening and the *only* rounding in the chain happens
    // at the store, which is where the hardware performs it.
    let program = program_with_tail(
        vec![
            ("fld", vec![mem("dword [rbp - 0x8]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let loaded = widened(Expr::var("stk_rbp_-8", 32), SINGLE);
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(loaded, DOUBLE)
    );
}

#[test]
fn test_x87_extended_store_writes_the_eighty_bit_image() {
    // The sort's pattern is 79 bits; the stored image is 80, because
    // x87 makes the leading significand bit explicit. The store must
    // widen across that bridge rather than write the sort's pattern.
    let program = program_with_tail(
        vec![("fld1", vec![]), ("fstp", vec![mem("xword [rbp - 0x20]")])],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    assert_eq!(first_store(&lifted.statements).1, MEMORY_BITS);
}

#[test]
fn test_x87_extended_load_declines_non_canonical_encodings() {
    // Stripping the explicit integer bit is only faithful when it
    // carries what IEEE would have inferred. The unsupported
    // double-extended encodings name a different number, so they must
    // resolve to a free value rather than to their canonical
    // counterpart — otherwise the solver gets a definite wrong answer.
    let program = program_with_tail(
        vec![
            ("fld", vec![mem("xword [rbp - 0x10]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let rendered = first_stored_value(&lifted.statements).to_string();
    assert!(
        rendered.contains(FREE_INPUT_PREFIX),
        "non-canonical encodings must fall to a free value, got: {rendered}"
    );
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
        narrowed(Expr::konst(ONE_BITS, SLOT_BITS), DOUBLE)
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
        narrowed(Expr::var("x87_in_0", SLOT_BITS), DOUBLE)
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
    let rendered = first_stored_value(&lifted.statements).to_string();
    assert!(
        rendered.contains(FREE_INPUT_PREFIX),
        "post-havoc reads must be free inputs, got {rendered}"
    );
}

#[test]
fn test_x87_stack_at_capacity_still_resolves_precisely() {
    // Teeth for the test above: eight pushes exactly fill the data
    // registers, so nothing declines and the store resolves to the
    // pushed constant with no free value anywhere in it.
    let mut body: Vec<(&str, Vec<Operand>)> = (0..8).map(|_| ("fld1", Vec::new())).collect();
    body.push(("fstp", vec![mem("qword [rbp - 0x20]")]));
    let program = program_with_tail(body, mem("qword [rbp - 0x20]"));
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(Expr::konst(ONE_BITS, SLOT_BITS), DOUBLE)
    );
}
// ---------------------------------------------------------------
// Control-word guard
// ---------------------------------------------------------------

/// `body`, the `mov` / `cmp` / `je` tail, and then `trailer` — placed
/// *after* the branch on purpose. The backward walk starts at the
/// branch and never visits it, which is exactly the blind spot the
/// function-scoped guard exists to cover: a control word has no
/// register name, so it defines nothing the live set can require.
fn program_with_trailer(
    body: Vec<(&str, Vec<Operand>)>,
    probe: Operand,
    trailer: Vec<(&str, Vec<Operand>)>,
) -> Program {
    let mut all = body;
    all.push(("mov", vec![reg("rax"), probe]));
    all.push(("cmp", vec![reg("rax"), op("0", OperandKind::Immediate)]));
    all.push(("je", vec![op("0x402000", OperandKind::Immediate)]));
    all.extend(trailer);
    program(all)
}

/// The rounding-dependent chain the guard has to protect: an add at the
/// extended sort, narrowed at the store.
fn rounding_chain() -> Vec<(&'static str, Vec<Operand>)> {
    vec![
        ("fld1", vec![]),
        ("fld1", vec![]),
        ("faddp", vec![reg("st(1)")]),
        ("fstp", vec![mem("qword [rbp - 0x20]")]),
    ]
}

#[test]
fn test_x87_arithmetic_is_complete_without_a_control_word_write() {
    // Teeth for the guard below: untouched, the same chain resolves.
    let program = program_with_trailer(rounding_chain(), mem("qword [rbp - 0x20]"), Vec::new());
    assert_eq!(slice_of(&program).status, SliceStatus::Complete);
}

#[test]
fn test_x87_arithmetic_truncates_when_a_control_word_is_reloaded() {
    // `fldcw` reprograms the rounding mode *and* the precision control,
    // the latter being the one no fixed sort can express: it sets the
    // significand width of every subsequent arithmetic result. The
    // writer sits past the branch, so nothing but the function-scoped
    // guard can catch it.
    for writer in ["fldcw", "fldenv", "frstor", "fxrstor", "ldmxcsr"] {
        let program = program_with_trailer(
            rounding_chain(),
            mem("qword [rbp - 0x20]"),
            vec![(writer, vec![mem("word [rbp - 0x40]")])],
        );
        assert!(
            matches!(slice_of(&program).status, SliceStatus::Truncated { .. }),
            "`{writer}` must invalidate the pinned control fields"
        );
    }
}

#[test]
fn test_x87_exact_chain_is_unaffected_by_a_control_word_write() {
    // The guard fires on what *computes*, not on anything x87. A load
    // and an extended store neither round nor depend on the significand
    // width, so this chain survives an `fldcw` in the same function.
    let program = program_with_trailer(
        vec![
            ("fld", vec![mem("xword [rbp - 0x10]")]),
            ("fstp", vec![mem("xword [rbp - 0x30]")]),
        ],
        mem("qword [rbp - 0x30]"),
        vec![("fldcw", vec![mem("word [rbp - 0x40]")])],
    );
    assert_eq!(slice_of(&program).status, SliceStatus::Complete);
}

#[test]
fn test_x87_control_word_read_stays_unmodelled() {
    // `fnstcw` takes the same posture as `stmxcsr`: it is not in the
    // modelled set, so it stays `Other` and truncates a pending slice
    // rather than being stepped over as if it had no effect.
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "fnstcw", vec![mem("word [rbp - 0x40]")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::Other);
}

// --- the spellings radare2 actually emits ---
//
// Everything below pins a form against real disassembler output rather
// than the Intel manual's rendering. Each one declined before this
// round while a hand-written fixture in the manual's spelling passed,
// which is exactly how the gap survived.

#[test]
fn test_x87_popping_arithmetic_takes_the_one_operand_spelling() {
    // `de c1` disassembles as `faddp st(1)`, never as a bare `faddp`
    // and never with two operands. The one-operand popping form is the
    // most common x87 arithmetic encoding a compiler emits, and it
    // reached the classifier's catch-all.
    for mnemonic in ["faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp"] {
        let i = insn(FUNCTION_ADDRESS, mnemonic, vec![reg("st(1)")]);
        assert_eq!(
            analyze(&i, Arch::X86_64).kind,
            InstructionKind::X87,
            "{mnemonic} st(1)"
        );
    }
}

#[test]
fn test_x87_non_popping_register_arithmetic_targets_the_top_of_stack() {
    // `d8 c1` is `fadd st(1)`, meaning ST(0) := ST(0) + ST(1). The
    // operand count is what names the destination: one operand means
    // the top of stack, two mean the register that is spelled.
    let i = insn(FUNCTION_ADDRESS, "fadd", vec![reg("st(1)")]);
    assert_eq!(analyze(&i, Arch::X86_64).kind, InstructionKind::X87);
}

#[test]
fn test_x87_extended_memory_uses_the_disassembler_width_keyword() {
    // radare2 writes the 80-bit format as `xword`; `tbyte` is the
    // Intel / MASM keyword. Only the latter was recognised, so the
    // whole 79-to-80 bit bridge was unreachable on real input.
    for raw in ["xword [rbp - 0x10]", "tbyte [rbp - 0x10]"] {
        let i = insn(FUNCTION_ADDRESS, "fld", vec![mem(raw)]);
        assert_eq!(
            analyze(&i, Arch::X86_64).kind,
            InstructionKind::X87,
            "{raw}"
        );
    }
}

// ---------------------------------------------------------------
// Integer-operand arithmetic
// ---------------------------------------------------------------

#[test]
fn test_x87_integer_operand_arithmetic_is_modelled() {
    // `de 00` is `fiadd word [eax]` and `da 00` `fiadd dword [eax]`.
    for mnemonic in ["fiadd", "fisub", "fisubr", "fimul", "fidiv", "fidivr"] {
        for width in ["word", "dword"] {
            let i = insn(
                FUNCTION_ADDRESS,
                mnemonic,
                vec![mem(&format!("{width} [rbp - 0x8]"))],
            );
            assert_eq!(
                analyze(&i, Arch::X86_64).kind,
                InstructionKind::X87,
                "{mnemonic} {width}"
            );
        }
    }
}

#[test]
fn test_x87_integer_operand_arithmetic_has_no_qword_encoding() {
    // Opcodes `DE` and `DA` give the family `m16int` and `m32int`
    // alone. Reusing `fild`'s width table would accept a `m64int`
    // operand no encoding can produce.
    let i = insn(FUNCTION_ADDRESS, "fiadd", vec![mem("qword [rbp - 0x8]")]);
    assert_eq!(analyze(&i, Arch::X86_64).kind, InstructionKind::Other);
}

#[test]
fn test_x87_integer_compare_defines_the_status_word() {
    let effect = analyze(
        &insn(FUNCTION_ADDRESS, "ficom", vec![mem("dword [rbp - 0x8]")]),
        Arch::X86_64,
    );
    assert_eq!(effect.kind, InstructionKind::X87);
    assert!(effect.defs.contains(&"fsw"));
}

#[test]
fn test_x87_integer_arithmetic_reads_its_operand_as_an_integer() {
    // The whole point of carrying the format on the source: the same
    // operand shape spells both families, and reading `m32int` as an
    // `m32fp` bit pattern would be a definite wrong number rather than
    // a decline.
    let program = program_with_tail(
        vec![
            ("fld1", vec![]),
            ("fiadd", vec![mem("dword [rbp - 0x8]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let sum = Expr::fp_to_ieee_bv(Expr::fadd(
        as_extended(Expr::konst(ONE_BITS, SLOT_BITS)),
        as_extended(from_integer(Expr::var("stk_rbp_-8", 32))),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(sum, DOUBLE)
    );
}

#[test]
fn test_x87_integer_reverse_divide_swaps_the_operands() {
    // `fidivr m32int` is `ST(0) := m / ST(0)`, the mirror of `fidiv`.
    let program = program_with_tail(
        vec![
            ("fld1", vec![]),
            ("fidivr", vec![mem("dword [rbp - 0x8]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    let quotient = Expr::fp_to_ieee_bv(Expr::fdiv(
        as_extended(from_integer(Expr::var("stk_rbp_-8", 32))),
        as_extended(Expr::konst(ONE_BITS, SLOT_BITS)),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(quotient, DOUBLE)
    );
}

#[test]
fn test_x87_integer_compare_pop_discards_the_top_of_stack() {
    // `ficomp` is dispatched ahead of the suffix-stripping arithmetic
    // classifier, which would otherwise peel its trailing `p` and read
    // it as the non-popping `ficom`. The pop is observable: with it,
    // the store writes the `+1.0` underneath; without it, the `+0.0`
    // the compare examined.
    let program = program_with_tail(
        vec![
            ("fld1", vec![]),
            ("fldz", vec![]),
            ("ficomp", vec![mem("dword [rbp - 0x8]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
    );
    let lifted = lift_slice(&slice_of(&program), Arch::X86_64);
    assert_eq!(
        first_stored_value(&lifted.statements),
        narrowed(Expr::konst(ONE_BITS, SLOT_BITS), DOUBLE)
    );
}

#[test]
fn test_x87_integer_arithmetic_truncates_when_a_control_word_is_reloaded() {
    // `fiadd` rounds its result exactly as `fadd` does, so precision
    // control reaches it and the guard has to cover it.
    let program = program_with_trailer(
        vec![
            ("fld1", vec![]),
            ("fiadd", vec![mem("dword [rbp - 0x8]")]),
            ("fstp", vec![mem("qword [rbp - 0x20]")]),
        ],
        mem("qword [rbp - 0x20]"),
        vec![("fldcw", vec![mem("word [rbp - 0x40]")])],
    );
    assert!(matches!(
        slice_of(&program).status,
        SliceStatus::Truncated { .. }
    ));
}

#[test]
fn test_x87_integer_compare_is_unaffected_by_a_control_word_write() {
    // Teeth for the test above, and the reason `ficom` / `ficomp` stay
    // out of the pinning set: a compare computes no result, so neither
    // the rounding mode nor the precision control can change what it
    // reports.
    let fixture = program(vec![
        ("fld1", vec![]),
        ("ficom", vec![mem("dword [rbp - 0x8]")]),
        ("fnstsw", vec![reg("ax")]),
        (
            "test",
            vec![reg("ah"), op(NOT_GREATER_MASK, OperandKind::Immediate)],
        ),
        ("jz", vec![op("0x402000", OperandKind::Immediate)]),
        ("fldcw", vec![mem("word [rbp - 0x40]")]),
    ]);
    assert_eq!(slice_of(&fixture).status, SliceStatus::Complete);
}

#[test]
fn test_x87_popping_flag_compare_takes_the_disassembler_spelling() {
    // `df f1` is `fcompi st(1)`, not the manual's `fcomip`.
    for mnemonic in ["fcompi", "fucompi", "fcomip", "fucomip"] {
        let i = insn(FUNCTION_ADDRESS, mnemonic, vec![reg("st(1)")]);
        assert_eq!(
            analyze(&i, Arch::X86_64).kind,
            InstructionKind::X87,
            "{mnemonic}"
        );
    }
}
