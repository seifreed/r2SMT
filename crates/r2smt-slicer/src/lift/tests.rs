#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};

use super::*;
use crate::collector::collect_branches;
use crate::condition::BranchKind;
use crate::slice::{SliceLimits, slice_branch};

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

fn one_block_program(insns: Vec<Instruction>) -> Program {
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

fn lift_first(program: &Program, arch: Arch) -> LiftedSlice {
    let candidates = collect_branches(program);
    let cand = candidates.first().expect("at least one branch");
    let slice = slice_branch(
        cand,
        &program.functions[0],
        &SliceLimits::default(),
        program.arch,
    );
    lift_slice(&slice, arch)
}

fn find_assign<'a>(stmts: &'a [IrStmt], dst_name: &str) -> Option<&'a IrStmt> {
    stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == dst_name))
}

#[test]
fn cmp_emits_tmp_and_flag_assignments() {
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("2", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1003,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let lifted = lift_first(&program, Arch::X86);
    // Statements: tmp := eax - 2, ZF := tmp == 0, SF, CF, OF, PF.
    assert!(lifted.statements.len() >= 5);
    let zf = find_assign(&lifted.statements, "ZF").expect("ZF set");
    if let IrStmt::Assign {
        src: Expr::Eq(_, rhs),
        ..
    } = zf
    {
        assert_eq!(**rhs, Expr::konst(0, 32));
    } else {
        panic!("ZF should be `tmp == 0`, got {zf:?}");
    }
    // Branch condition for `jne` is `ZF == 0`.
    assert_eq!(
        lifted.condition,
        Expr::eq(Expr::flag("ZF"), Expr::konst(0, 1))
    );
}

#[test]
fn subregister_cmp_uses_register_width_not_immediate_pseudo_width() {
    // `cmp al, 1` is an 8-bit subtraction. The compare width must be
    // `al`'s 8 bits, not the immediate's pointer-width pseudo-size:
    // widening corrupts SF (e.g. al=0xFF → 8-bit 0xFF-1=0xFE is
    // negative, but 32-bit 254 is not), flipping `js`/`jns`.
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "cmp",
            vec![
                op("al", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1002,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let lifted = lift_first(&program, Arch::X86);
    let sf = find_assign(&lifted.statements, "SF").expect("SF set");
    if let IrStmt::Assign {
        src: Expr::Slt(_, rhs),
        ..
    } = sf
    {
        assert_eq!(**rhs, Expr::konst(0, 8));
    } else {
        panic!("SF should be `slt(tmp, 0)`, got {sf:?}");
    }
}

#[test]
fn xor_same_reg_emits_zero_assignment_and_zf_one() {
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "xor",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            2,
            "test",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1004,
            6,
            "jnz",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let lifted = lift_first(&program, Arch::X86);
    let eax = find_assign(&lifted.statements, "rax").expect("rax assigned");
    if let IrStmt::Assign { src, .. } = eax {
        assert_eq!(*src, Expr::konst(0, 32));
    }
}

#[test]
fn opaque_predicate_yields_full_chain() {
    // mov eax, ecx ; imul eax, eax ; and eax, 1 ; cmp eax, 2 ; jne junk
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            2,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("ecx", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1002,
            3,
            "imul",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        insn(
            0x40_1005,
            3,
            "and",
            vec![
                op("eax", OperandKind::Register),
                op("1", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1008,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("2", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_100b,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let lifted = lift_first(&program, Arch::X86);
    // Expect at least: rax := rcx, rax := rax * rax, rax := rax & 1,
    // tmp := rax - 2, ZF := (tmp == 0), …
    let mnemonics: Vec<String> = lifted
        .statements
        .iter()
        .filter_map(|s| match s {
            IrStmt::Assign { dst, .. } => Some(dst.name.clone()),
            _ => None,
        })
        .collect();
    assert!(mnemonics.contains(&"rax".to_string()));
    assert!(mnemonics.iter().any(|n| n.starts_with("t_")));
    assert!(mnemonics.contains(&"ZF".to_string()));
    assert_eq!(
        lifted.condition,
        Expr::eq(Expr::flag("ZF"), Expr::konst(0, 1))
    );
}

#[test]
fn unsupported_mnemonic_is_marked() {
    // Take a small program with a synthetic unsupported insn that the
    // slicer would (incorrectly for this test) include. We bypass
    // the slicer entirely.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86);
        ctx.lift_instruction(&insn(
            0x40_1000,
            3,
            "vpxor",
            vec![op("xmm0", OperandKind::Register)],
        ));
        ctx.stmts
    };
    assert!(matches!(stmts[0], IrStmt::Unsupported { .. }));
}

#[test]
fn branch_condition_above_combines_cf_and_zf() {
    let cand = BranchCandidate {
        address: Address(0),
        function: Address(0),
        block: Address(0),
        kind: BranchKind::Jcc,
        mnemonic: "ja".into(),
        condition: BranchCondition::Above,
        formula: "CF == 0 && ZF == 0".into(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: None,
        bit_index: None,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    };
    let expr = lift_branch_condition(&cand, Arch::X86);
    assert_eq!(
        expr,
        Expr::bool_and(
            Expr::eq(Expr::flag("CF"), Expr::konst(0, 1)),
            Expr::eq(Expr::flag("ZF"), Expr::konst(0, 1)),
        )
    );
}

fn aarch64_branch_cand(
    condition: BranchCondition,
    reg: &str,
    bit_index: Option<u8>,
) -> BranchCandidate {
    BranchCandidate {
        address: Address(0),
        function: Address(0),
        block: Address(0),
        kind: BranchKind::Jcc,
        mnemonic: "cbz".into(),
        condition,
        formula: String::new(),
        taken_target: None,
        fallthrough_target: None,
        compare_register: Some(reg.into()),
        bit_index,
        upstream_resolved: None,
        operand_raws: Vec::new(),
        is_thumb: false,
    }
}

#[test]
fn cbz_xzr_resolves_to_constant_zero_not_free_var() {
    // `xzr` always reads 0; modelling it as a free `Expr::var("xzr")`
    // loses the precise `cbz xzr` → always-taken verdict.
    let cand = aarch64_branch_cand(BranchCondition::RegisterZero, "xzr", None);
    let expr = lift_branch_condition(&cand, Arch::Aarch64);
    assert_eq!(expr, Expr::eq(Expr::konst(0, 64), Expr::konst(0, 64)));
}

#[test]
fn tbz_unparsed_bit_index_is_free_not_silently_bit_zero() {
    // `bit_index == None` must NOT substitute bit 0 — that fabricates a
    // different concrete predicate. A free symbolic value is sound.
    let cand = aarch64_branch_cand(BranchCondition::BitZero, "x0", None);
    let expr = lift_branch_condition(&cand, Arch::Aarch64);
    assert!(matches!(expr, Expr::Unknown(_)), "got {expr:?}");
}

#[test]
fn tbz_out_of_range_bit_index_is_free() {
    // `w0` is 32-bit; bit 40 cannot be extracted. Surface as free
    // rather than emit an invalid `Extract`.
    let cand = aarch64_branch_cand(BranchCondition::BitZero, "w0", Some(40));
    let expr = lift_branch_condition(&cand, Arch::Aarch64);
    assert!(matches!(expr, Expr::Unknown(_)), "got {expr:?}");
}

#[test]
fn tbz_valid_bit_index_extracts_that_bit() {
    let cand = aarch64_branch_cand(BranchCondition::BitZero, "x0", Some(5));
    let expr = lift_branch_condition(&cand, Arch::Aarch64);
    assert_eq!(
        expr,
        Expr::eq(Expr::extract(Expr::var("x0", 64), 5, 5), Expr::konst(0, 1))
    );
}

#[test]
fn json_round_trips() {
    let program = one_block_program(vec![
        insn(
            0x40_1000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("2", OperandKind::Immediate),
            ],
        ),
        insn(
            0x40_1003,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ),
    ]);
    let lifted = lift_first(&program, Arch::X86);
    let json = serde_json::to_string(&lifted).unwrap();
    let back: LiftedSlice = serde_json::from_str(&json).unwrap();
    assert_eq!(back, lifted);
}

#[test]
fn parse_immediate_supports_hex_decimal_negative() {
    assert_eq!(parse_immediate("0x10"), Some(0x10));
    assert_eq!(parse_immediate("16"), Some(16));
    assert_eq!(parse_immediate("-2"), Some(u64::MAX - 1));
    assert!(parse_immediate("foo").is_none());
}

#[test]
fn parse_immediate_strips_arm_hash_prefix() {
    // AArch64 / AArch32 disassembly emits `#`-prefixed immediates.
    assert_eq!(parse_immediate("#0x10"), Some(0x10));
    assert_eq!(parse_immediate("#42"), Some(42));
    assert_eq!(parse_immediate("#-1"), Some(u64::MAX));
    assert_eq!(parse_immediate("# 0x20"), Some(0x20));
}

#[test]
fn mov_al_preserves_upper_bits_of_rax() {
    // `mov al, 0x10` on x86_64: rax becomes
    //   Concat(Extract(rax, 63, 8), 0x10:8).
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            2,
            "mov",
            vec![
                op("al", OperandKind::Register),
                op("0x10", OperandKind::Immediate),
            ],
        ));
        ctx.stmts
    };
    let assign = stmts.first().expect("mov produces an assignment");
    match assign {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(dst.bits, 64);
            match src {
                Expr::Concat { high, low } => {
                    assert_eq!(
                        **high,
                        Expr::extract(Expr::var("rax", 64), 63, 8),
                        "high preserve must extract bits 63:8 of rax"
                    );
                    assert_eq!(**low, Expr::konst(0x10, 8));
                }
                other => panic!("expected Concat RHS, got {other:?}"),
            }
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn mov_ah_preserves_low_and_high_bits_of_rax() {
    // `mov ah, 0x5` on x86_64: rax becomes
    //   Concat(Concat(Extract(rax, 63, 16), 0x5:8), Extract(rax, 7, 0)).
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            2,
            "mov",
            vec![
                op("ah", OperandKind::Register),
                op("0x5", OperandKind::Immediate),
            ],
        ));
        ctx.stmts
    };
    let assign = stmts.first().expect("mov produces an assignment");
    match assign {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(dst.bits, 64);
            // Outer concat: high = bits 63:16, low = (concat(0x5:8, bits 7:0))
            match src {
                Expr::Concat { high, low } => {
                    assert_eq!(**high, Expr::extract(Expr::var("rax", 64), 63, 16));
                    match &**low {
                        Expr::Concat {
                            high: inner_high,
                            low: inner_low,
                        } => {
                            assert_eq!(**inner_high, Expr::konst(0x5, 8));
                            assert_eq!(**inner_low, Expr::extract(Expr::var("rax", 64), 7, 0));
                        }
                        other => panic!("inner concat expected, got {other:?}"),
                    }
                }
                other => panic!("expected outer Concat, got {other:?}"),
            }
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn mov_eax_zero_extends_to_rax_on_x86_64() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            5,
            "mov",
            vec![
                op("eax", OperandKind::Register),
                op("0x12345678", OperandKind::Immediate),
            ],
        ));
        ctx.stmts
    };
    let assign = stmts.first().unwrap();
    match assign {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(dst.bits, 64);
            assert_eq!(*src, Expr::zero_ext(Expr::konst(0x1234_5678, 32), 64));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn mov_rax_full_replace_on_x86_64() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            7,
            "mov",
            vec![
                op("rax", OperandKind::Register),
                op("rbx", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let assign = stmts.first().unwrap();
    match assign {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(dst.bits, 64);
            assert_eq!(*src, Expr::var("rbx", 64));
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn xor_ah_al_produces_real_arithmetic_not_unknown() {
    // Regression: previously emitted Expr::Unknown via the
    // sub_register_alias guard. With the precise model the lifter
    // must produce a concat of bits 63:16 + (al XOR ah) + bits 7:0
    // (because `ah` is the destination — its slot is 15:8). The
    // RHS of the XOR is read from al, which is bits 7:0 of rax.
    //
    // Post flag-ordering fix the XOR lives in a synthetic temp; the
    // rax assignment is a concat that splices that temp into the
    // `ah` slot. Both the temp and the rax assignment must avoid
    // `Expr::Unknown`.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            2,
            "xor",
            vec![
                op("ah", OperandKind::Register),
                op("al", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let temp = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with("t_")))
        .expect("temp assignment present");
    let temp_src = match temp {
        IrStmt::Assign { src, .. } => format!("{src}"),
        _ => unreachable!(),
    };
    assert!(
        !temp_src.contains('?'),
        "xor ah, al must not collapse to Unknown: {temp_src}"
    );
    assert!(
        temp_src.contains('^'),
        "expected xor in temp RHS: {temp_src}"
    );
    let rax = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "rax"))
        .expect("rax assignment present");
    let rax_src = match rax {
        IrStmt::Assign { src, .. } => format!("{src}"),
        _ => unreachable!(),
    };
    assert!(
        !rax_src.contains('?'),
        "rax assignment must not contain Unknown: {rax_src}"
    );
    assert!(
        rax_src.contains("t_"),
        "rax assignment should splice the temp into the ah slot: {rax_src}"
    );
}

#[test]
fn sub_dst_dst_flags_reference_pre_op_value_not_post_op() {
    // Regression for the x86 RMW flag-ordering bug: `sub eax, eax`
    // followed by `je target`. Pre-op: eax-eax == 0, so ZF should be
    // 1 and the branch unconditional. The flag *value* expression
    // must reference the same operands the destination was computed
    // from, not the post-write `rax` (which under SSA would create a
    // tautological self-reference once the lifter stashes the result
    // in a temp). After the fix, `lift_add_sub` introduces a temp
    // and the ZF assignment reads from that temp.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            2,
            "sub",
            vec![
                op("eax", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    // Find the ZF assignment.
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("ZF assigned");
    // Its src must compare `tmp == 0`, where `tmp` is the synthetic
    // temp the lifter emits. The temp is the only `Var` on the LHS
    // of the equality; no `rax` / `Extract(rax, ...)` should appear.
    match zf {
        IrStmt::Assign {
            src: Expr::Eq(lhs, _rhs),
            ..
        } => {
            let rendered = format!("{lhs}");
            assert!(
                rendered.starts_with("t_"),
                "ZF LHS should reference a temp, got `{rendered}`"
            );
            assert!(
                !rendered.contains("rax"),
                "ZF must not read `rax` post-write: got `{rendered}`"
            );
        }
        other => panic!("expected ZF := Eq(.., ..), got {other:?}"),
    }
    // The destination write must source from the same temp.
    let rax = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "rax"))
        .expect("rax assigned");
    let rax_src = format!("{rax:?}");
    assert!(
        rax_src.contains("t_"),
        "rax assignment should source from the temp, got {rax_src}"
    );
}

#[test]
fn add_flags_reference_pre_op_value_not_post_op() {
    // Sibling regression for `add` — same mechanism as `sub`.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            3,
            "add",
            vec![
                op("eax", OperandKind::Register),
                op("ebx", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("ZF assigned");
    match zf {
        IrStmt::Assign {
            src: Expr::Eq(lhs, _),
            ..
        } => {
            let rendered = format!("{lhs}");
            assert!(
                rendered.starts_with("t_"),
                "ZF LHS should reference a temp, got `{rendered}`"
            );
        }
        other => panic!("expected ZF := Eq(.., ..), got {other:?}"),
    }
}

#[test]
fn and_flags_reference_pre_op_value_not_post_op() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            3,
            "and",
            vec![
                op("eax", OperandKind::Register),
                op("0x1", OperandKind::Immediate),
            ],
        ));
        ctx.stmts
    };
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("ZF assigned");
    match zf {
        IrStmt::Assign {
            src: Expr::Eq(lhs, _),
            ..
        } => {
            let rendered = format!("{lhs}");
            assert!(
                rendered.starts_with("t_"),
                "ZF LHS should reference a temp, got `{rendered}`"
            );
        }
        other => panic!("expected ZF := Eq(.., ..), got {other:?}"),
    }
}

#[test]
fn aarch64_subs_dst_overlap_emits_flags_before_destination_write() {
    // `subs x0, x0, x1` — destination overlaps source. The flag
    // updates must reference the pre-op `x0`, not the post-write
    // version. After the fix `aarch64_set_arith_flags` is called
    // before the destination write so SSA renames the lhs/rhs reads
    // inside CF to the unwritten `x0`.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::Aarch64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            4,
            "subs",
            vec![
                op("x0", OperandKind::Register),
                op("x0", OperandKind::Register),
                op("x1", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    // Locate the position of the `x0` write and the CF assignment.
    let x0_pos = stmts
        .iter()
        .position(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "x0"))
        .expect("x0 assignment present");
    let cf_pos = stmts
        .iter()
        .position(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "CF"))
        .expect("CF assignment present");
    assert!(
        cf_pos < x0_pos,
        "CF must be emitted before the destination write \
         (cf at {cf_pos}, x0 at {x0_pos})"
    );
}

#[test]
fn shl_flags_reference_pre_op_value_not_post_op() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            3,
            "shl",
            vec![
                op("eax", OperandKind::Register),
                op("0x2", OperandKind::Immediate),
            ],
        ));
        ctx.stmts
    };
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("ZF assigned");
    match zf {
        IrStmt::Assign {
            src: Expr::Eq(lhs, _),
            ..
        } => {
            let rendered = format!("{lhs}");
            assert!(
                rendered.starts_with("t_"),
                "ZF LHS should reference a temp, got `{rendered}`"
            );
        }
        other => panic!("expected ZF := Eq(.., ..), got {other:?}"),
    }
}

#[test]
fn aarch32_rsb_swaps_operands_and_subtracts() {
    // `rsbs r0, r1, r2` should compute `r2 - r1` and set flags from it.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::Arm);
        ctx.lift_instruction(&insn(
            0x40_1000,
            4,
            "rsbs",
            vec![
                op("r0", OperandKind::Register),
                op("r1", OperandKind::Register),
                op("r2", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let temp = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with("t_")))
        .expect("rsbs produces a temp assignment");
    let rendered = match temp {
        IrStmt::Assign { src, .. } => format!("{src}"),
        _ => unreachable!(),
    };
    // r2 - r1, so r2 appears on the left of the subtraction.
    assert!(
        rendered.contains("r2") && rendered.contains("r1") && rendered.contains('-'),
        "rsb temp should compute r2 - r1, got `{rendered}`"
    );
}

#[test]
fn aarch32_vpush_stores_each_doubleword_register() {
    // `vpush {d8, d9}` stores two 64-bit registers and decrements sp.
    let stmts = lift_aarch32("vpush", vec![op("{d8, d9}", OperandKind::Unknown)]);
    let stores = stmts
        .iter()
        .filter(|s| matches!(s, IrStmt::StoreMem { bits: 64, .. }))
        .count();
    assert_eq!(stores, 2, "{stmts:?}");
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
    );
}

#[test]
fn aarch32_vpop_range_expands_every_member() {
    // `vpop {s0-s3}` names a range: all four single-precision registers
    // must load, not just the two endpoints.
    let stmts = lift_aarch32("vpop", vec![op("{s0-s3}", OperandKind::Unknown)]);
    let loads = stmts
        .iter()
        .filter(|s| matches!(s, IrStmt::LoadMem { bits: 32, .. }))
        .count();
    assert_eq!(loads, 4, "{stmts:?}");
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
    );
}

#[test]
fn aarch32_vldr_loads_the_register_width_through_the_byte_model() {
    // `vldr s0, [r0, #4]` loads 32 bits (the `s` register width) into a
    // temp and merges it into the vector register.
    let stmts = lift_aarch32("vldr", vec![reg("s0"), op("[r0, #4]", OperandKind::Memory)]);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { bits: 32, .. })),
        "{stmts:?}"
    );
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
    );
}

#[test]
fn aarch32_vstr_stores_the_register_width() {
    // `vstr d0, [r1]` stores 64 bits (the `d` register width).
    let stmts = lift_aarch32("vstr", vec![reg("d0"), op("[r1]", OperandKind::Memory)]);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::StoreMem { bits: 64, .. })),
        "{stmts:?}"
    );
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
    );
}

fn aarch32_lifts_cleanly(stmts: &[IrStmt]) -> bool {
    !stmts
        .iter()
        .any(|s| matches!(s, IrStmt::Unsupported { .. }))
}

#[test]
fn aarch32_register_post_index_updates_its_base_by_the_delta_register() {
    // `ldr r0, [r1], r2` loads from `r1` and then adds `r2` to it. The
    // base update is the part that used to be dropped silently.
    let stmts = lift_aarch32(
        "ldr",
        vec![reg("r0"), op("[r1]", OperandKind::Memory), reg("r2")],
    );
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts.iter().any(|s| matches!(
            s,
            IrStmt::Assign { dst, src }
                if dst.name == "r1" && format!("{src:?}").contains("r2")
        )),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_predicated_load_wraps_its_destination_in_an_ite() {
    let stmts = lift_aarch32("ldreq", vec![reg("r0"), op("[r1]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts.iter().any(|s| matches!(
            s,
            IrStmt::Assign { dst, src } if dst.name == "r0" && format!("{src:?}").contains("Ite")
        )),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_predicated_store_becomes_a_read_modify_write() {
    // The IR has no conditional store. Wrapping only `Assign` would let
    // `streq` write unconditionally — the memory would take the new
    // value on a path the machine never stores on. Instead the store
    // writes back the bytes already there when the predicate is false.
    let stmts = lift_aarch32("streq", vec![reg("r0"), op("[r1]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    let store = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::StoreMem { value, .. } => Some(format!("{value:?}")),
            _ => None,
        })
        .expect("streq must still store");
    assert!(store.contains("Ite"), "{store}");
}

#[test]
fn aarch32_predicated_store_loads_the_old_bytes_before_storing() {
    // The read-modify-write only works if the load is emitted ahead of
    // the store that consumes it.
    let stmts = lift_aarch32("streq", vec![reg("r0"), op("[r1]", OperandKind::Memory)]);
    let load_at = stmts
        .iter()
        .position(|s| matches!(s, IrStmt::LoadMem { .. }));
    let store_at = stmts
        .iter()
        .position(|s| matches!(s, IrStmt::StoreMem { .. }));
    assert!(load_at < store_at, "{stmts:?}");
    assert!(load_at.is_some(), "{stmts:?}");
}

#[test]
fn aarch32_unpredicated_store_stays_a_plain_store() {
    // Companion direction: an unconditional `str` must not pay for the
    // read-modify-write.
    let stmts = lift_aarch32("str", vec![reg("r0"), op("[r1]", OperandKind::Memory)]);
    assert!(
        !stmts.iter().any(|s| matches!(s, IrStmt::LoadMem { .. })),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_predicated_pop_lifts_rather_than_declining() {
    // `popne` appears in every conditional ARM epilogue.
    let stmts = lift_aarch32("popne", vec![op("{r4, pc}", OperandKind::Unknown)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
}

#[test]
fn aarch32_sxtb_sign_extends_the_low_byte() {
    let stmts = lift_aarch32("sxtb", vec![reg("r0"), reg("r1")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(format!("{stmts:?}").contains("SignExtend"), "{stmts:?}");
}

#[test]
fn aarch32_uxth_zero_extends_the_low_halfword() {
    let stmts = lift_aarch32("uxth", vec![reg("r0"), reg("r1")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(format!("{stmts:?}").contains("ZeroExtend"), "{stmts:?}");
}

#[test]
fn aarch32_mla_accumulates_the_fourth_operand() {
    // `mla Rd, Rn, Rm, Ra` = Ra + Rn*Rm. The accumulator is operand 3,
    // not operand 1 — reading it positionally like every other
    // 3-operand form would multiply the wrong pair.
    let stmts = lift_aarch32("mla", vec![reg("r0"), reg("r1"), reg("r2"), reg("r3")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    let body = format!("{stmts:?}");
    assert!(body.contains("Add(Var(Var { name: \"r3\""), "{stmts:?}");
    assert!(body.contains("Mul"), "{stmts:?}");
}

#[test]
fn aarch32_mls_subtracts_the_product_from_the_accumulator() {
    let stmts = lift_aarch32("mls", vec![reg("r0"), reg("r1"), reg("r2"), reg("r3")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        format!("{stmts:?}").contains("Sub(Var(Var { name: \"r3\""),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_umull_widens_before_multiplying() {
    // The whole point of the long multiply is the high half. Computing
    // the product at the source width would discard exactly the bits
    // `RdHi` receives.
    let stmts = lift_aarch32("umull", vec![reg("r0"), reg("r1"), reg("r2"), reg("r3")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.bits == 64)),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_smull_writes_both_halves_to_different_registers() {
    let stmts = lift_aarch32("smull", vec![reg("r0"), reg("r1"), reg("r2"), reg("r3")]);
    let written: Vec<&str> = stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::Assign { dst, .. } if dst.name == "r0" || dst.name == "r1" => {
                Some(dst.name.as_str())
            }
            _ => None,
        })
        .collect();
    assert!(
        written.contains(&"r0") && written.contains(&"r1"),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_rev_reverses_every_byte() {
    let stmts = lift_aarch32("rev", vec![reg("r0"), reg("r1")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    // Four bytes reversed is three nested Concats.
    assert_eq!(
        format!("{stmts:?}").matches("Concat").count(),
        3,
        "{stmts:?}"
    );
}

#[test]
fn aarch32_adc_adds_the_carry_flag() {
    let stmts = lift_aarch32("adc", vec![reg("r0"), reg("r1"), reg("r2")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Assign { src, .. } if format!("{src:?}").contains("CF"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_sbc_subtracts_the_borrow_not_the_carry() {
    // `sbc` subtracts `NOT C`, so the lowering must carry a `1 - CF`
    // term. Subtracting `CF` directly would be off by one whenever the
    // carry is clear — a wrong value, not a decline.
    let stmts = lift_aarch32("sbc", vec![reg("r0"), reg("r1"), reg("r2")]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    let assigns = format!("{stmts:?}");
    assert!(assigns.contains("CF"), "{stmts:?}");
    assert!(assigns.contains("value: 1"), "{stmts:?}");
}

#[test]
fn aarch32_adcs_leaves_carry_and_overflow_imprecise() {
    // N and Z come from the carry-inclusive result; C and V would need
    // a wider computation, so they are Unknown rather than a wrong
    // value borrowed from the plain add formula.
    let stmts = lift_aarch32("adcs", vec![reg("r0"), reg("r1"), reg("r2")]);
    let carry = stmts.iter().find_map(|s| match s {
        IrStmt::Assign { dst, src } if dst.name == "CF" => Some(format!("{src:?}")),
        _ => None,
    });
    assert_eq!(carry.as_deref(), Some("Unknown(\"\")"), "{stmts:?}");
}

#[test]
fn aarch32_movw_clears_the_top_half() {
    let stmts = lift_aarch32(
        "movw",
        vec![reg("r0"), op("0x1234", OperandKind::Immediate)],
    );
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts.iter().any(
            |s| matches!(s, IrStmt::Assign { src, .. } if format!("{src:?}").contains("ZeroExtend"))
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_movt_preserves_the_bottom_half() {
    // `movt` is a partial write. Overwriting the whole register would
    // discard the `movw` that almost always precedes it.
    let stmts = lift_aarch32(
        "movt",
        vec![reg("r0"), op("0x5678", OperandKind::Immediate)],
    );
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    let assigns = format!("{stmts:?}");
    assert!(assigns.contains("Concat"), "{stmts:?}");
    assert!(assigns.contains("\"r0\""), "{stmts:?}");
}

/// The store addresses a register-list transfer touches, in order.
fn aarch32_store_offsets(stmts: &[IrStmt]) -> Vec<String> {
    stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::StoreMem { address, .. } => Some(format!("{address:?}")),
            _ => None,
        })
        .collect()
}

#[test]
fn aarch32_stmdb_starts_a_full_word_below_the_base() {
    // Two registers, decrement-before: the run starts at base-8, which
    // is also what `push` computes.
    let stmts = lift_aarch32(
        "stmdb",
        vec![reg("r0!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let offsets = aarch32_store_offsets(&stmts);
    assert_eq!(offsets.len(), 2, "{stmts:?}");
    assert!(offsets[0].contains("4294967288"), "{offsets:?}");
}

#[test]
fn aarch32_stmia_starts_at_the_base() {
    let stmts = lift_aarch32(
        "stmia",
        vec![reg("r0!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let offsets = aarch32_store_offsets(&stmts);
    assert_eq!(offsets.len(), 2, "{stmts:?}");
    assert!(!offsets[0].contains("Add"), "{offsets:?}");
}

#[test]
fn aarch32_stmib_starts_one_word_above_the_base() {
    let stmts = lift_aarch32(
        "stmib",
        vec![reg("r0!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let offsets = aarch32_store_offsets(&stmts);
    assert!(offsets[0].contains("value: 4"), "{offsets:?}");
}

#[test]
fn aarch32_stmda_ends_at_the_base() {
    // Decrement-after: the *last* word lands on the base itself, so the
    // run starts at base-4 for two registers.
    let stmts = lift_aarch32(
        "stmda",
        vec![reg("r0!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let offsets = aarch32_store_offsets(&stmts);
    assert!(offsets[0].contains("4294967292"), "{offsets:?}");
}

#[test]
fn aarch32_stmfd_is_the_descending_stack_push() {
    // The stack synonyms are direction-relative: `stmfd` is `stmdb`.
    let full_descending = lift_aarch32(
        "stmfd",
        vec![reg("sp!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let decrement_before = lift_aarch32(
        "stmdb",
        vec![reg("sp!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    assert_eq!(
        aarch32_store_offsets(&full_descending),
        aarch32_store_offsets(&decrement_before)
    );
}

#[test]
fn aarch32_ldmfd_is_the_ascending_pop_not_the_descending_push() {
    // The same suffix means the opposite thing on a load: `ldmfd` pops,
    // so it is `ldmia`. Resolving it as `ldmdb` would walk memory the
    // wrong way — a wrong value, not a decline.
    let full_descending = lift_aarch32(
        "ldmfd",
        vec![reg("sp!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let increment_after = lift_aarch32(
        "ldmia",
        vec![reg("sp!"), op("{r1, r2}", OperandKind::Unknown)],
    );
    let loads = |stmts: &[IrStmt]| {
        stmts
            .iter()
            .filter_map(|s| match s {
                IrStmt::LoadMem { address, .. } => Some(format!("{address:?}")),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(loads(&full_descending), loads(&increment_after));
    assert!(!loads(&full_descending).is_empty());
}

#[test]
fn aarch32_ldrsb_sign_extends_the_loaded_byte() {
    // The whole difference from `ldrb` is the extension: a byte of
    // 0xff must reach the register as -1, not as 255.
    let stmts = lift_aarch32("ldrsb", vec![reg("r0"), op("[r1, 4]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts.iter().any(
            |s| matches!(s, IrStmt::Assign { src, .. } if format!("{src:?}").contains("SignExtend"))
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_ldrsh_sign_extends_the_loaded_halfword() {
    let stmts = lift_aarch32("ldrsh", vec![reg("r0"), op("[r1]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { bits: 16, .. })),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_ldrb_still_zero_extends() {
    // Companion direction: the unsigned form must not have picked up
    // the sign extension.
    let stmts = lift_aarch32("ldrb", vec![reg("r0"), op("[r1]", OperandKind::Memory)]);
    assert!(
        !stmts.iter().any(
            |s| matches!(s, IrStmt::Assign { src, .. } if format!("{src:?}").contains("SignExtend"))
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_ldrd_loads_two_consecutive_words() {
    // `ldrd r0, r1, [r2, 8]` — radare2 writes the memory operand third.
    let stmts = lift_aarch32(
        "ldrd",
        vec![reg("r0"), reg("r1"), op("[r2, 8]", OperandKind::Memory)],
    );
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert_eq!(
        stmts
            .iter()
            .filter(|s| matches!(s, IrStmt::LoadMem { bits: 32, .. }))
            .count(),
        2,
        "{stmts:?}"
    );
}

#[test]
fn aarch32_ldrd_puts_the_second_register_one_word_higher() {
    // Both words at the same address would be a wrong value rather than
    // a decline, so the offset is what this pins.
    let stmts = lift_aarch32(
        "ldrd",
        vec![reg("r0"), reg("r1"), op("[r2]", OperandKind::Memory)],
    );
    let addresses: Vec<String> = stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::LoadMem { address, .. } => Some(format!("{address:?}")),
            _ => None,
        })
        .collect();
    assert_eq!(addresses.len(), 2, "{stmts:?}");
    assert_ne!(addresses[0], addresses[1], "{stmts:?}");
    assert!(addresses[1].contains("value: 4"), "{addresses:?}");
}

#[test]
fn aarch32_strd_stores_two_consecutive_words() {
    let stmts = lift_aarch32(
        "strd",
        vec![reg("r0"), reg("r1"), op("[r2, 8]", OperandKind::Memory)],
    );
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert_eq!(
        stmts
            .iter()
            .filter(|s| matches!(s, IrStmt::StoreMem { bits: 32, .. }))
            .count(),
        2,
        "{stmts:?}"
    );
}

#[test]
fn aarch32_register_offset_load_addresses_base_plus_index() {
    // `ldr r0, [r1, r2]` — the index is a register, not an immediate.
    let stmts = lift_aarch32("ldr", vec![reg("r0"), op("[r1, r2]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts.iter().any(|s| matches!(
            s,
            IrStmt::LoadMem { address, .. }
                if format!("{address:?}").contains("r1") && format!("{address:?}").contains("r2")
        )),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_shifted_register_offset_scales_the_index() {
    // radare2 spells the shift `lsl 2`, with no `#` on the amount.
    let stmts = lift_aarch32(
        "ldr",
        vec![reg("r0"), op("[r1, r2, lsl 2]", OperandKind::Memory)],
    );
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { address, .. } if format!("{address:?}").contains("Shl"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_subtracted_register_offset_negates_the_index() {
    let stmts = lift_aarch32("ldr", vec![reg("r0"), op("[r1, -r2]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { address, .. } if format!("{address:?}").contains("Sub"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_register_pre_index_writes_back_the_index_register() {
    // `[r1, r2]!` adds `r2` to `r1` as well as addressing with it.
    let stmts = lift_aarch32("ldr", vec![reg("r0"), op("[r1, r2]!", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts.iter().any(|s| matches!(
            s,
            IrStmt::Assign { dst, src }
                if dst.name == "r1" && format!("{src:?}").contains("r2")
        )),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_rotated_register_offset_declines() {
    // `ror` has no IR node; composing one from two shifts and an `Or`
    // is a lowering this form does not earn.
    let stmts = lift_aarch32(
        "ldr",
        vec![reg("r0"), op("[r1, r2, ror 4]", OperandKind::Memory)],
    );
    assert!(!aarch32_lifts_cleanly(&stmts), "{stmts:?}");
}

#[test]
fn aarch32_store_register_offset_addresses_base_plus_index() {
    let stmts = lift_aarch32("str", vec![reg("r0"), op("[r1, r2]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::StoreMem { address, .. } if format!("{address:?}").contains("r2"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_vldr_pc_relative_literal_pool_resolves_its_address() {
    // A PC-relative literal load used to decline for want of a PC
    // value. The architecture fixes PC, so the address is now a
    // constant and the load reads free bytes instead of truncating the
    // slice — sound either way, but Complete rather than Unsound.
    let stmts = lift_aarch32("vldr", vec![reg("s0"), op("[pc, #8]", OperandKind::Memory)]);
    assert!(aarch32_lifts_cleanly(&stmts), "{stmts:?}");
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { address, .. } if !format!("{address:?}").contains("Var"))),
        "the literal address must be constant: {stmts:?}"
    );
}

#[test]
fn aarch32_thumb_pc_reads_one_word_ahead_and_word_aligned() {
    // Thumb fixes PC at address + 4, and a literal access reads
    // Align(PC, 4). At 0x1002 that is 0x1004, not 0x1006 — using the
    // ARM offset or skipping the alignment would give a wrong address
    // rather than a decline.
    let mut thumb = insn(
        0x1002,
        2,
        "ldr",
        vec![reg("r0"), op("[pc]", OperandKind::Memory)],
    );
    thumb.is_thumb = true;
    let stmts = lift_per_mnemonic(&thumb, Arch::Arm);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { address, .. } if format!("{address:?}").contains("4100"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_arm_pc_reads_two_instructions_ahead() {
    // ARM fixes PC at address + 8. `lift_aarch32` places the
    // instruction at 0x1000, so `[pc]` addresses 0x1008.
    let stmts = lift_aarch32("ldr", vec![reg("r0"), op("[pc]", OperandKind::Memory)]);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { address, .. } if format!("{address:?}").contains("4104"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_pc_relative_load_no_longer_reads_a_free_register() {
    // Before the PC model this lifted with `r15` as a free symbolic
    // input: sound, but the address could alias anything.
    let stmts = lift_aarch32("ldr", vec![reg("r0"), op("[pc, 8]", OperandKind::Memory)]);
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, IrStmt::LoadMem { address, .. } if format!("{address:?}").contains("r15"))),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_vmrs_flag_transfer_is_a_modelled_nop() {
    // `vmrs APSR_nzcv, FPSCR` is an identity in this model (vcmp wrote
    // NZCV directly), lifted as a Nop so the slice keeps walking past it
    // rather than declining.
    let stmts = lift_aarch32("vmrs", vec![reg("APSR_nzcv"), reg("FPSCR")]);
    assert_eq!(stmts, vec![IrStmt::Nop]);
}

#[test]
fn aarch32_vmrs_gpr_read_declines() {
    // `vmrs r0, FPSCR` reads the status word into a GPR — not modelled.
    let stmts = lift_aarch32("vmrs", vec![reg("r0"), reg("FPSCR")]);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. }))
    );
}

#[test]
fn aarch32_movs_sets_zero_flag_from_the_moved_value() {
    // `movs` is a flag-setting move: N/Z come from the value, C/V are
    // untouched. `movs r0, #0` must set ZF, unlike plain `mov`.
    let stmts = lift_aarch32("movs", vec![reg("r0"), op("0x0", OperandKind::Immediate)]);
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("movs must set ZF");
    assert!(matches!(
        zf,
        IrStmt::Assign {
            src: Expr::Eq(..),
            ..
        }
    ));
    // Plain `mov` sets no flags.
    let plain = lift_aarch32("mov", vec![reg("r0"), op("0x0", OperandKind::Immediate)]);
    assert!(
        !plain
            .iter()
            .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF")),
        "plain mov must not set ZF"
    );
}

#[test]
fn aarch32_movs_leaves_carry_and_overflow_untouched() {
    // MOVS updates N/Z only; C and V are unchanged, so the lifter must
    // not fabricate CF / OF assignments.
    let stmts = lift_aarch32("movs", vec![reg("r0"), reg("r1")]);
    for flag in ["CF", "OF"] {
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == flag)),
            "movs must not write {flag}"
        );
    }
}

#[test]
fn aarch32_thumb_two_operand_arith_expands_the_destination() {
    // `add r0, r1` is the Thumb narrow form of `add r0, r0, r1`. Both
    // must lift identically.
    let short = lift_aarch32("add", vec![reg("r0"), reg("r1")]);
    let full = lift_aarch32("add", vec![reg("r0"), reg("r0"), reg("r1")]);
    assert_eq!(format!("{short:?}"), format!("{full:?}"));
}

#[test]
fn aarch32_thumb_two_operand_subs_sets_flags_from_the_destination() {
    // `subs r0, r1` computes `r0 - r1` and sets flags; the flag terms
    // must read the pre-op `r0`, so ZF is present and non-Ite.
    let stmts = lift_aarch32("subs", vec![reg("r0"), reg("r1")]);
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("subs must set ZF");
    assert!(matches!(
        zf,
        IrStmt::Assign {
            src: Expr::Eq(..),
            ..
        }
    ));
}

#[test]
fn aarch32_thumb_wide_suffix_dispatches_as_the_base_mnemonic() {
    // `.w` is a Thumb-2 encoding-width hint, not semantics: `add.w`
    // must lift exactly as `add`, and `ldr.w` / `mov.w` must not
    // decline to `Unsupported`.
    let wide = lift_aarch32(
        "add.w",
        vec![reg("r0"), reg("r1"), op("0x4", OperandKind::Immediate)],
    );
    let base = lift_aarch32(
        "add",
        vec![reg("r0"), reg("r1"), op("0x4", OperandKind::Immediate)],
    );
    assert_eq!(format!("{wide:?}"), format!("{base:?}"));

    for mnem in ["ldr.w", "mov.w"] {
        let stmts = match mnem {
            "ldr.w" => lift_aarch32(mnem, vec![reg("r0"), op("[r1]", OperandKind::Memory)]),
            _ => lift_aarch32(mnem, vec![reg("r0"), op("0x1", OperandKind::Immediate)]),
        };
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Unsupported { .. })),
            "{mnem} should lift, got {stmts:?}"
        );
    }
}

#[test]
fn aarch32_lsls_sets_flags_and_is_not_a_conditional_lsl() {
    // `lsls` ends in the `ls` condition spelling, so a cond-suffix peel
    // that runs before the exact match would strip it to a supported
    // base `lsl` and mis-lift `lsls r0, r1, #2` as a *conditional* `lsl`
    // under LS — no flags, and the write wrapped in `Ite`. The
    // flag-setting arm must win: ZF is a direct comparison, not an Ite.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::Arm);
        ctx.lift_instruction(&insn(
            0x40_1000,
            4,
            "lsls",
            vec![
                op("r0", OperandKind::Register),
                op("r1", OperandKind::Register),
                op("0x2", OperandKind::Immediate),
            ],
        ));
        ctx.stmts
    };
    let zf = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "ZF"))
        .expect("lsls must set ZF (a conditional lsl would set no flags)");
    match zf {
        IrStmt::Assign {
            src: Expr::Eq(..), ..
        } => {}
        other => panic!("expected ZF := Eq(.., ..), got {other:?}"),
    }
}

#[test]
fn aarch32_bic_emits_and_not_pattern() {
    // `bic r0, r1, r2` should compute `r1 & ~r2`, encoded as
    // `r1 & (r2 ^ all_ones)`.
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::Arm);
        ctx.lift_instruction(&insn(
            0x40_1000,
            4,
            "bic",
            vec![
                op("r0", OperandKind::Register),
                op("r1", OperandKind::Register),
                op("r2", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let temp = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with("t_")))
        .expect("bic produces a temp assignment");
    let rendered = match temp {
        IrStmt::Assign { src, .. } => format!("{src}"),
        _ => unreachable!(),
    };
    assert!(
        rendered.contains('&') && rendered.contains('^'),
        "bic temp should be `r1 & (r2 ^ ones)`, got `{rendered}`"
    );
}

#[test]
fn aarch32_cmn_sets_flags_from_addition() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::Arm);
        ctx.lift_instruction(&insn(
            0x40_1000,
            4,
            "cmn",
            vec![
                op("r0", OperandKind::Register),
                op("r1", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let temp = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with("t_")))
        .expect("cmn produces a temp assignment");
    match temp {
        IrStmt::Assign { src, .. } => {
            let rendered = format!("{src}");
            assert!(
                rendered.contains('+'),
                "cmn temp should be `r0 + r1`, got `{rendered}`"
            );
        }
        _ => unreachable!(),
    }
    // cmn has no destination register write — only flag updates.
    assert!(
        !stmts.iter().any(
            |s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "r0" || dst.name == "r1")
        ),
        "cmn must not write a register destination",
    );
}

#[test]
fn aarch32_teq_sets_flags_from_xor() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::Arm);
        ctx.lift_instruction(&insn(
            0x40_1000,
            4,
            "teq",
            vec![
                op("r0", OperandKind::Register),
                op("r1", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let temp = stmts
        .iter()
        .find(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with("t_")))
        .expect("teq produces a temp assignment");
    match temp {
        IrStmt::Assign { src, .. } => {
            let rendered = format!("{src}");
            assert!(
                rendered.contains('^'),
                "teq temp should be `r0 ^ r1`, got `{rendered}`"
            );
        }
        _ => unreachable!(),
    }
}

#[test]
fn movsx_eax_ah_sign_extends_to_eax_then_zero_extends_to_rax() {
    let stmts = {
        let mut ctx = LiftCtx::new(Arch::X86_64);
        ctx.lift_instruction(&insn(
            0x40_1000,
            3,
            "movsx",
            vec![
                op("eax", OperandKind::Register),
                op("ah", OperandKind::Register),
            ],
        ));
        ctx.stmts
    };
    let assign = stmts.first().unwrap();
    match assign {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(dst.bits, 64);
            // Outer is the x86_64 zero-extension of the dword write;
            // inner is sign-extending ah (Extract bits 15:8) to 32 bits.
            match src {
                Expr::ZeroExtend {
                    src: inner_src,
                    to_bits,
                } => {
                    assert_eq!(*to_bits, 64);
                    match &**inner_src {
                        Expr::SignExtend {
                            src: ext_src,
                            to_bits: ext_to,
                        } => {
                            assert_eq!(*ext_to, 32);
                            assert_eq!(**ext_src, Expr::extract(Expr::var("rax", 64), 15, 8));
                        }
                        other => panic!("expected SignExtend, got {other:?}"),
                    }
                }
                other => panic!("expected outer ZeroExtend, got {other:?}"),
            }
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn cmp_dword_ptr_memory_loads_at_prefix_width_not_pointer_width() {
    // `cmp dword ptr [rax], 5` — the compare width comes from the
    // memory operand's `dword ptr` prefix (32), so the emitted
    // `LoadMem` reads 32 bits, not the 64-bit pointer default.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            3,
            "cmp",
            vec![
                op("dword ptr [rax]", OperandKind::Memory),
                op("5", OperandKind::Immediate),
            ],
        ),
        Arch::X86_64,
    );
    let load = stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::LoadMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .expect("cmp against a memory operand emits a LoadMem");
    assert_eq!(load, 32);
}

#[test]
fn segment_prefixed_memory_stays_opaque_not_absolute_load() {
    // `mov rax, qword fs:[0x30]` — a segment-relative access whose
    // true address is `fs_base + 0x30`. Modelling it as an absolute
    // load at `0x30` would alias with `[0x30]` and could fabricate a
    // verdict, so the operand stays opaque: no LoadMem is emitted.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            8,
            "mov",
            vec![
                op("rax", OperandKind::Register),
                op("qword fs:[0x30]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    assert!(
        !stmts.iter().any(|s| matches!(s, IrStmt::LoadMem { .. })),
        "segment-prefixed memory must not lower to a LoadMem: {stmts:?}"
    );
}

fn simd_dst_src<'a>(stmts: &'a [IrStmt], dst: &str) -> &'a Expr {
    stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::Assign { dst: d, src } if d.name == dst && d.bits == 512 => Some(src),
            _ => None,
        })
        .expect("expected a 512-bit SIMD parent assignment")
}

/// Read a SIMD register view: the low `bits` slice of the 512-bit parent.
fn simd_read(parent: &str, bits: u16) -> Expr {
    Expr::extract(Expr::var(parent, 512), bits - 1, 0)
}

/// Legacy-SSE write reconstruction: the value in the low `bits`, the
/// parent's prior contents preserved above.
fn preserve_upper(parent: &str, bits: u16, low: Expr) -> Expr {
    Expr::concat(Expr::extract(Expr::var(parent, 512), 511, bits), low)
}

#[test]
fn vpxor_same_register_lifts_to_zero_with_upper_bits_zeroed() {
    // `vpxor xmm0, xmm0, xmm0` — the SIMD zero idiom. The low 128 bits
    // are 0 independent of inputs; being VEX-encoded, the parent bits
    // above the view are zeroed.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "vpxor",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm0", OperandKind::Register),
                op("xmm0", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        Expr::zero_ext(Expr::konst(0, 128), 512)
    );
}

#[test]
fn movaps_copies_xmm_view_preserving_upper_parent_bits() {
    // `movaps xmm0, xmm1` — a 128-bit copy; legacy SSE preserves the
    // parent bits above the view.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "movaps",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper("zmm0", 128, simd_read("zmm1", 128))
    );
}

#[test]
fn movss_reg_reg_merges_the_low_lane_preserving_upper_parent_bits() {
    // `movss xmm0, xmm1` — a *scalar* move: only the low 32-bit lane is
    // written, every bit above it keeps its prior value. Copying the
    // whole 128-bit view (as `movaps` does) would clobber three lanes.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "movss",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper("zmm0", 32, Expr::extract(Expr::var("zmm1", 512), 31, 0))
    );
}

#[test]
fn movsd_reg_reg_merges_the_low_64_bits() {
    // `movsd xmm0, xmm1` — the double-precision lane width.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "movsd",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper("zmm0", 64, Expr::extract(Expr::var("zmm1", 512), 63, 0))
    );
}

#[test]
fn vmovss_three_operand_takes_the_upper_lanes_from_src1_and_zeroes_above_the_view() {
    // `vmovss xmm0, xmm1, xmm2` — the low lane comes from `xmm2`, the
    // rest of the 128-bit view from `xmm1`, and the VEX encoding zeroes
    // everything above the view.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "vmovss",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
                op("xmm2", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let merged = Expr::concat(
        Expr::extract(simd_read("zmm1", 128), 127, 32),
        Expr::extract(Expr::var("zmm2", 512), 31, 0),
    );
    assert_eq!(*simd_dst_src(&stmts, "zmm0"), Expr::zero_ext(merged, 512));
}

#[test]
fn vmovsd_three_operand_effect_does_not_read_its_destination() {
    // The VEX form overwrites the whole register, so the destination is
    // a def only — unlike the legacy 2-operand merge below.
    let i = insn(
        0x1000,
        4,
        "vmovsd",
        vec![
            op("xmm0", OperandKind::Register),
            op("xmm1", OperandKind::Register),
            op("xmm2", OperandKind::Register),
        ],
    );
    let e = crate::effect::analyze(&i, Arch::X86_64);
    assert_eq!(e.uses, vec!["zmm1", "zmm2"]);
}

#[test]
fn movsd_reg_reg_effect_reads_its_destination() {
    // The legacy scalar move preserves the lanes above the one it
    // writes, so dropping the destination use would let the slicer
    // treat the prior vector value as dead.
    let i = insn(
        0x1000,
        4,
        "movsd",
        vec![
            op("xmm0", OperandKind::Register),
            op("xmm1", OperandKind::Register),
        ],
    );
    let e = crate::effect::analyze(&i, Arch::X86_64);
    assert!(e.uses.contains(&"zmm0"), "{:?}", e.uses);
}

#[test]
fn movsd_string_form_is_not_claimed_by_the_scalar_move_handler() {
    // `movsd` also names the string instruction (opcode `A5`), which
    // radare2 spells `movsd dword [rdi], dword [rsi]`. It shares nothing
    // with the SSE scalar move, so it must keep falling through to the
    // ESIL ladder rather than being routed into the SIMD handlers.
    let i = insn(
        0x1000,
        1,
        "movsd",
        vec![
            op("dword [rdi]", OperandKind::Memory),
            op("dword [rsi]", OperandKind::Memory),
        ],
    );
    assert_eq!(
        crate::effect::analyze(&i, Arch::X86_64).kind,
        crate::effect::InstructionKind::Other
    );
}

// ---------------------------------------------------------------
// Sub-lane vector views.
//
// x86 and AArch64 put every vector view at bit 0 of its parent, so
// nothing there can catch a helper that ignores `layout.lo`. AArch32
// can: `d1` is the *upper* half of `v0` and `s3` its top quarter. These
// exercise the shared SIMD helpers directly, ahead of any AArch32
// vector handler, because a helper that indexed from bit 0 would read
// and write the wrong part of the register without failing loudly.
// ---------------------------------------------------------------

#[test]
fn aarch32_upper_half_register_reads_the_upper_half_of_its_parent() {
    // `d1` is bits [127:64] of v0, not [63:0].
    let mut ctx = LiftCtx::new(Arch::Arm);
    let value = ctx
        .simd_operand_value(&op("d1", OperandKind::Register), 64)
        .expect("d1 resolves to a vector view");
    assert_eq!(value, Expr::extract(Expr::var("v0", 128), 127, 64));
}

#[test]
fn aarch32_top_quarter_register_reads_the_top_quarter_of_its_parent() {
    // `s3` is bits [127:96] of v0.
    let mut ctx = LiftCtx::new(Arch::Arm);
    let value = ctx
        .simd_operand_value(&op("s3", OperandKind::Register), 32)
        .expect("s3 resolves to a vector view");
    assert_eq!(value, Expr::extract(Expr::var("v0", 128), 127, 96));
}

#[test]
fn aarch32_upper_half_write_preserves_the_lower_half() {
    // Writing `d1` must leave `d0` — the bits *below* it — standing.
    // Only a sub-lane view can catch a splice that assumes offset zero.
    let mut ctx = LiftCtx::new(Arch::Arm);
    assert!(ctx.write_simd_lane(&op("d1", OperandKind::Register), Expr::konst(0, 64), 64, 0));
    let stmts = ctx.stmts;
    assert_eq!(
        *simd_parent_src(&stmts, "v0"),
        Expr::concat(
            Expr::konst(0, 64),
            Expr::extract(Expr::var("v0", 128), 63, 0)
        )
    );
}

#[test]
fn aarch32_top_quarter_write_preserves_the_bits_on_both_sides() {
    // `s3` has neighbours above nothing and below plenty; the splice
    // has to reconstruct the parent from both directions.
    let mut ctx = LiftCtx::new(Arch::Arm);
    assert!(ctx.write_simd_lane(&op("s3", OperandKind::Register), Expr::konst(0, 32), 32, 0));
    let stmts = ctx.stmts;
    assert_eq!(
        *simd_parent_src(&stmts, "v0"),
        Expr::concat(
            Expr::konst(0, 32),
            Expr::extract(Expr::var("v0", 128), 95, 0)
        )
    );
}

#[test]
fn aarch32_middle_register_write_preserves_bits_above_and_below() {
    // `s1` sits at [63:32] of v0 — bits on both sides must survive.
    let mut ctx = LiftCtx::new(Arch::Arm);
    assert!(ctx.write_simd_lane(&op("s1", OperandKind::Register), Expr::konst(0, 32), 32, 0));
    let stmts = ctx.stmts;
    assert_eq!(
        *simd_parent_src(&stmts, "v0"),
        Expr::concat(
            Expr::concat(
                Expr::extract(Expr::var("v0", 128), 127, 64),
                Expr::konst(0, 32)
            ),
            Expr::extract(Expr::var("v0", 128), 31, 0)
        )
    );
}

#[test]
fn aarch32_gpr_alias_is_not_mistaken_for_a_vector_register() {
    // `v1` on AArch32 is the AAPCS alias for the general-purpose `r4`,
    // not a vector register. The SIMD helpers must decline it, or every
    // ARM callee-saved register would masquerade as a vector.
    let ctx = LiftCtx::new(Arch::Arm);
    assert!(!ctx.is_simd_register(&op("v1", OperandKind::Register)));
}

#[test]
fn arm_fpcr_write_is_recognised_through_its_operand() {
    // `msr` writes any system register; only the operand says which.
    // Matching on the mnemonic alone would either miss the FPCR write
    // or condemn every system-register write in the function.
    let write_fpcr = insn(
        0x1000,
        4,
        "msr",
        vec![
            op("fpcr", OperandKind::Register),
            op("x0", OperandKind::Register),
        ],
    );
    assert!(crate::lift::writes_rounding_control(
        &write_fpcr,
        Arch::Aarch64
    ));
}

#[test]
fn arm_unrelated_system_register_write_is_not_a_rounding_control_write() {
    let write_tpidr = insn(
        0x1000,
        4,
        "msr",
        vec![
            op("tpidr_el0", OperandKind::Register),
            op("x0", OperandKind::Register),
        ],
    );
    assert!(!crate::lift::writes_rounding_control(
        &write_tpidr,
        Arch::Aarch64
    ));
}

#[test]
fn aarch32_fpscr_write_is_recognised() {
    let write_fpscr = insn(
        0x1000,
        4,
        "vmsr",
        vec![
            op("fpscr", OperandKind::Register),
            op("r0", OperandKind::Register),
        ],
    );
    assert!(crate::lift::writes_rounding_control(
        &write_fpscr,
        Arch::Arm
    ));
}

/// The source of the assignment to a 128-bit vector parent.
fn simd_parent_src<'a>(stmts: &'a [IrStmt], parent: &str) -> &'a Expr {
    stmts
        .iter()
        .find_map(|s| match s {
            IrStmt::Assign { dst, src } if dst.name == parent && dst.bits == 128 => Some(src),
            _ => None,
        })
        .expect("expected a 128-bit vector parent assignment")
}

/// The `LoadMem` widths a lifting produced, in order.
fn load_widths(stmts: &[IrStmt]) -> Vec<u16> {
    stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::LoadMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .collect()
}

/// The `StoreMem` widths a lifting produced, in order.
fn store_widths(stmts: &[IrStmt]) -> Vec<u16> {
    stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::StoreMem { bits, .. } => Some(*bits),
            _ => None,
        })
        .collect()
}

#[test]
fn movss_load_names_its_temp_after_the_stack_slot() {
    // The load temp keeps the `stk_<base>_<off>` name so the analyst
    // alias (`var_4h`) still resolves downstream.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            5,
            "movss",
            vec![
                op("xmm0", OperandKind::Register),
                op("dword [rbp - 8]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    let names: Vec<&str> = stmts
        .iter()
        .filter_map(|s| match s {
            IrStmt::LoadMem { dst, .. } => Some(dst.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["stk_rbp_-8"], "{stmts:?}");
}

#[test]
fn movss_load_reads_the_lane_width_not_the_vector_view() {
    // `movss xmm0, dword [rbp - 8]` reads four bytes, not sixteen.
    // Loading the whole view would pull in three neighbouring lanes
    // that the instruction never touches.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            5,
            "movss",
            vec![
                op("xmm0", OperandKind::Register),
                op("dword [rbp - 8]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(load_widths(&stmts), vec![32], "{stmts:?}");
}

#[test]
fn movss_load_from_memory_zeroes_everything_above_the_lane() {
    // Per the SDM the load form zeroes the rest of the register, unlike
    // the register-to-register form, which merges. There is no prior
    // value to merge with.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            5,
            "movss",
            vec![
                op("xmm0", OperandKind::Register),
                op("dword [rbp - 8]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    assert!(
        matches!(simd_dst_src(&stmts, "zmm0"), Expr::ZeroExtend { .. }),
        "{stmts:?}"
    );
}

#[test]
fn movss_store_writes_only_the_lane() {
    // `movss dword [rbp - 8], xmm0` writes four bytes and leaves the
    // rest of the stack slot alone.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            5,
            "movss",
            vec![
                op("dword [rbp - 8]", OperandKind::Memory),
                op("xmm0", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(store_widths(&stmts), vec![32], "{stmts:?}");
}

#[test]
fn movaps_load_reads_the_whole_vector_view_in_one_load() {
    // `movaps xmm1, xmmword [rbp - 0x10]` is a single 16-byte load; the
    // size prefix radare2 attaches is what supplies the width.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "movaps",
            vec![
                op("xmm1", OperandKind::Register),
                op("xmmword [rbp - 0x10]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(load_widths(&stmts), vec![128], "{stmts:?}");
}

#[test]
fn movaps_store_writes_the_whole_vector_view() {
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "movaps",
            vec![
                op("xmmword [rbp - 0x10]", OperandKind::Memory),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(store_widths(&stmts), vec![128], "{stmts:?}");
}

#[test]
fn packed_arithmetic_against_memory_loads_the_operand_once() {
    // `addps xmm0, xmmword [rsi]` applies four lane additions but must
    // read memory a single time — one load per lane would quadruple the
    // byte-store chain the encoder walks for no gain.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "addps",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmmword [rsi]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(load_widths(&stmts), vec![128], "{stmts:?}");
}

#[test]
fn simd_memory_the_lifter_cannot_address_still_analyzes_as_other() {
    // A segment-prefixed operand has no address expression, so the
    // effect table must keep demoting it — otherwise the slicer would
    // retain an instruction whose store the lifter drops, and a later
    // load would read a stale value.
    let i = insn(
        0x1000,
        9,
        "movaps",
        vec![
            op("xmm0", OperandKind::Register),
            op("xmmword fs:[0x30]", OperandKind::Memory),
        ],
    );
    assert_eq!(
        crate::effect::analyze(&i, Arch::X86_64).kind,
        crate::effect::InstructionKind::Other
    );
}

#[test]
fn simd_memory_without_a_size_prefix_is_declined() {
    // Nothing says how many bytes to read, and guessing would silently
    // load the wrong width.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "movaps",
            vec![
                op("xmm0", OperandKind::Register),
                op("[rsi]", OperandKind::Memory),
            ],
        ),
        Arch::X86_64,
    );
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{stmts:?}"
    );
}

#[test]
fn pxor_distinct_registers_lifts_to_xor_preserving_upper() {
    // `pxor xmm0, xmm1` (2-operand RMW) — low 128 = `xmm0 ^ xmm1`,
    // upper parent bits preserved (legacy SSE).
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "pxor",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper(
            "zmm0",
            128,
            Expr::bv_xor(simd_read("zmm0", 128), simd_read("zmm1", 128))
        )
    );
}

#[test]
fn pandn_distinct_registers_lifts_to_andnot_preserving_upper() {
    // `pandn xmm0, xmm1` (2-operand RMW) — low 128 = `(~xmm0) & xmm1`.
    // Regression: `pandn`/`vpandn` was classified `Simd` by the effect
    // table but had no lifter arm, so it emitted a silent `Unsupported`
    // no-op — a stale-def fabrication once an xmm→GPR bridge exists.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "pandn",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper(
            "zmm0",
            128,
            Expr::bv_and(
                Expr::bv_xor(simd_read("zmm0", 128), Expr::konst(u128::MAX, 128)),
                simd_read("zmm1", 128)
            )
        )
    );
}

#[test]
fn addss_lifts_to_scalar_fp_add_on_low_lane() {
    use r2smt_ir::expr::RoundingMode;
    // `addss xmm0, xmm1` — the low 32-bit lane of each register is
    // reinterpreted as an IEEE single, added (round-nearest-even), and
    // written back to the low lane with the upper parent bits preserved.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "addss",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let lane = |p: &str| Expr::bv_to_fp(Expr::extract(Expr::var(p, 512), 31, 0), 8, 24);
    let result = Expr::fp_to_ieee_bv(Expr::fadd(
        lane("zmm0"),
        lane("zmm1"),
        RoundingMode::NearestTiesEven,
    ));
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        Expr::concat(Expr::extract(Expr::var("zmm0", 512), 511, 32), result)
    );
}

/// Lift a register-to-register form of `mnemonic` on `xmm0, xmm1`.
fn lift_xmm_pair(mnemonic: &str) -> Vec<IrStmt> {
    lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            mnemonic,
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    )
}

/// The source assigned to flag `name`, or `None` if it is not defined.
fn flag_src<'a>(stmts: &'a [IrStmt], name: &str) -> Option<&'a Expr> {
    stmts.iter().find_map(|s| match s {
        IrStmt::Assign { dst, src } if dst.name == name => Some(src),
        _ => None,
    })
}

#[test]
fn ucomiss_lifts_parity_flag_to_the_exact_unordered_predicate() {
    // PF after a scalar FP compare *is* the unordered predicate, so it
    // lifts exactly — unlike the integer path, where PF degrades to
    // `Unknown`. This is what makes the `ucomiss` + `jp` NaN check that
    // compilers emit resolvable.
    let stmts = lift_xmm_pair("ucomiss");
    let lane = |p: &str| Expr::bv_to_fp(Expr::extract(Expr::var(p, 512), 31, 0), 8, 24);
    assert_eq!(
        flag_src(&stmts, "PF"),
        Some(&Expr::bool_or(
            Expr::fisnan(lane("zmm0")),
            Expr::fisnan(lane("zmm1"))
        ))
    );
}

#[test]
fn ucomiss_lifts_carry_flag_to_less_than_or_unordered() {
    let stmts = lift_xmm_pair("ucomiss");
    let lane = |p: &str| Expr::bv_to_fp(Expr::extract(Expr::var(p, 512), 31, 0), 8, 24);
    assert_eq!(
        flag_src(&stmts, "CF"),
        Some(&Expr::bool_or(
            Expr::bool_or(Expr::fisnan(lane("zmm0")), Expr::fisnan(lane("zmm1"))),
            Expr::flt(lane("zmm0"), lane("zmm1"))
        ))
    );
}

#[test]
fn scalar_fp_compare_defines_flags_without_touching_its_destination() {
    // `comiss` writes EFLAGS only. Claiming its destination register as
    // a def would fabricate a value for a register the instruction
    // never writes.
    for m in ["comiss", "ucomiss", "comisd", "ucomisd"] {
        let stmts = lift_xmm_pair(m);
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with("zmm"))),
            "{m}: defined a vector register"
        );
        for flag in ["ZF", "CF", "PF"] {
            assert!(flag_src(&stmts, flag).is_some(), "{m}: {flag} undefined");
        }
    }
}

#[test]
fn scalar_fp_compare_effect_reads_both_operands_and_writes_flags() {
    for m in ["comiss", "ucomiss", "comisd", "ucomisd"] {
        let i = insn(
            0x1000,
            4,
            m,
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        );
        let e = crate::effect::analyze(&i, Arch::X86_64);
        assert!(e.defines_flags, "{m}: does not define flags");
        assert!(e.defs.is_empty(), "{m}: claims a register def");
        assert_eq!(e.uses.len(), 2, "{m}: does not read both operands");
    }
}

#[test]
fn cvtsi2ss_converts_a_signed_register_into_the_low_lane() {
    use r2smt_ir::expr::RoundingMode;
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "cvtsi2ss",
            vec![
                op("xmm0", OperandKind::Register),
                op("eax", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let converted = Expr::fp_to_ieee_bv(Expr::sbv_to_fp(
        Expr::extract(Expr::var("rax", 64), 31, 0),
        RoundingMode::NearestTiesEven,
        8,
        24,
    ));
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        Expr::concat(Expr::extract(Expr::var("zmm0", 512), 511, 32), converted)
    );
}

#[test]
fn cvttss2si_truncates_toward_zero_into_the_destination_register() {
    use r2smt_ir::expr::RoundingMode;
    // The truncating form carries its rounding mode in the opcode, so
    // it is round-toward-zero regardless of MXCSR.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "cvttss2si",
            vec![
                op("eax", OperandKind::Register),
                op("xmm0", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let lane = Expr::bv_to_fp(Expr::extract(Expr::var("zmm0", 512), 31, 0), 8, 24);
    assert_eq!(
        find_assign(&stmts, "rax").and_then(|s| match s {
            IrStmt::Assign { src, .. } => Some(src),
            _ => None,
        }),
        Some(&Expr::zero_ext(
            Expr::fp_to_sbv(lane, RoundingMode::TowardZero, 32),
            64
        ))
    );
}

#[test]
fn cvtss2si_without_truncation_uses_the_default_rounding_mode() {
    use r2smt_ir::expr::RoundingMode;
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "cvtss2si",
            vec![
                op("eax", OperandKind::Register),
                op("xmm0", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let rendered = format!("{:?}", find_assign(&stmts, "rax").expect("rax defined"));
    assert!(
        rendered.contains(&format!("{:?}", RoundingMode::NearestTiesEven)),
        "{rendered}"
    );
}

#[test]
fn integer_to_float_conversion_reads_its_destination_register() {
    // `cvtsi2ss` preserves the lanes above the one it writes, so the
    // destination is both a def and a use. Dropping the use would let
    // the slicer treat the prior vector value as dead.
    let i = insn(
        0x1000,
        4,
        "cvtsi2ss",
        vec![
            op("xmm0", OperandKind::Register),
            op("eax", OperandKind::Register),
        ],
    );
    let e = crate::effect::analyze(&i, Arch::X86_64);
    assert!(e.uses.contains(&"zmm0"), "{:?}", e.uses);
}

#[test]
fn float_to_integer_conversion_does_not_read_its_destination_register() {
    let i = insn(
        0x1000,
        4,
        "cvttss2si",
        vec![
            op("eax", OperandKind::Register),
            op("xmm0", OperandKind::Register),
        ],
    );
    let e = crate::effect::analyze(&i, Arch::X86_64);
    assert_eq!(e.defs, vec!["rax"]);
    assert_eq!(e.uses, vec!["zmm0"]);
}

#[test]
fn cvtss2sd_widens_the_low_lane_to_the_double_sort() {
    use r2smt_ir::expr::RoundingMode;
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "cvtss2sd",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let converted = Expr::fp_to_ieee_bv(Expr::fp_to_fp(
        Expr::bv_to_fp(Expr::extract(Expr::var("zmm1", 512), 31, 0), 8, 24),
        RoundingMode::NearestTiesEven,
        11,
        53,
    ));
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        Expr::concat(Expr::extract(Expr::var("zmm0", 512), 511, 64), converted)
    );
}

#[test]
fn cvtsd2ss_narrows_the_low_lane_to_the_single_sort() {
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "cvtsd2ss",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    // Narrowing reads a double lane and produces a single: the source
    // sort must come from the mnemonic, not the destination width.
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert!(rendered.contains("bits: 11"), "{rendered}");
    assert!(rendered.contains("ebits: 8"), "{rendered}");
}

/// The IEEE float of lane `index` of `parent` at `lane_bits` width.
fn fp_lane(parent: &str, lane_bits: u16, index: u16) -> Expr {
    let lo = lane_bits * index;
    let (ebits, sbits) = if lane_bits == 32 { (8, 24) } else { (11, 53) };
    Expr::bv_to_fp(
        Expr::extract(Expr::var(parent, 512), lo + lane_bits - 1, lo),
        ebits,
        sbits,
    )
}

#[test]
fn addps_applies_the_lane_operation_to_all_four_single_lanes() {
    use r2smt_ir::expr::RoundingMode;
    let stmts = lift_xmm_pair("addps");
    let lane = |i: u16| {
        Expr::fp_to_ieee_bv(Expr::fadd(
            fp_lane("zmm0", 32, i),
            fp_lane("zmm1", 32, i),
            RoundingMode::NearestTiesEven,
        ))
    };
    let packed = Expr::concat(
        lane(3),
        Expr::concat(lane(2), Expr::concat(lane(1), lane(0))),
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper("zmm0", 128, packed)
    );
}

#[test]
fn addpd_uses_two_double_lanes_not_four_single_ones() {
    // The lane width comes from the mnemonic suffix, so `pd` must halve
    // the lane count relative to `ps` at the same view width.
    let stmts = lift_xmm_pair("addpd");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("FAdd").count(), 2, "{rendered}");
}

#[test]
fn vaddps_on_ymm_covers_eight_lanes_and_zeroes_the_upper_parent_bits() {
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "vaddps",
            vec![
                op("ymm0", OperandKind::Register),
                op("ymm1", OperandKind::Register),
                op("ymm2", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    let src = simd_dst_src(&stmts, "zmm0");
    let rendered = format!("{src:?}");
    assert_eq!(rendered.matches("FAdd").count(), 8, "{rendered}");
    // A VEX write zeroes above the view rather than preserving it.
    assert!(
        matches!(src, Expr::ZeroExtend { to_bits: 512, .. }),
        "{rendered}"
    );
    // The 3-operand form reads its two explicit sources, not the dest.
    assert!(!rendered.contains("zmm0"), "{rendered}");
}

#[test]
fn maxps_selects_the_second_operand_rather_than_computing_an_fp_max() {
    // Intel: `IF SRC1 > SRC2 THEN DEST := SRC1 ELSE DEST := SRC2`. The
    // comparison is false when either operand is NaN, so SRC2 wins on
    // unordered *and* on equality. Modelling this as SMT-LIB `fp.max`
    // would be wrong on exactly those cases, so the lifter must emit a
    // select over the raw lane bits with a `<` guard.
    let stmts = lift_xmm_pair("maxps");
    let lane0 = |parent: &str| Expr::extract(Expr::var(parent, 512), 31, 0);
    let expected = Expr::Ite {
        cond: Box::new(Expr::flt(fp_lane("zmm1", 32, 0), fp_lane("zmm0", 32, 0))),
        then_expr: Box::new(lane0("zmm0")),
        else_expr: Box::new(lane0("zmm1")),
    };
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert!(rendered.contains(&format!("{expected:?}")), "{rendered}");
}

#[test]
fn minps_guards_with_the_operands_in_the_opposite_order_to_maxps() {
    let stmts = lift_xmm_pair("minps");
    let expected_cond = Expr::flt(fp_lane("zmm0", 32, 0), fp_lane("zmm1", 32, 0));
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert!(
        rendered.contains(&format!("{expected_cond:?}")),
        "{rendered}"
    );
}

#[test]
fn maxss_writes_only_the_low_lane_and_preserves_the_rest() {
    let stmts = lift_xmm_pair("maxss");
    let src = simd_dst_src(&stmts, "zmm0");
    let rendered = format!("{src:?}");
    assert_eq!(rendered.matches("FLt").count(), 1, "{rendered}");
    assert!(rendered.contains("hi: 511"), "{rendered}");
}

#[test]
fn sqrtps_takes_the_root_of_every_lane() {
    let stmts = lift_xmm_pair("sqrtps");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("FSqrt").count(), 4, "{rendered}");
    // Unary: all four roots read the source. The destination appears
    // exactly once, in the preserved-upper slice of the legacy-SSE
    // write — never as an operand of a root.
    assert_eq!(rendered.matches("\"zmm1\"").count(), 4, "{rendered}");
    assert_eq!(rendered.matches("\"zmm0\"").count(), 1, "{rendered}");
}

#[test]
fn sqrtss_roots_only_the_low_lane_and_preserves_the_rest() {
    let stmts = lift_xmm_pair("sqrtss");
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        preserve_upper(
            "zmm0",
            32,
            Expr::fp_to_ieee_bv(Expr::fsqrt(
                fp_lane("zmm1", 32, 0),
                r2smt_ir::expr::RoundingMode::NearestTiesEven
            ))
        )
    );
}

#[test]
fn cmpeqps_writes_an_all_ones_mask_per_lane() {
    let stmts = lift_xmm_pair("cmpeqps");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    // One predicate and one mask select per 32-bit lane of the view.
    assert_eq!(rendered.matches("FEq").count(), 4, "{rendered}");
    assert_eq!(rendered.matches("4294967295").count(), 4, "{rendered}");
}

#[test]
fn cmpneqps_is_the_negation_of_the_equality_predicate() {
    // The `n`-prefixed predicates are the unordered variants precisely
    // because they negate a comparison that is false on NaN.
    let stmts = lift_xmm_pair("cmpneqps");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("BoolNot").count(), 4, "{rendered}");
    assert_eq!(rendered.matches("FEq").count(), 4, "{rendered}");
}

#[test]
fn cmpunordpd_tests_both_operands_for_nan_over_two_lanes() {
    let stmts = lift_xmm_pair("cmpunordpd");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("FIsNaN").count(), 4, "{rendered}");
    assert!(!rendered.contains("BoolNot"), "{rendered}");
}

#[test]
fn cmpordps_negates_the_unordered_predicate() {
    let stmts = lift_xmm_pair("cmpordps");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("BoolNot").count(), 4, "{rendered}");
    assert_eq!(rendered.matches("FIsNaN").count(), 8, "{rendered}");
}

#[test]
fn cmpltss_masks_only_the_low_lane() {
    let stmts = lift_xmm_pair("cmpltss");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("FLt").count(), 1, "{rendered}");
    assert!(rendered.contains("hi: 511"), "{rendered}");
}

#[test]
fn plain_integer_cmp_is_not_parsed_as_a_float_compare() {
    // `cmp` and the string-compare mnemonics share the prefix but carry
    // no predicate, so they must not be captured by the compare family.
    for m in ["cmp", "cmpsd", "cmpsb", "cmpxchg"] {
        assert!(
            !crate::lift::is_fp_compare_mnemonic(m),
            "{m}: wrongly parsed as a floating-point compare"
        );
    }
}

/// `[ldmxcsr] ; addss ; ucomiss ; jp` — the shape where a reprogrammed
/// rounding mode is invisible to the backward walk.
fn rounding_program(with_ldmxcsr: bool) -> Program {
    let mut insns = Vec::new();
    if with_ldmxcsr {
        insns.push(insn(
            0x40_0ff8,
            7,
            "ldmxcsr",
            vec![op("[rip + 0x1000]", OperandKind::Memory)],
        ));
    }
    insns.push(insn(
        0x40_1000,
        4,
        "addss",
        vec![
            op("xmm0", OperandKind::Register),
            op("xmm1", OperandKind::Register),
        ],
    ));
    insns.push(insn(
        0x40_1004,
        4,
        "ucomiss",
        vec![
            op("xmm0", OperandKind::Register),
            op("xmm2", OperandKind::Register),
        ],
    ));
    insns.push(insn(
        0x40_1008,
        6,
        "jp",
        vec![op("0x401080", OperandKind::Immediate)],
    ));
    one_block_program(insns)
}

fn first_slice_status(program: &Program) -> crate::slice::SliceStatus {
    let candidates = collect_branches(program);
    let cand = candidates.first().expect("a branch");
    slice_branch(
        cand,
        &program.functions[0],
        &SliceLimits::default(),
        program.arch,
    )
    .status
}

#[test]
fn floating_point_slice_truncates_when_the_function_reprograms_mxcsr() {
    // The backward walk stops once the live set is satisfied, so it
    // never reaches the `ldmxcsr`. Without the guard this slice would
    // report Complete while assuming a rounding mode the program had
    // replaced.
    let status = first_slice_status(&rounding_program(true));
    assert!(
        matches!(status, crate::slice::SliceStatus::Truncated { .. }),
        "{status:?}"
    );
}

#[test]
fn floating_point_slice_stays_complete_without_an_mxcsr_write() {
    let status = first_slice_status(&rounding_program(false));
    assert!(
        matches!(status, crate::slice::SliceStatus::Complete),
        "{status:?}"
    );
}

#[test]
fn rounding_insensitive_floating_point_survives_an_mxcsr_write() {
    // `maxss` selects an operand, `cvttss2si` carries its rounding mode
    // in the opcode, and the scalar moves transfer a bit pattern without
    // computing anything, so none of them depends on MXCSR — guarding
    // them would cost precision for nothing.
    for m in [
        "maxss",
        "minps",
        "cvttss2si",
        "cmpeqps",
        "ucomiss",
        "movss",
        "movsd",
    ] {
        assert!(
            !crate::lift::pins_rounding_mode(&insn(0, 1, m, vec![]), Arch::X86_64),
            "{m}: wrongly treated as rounding-mode dependent"
        );
    }
    for m in ["addss", "mulpd", "sqrtss", "cvtsi2ss", "cvtss2sd"] {
        assert!(
            crate::lift::pins_rounding_mode(&insn(0, 1, m, vec![]), Arch::X86_64),
            "{m}: should be guarded"
        );
    }
}

#[test]
fn vcvtph2ps_widens_four_half_lanes_into_single_lanes() {
    let stmts = lift_xmm_pair("vcvtph2ps");
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    // Four conversions, each reading a 16-bit lane of the source.
    assert_eq!(rendered.matches("FpToFp").count(), 4, "{rendered}");
    assert!(rendered.contains("ebits: 5"), "{rendered}");
    assert!(rendered.contains("ebits: 8"), "{rendered}");
}

#[test]
fn vcvtps2ph_declines_when_the_immediate_defers_to_mxcsr() {
    // Bit 2 of the immediate means "round per MXCSR", which this lifter
    // cannot pin — so it declines instead of assuming a mode.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            5,
            "vcvtps2ph",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
                op("4", OperandKind::Immediate),
            ],
        ),
        Arch::X86_64,
    );
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{stmts:?}"
    );
}

#[test]
fn vcvtps2ph_narrows_with_an_explicit_rounding_immediate() {
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            5,
            "vcvtps2ph",
            vec![
                op("xmm0", OperandKind::Register),
                op("xmm1", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        ),
        Arch::X86_64,
    );
    let rendered = format!("{:?}", simd_dst_src(&stmts, "zmm0"));
    assert_eq!(rendered.matches("FpToFp").count(), 4, "{rendered}");
    // The VEX write zeroes everything above the 64 bits of halves.
    assert!(rendered.contains("ZeroExtend"), "{rendered}");
}

#[test]
fn aarch32_families_are_covered_by_both_effect_and_lifter() {
    // Parity guard for the AArch32 seam: a mnemonic the effect table
    // keeps (any kind other than `Other`, which truncates) must be
    // modelled by the lifter, or the slicer retains an instruction that
    // lowers to `Unsupported` and the vector parent is left a stale def.
    // Branch mnemonics are excluded — they are lowered by the branch
    // machinery, not this per-mnemonic dispatch.
    let cases: &[(&str, &[&str])] = &[
        ("add", &["r0", "r1", "r2"]),
        ("add.w", &["r0", "r1", "r2"]),
        ("movs", &["r0", "0x0"]),
        ("vmrs", &["APSR_nzcv", "FPSCR"]),
        ("vldr", &["s0", "[r0]"]),
        ("vstr", &["s0", "[r0]"]),
        ("vpush", &["{d8, d9}"]),
        ("vpop", &["{d0-d3}"]),
        ("vadd.i32", &["d0", "d1", "d2"]),
        ("vpadd.f32", &["d0", "d2", "d4"]),
        ("vpmax.f32", &["d0", "d2", "d4"]),
    ];
    for (mnem, operands) in cases {
        let ops: Vec<Operand> = operands
            .iter()
            .map(|o| {
                let kind = if o.starts_with('[') {
                    OperandKind::Memory
                } else if o.starts_with('{') {
                    OperandKind::Unknown
                } else if o.starts_with("0x") {
                    OperandKind::Immediate
                } else {
                    OperandKind::Register
                };
                op(o, kind)
            })
            .collect();
        let i = insn(0x1000, 4, mnem, ops);
        assert_ne!(
            crate::effect::analyze(&i, Arch::Arm).kind,
            crate::effect::InstructionKind::Other,
            "{mnem}: effect table declined a modelled mnemonic"
        );
        let stmts = lift_per_mnemonic(&i, Arch::Arm);
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Unsupported { .. })),
            "{mnem}: lifter emitted Unsupported: {stmts:?}"
        );
    }
}

#[test]
fn every_simd_mnemonic_is_covered_by_both_effect_and_lifter() {
    // Parity guard: the effect table keeps an instruction iff the lifter
    // models it. A mnemonic classified `Simd` by `analyze` but absent
    // from the lifter (the historical `pandn` bug) leaves the vector
    // parent undefined — a stale-def fabrication. Assert both directions
    // stay in lockstep for every SIMD mnemonic.
    let mnemonics = [
        "movaps", "movups", "movapd", "movupd", "movdqa", "movdqu", "vmovaps", "vmovups",
        "vmovapd", "vmovupd", "vmovdqa", "vmovdqu", "pxor", "vpxor", "pand", "vpand", "por",
        "vpor", "pandn", "vpandn", "addss", "subss", "mulss", "divss", "addsd", "subsd", "mulsd",
        "divsd", "addps", "subps", "mulps", "divps", "addpd", "subpd", "mulpd", "divpd", "maxps",
        "minps", "maxpd", "minpd", "maxss", "minss", "maxsd", "minsd", "sqrtps", "sqrtpd",
        "sqrtss", "sqrtsd", "movss", "movsd", "pcmpeqb", "pcmpeqw", "pcmpeqd", "pcmpeqq",
        "pcmpgtb", "pcmpgtw", "pcmpgtd", "pcmpgtq", "vpcmpeqb", "vpcmpeqd", "vpcmpgtb", "vpcmpgtd",
        "paddb", "paddw", "paddd", "paddq", "psubb", "psubw", "psubd", "psubq", "pmullw", "pmulld",
        "vpaddd", "vpsubb",
    ];
    // The vector-to-general transfers need realistic operands: their
    // destination is a GPR, not a vector register.
    let transfers = [
        ("pmovmskb", "eax", "xmm1"),
        ("movd", "eax", "xmm1"),
        ("movq", "xmm0", "xmm1"),
        ("pslldq", "xmm0", "4"),
        ("psrldq", "xmm0", "4"),
        ("psllw", "xmm0", "3"),
        ("psrld", "xmm0", "3"),
        ("psraw", "xmm0", "3"),
    ];
    let cases = mnemonics
        .iter()
        .map(|m| (*m, "xmm0", "xmm1"))
        .chain(transfers);
    for (m, dst, src) in cases {
        let i = insn(
            0x1000,
            4,
            m,
            vec![
                op(dst, OperandKind::Register),
                op(
                    src,
                    if src.starts_with(|c: char| c.is_ascii_digit()) {
                        OperandKind::Immediate
                    } else {
                        OperandKind::Register
                    },
                ),
            ],
        );
        assert!(
            is_x86_simd_instruction(&i),
            "{m}: not recognised by is_x86_simd_instruction"
        );
        assert_eq!(
            crate::effect::analyze(&i, Arch::X86_64).kind,
            crate::effect::InstructionKind::Simd,
            "{m}: effect table does not classify it as Simd"
        );
        // The invariant that matters: whatever the effect table claims
        // the instruction defines, the lifter must actually define.
        // A mnemonic the slicer retains but no handler writes leaves
        // its destination undefined, and a later read binds to a stale
        // value — the `pandn` bug, and the shape of the ARM def/use
        // gaps that fabricated verdicts.
        let stmts = lift_per_mnemonic(&i, Arch::X86_64);
        for def in crate::effect::analyze(&i, Arch::X86_64).defs {
            assert!(
                stmts
                    .iter()
                    .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == def)),
                "{m}: effect table claims it defines {def}, lifter does not"
            );
        }
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Unsupported { .. })),
            "{m}: lifter emitted Unsupported"
        );
    }
}

#[test]
fn vpxor_ymm_lifts_to_256bit_xor_with_upper_bits_zeroed() {
    // `vpxor ymm0, ymm1, ymm2` (3-operand VEX at 256-bit view) —
    // `ymm0 := ymm1 ^ ymm2`, parent bits above 256 zeroed.
    let stmts = lift_per_mnemonic(
        &insn(
            0x1000,
            4,
            "vpxor",
            vec![
                op("ymm0", OperandKind::Register),
                op("ymm1", OperandKind::Register),
                op("ymm2", OperandKind::Register),
            ],
        ),
        Arch::X86_64,
    );
    assert_eq!(
        *simd_dst_src(&stmts, "zmm0"),
        Expr::zero_ext(
            Expr::bv_xor(simd_read("zmm1", 256), simd_read("zmm2", 256)),
            512
        )
    );
}

// ---------------------------------------------------------------
// AArch64 scalar floating point.
// ---------------------------------------------------------------

/// Lift one `AArch64` instruction through the per-mnemonic dispatch.
fn lift_aarch64(mnemonic: &str, operands: Vec<Operand>) -> Vec<IrStmt> {
    lift_per_mnemonic(&insn(0x1000, 4, mnemonic, operands), Arch::Aarch64)
}

fn reg(name: &str) -> Operand {
    op(name, OperandKind::Register)
}

#[test]
fn aarch64_fadd_single_operates_on_the_32bit_lane_of_the_vector_parent() {
    let stmts = lift_aarch64("fadd", vec![reg("s0"), reg("s1"), reg("s2")]);
    let lane = |p: &str| Expr::bv_to_fp(Expr::extract(Expr::var(p, 128), 31, 0), 8, 24);
    let result = Expr::fp_to_ieee_bv(Expr::fadd(
        lane("v1"),
        lane("v2"),
        r2smt_ir::expr::RoundingMode::NearestTiesEven,
    ));
    assert_eq!(
        *simd_parent_src(&stmts, "v0"),
        Expr::zero_ext(result, 128),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_fadd_double_uses_the_64bit_sort() {
    let stmts = lift_aarch64("fadd", vec![reg("d0"), reg("d1"), reg("d2")]);
    let rendered = format!("{:?}", simd_parent_src(&stmts, "v0"));
    assert!(rendered.contains("ebits: 11"), "{rendered}");
}

#[test]
fn aarch64_scalar_write_zeroes_the_rest_of_the_vector_register() {
    // Unlike legacy SSE, which merges, an AArch64 scalar write clears
    // everything above the lane it writes.
    let stmts = lift_aarch64("fmul", vec![reg("s0"), reg("s1"), reg("s2")]);
    assert!(
        matches!(simd_parent_src(&stmts, "v0"), Expr::ZeroExtend { .. }),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_fneg_flips_only_the_sign_bit() {
    let stmts = lift_aarch64("fneg", vec![reg("s0"), reg("s1")]);
    let expected = Expr::bv_xor(
        Expr::extract(Expr::var("v1", 128), 31, 0),
        Expr::konst(1 << 31, 32),
    );
    assert_eq!(
        *simd_parent_src(&stmts, "v0"),
        Expr::zero_ext(expected, 128),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_fcmp_marks_unordered_in_the_overflow_flag() {
    // `b.vs` after a compare means "unordered" on AArch64, and it
    // lowers to `OF == 1`, so the unordered predicate has to land
    // there rather than being zeroed.
    let stmts = lift_aarch64("fcmp", vec![reg("s0"), reg("s1")]);
    let rendered = format!("{:?}", flag_src(&stmts, "OF").expect("OF defined"));
    assert!(rendered.contains("FIsNaN"), "{rendered}");
}

#[test]
fn aarch64_fcmp_equality_flag_is_false_when_unordered() {
    // ZF is *ordered* equality: `b.eq` must not fire on NaN.
    let stmts = lift_aarch64("fcmp", vec![reg("s0"), reg("s1")]);
    let rendered = format!("{:?}", flag_src(&stmts, "ZF").expect("ZF defined"));
    assert!(rendered.starts_with("FEq"), "{rendered}");
}

#[test]
fn aarch64_fcmp_against_the_zero_immediate_is_supported() {
    // `fcmp s0, #0.0` is the ISA's only immediate form.
    let stmts = lift_aarch64("fcmp", vec![reg("s0"), op("#0.0", OperandKind::Immediate)]);
    assert!(
        !stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Unsupported { .. })),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_fmov_between_general_and_vector_registers_moves_raw_bits() {
    // `fmov w0, s0` reinterprets rather than converts, so no rounding
    // node may appear.
    let stmts = lift_aarch64("fmov", vec![reg("w0"), reg("s0")]);
    let rendered = format!("{stmts:?}");
    assert!(!rendered.contains("FpToFp"), "{rendered}");
}

#[test]
fn aarch64_ucvtf_converts_the_unsigned_range_exactly() {
    // The IR carries only the signed conversion; an unsigned source is
    // zero-extended by one bit so the signed node covers its whole
    // range without approximating.
    let stmts = lift_aarch64("ucvtf", vec![reg("s0"), reg("w0")]);
    let rendered = format!("{:?}", simd_parent_src(&stmts, "v0"));
    assert!(rendered.contains("ZeroExtend"), "{rendered}");
    assert!(rendered.contains("SbvToFp"), "{rendered}");
}

#[test]
fn aarch64_fp_effect_defines_the_vector_parent_without_reading_it() {
    let i = insn(0x1000, 4, "fadd", vec![reg("s0"), reg("s1"), reg("s2")]);
    let e = crate::effect::analyze(&i, Arch::Aarch64);
    assert_eq!(e.defs, vec!["v0"], "{e:?}");
}

#[test]
fn aarch64_fp_mnemonics_are_covered_by_both_effect_and_lifter() {
    // The same parity guard the x86 SIMD family carries: an effect
    // table that keeps an instruction the lifter declines would leave
    // the vector parent undefined and let a stale value stand in.
    for m in [
        "fadd", "fsub", "fmul", "fdiv", "fmax", "fmin", "fsqrt", "fabs", "fneg", "fmov", "fcvt",
    ] {
        let operands = match m {
            "fsqrt" | "fabs" | "fneg" | "fmov" | "fcvt" => vec![reg("s0"), reg("s1")],
            _ => vec![reg("s0"), reg("s1"), reg("s2")],
        };
        let i = insn(0x1000, 4, m, operands);
        assert_eq!(
            crate::effect::analyze(&i, Arch::Aarch64).kind,
            crate::effect::InstructionKind::Simd,
            "{m}: effect table does not classify it as Simd"
        );
        let stmts = lift_per_mnemonic(&i, Arch::Aarch64);
        assert!(
            !stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Unsupported { .. })),
            "{m}: lifter emitted Unsupported"
        );
        assert!(
            stmts
                .iter()
                .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name.starts_with('v'))),
            "{m}: lifter did not define the vector parent"
        );
    }
}

// ---------------------------------------------------------------
// AArch32 VFP scalar floating point.
// ---------------------------------------------------------------

fn lift_aarch32(mnemonic: &str, operands: Vec<Operand>) -> Vec<IrStmt> {
    lift_per_mnemonic(&insn(0x1000, 4, mnemonic, operands), Arch::Arm)
}

#[test]
fn aarch32_vadd_takes_its_precision_from_the_mnemonic() {
    // Unlike `AArch64`, the lane width is spelled `.f32` / `.f64` in
    // the mnemonic rather than in the register letter.
    let stmts = lift_aarch32("vadd.f64", vec![reg("d0"), reg("d1"), reg("d2")]);
    let rendered = format!("{:?}", simd_parent_src(&stmts, "v0"));
    assert!(rendered.contains("ebits: 11"), "{rendered}");
}

#[test]
fn aarch32_vfp_write_preserves_the_rest_of_the_register_file() {
    // `AArch32` VFP writes only the addressed slice — the opposite of
    // `AArch64`, which zeroes the rest. `d1` is the upper half of `v0`,
    // so the lower half must survive.
    let stmts = lift_aarch32("vadd.f64", vec![reg("d1"), reg("d1"), reg("d1")]);
    let rendered = format!("{:?}", simd_parent_src(&stmts, "v0"));
    assert!(rendered.contains("hi: 63"), "{rendered}");
}

#[test]
fn aarch32_integer_typed_vector_mnemonic_is_not_scalar_floating_point() {
    // `vadd.i32` is packed integer, not scalar float — claiming it
    // would lift integer lanes as IEEE values.
    assert!(crate::lift::vfp_scalar("vadd.i32").is_none());
}

#[test]
fn aarch32_vcmp_writes_flags_and_defines_no_register() {
    let i = insn(0x1000, 4, "vcmp.f32", vec![reg("s0"), reg("s1")]);
    let e = crate::effect::analyze(&i, Arch::Arm);
    assert!(e.defines_flags && e.defs.is_empty(), "{e:?}");
}

#[test]
fn aarch32_vfp_arithmetic_reads_its_destination() {
    // The write preserves the surrounding register file, so the prior
    // value stays live and the destination is a use as well as a def.
    let i = insn(0x1000, 4, "vadd.f32", vec![reg("s0"), reg("s1"), reg("s2")]);
    let e = crate::effect::analyze(&i, Arch::Arm);
    assert!(e.uses.contains(&"v0"), "{e:?}");
}

#[test]
fn aarch64_scalar_arithmetic_is_unaffected_by_the_shape_guard() {
    let i = insn(0x1000, 4, "add", vec![reg("x0"), reg("x1"), reg("x2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "x0")),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_indexed_lane_operand_declines_at_the_lifter() {
    let i = insn(0x1000, 4, "vmov", vec![reg("r0"), reg("d0[1]")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_aapcs_vector_alias_is_unaffected_by_the_shape_guard() {
    // `v1` on AArch32 is the AAPCS alias for `r4`, a bare GPR name that
    // the shape guard must leave alone.
    let i = insn(0x1000, 4, "add", vec![reg("v1"), reg("v2"), reg("v3")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert!(
        stmts
            .iter()
            .any(|s| matches!(s, IrStmt::Assign { dst, .. } if dst.name == "r4")),
        "{stmts:?}"
    );
}

// --- ARM NEON packed data processing ---

/// The lane-width `Add` nodes in a lowered packed integer result.
fn packed_add_lane_widths(stmts: &[IrStmt]) -> Vec<u16> {
    fn walk(expr: &Expr, out: &mut Vec<u16>) {
        match expr {
            Expr::Add(a, b) => {
                out.push(expr_bits(a).max(expr_bits(b)));
                walk(a, out);
                walk(b, out);
            }
            Expr::Concat { high, low } => {
                walk(high, out);
                walk(low, out);
            }
            Expr::ZeroExtend { src, .. } | Expr::Extract { src, .. } => walk(src, out),
            _ => {}
        }
    }
    fn expr_bits(expr: &Expr) -> u16 {
        match expr {
            Expr::Extract { hi, lo, .. } => hi - lo + 1,
            Expr::Var(v) => v.bits,
            _ => 0,
        }
    }
    let mut out = Vec::new();
    for stmt in stmts {
        if let IrStmt::Assign { src, .. } = stmt {
            walk(src, &mut out);
        }
    }
    out
}

/// How many floating-point additions a lowering contains, and at what
/// significand width.
fn packed_fadd_sorts(stmts: &[IrStmt]) -> Vec<u16> {
    fn walk(expr: &Expr, out: &mut Vec<u16>) {
        if let Expr::FAdd(a, b, _) = expr {
            if let Expr::BvToFp { sbits, .. } = &**a {
                out.push(*sbits);
            }
            walk(a, out);
            walk(b, out);
            return;
        }
        match expr {
            Expr::Concat { high, low } => {
                walk(high, out);
                walk(low, out);
            }
            Expr::FpToIeeeBv(s)
            | Expr::BvToFp { src: s, .. }
            | Expr::ZeroExtend { src: s, .. }
            | Expr::Extract { src: s, .. } => walk(s, out),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for stmt in stmts {
        if let IrStmt::Assign { src, .. } = stmt {
            walk(src, &mut out);
        }
    }
    out
}

#[test]
fn aarch64_packed_integer_add_lifts_one_addition_per_lane() {
    // The whole point of the arrangement: `v0.4s` is four independent
    // 32-bit lanes, not one 128-bit value. A single wide `Add` would
    // propagate carries across every lane boundary.
    let i = insn(
        0x1000,
        4,
        "add",
        vec![reg("v0.4s"), reg("v1.4s"), reg("v2.4s")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert_eq!(packed_add_lane_widths(&stmts), vec![32, 32, 32, 32]);
}

#[test]
fn aarch64_packed_integer_add_writes_the_vector_parent() {
    let i = insn(
        0x1000,
        4,
        "add",
        vec![reg("v0.4s"), reg("v1.4s"), reg("v2.4s")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Assign { dst, .. }] if dst.name == "v0" && dst.bits == 128),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_packed_double_precision_add_lifts_one_addition_per_lane() {
    let i = insn(
        0x1000,
        4,
        "fadd",
        vec![reg("v0.2d"), reg("v1.2d"), reg("v2.2d")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    // Two lanes, each an IEEE double (significand 53).
    assert_eq!(packed_fadd_sorts(&stmts), vec![53, 53]);
}

#[test]
fn aarch64_packed_single_precision_add_lifts_four_lanes() {
    let i = insn(
        0x1000,
        4,
        "fadd",
        vec![reg("v0.4s"), reg("v1.4s"), reg("v2.4s")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert_eq!(packed_fadd_sorts(&stmts), vec![24, 24, 24, 24]);
}

#[test]
fn aarch64_packed_bitwise_and_lowers_once_over_the_whole_view() {
    // No carry crosses a lane boundary in a bitwise operation, so
    // sixteen byte lanes would grow the formula for an identical result.
    let i = insn(
        0x1000,
        4,
        "and",
        vec![reg("v0.16b"), reg("v1.16b"), reg("v2.16b")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(
            stmts.as_slice(),
            [IrStmt::Assign { src: Expr::And(a, b), .. }]
                if matches!(&**a, Expr::Var(v) if v.bits == 128)
                    && matches!(&**b, Expr::Var(v) if v.bits == 128)
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_half_width_arrangement_write_zeroes_the_upper_half() {
    // An `AArch64` SIMD write has no merging form: writing `v0.8b`
    // zeroes bits 127:64 of `v0`.
    let i = insn(0x1000, 4, "mvn", vec![reg("v0.8b"), reg("v1.8b")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(
            stmts.as_slice(),
            [IrStmt::Assign {
                src: Expr::ZeroExtend { to_bits: 128, .. },
                ..
            }]
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_unmodelled_vector_mnemonic_declines() {
    // `fmulx` is the canary now that `fmla` lifts: it is a float
    // multiply extended (`0 * inf` gives 2.0, not NaN), which no family
    // claims. A mnemonic no family claims must decline rather than fall
    // into a same-width handler.
    let i = insn(
        0x1000,
        4,
        "fmulx",
        vec![reg("v0.4s"), reg("v1.4s"), reg("v2.4s")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_mismatched_arrangement_on_a_modelled_mnemonic_declines() {
    let i = insn(
        0x1000,
        4,
        "add",
        vec![reg("v0.4s"), reg("v1.4s"), reg("v2.2d")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_by_element_operand_reads_the_named_element() {
    // `mul v0.4s, v1.4s, v2.s[0]` multiplies lane 0 of `v2` into every
    // destination lane, so `v2` is read at one fixed index while `v1`
    // walks its own.
    let i = insn(
        0x1000,
        4,
        "mul",
        vec![reg("v0.4s"), reg("v1.4s"), reg("v2.s[1]")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Assign { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_byte_arrangement_floating_point_declines() {
    // `.16b` names an 8-bit element, which is not an IEEE sort.
    let i = insn(
        0x1000,
        4,
        "fadd",
        vec![reg("v0.16b"), reg("v1.16b"), reg("v2.16b")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_packed_integer_add_lifts_one_addition_per_lane() {
    let i = insn(0x1000, 4, "vadd.i32", vec![reg("q0"), reg("q1"), reg("q2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert_eq!(packed_add_lane_widths(&stmts), vec![32, 32, 32, 32]);
}

#[test]
fn aarch32_saturating_shift_left_immediate_lifts_but_register_form_declines() {
    // `vqshl` with an immediate amount saturates; the register form,
    // whose amount is a vector whose sign chooses the direction, is a
    // different lowering and stays declining.
    let immediate = insn(
        0x1000,
        4,
        "vqshl.s16",
        vec![reg("q0"), reg("q1"), op("3", OperandKind::Immediate)],
    );
    assert!(
        crate::lift::lift_per_mnemonic(&immediate, Arch::Arm)
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
    );
    let register = insn(
        0x1000,
        4,
        "vqshl.s16",
        vec![reg("q0"), reg("q1"), reg("q2")],
    );
    assert!(matches!(
        crate::lift::lift_per_mnemonic(&register, Arch::Arm).as_slice(),
        [IrStmt::Unsupported { .. }]
    ));
}

#[test]
fn aarch32_rounding_right_shifts_lift() {
    // `vrshr` / `vrsra` are the shift family's rounding forms; they lift
    // through the same immediate-shift path with the rounding flag set.
    for mnemonic in ["vrshr.u32", "vrsra.s16"] {
        let i = insn(
            0x1000,
            4,
            mnemonic,
            vec![reg("q0"), reg("q1"), op("2", OperandKind::Immediate)],
        );
        let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
        assert!(
            stmts
                .iter()
                .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
            "{mnemonic}: {stmts:?}"
        );
    }
}

#[test]
fn aarch32_single_precision_on_a_d_register_covers_both_lanes() {
    // `AArch32` puts only the element type in the mnemonic, so the lane
    // count comes from the destination: a `d` register holds two
    // single-precision elements. The scalar VFP handler would compute
    // only the low one and leave the high one at its stale value.
    let i = insn(0x1000, 4, "vadd.f32", vec![reg("d0"), reg("d1"), reg("d2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert_eq!(packed_fadd_sorts(&stmts), vec![24, 24]);
}

#[test]
fn aarch32_single_precision_on_an_s_register_stays_scalar() {
    // One element in the destination view is the scalar VFP form, which
    // the packed dispatch must leave to the existing handler.
    let i = insn(0x1000, 4, "vadd.f32", vec![reg("s0"), reg("s1"), reg("s2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert_eq!(packed_fadd_sorts(&stmts), vec![24]);
}

#[test]
fn aarch32_double_precision_on_a_d_register_stays_scalar() {
    let i = insn(0x1000, 4, "vadd.f64", vec![reg("d0"), reg("d1"), reg("d2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert_eq!(packed_fadd_sorts(&stmts), vec![53]);
}

#[test]
fn aarch32_untyped_bitwise_neon_lowers_over_the_whole_view() {
    // The bitwise NEON mnemonics make the element type optional, so
    // disassemblers emit them bare.
    let i = insn(0x1000, 4, "vand", vec![reg("q0"), reg("q1"), reg("q2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert!(
        matches!(
            stmts.as_slice(),
            [IrStmt::Assign { src: Expr::And(a, b), .. }]
                if matches!(&**a, Expr::Var(v) if v.bits == 128)
                    && matches!(&**b, Expr::Var(v) if v.bits == 128)
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_packed_write_preserves_the_rest_of_the_vector_register() {
    // Unlike `AArch64`, an `AArch32` vector write merges: `d1` survives
    // a write to `d0`, both being halves of `q0`.
    let i = insn(0x1000, 4, "vadd.i32", vec![reg("d0"), reg("d1"), reg("d2")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert!(
        matches!(
            stmts.as_slice(),
            [IrStmt::Assign { src: Expr::Concat { high, .. }, .. }]
                if matches!(&**high, Expr::Extract { hi: 127, lo: 64, .. })
        ),
        "{stmts:?}"
    );
}

// --- the bit-moving NEON families, whose element is a bare width ---

#[test]
fn aarch32_bare_width_element_no_longer_hides_the_permutation_family() {
    // `neon_element_type` reads the first character as the signedness
    // class, so `.8` names none and the whole family used to fall
    // through the mnemonic dispatch untouched.
    let i = insn(
        0x1000,
        4,
        "vext.8",
        vec![
            reg("q0"),
            reg("q1"),
            reg("q2"),
            op("3", OperandKind::Immediate),
        ],
    );
    assert!(
        crate::lift::lift_per_mnemonic(&i, Arch::Arm)
            .iter()
            .all(|s| !matches!(s, IrStmt::Unsupported { .. })),
    );
}

#[test]
fn aarch32_permutation_destination_is_a_definition_the_slicer_keeps() {
    // The trap this guards: a vector operand resolves as a `use` but
    // not as a `def`, which leaves the instruction neither truncating
    // the slice nor defining its destination, and a later read binds to
    // a stale value.
    let i = insn(0x1000, 4, "vrev64.8", vec![reg("q0"), reg("q1")]);
    let e = crate::effect::analyze(&i, Arch::Arm);
    assert_eq!(e.kind, crate::effect::InstructionKind::Simd);
    assert!(e.defs.contains(&"v0"), "{e:?}");
}

#[test]
fn aarch32_permutation_destination_is_also_a_use_because_the_write_merges() {
    // An `AArch32` vector write preserves the parent bits outside the
    // destination's view, so whatever defined the register before is
    // still live.
    let i = insn(0x1000, 4, "vrev64.8", vec![reg("d0"), reg("d2")]);
    let e = crate::effect::analyze(&i, Arch::Arm);
    assert!(e.uses.contains(&"v0"), "{e:?}");
}

#[test]
fn aarch32_paired_permutation_defines_both_named_registers() {
    // The whole reason `vzip` / `vuzp` / `vtrn` need an effect entry of
    // their own: `AArch64` reaches the same result with two
    // single-destination instructions, so a port that recorded one
    // definition would let the slicer drop whatever defined the second
    // register and leave a later read on a stale value.
    let i = insn(0x1000, 4, "vzip.8", vec![reg("q0"), reg("q1")]);
    let e = crate::effect::analyze(&i, Arch::Arm);
    assert!(e.defs.contains(&"v0") && e.defs.contains(&"v1"), "{e:?}");
}

#[test]
fn aarch32_paired_permutation_stashes_the_second_result_before_writing() {
    // SSA renames by statement position, so an assignment to the second
    // register placed after the first would read the freshly permuted
    // value as its input. The temp between them is what pins the read
    // to the original.
    let i = insn(0x1000, 4, "vzip.8", vec![reg("q0"), reg("q1")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Arm);
    assert!(
        matches!(
            stmts.first(),
            Some(IrStmt::Assign { dst, .. }) if dst.name.starts_with("t_")
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch32_vdup_from_a_vector_element_declines_at_the_lifter() {
    // `d2[1]` names one lane; the register table maps it onto the whole
    // `d2` slice, so lifting it would read 64 bits where the
    // instruction names 32.
    let i = insn(0x1000, 4, "vdup.32", vec![reg("q0"), reg("d2[1]")]);
    assert!(matches!(
        crate::lift::lift_per_mnemonic(&i, Arch::Arm).as_slice(),
        [IrStmt::Unsupported { .. }]
    ));
}

#[test]
fn aarch32_empty_element_type_suffix_does_not_panic() {
    // `split_once('.')` hands the element-type parser an empty string
    // for a mnemonic spelled `vadd.`, and `split_at(1)` panicked on it.
    let i = insn(0x1000, 4, "vadd.", vec![reg("q0"), reg("q1"), reg("q2")]);
    assert!(matches!(
        crate::lift::lift_per_mnemonic(&i, Arch::Arm).as_slice(),
        [IrStmt::Unsupported { .. }]
    ));
}

// --- rounding-mode pinning on ARM ---

/// The guard takes the whole instruction because x87 needs the operand
/// to tell a narrowing store from a widening one; on ARM the mnemonic
/// carries the whole answer, so the operand list stays empty.
fn arm_pin_probe(mnemonic: &str) -> Instruction {
    insn(0x40_1000, 4, mnemonic, vec![])
}

#[test]
fn aarch64_floating_point_arithmetic_pins_the_rounding_mode() {
    for mnemonic in ["fadd", "fsub", "fmul", "fdiv", "fsqrt", "fcvt", "scvtf"] {
        assert!(
            crate::lift::pins_rounding_mode(&arm_pin_probe(mnemonic), Arch::Aarch64),
            "{mnemonic}"
        );
    }
}

#[test]
fn aarch64_non_rounding_floating_point_does_not_pin_the_rounding_mode() {
    // `fmax` / `fmin` select an operand, `fcvtzs` / `fcvtzu` carry
    // round-toward-zero in the opcode, and `fmov` / `fabs` / `fneg` /
    // `fcmp` move or inspect a bit pattern.
    for mnemonic in [
        "fmax", "fmin", "fcvtzs", "fcvtzu", "fmov", "fabs", "fneg", "fcmp",
    ] {
        assert!(
            !crate::lift::pins_rounding_mode(&arm_pin_probe(mnemonic), Arch::Aarch64),
            "{mnemonic}"
        );
    }
}

#[test]
fn aarch32_floating_point_arithmetic_pins_the_rounding_mode() {
    for mnemonic in ["vadd.f32", "vsub.f64", "vmul.f32", "vdiv.f32", "vsqrt.f32"] {
        assert!(
            crate::lift::pins_rounding_mode(&arm_pin_probe(mnemonic), Arch::Arm),
            "{mnemonic}"
        );
    }
}

#[test]
fn aarch32_non_rounding_floating_point_does_not_pin_the_rounding_mode() {
    for mnemonic in [
        "vmov.f32", "vcmp.f32", "vabs.f32", "vneg.f64", "vadd.i32", "vand",
    ] {
        assert!(
            !crate::lift::pins_rounding_mode(&arm_pin_probe(mnemonic), Arch::Arm),
            "{mnemonic}"
        );
    }
}

// --- N3a: NEON broadcast and permutation ---

/// The single assignment a NEON lowering emits, rendered.
fn neon_lowering(mnemonic: &str, operands: &[&str]) -> String {
    let i = insn(
        0x1000,
        4,
        mnemonic,
        operands.iter().map(|o| reg(o)).collect(),
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert_eq!(stmts.len(), 1, "expected one statement: {stmts:?}");
    stmts[0].to_string()
}

/// Whether a NEON mnemonic + operand shape declines.
fn neon_declines(mnemonic: &str, operands: &[&str]) -> bool {
    let i = insn(
        0x1000,
        4,
        mnemonic,
        operands.iter().map(|o| reg(o)).collect(),
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }])
}

#[test]
fn aarch64_movi_replicates_the_immediate_to_every_lane() {
    assert_eq!(
        neon_lowering("movi", &["v0.4s", "#1"]),
        "v0 := concat(0x1:32, concat(0x1:32, concat(0x1:32, 0x1:32)))"
    );
}

#[test]
fn aarch64_mvni_replicates_the_inverted_immediate() {
    // The inversion is per lane and masked to the lane width, so a `4s`
    // arrangement gives `~1` at 32 bits rather than at 64.
    assert_eq!(
        neon_lowering("mvni", &["v0.4s", "#1"]),
        "v0 := concat(0xfffffffe:32, concat(0xfffffffe:32, concat(0xfffffffe:32, 0xfffffffe:32)))"
    );
}

#[test]
fn aarch64_movi_shifts_the_immediate_before_replicating() {
    assert_eq!(
        neon_lowering("movi", &["v0.4s", "#1", "lsl #8"]),
        "v0 := concat(0x100:32, concat(0x100:32, concat(0x100:32, 0x100:32)))"
    );
}

#[test]
fn aarch64_movi_fills_the_mask_shift_form_with_ones() {
    // `msl` shifts ones in, not zeroes, so the lane is 0x1ff rather
    // than the 0x100 the `lsl` spelling above produces.
    assert_eq!(
        neon_lowering("movi", &["v0.4s", "#1", "msl #8"]),
        "v0 := concat(0x1ff:32, concat(0x1ff:32, concat(0x1ff:32, 0x1ff:32)))"
    );
}

#[test]
fn aarch64_dup_from_a_general_register_replicates_the_low_element() {
    assert_eq!(
        neon_lowering("dup", &["v0.4s", "w1"]),
        "v0 := concat(x1[31:0], concat(x1[31:0], concat(x1[31:0], x1[31:0])))"
    );
}

#[test]
fn aarch64_dup_from_an_element_reads_the_indexed_lane() {
    // `v1.s[2]` is bits 95:64 — the lane index has to reach the read,
    // which is why the lane helpers take one.
    assert_eq!(
        neon_lowering("dup", &["v0.4s", "v1.s[2]"]),
        "v0 := concat(v1[95:64], concat(v1[95:64], concat(v1[95:64], v1[95:64])))"
    );
}

#[test]
fn aarch64_ext_windows_the_two_sources_with_the_first_at_the_low_end() {
    // ARM concatenates Vm:Vn with Vn low, then takes `index` bytes up.
    assert_eq!(
        neon_lowering("ext", &["v0.16b", "v1.16b", "v2.16b", "#4"]),
        "v0 := concat(v2, v1)[159:32]"
    );
}

#[test]
fn aarch64_ext_declines_a_non_byte_arrangement() {
    // Only `.8b` and `.16b` are architecturally valid for `ext`.
    assert!(neon_declines("ext", &["v0.4s", "v1.4s", "v2.4s", "#1"]));
}

#[test]
fn aarch64_ext_declines_an_out_of_range_byte_index() {
    assert!(neon_declines("ext", &["v0.16b", "v1.16b", "v2.16b", "#16"]));
}

#[test]
fn aarch64_zip1_interleaves_the_lower_halves() {
    // result[0]=Vn[0], result[1]=Vm[0], result[2]=Vn[1], result[3]=Vm[1].
    assert_eq!(
        neon_lowering("zip1", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat(v2[63:32], concat(v1[63:32], concat(v2[31:0], v1[31:0])))"
    );
}

#[test]
fn aarch64_zip2_interleaves_the_upper_halves() {
    assert_eq!(
        neon_lowering("zip2", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat(v2[127:96], concat(v1[127:96], concat(v2[95:64], v1[95:64])))"
    );
}

#[test]
fn aarch64_uzp1_takes_the_even_lanes_of_each_source_in_turn() {
    // The low half of the destination walks Vn's even lanes, the high
    // half Vm's.
    assert_eq!(
        neon_lowering("uzp1", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat(v2[95:64], concat(v2[31:0], concat(v1[95:64], v1[31:0])))"
    );
}

#[test]
fn aarch64_uzp2_takes_the_odd_lanes() {
    assert_eq!(
        neon_lowering("uzp2", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat(v2[127:96], concat(v2[63:32], concat(v1[127:96], v1[63:32])))"
    );
}

#[test]
fn aarch64_trn1_transposes_the_even_lanes() {
    assert_eq!(
        neon_lowering("trn1", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat(v2[95:64], concat(v1[95:64], concat(v2[31:0], v1[31:0])))"
    );
}

#[test]
fn aarch64_trn2_transposes_the_odd_lanes() {
    assert_eq!(
        neon_lowering("trn2", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat(v2[127:96], concat(v1[127:96], concat(v2[63:32], v1[63:32])))"
    );
}

#[test]
fn aarch64_rev64_reverses_elements_within_each_doubleword() {
    // Two `s` elements per 64-bit container, so each pair swaps.
    assert_eq!(
        neon_lowering("rev64", &["v0.4s", "v1.4s"]),
        "v0 := concat(v1[95:64], concat(v1[127:96], concat(v1[31:0], v1[63:32])))"
    );
}

#[test]
fn aarch64_rev16_reverses_bytes_within_each_halfword() {
    assert_eq!(
        neon_lowering("rev16", &["v0.8b", "v1.8b"]),
        "v0 := zext(concat(v1[55:48], concat(v1[63:56], concat(v1[39:32], \
concat(v1[47:40], concat(v1[23:16], concat(v1[31:24], concat(v1[7:0], v1[15:8]))))))), 128)"
    );
}

#[test]
fn aarch64_rev_declines_a_container_no_wider_than_its_element() {
    // `rev32 v0.4s` would reverse a single element inside its own
    // container, which the encoding does not admit.
    assert!(neon_declines("rev32", &["v0.4s", "v1.4s"]));
}

#[test]
fn aarch64_umov_zero_extends_the_element_into_the_general_register() {
    assert_eq!(
        neon_lowering("umov", &["w0", "v1.s[1]"]),
        "x0 := zext(v1[63:32], 64)"
    );
}

#[test]
fn aarch64_smov_sign_extends_the_element() {
    assert_eq!(
        neon_lowering("smov", &["x0", "v1.h[3]"]),
        "x0 := sext(v1[63:48], 64)"
    );
}

#[test]
fn aarch64_ins_from_a_general_register_preserves_the_other_lanes() {
    assert_eq!(
        neon_lowering("ins", &["v0.s[1]", "w0"]),
        "v0 := concat(concat(v0[127:64], x0[31:0]), v0[31:0])"
    );
}

#[test]
fn aarch64_ins_from_an_element_moves_between_lanes() {
    assert_eq!(
        neon_lowering("ins", &["v0.s[1]", "v1.s[3]"]),
        "v0 := concat(concat(v0[127:64], v1[127:96]), v0[31:0])"
    );
}

#[test]
fn aarch64_ins_declines_a_source_element_of_a_different_width() {
    assert!(neon_declines("ins", &["v0.s[1]", "v1.h[3]"]));
}

#[test]
fn aarch64_indexed_element_declines_an_out_of_range_index() {
    // `v1.s[4]` names a fifth 32-bit element of a 128-bit register.
    assert!(neon_declines("umov", &["w0", "v1.s[4]"]));
}

#[test]
fn aarch64_mov_alias_routes_by_operand_shape() {
    // `mov` is a lane-wise copy, an element insert and an element read
    // depending only on its operands.
    assert_eq!(neon_lowering("mov", &["v0.16b", "v1.16b"]), "v0 := v1");
    assert_eq!(
        neon_lowering("mov", &["v0.s[1]", "w0"]),
        "v0 := concat(concat(v0[127:64], x0[31:0]), v0[31:0])"
    );
    assert_eq!(
        neon_lowering("mov", &["w0", "v1.s[1]"]),
        "x0 := zext(v1[63:32], 64)"
    );
}

// --- N3b: NEON widening and narrowing ---

#[test]
fn aarch64_ushll_extends_each_element_before_shifting() {
    // Extending first is the point of the family: shifting at the
    // source width would drop the bits the long form exists to keep.
    assert_eq!(
        neon_lowering("ushll", &["v0.4s", "v1.4h", "#3"]),
        "v0 := concat((zext(v1[63:48], 32) << 0x3:32), concat((zext(v1[47:32], 32) << 0x3:32), \
concat((zext(v1[31:16], 32) << 0x3:32), (zext(v1[15:0], 32) << 0x3:32))))"
    );
}

#[test]
fn aarch64_ushll2_reads_the_upper_half_of_its_source() {
    assert_eq!(
        neon_lowering("ushll2", &["v0.4s", "v1.8h", "#3"]),
        "v0 := concat((zext(v1[127:112], 32) << 0x3:32), concat((zext(v1[111:96], 32) << 0x3:32), \
concat((zext(v1[95:80], 32) << 0x3:32), (zext(v1[79:64], 32) << 0x3:32))))"
    );
}

#[test]
fn aarch64_uxtl_is_the_zero_shift_alias() {
    assert_eq!(
        neon_lowering("uxtl", &["v0.4s", "v1.4h"]),
        "v0 := concat(zext(v1[63:48], 32), concat(zext(v1[47:32], 32), \
concat(zext(v1[31:16], 32), zext(v1[15:0], 32))))"
    );
}

#[test]
fn aarch64_sxtl_sign_extends() {
    assert_eq!(
        neon_lowering("sxtl", &["v0.4s", "v1.4h"]),
        "v0 := concat(sext(v1[63:48], 32), concat(sext(v1[47:32], 32), \
concat(sext(v1[31:16], 32), sext(v1[15:0], 32))))"
    );
}

#[test]
fn aarch64_xtn_zeroes_the_destination_upper_half() {
    // A 64-bit arrangement is still a whole-register write on AArch64.
    assert_eq!(
        neon_lowering("xtn", &["v0.4h", "v1.4s"]),
        "v0 := zext(concat(v1[127:96][15:0], concat(v1[95:64][15:0], \
concat(v1[63:32][15:0], v1[31:0][15:0]))), 128)"
    );
}

#[test]
fn aarch64_xtn2_preserves_the_destination_lower_half() {
    assert_eq!(
        neon_lowering("xtn2", &["v0.8h", "v1.4s"]),
        "v0 := concat(concat(v1[127:96][15:0], concat(v1[95:64][15:0], \
concat(v1[63:32][15:0], v1[31:0][15:0]))), v0[63:0])"
    );
}

#[test]
fn aarch64_uaddl_widens_both_sources_before_adding() {
    assert_eq!(
        neon_lowering("uaddl", &["v0.4s", "v1.4h", "v2.4h"]),
        "v0 := concat((zext(v1[63:48], 32) + zext(v2[63:48], 32)), \
concat((zext(v1[47:32], 32) + zext(v2[47:32], 32)), \
concat((zext(v1[31:16], 32) + zext(v2[31:16], 32)), \
(zext(v1[15:0], 32) + zext(v2[15:0], 32)))))"
    );
}

#[test]
fn aarch64_saddl_sign_extends_both_sources() {
    assert_eq!(
        neon_lowering("saddl", &["v0.4s", "v1.4h", "v2.4h"]),
        "v0 := concat((sext(v1[63:48], 32) + sext(v2[63:48], 32)), \
concat((sext(v1[47:32], 32) + sext(v2[47:32], 32)), \
concat((sext(v1[31:16], 32) + sext(v2[31:16], 32)), \
(sext(v1[15:0], 32) + sext(v2[15:0], 32)))))"
    );
}

#[test]
fn aarch64_uaddw_reads_its_first_source_at_the_destination_width() {
    // The `w` form's first operand is already wide and is not extended.
    assert_eq!(
        neon_lowering("uaddw", &["v0.4s", "v1.4s", "v2.4h"]),
        "v0 := concat((v1[127:96] + zext(v2[63:48], 32)), \
concat((v1[95:64] + zext(v2[47:32], 32)), \
concat((v1[63:32] + zext(v2[31:16], 32)), \
(v1[31:0] + zext(v2[15:0], 32)))))"
    );
}

#[test]
fn aarch64_uaddw2_halves_only_the_narrow_source() {
    // The wide source keeps lane `i`; only the narrow one is read from
    // the register's upper half.
    assert_eq!(
        neon_lowering("uaddw2", &["v0.4s", "v1.4s", "v2.8h"]),
        "v0 := concat((v1[127:96] + zext(v2[127:112], 32)), \
concat((v1[95:64] + zext(v2[111:96], 32)), \
concat((v1[63:32] + zext(v2[95:80], 32)), \
(v1[31:0] + zext(v2[79:64], 32)))))"
    );
}

#[test]
fn aarch64_umull_multiplies_at_the_widened_width() {
    // A same-width multiply would discard exactly the high half the
    // long form exists to compute.
    assert_eq!(
        neon_lowering("umull", &["v0.4s", "v1.4h", "v2.4h"]),
        "v0 := concat((zext(v1[63:48], 32) * zext(v2[63:48], 32)), \
concat((zext(v1[47:32], 32) * zext(v2[47:32], 32)), \
concat((zext(v1[31:16], 32) * zext(v2[31:16], 32)), \
(zext(v1[15:0], 32) * zext(v2[15:0], 32)))))"
    );
}

#[test]
fn aarch64_ssubl_subtracts_at_the_widened_width() {
    assert_eq!(
        neon_lowering("ssubl", &["v0.4s", "v1.4h", "v2.4h"]),
        "v0 := concat((sext(v1[63:48], 32) - sext(v2[63:48], 32)), \
concat((sext(v1[47:32], 32) - sext(v2[47:32], 32)), \
concat((sext(v1[31:16], 32) - sext(v2[31:16], 32)), \
(sext(v1[15:0], 32) - sext(v2[15:0], 32)))))"
    );
}

#[test]
fn aarch64_widening_declines_a_source_of_the_wrong_width() {
    // `uaddl` needs sources half the destination's element width.
    assert!(neon_declines("uaddl", &["v0.4s", "v1.4s", "v2.4s"]));
}

#[test]
fn aarch64_widening_declines_a_non_two_form_reading_a_full_register() {
    // `uaddl v0.4s, v1.8h, v2.8h` is the `uaddl2` shape without the
    // suffix that says to take the upper half.
    assert!(neon_declines("uaddl", &["v0.4s", "v1.8h", "v2.8h"]));
}

#[test]
fn aarch64_two_form_declines_a_half_register_source() {
    assert!(neon_declines("uaddl2", &["v0.4s", "v1.4h", "v2.4h"]));
}

// --- N3c: NEON multiply-accumulate ---

#[test]
fn aarch64_mla_accumulates_the_product_into_the_destination() {
    assert_eq!(
        neon_lowering("mla", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat((v0[127:96] + (v1[127:96] * v2[127:96])), \
concat((v0[95:64] + (v1[95:64] * v2[95:64])), \
concat((v0[63:32] + (v1[63:32] * v2[63:32])), \
(v0[31:0] + (v1[31:0] * v2[31:0])))))"
    );
}

#[test]
fn aarch64_mls_subtracts_the_product_from_the_destination() {
    assert_eq!(
        neon_lowering("mls", &["v0.4s", "v1.4s", "v2.4s"]),
        "v0 := concat((v0[127:96] - (v1[127:96] * v2[127:96])), \
concat((v0[95:64] - (v1[95:64] * v2[95:64])), \
concat((v0[63:32] - (v1[63:32] * v2[63:32])), \
(v0[31:0] - (v1[31:0] * v2[31:0])))))"
    );
}

#[test]
fn aarch64_umlal_multiplies_at_the_accumulator_width() {
    assert_eq!(
        neon_lowering("umlal", &["v0.4s", "v1.4h", "v2.4h"]),
        "v0 := concat((v0[127:96] + (zext(v1[63:48], 32) * zext(v2[63:48], 32))), \
concat((v0[95:64] + (zext(v1[47:32], 32) * zext(v2[47:32], 32))), \
concat((v0[63:32] + (zext(v1[31:16], 32) * zext(v2[31:16], 32))), \
(v0[31:0] + (zext(v1[15:0], 32) * zext(v2[15:0], 32))))))"
    );
}

#[test]
fn aarch64_smlal2_reads_the_upper_half_of_its_sources() {
    assert_eq!(
        neon_lowering("smlal2", &["v0.4s", "v1.8h", "v2.8h"]),
        "v0 := concat((v0[127:96] + (sext(v1[127:112], 32) * sext(v2[127:112], 32))), \
concat((v0[95:64] + (sext(v1[111:96], 32) * sext(v2[111:96], 32))), \
concat((v0[63:32] + (sext(v1[95:80], 32) * sext(v2[95:80], 32))), \
(v0[31:0] + (sext(v1[79:64], 32) * sext(v2[79:64], 32))))))"
    );
}

#[test]
fn aarch64_umlsl_subtracts_the_widened_product() {
    assert_eq!(
        neon_lowering("umlsl", &["v0.4s", "v1.4h", "v2.4h"]),
        "v0 := concat((v0[127:96] - (zext(v1[63:48], 32) * zext(v2[63:48], 32))), \
concat((v0[95:64] - (zext(v1[47:32], 32) * zext(v2[47:32], 32))), \
concat((v0[63:32] - (zext(v1[31:16], 32) * zext(v2[31:16], 32))), \
(v0[31:0] - (zext(v1[15:0], 32) * zext(v2[15:0], 32))))))"
    );
}

#[test]
fn aarch64_same_width_accumulate_has_no_two_form() {
    // `mla2` is not an encoding; only the long forms have one.
    assert!(neon_declines("mla2", &["v0.4s", "v1.4s", "v2.4s"]));
}

#[test]
fn aarch64_accumulate_lifts_a_by_element_source() {
    // `mla v0.4s, v1.4s, v2.s[0]` multiplies one lane of the second
    // source into every destination lane. It used to decline on the
    // shared "every operand carries the same arrangement" check.
    assert!(!neon_declines("mla", &["v0.4s", "v1.4s", "v2.s[0]"]));
}

// --- N3f: NEON saturation ---

#[test]
fn aarch64_saturating_add_writes_the_vector_parent() {
    let i = insn(
        0x1000,
        4,
        "sqadd",
        vec![reg("v0.4h"), reg("v1.4h"), reg("v2.4h")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Assign { dst, .. }] if dst.name == "v0"),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_doubling_multiply_declines_an_unencodable_element_width() {
    // `sqdmulh` exists only for 16- and 32-bit elements; a byte or
    // doubleword arrangement is not an encoding.
    assert!(neon_declines("sqdmulh", &["v0.16b", "v1.16b", "v2.16b"]));
    assert!(neon_declines("sqdmulh", &["v0.2d", "v1.2d", "v2.2d"]));
}

#[test]
fn aarch64_shift_narrow_declines_a_shift_past_the_element_width() {
    // Beyond the destination's element width the surviving bits would be
    // sign or zero fill rather than source bits, which is outside the
    // encoding's shift range.
    assert!(neon_declines("rshrn", &["v0.8b", "v1.8h", "#9"]));
}

#[test]
fn aarch64_shift_narrow_declines_a_zero_shift() {
    assert!(neon_declines("rshrn", &["v0.8b", "v1.8h", "#0"]));
}

#[test]
fn aarch64_saturating_narrow_declines_mismatched_source_width() {
    // `sqxtn` narrows by exactly half.
    assert!(neon_declines("sqxtn", &["v0.8b", "v1.4s"]));
}

#[test]
fn aarch64_same_width_saturating_has_no_two_form() {
    assert!(neon_declines("sqadd2", &["v0.4h", "v1.4h", "v2.4h"]));
}

#[test]
fn aarch64_saturating_narrow_two_form_preserves_the_lower_half() {
    let i = insn(0x1000, 4, "sqxtn2", vec![reg("v0.16b"), reg("v1.8h")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(
            stmts.as_slice(),
            [IrStmt::Assign { src: Expr::Concat { low, .. }, .. }]
                if matches!(&**low, Expr::Extract { hi: 63, lo: 0, .. })
        ),
        "the surviving lower half must read the destination: {stmts:?}"
    );
}

// --- N3d: NEON same-width shifts ---

#[test]
fn aarch64_left_shift_declines_an_amount_at_the_element_width() {
    // `shl` encodes shifts in `0 .. esize-1`; the element width itself
    // is not an encoding.
    assert!(neon_declines("shl", &["v0.4h", "v1.4h", "#16"]));
}

#[test]
fn aarch64_right_shift_declines_a_zero_amount() {
    // `ushr` encodes shifts in `1 .. esize`.
    assert!(neon_declines("ushr", &["v0.4h", "v1.4h", "#0"]));
}

#[test]
fn aarch64_right_shift_declines_an_amount_past_the_element_width() {
    assert!(neon_declines("ushr", &["v0.4h", "v1.4h", "#17"]));
}

#[test]
fn aarch64_register_shift_declines_an_immediate_amount() {
    // `ushl` takes a vector of per-lane amounts, not an immediate.
    assert!(neon_declines("ushl", &["v0.4h", "v1.4h", "#4"]));
}

#[test]
fn aarch64_immediate_shift_declines_a_vector_amount() {
    assert!(neon_declines("ushr", &["v0.4h", "v1.4h", "v2.4h"]));
}

#[test]
fn aarch64_register_shift_builds_both_directions() {
    // The per-lane amount's sign chooses the direction at run time, so
    // the lowering selects between a left and a right shift rather than
    // committing to one.
    let i = insn(
        0x1000,
        4,
        "ushl",
        vec![reg("v0.4h"), reg("v1.4h"), reg("v2.4h")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    let rendered = stmts
        .first()
        .map(std::string::ToString::to_string)
        .unwrap_or_default();
    assert!(rendered.contains("ite"), "{rendered}");
}

// --- N3e: NEON conversions and compares ---

#[test]
fn aarch64_vector_compare_writes_a_mask_not_a_flag() {
    // The scalar compare path writes NZCV; a vector compare writes the
    // destination register, because its result feeds another vector op.
    let i = insn(
        0x1000,
        4,
        "cmgt",
        vec![reg("v0.4h"), reg("v1.4h"), reg("v2.4h")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Assign { dst, .. }] if dst.name == "v0"),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_vector_compare_sets_no_flags() {
    let i = insn(
        0x1000,
        4,
        "cmgt",
        vec![reg("v0.4h"), reg("v1.4h"), reg("v2.4h")],
    );
    let effect = crate::effect::analyze(&i, Arch::Aarch64);
    assert!(!effect.defines_flags, "{effect:?}");
}

#[test]
fn aarch64_float_compare_declines_a_byte_arrangement() {
    // `.16b` names an 8-bit element, which is not an IEEE sort.
    assert!(neon_declines("fcmgt", &["v0.16b", "v1.16b", "v2.16b"]));
}

#[test]
fn aarch64_test_bits_has_no_compare_with_zero_form() {
    // `cmtst v0.4h, v1.4h, #0` is not an encoding — the AND against zero
    // would be constant.
    assert!(neon_declines("cmtst", &["v0.4h", "v1.4h", "#0"]));
}

#[test]
fn aarch64_compare_declines_a_non_zero_immediate() {
    // The immediate form compares against zero only.
    assert!(neon_declines("cmgt", &["v0.4h", "v1.4h", "#1"]));
}

#[test]
fn aarch64_fixed_point_convert_lifts_at_a_half_width_arrangement() {
    // The three-operand form scales by a fractional-bit count. It used
    // to decline; the scale is an exact power of two, so it costs one
    // multiply and introduces no rounding of its own.
    assert!(!neon_declines("scvtf", &["v0.2s", "v1.2s", "#4"]));
}

#[test]
fn aarch64_convert_declines_mismatched_lane_geometry() {
    // `scvtf` converts element-for-element.
    assert!(neon_declines("scvtf", &["v0.2d", "v1.2s"]));
}

#[test]
fn aarch64_float_widen_two_form_reads_the_upper_half() {
    let i = insn(0x1000, 4, "fcvtl2", vec![reg("v0.2d"), reg("v1.4s")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    let rendered = stmts
        .first()
        .map(std::string::ToString::to_string)
        .unwrap_or_default();
    assert!(rendered.contains("127:96"), "{rendered}");
}

// --- N3g: NEON bitwise select ---

#[test]
fn aarch64_bitwise_select_lowers_once_over_the_whole_view() {
    // Bit-granular, so there is nothing to do per lane.
    let i = insn(
        0x1000,
        4,
        "bsl",
        vec![reg("v0.16b"), reg("v1.16b"), reg("v2.16b")],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(
            stmts.as_slice(),
            [IrStmt::Assign {
                src: Expr::Or(..),
                ..
            }]
        ),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_bitwise_select_declines_a_non_byte_arrangement() {
    assert!(neon_declines("bsl", &["v0.4s", "v1.4s", "v2.4s"]));
}

#[test]
fn aarch64_two_form_declines_a_half_width_full_register_operand() {
    // The `2` suffix names the half of the 128-bit register the base
    // form does not touch, so the operand it halves must span the whole
    // register. A 64-bit arrangement there is not an encoding.
    assert!(neon_declines("xtn2", &["v0.4h", "v1.4s"]));
    assert!(neon_declines("uaddl2", &["v0.2s", "v1.4h", "v2.4h"]));
    assert!(neon_declines("sqxtn2", &["v0.8b", "v1.8h"]));
}

#[test]
fn all_ones_below_the_payload_width_is_a_single_constant() {
    assert_eq!(crate::lift::simd::all_ones(8), Expr::konst(0xff, 8));
    assert_eq!(
        crate::lift::simd::all_ones(64),
        Expr::konst(u64::MAX.into(), 64)
    );
    assert_eq!(
        crate::lift::simd::all_ones(128),
        Expr::konst(u128::MAX, 128)
    );
}

#[test]
fn all_ones_above_the_payload_width_concatenates_full_chunks() {
    // A `Const` carries a `u128`, so a wider mask cannot be one
    // literal. Building it as `konst(u128::MAX, 256)` would denote a
    // *zero-extended* `u128::MAX` — a valid, wrong mask whose top half
    // is clear, which reads as correct at a glance and would silently
    // invert only the low half of an `x XOR all-ones` complement.
    let full = Expr::konst(u128::MAX, 128);
    assert_eq!(
        crate::lift::simd::all_ones(256),
        Expr::concat(full.clone(), full.clone())
    );
    assert_eq!(
        crate::lift::simd::all_ones(512),
        Expr::concat(
            Expr::concat(Expr::concat(full.clone(), full.clone()), full.clone()),
            full
        )
    );
}

#[test]
fn aarch64_across_lane_reduction_lifts_from_the_source_arrangement() {
    // The destination is a scalar and carries no arrangement, so the
    // geometry has to come from operand 1. A resolver that read
    // operand 0 the way every other family does would decline here.
    assert!(!neon_declines("addv", &["s0", "v1.4s"]));
    assert!(!neon_declines("uaddlv", &["h0", "v1.8b"]));
    assert!(!neon_declines("smaxv", &["b0", "v1.16b"]));
}

#[test]
fn aarch64_across_lane_reduction_declines_a_mismatched_destination_width() {
    // `uaddlv` widens, so an 8-bit source element produces a 16-bit
    // one — a byte destination is a different instruction.
    assert!(neon_declines("uaddlv", &["b0", "v1.8b"]));
    assert!(neon_declines("addv", &["h0", "v1.4s"]));
}

#[test]
fn aarch64_across_lane_reduction_declines_an_unencodable_arrangement() {
    // ARM ARM C7.2: a 64-bit element has no across-lane encoding, and
    // a 32-bit one requires the full-width arrangement.
    assert!(neon_declines("addv", &["d0", "v1.2d"]));
    assert!(neon_declines("addv", &["s0", "v1.2s"]));
}

#[test]
fn aarch64_across_lane_reduction_declines_an_arranged_destination() {
    // `addv v0.4s, v1.4s` is not an encoding: the destination of a
    // reduction holds one element, not a vector of them.
    assert!(neon_declines("addv", &["v0.4s", "v1.4s"]));
}

#[test]
fn aarch64_movi_with_a_ones_shift_lifts() {
    // `msl` shifts ones in, which is what builds the `0x0000ffff`-style
    // masks a bare `lsl` cannot reach.
    assert!(!neon_declines_mixed("movi", &["v0.4s", "1", "msl 8"]));
    assert!(!neon_declines_mixed("mvni", &["v0.4s", "1", "msl 16"]));
}

#[test]
fn aarch64_movi_ones_shift_declines_outside_its_encoding() {
    // ARM ARM C7.2: `msl` is encoded for 32-bit elements only, and only
    // for a shift of 8 or 16.
    assert!(neon_declines_mixed("movi", &["v0.8h", "2", "msl 8"]));
    assert!(neon_declines_mixed("movi", &["v0.4s", "1", "msl 4"]));
}

#[test]
fn aarch64_fixed_point_conversion_lifts() {
    assert!(!neon_declines_mixed("scvtf", &["v0.4s", "v1.4s", "4"]));
    assert!(!neon_declines_mixed("fcvtzu", &["v0.2d", "v1.2d", "64"]));
}

#[test]
fn aarch64_fixed_point_conversion_declines_an_out_of_range_fraction_width() {
    // ARM ARM C7.2 bounds the fraction width by `1 <= fbits <= esize`.
    assert!(neon_declines_mixed("scvtf", &["v0.4s", "v1.4s", "0"]));
    assert!(neon_declines_mixed("scvtf", &["v0.4s", "v1.4s", "33"]));
}

#[test]
fn aarch64_float_width_conversion_has_no_fixed_point_form() {
    // `fcvtl` / `fcvtn` change the float format; there is no fixed
    // point on either side, so a third operand is not an encoding.
    assert!(neon_declines_mixed("fcvtl", &["v0.4s", "v1.4h", "2"]));
}

/// [`neon_declines`] for an instruction whose operands are not all
/// registers — the immediate-bearing forms.
fn neon_declines_mixed(mnemonic: &str, operands: &[&str]) -> bool {
    let i = insn(
        0x1000,
        4,
        mnemonic,
        operands
            .iter()
            .map(|raw| {
                let kind = if raw.starts_with('v') {
                    OperandKind::Register
                } else {
                    OperandKind::Immediate
                };
                op(raw, kind)
            })
            .collect(),
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }])
}

#[test]
fn aarch64_by_element_long_form_lifts() {
    assert!(!neon_declines("umlal", &["v0.4s", "v1.4h", "v2.h[3]"]));
    assert!(!neon_declines("umlal2", &["v0.4s", "v1.8h", "v2.h[3]"]));
    assert!(!neon_declines("umull", &["v0.4s", "v1.4h", "v2.h[1]"]));
    assert!(!neon_declines("fmul", &["v0.4s", "v1.4s", "v2.s[1]"]));
}

#[test]
fn aarch64_fused_multiply_add_lifts_for_single_lanes() {
    // `fmla` / `fmls` over binary32 lanes lower through the binary64
    // fused step, in both the whole-vector and the by-element spelling.
    assert!(!neon_declines("fmla", &["v0.4s", "v1.4s", "v2.4s"]));
    assert!(!neon_declines("fmls", &["v0.2s", "v1.2s", "v2.2s"]));
    assert!(!neon_declines("fmla", &["v0.4s", "v1.4s", "v2.s[1]"]));
    assert!(!neon_declines("fmls", &["v0.2s", "v1.2s", "v2.s[0]"]));
}

#[test]
fn aarch64_by_element_fused_multiply_declines_double_lanes() {
    // The by-element form inherits the whole-vector width gate: a
    // binary64 lane would need binary128 to keep the fused step exact.
    assert!(neon_declines("fmla", &["v0.2d", "v1.2d", "v2.d[0]"]));
}

#[test]
fn aarch64_fused_multiply_add_declines_double_lanes() {
    // A binary64 lane would need binary128 to stay exact, so the `.2d`
    // arrangement declines rather than round twice.
    assert!(neon_declines("fmla", &["v0.2d", "v1.2d", "v2.2d"]));
    assert!(neon_declines("frecps", &["v0.2d", "v1.2d", "v2.2d"]));
}

#[test]
fn aarch64_by_element_declines_a_mismatched_element_width() {
    // The named element is the *source* width, so a long form over
    // halfword sources cannot name a word element.
    assert!(neon_declines("umlal", &["v0.4s", "v1.4h", "v2.s[1]"]));
    // And the byte element the dot products use has no by-element
    // multiply encoding at all.
    assert!(neon_declines("mul", &["v0.16b", "v1.16b", "v2.b[1]"]));
}

#[test]
fn aarch64_dot_product_lifts() {
    assert!(!neon_declines("sdot", &["v0.4s", "v1.16b", "v2.16b"]));
    assert!(!neon_declines("udot", &["v0.2s", "v1.8b", "v2.8b"]));
}

#[test]
fn aarch64_dot_product_declines_a_mismatched_source_geometry() {
    // Four byte products feed one 32-bit lane, so `.4s` needs `.16b`.
    assert!(neon_declines("sdot", &["v0.4s", "v1.8b", "v2.8b"]));
    assert!(neon_declines("sdot", &["v0.8h", "v1.16b", "v2.16b"]));
}

#[test]
fn aarch64_dot_product_lifts_the_by_element_spelling() {
    // `v2.4b[1]` names an arrangement and an index at once; its own
    // parser resolves it, and the index selects the four-byte group
    // broadcast to every destination lane.
    assert!(!neon_declines("sdot", &["v0.4s", "v1.16b", "v2.4b[1]"]));
    assert!(!neon_declines("udot", &["v0.2s", "v1.8b", "v2.4b[3]"]));
}

#[test]
fn aarch64_dot_product_by_element_declines_an_out_of_range_group() {
    // Four four-byte groups fit the 128-bit register, so index 4 is off
    // the end.
    assert!(neon_declines("sdot", &["v0.4s", "v1.16b", "v2.4b[4]"]));
}

#[test]
fn aarch64_float_across_lane_reduction_lifts() {
    assert!(!neon_declines("fmaxv", &["s0", "v1.4s"]));
    assert!(!neon_declines("fminv", &["h0", "v1.8h"]));
}

#[test]
fn aarch64_float_across_lane_reduction_declines_a_non_ieee_lane() {
    // A byte element names no float format, and a doubleword one has
    // no across-lane encoding.
    assert!(neon_declines("fmaxv", &["b0", "v1.8b"]));
    assert!(neon_declines("fmaxv", &["d0", "v1.2d"]));
}

#[test]
fn aarch64_number_flavoured_float_reduction_lifts() {
    // `fmaxnmv` / `fminnmv` are `FPMaxNum` / `FPMinNum`, which ignore a
    // quiet NaN operand rather than propagating it — a distinct fold
    // from `fmaxv`, resolved through the same reduction shape.
    assert!(!neon_declines("fmaxnmv", &["s0", "v1.4s"]));
    assert!(!neon_declines("fminnmv", &["s0", "v1.4s"]));
}

#[test]
fn aarch64_table_lookup_lifts_the_single_register_form() {
    let i = insn(
        0x1000,
        4,
        "tbl",
        vec![
            reg("v0.8b"),
            op("{v1.16b}", OperandKind::Unknown),
            reg("v2.8b"),
        ],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Assign { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_table_lookup_lifts_a_multi_register_table() {
    // A two-register table concatenates the members low-to-high into a
    // 32-byte lookup table.
    let i = insn(
        0x1000,
        4,
        "tbl",
        vec![
            reg("v0.16b"),
            op("{v1.16b, v2.16b}", OperandKind::Unknown),
            reg("v3.16b"),
        ],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Assign { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_table_lookup_declines_a_half_width_table() {
    // A table register is always the full 128 bits, whatever the
    // destination's arrangement.
    let i = insn(
        0x1000,
        4,
        "tbl",
        vec![
            reg("v0.8b"),
            op("{v1.8b}", OperandKind::Unknown),
            reg("v2.8b"),
        ],
    );
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(
        matches!(stmts.as_slice(), [IrStmt::Unsupported { .. }]),
        "{stmts:?}"
    );
}

#[test]
fn aarch64_polynomial_multiply_lifts_both_encodings() {
    assert!(!neon_declines("pmull", &["v0.8h", "v1.8b", "v2.8b"]));
    assert!(!neon_declines("pmull2", &["v0.8h", "v1.16b", "v2.16b"]));
    // The AES / GHASH shape, whose destination is the one-lane 128-bit
    // arrangement.
    assert!(!neon_declines("pmull", &["v0.1q", "v1.1d", "v2.1d"]));
    assert!(!neon_declines("pmull2", &["v0.1q", "v1.2d", "v2.2d"]));
}

#[test]
fn aarch64_polynomial_multiply_declines_an_unencodable_element() {
    // The architecture encodes `8B` into `8H` and `1D` into `1Q`, and
    // nothing between them.
    assert!(neon_declines("pmull", &["v0.4s", "v1.4h", "v2.4h"]));
}

#[test]
fn aarch64_same_width_polynomial_multiply_still_declines() {
    // `pmul` keeps the element width and is a different instruction
    // from `pmull`; nothing claims it.
    assert!(neon_declines("pmul", &["v0.8b", "v1.8b", "v2.8b"]));
}

#[test]
fn aarch64_reciprocal_estimate_assigns_the_destination() {
    // Emitting *something* is the point: a decline would truncate the
    // slice, while an assignment to a never-defined temp keeps it
    // Complete with the estimate as a free input.
    assert!(!neon_declines("frecpe", &["v0.4s", "v1.4s"]));
    assert!(!neon_declines("frsqrte", &["v0.2d", "v1.2d"]));
}

#[test]
fn aarch64_reciprocal_estimate_never_reads_its_source() {
    // The architecture fixes only a relative error bound, so the value
    // is implementation-defined and nothing about the source may reach
    // the result — an expression mentioning `v1` would be a definite
    // number the machine need not produce.
    let i = insn(0x1000, 4, "frecpe", vec![reg("v0.4s"), reg("v1.4s")]);
    let stmts = crate::lift::lift_per_mnemonic(&i, Arch::Aarch64);
    assert!(!format!("{stmts:?}").contains("v1"), "{stmts:?}");
}

#[test]
fn aarch64_reciprocal_estimate_declines_a_non_ieee_lane() {
    assert!(neon_declines("frecpe", &["v0.16b", "v1.16b"]));
}

#[test]
fn aarch64_reciprocal_step_lifts_for_single_lanes() {
    // `frecps` / `frsqrts` are `2.0 - x*y` and `(3.0 - x*y) / 2.0`
    // computed through `FPRecipStepFused` — one rounding over the whole
    // expression, which the binary64 fused step reproduces for binary32
    // lanes.
    assert!(!neon_declines("frecps", &["v0.4s", "v1.4s", "v2.4s"]));
    assert!(!neon_declines("frsqrts", &["v0.4s", "v1.4s", "v2.4s"]));
}
