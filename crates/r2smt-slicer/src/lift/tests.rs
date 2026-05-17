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
