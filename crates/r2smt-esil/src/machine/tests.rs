#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;

#[test]
fn assign_constant_to_register_64bit() {
    // ESIL for `mov rax, 1` on x86_64: "1,rax,=". `rax` is 64-bit
    // by table so the widen step is a no-op.
    let lift = lift_esil("1,rax,=", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(dst.bits, 64);
            assert_eq!(*src, Expr::konst(1, 64));
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn assign_constant_to_subregister_narrows_value() {
    // ESIL `mov eax, 1`: target is 32-bit, immediate enters as
    // 64-bit so the widen step extracts the low 32 bits.
    let lift = lift_esil("1,eax,=", Arch::X86_64).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "eax");
            assert_eq!(dst.bits, 32);
            assert_eq!(*src, Expr::extract(Expr::konst(1, 64), 31, 0));
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn compound_assignment_unwraps_into_self_referential_expression() {
    // ESIL for `add rax, 1`: "1,rax,+="
    let lift = lift_esil("1,rax,+=", Arch::X86_64).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            let expected = Expr::add(Expr::Var(Var::new("rax", 64)), Expr::konst(1, 64));
            assert_eq!(*src, expected);
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn test_esil_lnot_zero_yields_one() {
    // ESIL `!` is logical-not: `1` when the operand is zero, else
    // `0` (MicroSMT `m_lnot` parity). For the literal-zero operand
    // `0,!,rax,=` the modelled selector compares `0 == 0` and the
    // taken (`then`) branch is the 1-bit constant `1`.
    let lift = lift_esil("0,!,rax,=", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            let expected = Expr::zero_ext(
                Expr::Ite {
                    cond: Box::new(Expr::eq(Expr::konst(0, 64), Expr::konst(0, 64))),
                    then_expr: Box::new(Expr::konst(1, 1)),
                    else_expr: Box::new(Expr::konst(0, 1)),
                },
                64,
            );
            assert_eq!(*src, expected);
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn test_esil_lnot_nonzero_yields_zero() {
    // For a symbolic operand `eax,!,rax,=` the model is the same
    // single `Ite(eax == 0 ? 1 : 0)`; the not-taken (`else`) branch
    // is the 1-bit constant `0`, encoding `!x == 0` whenever
    // `x != 0`. The comparison is against a zero of the operand's
    // own width (32 here), not the pointer width.
    let lift = lift_esil("eax,!,rax,=", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            let expected = Expr::zero_ext(
                Expr::Ite {
                    cond: Box::new(Expr::eq(Expr::Var(Var::new("eax", 32)), Expr::konst(0, 32))),
                    then_expr: Box::new(Expr::konst(1, 1)),
                    else_expr: Box::new(Expr::konst(0, 1)),
                },
                64,
            );
            assert_eq!(*src, expected);
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn test_esil_lnot_width_is_1bit() {
    // The `!` result is a 1-bit truthiness value. Assigning it to
    // the 8-bit `al` must zero-extend a 1-bit core to 8 bits — the
    // `ZeroExtend { to_bits: 8 }` over an `Ite` whose branches are
    // `konst(_, 1)` proves the pushed value was exactly 1 bit wide.
    let lift = lift_esil("eax,!,al,=", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.bits, 8);
            let expected = Expr::zero_ext(
                Expr::Ite {
                    cond: Box::new(Expr::eq(Expr::Var(Var::new("eax", 32)), Expr::konst(0, 32))),
                    then_expr: Box::new(Expr::konst(1, 1)),
                    else_expr: Box::new(Expr::konst(0, 1)),
                },
                8,
            );
            assert_eq!(*src, expected);
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn flag_token_falls_back_to_free_var_without_arith_context() {
    // ESIL: "$z,zf,=" — copies the `$z` synthetic flag into the
    // 1-bit zf register. With no prior arithmetic operation the
    // machine has nothing to derive `$z` from and falls back to
    // the canonical `Var("ZF", 1)`. The lowercase `zf` target
    // also normalises to the uppercase canonical form.
    let lift = lift_esil("$z,zf,=", Arch::X86_64).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "ZF");
            assert_eq!(dst.bits, 1);
            assert_eq!(*src, Expr::Var(Var::new("ZF", 1)));
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn flag_token_derives_zero_bit_from_last_arith() {
    // After `1,eax,-` the machine remembers `1 - eax_widened` as
    // the latest arithmetic result, so `$z` becomes
    // `Ite(result == 0, 1, 0)` rather than a free flag variable.
    let lift = lift_esil("1,eax,-,$z,zf,=", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "ZF");
            // The src must be an Ite collapsing the last
            // arithmetic delta to a 1-bit flag.
            assert!(matches!(src, Expr::Ite { .. }));
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn unclosed_block_returns_unsupported_control_flow() {
    // `?{` without a matching `}` is malformed and must abort so
    // the slicer falls back to the per-mnemonic handler. Without
    // this check the unwrapped statements would be committed as
    // if they were unconditional.
    let err = lift_esil("rax,0,==,?{", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::UnsupportedControlFlow);
}

#[test]
fn block_simple_predicated_assign_wraps_with_ite() {
    // ESIL `0,rax,==,?{,2,rax,=,}`: "if rax == 0 then rax := 2".
    // The block close must turn the inner `rax := 2` into
    // `rax := Ite(rax == 0, 2, rax)`.
    let lift = lift_esil("0,rax,==,?{,2,rax,=,}", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            match src {
                Expr::Ite {
                    cond,
                    then_expr,
                    else_expr,
                } => {
                    // Condition is a 1-bit equality predicate.
                    assert!(matches!(cond.as_ref(), Expr::Eq(_, _)));
                    // Then-branch is the constant write.
                    assert_eq!(then_expr.as_ref().clone(), Expr::konst(2, 64));
                    // Else-branch preserves the prior value of rax.
                    assert_eq!(else_expr.as_ref().clone(), Expr::Var(Var::new("rax", 64)));
                }
                other => panic!("expected Ite, got {other:?}"),
            }
        }
        _ => panic!("expected Assign"),
    }
}

#[test]
fn block_nested_wraps_with_outer_then_inner_ite() {
    // ESIL `0,rax,==,?{,1,rbx,==,?{,2,rax,=,},}`: outer cond
    // wraps over the inner block; the inner block already
    // wrapped the assignment once. Result: nested Ite.
    let lift = lift_esil("0,rax,==,?{,1,rbx,==,?{,2,rax,=,},}", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    match &lift.statements[0] {
        IrStmt::Assign { src, .. } => match src {
            Expr::Ite {
                cond: outer_cond,
                then_expr: outer_then,
                ..
            } => {
                assert!(matches!(outer_cond.as_ref(), Expr::Eq(_, _)));
                // Inner is itself an Ite.
                assert!(matches!(outer_then.as_ref(), Expr::Ite { .. }));
            }
            other => panic!("expected outer Ite, got {other:?}"),
        },
        _ => panic!("expected Assign"),
    }
}

#[test]
fn block_with_store_returns_unsupported() {
    // Stores cannot be made conditional in the IR — the close
    // handler aborts the lift.
    let err = lift_esil("0,rax,==,?{,rax,rbx,=[8],}", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::UnsupportedControlFlow);
}

#[test]
fn block_with_load_returns_unsupported() {
    // Loads have an unconditional side effect; the block close
    // refuses to wrap them.
    let err = lift_esil("0,rax,==,?{,rsp,[4],}", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::UnsupportedControlFlow);
}

#[test]
fn block_close_without_open_returns_unsupported() {
    // `1,rax,=,}` is a valid `mov rax, 1` followed by a stray `}`
    // — the close handler must reject the orphan block close.
    let err = lift_esil("1,rax,=,}", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::UnsupportedControlFlow);
}

#[test]
fn unknown_token_surfaces_with_text() {
    // `??` is not an identifier (starts with a non-alpha char)
    // and not a recognised operator — it must surface as Unknown.
    let err = lift_esil("rax,??", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::UnknownToken("??".to_string()));
}

#[test]
fn stack_underflow_reports_context() {
    let err = lift_esil("+", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::StackUnderflow("binary rhs"));
}

#[test]
fn memory_load_emits_loadmem() {
    // ESIL: load 4 bytes from rsp into a temporary.
    //   "rsp,[4]"
    let lift = lift_esil("rsp,[4]", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    assert!(matches!(
        lift.statements[0],
        IrStmt::LoadMem { bits: 32, .. }
    ));
}

#[test]
fn memory_store_emits_storemem() {
    // ESIL: store rax into [rbp]: "rax,rbp,=[8]"
    let lift = lift_esil("rax,rbp,=[8]", Arch::X86_64).expect("lift ok");
    assert_eq!(lift.statements.len(), 1);
    assert!(matches!(
        lift.statements[0],
        IrStmt::StoreMem { bits: 64, .. }
    ));
}
