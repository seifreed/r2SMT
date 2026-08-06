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
    // After `1,eax,-` the machine remembers `eax_widened - 1` as
    // the latest arithmetic result — ESIL reads `a,b,OP` as
    // `b OP a` — so `$z` becomes
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
    // The top of the stack is the *left*-hand side, since ESIL reads
    // `a,b,OP` as `b OP a`, so it is the operand popped first and the
    // one an empty stack fails to supply.
    let err = lift_esil("+", Arch::X86_64).expect_err("must reject");
    assert_eq!(err, EsilError::StackUnderflow("binary lhs"));
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

#[test]
fn a_store_takes_its_address_from_the_top_of_the_stack() {
    // The assertion above only checks the *width*, so an address/value
    // swap in `apply_store` passes it — the exact shape of the operand
    // inversion found in `apply_binary`, in the busiest path there is:
    // stores are ~6 200 of the liftable ESIL strings across three ISAs.
    //
    // The ground truth is a real instruction rather than the ESIL VM.
    // x86 `push 0x14` lowers to `20,4,esp,-,=[4]`, which must write the
    // value 20 to the address `esp - 4`. So `=[N]` pops the *address*
    // first and the value second.
    let lift = lift_esil("20,4,esp,-,=[4]", Arch::X86_64).expect("lift ok");
    let store = lift
        .statements
        .iter()
        .find(|s| matches!(s, IrStmt::StoreMem { .. }))
        .expect("a store");
    match store {
        IrStmt::StoreMem {
            address,
            value,
            bits,
        } => {
            assert_eq!(*bits, 32);
            // The value is the literal being pushed, not the address.
            assert_eq!(*value, Expr::extract(Expr::konst(20, 64), 31, 0));
            // And the address is `esp - 4`, not `4 - esp`. `esp` is
            // 32-bit while the literal enters at the pointer width, so
            // the binary widens both to 64 — the operand *order* is what
            // this pins.
            assert_eq!(
                *address,
                Expr::sub(
                    Expr::zero_ext(Expr::Var(Var::new("esp", 32)), 64),
                    Expr::konst(4, 64)
                )
            );
        }
        other => panic!("expected StoreMem, got {other:?}"),
    }
}

#[test]
fn compound_assignment_subtracts_the_value_from_the_target() {
    // The only compound-assign test uses `+=`, which is commutative and
    // therefore passes on an inverted implementation. `-=` is 971 of the
    // liftable ESIL strings measured and is order-sensitive.
    //
    // Measured: `r2 -a x86 -b 64 -qc 'aer rax=10; ae 4,rax,-=; aer rax'`
    // → 6, so the target is the left-hand side.
    let lift = lift_esil("4,rax,-=", Arch::X86_64).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(
                *src,
                Expr::sub(Expr::Var(Var::new("rax", 64)), Expr::konst(4, 64))
            );
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn compound_shift_shifts_the_target_by_the_value() {
    // Same hazard, same direction. Measured
    // `aer rax=10; ae 4,rax,<<=; aer rax` → 0xa0, i.e. `10 << 4` and
    // never `4 << 10`.
    //
    // `>>=` is deliberately not pinned here: radare2 6.1.8 leaves the
    // register untouched for every input tried, so there is no
    // behaviour to measure against, and it appears zero times in the
    // liftable corpus anyway.
    let lift = lift_esil("4,rax,<<=", Arch::X86_64).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, src } => {
            assert_eq!(dst.name, "rax");
            assert_eq!(
                *src,
                Expr::shl(Expr::Var(Var::new("rax", 64)), Expr::konst(4, 64))
            );
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn the_negate_all_bits_operator_is_not_modelled_as_a_comparison() {
    // radare2's `ae???` spells `!=` as `(1 -- 0)` tagged `math+regw`:
    // it pops **one** operand and writes a register with its bitwise
    // complement, pushing nothing. It was modelled as `Expr::ne` — wrong
    // in kind, not merely in stack effect — and pinned that way by two
    // parser tests, which is worse than no test at all.
    //
    // It declines rather than being implemented because it is measured
    // unreachable: zero occurrences in liftable ESIL across three ISAs,
    // since every real use sits in a string that also carries `:=`.
    let err = lift_esil("rax,!=", Arch::X86_64).expect_err("`!=` must not lift");
    assert!(
        matches!(err, EsilError::UnknownToken(ref t) if t == "!="),
        "{err:?}"
    );
}

// --- ARM, which this file did not cover at all until now -------------

#[test]
fn arm_negative_flag_lands_in_the_sign_flag() {
    // `nf` is ARM's name for the flag this pipeline calls `SF`, and
    // every signed ARM condition (`lt`, `ge`, `gt`, `le`, `mi`, `pl`)
    // reads `SF`. Left unmapped, an ESIL-lifted ARM instruction defined
    // a register nothing consults and the branch predicate took a free
    // input — sound, and blind on every signed comparison.
    let lift = lift_esil("1,nf,=", Arch::Arm).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, .. } => assert_eq!(dst.name, "SF"),
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn arm_overflow_flag_lands_in_the_overflow_flag() {
    // The same for `vf`, which the signed conditions read alongside
    // `SF`: `lt` is `SF != OF`, so one of the two being blind is enough
    // to lose the branch.
    let lift = lift_esil("1,vf,=", Arch::Arm).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, .. } => assert_eq!(dst.name, "OF"),
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn the_arm_flag_names_do_not_disturb_x86() {
    // The mapping is unconditional, so it is worth pinning that x86
    // never spells a flag `nf` or `vf` — the names are ARM vocabulary
    // and x86 ESIL keeps using `sf` / `of`.
    let lift = lift_esil("1,sf,=", Arch::X86_64).expect("lift ok");
    match &lift.statements[0] {
        IrStmt::Assign { dst, .. } => assert_eq!(dst.name, "SF"),
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn the_arm_flag_writing_operator_is_still_unsupported() {
    // What the mapping above does *not* buy, stated so nobody reads its
    // presence as coverage. radare2 writes ARM flags with `:=` — the
    // whole tail of `ands r0, r1, r2` is
    // `…,$z,zf,:=,31,$s,nf,:=` — and this machine has no such token, so
    // the lift fails and the ladder falls through to the per-mnemonic
    // handler for **every flag-setting ARM instruction**.
    //
    // Supporting it is not just a parser entry: ARM's ESIL writes ARM's
    // own `C` into `cf`, while this pipeline stores its inverse (see
    // `aarch32_carry_convention_contracts.rs`), so the bridge would
    // need to invert on the ARM path before that family may lift here.
    let err = lift_esil("$z,zf,:=", Arch::Arm).expect_err("`:=` is unsupported");
    assert!(
        matches!(err, EsilError::UnknownToken(ref t) if t == ":="),
        "{err:?}"
    );
}

/// Fold a constant-only expression, covering exactly the operators the
/// operand-order table below exercises. Returns `None` for anything
/// else so a shape change surfaces as a failure rather than a pass.
fn eval_const(expr: &Expr) -> Option<u128> {
    let mask = u128::from(u64::MAX);
    match expr {
        Expr::Const { value, .. } => Some(*value),
        Expr::Sub(a, b) => Some(eval_const(a)?.wrapping_sub(eval_const(b)?) & mask),
        Expr::UDiv(a, b) => eval_const(a)?.checked_div(eval_const(b)?),
        Expr::URem(a, b) => eval_const(a)?.checked_rem(eval_const(b)?),
        Expr::Shl(a, b) => Some((eval_const(a)? << eval_const(b)?) & mask),
        Expr::LShr(a, b) => Some(eval_const(a)? >> eval_const(b)?),
        Expr::Ult(a, b) => Some(u128::from(eval_const(a)? < eval_const(b)?)),
        Expr::Ule(a, b) => Some(u128::from(eval_const(a)? <= eval_const(b)?)),
        _ => None,
    }
}

#[test]
fn non_commutative_operators_match_radare2_operand_order() {
    // ESIL evaluates `a,b,OP` as `b OP a`: the second token is the
    // left-hand side. Every expectation below is the value printed by
    // `r2 -qc '"ae 4,10,<op>"'` on radare2 6.1.8, so each entry is
    // `10 OP 4` and never `4 OP 10`.
    const CASES: &[(&str, &str, u128)] = &[
        ("-", "rax", 6),
        ("/", "rax", 2),
        ("%", "rax", 2),
        ("<<", "rax", 0xa0),
        (">>", "rax", 0),
        ("<", "zf", 0),
        ("<=", "zf", 0),
        (">", "zf", 1),
        (">=", "zf", 1),
    ];
    let measured: Vec<(&str, Option<u128>)> = CASES
        .iter()
        .map(|(op, target, _)| {
            let lift = lift_esil(&format!("4,10,{op},{target},="), Arch::X86_64).expect("lift ok");
            match lift.statements.first() {
                Some(IrStmt::Assign { src, .. }) => (*op, eval_const(src)),
                _ => (*op, None),
            }
        })
        .collect();
    let expected: Vec<(&str, Option<u128>)> = CASES
        .iter()
        .map(|(op, _, want)| (*op, Some(*want)))
        .collect();
    assert_eq!(measured, expected);
}

#[test]
fn the_alphabetic_esil_operators_are_not_registers() {
    // Every one of these is an operator in radare2's own `ae???` table,
    // and each used to parse as a *register* — so `r2,r1,ROR,r0,=` lifted
    // to `r0 = <free value named "ror">`. A fabricated value, not a lost
    // one, and invisible because the lift still returned `Ok`.
    //
    // Prevalence is why this matters: one AArch64 sample emits `DUP`
    // 1 121 times, `ROR` 641, `ASR` 130, `SWAP` 57.
    for op in [
        "DUP", "SWAP", "POP", "ASR", "LSL", "LSR", "ROR", "ROL", "GOTO", "BREAK", "TODO", "TRAP",
        "NAN", "SQRT", "NUM", "CLEAR", "STACK", "BITS", "SETD", "SETJT", "I2D", "U2D", "D2I",
        "D2F", "F2D", "CEIL", "FLOOR", "ROUND",
    ] {
        let err = lift_esil(&format!("r1,r2,{op},r0,="), Arch::Arm)
            .expect_err("an ESIL operator must not lift as a register");
        assert!(
            matches!(err, EsilError::UnknownToken(ref t) if t == op),
            "{op}: {err:?}"
        );
    }
}

#[test]
fn a_lower_case_register_is_still_a_register() {
    // The other direction of the same rule, so the fix cannot be
    // "reject everything alphabetic". radare2 spells register names
    // lower-case, which is what makes the case test safe.
    let lift = lift_esil("rbx,rax,=", Arch::X86_64).expect("lift ok");
    match lift.statements.first() {
        Some(IrStmt::Assign { dst, src }) => {
            assert_eq!(dst.name, "rax");
            assert!(matches!(src, Expr::Var(v) if v.name == "rbx"), "{src:?}");
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}
