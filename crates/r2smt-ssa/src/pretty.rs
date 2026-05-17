//! Pretty-print the branch condition with SSA definitions substituted
//! into the expression tree.
//!
//! The default `Display` impl for [`Expr`] renders the IR verbatim,
//! which means the condition shows up as `ZF#1 == 0` — useful for
//! debugging the SSA pass but unhelpful for analysts who want to see
//! the arithmetic that actually drives the branch.
//!
//! [`pretty_condition`] walks [`SsaLiftedSlice::condition`] and, for
//! every [`Expr::Var`] reference, follows the chain of `IrStmt::Assign`
//! statements back to the original input or constant. Variables that
//! cannot be resolved (free inputs, the right-hand side of unsupported
//! statements, memory loads, …) are rendered with their SSA name
//! unchanged so the output still uniquely identifies the symbol.
//!
//! Substitution depth is bounded to defend against pathological IRs;
//! the bound only matters for malformed input — well-formed SSA never
//! creates cycles.

use std::collections::HashMap;
use std::fmt::Write as _;

use r2smt_common::Arch;
use r2smt_ir::expr::Expr;
use r2smt_ir::name_hints::NameHints;
use r2smt_ir::stmt::IrStmt;

use crate::convert::SsaLiftedSlice;

const MAX_DEPTH: usize = 64;

/// Render the slice's condition with every SSA reference substituted by
/// its defining expression (recursive substitution is bounded to keep
/// pathological cycles from blowing the stack).
///
/// Result format mirrors [`Expr`]'s `Display` impl — bit-vector arithmetic and
/// comparisons appear in infix notation, with constants annotated by
/// their bit width (e.g. `0x2:32`). Free inputs keep their SSA name.
#[must_use]
pub fn pretty_condition(slice: &SsaLiftedSlice) -> String {
    let hints = NameHints::default();
    pretty_condition_with_hints(slice, &hints)
}

/// Variant of [`pretty_condition`] that consults `hints` to swap
/// canonical names (`stk_rbp_-4`, `stk_rsp_+8`) for their r2-supplied
/// aliases (`var_4h`, `arg_8h`, …) when they exist. The canonical
/// name is appended in square brackets so the analyst still has a
/// stable handle into the IR.
#[must_use]
pub fn pretty_condition_with_hints(slice: &SsaLiftedSlice, hints: &NameHints) -> String {
    let defs = collect_defs(&slice.statements);
    let ctx = FmtCtx {
        defs: &defs,
        hints,
        arch: slice.arch,
    };
    let mut out = String::new();
    write_expr(&mut out, &slice.condition, &ctx, 0);
    out
}

/// Bundle the read-only context shared by every internal helper —
/// keeps argument counts at each call site below the
/// `clippy::too_many_arguments` threshold while threading the
/// arch-aware data-flow downstream.
struct FmtCtx<'a> {
    defs: &'a HashMap<String, &'a Expr>,
    hints: &'a NameHints,
    arch: Arch,
}

fn collect_defs(statements: &[IrStmt]) -> HashMap<String, &Expr> {
    let mut out: HashMap<String, &Expr> = HashMap::new();
    for stmt in statements {
        if let IrStmt::Assign { dst, src } = stmt {
            out.insert(dst.name.clone(), src);
        }
    }
    out
}

fn write_expr(out: &mut String, expr: &Expr, ctx: &FmtCtx<'_>, depth: usize) {
    if depth >= MAX_DEPTH {
        let _ = write!(out, "<depth-limit>");
        return;
    }
    match expr {
        Expr::Var(v) => match ctx.defs.get(&v.name) {
            Some(src) => write_expr(out, src, ctx, depth + 1),
            None => write_var_name(out, &v.name, ctx),
        },
        Expr::Const { value, bits } => {
            let _ = write!(out, "{value:#x}:{bits}");
        }
        Expr::Add(a, b) => write_binary(out, ctx, depth, a, "+", b),
        Expr::Sub(a, b) => write_binary(out, ctx, depth, a, "-", b),
        Expr::Mul(a, b) => write_binary(out, ctx, depth, a, "*", b),
        Expr::UDiv(a, b) => write_binary(out, ctx, depth, a, "/u", b),
        Expr::URem(a, b) => write_binary(out, ctx, depth, a, "%u", b),
        Expr::SDiv(a, b) => write_binary(out, ctx, depth, a, "/s", b),
        Expr::SRem(a, b) => write_binary(out, ctx, depth, a, "%s", b),
        Expr::And(a, b) => write_binary(out, ctx, depth, a, "&", b),
        Expr::Or(a, b) => write_binary(out, ctx, depth, a, "|", b),
        Expr::Xor(a, b) => write_binary(out, ctx, depth, a, "^", b),
        Expr::Shl(a, b) => write_binary(out, ctx, depth, a, "<<", b),
        Expr::LShr(a, b) => write_binary(out, ctx, depth, a, ">>u", b),
        Expr::AShr(a, b) => write_binary(out, ctx, depth, a, ">>s", b),
        Expr::Eq(a, b) => write_binary(out, ctx, depth, a, "==", b),
        Expr::Ne(a, b) => write_binary(out, ctx, depth, a, "!=", b),
        Expr::Ult(a, b) => write_binary(out, ctx, depth, a, "<u", b),
        Expr::Ule(a, b) => write_binary(out, ctx, depth, a, "<=u", b),
        Expr::Slt(a, b) => write_binary(out, ctx, depth, a, "<s", b),
        Expr::Sle(a, b) => write_binary(out, ctx, depth, a, "<=s", b),
        Expr::BoolAnd(a, b) => write_binary(out, ctx, depth, a, "&&", b),
        Expr::BoolOr(a, b) => write_binary(out, ctx, depth, a, "||", b),
        Expr::BoolNot(e) => {
            let _ = write!(out, "!(");
            write_expr(out, e, ctx, depth + 1);
            let _ = write!(out, ")");
        }
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            let _ = write!(out, "ite(");
            write_expr(out, cond, ctx, depth + 1);
            let _ = write!(out, ", ");
            write_expr(out, then_expr, ctx, depth + 1);
            let _ = write!(out, ", ");
            write_expr(out, else_expr, ctx, depth + 1);
            let _ = write!(out, ")");
        }
        Expr::Extract { src, hi, lo } => {
            write_extract(out, src, *hi, *lo, ctx, depth);
        }
        Expr::Concat { high, low } => {
            let _ = write!(out, "concat(");
            write_expr(out, high, ctx, depth + 1);
            let _ = write!(out, ", ");
            write_expr(out, low, ctx, depth + 1);
            let _ = write!(out, ")");
        }
        Expr::ZeroExtend { src, to_bits } => {
            let _ = write!(out, "zext(");
            write_expr(out, src, ctx, depth + 1);
            let _ = write!(out, ", {to_bits})");
        }
        Expr::SignExtend { src, to_bits } => {
            let _ = write!(out, "sext(");
            write_expr(out, src, ctx, depth + 1);
            let _ = write!(out, ", {to_bits})");
        }
        Expr::Unknown(reason) => {
            if reason.is_empty() {
                let _ = write!(out, "?");
            } else {
                let _ = write!(out, "?({reason})");
            }
        }
    }
}

/// Render `Extract(src, hi, lo)` as either the named sub-register
/// alias (when `src` is a free input and the slice matches a known
/// register — e.g. `Extract(rcx_free, 31, 0)` → `ecx`) or as the
/// substituted expression with a trailing `[hi:lo]` decoration so the
/// analyst can follow the data-flow all the way back to its origin.
fn write_extract(out: &mut String, src: &Expr, hi: u8, lo: u8, ctx: &FmtCtx<'_>, depth: usize) {
    if let Expr::Var(v) = src
        && !ctx.defs.contains_key(&v.name)
        && let Some(alias) =
            r2smt_slicer::registers::alias_for(strip_ssa_suffix(&v.name), hi, lo, ctx.arch)
    {
        let _ = write!(out, "{alias}");
        return;
    }
    write_expr(out, src, ctx, depth + 1);
    let _ = write!(out, "[{hi}:{lo}]");
}

fn strip_ssa_suffix(name: &str) -> &str {
    name.split_once('#').map_or(name, |(base, _)| base)
}

fn write_var_name(out: &mut String, canonical: &str, ctx: &FmtCtx<'_>) {
    // Strip the SSA `#N` suffix so user-facing output never carries
    // version numbers. Stack-slot names route through the analyst
    // alias (`var_4h`). Register names route through `hints.register`
    // — which today is identity, but the hook lets future adapters
    // surface analyst-supplied register aliases without touching the
    // pretty-printer.
    let base = strip_ssa_suffix(canonical);
    if base.starts_with("stk_") {
        let alias = ctx.hints.stack_slot(base);
        if alias != base {
            let _ = write!(out, "{alias}[{base}]");
            return;
        }
        let _ = write!(out, "{base}");
        return;
    }
    if r2smt_slicer::register_layout(base, ctx.arch).is_some() {
        let _ = write!(out, "{}", ctx.hints.register(base));
        return;
    }
    let _ = write!(out, "{base}");
}

fn write_binary(
    out: &mut String,
    ctx: &FmtCtx<'_>,
    depth: usize,
    lhs: &Expr,
    op: &str,
    rhs: &Expr,
) {
    let _ = write!(out, "(");
    write_expr(out, lhs, ctx, depth + 1);
    let _ = write!(out, " {op} ");
    write_expr(out, rhs, ctx, depth + 1);
    let _ = write!(out, ")");
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};

    use super::*;
    use crate::convert::ssa_convert;

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

    fn one_block(insns: Vec<Instruction>) -> Program {
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

    fn ssa_first(program: &Program) -> SsaLiftedSlice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        let slice = slice_branch(
            cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        );
        let lifted = lift_slice(&slice, program.arch);
        ssa_convert(&lifted)
    }

    #[test]
    fn flat_flag_predicate_uses_input_names_when_no_def() {
        // Empty slice — the branch's flags are free inputs. The pretty
        // form should still emit the flag predicate.
        let program = one_block(vec![insn(
            0x40_1000,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        )]);
        let ssa = ssa_first(&program);
        let pretty = pretty_condition(&ssa);
        assert!(pretty.contains("ZF"));
        assert!(pretty.contains("=="));
    }

    #[test]
    fn opaque_predicate_substitutes_chain_back_to_input() {
        // The canonical SCC example: mov eax, ecx; imul eax, eax;
        // and eax, 1; cmp eax, 2; jne — the pretty form should
        // expose `ecx`-derived arithmetic, not just `ZF#N`.
        let program = one_block(vec![
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
        let ssa = ssa_first(&program);
        let pretty = pretty_condition(&ssa);
        // The expression must reference the original input register at
        // the sub-register slice it was read at — `ecx` is the 32-bit
        // view of `rcx`, and that's how the lifter surfaces it via the
        // `Extract(rcx, 31, 0)` alias lookup.
        assert!(
            pretty.contains("ecx") || pretty.contains("rcx"),
            "pretty form must surface the rcx/ecx input; got: {pretty}"
        );
        // It must keep the arithmetic shape — multiplication and the
        // constant `1` (bit mask) should both be visible.
        assert!(
            pretty.contains('*'),
            "pretty form must surface the multiplication; got: {pretty}"
        );
        assert!(
            pretty.contains("0x1:"),
            "pretty form must surface the AND mask; got: {pretty}"
        );
    }

    #[test]
    fn aarch64_extract_renders_as_wn_alias() {
        // Hand-build a synthetic SSA slice flagged as AArch64. The
        // condition references `Extract(x0, 31, 0)` against a free
        // input — `alias_for("x0", 31, 0, Arch::Aarch64) == Some("w0")`,
        // so the pretty form must surface `w0`, not `x0[31:0]`.
        use r2smt_common::Address;
        use r2smt_ir::expr::{Expr, Var};
        use r2smt_slicer::condition::{BranchCondition, BranchKind};
        use r2smt_slicer::{BranchCandidate, SliceStatus};

        let cond = Expr::eq(
            Expr::extract(Expr::var("x0", 64), 31, 0),
            Expr::konst(0, 32),
        );
        let ssa = SsaLiftedSlice {
            branch: BranchCandidate {
                address: Address(0x40_1000),
                function: Address(0x40_1000),
                block: Address(0x40_1000),
                kind: BranchKind::Jcc,
                mnemonic: "b.eq".into(),
                condition: BranchCondition::Equal,
                formula: String::new(),
                taken_target: None,
                fallthrough_target: None,
                compare_register: None,
                bit_index: None,
                upstream_resolved: None,
                operand_raws: Vec::new(),
                is_thumb: false,
            },
            statements: Vec::new(),
            condition: cond,
            status: SliceStatus::Complete,
            treat_truncation_as_inputs: false,
            inputs: vec![Var::new("x0", 64)],
            defs: Vec::new(),
            arch: Arch::Aarch64,
        };
        let pretty = pretty_condition(&ssa);
        assert!(
            pretty.contains("w0"),
            "AArch64 Extract(x0, 31, 0) must render as `w0`; got: {pretty}"
        );
        // The output must not leak x86 register names from the same
        // (parent, hi, lo) shape (e.g. `eax`) — that would mean the
        // alias dispatch picked the wrong arch table.
        assert!(
            !pretty.contains("eax"),
            "AArch64 render must not surface x86 aliases; got: {pretty}"
        );
    }

    #[test]
    fn register_alias_in_hints_overrides_canonical_name() {
        // The pretty-printer must consult `NameHints.registers` when
        // rendering a free input. This is the user-visible payoff of
        // wiring `afvj.reg[]` entries from radare2 through to
        // `NameHints` — analyst-named arguments (`arg1`, `userInput`)
        // should appear instead of the bare register.
        let program = one_block(vec![
            insn(
                0x40_1000,
                3,
                "cmp",
                vec![
                    op("rax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1003,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = ssa_first(&program);
        let mut hints = NameHints::default();
        hints.add_register("rax", "userInput");
        let pretty = pretty_condition_with_hints(&ssa, &hints);
        assert!(
            pretty.contains("userInput"),
            "register alias must appear in pretty output; got: {pretty}"
        );
        assert!(
            !pretty.contains("rax"),
            "canonical register name must be replaced; got: {pretty}"
        );
    }

    #[test]
    fn ssa_suffix_is_stripped_from_pretty_output() {
        let program = one_block(vec![
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
        let ssa = ssa_first(&program);
        let pretty = pretty_condition(&ssa);
        assert!(
            !pretty.contains('#'),
            "pretty form must not surface SSA suffixes; got: {pretty}"
        );
    }
}
