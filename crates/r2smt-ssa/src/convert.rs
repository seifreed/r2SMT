//! Single-pass SSA rename for a Phase 4 [`LiftedSlice`].

use std::collections::{BTreeMap, HashMap};

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;
use r2smt_slicer::{BranchCandidate, LiftedSlice, SliceStatus};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// SSA-renamed counterpart of [`LiftedSlice`].
///
/// Every [`Var`] defined inside the slice carries a `#N` suffix on its
/// name (`rax#0`, `rax#1`, `ZF#0`, …). Variables read before any
/// definition keep their plain name and are surfaced in [`inputs`].
///
/// [`inputs`]: SsaLiftedSlice::inputs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SsaLiftedSlice {
    /// The branch the slice belongs to.
    pub branch: BranchCandidate,
    /// SSA-renamed statements, in execution order.
    pub statements: Vec<IrStmt>,
    /// Branch condition expression with reads pointing at the most
    /// recent definitions.
    pub condition: Expr,
    /// Forwarded slice status.
    pub status: SliceStatus,
    /// Forwarded from [`LiftedSlice::treat_truncation_as_inputs`]:
    /// when set, the solver and classifier treat a truncated slice as
    /// if it were complete with the remaining `roots` materialised as
    /// free `Var`s in [`inputs`].
    ///
    /// [`inputs`]: SsaLiftedSlice::inputs
    /// [`LiftedSlice::treat_truncation_as_inputs`]: r2smt_slicer::LiftedSlice
    #[serde(default)]
    pub treat_truncation_as_inputs: bool,
    /// Free symbolic inputs (variables read but never defined inside
    /// the slice). Ordered by name for stable output.
    pub inputs: Vec<Var>,
    /// Variables defined by the slice, in definition order.
    pub defs: Vec<Var>,
    /// Architecture forwarded from the [`LiftedSlice`]. Carried so
    /// register-name resolution downstream (pretty-printer, report)
    /// picks the right ISA table.
    #[serde(default = "default_arch")]
    pub arch: Arch,
}

fn default_arch() -> Arch {
    Arch::X86_64
}

/// Convert a [`LiftedSlice`] into SSA form.
#[must_use]
pub fn ssa_convert(lifted: &LiftedSlice) -> SsaLiftedSlice {
    let mut versions: HashMap<String, u32> = HashMap::new();
    let mut latest: HashMap<String, u32> = HashMap::new();
    let mut inputs: BTreeMap<String, Var> = BTreeMap::new();
    let mut defs: Vec<Var> = Vec::new();
    let mut new_stmts: Vec<IrStmt> = Vec::with_capacity(lifted.statements.len());

    for stmt in &lifted.statements {
        match stmt {
            IrStmt::Assign { dst, src } => {
                let renamed_src = rename_reads(src, &latest, &mut inputs);
                let next_dst = bump_version(dst, &mut versions, &mut latest);
                defs.push(next_dst.clone());
                new_stmts.push(IrStmt::Assign {
                    dst: next_dst,
                    src: renamed_src,
                });
            }
            IrStmt::LoadMem { dst, address, bits } => {
                let renamed_addr = rename_reads(address, &latest, &mut inputs);
                let next_dst = bump_version(dst, &mut versions, &mut latest);
                defs.push(next_dst.clone());
                new_stmts.push(IrStmt::LoadMem {
                    dst: next_dst,
                    address: renamed_addr,
                    bits: *bits,
                });
            }
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => {
                let renamed_addr = rename_reads(address, &latest, &mut inputs);
                let renamed_val = rename_reads(value, &latest, &mut inputs);
                new_stmts.push(IrStmt::StoreMem {
                    address: renamed_addr,
                    value: renamed_val,
                    bits: *bits,
                });
            }
            IrStmt::Unsupported { mnemonic, comment } => {
                new_stmts.push(IrStmt::Unsupported {
                    mnemonic: mnemonic.clone(),
                    comment: comment.clone(),
                });
            }
            IrStmt::Nop => new_stmts.push(IrStmt::Nop),
        }
    }

    let renamed_condition = rename_reads(&lifted.condition, &latest, &mut inputs);

    debug!(
        target: "r2smt::ssa",
        at = %lifted.branch.address,
        statements = new_stmts.len(),
        inputs = inputs.len(),
        defs = defs.len(),
        "ssa conversion complete"
    );

    SsaLiftedSlice {
        branch: lifted.branch.clone(),
        statements: new_stmts,
        condition: renamed_condition,
        status: lifted.status.clone(),
        treat_truncation_as_inputs: lifted.treat_truncation_as_inputs,
        inputs: inputs.into_values().collect(),
        defs,
        arch: lifted.arch,
    }
}

fn bump_version(
    dst: &Var,
    versions: &mut HashMap<String, u32>,
    latest: &mut HashMap<String, u32>,
) -> Var {
    let next = match versions.get(&dst.name) {
        Some(v) => v + 1,
        None => 0,
    };
    versions.insert(dst.name.clone(), next);
    latest.insert(dst.name.clone(), next);
    Var {
        name: format!("{name}#{ver}", name = dst.name, ver = next),
        bits: dst.bits,
    }
}

// Exhaustive dispatch over every `Expr` variant — the function body
// contains no domain logic, only structural recursion through each
// arm of the enum. This matches the documented exception in
// CLAUDE.md (exhaustive dispatch tables on a closed enum) and grows
// linearly with `Expr`'s arms rather than with new behavior.
#[allow(clippy::too_many_lines)]
fn rename_reads(
    expr: &Expr,
    latest: &HashMap<String, u32>,
    inputs: &mut BTreeMap<String, Var>,
) -> Expr {
    match expr {
        Expr::Var(v) => {
            if let Some(ver) = latest.get(&v.name) {
                Expr::Var(Var {
                    name: format!("{name}#{ver}", name = v.name),
                    bits: v.bits,
                })
            } else {
                inputs.entry(v.name.clone()).or_insert_with(|| v.clone());
                Expr::Var(v.clone())
            }
        }
        Expr::Const { value, bits } => Expr::Const {
            value: *value,
            bits: *bits,
        },
        Expr::Add(a, b) => Expr::add(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Sub(a, b) => Expr::sub(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Mul(a, b) => Expr::mul(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::UDiv(a, b) => Expr::udiv(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::URem(a, b) => Expr::urem(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::SDiv(a, b) => Expr::sdiv(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::SRem(a, b) => Expr::srem(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::And(a, b) => Expr::bv_and(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Or(a, b) => Expr::bv_or(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Xor(a, b) => Expr::bv_xor(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Shl(a, b) => Expr::shl(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::LShr(a, b) => Expr::lshr(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::AShr(a, b) => Expr::ashr(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Eq(a, b) => Expr::eq(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Ne(a, b) => Expr::ne(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Ult(a, b) => Expr::ult(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Ule(a, b) => Expr::ule(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Slt(a, b) => Expr::slt(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::Sle(a, b) => Expr::sle(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::BoolAnd(a, b) => Expr::bool_and(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::BoolOr(a, b) => Expr::bool_or(
            rename_reads(a, latest, inputs),
            rename_reads(b, latest, inputs),
        ),
        Expr::BoolNot(inner) => Expr::bool_not(rename_reads(inner, latest, inputs)),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => Expr::Ite {
            cond: Box::new(rename_reads(cond, latest, inputs)),
            then_expr: Box::new(rename_reads(then_expr, latest, inputs)),
            else_expr: Box::new(rename_reads(else_expr, latest, inputs)),
        },
        Expr::Extract { src, hi, lo } => Expr::extract(rename_reads(src, latest, inputs), *hi, *lo),
        Expr::Concat { high, low } => Expr::concat(
            rename_reads(high, latest, inputs),
            rename_reads(low, latest, inputs),
        ),
        Expr::ZeroExtend { src, to_bits } => {
            Expr::zero_ext(rename_reads(src, latest, inputs), *to_bits)
        }
        Expr::SignExtend { src, to_bits } => {
            Expr::sign_ext(rename_reads(src, latest, inputs), *to_bits)
        }
        Expr::Unknown(reason) => Expr::Unknown(reason.clone()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::{
        BranchKind, SliceLimits, SliceStatus, collect_branches, lift_slice, slice_branch,
    };

    use super::*;

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

    fn convert_first(program: &Program, arch: Arch) -> SsaLiftedSlice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        let slice = slice_branch(
            cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        );
        let lifted = lift_slice(&slice, arch);
        ssa_convert(&lifted)
    }

    #[test]
    fn xor_zero_idiom_produces_versioned_rax_no_inputs() {
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
        let ssa = convert_first(&program, Arch::X86);
        // Inputs should be empty — every read is to a previously
        // defined SSA name.
        assert!(ssa.inputs.is_empty(), "inputs: {:?}", ssa.inputs);
        // We should have at least: rax#0, ZF#0, CF#0, SF#0, OF#0, PF#0
        // plus a temp for `test`.
        let names: Vec<String> = ssa.defs.iter().map(|v| v.name.clone()).collect();
        assert!(names.iter().any(|n| n == "rax#0"));
        assert!(names.iter().any(|n| n == "ZF#0"));
        // Branch is `jnz` → condition `(ZF == 0)`; after SSA `ZF` should
        // be the most recent, which is the one defined by `test`.
        assert!(
            format!("{}", ssa.condition).contains("ZF#"),
            "condition: {}",
            ssa.condition
        );
    }

    #[test]
    fn opaque_predicate_chain_versions_rax_three_times() {
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
        let ssa = convert_first(&program, Arch::X86);
        // rcx is the only free input.
        let input_names: Vec<String> = ssa.inputs.iter().map(|v| v.name.clone()).collect();
        assert_eq!(input_names, vec!["rcx".to_string()]);
        // rax should be defined three times: #0 (mov), #1 (imul), #2 (and).
        let rax_defs: Vec<&Var> = ssa
            .defs
            .iter()
            .filter(|v| v.name.starts_with("rax#"))
            .collect();
        assert_eq!(rax_defs.len(), 3, "{rax_defs:?}");
        assert_eq!(rax_defs[0].name, "rax#0");
        assert_eq!(rax_defs[1].name, "rax#1");
        assert_eq!(rax_defs[2].name, "rax#2");
        // Branch condition references the latest ZF (whichever version
        // it landed on — every flag-defining instruction bumps it).
        let cond_str = ssa.condition.to_string();
        assert!(cond_str.contains("ZF#"), "condition: {cond_str}");
        // Status survives.
        assert_eq!(ssa.status, SliceStatus::Complete);
        // Kind survives.
        assert_eq!(ssa.branch.kind, BranchKind::Jcc);
    }

    #[test]
    fn second_imul_reads_first_imul_result() {
        // mov eax, ecx ; imul eax, eax ; imul eax, eax ; test eax,eax ; jnz
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
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1008,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_100a,
                6,
                "jnz",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = convert_first(&program, Arch::X86);
        let rax_defs: Vec<&Var> = ssa
            .defs
            .iter()
            .filter(|v| v.name.starts_with("rax#"))
            .collect();
        // We need at least 3 rax versions.
        assert!(rax_defs.len() >= 3, "{rax_defs:?}");
        // Each successive imul must read the previous version.
        // Walk the SSA statements and look for an assign whose dst is
        // `rax#1` and whose src mentions `rax#0`.
        let stmt_strings: Vec<String> = ssa
            .statements
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let imul1 = stmt_strings.iter().find(|s| s.starts_with("rax#1 :="));
        let imul2 = stmt_strings.iter().find(|s| s.starts_with("rax#2 :="));
        let imul1 = imul1.expect("rax#1 statement");
        let imul2 = imul2.expect("rax#2 statement");
        assert!(imul1.contains("rax#0"), "{imul1}");
        assert!(imul2.contains("rax#1"), "{imul2}");
    }

    #[test]
    fn read_before_def_is_reported_as_input() {
        // test eax, eax ; je junk — eax is never defined inside the slice.
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = convert_first(&program, Arch::X86);
        let input_names: Vec<String> = ssa.inputs.iter().map(|v| v.name.clone()).collect();
        assert_eq!(input_names, vec!["rax".to_string()]);
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
        let ssa = convert_first(&program, Arch::X86);
        let json = serde_json::to_string(&ssa).unwrap();
        let back: SsaLiftedSlice = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ssa);
    }

    #[test]
    fn truncated_status_is_preserved() {
        // Construct a slice that the slicer truncates (call before cmp).
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let ssa = convert_first(&program, Arch::X86);
        assert!(matches!(ssa.status, SliceStatus::Truncated { .. }));
    }
}
