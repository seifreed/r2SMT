//! SMT-LIB2 emitter for a [`SsaLiftedSlice`].
//!
//! Produces a self-contained SMT-LIB2 script that any SMT solver
//! understanding the `QF_BV` theory can consume. The script declares
//! every free input, every SSA-assigned variable, asserts each
//! `IrStmt::Assign` as an equality, and pushes / pops a single
//! query for the branch condition's truth value.
//!
//! Two scripts are produced per branch: one assuming the condition
//! is `true`, one assuming `false`. Combining the two `(check-sat)`
//! outcomes gives the same verdict ladder as the Z3 backend
//! ([`crate::SmtResult`]).
//!
//! Output uses the `QF_BV` logic — no quantifiers, only the
//! fixed-width bit-vector theory the slicer's IR maps onto.

use std::collections::HashMap;
use std::fmt::Write as _;

use r2smt_common::smt::SolveOptions;
use r2smt_ir::expr::Expr;
use r2smt_ir::stmt::IrStmt;
use r2smt_ssa::SsaLiftedSlice;

/// Render the slice's preamble (declarations + statement assertions)
/// into SMT-LIB2 text, leaving the branch-condition query to the
/// caller. The output ends with a blank line so the caller can
/// append a `(push)`/`(assert <cond>)`/`(check-sat)`/`(pop)` block
/// directly.
#[must_use]
pub fn emit_preamble(slice: &SsaLiftedSlice, options: &SolveOptions) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "(set-logic QF_BV)");
    let _ = writeln!(out, "(set-option :produce-models false)");
    // Pin the subprocess backends' PRNG (standard SMT-LIB option,
    // honoured by Z3 / CVC5 / Bitwuzla) so a query's verdict is
    // reproducible run-to-run, mirroring the Z3 FFI `random_seed`.
    let _ = writeln!(
        out,
        "(set-option :random-seed {seed})",
        seed = options.random_seed
    );
    let _ = writeln!(out, "(set-info :status unknown)");
    let _ = writeln!(out, "; r2smt slice @ {addr}", addr = slice.branch.address);
    let _ = writeln!(out, "; timeout-ms: {ms}", ms = options.timeout_ms);
    let mut declared: Vec<String> = Vec::new();
    let mut mem = TextMemory::new(ptr_bits_for(slice.arch));
    for var in &slice.inputs {
        declare_bv(&mut out, &var.name, var.bits, &mut declared);
    }
    for stmt in &slice.statements {
        match stmt {
            IrStmt::Assign { dst, src } => {
                declare_bv(&mut out, &dst.name, dst.bits, &mut declared);
                let rhs = render_expr_with_width(src, dst.bits);
                let _ = writeln!(out, "(assert (= {lhs} {rhs}))", lhs = dst.name);
            }
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => mem.record_store(address, value, *bits),
            IrStmt::LoadMem { dst, address, bits } => {
                declare_bv(&mut out, &dst.name, dst.bits, &mut declared);
                mem.emit_load(&mut out, &mut declared, dst, address, *bits);
            }
            // Unsupported / Nop emit nothing; the SMT side stays an
            // over-approximation for those (extra freedom can only widen
            // `AlwaysX` to `BothPossible`, never fabricate one).
            IrStmt::Unsupported { .. } | IrStmt::Nop => {}
        }
    }
    let _ = writeln!(out);
    out
}

/// Pointer width in bits for the target architecture, matching the Z3
/// encoder's `ptr_bits`.
fn ptr_bits_for(arch: r2smt_common::Arch) -> u16 {
    match arch {
        r2smt_common::Arch::X86_64 | r2smt_common::Arch::Aarch64 => 64,
        _ => 32,
    }
}

/// Convenience helper combining [`emit_preamble`] with the
/// branch-condition query block. Two scripts are produced; the
/// caller decides which polarity to run first (typical pattern:
/// run "taken" first, then "not-taken").
#[must_use]
pub fn emit_query(slice: &SsaLiftedSlice, options: &SolveOptions, polarity: bool) -> String {
    let mut script = emit_preamble(slice, options);
    let cond = render_expr_with_width(&slice.condition, 1);
    let assertion = if polarity {
        format!("(= {cond} #b1)")
    } else {
        format!("(= {cond} #b0)")
    };
    let _ = writeln!(&mut script, "(assert {assertion})");
    let _ = writeln!(&mut script, "(check-sat)");
    script
}

/// Byte-store cap for the textual backend, matching the Z3 encoder's
/// `MEM_BYTE_STORE_CAP`. Overflow havocs the store list.
const MEM_BYTE_STORE_CAP: usize = 4096;

/// Byte-granular memory model for the textual SMT-LIB backend, mirroring
/// the Z3 encoder's `Ite`-chain (`encoder.rs`). Each `StoreMem` enqueues
/// its bytes little-endian; each `LoadMem` reads latest-write-wins by
/// folding one `ite` per byte over the store list **oldest → latest**,
/// so the latest write is the outermost `ite`. Bounded by
/// [`MEM_BYTE_STORE_CAP`]: overflow havocs the list and subsequent loads
/// read fresh free bytes — precision lost, soundness preserved.
struct TextMemory {
    stores: Vec<(String, String)>,
    havoced: bool,
    load_counter: u32,
    ptr_bits: u16,
    /// Load memo restoring read-read consistency, mirroring the Z3
    /// encoder's `load_memo`. Keyed by (canonical address rendering,
    /// width) → the destination name of the first identical load;
    /// cleared on every store and on havoc.
    load_memo: HashMap<(String, u16), String>,
}

impl TextMemory {
    fn new(ptr_bits: u16) -> Self {
        Self {
            stores: Vec::new(),
            havoced: false,
            load_counter: 0,
            ptr_bits,
            load_memo: HashMap::new(),
        }
    }

    fn byte_addr(&self, addr: &str, i: u16) -> String {
        if i == 0 {
            addr.to_string()
        } else {
            format!("(bvadd {addr} (_ bv{i} {bits}))", bits = self.ptr_bits)
        }
    }

    fn record_store(&mut self, address: &Expr, value: &Expr, bits: u16) {
        if bits == 0 {
            return;
        }
        // Any store may alias a memoized load; drop the memo.
        self.load_memo.clear();
        let nbytes = usize::from(bits.div_ceil(8));
        if self.stores.len().saturating_add(nbytes) > MEM_BYTE_STORE_CAP {
            self.stores.clear();
            self.havoced = true;
            return;
        }
        let addr = render_expr_with_width(address, self.ptr_bits);
        let value_s = render_expr_with_width(value, bits);
        for i in 0..u16::try_from(nbytes).unwrap_or(u16::MAX) {
            let byte_addr = self.byte_addr(&addr, i);
            let lo = u32::from(i) * 8;
            let hi = lo + 7;
            let byte = format!("((_ extract {hi} {lo}) {value_s})");
            self.stores.push((byte_addr, byte));
        }
    }

    fn emit_load(
        &mut self,
        out: &mut String,
        declared: &mut Vec<String>,
        dst: &r2smt_ir::expr::Var,
        address: &Expr,
        bits: u16,
    ) {
        if bits == 0 {
            return;
        }
        // Read-read consistency: reuse a prior identical load's
        // destination when no store has invalidated the memo since.
        let memo_key = (render_expr_with_width(address, self.ptr_bits), bits);
        if let Some(cached) = self.load_memo.get(&memo_key) {
            let _ = writeln!(out, "(assert (= {lhs} {cached}))", lhs = dst.name);
            return;
        }
        let nbytes = bits.div_ceil(8);
        let addr = render_expr_with_width(address, self.ptr_bits);
        let load_id = self.load_counter;
        self.load_counter = self.load_counter.wrapping_add(1);
        let mut acc: Option<String> = None;
        for i in 0..nbytes {
            let byte_addr = self.byte_addr(&addr, i);
            let fresh = format!("load_{load_id}_{i}");
            declare_bv(out, &fresh, 8, declared);
            let mut byte_val = fresh;
            if !self.havoced {
                for (store_addr, store_byte) in &self.stores {
                    byte_val =
                        format!("(ite (= {byte_addr} {store_addr}) {store_byte} {byte_val})");
                }
            }
            acc = Some(match acc.take() {
                None => byte_val,
                Some(prev) => format!("(concat {byte_val} {prev})"),
            });
        }
        let Some(loaded) = acc else {
            return;
        };
        let total_bits = nbytes.saturating_mul(8);
        let coerced = coerce(&loaded, total_bits, bits);
        let _ = writeln!(out, "(assert (= {lhs} {coerced}))", lhs = dst.name);
        self.load_memo.insert(memo_key, dst.name.clone());
    }
}

fn declare_bv(out: &mut String, name: &str, bits: u16, declared: &mut Vec<String>) {
    if declared.iter().any(|n| n == name) {
        return;
    }
    let _ = writeln!(out, "(declare-fun {name} () (_ BitVec {bits}))");
    declared.push(name.to_string());
}

/// Render an expression to SMT-LIB2 text, widening / narrowing to
/// the target bit width before returning. Used at every assertion
/// boundary so the produced SMT-LIB stays well-typed.
fn render_expr_with_width(expr: &Expr, target_bits: u16) -> String {
    let (rendered, bits) = render_expr(expr);
    coerce(&rendered, bits, target_bits)
}

// `clippy::too_many_lines`: exhaustive dispatch over `Expr`'s variants —
// each arm is a 1-3 line emitter; splitting per-arm would hide the parity
// between the variant set and the SMT-LIB renderer without removing any
// logic.
#[allow(clippy::too_many_lines)]
fn render_expr(expr: &Expr) -> (String, u16) {
    match expr {
        Expr::Var(v) => (v.name.clone(), v.bits),
        Expr::Const { value, bits } => (format!("(_ bv{value} {bits})"), *bits),
        Expr::Add(a, b) => bin_op("bvadd", a, b, Signedness::Unsigned),
        Expr::Sub(a, b) => bin_op("bvsub", a, b, Signedness::Unsigned),
        Expr::Mul(a, b) => bin_op("bvmul", a, b, Signedness::Unsigned),
        Expr::UDiv(a, b) => bin_op("bvudiv", a, b, Signedness::Unsigned),
        Expr::URem(a, b) => bin_op("bvurem", a, b, Signedness::Unsigned),
        Expr::SDiv(a, b) => bin_op("bvsdiv", a, b, Signedness::Signed),
        Expr::SRem(a, b) => bin_op("bvsrem", a, b, Signedness::Signed),
        Expr::And(a, b) => bin_op("bvand", a, b, Signedness::Unsigned),
        Expr::Or(a, b) => bin_op("bvor", a, b, Signedness::Unsigned),
        Expr::Xor(a, b) => bin_op("bvxor", a, b, Signedness::Unsigned),
        Expr::Shl(a, b) => bin_op("bvshl", a, b, Signedness::Unsigned),
        Expr::LShr(a, b) => bin_op("bvlshr", a, b, Signedness::Unsigned),
        Expr::AShr(a, b) => bin_op("bvashr", a, b, Signedness::Unsigned),
        Expr::Eq(a, b) => bool_op("=", a, b, Signedness::Unsigned),
        Expr::Ne(a, b) => bool_op("distinct", a, b, Signedness::Unsigned),
        Expr::Ult(a, b) => bool_op("bvult", a, b, Signedness::Unsigned),
        Expr::Ule(a, b) => bool_op("bvule", a, b, Signedness::Unsigned),
        Expr::Slt(a, b) => bool_op("bvslt", a, b, Signedness::Signed),
        Expr::Sle(a, b) => bool_op("bvsle", a, b, Signedness::Signed),
        Expr::BoolAnd(a, b) => bool_combiner("and", a, b),
        Expr::BoolOr(a, b) => bool_combiner("or", a, b),
        Expr::BoolNot(inner) => {
            let r = bool_of(inner);
            (format!("(ite (not {r}) #b1 #b0)"), 1)
        }
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond_str = bool_of(cond);
            let (then_str, then_bits) = render_expr(then_expr);
            let (else_str, else_bits) = render_expr(else_expr);
            let target = then_bits.max(else_bits);
            let then_coerced = coerce(&then_str, then_bits, target);
            let else_coerced = coerce(&else_str, else_bits, target);
            (
                format!("(ite {cond_str} {then_coerced} {else_coerced})"),
                target,
            )
        }
        Expr::Extract { src, hi, lo } => {
            let (src_str, _) = render_expr(src);
            let width = hi - lo + 1;
            (format!("((_ extract {hi} {lo}) {src_str})"), width)
        }
        Expr::Concat { high, low } => {
            let (high_str, hb) = render_expr(high);
            let (low_str, lb) = render_expr(low);
            (format!("(concat {high_str} {low_str})"), hb + lb)
        }
        Expr::ZeroExtend { src, to_bits } => {
            let (src_str, cur) = render_expr(src);
            (coerce(&src_str, cur, *to_bits), *to_bits)
        }
        Expr::SignExtend { src, to_bits } => {
            let (src_str, cur) = render_expr(src);
            match cur.cmp(to_bits) {
                std::cmp::Ordering::Equal => (src_str, *to_bits),
                std::cmp::Ordering::Less => (
                    format!("((_ sign_extend {n}) {src_str})", n = to_bits - cur),
                    *to_bits,
                ),
                std::cmp::Ordering::Greater => {
                    let hi = to_bits - 1;
                    (format!("((_ extract {hi} 0) {src_str})"), *to_bits)
                }
            }
        }
        // `Unknown` has no sound text encoding here: a constant under-
        // approximates the value set (unlike the Z3 backend's fresh
        // free var). This constant placeholder only keeps the script
        // parseable for render-only consumers; the CVC5 *verdict* path
        // (`crate::cvc5::solve_branch_cvc5`) declines any slice that
        // contains an `Unknown` before this is ever solved, so no
        // verdict is derived from this placeholder.
        Expr::Unknown(_) => ("(_ bv0 1)".to_string(), 1),
        // Floating point has no sound QF_BV text encoding. Like
        // `Unknown`, these placeholders only keep the script parseable
        // for render-only consumers; the CVC5 / Bitwuzla verdict paths
        // decline any slice containing a float (`slice_contains_float`)
        // before this is ever solved, so no verdict is derived from a
        // placeholder. Precise `QF_BVFP` rendering lands in P40-f-4.
        Expr::FAdd(..) | Expr::FSub(..) | Expr::FMul(..) | Expr::FDiv(..) | Expr::FpConst { .. }
        | Expr::BvToFp { .. } | Expr::FpToIeeeBv(_) | Expr::SbvToFp { .. } => {
            ("(_ bv0 1)".to_string(), 1)
        }
        Expr::FpToSbv { bits, .. } => (format!("(_ bv0 {bits})"), *bits),
        Expr::FEq(..) | Expr::FLt(..) | Expr::FLe(..) | Expr::FIsNaN(_) => {
            ("(_ bv0 1)".to_string(), 1)
        }
    }
}

fn bin_op(name: &str, a: &Expr, b: &Expr, sign: Signedness) -> (String, u16) {
    let (a_str, a_bits) = render_expr(a);
    let (b_str, b_bits) = render_expr(b);
    let target = a_bits.max(b_bits);
    let lhs = coerce_with_sign(&a_str, a_bits, target, sign);
    let rhs = coerce_with_sign(&b_str, b_bits, target, sign);
    (format!("({name} {lhs} {rhs})"), target)
}

fn bool_op(name: &str, a: &Expr, b: &Expr, sign: Signedness) -> (String, u16) {
    let (a_str, a_bits) = render_expr(a);
    let (b_str, b_bits) = render_expr(b);
    let target = a_bits.max(b_bits);
    let lhs = coerce_with_sign(&a_str, a_bits, target, sign);
    let rhs = coerce_with_sign(&b_str, b_bits, target, sign);
    // Wrap the SMT boolean back into a 1-bit BV so the encoder can
    // splice the result wherever a bit-vector is expected.
    (format!("(ite ({name} {lhs} {rhs}) #b1 #b0)"), 1)
}

/// Whether a binary operation interprets its operands as signed or
/// unsigned when one of them needs to be widened to match the other.
#[derive(Debug, Clone, Copy)]
enum Signedness {
    Signed,
    Unsigned,
}

fn coerce_with_sign(rendered: &str, cur: u16, target: u16, sign: Signedness) -> String {
    if cur >= target {
        return coerce(rendered, cur, target);
    }
    let n = target - cur;
    match sign {
        Signedness::Signed => format!("((_ sign_extend {n}) {rendered})"),
        Signedness::Unsigned => format!("((_ zero_extend {n}) {rendered})"),
    }
}

fn bool_combiner(name: &str, a: &Expr, b: &Expr) -> (String, u16) {
    let a_bool = bool_of(a);
    let b_bool = bool_of(b);
    (format!("(ite ({name} {a_bool} {b_bool}) #b1 #b0)"), 1)
}

fn bool_of(expr: &Expr) -> String {
    let (rendered, bits) = render_expr(expr);
    if bits == 1 {
        format!("(= {rendered} #b1)")
    } else {
        format!("(distinct {rendered} (_ bv0 {bits}))")
    }
}

fn coerce(rendered: &str, cur: u16, target: u16) -> String {
    if cur == target {
        return rendered.to_string();
    }
    if cur < target {
        let n = target - cur;
        return format!("((_ zero_extend {n}) {rendered})");
    }
    let hi = target - 1;
    format!("((_ extract {hi} 0) {rendered})")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use r2smt_common::smt::SolveOptions;
    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
    use r2smt_ssa::ssa_convert;

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

    fn build_ssa(program: &Program) -> SsaLiftedSlice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("branch");
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
    fn smtlib_query_for_constant_propagation_has_check_sat_line() {
        // mov eax, 1 ; cmp eax, 1 ; je dest — same fixture used by
        // the Z3 backend tests, but here we only verify the SMT-LIB
        // script structure (logic, declarations, check-sat).
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        insn(
                            0x40_1000,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1005,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1008,
                            6,
                            "je",
                            vec![op("0x401080", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        };
        let ssa = build_ssa(&program);
        let script = emit_query(&ssa, &SolveOptions::default(), true);
        assert!(script.starts_with("(set-logic QF_BV)"));
        assert!(script.contains("(check-sat)"));
        assert!(script.contains("(declare-fun "));
        assert!(script.contains("(assert ("));
    }

    fn mem_slice(statements: Vec<IrStmt>) -> SsaLiftedSlice {
        use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus};
        let z = Address::new(0x1000);
        SsaLiftedSlice {
            branch: BranchCandidate {
                address: z,
                function: z,
                block: z,
                kind: BranchKind::Jcc,
                mnemonic: "memtest".into(),
                condition: BranchCondition::NotEqual,
                formula: "memtest".into(),
                taken_target: None,
                fallthrough_target: None,
                compare_register: None,
                bit_index: None,
                upstream_resolved: None,
                operand_raws: Vec::new(),
                is_thumb: false,
            },
            statements,
            condition: Expr::konst(0, 1),
            status: SliceStatus::Complete,
            treat_truncation_as_inputs: false,
            inputs: Vec::new(),
            defs: Vec::new(),
            arch: Arch::Aarch64,
        }
    }

    #[test]
    fn smtlib_store_then_load_emits_byte_granular_ite_chain() {
        // A store to `sp` followed by a load from `sp` must constrain
        // the loaded value via the alias `ite` chain — the same
        // precision the Z3 backend has, not a free variable.
        use r2smt_ir::expr::Var;
        let sp = Expr::Var(Var::new("sp", 64));
        let ssa = mem_slice(vec![
            IrStmt::StoreMem {
                address: sp.clone(),
                value: Expr::konst(5, 64),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: Var::new("t0", 64),
                address: sp,
                bits: 64,
            },
        ]);
        let script = emit_preamble(&ssa, &SolveOptions::default());
        assert!(
            script.contains("(ite "),
            "expected alias ite chain: {script}"
        );
        assert!(
            script.contains("(concat "),
            "expected byte concat: {script}"
        );
    }

    #[test]
    fn smtlib_load_without_prior_store_stays_a_free_value() {
        // No store → the load reads a fresh free byte (no ite chain).
        use r2smt_ir::expr::Var;
        let ssa = mem_slice(vec![IrStmt::LoadMem {
            dst: Var::new("t0", 64),
            address: Expr::Var(Var::new("sp", 64)),
            bits: 64,
        }]);
        let script = emit_preamble(&ssa, &SolveOptions::default());
        assert!(
            !script.contains("(ite "),
            "a load with no prior store must not build an ite chain: {script}"
        );
    }
}
