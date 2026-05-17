//! P-code → `IrStmt` lifter (strict, sound subset).
//!
//! Soundness boundary: only the **Z** flag is mapped to the canonical
//! `ZF` the branch-condition composer reads (P-code `ZR` ≡ zero, no
//! polarity ambiguity). P-code `NG`/`CY`/`OV` are *not* mapped to the
//! canonical `SF`/`CF`/`OF`, because ARM `NZCV` polarity differs from
//! the per-mnemonic `AArch64` model and a name-level merge would be
//! unsound. They are lifted into distinct `pc_*` vars instead, so a
//! branch that needs C/V/N simply leaves the canonical flag a free
//! input downstream → the solver returns `BothPossible` (sound, never
//! a fabricated verdict). Branches that read only Z (`eq`/`ne`,
//! `cbz`/`cbnz`, `test;jz`, …) get a precise decompiler-grade slice.
//! Any opcode outside the modelled subset returns [`PcodeError`] so
//! the caller falls back to the ESIL / per-mnemonic lifter.

use std::collections::BTreeMap;

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::stmt::IrStmt;

use crate::parse::{ParseError, PcodeOp, Varnode, parse_pcode};

/// Reasons the P-code lifter declines (caller falls back to ESIL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PcodeError {
    /// The `pdgsd` text could not be structurally parsed.
    Parse(String),
    /// An opcode outside the modelled sound subset.
    UnsupportedOpcode(String),
    /// A varnode shape the lifter does not model (e.g. nested `ram`).
    BadVarnode(String),
    /// An op had the wrong operand arity for its opcode.
    Arity(String),
}

impl From<ParseError> for PcodeError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e.0)
    }
}

/// Lifter output — mirrors [`r2smt_esil::EsilLift`] so callers can
/// splice it into their statement list directly.
#[derive(Debug, Clone)]
pub struct PcodeLift {
    /// Statements produced, in execution order.
    pub statements: Vec<IrStmt>,
}

/// Lift a `pdgsd` dump under `arch` into IR statements.
///
/// # Errors
///
/// Returns [`PcodeError`] on a parse failure or any construct outside
/// the sound subset; the caller treats that as "use another IR".
pub fn lift_pcode(text: &str, arch: Arch) -> Result<PcodeLift, PcodeError> {
    let insns = parse_pcode(text)?;
    let mut m = Machine::new(arch);
    for insn in &insns {
        for op in &insn.ops {
            m.step(op)?;
        }
    }
    Ok(PcodeLift {
        statements: m.statements,
    })
}

struct Machine {
    arch: Arch,
    /// Defined width (in bits) per varnode key, so a later read uses
    /// the width its defining op assigned.
    widths: BTreeMap<String, u8>,
    statements: Vec<IrStmt>,
}

impl Machine {
    fn new(arch: Arch) -> Self {
        Self {
            arch,
            widths: BTreeMap::new(),
            statements: Vec::new(),
        }
    }

    fn step(&mut self, op: &PcodeOp) -> Result<(), PcodeError> {
        match op.opcode.as_str() {
            "STORE" => self.lift_store(op),
            // Control flow carries no data-flow definition. The branch
            // predicate is computed by the preceding flag ops and read
            // by the existing slicer; emit nothing.
            "BRANCH" | "CBRANCH" | "BRANCHIND" | "CALL" | "CALLIND" | "CALLOTHER" | "RETURN" => {
                Ok(())
            }
            _ => self.lift_defining(op),
        }
    }

    fn lift_store(&mut self, op: &PcodeOp) -> Result<(), PcodeError> {
        let [addr, value] = op.inputs.as_slice() else {
            return Err(PcodeError::Arity(format!("STORE inputs: {:?}", op.inputs)));
        };
        let addr_expr = self.read_mem_addr(addr)?;
        let val_expr = self.read(value)?;
        let bits = self.var_width(value);
        self.statements.push(IrStmt::StoreMem {
            address: addr_expr,
            value: val_expr,
            bits,
        });
        Ok(())
    }

    fn lift_defining(&mut self, op: &PcodeOp) -> Result<(), PcodeError> {
        let Some(out) = &op.out else {
            return Err(PcodeError::Arity(format!(
                "{} expects an output",
                op.opcode
            )));
        };

        if op.opcode == "LOAD" {
            let [src] = op.inputs.as_slice() else {
                return Err(PcodeError::Arity(format!("LOAD inputs: {:?}", op.inputs)));
            };
            let addr = self.read_mem_addr(src)?;
            let (dst, bits) = self.define(out, varnode_bits(out, self.arch));
            self.statements.push(IrStmt::LoadMem {
                dst,
                address: addr,
                bits,
            });
            return Ok(());
        }

        let out_bits = varnode_bits(out, self.arch);
        let expr = self.eval(&op.opcode, &op.inputs, out_bits)?;
        let (dst, _) = self.define(out, expr_bits(&expr, out_bits));
        self.statements.push(IrStmt::Assign { dst, src: expr });
        Ok(())
    }

    /// Build the value expression for a defining opcode.
    fn eval(&mut self, opcode: &str, inputs: &[Varnode], out_bits: u8) -> Result<Expr, PcodeError> {
        let bin = |m: &mut Self| -> Result<(Expr, Expr), PcodeError> {
            let [a, b] = inputs else {
                return Err(PcodeError::Arity(format!("{opcode} needs 2 inputs")));
            };
            Ok((m.read(a)?, m.read(b)?))
        };
        let un = |m: &mut Self| -> Result<Expr, PcodeError> {
            let [a] = inputs else {
                return Err(PcodeError::Arity(format!("{opcode} needs 1 input")));
            };
            m.read(a)
        };

        match opcode {
            "COPY" => un(self),
            "INT_ADD" => bin(self).map(|(a, b)| Expr::add(a, b)),
            "INT_SUB" => bin(self).map(|(a, b)| Expr::sub(a, b)),
            "INT_MULT" => bin(self).map(|(a, b)| Expr::mul(a, b)),
            "INT_AND" => bin(self).map(|(a, b)| Expr::bv_and(a, b)),
            "INT_OR" => bin(self).map(|(a, b)| Expr::bv_or(a, b)),
            "INT_XOR" => bin(self).map(|(a, b)| Expr::bv_xor(a, b)),
            "INT_LEFT" => bin(self).map(|(a, b)| Expr::shl(a, b)),
            "INT_RIGHT" => bin(self).map(|(a, b)| Expr::lshr(a, b)),
            "INT_SRIGHT" => bin(self).map(|(a, b)| Expr::ashr(a, b)),
            "INT_NEGATE" => un(self).map(|a| Expr::bv_xor(a, Expr::konst(u64::MAX, out_bits))),
            "INT_2COMP" => un(self).map(|a| Expr::sub(Expr::konst(0, out_bits), a)),
            "INT_ZEXT" => un(self).map(|a| Expr::zero_ext(a, out_bits)),
            "INT_SEXT" => un(self).map(|a| Expr::sign_ext(a, out_bits)),
            "INT_EQUAL" => bin(self).map(|(a, b)| Expr::eq(a, b)),
            "INT_NOTEQUAL" => bin(self).map(|(a, b)| Expr::ne(a, b)),
            "INT_LESS" => bin(self).map(|(a, b)| Expr::ult(a, b)),
            "INT_LESSEQUAL" => bin(self).map(|(a, b)| Expr::ule(a, b)),
            "INT_SLESS" => bin(self).map(|(a, b)| Expr::slt(a, b)),
            "INT_SLESSEQUAL" => bin(self).map(|(a, b)| Expr::sle(a, b)),
            "BOOL_NEGATE" => un(self).map(|a| Expr::Ite {
                cond: Box::new(Expr::eq(a, Expr::konst(0, 1))),
                then_expr: Box::new(Expr::konst(1, 1)),
                else_expr: Box::new(Expr::konst(0, 1)),
            }),
            "BOOL_AND" => bin(self).map(|(a, b)| Expr::bv_and(a, b)),
            "BOOL_OR" => bin(self).map(|(a, b)| Expr::bv_or(a, b)),
            "BOOL_XOR" => bin(self).map(|(a, b)| Expr::bv_xor(a, b)),
            "SUBPIECE" => {
                let [a, off] = inputs else {
                    return Err(PcodeError::Arity("SUBPIECE needs 2 inputs".into()));
                };
                let Varnode::Const {
                    value: byte_off, ..
                } = off
                else {
                    return Err(PcodeError::BadVarnode(format!(
                        "SUBPIECE offset must be const: {off:?}"
                    )));
                };
                let src = self.read(a)?;
                let lo = u8::try_from(byte_off.saturating_mul(8))
                    .map_err(|_| PcodeError::BadVarnode("SUBPIECE offset too large".into()))?;
                let hi = lo.saturating_add(out_bits.saturating_sub(1));
                Ok(Expr::extract(src, hi, lo))
            }
            other => Err(PcodeError::UnsupportedOpcode(other.to_string())),
        }
    }

    /// Resolve a varnode read to an [`Expr`].
    fn read(&mut self, vn: &Varnode) -> Result<Expr, PcodeError> {
        match vn {
            Varnode::Const { value, size } => {
                let bits = size.map_or(self.ptr_bits(), |s| s.saturating_mul(8));
                Ok(Expr::konst(*value, bits))
            }
            Varnode::Register(_) | Varnode::Unique { .. } => {
                let name = Self::var_name(vn);
                let bits = self.var_width(vn);
                Ok(Expr::Var(Var::new(name, bits)))
            }
            Varnode::Ram(_) | Varnode::CodeAddr(_) => Err(PcodeError::BadVarnode(format!(
                "value position cannot be {vn:?}"
            ))),
        }
    }

    fn read_mem_addr(&mut self, vn: &Varnode) -> Result<Expr, PcodeError> {
        match vn {
            Varnode::Ram(inner) => self.read(inner),
            Varnode::Register(_) | Varnode::Unique { .. } => self.read(vn),
            _ => Err(PcodeError::BadVarnode(format!("bad mem address: {vn:?}"))),
        }
    }

    /// Register the defined width for `out` and return its IR `Var`.
    fn define(&mut self, out: &Varnode, bits: u8) -> (Var, u8) {
        let name = Self::var_name(out);
        self.widths.insert(name.clone(), bits);
        (Var::new(name, bits), bits)
    }

    fn var_width(&self, vn: &Varnode) -> u8 {
        let name = Self::var_name(vn);
        if let Some(w) = self.widths.get(&name) {
            return *w;
        }
        varnode_bits(vn, self.arch)
    }

    /// Canonical IR variable name for a varnode. Only the Z flag is
    /// mapped onto the canonical `ZF`; N/C/V get distinct `pc_*` names
    /// so they never collide with the per-mnemonic flag model.
    fn var_name(vn: &Varnode) -> String {
        match vn {
            Varnode::Unique { offset, size } => format!("u_{offset:x}_{size}"),
            Varnode::Register(r) => map_register(r),
            other => format!("?{other:?}"),
        }
    }

    fn ptr_bits(&self) -> u8 {
        match self.arch {
            Arch::X86 | Arch::Arm => 32,
            _ => 64,
        }
    }
}

/// Map a P-code register/flag name to a canonical IR variable name.
fn map_register(r: &str) -> String {
    match r {
        // Z flag is polarity-free: P-code `ZR` ≡ canonical `ZF`.
        "ZR" | "tmpZR" => "ZF".to_string(),
        // N/C/V kept distinct from canonical SF/CF/OF (ARM polarity
        // differs from the per-mnemonic model — see module docs).
        "NG" | "tmpNG" => "pc_ng".to_string(),
        "CY" | "tmpCY" => "pc_cy".to_string(),
        "OV" | "tmpOV" => "pc_ov".to_string(),
        // `wN` (32-bit) and `xN` (64-bit) are kept as distinct vars.
        // P-code is explicit about the relationship — every w↔x
        // transition is materialised by its own `INT_ZEXT` /
        // `SUBPIECE` op — so faithfully mirroring the named varnodes
        // is sound; an unmaterialised alias merely leaves a free
        // input (→ sound `BothPossible`), never a wrong verdict.
        _ => r.to_string(),
    }
}

/// Natural width (bits) of a varnode before any defining op overrides.
fn varnode_bits(vn: &Varnode, arch: Arch) -> u8 {
    match vn {
        Varnode::Unique { size, .. } => size.saturating_mul(8).max(1),
        Varnode::Const { size, .. } => size.map_or(64, |s| s.saturating_mul(8)).max(1),
        Varnode::Register(r) => register_bits(r, arch),
        Varnode::Ram(_) | Varnode::CodeAddr(_) => 64,
    }
}

fn register_bits(r: &str, arch: Arch) -> u8 {
    match r {
        "ZR" | "tmpZR" | "NG" | "tmpNG" | "CY" | "tmpCY" | "OV" | "tmpOV" => 1,
        "sp" | "lr" | "fp" | "pc" => 64,
        _ if r.starts_with('x') && r[1..].chars().all(|c| c.is_ascii_digit()) => 64,
        _ if r.starts_with('w') && r[1..].chars().all(|c| c.is_ascii_digit()) => 32,
        _ if r.starts_with('r') && r[1..].chars().all(|c| c.is_ascii_digit()) => {
            if matches!(arch, Arch::X86) { 32 } else { 64 }
        }
        _ => 64,
    }
}

fn expr_bits(expr: &Expr, fallback: u8) -> u8 {
    match expr {
        Expr::Const { bits, .. } => *bits,
        Expr::Var(v) => v.bits,
        Expr::ZeroExtend { to_bits, .. } | Expr::SignExtend { to_bits, .. } => *to_bits,
        Expr::Eq(..)
        | Expr::Ne(..)
        | Expr::Ult(..)
        | Expr::Ule(..)
        | Expr::Slt(..)
        | Expr::Sle(..) => 1,
        Expr::Ite { then_expr, .. } => expr_bits(then_expr, fallback),
        Expr::Extract { hi, lo, .. } => hi.saturating_sub(*lo).saturating_add(1),
        _ => fallback,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn test_lift_int_arith_chain_to_assigns() {
        let txt = "\
0x100: mul w8, w8, w9
    (unique,0x2ae80,4) = INT_MULT w8, w9
    x8 = INT_ZEXT (unique,0x2ae80,4)";
        let lift = lift_pcode(txt, Arch::Aarch64).unwrap();
        assert_eq!(lift.statements.len(), 2);
        match &lift.statements[0] {
            IrStmt::Assign { dst, src } => {
                assert_eq!(dst.name, "u_2ae80_4");
                assert_eq!(dst.bits, 32);
                assert_eq!(
                    *src,
                    Expr::mul(Expr::Var(Var::new("w8", 32)), Expr::Var(Var::new("w9", 32)))
                );
            }
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    #[test]
    fn test_lift_z_flag_maps_to_canonical_zf() {
        // `tmpZR = INT_EQUAL (sub), 0 ; ZR = COPY tmpZR` — both must
        // resolve to canonical `ZF` so the branch composer reads it.
        let txt = "\
0x100: subs w8, w8, #2
    (unique,0x10,4) = INT_SUB w8, 0x2
    tmpZR = INT_EQUAL (unique,0x10,4), 0x0
    ZR = COPY tmpZR";
        let lift = lift_pcode(txt, Arch::Aarch64).unwrap();
        let IrStmt::Assign { dst, .. } = &lift.statements[2] else {
            panic!("expected Assign");
        };
        assert_eq!(dst.name, "ZF");
        assert_eq!(dst.bits, 1);
    }

    #[test]
    fn test_lift_ncv_flags_stay_non_canonical() {
        // `CY` must NOT become canonical `CF` (ARM polarity differs).
        let txt = "\
0x100: subs w8, w8, #2
    tmpCY = INT_LESSEQUAL 0x2, w8
    CY = COPY tmpCY";
        let lift = lift_pcode(txt, Arch::Aarch64).unwrap();
        let IrStmt::Assign { dst, .. } = &lift.statements[1] else {
            panic!("expected Assign");
        };
        assert_eq!(dst.name, "pc_cy");
        assert_ne!(dst.name, "CF");
    }

    #[test]
    fn test_lift_load_store_emit_mem_stmts() {
        let txt = "\
0x100: ldr w8, [sp, #8]
    (unique,0x60,8) = INT_ADD sp, 0x8
    (unique,0x247,4) = LOAD ram[(unique,0x60,8)]
0x104: str w8, [sp, #8]
    STORE ram[(unique,0x60,8)] = w8";
        let lift = lift_pcode(txt, Arch::Aarch64).unwrap();
        assert!(matches!(
            lift.statements[1],
            IrStmt::LoadMem { bits: 32, .. }
        ));
        assert!(matches!(
            lift.statements[2],
            IrStmt::StoreMem { bits: 32, .. }
        ));
    }

    #[test]
    fn test_unsupported_opcode_errors_for_fallback() {
        let txt = "\
0x100: fmul d0, d0, d1
    d0 = FLOAT_MULT d0, d1";
        let err = lift_pcode(txt, Arch::Aarch64).unwrap_err();
        assert_eq!(err, PcodeError::UnsupportedOpcode("FLOAT_MULT".into()));
    }

    #[test]
    fn test_parse_error_propagates_as_pcode_error() {
        let err = lift_pcode("    orphan = INT_ADD a, b", Arch::Aarch64).unwrap_err();
        assert!(matches!(err, PcodeError::Parse(_)));
    }
}
