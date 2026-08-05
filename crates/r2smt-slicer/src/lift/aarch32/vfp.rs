//! The `AArch32` VFP scalar family: its vocabulary and its lowering.
//!
//! Split out of the parent when the addressing and conversion work took
//! it to 1 998 lines, two short of the 2 000-line hard limit. The cut
//! follows `memory.rs`'s precedent one step further: there only the
//! lowerings moved and the model stayed behind, but the VFP *type*
//! vocabulary — `VfpOp`, `VfpConvert`, the two parsers — has exactly one
//! inbound edge from the dispatch and travels with its lowering, so
//! keeping them apart would put a module wall through a private pair.
//!
//! What stays in the parent is what the parent is: the mnemonic
//! dispatch, the addressing model, and the NEON packed tables. In
//! particular `neon_element_type` stays there and is read from here,
//! which is the reason this file is a *child* of `aarch32` and not a
//! sibling — a descendant sees its ancestor's private items, so the
//! split needed no visibility widening at all beyond making
//! `lift_aarch32_vfp` `pub(super)` for the dispatch to reach.

use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::Instruction;
use r2smt_ir::stmt::IrStmt;

use super::super::aarch64;
use super::super::aarch64::neon::lower::convert_lane;
use super::super::aarch64::neon::width::ConvertKind;
use super::super::simd;
use super::super::{
    FpArithOp, LiftCtx, fp_lane_result, fp_propagating_max_min, fp_sort_bits_checked,
    parse_immediate,
};
use super::{ElementKind, neon_element_type};

/// What a VFP scalar mnemonic does, and at what precision.
#[derive(Clone, Copy)]
pub(crate) enum VfpOp {
    /// Binary arithmetic on two source registers.
    Arith(FpArithOp),
    /// `vcmp` / `vcmpe` — compare into FPSCR.
    Compare,
    /// `vmov` — move a bit pattern without interpreting it.
    Move,
    /// `vneg` / `vabs` — sign-bit manipulation.
    Sign { negate: bool },
    /// `vsqrt`.
    Sqrt,
    /// `vrint<mode>` — round to an integral value, keeping the float
    /// sort.
    Round {
        /// The mode the result rounds with.
        rounding: RoundingMode,
        /// Whether that mode is FPSCR's default rather than one the
        /// encoding fixes. Only `vrintr` / `vrintx` read the control
        /// word; the directed forms each name their own, exactly as
        /// `vcvta` / `vcvtn` / `vcvtp` / `vcvtm` do beside `vcvtr`.
        reads_control: bool,
    },
    /// `vcvt` and its directed-rounding cousins — the one VFP family
    /// naming **two** element types (`vcvt.f32.s32`). The width
    /// [`vfp_scalar`] returns beside this is the *destination*
    /// element's; the source's is carried here.
    Convert(VfpConvert),
}

/// The source element and rounding of an `AArch32` VFP conversion.
///
/// Split from [`VfpOp`] because a conversion is the one VFP shape whose
/// two operands need not share a width, and because the rounding a
/// float-to-integer form uses is spelled by the mnemonic rather than
/// fixed.
#[derive(Clone, Copy)]
pub(crate) struct VfpConvert {
    /// Signedness class and width of the source element.
    pub(crate) source: (ElementKind, u16),
    /// Signedness class of the destination element. Its width is the
    /// one [`vfp_scalar`] returns beside this.
    pub(crate) dest_kind: ElementKind,
    /// The mode a float-to-integer conversion rounds with. The other
    /// directions round to the FPSCR default and ignore this.
    pub(crate) to_int_rounding: RoundingMode,
    /// Whether that mode is the FPSCR default rather than one the
    /// encoding fixes, so lifting it pins a control field.
    ///
    /// `vcvt.s32.f32` carries round-toward-zero in the opcode and
    /// `vcvta` / `vcvtn` / `vcvtp` / `vcvtm` each name their own, so
    /// only `vcvtr` — and every conversion *into* a float — depends on
    /// FPSCR.
    pub(crate) reads_control: bool,
}

/// The fraction width of a fixed-point `vcvt`, or zero for the
/// plain register form.
///
/// The immediate is bounded by the width of the *fixed-point* side,
/// which is whichever end is not the float (ARM ARM F6.1.35). An
/// out-of-range amount is not an instruction, so it declines rather
/// than being clamped into one.
fn vfp_convert_fraction_bits(
    insn: &Instruction,
    kind: ConvertKind,
    dest_bits: u16,
    source_bits: u16,
) -> Option<u16> {
    let Some(operand) = insn.operands.get(2) else {
        return Some(0);
    };
    // `fcvtl` / `fcvtn`'s AArch32 spelling changes only the float
    // format, so neither end is fixed point and a third operand
    // means this is a shape the parser mis-read.
    if !kind.scales() {
        return None;
    }
    let fixed_bits = match kind {
        ConvertKind::IntToFloat { .. } => source_bits,
        _ => dest_bits,
    };
    let amount = u16::try_from(parse_immediate(&operand.raw)?).ok()?;
    (amount > 0 && amount <= fixed_bits).then_some(amount)
}

/// Parse the two dotted element types of a conversion mnemonic —
/// `vcvt.f32.s32` is *to* `f32` *from* `s32`.
///
/// Separate from [`vfp_scalar`]'s single `split_once('.')` because that
/// hands the whole `"f32.s32"` tail to [`neon_element_type`], which
/// matches nothing: the reason the entire family declined before this
/// existed rather than lifting wrongly.
pub(in crate::lift::aarch32) fn vfp_convert(base: &str, ty: &str) -> Option<(VfpConvert, u16)> {
    // `vcvtr` differs from `vcvt` only in taking its rounding from
    // FPSCR instead of the opcode; `vcvtb` / `vcvtt` address the halves
    // of a half-precision pair and are a different operand shape, so
    // they are deliberately absent.
    let (to_int_rounding, reads_control) = match base {
        "vcvt" => (RoundingMode::TowardZero, false),
        "vcvtr" => (RoundingMode::NearestTiesEven, true),
        "vcvta" => (RoundingMode::NearestTiesAway, false),
        "vcvtn" => (RoundingMode::NearestTiesEven, false),
        "vcvtp" => (RoundingMode::TowardPositive, false),
        "vcvtm" => (RoundingMode::TowardNegative, false),
        _ => return None,
    };
    let (dest_ty, source_ty) = ty.split_once('.')?;
    let (dest_kind, dest_bits) = neon_element_type(dest_ty)?;
    let source = neon_element_type(source_ty)?;
    // One side must be a float: an integer-to-integer `vcvt` does not
    // exist, and `Untyped` names no signedness a conversion could use.
    let float_ends =
        u8::from(dest_kind == ElementKind::Float) + u8::from(source.0 == ElementKind::Float);
    if float_ends == 0 || dest_kind == ElementKind::Untyped || source.0 == ElementKind::Untyped {
        return None;
    }
    Some((
        VfpConvert {
            source,
            dest_kind,
            to_int_rounding,
            reads_control,
        },
        dest_bits,
    ))
}

/// Parse `v<op>.f<width>` — the `AArch32` spelling that carries the
/// precision in the mnemonic. Integer-typed vector forms (`vadd.i32`)
/// are not scalar floating point and decline here.
pub(crate) fn vfp_scalar(mnemonic: &str) -> Option<(VfpOp, u16)> {
    let (base, ty) = mnemonic.split_once('.')?;
    // Tried first: a conversion's tail is itself dotted, so the width
    // match below sees `"f32.s32"` and declines.
    if let Some((convert, dest_bits)) = vfp_convert(base, ty) {
        return Some((VfpOp::Convert(convert), dest_bits));
    }
    let lane = match ty {
        "f32" => 32,
        "f64" => 64,
        "f16" => 16,
        _ => return None,
    };
    let op = match base {
        "vadd" => VfpOp::Arith(FpArithOp::Add),
        "vsub" => VfpOp::Arith(FpArithOp::Sub),
        "vmul" => VfpOp::Arith(FpArithOp::Mul),
        "vdiv" => VfpOp::Arith(FpArithOp::Div),
        "vmax" => VfpOp::Arith(FpArithOp::Max),
        "vmin" => VfpOp::Arith(FpArithOp::Min),
        "vcmp" | "vcmpe" => VfpOp::Compare,
        "vmov" => VfpOp::Move,
        "vneg" => VfpOp::Sign { negate: true },
        "vabs" => VfpOp::Sign { negate: false },
        "vsqrt" => VfpOp::Sqrt,
        "vrinta" => VfpOp::Round {
            rounding: RoundingMode::NearestTiesAway,
            reads_control: false,
        },
        "vrintn" => VfpOp::Round {
            rounding: RoundingMode::NearestTiesEven,
            reads_control: false,
        },
        "vrintp" => VfpOp::Round {
            rounding: RoundingMode::TowardPositive,
            reads_control: false,
        },
        "vrintm" => VfpOp::Round {
            rounding: RoundingMode::TowardNegative,
            reads_control: false,
        },
        "vrintz" => VfpOp::Round {
            rounding: RoundingMode::TowardZero,
            reads_control: false,
        },
        // `vrintr` rounds by FPSCR, and `vrintx` computes the same value
        // while additionally being able to signal the inexact exception,
        // which the value model does not carry.
        "vrintr" | "vrintx" => VfpOp::Round {
            rounding: FPSCR_DEFAULT_ROUNDING,
            reads_control: true,
        },
        _ => return None,
    };
    Some((op, lane))
}

/// The rounding mode FPSCR holds out of reset (`RMode == 0b00`), which
/// the forms that read the control word are lifted against.
const FPSCR_DEFAULT_ROUNDING: RoundingMode = RoundingMode::NearestTiesEven;

impl LiftCtx {
    /// The VFP scalar family. Operand shapes mirror `AArch64`, so the
    /// lane helpers are shared; only the flag write differs, because
    /// `AArch32` compares land in FPSCR and are copied to the
    /// condition flags by a subsequent `vmrs`.
    pub(super) fn lift_aarch32_vfp(&mut self, insn: &Instruction, op: VfpOp, lane: u16) {
        let Some(dst) = insn.operands.first().cloned() else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        let result = match op {
            VfpOp::Arith(arith) => self.vfp_arith_value(insn, arith, lane),
            VfpOp::Sqrt => self.vfp_sqrt_value(insn, lane),
            VfpOp::Round { rounding, .. } => self.vfp_round_value(insn, lane, rounding),
            VfpOp::Sign { negate } => self.vfp_sign_value(insn, lane, negate),
            VfpOp::Move => insn
                .operands
                .get(1)
                .cloned()
                .and_then(|src| self.read_simd_lane_bits(&src, lane, 0)),
            VfpOp::Compare => {
                self.lift_aarch32_vcmp(insn, lane);
                return;
            }
            VfpOp::Convert(convert) => self.vfp_convert_value(insn, convert, lane),
        };
        let Some(value) = result else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        // `AArch32` VFP writes the addressed slice and preserves the
        // rest of the register file — the opposite of `AArch64`, and
        // the reason `write_simd_lane` has to honour the view's offset.
        if !self.write_simd_lane(&dst, value, lane, 0) {
            self.push_aarch32_vfp_unsupported(insn);
        }
    }

    /// One `vcvt`, as the conversion the destination and source element
    /// types name between them.
    ///
    /// `dest_bits` is the destination element's width, which is also the
    /// width the caller writes — so a `vcvt.s16.f32` merges its 16-bit
    /// result into the low half of an `s` register and leaves the top
    /// half standing, as an `AArch32` VFP write does.
    fn vfp_convert_value(
        &mut self,
        insn: &Instruction,
        convert: VfpConvert,
        dest_bits: u16,
    ) -> Option<Expr> {
        let (source_kind, source_bits) = convert.source;
        let kind = match (convert.dest_kind, source_kind) {
            (ElementKind::Float, ElementKind::Float) => ConvertKind::FloatToFloat {
                widening: dest_bits > source_bits,
            },
            (ElementKind::Float, signedness) => ConvertKind::IntToFloat {
                signed: signedness == ElementKind::Signed,
            },
            (signedness, ElementKind::Float) => ConvertKind::FloatToInt {
                signed: signedness == ElementKind::Signed,
            },
            // `vfp_convert` requires a float on one side, so this arm is
            // unreachable through the parser.
            _ => return None,
        };
        let fbits = vfp_convert_fraction_bits(insn, kind, dest_bits, source_bits)?;
        let src = insn.operands.get(1)?.clone();
        // A conversion is scalar only if each operand's register class
        // holds exactly one element. `vfp_convert` reads the mnemonic
        // alone, and the packed forms spell the same one — `vcvt.s32.f32
        // q0, q1` converts four lanes where this would convert lane
        // zero and leave the rest of the register standing. That is a
        // wrong value rather than a decline, so the check is here and
        // not in the parser, which cannot see the operands.
        let dst = insn.operands.first()?.clone();
        for (operand, bits) in [(&dst, dest_bits), (&src, source_bits)] {
            // The fixed-point forms hold a 16-bit element in an `s`
            // register, so the rule is the register class the width
            // needs, not equality with the element.
            let expected = if bits > 32 { bits } else { 32 };
            if self.simd_view_bits(operand)? != expected {
                return None;
            }
        }
        let element = self.read_simd_lane_bits(&src, source_bits, 0)?;
        convert_lane(
            kind,
            element,
            source_bits,
            dest_bits,
            fbits,
            convert.to_int_rounding,
        )
    }

    fn vfp_arith_value(&mut self, insn: &Instruction, arith: FpArithOp, lane: u16) -> Option<Expr> {
        let lhs = insn.operands.get(1)?.clone();
        let rhs = insn.operands.get(2)?.clone();
        let a = self.read_simd_lane_bits(&lhs, lane, 0)?;
        let b = self.read_simd_lane_bits(&rhs, lane, 0)?;
        // `vmax` / `vmin` are `FPMax` / `FPMin`, which propagate NaN and
        // combine the signs of a zero tie. [`fp_lane_result`] is Intel's
        // `MAXPS` and does neither.
        match arith {
            FpArithOp::Max | FpArithOp::Min => {
                fp_propagating_max_min(a, b, lane, matches!(arith, FpArithOp::Max), false)
            }
            FpArithOp::Add | FpArithOp::Sub | FpArithOp::Mul | FpArithOp::Div => {
                fp_lane_result(arith, a, b, lane)
            }
        }
    }

    fn vfp_sqrt_value(&mut self, insn: &Instruction, lane: u16) -> Option<Expr> {
        let src = insn.operands.get(1)?.clone();
        let value = self.read_simd_lane_fp(&src, lane, 0)?;
        Some(Expr::fp_to_ieee_bv(Expr::fsqrt(
            value,
            r2smt_ir::expr::RoundingMode::NearestTiesEven,
        )))
    }

    /// `vrint<mode>.f32 Sd, Sm` / `.f64 Dd, Dm` — round to an integral
    /// value without leaving the float sort.
    ///
    /// The register classes are checked rather than assumed, unlike
    /// every other VFP shape here: `vrint` also has a NEON form
    /// (`vrinta.f32 q0, q1`) spelling the identical mnemonic, and
    /// `neon_packed_op` does not claim the family, so nothing above
    /// this arm separates the two. Lowering the packed form here would
    /// round lane zero and leave the rest of the register standing —
    /// a wrong value rather than a decline.
    fn vfp_round_value(
        &mut self,
        insn: &Instruction,
        lane: u16,
        rounding: RoundingMode,
    ) -> Option<Expr> {
        let src = insn.operands.get(1)?.clone();
        let dst = insn.operands.first()?.clone();
        // A half-precision element lives in the low half of an `s`
        // register, so the rule is the register class the width needs
        // rather than equality with the element — as in `vcvt`.
        let expected = if lane > 32 { lane } else { 32 };
        for operand in [&dst, &src] {
            if self.simd_view_bits(operand)? != expected {
                return None;
            }
        }
        let value = self.read_simd_lane_fp(&src, lane, 0)?;
        Some(Expr::fp_to_ieee_bv(Expr::fround_to_integral(
            value, rounding,
        )))
    }

    fn vfp_sign_value(&mut self, insn: &Instruction, lane: u16, negate: bool) -> Option<Expr> {
        let src = insn.operands.get(1)?.clone();
        let bits = self.read_simd_lane_bits(&src, lane, 0)?;
        let sign = aarch64::sign_bit_mask(lane)?;
        Some(if negate {
            Expr::bv_xor(bits, sign)
        } else {
            Expr::bv_and(bits, Expr::bv_xor(sign, simd::all_ones(lane)))
        })
    }

    /// `vcmp` / `vcmpe` — same flag mapping as the `AArch64` compare:
    /// ordered equality in ZF, ordered less-than in SF, the unordered
    /// predicate in OF so `lt` covers less-than and unordered exactly
    /// as the architecture defines after a floating-point compare.
    fn lift_aarch32_vcmp(&mut self, insn: &Instruction, lane: u16) {
        let (Some(lhs), Some(rhs)) = (
            insn.operands.first().cloned(),
            insn.operands.get(1).cloned(),
        ) else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        let Some(a) = self.read_simd_lane_fp(&lhs, lane, 0) else {
            self.push_aarch32_vfp_unsupported(insn);
            return;
        };
        let b = if self.is_simd_register(&rhs) {
            let Some(value) = self.read_simd_lane_fp(&rhs, lane, 0) else {
                self.push_aarch32_vfp_unsupported(insn);
                return;
            };
            value
        } else {
            let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
                self.push_aarch32_vfp_unsupported(insn);
                return;
            };
            Expr::bv_to_fp(Expr::konst(0, lane), ebits, sbits)
        };
        let unordered = Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone()));
        self.set_flag("ZF", Expr::feq(a.clone(), b.clone()));
        self.set_flag("SF", Expr::flt(a.clone(), b.clone()));
        self.set_flag("CF", Expr::flt(a, b));
        self.set_flag("OF", unordered.clone());
        self.set_flag("PF", unordered);
    }

    fn push_aarch32_vfp_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable VFP operand at {addr}", addr = insn.address),
        });
    }
}
