//! `AArch64` scalar floating-point handlers.
//!
//! Split from the parent when it crossed the promote line. The cut takes
//! the mnemonic dispatch along with the handler bodies, which is what
//! keeps it cheap: only [`LiftCtx::lift_aarch64_fp_by_mnemonic`] needs to
//! be visible to the parent, so every handler stays private here.
//!
//! Two items the region reads stay behind in the parent on purpose.
//! `FPCR_DEFAULT_ROUNDING` has no consumer here at all — only the packed
//! resolver in `neon/arith.rs` reads it — and [`super::sign_bit_mask`] is
//! generic bit manipulation that `aarch32/vfp.rs` reaches by the parent's
//! path. A child reads its parent's private items, so neither move costs
//! anything here.

use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::{Instruction, Operand};
use r2smt_ir::stmt::IrStmt;

use crate::lift::simd::{all_ones, fp_multiply_extended_lane};
use crate::lift::{
    FpArithOp, LiftCtx, fp_lane_result, fp_propagating_max_min, fp_sort_bits_checked,
};

use super::neon;
use super::sign_bit_mask;

/// Unary scalar floating-point operation.
#[derive(Clone, Copy)]
enum FpUnaryOp {
    /// `fabs` — clear the sign bit.
    Abs,
    /// `fneg` — flip the sign bit.
    Neg,
}

impl LiftCtx {
    /// The scalar floating-point arm of the `AArch64` dispatch. Returns
    /// whether the mnemonic was claimed, so the caller can fall through
    /// to its `Unsupported` arm.
    ///
    /// The lane width comes from the register letter (`s0` → 32,
    /// `d0` → 64, `h0` → 16), which `simd_view_bits` already reports, so
    /// one handler covers every precision.
    pub(super) fn lift_aarch64_fp_by_mnemonic(&mut self, insn: &Instruction, mnem: &str) -> bool {
        match mnem {
            "fadd" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Add),
            "fsub" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Sub),
            "fmul" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Mul),
            // `fmulx` differs from `fmul` on one input class, so it goes
            // through the packed core's lane helper rather than the
            // scalar arithmetic one.
            "fmulx" => self.lift_aarch64_fmulx(insn),
            "fdiv" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Div),
            "fmax" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Max),
            "fmin" => self.lift_aarch64_fp_arith3(insn, FpArithOp::Min),
            "fsqrt" => self.lift_aarch64_fp_sqrt(insn),
            "fabs" => self.lift_aarch64_fp_unary(insn, FpUnaryOp::Abs),
            "fneg" => self.lift_aarch64_fp_unary(insn, FpUnaryOp::Neg),
            "fmov" => self.lift_aarch64_fmov(insn),
            "fcmp" | "fcmpe" => self.lift_aarch64_fcmp(insn),
            "fcvt" => self.lift_aarch64_fcvt(insn),
            "scvtf" => self.lift_aarch64_int_to_fp(insn, true),
            "ucvtf" => self.lift_aarch64_int_to_fp(insn, false),
            "fcvtzs" => self.lift_aarch64_fp_to_int(insn, true),
            "fcvtzu" => self.lift_aarch64_fp_to_int(insn, false),
            // Round to an integral value *keeping the float sort*. Five
            // of the seven name their mode in the opcode, so they are
            // exact whatever FPCR holds. `frinti` reads FPCR, and
            // `frintx` computes the same value while additionally being
            // able to signal the inexact exception — a flag the value
            // model does not carry, so the two lower identically.
            //
            // The `FEAT_FRINTTS` members (`frint32z` / `frint32x` /
            // `frint64z` / `frint64x`) come through the same arm: the
            // table they all consult is the packed resolver's, so the
            // scalar and packed spellings cannot drift about either the
            // mode or the saturation width.
            mnemonic if let Some((rounding, saturate)) = neon::round_to_integral_kind(mnemonic) => {
                self.lift_aarch64_fp_round(insn, rounding, saturate);
            }
            "frecpx" => self.lift_aarch64_frecpx(insn),
            _ => return false,
        }
        true
    }

    /// `fadd`/`fsub`/`fmul`/`fdiv`/`fmax`/`fmin Rd, Rn, Rm` — scalar
    /// floating point at the precision the register letter names.
    ///
    /// A write through a scalar view zeroes the rest of the vector
    /// register per `AArch64` SIMD&FP semantics, unlike legacy SSE,
    /// which merges — hence `zero_upper`.
    fn lift_aarch64_fp_arith3(&mut self, insn: &Instruction, op: FpArithOp) {
        let ops = &insn.operands;
        let (Some(dst), Some(lhs), Some(rhs)) = (ops.first(), ops.get(1), ops.get(2)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(a) = self.read_simd_lane_bits(lhs, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(b) = self.read_simd_lane_bits(rhs, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // `fmax` / `fmin` are `FPMax` / `FPMin`, which propagate NaN
        // and combine the signs of a zero tie. [`fp_lane_result`] is
        // Intel's `MAXPS` and does neither.
        let result = match op {
            FpArithOp::Max | FpArithOp::Min => {
                fp_propagating_max_min(a, b, lane, matches!(op, FpArithOp::Max), false)
            }
            FpArithOp::Add | FpArithOp::Sub | FpArithOp::Mul | FpArithOp::Div => {
                fp_lane_result(op, a, b, lane)
            }
        };
        let Some(result) = result else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fsqrt Rd, Rn`.
    fn lift_aarch64_fp_sqrt(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let root = Expr::fp_to_ieee_bv(Expr::fsqrt(value, RoundingMode::NearestTiesEven));
        if !self.write_simd_dst(dst, root, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `frint<mode> Rd, Rn` — round to an integral value without
    /// leaving the float sort.
    ///
    /// Not the integer round trip `fcvtzs` + `scvtf`: that agrees only
    /// on the values an integer register can hold, and turns an
    /// infinity, a NaN or an out-of-range magnitude into some in-range
    /// number — a wrong value rather than a lost one.
    ///
    /// `saturate` is the `FEAT_FRINTTS` clamp (`frint32z` / `frint32x` /
    /// `frint64z` / `frint64x`), which keeps the float sort but bounds
    /// the value to what a signed integer of that width holds. It shares
    /// [`neon::lower::round_to_integral_lane`] with the packed spelling
    /// so the two cannot disagree about the bound.
    fn lift_aarch64_fp_round(
        &mut self,
        insn: &Instruction,
        rounding: RoundingMode,
        saturate: Option<u16>,
    ) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(rounded) = neon::lower::round_to_integral_lane(value, lane, rounding, saturate)
        else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, rounded, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fmulx Rd, Rn, Rm` — the scalar spelling of the extended
    /// multiply.
    ///
    /// Shares [`crate::lift::simd::fp_multiply_extended_lane`] with the
    /// packed and by-element forms, so the `inf * 0` answer and its sign
    /// cannot differ between spellings of one instruction.
    fn lift_aarch64_fmulx(&mut self, insn: &Instruction) {
        let (Some(dst), Some(a), Some(b)) = (
            insn.operands.first(),
            insn.operands.get(1),
            insn.operands.get(2),
        ) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let (Some(a_bits), Some(b_bits)) = (
            self.read_simd_lane_bits(a, lane, 0),
            self.read_simd_lane_bits(b, lane, 0),
        ) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(result) = fp_multiply_extended_lane(&a_bits, &b_bits, lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `frecpx Rd, Rn` — the reciprocal *exponent*.
    ///
    /// Shares a prefix with `frecpe` and almost nothing else, which is
    /// the hazard worth naming. The estimate's value is
    /// implementation-defined inside an error bound, so it lowers to a
    /// free value; `FPRecpX` is defined exactly — the exponent field
    /// complemented, the significand cleared, the sign kept — so lowering
    /// it as a free value would throw away a fact the architecture
    /// guarantees, and lowering `frecpe` this way would invent one.
    ///
    /// Built on the bit pattern rather than on a float sort, which keeps
    /// the binary16 form renderable by the text backends as well as by
    /// Z3. The NaN arm reproduces `FPProcessNaN` under the reset FPCR
    /// (`DN == 0`): the operand with its quiet bit forced on, which
    /// leaves a quiet NaN alone and quietens a signalling one.
    ///
    /// There is no vector form of this instruction, so no arrangement
    /// ever reaches here.
    fn lift_aarch64_frecpx(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(result) = reciprocal_exponent(bits, lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fabs`/`fneg Rd, Rn` — sign-bit manipulation, modelled on the
    /// bit pattern rather than as arithmetic so no rounding is implied.
    fn lift_aarch64_fp_unary(&mut self, insn: &Instruction, op: FpUnaryOp) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(sign) = sign_bit_mask(lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let result = match op {
            // `abs` clears the sign bit: AND with its complement.
            FpUnaryOp::Abs => Expr::bv_and(bits, Expr::bv_xor(sign, all_ones(lane))),
            FpUnaryOp::Neg => Expr::bv_xor(bits, sign),
        };
        if !self.write_simd_dst(dst, result, true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fmov` in its three shapes: vector-to-vector, general-to-vector
    /// and vector-to-general. All three move a bit pattern without
    /// interpreting it, so none of them rounds.
    fn lift_aarch64_fmov(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        match (self.is_simd_register(dst), self.is_simd_register(src)) {
            (true, true) => {
                let Some(lane) = self.simd_view_bits(dst) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                if !self.write_simd_dst(dst, bits, true) {
                    self.push_aarch64_fp_unsupported(insn);
                }
            }
            // `fmov s0, w0` — the general register supplies the bits.
            (true, false) => {
                let Some(lane) = self.simd_view_bits(dst) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let Some(value) = self.read_register(src) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let sized = Expr::extract(value, lane - 1, 0);
                if !self.write_simd_dst(dst, sized, true) {
                    self.push_aarch64_fp_unsupported(insn);
                }
            }
            // `fmov w0, s0` — the vector lane supplies the bits.
            (false, true) => {
                let Some(lane) = self.simd_view_bits(src) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                let Some(bits) = self.read_simd_lane_bits(src, lane, 0) else {
                    self.push_aarch64_fp_unsupported(insn);
                    return;
                };
                if !self.write_register_to(dst, bits) {
                    self.push_aarch64_fp_unsupported(insn);
                }
            }
            (false, false) => self.push_aarch64_fp_unsupported(insn),
        }
    }

    /// `fcmp`/`fcmpe Rn, Rm` — the floating-point compare, written into
    /// the same x86-polarity flags the rest of the `AArch64` lifter
    /// uses so the existing condition-code table keeps working.
    ///
    /// The mapping is chosen so that every `AArch64` condition after an
    /// `fcmp` comes out with its architectural meaning:
    ///
    /// - `ZF` is ordered equality, so `b.eq` is equality and unordered
    ///   does not satisfy it.
    /// - `OF` is the unordered predicate, matching `V`, so `b.vs` after
    ///   a compare means "unordered".
    /// - `SF` is ordered less-than, matching `N`, so `b.mi` means "less
    ///   than".
    /// - `b.lt` lowers to `SF != OF`, which is therefore true for
    ///   less-than *and* for unordered — exactly what `LT` means after
    ///   an `AArch64` floating-point compare, and the reason the sign
    ///   and overflow flags cannot simply be zeroed here.
    ///
    /// `fcmp` and `fcmpe` differ only in which NaN raises the invalid-
    /// operation exception, which the value model does not track.
    fn lift_aarch64_fcmp(&mut self, insn: &Instruction) {
        let (Some(lhs), Some(rhs)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(lhs) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(a) = self.read_simd_lane_fp(lhs, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // `fcmp Rn, #0.0` is the ISA's only immediate form.
        let b = if self.is_simd_register(rhs) {
            let Some(value) = self.read_simd_lane_fp(rhs, lane, 0) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            value
        } else if is_fp_zero_immediate(rhs) {
            let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            Expr::bv_to_fp(Expr::konst(0, lane), ebits, sbits)
        } else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let unordered = Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone()));
        self.set_flag("ZF", Expr::feq(a.clone(), b.clone()));
        self.set_flag("SF", Expr::flt(a.clone(), b.clone()));
        self.set_flag("CF", Expr::flt(a, b));
        self.set_flag("OF", unordered.clone());
        // `AArch64` has no parity flag; carrying the unordered
        // predicate keeps it informative and costs nothing, since no
        // `AArch64` condition reads it.
        self.set_flag("PF", unordered);
    }

    /// `fcvt Rd, Rn` — convert between floating-point precisions. The
    /// source and destination letters give both sorts.
    fn lift_aarch64_fcvt(&mut self, insn: &Instruction) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let (Some(dst_lane), Some(src_lane)) = (self.simd_view_bits(dst), self.simd_view_bits(src))
        else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, src_lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some((ebits, sbits)) = fp_sort_bits_checked(dst_lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let converted = Expr::fp_to_fp(value, RoundingMode::NearestTiesEven, ebits, sbits);
        if !self.write_simd_dst(dst, Expr::fp_to_ieee_bv(converted), true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `scvtf`/`ucvtf Rd, Rn` — integer register to floating point.
    fn lift_aarch64_int_to_fp(&mut self, insn: &Instruction, signed: bool) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(dst) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // The source may be a general register (`scvtf s0, w0`) or a
        // vector one (`scvtf s0, s1`, the integer held *in* the vector
        // file). They are two register files, not two widths: reading
        // the vector one through `read_register` would size `v1` at the
        // pointer width, which is the mirror image of the write-side
        // defect `lift_aarch64_fp_to_int` branches around.
        let source_bits = self.operand_width(src);
        let int = if self.is_simd_register(src) {
            self.read_simd_lane_bits(src, source_bits, 0)
        } else {
            self.read_register(src)
        };
        let Some(int) = int else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some((ebits, sbits)) = fp_sort_bits_checked(lane) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        // The IR carries only the signed conversion. An unsigned value
        // zero-extended by one bit is the same number read as signed,
        // so `ucvtf` goes through the signed node exactly rather than
        // approximately.
        let source = if signed {
            int
        } else {
            let Some(wider) = source_bits.checked_add(1) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            Expr::zero_ext(int, wider)
        };
        let converted = Expr::sbv_to_fp(source, RoundingMode::NearestTiesEven, ebits, sbits);
        if !self.write_simd_dst(dst, Expr::fp_to_ieee_bv(converted), true) {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    /// `fcvtzs`/`fcvtzu Rd, Rn` — floating point to integer, into
    /// either a general register or a SIMD one. The `z` in the mnemonic
    /// is round-toward-zero, so the rounding mode is carried by the
    /// opcode and no control register is assumed.
    ///
    /// The destination's *class* decides how the result is written, and
    /// this used to ignore it — the doc said "integer register", which
    /// is the form it was written for, and the SIMD-destination form
    /// arrived at the same code without a branch. Two things then go
    /// wrong at once, and both are wrong values rather than declines.
    ///
    /// `write_register_to` sizes the parent at [`LiftCtx::bits`], 64 on
    /// this ISA, where the vector register is 128 — and the SSA pass
    /// versions by name alone, so a later vector read binds to that
    /// 64-bit definition. And its partial-write branch *preserves* the
    /// bits above the element, where an `AArch64` scalar SIMD write
    /// clears them. `write_simd_dst` answers both, which is why every
    /// other scalar-FP handler here already uses it and why
    /// `lift_aarch64_fmov` branches the same way.
    fn lift_aarch64_fp_to_int(&mut self, insn: &Instruction, signed: bool) {
        let (Some(dst), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(lane) = self.simd_view_bits(src) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let Some(value) = self.read_simd_lane_fp(src, lane, 0) else {
            self.push_aarch64_fp_unsupported(insn);
            return;
        };
        let width = self.operand_width(dst);
        // Same trick in reverse: converting into one extra bit of
        // signed range covers the whole unsigned range exactly, and the
        // destination keeps the low `width` bits.
        let converted = if signed {
            Expr::fp_to_sbv(value, RoundingMode::TowardZero, width)
        } else {
            let Some(wider) = width.checked_add(1) else {
                self.push_aarch64_fp_unsupported(insn);
                return;
            };
            Expr::extract(
                Expr::fp_to_sbv(value, RoundingMode::TowardZero, wider),
                width - 1,
                0,
            )
        };
        let written = if self.is_simd_register(dst) {
            self.write_simd_dst(dst, converted, true)
        } else {
            self.write_register_to(dst, converted)
        };
        if !written {
            self.push_aarch64_fp_unsupported(insn);
        }
    }

    fn push_aarch64_fp_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable operand at {addr}", addr = insn.address),
        });
    }
}

/// `FPRecpX` over the stored bit pattern of a `lane`-bit float.
///
/// The architecture's three cases, in the order it spells them. A NaN is
/// quietened and returned. A zero or subnormal — every encoding whose
/// exponent field is zero — yields `Ones(E) - 1`, which is one below the
/// infinity / NaN field and so the largest *finite* exponent; the
/// complement would give the all-ones field and turn a zero into an
/// infinity. Everything else takes the complemented field, which sends an
/// infinity to a zero of the same sign because the significand is cleared
/// either way.
fn reciprocal_exponent(bits: Expr, lane: u16) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(lane)?;
    let frac_bits = sbits.checked_sub(1)?;
    let all_ones = 1u128.checked_shl(u32::from(ebits))?.checked_sub(1)?;
    let sign = Expr::extract(bits.clone(), lane - 1, lane - 1);
    let exponent = Expr::extract(bits.clone(), lane - 2, frac_bits);
    let fraction = Expr::extract(bits.clone(), frac_bits - 1, 0);
    let reciprocal = Expr::Ite {
        cond: Box::new(Expr::eq(exponent.clone(), Expr::konst(0, ebits))),
        then_expr: Box::new(Expr::konst(all_ones.checked_sub(1)?, ebits)),
        else_expr: Box::new(Expr::bv_xor(exponent.clone(), Expr::konst(all_ones, ebits))),
    };
    let value = Expr::concat(Expr::concat(sign, reciprocal), Expr::konst(0, frac_bits));
    let is_nan = Expr::bool_and(
        Expr::eq(exponent, Expr::konst(all_ones, ebits)),
        Expr::ne(fraction, Expr::konst(0, frac_bits)),
    );
    let quiet = Expr::bv_or(
        bits,
        Expr::konst(1u128.checked_shl(u32::from(frac_bits - 1))?, lane),
    );
    Some(Expr::Ite {
        cond: Box::new(is_nan),
        then_expr: Box::new(quiet),
        else_expr: Box::new(value),
    })
}

/// Whether `op` is the `#0.0` immediate that `fcmp`'s compare-with-zero
/// form takes.
fn is_fp_zero_immediate(op: &Operand) -> bool {
    let raw = op.raw.trim().trim_start_matches('#');
    matches!(raw, "0" | "0.0" | "#0.0" | "0.00000")
}
