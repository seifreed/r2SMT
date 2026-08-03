//! The arch-neutral SIMD / packed-vector core of the lifter.
//!
//! Everything here is expressed in four concepts and nothing else: a
//! vector **parent** register, an operand's **view** of it, the **lanes**
//! that tile that view, and the **operation** applied to a lane. None of
//! those is ISA-specific, which is why `addps xmm0, xmm1` and
//! `fadd v0.4s, v1.4s, v2.4s` are the same computation over the same
//! model here — the per-ISA modules differ only in how they derive the
//! lane width (from the mnemonic on x86, from the arrangement on ARM)
//! and in the upper-bits rule they pass to [`LiftCtx::write_simd_dst`].
//!
//! Two consequences are worth stating because they are easy to break.
//! A lane offset is measured from the *view's* offset in the parent, not
//! from bit zero: `AArch32` puts `d1` at bits `[127:64]` of `v0` and `s3`
//! at `[127:96]`, so indexing from zero would silently read the wrong
//! half of the register. And an operand is materialised **once**, with
//! the lanes extracted from that value, so a memory operand costs one
//! load rather than one per lane.
//!
//! The one question this module cannot answer for itself is whether a
//! memory operand's address is modellable — that belongs to a per-ISA
//! memory model, and it arrives through the
//! [`LiftCtx::is_modellable_simd_memory`] hook next to the lifter's other
//! arch dispatch. This module therefore names no [`r2smt_common::Arch`]
//! variant at all.

use r2smt_ir::expr::{Expr, RoundingMode, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::effect::memory_operand_width;
use crate::registers::{RegisterLayout, is_simd_parent, register_layout, simd_parent_bits};

use super::aarch64::all_ones;
use super::{BinOp, LiftCtx};

/// Integer operation a packed ARM vector instruction applies to each
/// lane.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PackedIntOp {
    /// A lane-wise binary operation over two sources.
    Bin(BinOp),
    /// `bic` — `a & ~b`.
    BitClear,
    /// `mvn` / `not` — `~a`, one source.
    Not,
    /// `mov` — copy, one source.
    Copy,
}

/// What a packed ARM vector data-processing instruction computes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PackedOp {
    /// Integer lanes.
    Int(PackedIntOp),
    /// IEEE floating-point lanes.
    Fp(FpArithOp),
}

impl PackedOp {
    /// Number of operands the instruction's packed form carries,
    /// destination included.
    pub(super) const fn operand_count(self) -> usize {
        match self {
            Self::Int(PackedIntOp::Not | PackedIntOp::Copy) => 2,
            _ => 3,
        }
    }

    /// Whether applying the operation to the whole vector view at once
    /// yields the same bits as applying it to each lane separately.
    ///
    /// True for the bitwise family, where no carry crosses a lane
    /// boundary — lowering those lane by lane would multiply the formula
    /// the solver sees by the lane count (sixteen extracts and fifteen
    /// concatenations for a `.16b` operand) for an identical result.
    /// False for the arithmetic family, where the lane boundary is
    /// exactly what stops the carry.
    const fn is_lane_independent(self) -> bool {
        matches!(
            self,
            Self::Int(
                PackedIntOp::Bin(BinOp::And | BinOp::Or | BinOp::Xor)
                    | PackedIntOp::BitClear
                    | PackedIntOp::Not
                    | PackedIntOp::Copy
            )
        )
    }
}

/// Floating-point operation applied to one lane, shared by the scalar
/// (`addss`) and packed (`addps`) handlers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FpArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// IEEE interchange formats the pipeline can name and render.
const FP_HALF_BITS: u16 = 16;
/// IEEE binary32.
const FP_SINGLE_BITS: u16 = 32;
/// IEEE binary64.
pub(crate) const FP_DOUBLE_BITS: u16 = 64;

/// IEEE `(ebits, sbits)` sort for an FP lane width: 16→half `(5, 11)`,
/// 32→single `(8, 24)`, 64→double `(11, 53)`.
///
/// `None` for every other width, which is the only honest answer: the
/// IR carries the sort verbatim into the solver, so resolving an
/// unrecognised width to *some* sort would silently reinterpret the
/// value. x87's 80-bit double-extended format is exactly that case —
/// it is a real format this pipeline cannot render, and it has to
/// decline rather than be read as a double.
pub(crate) fn fp_sort_bits_checked(lane_bits: u16) -> Option<(u16, u16)> {
    Some(match lane_bits {
        FP_HALF_BITS => (5, 11),
        FP_SINGLE_BITS => (8, 24),
        FP_DOUBLE_BITS => (11, 53),
        _ => return None,
    })
}

/// Apply `op` to one lane, given both operands as raw IEEE bit-vectors
/// of `lane_bits`, and return the lane result in the same form.
///
/// The result is a bit-vector rather than a float because `max`/`min`
/// *select* an operand instead of computing one, and the IR's `Ite` is
/// bit-vector-typed — an `Ite` over float-sorted branches has no
/// rendering.
///
/// The rounding mode is the architectural MXCSR default; a function
/// that reprograms MXCSR is rejected by the slicer's guard rather than
/// silently rounded here.
///
/// `None` for a lane width with no IEEE sort. Callers derive the width
/// from an operand's vector view, which can name a width no float
/// format has, and reading such a lane as a double would be a definite
/// wrong value rather than a decline.
pub(crate) fn fp_lane_result(
    op: FpArithOp,
    a_bits: Expr,
    b_bits: Expr,
    lane_bits: u16,
) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
    let a = Expr::bv_to_fp(a_bits.clone(), ebits, sbits);
    let b = Expr::bv_to_fp(b_bits.clone(), ebits, sbits);
    let rm = RoundingMode::NearestTiesEven;
    let arith = match op {
        FpArithOp::Add => Expr::fadd(a, b, rm),
        FpArithOp::Sub => Expr::fsub(a, b, rm),
        FpArithOp::Mul => Expr::fmul(a, b, rm),
        FpArithOp::Div => Expr::fdiv(a, b, rm),
        // `MAXPS: IF SRC1 > SRC2 THEN DEST := SRC1 ELSE DEST := SRC2`
        // (and the mirror for MIN). The comparison is false when either
        // operand is NaN, so the second operand wins on unordered *and*
        // on equality — which is why this is an explicit select rather
        // than `fp.max`/`fp.min`, whose NaN and signed-zero behaviour
        // differs from x86's.
        FpArithOp::Max | FpArithOp::Min => {
            let cond = if matches!(op, FpArithOp::Max) {
                Expr::flt(b, a)
            } else {
                Expr::flt(a, b)
            };
            return Some(Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(a_bits),
                else_expr: Box::new(b_bits),
            });
        }
    };
    Some(Expr::fp_to_ieee_bv(arith))
}

/// Apply an integer packed operation to one lane (or, for the
/// lane-independent operations, to a whole view) of `bits` width.
///
/// `None` when a two-source operation was handed no second source — an
/// operand-count mismatch the caller declines on rather than inventing a
/// value for.
fn packed_int_lane(op: PackedIntOp, a: Expr, b: Option<Expr>, bits: u16) -> Option<Expr> {
    Some(match op {
        PackedIntOp::Bin(bin) => bin.apply(a, b?),
        // The IR has no bitwise NOT, so `~x` is `x XOR all-ones`.
        PackedIntOp::BitClear => Expr::bv_and(a, Expr::bv_xor(b?, all_ones(bits))),
        PackedIntOp::Not => Expr::bv_xor(a, all_ones(bits)),
        PackedIntOp::Copy => a,
    })
}

/// `Extract(Extract(x, inner_hi, 0), hi, lo)` denotes the same bits as
/// `Extract(x, hi, lo)` whenever the outer slice fits inside the inner
/// one, which it always does when the inner slice is an operand's
/// vector view and the outer one is a lane of that view.
///
/// Collapsing matters beyond tidiness: a lane read used to go straight
/// to the vector parent, and materialising the operand's view first
/// would otherwise nest one extract inside every lane of every packed
/// operation, growing the formula the solver sees for no gain.
fn extract_collapsing(value: Expr, hi: u16, lo: u16) -> Expr {
    if let Expr::Extract {
        src,
        hi: inner_hi,
        lo: 0,
    } = &value
        && hi <= *inner_hi
    {
        return Expr::extract((**src).clone(), hi, lo);
    }
    Expr::extract(value, hi, lo)
}

impl LiftCtx {
    /// The vector-register layout `op` names, if it names one.
    fn simd_layout(&self, op: &Operand) -> Option<RegisterLayout> {
        if op.kind != OperandKind::Register {
            return None;
        }
        let layout = register_layout(&op.raw, self.arch)?;
        is_simd_parent(layout.parent, self.arch).then_some(layout)
    }

    /// Whether `op` names a view of a SIMD vector register.
    pub(super) fn is_simd_register(&self, op: &Operand) -> bool {
        self.simd_layout(op).is_some()
    }

    /// Architectural width of this ISA's vector parent.
    fn simd_parent_width(&self) -> Option<u16> {
        simd_parent_bits(self.arch)
    }

    /// The full value of a SIMD operand at `view_bits`, as a single
    /// expression.
    ///
    /// A register view is the slice `[layout.hi:layout.lo]` of its
    /// canonical vector parent. The offset is load-bearing on `AArch32`,
    /// where `d1` sits at bits `[127:64]` of `v0` and `s3` at `[127:96]`
    /// — indexing every view from bit 0, as x86 and `AArch64` allow, would
    /// silently read the wrong half of the register there.
    ///
    /// A memory operand becomes **one** `LoadMem` of `view_bits` through
    /// [`LiftCtx::read_operand_lowered`], which also supplies the
    /// `stk_<base>_<off>` naming so an analyst alias still resolves.
    /// Reading the whole view once — rather than once per lane — is what
    /// keeps a packed operand to a single load.
    ///
    /// `None` for anything else. The caller marks the instruction
    /// unsupported rather than fabricating a value.
    pub(super) fn simd_operand_value(&mut self, op: &Operand, view_bits: u16) -> Option<Expr> {
        if self.is_modellable_simd_memory(op) {
            return Some(self.read_operand_lowered(op, view_bits));
        }
        let layout = self.simd_layout(op)?;
        let parent_bits = self.simd_parent_width()?;
        let parent = Expr::var(layout.parent, parent_bits);
        Some(if layout.width() == parent_bits {
            parent
        } else {
            Expr::extract(parent, layout.hi, layout.lo)
        })
    }

    /// The value of a SIMD operand at its own view width.
    pub(super) fn read_simd_operand(&mut self, op: &Operand) -> Option<Expr> {
        let view = self.simd_view_bits(op)?;
        self.simd_operand_value(op, view)
    }

    /// Read lane `index` of a SIMD operand and reinterpret it as an IEEE
    /// float of the matching sort (32→single, 64→double). Used by the
    /// scalar SSE FP handlers, which always read lane 0.
    pub(super) fn read_simd_lane_fp(
        &mut self,
        op: &Operand,
        lane_bits: u16,
        index: u16,
    ) -> Option<Expr> {
        let raw = self.read_simd_lane_bits(op, lane_bits, index)?;
        let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
        Some(Expr::bv_to_fp(raw, ebits, sbits))
    }

    /// The raw bit-vector of lane `index`, before any float
    /// reinterpretation. Kept separate so the compare handlers can build
    /// a mask without a pointless round trip through the float sort.
    ///
    /// A scalar memory operand is loaded at the *lane* width, not the
    /// vector view: `movss xmm0, dword [rbp - 8]` reads four bytes. Only
    /// lane 0 can be read that way — a lane index addresses a register's
    /// element, and no ISA spells an indexed element of a memory operand.
    pub(super) fn read_simd_lane_bits(
        &mut self,
        op: &Operand,
        lane_bits: u16,
        index: u16,
    ) -> Option<Expr> {
        if self.is_modellable_simd_memory(op) {
            if index != 0 {
                return None;
            }
            return Some(self.read_operand_lowered(op, lane_bits));
        }
        let view = self.simd_view_bits(op)?;
        let value = self.simd_operand_value(op, view)?;
        Self::extract_lane(value, lane_bits, index)
    }

    /// Extract lane `index` (counting from the least-significant) out of
    /// an already-materialised operand value.
    pub(super) fn extract_lane(value: Expr, lane_bits: u16, index: u16) -> Option<Expr> {
        let lo = lane_bits.checked_mul(index)?;
        let hi = lo.checked_add(lane_bits)?.checked_sub(1)?;
        Some(extract_collapsing(value, hi, lo))
    }

    /// Write `lane_value` (a bit-vector of `lane_bits`) to lane `index`
    /// of a SIMD destination, preserving every other bit of the parent.
    ///
    /// A register destination keeps the parent bits around the lane
    /// (legacy SSE scalar semantics, and every ARM element insert). A
    /// memory destination stores exactly `lane_bits` — there is nothing
    /// around the lane to preserve, which is why `movss [rbp - 8], xmm0`
    /// writes four bytes and leaves the rest of the slot alone; only lane
    /// 0 is addressable that way.
    ///
    /// The lane sits at `index` elements above the view's own offset in
    /// the parent, so on `AArch32` a write to `s3` preserves the bits
    /// *below* it as well as above.
    pub(super) fn write_simd_lane(
        &mut self,
        op: &Operand,
        lane_value: Expr,
        lane_bits: u16,
        index: u16,
    ) -> bool {
        if self.is_modellable_simd_memory(op) {
            return index == 0 && self.write_dst(op, lane_value, lane_bits);
        }
        let (Some(layout), Some(parent_bits)) = (self.simd_layout(op), self.simd_parent_width())
        else {
            return false;
        };
        let Some(lane_lo) = lane_bits
            .checked_mul(index)
            .and_then(|offset| layout.lo.checked_add(offset))
        else {
            return false;
        };
        let Some(lane_top) = lane_lo.checked_add(lane_bits) else {
            return false;
        };
        if lane_top > parent_bits {
            return false;
        }
        let parent = Expr::var(layout.parent, parent_bits);
        let value = Self::splice_into_parent(&parent, lane_value, lane_lo, lane_top, parent_bits);
        self.assign(Var::new(layout.parent, parent_bits), value);
        true
    }

    /// Rebuild a parent register with `value` occupying
    /// `[top-1 : lo]` and every other bit taken from its prior contents.
    fn splice_into_parent(parent: &Expr, value: Expr, lo: u16, top: u16, parent_bits: u16) -> Expr {
        let with_high = if top < parent_bits {
            Expr::concat(Expr::extract(parent.clone(), parent_bits - 1, top), value)
        } else {
            value
        };
        if lo > 0 {
            Expr::concat(with_high, Expr::extract(parent.clone(), lo - 1, 0))
        } else {
            with_high
        }
    }

    /// Number of `lane_bits`-wide lanes in a `view_bits`-wide vector
    /// view, or `None` when the view is not a whole multiple of the lane.
    pub(super) fn packed_lane_count(view_bits: u16, lane_bits: u16) -> Option<u16> {
        if lane_bits == 0 || view_bits % lane_bits != 0 {
            return None;
        }
        Some(view_bits / lane_bits)
    }

    /// Assemble per-lane bit-vectors, least-significant lane first, into
    /// one value of the full view width.
    pub(super) fn concat_lanes(lanes: Vec<Expr>) -> Option<Expr> {
        let mut iter = lanes.into_iter();
        let mut acc = iter.next()?;
        for lane in iter {
            acc = Expr::concat(lane, acc);
        }
        Some(acc)
    }

    /// Build the full-view result of a packed floating-point lane
    /// operation, or `None` when any operand is unmodellable.
    ///
    /// Each operand is materialised **once** and the lanes are extracted
    /// from that value, so a memory operand costs one load rather than
    /// one per lane.
    ///
    /// Arch-neutral: every step below is expressed in views, lanes and
    /// operand values, none of which is x86-specific. `addps xmm0, xmm1`
    /// and `fadd v0.4s, v1.4s, v2.4s` are the same computation over the
    /// same model, differing only in how the caller derived the lane
    /// width — from the mnemonic on x86, from the arrangement on ARM.
    pub(super) fn packed_fp_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: &Operand,
        op: FpArithOp,
        lane_bits: u16,
    ) -> Option<Expr> {
        let view = self.simd_instruction_view_bits(&[dst, a_op, b_op])?;
        let count = Self::packed_lane_count(view, lane_bits)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = self.simd_operand_value(b_op, view)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let b = Self::extract_lane(b_val.clone(), lane_bits, index)?;
            lanes.push(fp_lane_result(op, a, b, lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// The integer twin of [`Self::packed_fp_result`]. `b_op` is absent
    /// for the one-source forms (`mvn`, `mov`).
    ///
    /// A lane-independent operation is emitted once over the whole view
    /// instead of once per lane — see [`PackedOp::is_lane_independent`].
    fn packed_int_result(
        &mut self,
        dst: &Operand,
        a_op: &Operand,
        b_op: Option<&Operand>,
        op: PackedIntOp,
        lane_bits: u16,
    ) -> Option<Expr> {
        let mut refs = vec![dst, a_op];
        refs.extend(b_op);
        let view = self.simd_instruction_view_bits(&refs)?;
        let a_val = self.simd_operand_value(a_op, view)?;
        let b_val = match b_op {
            Some(b) => Some(self.simd_operand_value(b, view)?),
            None => None,
        };
        if PackedOp::Int(op).is_lane_independent() {
            return packed_int_lane(op, a_val, b_val, view);
        }
        let count = Self::packed_lane_count(view, lane_bits)?;
        let mut lanes = Vec::with_capacity(usize::from(count));
        for index in 0..count {
            let a = Self::extract_lane(a_val.clone(), lane_bits, index)?;
            let b = match b_val.as_ref() {
                Some(value) => Some(Self::extract_lane(value.clone(), lane_bits, index)?),
                None => None,
            };
            lanes.push(packed_int_lane(op, a, b, lane_bits)?);
        }
        Self::concat_lanes(lanes)
    }

    /// Lower one packed ARM vector data-processing instruction: the same
    /// lane operation applied independently to every lane of the
    /// destination's view.
    ///
    /// `zero_upper` is the ISA's rule for the vector-register bits above
    /// the view. `AArch64` zeroes them on every SIMD write; `AArch32`
    /// preserves them, because a `Dd` operand is one half of a `Qd` and
    /// the other half survives the instruction.
    ///
    /// A decline here is sound by the free-input boundary: the slicer
    /// consumed the destination as a definition and therefore stopped
    /// tracking its upstream definitions, so emitting no assignment
    /// leaves the register a free SSA input rather than bound to a stale
    /// value.
    pub(super) fn lift_packed_vector(
        &mut self,
        insn: &Instruction,
        op: PackedOp,
        lane_bits: u16,
        zero_upper: bool,
    ) {
        let ops = &insn.operands;
        let (Some(dst), Some(a_op)) = (ops.first(), ops.get(1)) else {
            self.push_packed_unsupported(insn);
            return;
        };
        let b_op = ops.get(2);
        let result = match op {
            PackedOp::Fp(fp) => match b_op {
                Some(b) => self.packed_fp_result(dst, a_op, b, fp, lane_bits),
                None => None,
            },
            PackedOp::Int(int) => self.packed_int_result(dst, a_op, b_op, int, lane_bits),
        };
        let Some(value) = result else {
            self.push_packed_unsupported(insn);
            return;
        };
        if !self.write_simd_dst(dst, value, zero_upper) {
            self.push_packed_unsupported(insn);
        }
    }

    fn push_packed_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable packed operand at {addr}", addr = insn.address),
        });
    }

    /// Width of a SIMD operand's view: 128 / 256 / 512 for a register,
    /// and for memory whatever size prefix radare2 attached
    /// (`xmmword [rsi]` → 128). `None` if `op` is neither, or is a
    /// memory operand with no size prefix — guessing a width there would
    /// silently load the wrong number of bytes.
    pub(super) fn simd_view_bits(&self, op: &Operand) -> Option<u16> {
        if self.is_modellable_simd_memory(op) {
            return memory_operand_width(&op.raw);
        }
        Some(self.simd_layout(op)?.width())
    }

    /// The view width an instruction operates at, given its operands.
    ///
    /// A memory operand carries its own size prefix, but a register one
    /// is the more reliable source (r2 always spells the register), so
    /// the first operand that resolves wins.
    pub(super) fn simd_instruction_view_bits(&self, ops: &[&Operand]) -> Option<u16> {
        ops.iter()
            .find(|op| self.is_simd_register(op))
            .or_else(|| ops.first())
            .and_then(|op| self.simd_view_bits(op))
    }

    /// Write a `value` (already sized to the operand's view width) to a
    /// SIMD **register** destination, reconstructing the full-width
    /// parent per the write's upper-bits rule: `zero_upper` (VEX-encoded
    /// `v*` forms, and every `AArch64` SIMD write) zeroes bits above the
    /// view; otherwise (legacy SSE, `AArch32` VFP) the bits above the
    /// view are preserved from the parent's prior value. A memory
    /// destination stores the view width verbatim — there are no parent
    /// bits to reconstruct.
    pub(super) fn write_simd_dst(&mut self, op: &Operand, value: Expr, zero_upper: bool) -> bool {
        if self.is_modellable_simd_memory(op) {
            let Some(width) = self.simd_view_bits(op) else {
                return false;
            };
            return self.write_dst(op, value, width);
        }
        let (Some(layout), Some(parent_bits)) = (self.simd_layout(op), self.simd_parent_width())
        else {
            return false;
        };
        let full = if layout.width() == parent_bits {
            value
        } else if zero_upper {
            Expr::zero_ext(value, parent_bits)
        } else {
            let parent = Expr::var(layout.parent, parent_bits);
            Self::splice_into_parent(&parent, value, layout.lo, layout.hi + 1, parent_bits)
        };
        self.assign(Var::new(layout.parent, parent_bits), full);
        true
    }
}
