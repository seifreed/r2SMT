//! The `AArch32` shift and rotate family: its vocabulary and its
//! lowering.
//!
//! Split out of the parent when adding the standalone `rrx` / `ror`
//! mnemonics took it to 1 818 lines, past the promote line. The cut
//! follows `vfp.rs`'s precedent: the vocabulary travels with the
//! lowering, because [`ShiftForm`] and [`CarryOut`] have exactly one
//! inbound edge each — the dispatch, and this file itself.
//!
//! One item goes the other way. [`rotate_right_through_carry`] is read
//! by the parent's addressing model, because `rrx` has two spellings —
//! a mnemonic and an index shift — and they are one operation. Keeping
//! them on one function is what stops them drifting apart, which is the
//! same argument the parent's own doc makes about restating a width.

use r2smt_ir::expr::{Expr, Var};
use r2smt_ir::program::{Instruction, OperandKind};

use super::super::{BinOp, LiftCtx, nonzero_width};
use super::parse_aarch32_immediate;

/// A register shift amount is `Rs[7:0]` (ARM DDI 0487, `Shift_C`).
const SHIFT_AMOUNT_MASK: u128 = 0xff;

/// Which shift a mnemonic performs. `rrx` is not a [`BinOp`] for the
/// same reason [`super::Aarch32Shift`] splits it out: it takes no amount
/// and it is not a function of the register alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShiftForm {
    /// `lsl` / `lsr` / `asr` / `ror`, whose amount is an operand.
    By(BinOp),
    /// `rrx`, one bit right through the carry.
    ThroughCarry,
}

/// What a shift does to the carry flag. The three answers are genuinely
/// different and collapsing any two of them is a bug: leaving C alone
/// where the shift defines it keeps a stale value the architecture
/// overwrote, and freeing it where the shift leaves it alone throws away
/// a carry the slice may already have computed.
enum CarryOut {
    /// The architecture leaves C as it found it — a zero shift amount.
    Unchanged,
    /// The bit that left the register.
    Bit(Expr),
    /// A shift by a register amount: whether C changes at all depends on
    /// a value this lift does not have.
    Free,
}

/// The bit a shift by `amount` pushes out of `src`.
///
/// `lsl` takes it from the top (`src[width - n]`); every right-moving
/// form takes `src[n - 1]`, rotate included — a rotate discards nothing,
/// but the bit that lands in C is still the last one to leave the low
/// end.
fn shift_carry_out(op: BinOp, src: &Expr, amount: u16, width: u16) -> CarryOut {
    if amount == 0 {
        return CarryOut::Unchanged;
    }
    let bit = match op {
        BinOp::Shl => width.checked_sub(amount),
        BinOp::Shr | BinOp::Sar | BinOp::Ror => amount.checked_sub(1),
        _ => None,
    };
    // An amount the encoding cannot spell (`lsl #32` and beyond) is not
    // a shift this handler can name a carry for.
    match bit.filter(|bit| *bit < width) {
        Some(bit) => CarryOut::Bit(Expr::extract(src.clone(), bit, bit)),
        None => CarryOut::Free,
    }
}

/// `rrx` — rotate `value` right through the carry flag by one.
///
/// The carry becomes the new top bit and every other bit moves down one,
/// which is exactly bits `[bits:1]` of the 33-bit concatenation. Shared
/// with the addressing model so the two spellings of one operation
/// cannot drift apart.
pub(super) fn rotate_right_through_carry(value: Expr, bits: u16) -> Expr {
    let through = Expr::concat(Expr::Var(Var::new("CF", 1)), value);
    Expr::extract(through, bits, 1)
}

impl LiftCtx {
    /// The shift and rotate family: `lsl` / `lsr` / `asr` / `ror` in
    /// their 3-operand shape, and `rrx` in its 2-operand one.
    ///
    /// It is one handler rather than an arm of the shared arithmetic
    /// helper because a shift's flags are not an arithmetic instruction's
    /// and its amount is not an arithmetic operand:
    ///
    /// - **C is the last bit shifted out**, which the arithmetic helper
    ///   cannot say and left free. For an immediate amount it is one
    ///   bit of the source, and the same one for `lsr` / `asr` / `ror`
    ///   (`src[n-1]`) because a rotate discards nothing.
    /// - **A zero amount leaves C alone**, so nothing is emitted for it.
    ///   Assigning a free value there would destroy a carry the slice may
    ///   already have computed.
    /// - **V is untouched by a shift.** The arithmetic helper writes it.
    /// - **A register amount is `Rs[7:0]`**, and the low eight bits are
    ///   load-bearing rather than cosmetic: `bvshl` by 0x100 gives zero
    ///   where the architecture shifts by nothing at all.
    pub(super) fn lift_aarch32_shift(
        &mut self,
        insn: &Instruction,
        form: ShiftForm,
        sets_flags: bool,
    ) {
        let (Some(dst), Some(src_op)) = (insn.operands.first(), insn.operands.get(1)) else {
            self.unsupported_aarch32(insn, "shift expects Rd, Rm[, amount]");
            return;
        };
        if dst.kind != OperandKind::Register {
            self.unsupported_aarch32(insn, "shift non-register destination");
            return;
        }
        let Some(width) = nonzero_width(self.operand_width(dst)) else {
            self.unsupported_aarch32(insn, "shift zero-width destination");
            return;
        };
        let src = self.read_operand_at(src_op, width);
        let (value, carry_out) = match form {
            ShiftForm::ThroughCarry => (
                rotate_right_through_carry(src.clone(), width),
                CarryOut::Bit(Expr::extract(src, 0, 0)),
            ),
            ShiftForm::By(op) => {
                let Some(amount_op) = insn.operands.get(2) else {
                    self.unsupported_aarch32(insn, "shift expects Rd, Rm, amount");
                    return;
                };
                let immediate = (amount_op.kind == OperandKind::Immediate)
                    .then(|| parse_aarch32_immediate(&amount_op.raw))
                    .flatten()
                    .and_then(|value| u16::try_from(value).ok());
                let amount = self.read_operand_at(amount_op, width);
                let (amount, carry) = match immediate {
                    Some(n) => (amount, shift_carry_out(op, &src, n, width)),
                    // Only the low byte of the amount register counts.
                    None => (
                        Expr::bv_and(amount, Expr::konst(SHIFT_AMOUNT_MASK, width)),
                        CarryOut::Free,
                    ),
                };
                (op.apply(src, amount), carry)
            }
        };
        let tmp = self.new_temp(insn.address, width);
        self.assign(tmp.clone(), value);
        let result = Expr::Var(tmp);
        if sets_flags {
            self.set_flag("ZF", Expr::eq(result.clone(), Expr::konst(0, width)));
            self.set_flag("SF", Expr::slt(result.clone(), Expr::konst(0, width)));
            match carry_out {
                CarryOut::Bit(carry) => self.set_flag("CF", carry),
                // A shift by a register amount can be zero, and then C
                // keeps its value — so which of the two happens is not
                // known at lift time and a free value is the honest
                // answer.
                CarryOut::Free => self.set_flag("CF", Expr::Unknown(String::new())),
                CarryOut::Unchanged => {}
            }
        }
        if !self.write_register_to(dst, result) {
            self.unsupported_aarch32(insn, "shift destination not a supported register");
        }
    }
}
