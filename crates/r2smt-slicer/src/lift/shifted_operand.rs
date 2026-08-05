//! The shift or extend specifier that follows a register operand on
//! both ARM ISAs.
//!
//! `eor r0, r1, r2, lsl 2` reaches the lifter as **four** top-level
//! operands, and `cmp r0, r1, lsl 2` as three — one more than the
//! handler expects in each case. A handler that reads a fixed number of
//! operands drops the specifier silently and computes the unshifted
//! value, which is a wrong answer rather than a decline.
//!
//! Three things about the spelling are worth writing down, all measured
//! against radare2 6.1.8 rather than assumed:
//!
//! - The amount is **bare** (`lsl 2`, never `lsl #2`) on both ISAs.
//! - `AArch64`'s zero-amount extends carry **no** amount at all
//!   (`add x0, x1, w2, sxtw`), so the specifier is a single token — and
//!   the r2 adapter types that one as `OperandKind::Register`, since it
//!   is all-alphanumeric. Matching on the text is therefore the only
//!   reliable test; the operand kind says nothing.
//! - r2 **canonicalises** the shift-only forms away: `mov r0, r1, asr 3`
//!   comes back as `asr r0, r1, 3`. That is why the `mov` handlers need
//!   no specifier support, and it is a dependency on the disassembler
//!   worth stating rather than rediscovering.

use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use super::aarch32::shift::rotate_right_through_carry;
use super::{BinOp, LiftCtx, parse_immediate};

/// A register shift amount is `Rs[7:0]` on `AArch32`; `AArch64` spells
/// only immediates, so masking is a no-op there.
const SHIFT_AMOUNT_MASK: u128 = 0xff;

/// Where `Operand2` sits in the three-operand shape `ands Rd, Rn, Op`.
pub(super) const OPERAND2_INDEX_ARITH3: usize = 2;

/// Where `Operand2` sits in the destination-less shape `tst Rn, Op`.
pub(super) const OPERAND2_INDEX_COMPARE: usize = 1;

/// What a specifier does to the register it follows.
enum ShiftSpec<'a> {
    /// `lsl` / `lsr` / `asr` / `ror` by an immediate or a register.
    By(BinOp, &'a str),
    /// `rrx` — one bit right through the carry, no amount.
    ThroughCarry,
    /// `AArch64` `uxtb 2` / `sxtw` — take the low `bits`, extend them
    /// across the destination, then shift left by `shift`.
    Extend {
        bits: u16,
        signed: bool,
        shift: &'a str,
    },
}

/// Parse a specifier, or `None` when the operand is an ordinary one.
///
/// Deliberately text-driven: the operand *kind* cannot tell these apart
/// (a bare `sxtw` arrives typed as a register), and the token set is
/// closed and small.
fn parse_shift_spec(raw: &str) -> Option<ShiftSpec<'_>> {
    let text = raw.trim();
    let (head, tail) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    let tail = tail.trim();
    let op = match head {
        "lsl" => Some(BinOp::Shl),
        "lsr" => Some(BinOp::Shr),
        "asr" => Some(BinOp::Sar),
        "ror" => Some(BinOp::Ror),
        _ => None,
    };
    if let Some(op) = op {
        // A bare `lsl` with no amount is not a spelling either ISA
        // produces; refusing it keeps the parser from claiming an
        // operand it cannot lower.
        return (!tail.is_empty()).then_some(ShiftSpec::By(op, tail));
    }
    if head == "rrx" && tail.is_empty() {
        return Some(ShiftSpec::ThroughCarry);
    }
    let (signed, width) = match head {
        "uxtb" => (false, 8),
        "uxth" => (false, 16),
        "uxtw" => (false, 32),
        "uxtx" => (false, 64),
        "sxtb" => (true, 8),
        "sxth" => (true, 16),
        "sxtw" => (true, 32),
        "sxtx" => (true, 64),
        _ => return None,
    };
    Some(ShiftSpec::Extend {
        bits: width,
        signed,
        shift: tail,
    })
}

impl LiftCtx {
    /// Read source operand `index`, folding a shift or extend specifier
    /// at `index + 1` into the value.
    ///
    /// Every handler whose last source can carry one must read it
    /// through here. Reading the operand alone is the silent-wrong-value
    /// bug this module exists for.
    pub(super) fn read_source_with_shift(
        &mut self,
        insn: &Instruction,
        index: usize,
        width: u16,
    ) -> Expr {
        let Some(op) = insn.operands.get(index) else {
            return Expr::Unknown(String::new());
        };
        let value = self.read_operand_at(op, width);
        let Some(spec) = insn
            .operands
            .get(index + 1)
            .and_then(|next| parse_shift_spec(&next.raw))
        else {
            return value;
        };
        match spec {
            ShiftSpec::ThroughCarry => rotate_right_through_carry(value, width),
            ShiftSpec::By(op, amount) => {
                let amount = self.shift_amount(amount, width);
                op.apply(value, amount)
            }
            ShiftSpec::Extend {
                bits,
                signed,
                shift,
            } => {
                let narrowed = match bits.checked_sub(1) {
                    Some(hi) => Expr::extract(value, hi, 0),
                    None => value,
                };
                let extended = if signed {
                    Expr::sign_ext(narrowed, width)
                } else {
                    Expr::zero_ext(narrowed, width)
                };
                if shift.is_empty() {
                    extended
                } else {
                    let amount = self.shift_amount(shift, width);
                    Expr::shl(extended, amount)
                }
            }
        }
    }

    /// Write the `C` an `AArch32` logical s-form leaves behind, given the
    /// operand index its `Operand2` sits at.
    ///
    /// C is the **shifter carry-out of Operand2** (ARM DDI 0487, A32
    /// `ANDS` / `ORRS` / `EORS` / `BICS` / `TST` / `TEQ`), and V is not
    /// written at all — which is why neither flag is assigned here unless
    /// there is a specifier to take a carry from.
    ///
    /// With no specifier the shifter carry-out *is* the carry-in, so C
    /// keeps its value and nothing may be emitted: assigning even a free
    /// value there would destroy a carry the slice already holds, and
    /// assigning a constant would fabricate one. With a specifier C is a
    /// bit of the *unshifted* source, which this caller no longer has, so
    /// a free value is the honest answer — it can only widen a verdict.
    pub(super) fn set_aarch32_logical_carry(&mut self, insn: &Instruction, index: usize) {
        let shifted = insn
            .operands
            .get(index + 1)
            .is_some_and(|next| parse_shift_spec(&next.raw).is_some());
        if shifted {
            self.set_flag("CF", Expr::Unknown(String::new()));
        }
    }

    /// The shift amount an specifier spells: an immediate as a constant,
    /// a register masked to its low byte (`Rs[7:0]`, ARM DDI 0487
    /// `Shift_C`) because a bit-vector shift by `0x100` clears the
    /// register where the machine shifts by nothing.
    fn shift_amount(&self, raw: &str, width: u16) -> Expr {
        if let Some(value) = parse_immediate(raw) {
            return Expr::konst(u128::from(value), width);
        }
        let operand = Operand {
            raw: raw.to_string(),
            kind: OperandKind::Register,
        };
        let register = self.read_operand_at(&operand, width);
        Expr::bv_and(register, Expr::konst(SHIFT_AMOUNT_MASK, width))
    }
}
