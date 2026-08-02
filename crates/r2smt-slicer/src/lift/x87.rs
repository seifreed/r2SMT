//! x87 floating-point lifting.
//!
//! The x87 data registers are a *stack*: eight physical registers
//! addressed as `ST(0)`..`ST(7)` relative to a rotating TOP pointer,
//! where almost every instruction pushes or pops (Intel SDM Vol. 1
//! §8.1). The flat [`RegisterLayout`](crate::registers::RegisterLayout)
//! table cannot express that — `st(1)` names a different physical
//! register after every `fld` — so the stack is modelled twice, at two
//! granularities that answer two different questions:
//!
//! * **The effect table** (`effect/x86.rs`) collapses the whole file
//!   onto one pseudo-register `st` that every x87 instruction both
//!   defines and uses. That is deliberately blunt: a slicer which
//!   cannot tell `st(0)` from `st(1)` keeps the entire chain rather
//!   than stepping over the instruction that rotated TOP underneath it.
//! * **The lifter** keeps [`X87Stack`], a bounded `Vec<Expr>` living in
//!   `LiftCtx` for one slice. A slice is a linear instruction sequence,
//!   so pushes and pops compose exactly and no numbering is needed:
//!   `fld [a]; fld [b]; faddp; fstp [c]` resolves to a single store of
//!   `a + b`, and no x87 value ever becomes an IR variable.
//!
//! ## Value model
//!
//! Stack slots hold **IEEE binary64 bit patterns**, keeping the
//! pipeline-wide invariant that registers are bit-vector-typed (floats
//! appear only inside an expression, wrapped back up by
//! `Expr::FpToIeeeBv` before they reach a slot).
//!
//! The hardware computes at 80-bit double-extended precision, so this
//! is the semantics of an FPU whose control-word precision-control
//! field selects double rather than the architectural default of
//! extended. Two consequences, both stated rather than hidden:
//!
//! * Loads and stores of `m32fp` / `m64fp` are exact — widening
//!   binary32 to binary64 loses nothing, and a `m64fp` store of a
//!   value that only ever passed through binary64 arithmetic is the
//!   value itself. For `add` and `mul` of single-precision inputs the
//!   round-to-single at the store is also the hardware's single
//!   rounding, because the exact product of two binary32 values fits a
//!   binary64 significand. `div` and `sqrt` can double-round and so
//!   differ from the hardware in the last bit.
//! * An 80-bit operand (`tbyte`) is **declined**, not approximated:
//!   `fp_sort_bits_checked` has no sort for it, the effect table
//!   therefore reports [`InstructionKind::Other`](crate::effect::InstructionKind::Other),
//!   and the slicer truncates.
//!
//! The rounding mode is likewise pinned to the control word's default
//! (round to nearest, ties to even) with no guard yet on `fldcw`; that
//! guard, the compare / status-word idiom (`fcom` / `fnstsw` / `fcomi`)
//! and the 80-bit sort are follow-up work. Every mnemonic they involve
//! is outside [`classify`], so today they truncate the slice.
//!
//! ## Declining
//!
//! Everything [`classify`] rejects is rejected on the *instruction*
//! alone, so the effect table sees the same answer and the slicer never
//! keeps an instruction the lifter would drop. The one decline that
//! depends on lifter state instead is a push into a full stack, which
//! the instruction cannot predict. It emits
//! [`IrStmt::Unsupported`] — which the SMT layer renders as nothing at
//! all — so it additionally *havocs* the modelled stack: every later
//! read then produces a free symbolic input rather than a stale value
//! from a slot the hardware would have overwritten.

use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::effect::memory_operand_width;

use super::{
    FP_DOUBLE_BITS, FpArithOp, LiftCtx, fp_lane_result, fp_sort_bits_checked, x86_memory_modellable,
};

/// The single canonical data-flow node the x87 register stack collapses
/// onto in the effect tables. Recovered from the disassembler spelling
/// `st(0)`..`st(7)` by [`crate::effect::registers_in_operand`], which
/// splits on non-alphanumerics and so yields the bare token `st`.
pub(crate) const X87_STACK_REGISTER: &str = "st";

/// Number of x87 data registers (Intel SDM Vol. 1 §8.1.2).
const X87_STACK_DEPTH: usize = 8;

/// Width every modelled stack slot carries: IEEE binary64.
const X87_SLOT_BITS: u16 = FP_DOUBLE_BITS;

/// `+1.0` as an IEEE binary64 bit pattern — what `fld1` pushes.
const X87_ONE: u128 = 0x3ff0_0000_0000_0000;

/// `+0.0` as an IEEE binary64 bit pattern — what `fldz` pushes.
const X87_ZERO: u128 = 0;

/// IEEE binary64 sign bit, flipped by `fchs`.
const X87_SIGN_BIT: u128 = 0x8000_0000_0000_0000;

/// IEEE binary64 magnitude mask (every bit but the sign), which `fabs`
/// applies. Clearing the sign bit is exact for every value including
/// the NaNs and infinities, so it needs no float sort at all.
const X87_MAGNITUDE_MASK: u128 = 0x7fff_ffff_ffff_ffff;

/// Memory widths the floating-point load / store family accepts:
/// `m32fp` and `m64fp`. `m80fp` is absent by design — see the module
/// docs.
const X87_FLOAT_WIDTHS: [u16; 2] = [32, 64];

/// Memory widths `fild` / `fistp` accept: `m16int`, `m32int`, `m64int`.
const X87_INT_WIDTHS: [u16; 3] = [16, 32, 64];

/// Memory widths the *non-popping* `fist` accepts. The SDM gives it
/// `m16int` and `m32int` only; the 64-bit integer store exists solely
/// in the popping form.
const X87_INT_STORE_WIDTHS: [u16; 2] = [16, 32];

/// Rounding mode every x87 conversion and arithmetic operation assumes:
/// the control word's reset value, round to nearest with ties to even
/// (Intel SDM Vol. 1 §8.1.5.3).
const X87_ROUNDING: RoundingMode = RoundingMode::NearestTiesEven;

/// Why an x87 instruction declined at lift time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X87Error {
    /// A push found all eight data registers occupied. The hardware
    /// signals a stack-overflow exception and writes its own
    /// indefinite quiet NaN; the value model has no term for that, so
    /// the instruction declines and the modelled stack is havoced.
    StackOverflow,
    /// The instruction's operand list did not survive to the handler.
    /// [`classify`] guarantees the shape, so this is unreachable
    /// through the normal path and exists to keep the handler total.
    MalformedOperands,
    /// A width reached the value model that has no renderable IEEE
    /// sort. [`classify`] filters these too; the second check keeps the
    /// conversion helpers honest on their own terms.
    UnmodelledWidth,
    /// The destination could not be written — an address the byte model
    /// declines to build.
    UnwritableDestination,
}

impl X87Error {
    /// Analyst-facing reason, recorded on the `Unsupported` statement.
    const fn reason(self) -> &'static str {
        match self {
            Self::StackOverflow => "x87 stack overflow",
            Self::MalformedOperands => "malformed x87 operands",
            Self::UnmodelledWidth => "unmodelled x87 operand width",
            Self::UnwritableDestination => "unmodellable x87 destination",
        }
    }
}

/// Symbolic model of the eight x87 data registers over one slice.
///
/// Slots are stored bottom-first, so `ST(0)` is the last element and a
/// push is a plain `Vec::push`. Reading below the modelled bottom is
/// *underflow*, and it is sound rather than an error: the value was
/// there before the slice started, so a fresh free symbolic input names
/// it exactly. Pushing past [`X87_STACK_DEPTH`] is not — the hardware
/// would have raised an exception instead of storing the value — so it
/// declines.
pub(super) struct X87Stack {
    slots: Vec<Expr>,
    free_inputs: u32,
}

impl X87Stack {
    pub(super) const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_inputs: 0,
        }
    }

    /// Forget every modelled slot. Later reads produce fresh free
    /// inputs, which is the sound reading of "this analysis no longer
    /// knows what the stack holds".
    fn havoc(&mut self) {
        self.slots.clear();
    }

    fn fresh_input(&mut self) -> Expr {
        let value = Expr::var(format!("x87_in_{n}", n = self.free_inputs), X87_SLOT_BITS);
        self.free_inputs = self.free_inputs.saturating_add(1);
        value
    }

    /// Materialise slots below the modelled bottom until `ST(index)`
    /// exists, so repeated reads of the same underflowing slot observe
    /// the same free input.
    fn ensure(&mut self, index: usize) {
        while self.slots.len() <= index {
            let value = self.fresh_input();
            self.slots.insert(0, value);
        }
    }

    /// Position of `ST(index)` in the bottom-first slot vector.
    fn offset(&self, index: usize) -> Option<usize> {
        self.slots.len().checked_sub(index.checked_add(1)?)
    }

    fn read(&mut self, index: usize) -> Expr {
        self.ensure(index);
        match self.offset(index).and_then(|at| self.slots.get(at)) {
            Some(value) => value.clone(),
            None => self.fresh_input(),
        }
    }

    fn write(&mut self, index: usize, value: Expr) {
        self.ensure(index);
        if let Some(at) = self.offset(index)
            && let Some(slot) = self.slots.get_mut(at)
        {
            *slot = value;
        }
    }

    fn push(&mut self, value: Expr) -> Result<(), X87Error> {
        if self.slots.len() >= X87_STACK_DEPTH {
            self.havoc();
            return Err(X87Error::StackOverflow);
        }
        self.slots.push(value);
        Ok(())
    }

    /// Move TOP up one slot without reading the value it discarded.
    fn drop_top(&mut self) {
        self.slots.truncate(self.slots.len().saturating_sub(1));
    }

    fn pop(&mut self) -> Expr {
        let top = self.read(0);
        self.drop_top();
        top
    }

    fn exchange(&mut self, index: usize) {
        let top = self.read(0);
        let other = self.read(index);
        self.write(0, other);
        self.write(index, top);
    }
}

/// The value format a memory operand carries.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemFormat {
    /// IEEE binary32 / binary64 (`m32fp` / `m64fp`).
    Float,
    /// Two's-complement integer (`m16int` / `m32int` / `m64int`).
    Integer,
}

/// Where an arithmetic instruction's second input comes from.
#[derive(Clone, Copy)]
enum ArithSrc {
    /// Another stack slot, `ST(i)`.
    Slot(usize),
    /// A memory operand of this width, always floating-point — the
    /// integer forms (`fiadd`, `fimul`, …) are not modelled.
    Memory(u16),
}

/// In-place unary operations on `ST(0)`.
#[derive(Clone, Copy)]
enum UnaryOp {
    /// `fabs` — clear the sign bit.
    Abs,
    /// `fchs` — flip the sign bit.
    Chs,
    /// `fsqrt` — square root under the default rounding mode.
    Sqrt,
}

/// A recognised x87 instruction, resolved to what it does to the stack.
enum X87Form {
    /// Push a fixed IEEE binary64 bit pattern (`fld1`, `fldz`).
    PushConst(u128),
    /// Push a value converted from `operands[0]` (`fld`, `fild`).
    PushMemory { width: u16, format: MemFormat },
    /// Push a copy of `ST(i)` (`fld st(i)`).
    PushSlot(usize),
    /// Convert `ST(0)` into `operands[0]` (`fst`/`fstp`, `fist`/`fistp`).
    StoreMemory {
        width: u16,
        format: MemFormat,
        pop: bool,
    },
    /// Copy `ST(0)` into `ST(i)` (`fst`/`fstp st(i)`).
    StoreSlot { index: usize, pop: bool },
    /// `ST(dst) := ST(dst) op src`, or its operand-reversed form.
    Arith {
        op: FpArithOp,
        dst: usize,
        src: ArithSrc,
        reversed: bool,
        pop: bool,
    },
    /// Swap `ST(0)` with `ST(i)` (`fxch`).
    Exchange(usize),
    /// In-place unary on `ST(0)`.
    Unary(UnaryOp),
}

/// Whether `insn` is an x87 instruction in a shape the lifter models.
///
/// The single source of truth for two gates that must agree: the effect
/// table (which decides whether the slicer keeps the instruction or
/// truncates) and the pre-empt in [`super::is_simd_instruction`] (which
/// decides whether the per-mnemonic handler runs instead of the ESIL
/// ladder). Both consult this, so neither can drift from what
/// [`LiftCtx::lift_instruction_x87`] actually lowers.
pub(crate) fn is_modelled_x87(insn: &Instruction) -> bool {
    classify(insn).is_some()
}

/// Resolve `insn` into the stack effect the lifter will apply, or
/// `None` when the mnemonic is outside the modelled set or its operands
/// are in a shape that cannot be lowered soundly (an unmodellable
/// address, an absent or unrenderable size prefix, an operand count the
/// encoding does not have).
fn classify(insn: &Instruction) -> Option<X87Form> {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    let ops = insn.operands.as_slice();
    match mnemonic.as_str() {
        "fld1" => no_operands(ops, X87Form::PushConst(X87_ONE)),
        "fldz" => no_operands(ops, X87Form::PushConst(X87_ZERO)),
        "fabs" => no_operands(ops, X87Form::Unary(UnaryOp::Abs)),
        "fchs" => no_operands(ops, X87Form::Unary(UnaryOp::Chs)),
        "fsqrt" => no_operands(ops, X87Form::Unary(UnaryOp::Sqrt)),
        "fxch" => classify_exchange(ops),
        "fld" => classify_load(ops, MemFormat::Float, &X87_FLOAT_WIDTHS),
        "fild" => classify_load(ops, MemFormat::Integer, &X87_INT_WIDTHS),
        "fst" => classify_store(ops, MemFormat::Float, &X87_FLOAT_WIDTHS, false),
        "fstp" => classify_store(ops, MemFormat::Float, &X87_FLOAT_WIDTHS, true),
        "fist" => classify_store(ops, MemFormat::Integer, &X87_INT_STORE_WIDTHS, false),
        "fistp" => classify_store(ops, MemFormat::Integer, &X87_INT_WIDTHS, true),
        other => classify_arith(other, ops),
    }
}

fn no_operands(ops: &[Operand], form: X87Form) -> Option<X87Form> {
    ops.is_empty().then_some(form)
}

fn only_operand(ops: &[Operand]) -> Option<&Operand> {
    match ops {
        [op] => Some(op),
        _ => None,
    }
}

/// `fxch` with no operand is `fxch st(1)`.
fn classify_exchange(ops: &[Operand]) -> Option<X87Form> {
    match ops {
        [] => Some(X87Form::Exchange(1)),
        [op] => Some(X87Form::Exchange(slot_index(op)?)),
        _ => None,
    }
}

/// `fld` has a register-to-register encoding (`fld st(i)`); `fild`, an
/// integer conversion, does not.
fn classify_load(ops: &[Operand], format: MemFormat, widths: &[u16]) -> Option<X87Form> {
    let op = only_operand(ops)?;
    if let Some(index) = slot_index(op) {
        return (format == MemFormat::Float).then_some(X87Form::PushSlot(index));
    }
    Some(X87Form::PushMemory {
        width: modellable_memory_width(op, widths)?,
        format,
    })
}

/// Same asymmetry as [`classify_load`]: `fst`/`fstp` can name a stack
/// slot, `fist`/`fistp` cannot.
fn classify_store(
    ops: &[Operand],
    format: MemFormat,
    widths: &[u16],
    pop: bool,
) -> Option<X87Form> {
    let op = only_operand(ops)?;
    if let Some(index) = slot_index(op) {
        return (format == MemFormat::Float).then_some(X87Form::StoreSlot { index, pop });
    }
    Some(X87Form::StoreMemory {
        width: modellable_memory_width(op, widths)?,
        format,
        pop,
    })
}

/// The arithmetic family, in Intel operand order.
///
/// `FSUB ST(i), ST(0)` computes `ST(i) − ST(0)` and `FSUBR ST(i),
/// ST(0)` computes `ST(0) − ST(i)`; the popping forms behave the same
/// and then discard `ST(0)`. This is worth naming because the AT&T
/// assemblers famously swap the two mnemonics for the register-only
/// encodings — radare2 disassembles Intel syntax, so what is written is
/// what is meant.
fn classify_arith(mnemonic: &str, ops: &[Operand]) -> Option<X87Form> {
    let (base, pop) = match mnemonic.strip_suffix('p') {
        Some(base) if !base.is_empty() => (base, true),
        _ => (mnemonic, false),
    };
    let (op, reversed) = match base {
        "fadd" => (FpArithOp::Add, false),
        "fmul" => (FpArithOp::Mul, false),
        "fsub" => (FpArithOp::Sub, false),
        "fsubr" => (FpArithOp::Sub, true),
        "fdiv" => (FpArithOp::Div, false),
        "fdivr" => (FpArithOp::Div, true),
        _ => return None,
    };
    match ops {
        // The popping forms have a no-operand encoding meaning
        // `<op>p st(1), st(0)`; the non-popping ones do not.
        [] => pop.then_some(X87Form::Arith {
            op,
            dst: 1,
            src: ArithSrc::Slot(0),
            reversed,
            pop,
        }),
        // `fadd m32fp` — `ST(0) := ST(0) op m`. There is no popping
        // memory encoding, and a bare `<op> st(i)` is not a spelling
        // the disassembler emits: without both operands there is no
        // saying which one is the destination, so it declines.
        [mem] if !pop => Some(X87Form::Arith {
            op,
            dst: 0,
            src: ArithSrc::Memory(modellable_memory_width(mem, &X87_FLOAT_WIDTHS)?),
            reversed,
            pop,
        }),
        [dst, src] => Some(X87Form::Arith {
            op,
            dst: slot_index(dst)?,
            src: ArithSrc::Slot(slot_index(src)?),
            reversed,
            pop,
        }),
        _ => None,
    }
}

/// Index of the x87 stack slot an operand names, in radare2's
/// disassembly spelling `st(0)`..`st(7)`.
///
/// The bare `st0` form is deliberately not accepted: that is what
/// radare2 emits in ESIL, and an ESIL string never reaches an operand
/// list.
fn slot_index(op: &Operand) -> Option<usize> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    let index: usize = raw.strip_prefix("st(")?.strip_suffix(')')?.parse().ok()?;
    (index < X87_STACK_DEPTH).then_some(index)
}

/// Width of a memory operand the byte-granular model can address and
/// the value model can name, restricted to the widths this family
/// accepts.
///
/// An operand with no size prefix declines rather than assuming pointer
/// width — the x87 families span 16 to 80 bits, so guessing would read
/// the wrong number of bytes.
fn modellable_memory_width(op: &Operand, widths: &[u16]) -> Option<u16> {
    if op.kind != OperandKind::Memory || !x86_memory_modellable(op) {
        return None;
    }
    let width = memory_operand_width(&op.raw)?;
    widths.contains(&width).then_some(width)
}

/// Convert a value read from memory into a slot value.
///
/// Widening binary32 to binary64 and converting a 16- or 32-bit integer
/// are both exact; a 64-bit integer rounds under the control word's
/// mode, which is why the rounding mode is threaded through rather than
/// asserted away.
fn to_slot(raw: Expr, width: u16, format: MemFormat) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(X87_SLOT_BITS)?;
    Some(match format {
        MemFormat::Float => {
            if width == X87_SLOT_BITS {
                raw
            } else {
                let (src_e, src_s) = fp_sort_bits_checked(width)?;
                Expr::fp_to_ieee_bv(Expr::fp_to_fp(
                    Expr::bv_to_fp(raw, src_e, src_s),
                    X87_ROUNDING,
                    ebits,
                    sbits,
                ))
            }
        }
        MemFormat::Integer => Expr::fp_to_ieee_bv(Expr::sbv_to_fp(raw, X87_ROUNDING, ebits, sbits)),
    })
}

/// Convert a slot value into the form a memory destination stores.
fn from_slot(value: Expr, width: u16, format: MemFormat) -> Option<Expr> {
    let (ebits, sbits) = fp_sort_bits_checked(X87_SLOT_BITS)?;
    Some(match format {
        MemFormat::Float => {
            if width == X87_SLOT_BITS {
                value
            } else {
                let (dst_e, dst_s) = fp_sort_bits_checked(width)?;
                Expr::fp_to_ieee_bv(Expr::fp_to_fp(
                    Expr::bv_to_fp(value, ebits, sbits),
                    X87_ROUNDING,
                    dst_e,
                    dst_s,
                ))
            }
        }
        // `fist` / `fistp` round per the control word, unlike SSE's
        // `cvtt*` family which truncates by opcode.
        MemFormat::Integer => {
            Expr::fp_to_sbv(Expr::bv_to_fp(value, ebits, sbits), X87_ROUNDING, width)
        }
    })
}

impl LiftCtx {
    pub(super) fn lift_instruction_x87(&mut self, insn: &Instruction) {
        let Some(form) = classify(insn) else {
            self.decline_x87(insn, "unmodelled x87 form");
            return;
        };
        if let Err(err) = self.apply_x87(insn, &form) {
            self.decline_x87(insn, err.reason());
        }
    }

    fn apply_x87(&mut self, insn: &Instruction, form: &X87Form) -> Result<(), X87Error> {
        match *form {
            X87Form::PushConst(bits) => self.x87.push(Expr::konst(bits, X87_SLOT_BITS)),
            X87Form::PushSlot(index) => {
                let value = self.x87.read(index);
                self.x87.push(value)
            }
            X87Form::PushMemory { width, format } => {
                let value = self.x87_memory_read(insn, width, format)?;
                self.x87.push(value)
            }
            X87Form::StoreMemory { width, format, pop } => {
                self.x87_store_memory(insn, width, format, pop)
            }
            X87Form::StoreSlot { index, pop } => {
                let value = self.x87.read(0);
                self.x87.write(index, value);
                if pop {
                    self.x87.drop_top();
                }
                Ok(())
            }
            X87Form::Arith {
                op,
                dst,
                src,
                reversed,
                pop,
            } => self.x87_arith(insn, op, dst, src, reversed, pop),
            X87Form::Exchange(index) => {
                self.x87.exchange(index);
                Ok(())
            }
            X87Form::Unary(op) => self.x87_unary(op),
        }
    }

    /// Load `operands[0]` through the byte-granular memory model and
    /// convert it to a slot value.
    fn x87_memory_read(
        &mut self,
        insn: &Instruction,
        width: u16,
        format: MemFormat,
    ) -> Result<Expr, X87Error> {
        let op = insn
            .operands
            .first()
            .ok_or(X87Error::MalformedOperands)?
            .clone();
        let raw = self.read_operand_lowered(&op, width);
        to_slot(raw, width, format).ok_or(X87Error::UnmodelledWidth)
    }

    fn x87_store_memory(
        &mut self,
        insn: &Instruction,
        width: u16,
        format: MemFormat,
        pop: bool,
    ) -> Result<(), X87Error> {
        let value = if pop {
            self.x87.pop()
        } else {
            self.x87.read(0)
        };
        let stored = from_slot(value, width, format).ok_or(X87Error::UnmodelledWidth)?;
        let op = insn.operands.first().ok_or(X87Error::MalformedOperands)?;
        if self.write_dst(op, stored, width) {
            Ok(())
        } else {
            Err(X87Error::UnwritableDestination)
        }
    }

    fn x87_arith(
        &mut self,
        insn: &Instruction,
        op: FpArithOp,
        dst: usize,
        src: ArithSrc,
        reversed: bool,
        pop: bool,
    ) -> Result<(), X87Error> {
        let a = self.x87.read(dst);
        let b = match src {
            ArithSrc::Slot(index) => self.x87.read(index),
            ArithSrc::Memory(width) => self.x87_memory_read(insn, width, MemFormat::Float)?,
        };
        let (lhs, rhs) = if reversed { (b, a) } else { (a, b) };
        self.x87
            .write(dst, fp_lane_result(op, lhs, rhs, X87_SLOT_BITS));
        if pop {
            self.x87.drop_top();
        }
        Ok(())
    }

    fn x87_unary(&mut self, op: UnaryOp) -> Result<(), X87Error> {
        let top = self.x87.read(0);
        let result = match op {
            UnaryOp::Abs => Expr::bv_and(top, Expr::konst(X87_MAGNITUDE_MASK, X87_SLOT_BITS)),
            UnaryOp::Chs => Expr::bv_xor(top, Expr::konst(X87_SIGN_BIT, X87_SLOT_BITS)),
            UnaryOp::Sqrt => {
                let (ebits, sbits) =
                    fp_sort_bits_checked(X87_SLOT_BITS).ok_or(X87Error::UnmodelledWidth)?;
                Expr::fp_to_ieee_bv(Expr::fsqrt(Expr::bv_to_fp(top, ebits, sbits), X87_ROUNDING))
            }
        };
        self.x87.write(0, result);
        Ok(())
    }

    /// Record an x87 instruction the lifter could not lower, and forget
    /// the modelled stack.
    ///
    /// The havoc is the load-bearing half. `IrStmt::Unsupported`
    /// lowers to nothing at all downstream, so a declined push would
    /// otherwise leave the slot the hardware wrote holding this
    /// analysis's stale idea of it, and a later `fstp` would store the
    /// wrong value. Dropping every slot makes the subsequent reads free
    /// symbolic inputs, which can only widen a verdict.
    fn decline_x87(&mut self, insn: &Instruction, reason: &str) {
        self.x87.havoc();
        self.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("{reason} at {addr}", addr = insn.address),
        });
    }
}
