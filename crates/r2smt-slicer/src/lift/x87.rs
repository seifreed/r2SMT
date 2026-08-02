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
//! Slots hold the **IEEE bit pattern of the double-extended sort**,
//! `FloatingPoint(15, 64)`, which is what the hardware actually
//! computes in. Keeping them bit patterns rather than floats preserves
//! the pipeline-wide invariant that a stored value is bit-vector-typed;
//! floats appear only inside an expression, wrapped back up by
//! `Expr::FpToIeeeBv` before they reach a slot.
//!
//! That makes every rounding this module performs the same rounding the
//! hardware performs, in the same place. `fld` of `m32fp` / `m64fp`
//! widens exactly, arithmetic runs at the 64-bit significand, and
//! `fstp` to a narrower format rounds exactly once — at the store,
//! where the hardware rounds. The predecessor of this model computed at
//! binary64 and could differ from the hardware by one ulp on a `m64fp`
//! store, because 64 intermediate bits do not clear the 108 that a
//! binary64 target needs for double rounding to be innocuous. A
//! differing definite value is a fabricated verdict, not a widening,
//! which is why this had to land before any x87 compare does.
//!
//! ### The 79-vs-80 bit bridge
//!
//! `ebits + sbits` is 79, but the stored image (`m80fp`, spelled
//! `tbyte`) is 80 bits: x87 makes the significand's leading bit
//! explicit at position 63, where the SMT sort leaves it implicit.
//! Every width computation in the encoder and the renderer derives the
//! pattern width as `ebits + sbits`, and all of them are *right* — 79
//! is the width of this sort's pattern. So the bridge lives here,
//! entirely outside them: [`extended_from_memory`] strips the integer
//! bit on a load and [`extended_to_memory`] reinserts it on a store,
//! and no `Expr::FpToIeeeBv` in the IR is ever 80 bits wide.
//!
//! The strip is only faithful when the explicit bit carries what IEEE
//! would have inferred — zero exactly when the exponent field is zero.
//! The other two combinations are the double-extended encodings the SDM
//! calls unsupported; naming them by their canonical counterpart would
//! be a different number, so they resolve to a free symbolic value
//! instead.
//!
//! Because `(15, 64)` is outside `is_renderable_fp_sort`, cvc5 and
//! Bitwuzla decline any slice carrying an x87 value and x87 is
//! effectively **Z3-only**. That decline is the text backends' own,
//! and each pins it with a contract test.
//!
//! The rounding mode is pinned to the control word's reset value (round
//! to nearest, ties to even) and the precision-control field is assumed
//! to hold its reset value too — extended, which is what this sort
//! models. There is no guard yet on `fldcw`; that, and the compare /
//! status-word idiom (`fcom` / `fnstsw` / `fcomi`), are follow-up work.
//! Every mnemonic they involve is outside [`classify`], so today they
//! truncate the slice.
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

use super::{LiftCtx, fp_sort_bits_checked, x86_memory_modellable};

/// The single canonical data-flow node the x87 register stack collapses
/// onto in the effect tables. Recovered from the disassembler spelling
/// `st(0)`..`st(7)` by [`crate::effect::registers_in_operand`], which
/// splits on non-alphanumerics and so yields the bare token `st`.
pub(crate) const X87_STACK_REGISTER: &str = "st";

/// Number of x87 data registers (Intel SDM Vol. 1 §8.1.2).
const X87_STACK_DEPTH: usize = 8;

/// Exponent width of the x87 double-extended sort.
const X87_EBITS: u16 = 15;

/// Significand width of the x87 double-extended sort, counted the
/// SMT-LIB way — including the leading bit, which the sort keeps
/// implicit and the stored image makes explicit.
const X87_SBITS: u16 = 64;

/// Bits of the significand that appear in either bit pattern.
const X87_FRACTION_BITS: u16 = X87_SBITS - 1;

/// Width of the sort's IEEE bit pattern: sign, exponent, fraction.
/// Equals `X87_EBITS + X87_SBITS`, which is what every width
/// computation downstream derives, and it is correct there.
const X87_SLOT_BITS: u16 = 1 + X87_EBITS + X87_FRACTION_BITS;

/// Width of the stored double-extended image (`m80fp` / `tbyte`), one
/// bit wider because it carries the explicit integer bit.
const X87_MEMORY_BITS: u16 = X87_SLOT_BITS + 1;

/// Position of the explicit integer bit in the stored image: directly
/// above the fraction.
const X87_INTEGER_BIT: u16 = X87_FRACTION_BITS;

/// `+1.0` in the sort's bit pattern: exponent at the bias
/// (`2^(ebits-1) - 1` = 16383), zero fraction, positive sign.
const X87_ONE: u128 = 0x1fff_8000_0000_0000_0000;

/// `+0.0` in the sort's bit pattern — what `fldz` pushes.
const X87_ZERO: u128 = 0;

/// Sign bit of the sort's bit pattern, flipped by `fchs`.
const X87_SIGN_BIT: u128 = 1 << (X87_SLOT_BITS - 1);

/// Every bit of the sort's pattern but the sign, which `fabs` keeps.
/// Clearing the sign bit is exact for every value including the NaNs
/// and infinities, so it needs no float sort at all.
const X87_MAGNITUDE_MASK: u128 = X87_SIGN_BIT - 1;

/// Memory widths `fld` and `fstp` accept: `m32fp`, `m64fp`, and the
/// double-extended `m80fp`.
const X87_FLOAT_WIDTHS: [u16; 3] = [32, 64, X87_MEMORY_BITS];

/// Memory widths the non-popping `fst` and the arithmetic family
/// accept. The SDM gives the double-extended form to `fld` and `fstp`
/// alone: there is no `m80fp` arithmetic form, and no way to store one
/// without popping.
const X87_FLOAT_SHORT_WIDTHS: [u16; 2] = [32, 64];

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

/// The arithmetic x87 performs on its stack. Narrower than the SSE
/// [`super::FpArithOp`] on purpose: x87 has no `max` / `min`, so there
/// is no operand-selecting case and every result is a genuine rounded
/// computation at the extended sort.
#[derive(Clone, Copy)]
enum X87Arith {
    Add,
    Sub,
    Mul,
    Div,
}

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
    /// Push a fixed double-extended bit pattern (`fld1`, `fldz`).
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
        op: X87Arith,
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
        "fst" => classify_store(ops, MemFormat::Float, &X87_FLOAT_SHORT_WIDTHS, false),
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
        "fadd" => (X87Arith::Add, false),
        "fmul" => (X87Arith::Mul, false),
        "fsub" => (X87Arith::Sub, false),
        "fsubr" => (X87Arith::Sub, true),
        "fdiv" => (X87Arith::Div, false),
        "fdivr" => (X87Arith::Div, true),
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
            src: ArithSrc::Memory(modellable_memory_width(mem, &X87_FLOAT_SHORT_WIDTHS)?),
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

/// Read a slot value as a double-extended float.
fn as_extended(pattern: Expr) -> Expr {
    Expr::bv_to_fp(pattern, X87_EBITS, X87_SBITS)
}

/// Write a double-extended float back as a slot value.
fn from_extended(value: Expr) -> Expr {
    Expr::fp_to_ieee_bv(value)
}

/// `ST(dst) := ST(dst) op src` at the extended sort — the one place x87
/// arithmetic rounds, and it rounds exactly where the hardware does.
fn extended_arith(op: X87Arith, lhs: Expr, rhs: Expr) -> Expr {
    let (a, b) = (as_extended(lhs), as_extended(rhs));
    from_extended(match op {
        X87Arith::Add => Expr::fadd(a, b, X87_ROUNDING),
        X87Arith::Sub => Expr::fsub(a, b, X87_ROUNDING),
        X87Arith::Mul => Expr::fmul(a, b, X87_ROUNDING),
        X87Arith::Div => Expr::fdiv(a, b, X87_ROUNDING),
    })
}

/// The sort's 79-bit pattern of a value held in its 80-bit stored
/// image, or `free` when the image is one of the double-extended
/// encodings the SDM calls unsupported.
///
/// The stored image makes the significand's leading bit explicit at
/// [`X87_INTEGER_BIT`]; the sort leaves it implicit, so the bridge drops
/// it. That is faithful exactly when the bit carries what IEEE would
/// have inferred — zero when the exponent field is zero, one otherwise.
/// The other two combinations are the unnormals, pseudo-denormals,
/// pseudo-infinities and pseudo-NaNs, whose value is *not* the one the
/// canonical pattern names. Dropping their integer bit would hand the
/// solver a different definite number, which fabricates rather than
/// widens, so they resolve to a free symbolic value instead.
fn extended_from_memory(image: &Expr, free: Expr) -> Expr {
    let sign = Expr::extract(image.clone(), X87_MEMORY_BITS - 1, X87_MEMORY_BITS - 1);
    let exponent = Expr::extract(image.clone(), X87_MEMORY_BITS - 2, X87_INTEGER_BIT + 1);
    let integer = Expr::extract(image.clone(), X87_INTEGER_BIT, X87_INTEGER_BIT);
    let fraction = Expr::extract(image.clone(), X87_FRACTION_BITS - 1, 0);
    let zero_exponent = Expr::eq(exponent.clone(), Expr::konst(0, X87_EBITS));
    let leading = Expr::eq(integer, Expr::konst(1, 1));
    let canonical = Expr::bool_or(
        Expr::bool_and(zero_exponent.clone(), Expr::bool_not(leading.clone())),
        Expr::bool_and(Expr::bool_not(zero_exponent), leading),
    );
    Expr::Ite {
        cond: Box::new(canonical),
        then_expr: Box::new(Expr::concat(sign, Expr::concat(exponent, fraction))),
        else_expr: Box::new(free),
    }
}

/// The 80-bit stored image of a slot value: the sort's pattern with the
/// implicit leading significand bit made explicit at
/// [`X87_INTEGER_BIT`]. Always canonical by construction, which is what
/// makes the inverse bridge exact for anything this module stored.
fn extended_to_memory(pattern: &Expr) -> Expr {
    let sign = Expr::extract(pattern.clone(), X87_SLOT_BITS - 1, X87_SLOT_BITS - 1);
    let exponent = Expr::extract(pattern.clone(), X87_SLOT_BITS - 2, X87_FRACTION_BITS);
    let fraction = Expr::extract(pattern.clone(), X87_FRACTION_BITS - 1, 0);
    let integer = Expr::Ite {
        cond: Box::new(Expr::eq(exponent.clone(), Expr::konst(0, X87_EBITS))),
        then_expr: Box::new(Expr::konst(0, 1)),
        else_expr: Box::new(Expr::konst(1, 1)),
    };
    Expr::concat(
        sign,
        Expr::concat(exponent, Expr::concat(integer, fraction)),
    )
}

/// Convert a value read from memory into a slot value.
///
/// Every widening here is exact: binary32 and binary64 both nest inside
/// the extended sort, and a 64-bit integer fits its 64-bit significand
/// exactly. The rounding mode is still threaded through because the IR
/// node carries one, not because any of these conversions round.
///
/// The double-extended case is absent: it needs a free symbolic value
/// for the non-canonical encodings and so lives on [`LiftCtx`].
fn to_slot(raw: Expr, width: u16, format: MemFormat) -> Option<Expr> {
    Some(match format {
        MemFormat::Float => {
            let (src_e, src_s) = fp_sort_bits_checked(width)?;
            from_extended(Expr::fp_to_fp(
                Expr::bv_to_fp(raw, src_e, src_s),
                X87_ROUNDING,
                X87_EBITS,
                X87_SBITS,
            ))
        }
        MemFormat::Integer => {
            from_extended(Expr::sbv_to_fp(raw, X87_ROUNDING, X87_EBITS, X87_SBITS))
        }
    })
}

/// Convert a slot value into the form a memory destination stores.
///
/// This is where the narrowing rounding happens, once, at the store —
/// exactly where the hardware performs it.
fn from_slot(value: Expr, width: u16, format: MemFormat) -> Option<Expr> {
    Some(match format {
        MemFormat::Float if width == X87_MEMORY_BITS => extended_to_memory(&value),
        MemFormat::Float => {
            let (dst_e, dst_s) = fp_sort_bits_checked(width)?;
            from_extended(Expr::fp_to_fp(
                as_extended(value),
                X87_ROUNDING,
                dst_e,
                dst_s,
            ))
        }
        // `fist` / `fistp` round per the control word, unlike SSE's
        // `cvtt*` family which truncates by opcode.
        MemFormat::Integer => Expr::fp_to_sbv(as_extended(value), X87_ROUNDING, width),
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
            X87Form::Unary(op) => {
                self.x87_unary(op);
                Ok(())
            }
        }
    }

    /// Load `operands[0]` through the byte-granular memory model and
    /// convert it to a slot value.
    ///
    /// The double-extended case takes the bridge rather than a float
    /// conversion, and needs a fresh free value for the encodings the
    /// bridge refuses to name — which is why this is a method and
    /// [`to_slot`] is not.
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
        if format == MemFormat::Float && width == X87_MEMORY_BITS {
            let free = self.x87.fresh_input();
            return Ok(extended_from_memory(&raw, free));
        }
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
        op: X87Arith,
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
        self.x87.write(dst, extended_arith(op, lhs, rhs));
        if pop {
            self.x87.drop_top();
        }
        Ok(())
    }

    /// The in-place unaries. Total rather than fallible: `fabs` and
    /// `fchs` are bit manipulations and `fsqrt` runs at the sort the
    /// slot already holds, so none of them can present an unmodelled
    /// width.
    fn x87_unary(&mut self, op: UnaryOp) {
        let top = self.x87.read(0);
        let result = match op {
            UnaryOp::Abs => Expr::bv_and(top, Expr::konst(X87_MAGNITUDE_MASK, X87_SLOT_BITS)),
            UnaryOp::Chs => Expr::bv_xor(top, Expr::konst(X87_SIGN_BIT, X87_SLOT_BITS)),
            UnaryOp::Sqrt => from_extended(Expr::fsqrt(as_extended(top), X87_ROUNDING)),
        };
        self.x87.write(0, result);
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
