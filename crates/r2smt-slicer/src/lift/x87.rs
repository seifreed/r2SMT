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
//! widens exactly, `fild` and the integer-operand arithmetic convert
//! their `m16int` / `m32int` operand exactly into it too — the 64-bit
//! significand holds either without loss — arithmetic runs at that
//! significand, and
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
//! models. `fldcw` reprograms both, so it is in `writes_fp_control` and
//! truncates any slice whose result depends on either.
//!
//! One member of the family escapes that: `fisttp` carries
//! round-toward-zero in its opcode rather than reading the control word,
//! so it pins nothing and its chains survive an `fldcw` that truncates
//! the `fistp` chain beside it. That asymmetry is the whole difference
//! between the two mnemonics in this model, and it is pinned from both
//! sides.
//!
//! ## Compares, and the two idioms
//!
//! x87 has two ways to report a comparison, and both are lifted.
//!
//! `fcomi` / `fucomi` write ZF, PF and CF directly, in the same
//! encoding the SSE scalar compares use, so the existing `jcc` lowering
//! reads them with no change at all.
//!
//! The older `fcom` family writes the status word instead, which is
//! modelled as a second pseudo-register [`X87_STATUS_REGISTER`] with C0
//! at bit 8, C2 at bit 10 and C3 at bit 14. Everything else in that
//! word is free — the exception flags are sticky state this model does
//! not carry, and TOP is the stack pointer the lift-time stack
//! deliberately does not number.
//!
//! Both endings of the classic idiom then fall out of those positions
//! rather than needing code of their own. `fnstsw ax` copies the word
//! into AX, so status bits 15..8 land in AH and put C0 at AH bit 0, C2
//! at bit 2 and C3 at bit 6. Those are exactly the CF, PF and ZF
//! positions `sahf` transfers — and exactly the bits `test ah, 0x41`
//! masks, which needs nothing from this module at all: `test` and `ah`
//! were already modelled, so once `fnstsw` has written `ax` the chain
//! resolves through the ordinary integer path.
//!
//! `fcmov<cc>` reads the same EFLAGS bits back and selects between two
//! slots. It is the one x87 shape that *reads* the flags rather than
//! writing them, which the effect table has to know or the slicer steps
//! over whatever defined the condition and the select runs on a free
//! bit. The lowering is an `Ite`, so the slice stays linear.
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

use r2smt_common::Arch;
use r2smt_ir::expr::{Expr, RoundingMode, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;

use crate::effect::memory_operand_width;

use super::{LiftCtx, fp_sort_bits_checked, x86_memory_modellable};

/// The single canonical data-flow node the x87 register stack collapses
/// onto in the effect tables. Recovered from the disassembler spelling
/// `st(0)`..`st(7)` by [`crate::effect::registers_in_operand`], which
/// splits on non-alphanumerics and so yields the bare token `st`.
pub(crate) const X87_STACK_REGISTER: &str = "st";

/// The x87 status word, as a data-flow node of its own. The `fcom`
/// family writes it and `fnstsw` reads it; no instruction names it as
/// an operand, so unlike `st` it is never recovered from operand text.
pub(crate) const X87_STATUS_REGISTER: &str = "fsw";

/// Width of the status word.
const X87_STATUS_BITS: u16 = 16;

/// Status-word bit positions of the condition codes (Intel SDM Vol. 1
/// §8.1.3). Their spacing is the whole reason the classic idiom works:
/// `fnstsw ax` puts bits 15..8 in AH, so C0 lands at AH bit 0, C2 at
/// bit 2 and C3 at bit 6 — exactly the CF, PF and ZF positions `sahf`
/// transfers.
const X87_C0_BIT: u16 = 8;
/// C2 — the unordered indicator after a compare. C1, the bit between
/// them, signals a stack fault, which the bounded stack model declines
/// on rather than records, so it is written zero.
const X87_C2_BIT: u16 = 10;
/// C3 — the equality indicator after a compare.
const X87_C3_BIT: u16 = 14;

/// Mnemonic prefix of the conditional-move family, whose suffix names
/// an EFLAGS condition rather than being part of a fixed mnemonic.
const X87_CONDITIONAL_MOVE_PREFIX: &str = "fcmov";

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

/// Memory widths the integer-operand arithmetic family accepts.
///
/// Deliberately *not* [`X87_INT_WIDTHS`]: `FIADD` and its siblings are
/// encoded by opcode `DE` (word) and `DA` (dword) only, and the SDM
/// gives them no `m64int` form at all. Reusing the load table would
/// accept a qword operand no encoding can produce.
const X87_INT_ARITH_WIDTHS: [u16; 2] = [16, 32];

/// Memory widths the *non-popping* `fist` accepts. The SDM gives it
/// `m16int` and `m32int` only; the 64-bit integer store exists solely
/// in the popping form.
const X87_INT_STORE_WIDTHS: [u16; 2] = [16, 32];

/// Rounding mode every x87 conversion and arithmetic operation assumes:
/// the control word's reset value, round to nearest with ties to even
/// (Intel SDM Vol. 1 §8.1.5.3).
const X87_ROUNDING: RoundingMode = RoundingMode::NearestTiesEven;

/// Rounding `fisttp` performs regardless of the control word: the SSE3
/// store-and-pop family encodes round-toward-zero in the opcode, which
/// is exactly what makes its chains survive an `fldcw` where an `fistp`
/// chain truncates.
const X87_TRUNCATING: RoundingMode = RoundingMode::TowardZero;

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
        self.fresh(X87_SLOT_BITS)
    }

    /// A fresh free status word, for the bits of it a compare does not
    /// define. Shares the counter with the slot inputs so no two free
    /// values ever collide.
    fn fresh_status(&mut self) -> Expr {
        self.fresh(X87_STATUS_BITS)
    }

    fn fresh(&mut self, bits: u16) -> Expr {
        let value = Expr::var(format!("x87_in_{n}", n = self.free_inputs), bits);
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
    /// A memory operand, in the width and value format the encoding
    /// names. The format is carried rather than assumed because the
    /// same operand shape spells both families: `fadd dword [eax]`
    /// reads an `m32fp` and `fiadd dword [eax]` an `m32int`, and the
    /// conversion into a slot differs accordingly.
    Memory { width: u16, format: MemFormat },
}

/// The three IEEE predicates every x87 compare records, whichever
/// destination it records them in.
struct Compare {
    unordered: Expr,
    equal: Expr,
    less: Expr,
}

/// The condition an `FCMOV<cc>` tests, as the SDM states it: over the
/// EFLAGS bits an integer `cmp` or the `fcomi` family leaves behind,
/// never over the x87 status word.
///
/// The suffixes are the integer `cmov` ones with `u` / `nu` in place of
/// `p` / `np`, because after a floating-point compare parity *is* the
/// unordered indicator. That renaming is the only reason this cannot
/// reuse [`crate::condition::BranchCondition::from_suffix`].
#[derive(Clone, Copy)]
enum X87Condition {
    /// `fcmovb` — below, `CF = 1`.
    Below,
    /// `fcmove` — equal, `ZF = 1`.
    Equal,
    /// `fcmovbe` — below or equal, `CF = 1 or ZF = 1`.
    BelowOrEqual,
    /// `fcmovu` — unordered, `PF = 1`.
    Unordered,
    /// `fcmovnb` — not below, `CF = 0`.
    NotBelow,
    /// `fcmovne` — not equal, `ZF = 0`.
    NotEqual,
    /// `fcmovnbe` — neither below nor equal, `CF = 0 and ZF = 0`.
    NotBelowOrEqual,
    /// `fcmovnu` — not unordered, `PF = 0`.
    NotUnordered,
}

impl X87Condition {
    /// The condition spelled by a `fcmov` mnemonic's suffix.
    fn from_suffix(suffix: &str) -> Option<Self> {
        Some(match suffix {
            "b" => Self::Below,
            "e" => Self::Equal,
            "be" => Self::BelowOrEqual,
            "u" => Self::Unordered,
            "nb" => Self::NotBelow,
            "ne" => Self::NotEqual,
            "nbe" => Self::NotBelowOrEqual,
            "nu" => Self::NotUnordered,
            _ => return None,
        })
    }

    /// The condition as a boolean over the lifter's flag variables.
    fn predicate(self) -> Expr {
        let cf = || Expr::flag("CF");
        let zf = || Expr::flag("ZF");
        let pf = || Expr::flag("PF");
        let set = |flag: Expr| Expr::eq(flag, Expr::konst(1, 1));
        let clear = |flag: Expr| Expr::eq(flag, Expr::konst(0, 1));
        match self {
            Self::Below => set(cf()),
            Self::Equal => set(zf()),
            Self::BelowOrEqual => Expr::bool_or(set(cf()), set(zf())),
            Self::Unordered => set(pf()),
            Self::NotBelow => clear(cf()),
            Self::NotEqual => clear(zf()),
            Self::NotBelowOrEqual => Expr::bool_and(clear(cf()), clear(zf())),
            Self::NotUnordered => clear(pf()),
        }
    }
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
    /// Convert `ST(0)` into `operands[0]` (`fst`/`fstp`, `fist`/`fistp`,
    /// `fisttp`).
    StoreMemory {
        width: u16,
        format: MemFormat,
        pop: bool,
        /// How the conversion rounds. The control word's mode for the
        /// whole family bar `fisttp`, which carries round-toward-zero
        /// in its opcode — the one x87 store whose result does not
        /// depend on the control word at all.
        rounding: RoundingMode,
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
    /// Compare `ST(0)` against a source and record the outcome in the
    /// status word's condition codes (`fcom` family), popping `pops`
    /// slots afterwards.
    CompareStatus { src: ArithSrc, pops: u8 },
    /// Compare `ST(0)` against `ST(src)` and write EFLAGS directly
    /// (`fcomi` family).
    CompareFlags { src: usize, pop: bool },
    /// `ST(0) := ST(src)` when `condition` holds (`fcmov<cc>`).
    ConditionalMove { src: usize, condition: X87Condition },
    /// Copy the status word into `operands[0]` (`fnstsw` / `fstsw`).
    StoreStatus,
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

/// What an x87 instruction touches beyond the registers its operands
/// name, as the effect table needs to see it.
pub(crate) struct X87Effect {
    /// Pseudo-registers the instruction defines.
    pub(crate) defs: Vec<&'static str>,
    /// Pseudo-registers the instruction reads.
    pub(crate) uses: Vec<&'static str>,
    /// Canonical register the instruction writes, if it writes one.
    pub(crate) register_def: Option<&'static str>,
    /// Whether the instruction writes EFLAGS.
    pub(crate) defines_flags: bool,
    /// Whether the instruction reads EFLAGS, and so needs whatever
    /// defined them kept alive in the slice.
    pub(crate) reads_flags: bool,
}

impl X87Effect {
    /// The uniform footprint: the whole data-register file, both
    /// defined and used.
    fn stack() -> Self {
        Self {
            defs: vec![X87_STACK_REGISTER],
            uses: vec![X87_STACK_REGISTER],
            register_def: None,
            defines_flags: false,
            reads_flags: false,
        }
    }
}

/// The pseudo-register footprint of a modelled x87 instruction, or
/// `None` when the instruction is not one.
///
/// Most of the family is uniform — every instruction that touches the
/// data registers both defines and uses the whole stack. Four shapes are
/// not: the `fcom` family also defines the status word, the `fcomi`
/// family writes EFLAGS instead of the status word, the `fcmov` family
/// *reads* EFLAGS, and `fnstsw` reads the status word into a register
/// without touching the data registers at all.
pub(crate) fn x87_effect(insn: &Instruction) -> Option<X87Effect> {
    Some(match classify(insn)? {
        X87Form::CompareStatus { .. } => X87Effect {
            defs: vec![X87_STACK_REGISTER, X87_STATUS_REGISTER],
            ..X87Effect::stack()
        },
        X87Form::CompareFlags { .. } => X87Effect {
            defines_flags: true,
            ..X87Effect::stack()
        },
        // Without this the slicer would step over the `cmp` or `fcomi`
        // that produced the condition, and the `Ite` would select on a
        // free flag.
        X87Form::ConditionalMove { .. } => X87Effect {
            reads_flags: true,
            ..X87Effect::stack()
        },
        X87Form::StoreStatus => X87Effect {
            defs: Vec::new(),
            uses: vec![X87_STATUS_REGISTER],
            register_def: insn
                .operands
                .first()
                .and_then(|op| crate::effect::canonical_register(&op.raw, Arch::X86_64)),
            defines_flags: false,
            reads_flags: false,
        },
        _ => X87Effect::stack(),
    })
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
        "fst" => classify_store(
            ops,
            &StoreSpec::pinned(MemFormat::Float, &X87_FLOAT_SHORT_WIDTHS, false),
        ),
        "fstp" => classify_store(
            ops,
            &StoreSpec::pinned(MemFormat::Float, &X87_FLOAT_WIDTHS, true),
        ),
        "fist" => classify_store(
            ops,
            &StoreSpec::pinned(MemFormat::Integer, &X87_INT_STORE_WIDTHS, false),
        ),
        "fistp" => classify_store(
            ops,
            &StoreSpec::pinned(MemFormat::Integer, &X87_INT_WIDTHS, true),
        ),
        // `fisttp` (SSE3) is `fistp` with the rounding fixed by the
        // opcode instead of read from the control word. It has no
        // non-popping form, and no stack-slot destination — the
        // truncation only makes sense against an integer format.
        "fisttp" => classify_store(
            ops,
            &StoreSpec {
                format: MemFormat::Integer,
                widths: &X87_INT_WIDTHS,
                pop: true,
                rounding: X87_TRUNCATING,
            },
        ),
        // The ordered and unordered compares differ only in which NaN
        // raises the invalid-operation exception, which the value model
        // does not track — so they lift identically, and are told apart
        // only by the operand shapes each encoding allows.
        "fcom" => classify_status_compare(ops, 0, MemorySource::Allowed),
        "fcomp" => classify_status_compare(ops, 1, MemorySource::Allowed),
        "fucom" => classify_status_compare(ops, 0, MemorySource::Forbidden),
        "fucomp" => classify_status_compare(ops, 1, MemorySource::Forbidden),
        "fcompp" | "fucompp" => classify_paired_compare(ops),
        "fcomi" | "fucomi" => classify_flag_compare(ops, false),
        // radare2 spells the popping EFLAGS compares `fcompi` /
        // `fucompi`, not the SDM's `fcomip` / `fucomip`. Both are
        // accepted so the arm does not depend on which spelling a
        // given disassembler build chooses.
        "fcompi" | "fucompi" | "fcomip" | "fucomip" => classify_flag_compare(ops, true),
        "fnstsw" | "fstsw" => classify_store_status(ops),
        other if other.starts_with(X87_CONDITIONAL_MOVE_PREFIX) => other
            .strip_prefix(X87_CONDITIONAL_MOVE_PREFIX)
            .and_then(|suffix| classify_conditional_move(suffix, ops)),
        // The integer-operand family is dispatched *before*
        // [`classify_arith`], which strips a trailing `p` to recognise
        // the popping forms: `ficomp` would otherwise be mis-stripped
        // to `ficom` and read as a non-popping compare.
        "fiadd" | "fisub" | "fisubr" | "fimul" | "fidiv" | "fidivr" | "ficom" | "ficomp" => {
            classify_integer_arith(mnemonic.as_str(), ops)
        }
        other => classify_arith(other, ops),
    }
}

/// The integer-operand arithmetic and compare family (`fiadd`,
/// `fisub`, `ficomp`, …).
///
/// Every member has exactly one memory operand of
/// [`X87_INT_ARITH_WIDTHS`] and takes `ST(0)` as the other input; none
/// has a register-to-register or popping-arithmetic encoding, so the
/// operand shape is fixed and there is no `p` suffix to peel. The
/// conversion into the extended sort is the same one `fild` performs.
fn classify_integer_arith(mnemonic: &str, ops: &[Operand]) -> Option<X87Form> {
    let op = only_operand(ops)?;
    let src = ArithSrc::Memory {
        width: modellable_memory_width(op, &X87_INT_ARITH_WIDTHS)?,
        format: MemFormat::Integer,
    };
    let (arith, reversed) = match mnemonic {
        "ficom" => return Some(X87Form::CompareStatus { src, pops: 0 }),
        "ficomp" => return Some(X87Form::CompareStatus { src, pops: 1 }),
        "fiadd" => (X87Arith::Add, false),
        "fimul" => (X87Arith::Mul, false),
        "fisub" => (X87Arith::Sub, false),
        "fisubr" => (X87Arith::Sub, true),
        "fidiv" => (X87Arith::Div, false),
        "fidivr" => (X87Arith::Div, true),
        _ => return None,
    };
    Some(X87Form::Arith {
        op: arith,
        dst: 0,
        src,
        reversed,
        pop: false,
    })
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

/// The fixed part of a store encoding: the value format the memory
/// destination holds, the widths it accepts, whether it pops afterwards,
/// and how the conversion rounds.
struct StoreSpec {
    format: MemFormat,
    widths: &'static [u16],
    pop: bool,
    rounding: RoundingMode,
}

impl StoreSpec {
    /// The ordinary case: rounding taken from the control word, whose
    /// reset value this module pins.
    const fn pinned(format: MemFormat, widths: &'static [u16], pop: bool) -> Self {
        Self {
            format,
            widths,
            pop,
            rounding: X87_ROUNDING,
        }
    }
}

/// Same asymmetry as [`classify_load`]: `fst`/`fstp` can name a stack
/// slot, `fist`/`fistp` cannot.
fn classify_store(ops: &[Operand], spec: &StoreSpec) -> Option<X87Form> {
    let op = only_operand(ops)?;
    if let Some(index) = slot_index(op) {
        return (spec.format == MemFormat::Float).then_some(X87Form::StoreSlot {
            index,
            pop: spec.pop,
        });
    }
    Some(X87Form::StoreMemory {
        width: modellable_memory_width(op, spec.widths)?,
        format: spec.format,
        pop: spec.pop,
        rounding: spec.rounding,
    })
}

/// Whether a compare encoding has a memory form. `FCOM` / `FCOMP` do;
/// the unordered `FUCOM` / `FUCOMP` are register-only.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemorySource {
    Allowed,
    Forbidden,
}

/// `fcom` / `fcomp` / `fucom` / `fucomp` — compare `ST(0)` against the
/// source. With no operand the source is `ST(1)`.
fn classify_status_compare(ops: &[Operand], pops: u8, memory: MemorySource) -> Option<X87Form> {
    let src = match ops {
        [] => ArithSrc::Slot(1),
        [op] if op.kind == OperandKind::Register => ArithSrc::Slot(slot_index(op)?),
        [op] if memory == MemorySource::Allowed => ArithSrc::Memory {
            width: modellable_memory_width(op, &X87_FLOAT_SHORT_WIDTHS)?,
            format: MemFormat::Float,
        },
        _ => return None,
    };
    Some(X87Form::CompareStatus { src, pops })
}

/// `fcompp` / `fucompp` — compare `ST(0)` against `ST(1)` and pop both.
/// The encoding carries no operand.
fn classify_paired_compare(ops: &[Operand]) -> Option<X87Form> {
    no_operands(
        ops,
        X87Form::CompareStatus {
            src: ArithSrc::Slot(1),
            pops: 2,
        },
    )
}

/// `fcomi` / `fucomi` and their popping forms — register-only, and the
/// destination operand is always `ST(0)`, so a one-operand spelling
/// names the source directly.
fn classify_flag_compare(ops: &[Operand], pop: bool) -> Option<X87Form> {
    let src = match ops {
        [src] => slot_index(src)?,
        [_dst, src] => slot_index(src)?,
        _ => return None,
    };
    Some(X87Form::CompareFlags { src, pop })
}

/// `fcmov<cc> st(0), st(i)` — move `ST(i)` into `ST(0)` when the
/// condition holds, and leave `ST(0)` alone otherwise.
///
/// `db c9` disassembles as `fcmovne st(0), st(1)`: the encoding's
/// destination is always the top of stack, so a spelling naming any
/// other slot is not this instruction and declines.
fn classify_conditional_move(suffix: &str, ops: &[Operand]) -> Option<X87Form> {
    let condition = X87Condition::from_suffix(suffix)?;
    let [dst, src] = ops else { return None };
    (slot_index(dst)? == 0).then_some(X87Form::ConditionalMove {
        src: slot_index(src)?,
        condition,
    })
}

/// `fnstsw` / `fstsw` — the status word into `AX` or a 16-bit memory
/// destination. The two mnemonics differ only in the `FWAIT` prefix,
/// which is exception sequencing rather than value semantics.
fn classify_store_status(ops: &[Operand]) -> Option<X87Form> {
    let op = only_operand(ops)?;
    match op.kind {
        // The encoding has exactly one register form, and it names AX.
        OperandKind::Register => op
            .raw
            .trim()
            .eq_ignore_ascii_case("ax")
            .then_some(X87Form::StoreStatus),
        OperandKind::Memory => {
            modellable_memory_width(op, &[X87_STATUS_BITS]).map(|_| X87Form::StoreStatus)
        }
        _ => None,
    }
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
        // Defensive only: radare2 always prints the destination, so
        // the bare no-operand spelling does not occur in practice.
        [] => pop.then_some(X87Form::Arith {
            op,
            dst: 1,
            src: ArithSrc::Slot(0),
            reversed,
            pop,
        }),
        // One operand, and its kind decides the family.
        //
        // A stack slot means a register-to-register encoding, where
        // the *operand count* names the destination — which is what
        // makes this decidable, contrary to an earlier note here. The
        // one-operand spelling is the only one radare2 emits for the
        // `D8`/`DE` register forms:
        //
        //   d8 c1  fadd  st(1)   `D8 C0+i`  ST(0) := ST(0) + ST(1)
        //   de c1  faddp st(1)   `DE C0+i`  ST(1) := ST(1) + ST(0), pop
        //
        // The `r` suffix keeps its meaning through `reversed`, which
        // matters because Intel swaps the mnemonics between the two
        // opcode groups: `dc f1` is `fdivr st(1), st(0)` while
        // `d8 f1` is `fdiv st(1)`. radare2 reports each faithfully, so
        // no correction belongs here.
        [only] => match slot_index(only) {
            Some(index) if pop => Some(X87Form::Arith {
                op,
                dst: index,
                src: ArithSrc::Slot(0),
                reversed,
                pop,
            }),
            Some(index) => Some(X87Form::Arith {
                op,
                dst: 0,
                src: ArithSrc::Slot(index),
                reversed,
                pop,
            }),
            // `fadd m32fp` — `ST(0) := ST(0) op m`. There is no
            // popping memory encoding.
            None if !pop => Some(X87Form::Arith {
                op,
                dst: 0,
                src: ArithSrc::Memory {
                    width: modellable_memory_width(only, &X87_FLOAT_SHORT_WIDTHS)?,
                    format: MemFormat::Float,
                },
                reversed,
                pop,
            }),
            None => None,
        },
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

/// Convert a slot value into the form a memory destination stores,
/// under the rounding the encoding selects.
///
/// This is where the narrowing rounding happens, once, at the store —
/// exactly where the hardware performs it.
fn from_slot(value: Expr, width: u16, format: MemFormat, rounding: RoundingMode) -> Option<Expr> {
    Some(match format {
        MemFormat::Float if width == X87_MEMORY_BITS => extended_to_memory(&value),
        MemFormat::Float => {
            let (dst_e, dst_s) = fp_sort_bits_checked(width)?;
            from_extended(Expr::fp_to_fp(as_extended(value), rounding, dst_e, dst_s))
        }
        // `fist` / `fistp` round per the control word; `fisttp`
        // truncates by opcode, the way SSE's `cvtt*` family does.
        MemFormat::Integer => Expr::fp_to_sbv(as_extended(value), rounding, width),
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
            X87Form::StoreMemory {
                width,
                format,
                pop,
                rounding,
            } => self.x87_store_memory(insn, width, format, pop, rounding),
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
            X87Form::CompareStatus { src, pops } => self.x87_compare_status(insn, src, pops),
            X87Form::CompareFlags { src, pop } => {
                self.x87_compare_flags(src, pop);
                Ok(())
            }
            X87Form::ConditionalMove { src, condition } => {
                self.x87_conditional_move(src, condition);
                Ok(())
            }
            X87Form::StoreStatus => self.x87_store_status(insn),
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
        rounding: RoundingMode,
    ) -> Result<(), X87Error> {
        let value = if pop {
            self.x87.pop()
        } else {
            self.x87.read(0)
        };
        let stored = from_slot(value, width, format, rounding).ok_or(X87Error::UnmodelledWidth)?;
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
            ArithSrc::Memory { width, format } => self.x87_memory_read(insn, width, format)?,
        };
        let (lhs, rhs) = if reversed { (b, a) } else { (a, b) };
        self.x87.write(dst, extended_arith(op, lhs, rhs));
        if pop {
            self.x87.drop_top();
        }
        Ok(())
    }

    /// The three predicates every x87 compare is built from, over
    /// `ST(0)` and a source, in that order.
    ///
    /// Identical to the SSE scalar compare's, and for the same reason:
    /// the comparison itself is IEEE, and what differs between the two
    /// instruction families is only where the answer is written.
    fn x87_compare_predicates(
        &mut self,
        insn: &Instruction,
        src: ArithSrc,
    ) -> Result<Compare, X87Error> {
        let a = as_extended(self.x87.read(0));
        let b = as_extended(match src {
            ArithSrc::Slot(index) => self.x87.read(index),
            ArithSrc::Memory { width, format } => self.x87_memory_read(insn, width, format)?,
        });
        Ok(Compare {
            unordered: Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone())),
            equal: Expr::feq(a.clone(), b.clone()),
            less: Expr::flt(a, b),
        })
    }

    /// `fcom` / `fcomp` / `fcompp` and their unordered forms — the
    /// outcome goes into the status word's condition codes.
    ///
    /// SDM Vol. 2, "FCOM": greater clears C3/C2/C0, less sets C0 alone,
    /// equal sets C3 alone, and unordered sets all three. C1 stays zero:
    /// after a compare it signals a stack fault, and an empty stack is
    /// already a decline here.
    fn x87_compare_status(
        &mut self,
        insn: &Instruction,
        src: ArithSrc,
        pops: u8,
    ) -> Result<(), X87Error> {
        let cmp = self.x87_compare_predicates(insn, src)?;
        let c0 = Expr::bool_or(cmp.unordered.clone(), cmp.less);
        let c2 = cmp.unordered.clone();
        let c3 = Expr::bool_or(cmp.unordered, cmp.equal);
        let status = self.x87_status_word(c0, c2, c3);
        self.assign(Var::new(X87_STATUS_REGISTER, X87_STATUS_BITS), status);
        for _ in 0..pops {
            self.x87.drop_top();
        }
        Ok(())
    }

    /// Assemble the status word around the three condition codes a
    /// compare defines.
    ///
    /// Everything else in the word is left *free*: the exception flags
    /// are sticky state this model does not carry, and TOP is the stack
    /// pointer, which the lift-time stack deliberately does not number.
    /// Writing zeros there would be a definite wrong answer for any
    /// program that read them — and the idioms that matter
    /// (`test ah, 0x41`, `sahf`) mask or ignore every free bit.
    fn x87_status_word(&mut self, c0: Expr, c2: Expr, c3: Expr) -> Expr {
        let free = self.x87.fresh_status();
        let bit = |from: &Expr, index: u16| Expr::extract(from.clone(), index, index);
        let span = |from: &Expr, hi: u16, lo: u16| Expr::extract(from.clone(), hi, lo);
        // Assembled most-significant first: B, C3, TOP, C2, C1, C0, and
        // the exception-flag byte.
        Expr::concat(
            bit(&free, X87_STATUS_BITS - 1),
            Expr::concat(
                c3,
                Expr::concat(
                    span(&free, X87_C3_BIT - 1, X87_C2_BIT + 1),
                    Expr::concat(
                        c2,
                        Expr::concat(
                            Expr::konst(0, 1),
                            Expr::concat(c0, span(&free, X87_C0_BIT - 1, 0)),
                        ),
                    ),
                ),
            ),
        )
    }

    /// `fcomi` / `fucomi` and their popping forms — the outcome goes
    /// straight into EFLAGS, in the same encoding the SSE scalar
    /// compares use, which is what lets the existing `jcc` lowering read
    /// it unchanged. OF and SF are cleared per the SDM.
    fn x87_compare_flags(&mut self, src: usize, pop: bool) {
        let a = as_extended(self.x87.read(0));
        let b = as_extended(self.x87.read(src));
        let unordered = Expr::bool_or(Expr::fisnan(a.clone()), Expr::fisnan(b.clone()));
        self.set_flag("PF", unordered.clone());
        self.set_flag(
            "ZF",
            Expr::bool_or(unordered.clone(), Expr::feq(a.clone(), b.clone())),
        );
        self.set_flag("CF", Expr::bool_or(unordered, Expr::flt(a, b)));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("SF", Expr::konst(0, 1));
        if pop {
            self.x87.drop_top();
        }
    }

    /// `fnstsw` / `fstsw` — copy the status word into AX or a 16-bit
    /// memory destination.
    fn x87_store_status(&mut self, insn: &Instruction) -> Result<(), X87Error> {
        let op = insn.operands.first().ok_or(X87Error::MalformedOperands)?;
        let status = Expr::var(X87_STATUS_REGISTER, X87_STATUS_BITS);
        if self.write_dst(op, status, X87_STATUS_BITS) {
            Ok(())
        } else {
            Err(X87Error::UnwritableDestination)
        }
    }

    /// `fcmov<cc>` — select between two slot values on an EFLAGS
    /// predicate.
    ///
    /// A select, not a branch: both slots are read unconditionally and
    /// an `Ite` picks between them, so the slice stays linear and the
    /// SSA pass and the encoder need nothing new. Total, because both
    /// reads are of the modelled stack and neither can present an
    /// unmodelled width.
    fn x87_conditional_move(&mut self, src: usize, condition: X87Condition) {
        let taken = self.x87.read(src);
        let kept = self.x87.read(0);
        self.x87.write(
            0,
            Expr::Ite {
                cond: Box::new(condition.predicate()),
                then_expr: Box::new(taken),
                else_expr: Box::new(kept),
            },
        );
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
