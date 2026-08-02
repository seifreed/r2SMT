//! Lift a [`Slice`] of x86 / `x86_64` instructions into a list of
//! [`IrStmt`]s plus a symbolic [`Expr`] for the branch's condition.
//!
//! The lifter is intentionally narrow: it handles the mnemonic set
//! produced by the Phase 3 slicer (`mov`, `xor` (zero idiom),
//! `and / or / add / sub / imul`, `cmp`, `test`, `shl / sal / shr / sar`,
//! `lea`). For each instruction it emits one or more [`IrStmt::Assign`]
//! statements that capture the data-flow effect plus, when relevant, the
//! flag updates that the SMT backend reads through
//! [`BranchCandidate`].
//!
//! Flags are modelled as 1-bit [`r2smt_ir::Var`]s named `ZF`, `CF`,
//! `SF`, `OF`, `PF`. Flags we cannot model precisely yet (`OF`, `PF`,
//! `AF`) are set to [`Expr::Unknown`] so the SMT translator can treat
//! them as free symbolic inputs without producing wrong answers.
//!
//! Sub-register precision: register reads and writes consult
//! [`crate::registers::register_layout`] so that operands like `al`,
//! `ah`, `ax`, `eax` are modelled as bit-slices of the canonical
//! 64-bit parent (`rax`) via [`Expr::Extract`] / [`Expr::Concat`] /
//! [`Expr::ZeroExtend`]. The lifter still tracks the parent as the
//! single SSA-renamed variable; bit-precise rewrites live inside the
//! right-hand side of the assignment.

use r2smt_common::{Address, Arch};
use r2smt_ir::expr::{Expr, RoundingMode, Var};
use r2smt_ir::program::{Instruction, Operand, OperandKind};
use r2smt_ir::stmt::IrStmt;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::collector::BranchCandidate;
use crate::condition::BranchCondition;
use crate::effect::{memory_operand_width, stack_slot};
use crate::registers::{RegisterLayout, is_simd_parent, register_layout, simd_parent_bits};
use crate::slice::{Slice, SliceMerge, SliceStatus};

mod aarch32;
mod aarch64;
mod merge;
mod x86;
mod x87;
pub(crate) use aarch32::{VfpOp, vfp_scalar};
use merge::lower_merge;
pub(crate) use x86::{is_fp_compare_mnemonic, sse_scalar_move_lane};
use x87::X87Stack;
pub(crate) use x87::{X87Effect, is_modelled_x87, x87_effect};

/// IR representation of a [`Slice`] plus the branch's symbolic
/// condition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiftedSlice {
    /// The branch the slice belongs to.
    pub branch: BranchCandidate,
    /// IR statements produced by lifting `slice.instructions`, in
    /// execution order.
    pub statements: Vec<IrStmt>,
    /// Symbolic condition the branch evaluates over the flags emitted
    /// by `statements`.
    pub condition: Expr,
    /// Status forwarded from the slice.
    pub status: SliceStatus,
    /// Forwarded from [`Slice::treat_truncation_as_inputs`]: when set,
    /// downstream stages treat a truncated slice as if it were
    /// complete, with the remaining roots becoming free symbolic
    /// inputs at the SSA layer.
    #[serde(default)]
    pub treat_truncation_as_inputs: bool,
    /// Architecture the slice was lifted under. Surfaced downstream
    /// (SSA, pretty-printer, report) so register-name resolution can
    /// pick the correct ISA table without re-deriving it.
    #[serde(default = "default_arch")]
    pub arch: Arch,
}

fn default_arch() -> Arch {
    Arch::X86_64
}

const FLAGS: &[&str] = &["ZF", "CF", "SF", "OF", "PF"];

/// SSE / AVX / x87 mnemonics whose per-mnemonic handler must win over
/// the ESIL and P-code ladders, which model an `xmm` operand at pointer
/// width instead of canonicalising it to its vector parent.
///
/// Most of these define the vector register; the scalar compares
/// (`comiss` and friends) define only flags, but need the same override
/// for the same reason — their operands are vector-register lanes.
/// Kept in sync with the dispatch in `effect/x86.rs`.
///
/// Takes the whole instruction rather than the mnemonic because one
/// member of the set cannot be recognised from its mnemonic alone: see
/// [`x86::sse_scalar_move_lane`].
///
/// Arch-dispatched so the pre-empt can be extended to ARM alongside its
/// vector handlers. It has to be: ESIL describes several ARM FP moves
/// with tokens the mini stack machine *does* support (`fmov s0, s1`
/// lowers as `"s1,s0,="`), so without an override those would lift into
/// a differently-named, differently-sized node than the vector lifter
/// uses, splitting the data-flow graph in a way no unit test would
/// notice.
///
/// x87 is in the same boat for a sharper reason. ESIL describes `fld1`
/// as eight register moves plus a 64-bit immediate — `st6,st7,=,…,
/// 0x3ff0000000000000,st0,=` — every token of which the mini stack
/// machine evaluates happily, producing eight `st<n>` variables at
/// pointer width that no x87 handler will ever read. `fldz`, `fld
/// st(i)` and `fxch` are the same shape. The x87 handlers do not model
/// the stack as registers at all (see `lift/x87.rs`), so without this
/// pre-empt the two models would silently coexist in one slice.
fn is_simd_instruction(insn: &Instruction, arch: Arch) -> bool {
    match arch {
        Arch::X86 | Arch::X86_64 => is_x86_simd_instruction(insn) || is_modelled_x87(insn),
        Arch::Aarch64 => is_aarch64_fp_instruction(insn),
        Arch::Arm => vfp_scalar(&insn.mnemonic.trim().to_ascii_lowercase()).is_some(),
        _ => false,
    }
}

/// `AArch64` scalar floating-point mnemonics whose per-mnemonic handler
/// must win over the ESIL and P-code ladders.
///
/// `fmov s0, s1` is the reason this cannot wait: ESIL describes it as
/// `"s1,s0,="`, which the mini stack machine evaluates happily, into a
/// register node named and sized unlike anything the vector lifter
/// produces.
fn is_aarch64_fp_instruction(insn: &Instruction) -> bool {
    matches!(
        insn.mnemonic.trim().to_ascii_lowercase().as_str(),
        "fadd"
            | "fsub"
            | "fmul"
            | "fdiv"
            | "fmax"
            | "fmin"
            | "fsqrt"
            | "fabs"
            | "fneg"
            | "fmov"
            | "fcmp"
            | "fcmpe"
            | "fcvt"
            | "scvtf"
            | "ucvtf"
            | "fcvtzs"
            | "fcvtzu"
    )
}

fn is_x86_simd_instruction(insn: &Instruction) -> bool {
    // The compare family is 64 mnemonics once the eight predicates are
    // crossed with ps/pd/ss/sd and the VEX prefix, so it is recognised
    // structurally rather than listed.
    if x86::is_fp_compare_mnemonic(&insn.mnemonic) || x86::sse_scalar_move_lane(insn).is_some() {
        return true;
    }
    matches!(
        insn.mnemonic.trim().to_ascii_lowercase().as_str(),
        "movaps"
            | "movups"
            | "movapd"
            | "movupd"
            | "movdqa"
            | "movdqu"
            | "vmovaps"
            | "vmovups"
            | "vmovapd"
            | "vmovupd"
            | "vmovdqa"
            | "vmovdqu"
            | "pxor"
            | "vpxor"
            | "pand"
            | "vpand"
            | "por"
            | "vpor"
            | "pandn"
            | "vpandn"
            | "addss"
            | "subss"
            | "mulss"
            | "divss"
            | "addsd"
            | "subsd"
            | "mulsd"
            | "divsd"
            | "addps"
            | "subps"
            | "mulps"
            | "divps"
            | "vaddps"
            | "vsubps"
            | "vmulps"
            | "vdivps"
            | "addpd"
            | "subpd"
            | "mulpd"
            | "divpd"
            | "vaddpd"
            | "vsubpd"
            | "vmulpd"
            | "vdivpd"
            | "maxps"
            | "minps"
            | "maxpd"
            | "minpd"
            | "vmaxps"
            | "vminps"
            | "vmaxpd"
            | "vminpd"
            | "maxss"
            | "minss"
            | "maxsd"
            | "minsd"
            | "sqrtps"
            | "sqrtpd"
            | "vsqrtps"
            | "vsqrtpd"
            | "sqrtss"
            | "sqrtsd"
            | "comiss"
            | "ucomiss"
            | "comisd"
            | "ucomisd"
            | "cvtsi2ss"
            | "cvtsi2sd"
            | "cvtss2si"
            | "cvtsd2si"
            | "cvttss2si"
            | "cvttsd2si"
            | "cvtss2sd"
            | "cvtsd2ss"
            | "vcvtph2ps"
            | "vcvtps2ph"
    )
}

/// Whether `insn` reprograms an FPU control register that
/// [`pins_rounding_mode`] assumes holds its architectural default.
///
/// x86 names the write in the mnemonic — `ldmxcsr` for the SSE control
/// word, `fldcw` for the x87 one, and the state restores that reload
/// either wholesale. ARM does not: `msr` and `vmsr` write *any* system
/// register, and only the operand says which, so the mnemonic alone
/// would either miss the FPCR write or condemn every system-register
/// write in the function.
#[must_use]
pub(crate) fn writes_rounding_control(insn: &Instruction, arch: Arch) -> bool {
    match arch {
        Arch::X86 | Arch::X86_64 => x86::writes_fp_control(&insn.mnemonic),
        Arch::Aarch64 | Arch::Arm => {
            let mnem = insn.mnemonic.trim().to_ascii_lowercase();
            if !matches!(mnem.as_str(), "msr" | "vmsr") {
                return false;
            }
            insn.operands.first().is_some_and(|op| {
                let target = op.raw.trim().to_ascii_lowercase();
                target == "fpcr" || target == "fpscr"
            })
        }
        _ => false,
    }
}

/// Whether lifting `insn` pins a floating-point control field to
/// its architectural default, and so depends on nothing having
/// reprogrammed it. On x87 that is the rounding mode *and* the
/// precision control.
#[must_use]
pub(crate) fn pins_rounding_mode(insn: &Instruction, arch: Arch) -> bool {
    match arch {
        Arch::X86 | Arch::X86_64 => x86::pins_rounding_mode(insn),
        // ARM floating-point arithmetic is not lifted yet. The guard is
        // wired ahead of it deliberately: the moment a handler pins a
        // rounding mode it has to be listed here, or the slice would
        // report `Complete` while the hardware rounded some other way.
        _ => false,
    }
}

/// Floating-point operation applied to one lane, shared by the scalar
/// (`addss`) and packed (`addps`) handlers.
#[derive(Clone, Copy)]
pub(crate) enum FpArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
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
pub(crate) fn fp_lane_result(op: FpArithOp, a_bits: Expr, b_bits: Expr, lane_bits: u16) -> Expr {
    let (ebits, sbits) = fp_sort_bits(lane_bits);
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
            return Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(a_bits),
                else_expr: Box::new(b_bits),
            };
        }
    };
    Expr::fp_to_ieee_bv(arith)
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

/// Total form of [`fp_sort_bits_checked`], for the handlers that select
/// a lane width from a closed set of literals and so cannot present an
/// unmodelled one.
///
/// The fallback exists because `fp_lane_result` — shared with the ARM
/// scalar-FP handlers — returns an `Expr` rather than an `Option`.
/// Prefer the checked form anywhere a width is derived from an operand.
fn fp_sort_bits(lane_bits: u16) -> (u16, u16) {
    fp_sort_bits_checked(lane_bits).unwrap_or(FP_DOUBLE_SORT)
}

/// `(ebits, sbits)` of IEEE binary64.
const FP_DOUBLE_SORT: (u16, u16) = (11, 53);

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

/// Lift `slice` under `arch`. The lifter still only handles x86
/// mnemonics; passing a non-x86 arch results in every instruction
/// falling through to `IrStmt::Unsupported`. The arch flows into
/// `register_layout` lookups so register naming is unambiguous if a
/// future caller hand-builds non-x86 IR.
#[must_use]
pub fn lift_slice(slice: &Slice, arch: Arch) -> LiftedSlice {
    let mut ctx = LiftCtx::new(arch);
    // Lower Φ-merges interleaved with the kept-instruction stream in
    // execution order. Each merge's `tail_instructions` is the number
    // of kept instructions that execute *after* it, so its execution
    // index is `total - tail`; a merge is lowered just before the
    // instruction at that index. A single merge captures the whole tail
    // (index 0) and is lowered first, reproducing the historical order.
    // Chained/nested diamonds anchor at their true positions so an
    // upstream merge's `Ite` precedes the instructions and merges that
    // consume it. A merge that turns out not to be soundly foldable at
    // lift time emits nothing — the merged register then surfaces as a
    // free SSA input through its use, the sound free-input boundary
    // (widen-only, never a fabricated verdict).
    let total = slice.instructions.len();
    let mut merges_by_pos: Vec<Vec<&SliceMerge>> = vec![Vec::new(); total + 1];
    for merge in &slice.merges {
        let pos = total.saturating_sub(merge.tail_instructions);
        merges_by_pos[pos].push(merge);
    }
    for (pos, bucket) in merges_by_pos.iter().enumerate() {
        // Later-recorded (upstream) merges at the same position lower
        // first, so a nested inner diamond precedes the outer one.
        for merge in bucket.iter().rev() {
            lower_merge(&mut ctx, merge, arch);
        }
        if let Some(insn) = slice.instructions.get(pos) {
            ctx.lift_instruction(insn);
        }
    }
    let condition = lift_branch_condition(&slice.branch, arch);
    debug!(
        target: "r2smt::lift",
        at = %slice.branch.address,
        statements = ctx.stmts.len(),
        "slice lifted"
    );
    LiftedSlice {
        branch: slice.branch.clone(),
        statements: ctx.stmts,
        condition,
        status: slice.status.clone(),
        treat_truncation_as_inputs: slice.treat_truncation_as_inputs,
        arch,
    }
}

/// Lift a branch's [`BranchCondition`] into an [`Expr`] over the flag
/// variables. The arch's pointer width drives the `CxZero` family.
#[must_use]
pub fn lift_branch_condition(candidate: &BranchCandidate, arch: Arch) -> Expr {
    let bits = arch.pointer_bits();
    let zf = Expr::flag("ZF");
    let cf = Expr::flag("CF");
    let sf = Expr::flag("SF");
    let of = Expr::flag("OF");
    let pf = Expr::flag("PF");
    let one = || Expr::konst(1, 1);
    let zero = || Expr::konst(0, 1);
    match candidate.condition {
        BranchCondition::Equal => Expr::eq(zf, one()),
        BranchCondition::NotEqual => Expr::eq(zf, zero()),
        BranchCondition::Above => Expr::bool_and(Expr::eq(cf, zero()), Expr::eq(zf, zero())),
        BranchCondition::AboveOrEqual => Expr::eq(cf, zero()),
        BranchCondition::Below => Expr::eq(cf, one()),
        BranchCondition::BelowOrEqual => Expr::bool_or(Expr::eq(cf, one()), Expr::eq(zf, one())),
        BranchCondition::Greater => Expr::bool_and(Expr::eq(zf, zero()), Expr::eq(sf, of)),
        BranchCondition::GreaterOrEqual => Expr::eq(sf, of),
        BranchCondition::Less => Expr::ne(sf, of),
        BranchCondition::LessOrEqual => Expr::bool_or(Expr::eq(zf, one()), Expr::ne(sf, of)),
        BranchCondition::Sign => Expr::eq(sf, one()),
        BranchCondition::NotSign => Expr::eq(sf, zero()),
        BranchCondition::Overflow => Expr::eq(of, one()),
        BranchCondition::NotOverflow => Expr::eq(of, zero()),
        BranchCondition::ParityEven => Expr::eq(pf, one()),
        BranchCondition::ParityOdd => Expr::eq(pf, zero()),
        BranchCondition::CxZero => {
            // Modelled at the program's pointer width; the actual
            // register width depends on the mnemonic (`jcxz` ≡ cx,
            // `jecxz` ≡ ecx, `jrcxz` ≡ rcx) but our canonical naming
            // collapses them.
            Expr::eq(Expr::var("rcx", bits), Expr::konst(0, bits))
        }
        BranchCondition::RegisterZero | BranchCondition::RegisterNotZero => {
            // `cbz Rn` / `cbnz Rn` — read the register the collector
            // parsed out of operand[0], canonicalise via the
            // arch-aware layout table, and emit `Rn == 0` / `Rn != 0`.
            let (var, vbits) = aarch64_branch_var(candidate, arch);
            let cmp = Expr::eq(var, Expr::konst(0, vbits));
            match candidate.condition {
                BranchCondition::RegisterZero => cmp,
                _ => Expr::bool_not(cmp),
            }
        }
        BranchCondition::BitZero | BranchCondition::BitNotZero => {
            // `tbz Rn, #bit` / `tbnz Rn, #bit` — extract a single bit
            // and compare against zero.
            let (var, vbits) = aarch64_branch_var(candidate, arch);
            match candidate.bit_index {
                Some(bit) if u16::from(bit) < vbits => {
                    let bit = u16::from(bit);
                    let slice = Expr::extract(var, bit, bit);
                    let cmp = Expr::eq(slice, Expr::konst(0, 1));
                    match candidate.condition {
                        BranchCondition::BitZero => cmp,
                        _ => Expr::bool_not(cmp),
                    }
                }
                // Unparsed or out-of-range bit index: substituting a
                // concrete bit would fabricate a different predicate
                // (an unsound `AlwaysX`). Surface as a free symbolic
                // value instead — the solver can only widen it to
                // `BothPossible`, never fabricate a verdict.
                _ => Expr::Unknown(format!(
                    "tbz/tbnz bit-index unresolved for `{reg}`",
                    reg = candidate.compare_register.as_deref().unwrap_or("")
                )),
            }
        }
    }
}

/// Resolve a `cbz`/`cbnz`/`tbz`/`tbnz` candidate's compare register
/// against the arch's layout table. Falls back to a parent-width
/// `Unknown(name)` for unrecognised tokens so the SMT backend
/// produces a free input rather than silently dropping the operand.
fn aarch64_branch_var(candidate: &BranchCandidate, arch: Arch) -> (Expr, u16) {
    let raw = candidate.compare_register.as_deref().unwrap_or("");
    if let Some(layout) = register_layout(raw, arch) {
        // `xzr`/`wzr` always read 0. Mirrors `read_register` so
        // `cbz xzr` / `tbz xzr, #n` resolve precisely instead of
        // getting stuck on a free input named `xzr`.
        if layout.parent == "xzr" {
            return (Expr::konst(0, layout.width()), layout.width());
        }
        let parent_bits = arch.pointer_bits();
        if layout.hi < parent_bits {
            let parent = Expr::var(layout.parent, parent_bits);
            let width = layout.width();
            let var = if layout.lo == 0 && layout.hi + 1 == parent_bits {
                parent
            } else {
                Expr::extract(parent, layout.hi, layout.lo)
            };
            return (var, width);
        }
    }
    let bits = arch.pointer_bits();
    (Expr::Unknown(raw.to_string()), bits)
}

/// Lift a single instruction through **only** the per-mnemonic
/// handler dispatch, bypassing the P-code-first / ESIL-first ladder
/// that [`lift_slice`] runs.
///
/// This is the differential-lifting seam consumed by the
/// `r2smt-difflift` harness: it exposes the per-mnemonic lowering in
/// isolation so it can be cross-checked against the independent ESIL
/// ([`r2smt_esil::lift_esil`]) and P-code ([`r2smt_pcode::lift_pcode`])
/// lowerings of the same instruction. The production dispatch ladder
/// in [`lift_slice`] is deliberately **not** routed through this
/// function — its observable behaviour is unchanged.
#[must_use]
pub fn lift_per_mnemonic(insn: &Instruction, arch: Arch) -> Vec<IrStmt> {
    let mut ctx = LiftCtx::new(arch);
    // Closed structural dispatch over the supported ISAs (the
    // documented exhaustive-dispatch-table exception). Mirrors the
    // tail of `LiftCtx::lift_instruction` *without* the ESIL / P-code
    // short-circuits, which is the entire purpose of this seam.
    match arch {
        Arch::X86 | Arch::X86_64 => ctx.lift_instruction_x86(insn),
        Arch::Aarch64 => ctx.lift_instruction_aarch64(insn),
        Arch::Arm => ctx.lift_instruction_aarch32(insn),
        _ => ctx.stmts.push(IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!(
                "at {addr} (arch {arch:?})",
                addr = insn.address,
                arch = arch
            ),
        }),
    }
    ctx.stmts
}

struct LiftCtx {
    stmts: Vec<IrStmt>,
    bits: u16,
    arch: Arch,
    temp_counter: u32,
    cur_addr: Address,
    /// Symbolic x87 register stack, valid for the duration of one
    /// slice. A slice is a linear instruction sequence, so pushes and
    /// pops compose exactly; see `lift/x87.rs`.
    x87: X87Stack,
}

impl LiftCtx {
    fn new(arch: Arch) -> Self {
        Self {
            stmts: Vec::new(),
            bits: arch.pointer_bits(),
            arch,
            temp_counter: 0,
            cur_addr: Address::new(0),
            x87: X87Stack::new(),
        }
    }

    fn new_temp(&mut self, address: Address, width: u16) -> Var {
        let name = format!(
            "t_{addr:x}_{n}",
            addr = address.get(),
            n = self.temp_counter
        );
        self.temp_counter += 1;
        Var::new(name, width)
    }

    fn lift_instruction(&mut self, insn: &Instruction) {
        self.cur_addr = insn.address;
        // Integer-SIMD override: the ESIL / P-code lowerings model an
        // `xmm` register at the pointer width (64 bits) and do not
        // canonicalise it to its vector parent, so they produce a
        // disconnected, wrong-width value. Our per-mnemonic handlers
        // model these precisely at 128 bits — prefer them, bypassing
        // the ESIL/P-code ladder for this closed mnemonic set.
        if is_simd_instruction(insn, self.arch) {
            match self.arch {
                Arch::Aarch64 => self.lift_instruction_aarch64(insn),
                Arch::Arm => self.lift_instruction_aarch32(insn),
                _ => self.lift_instruction_x86(insn),
            }
            return;
        }
        // ESIL-first path: when radare2 has attached an ESIL string
        // to the instruction and the mini stack machine can evaluate
        // it, splice the resulting IrStmts straight into the buffer.
        // This covers every opcode r2 knows how to disassemble
        // without writing a per-mnemonic handler. Failures (unknown
        // tokens, control-flow markers, …) fall through to the
        // arch-specific dispatcher below — which keeps Fase C's
        // bespoke handlers as overrides for mnemonics ESIL describes
        // imprecisely.
        // P-code-first path: when the r2ghidra adapter attached SLEIGH
        // P-code for this instruction (opt-in `--ir pcode|auto`),
        // prefer it — decompiler-grade IR with explicit varnodes and
        // flag derivation. Any [`r2smt_pcode::PcodeError`] (opcode
        // outside the sound subset, …) falls through to ESIL exactly
        // as before; a declined P-code lift never emits output, so
        // this is sound and the default (no `pcode` attached) path is
        // byte-identical to before.
        if let Some(pcode) = insn.pcode.as_deref()
            && let Ok(lift) = r2smt_pcode::lift_pcode(pcode, self.arch)
        {
            debug!(
                target: "r2smt::lift",
                at = %insn.address,
                stmts = lift.statements.len(),
                "pcode-hit"
            );
            self.stmts.extend(lift.statements);
            return;
        }

        if let Some(esil) = insn.esil.as_deref()
            && let Ok(lift) = r2smt_esil::lift_esil(esil, self.arch)
        {
            debug!(
                target: "r2smt::lift",
                at = %insn.address,
                stmts = lift.statements.len(),
                "esil-hit"
            );
            self.stmts.extend(lift.statements);
            return;
        }
        debug!(
            target: "r2smt::lift",
            at = %insn.address,
            "esil-miss"
        );
        match self.arch {
            Arch::X86 | Arch::X86_64 => self.lift_instruction_x86(insn),
            Arch::Aarch64 => self.lift_instruction_aarch64(insn),
            Arch::Arm => self.lift_instruction_aarch32(insn),
            _ => self.stmts.push(IrStmt::Unsupported {
                mnemonic: insn.mnemonic.clone(),
                comment: format!(
                    "at {addr} (arch {arch:?})",
                    addr = insn.address,
                    arch = self.arch
                ),
            }),
        }
    }

    /// Width of the operand at its natural granularity (sub-register
    /// width, stack-slot width, or pointer width for immediates).
    fn operand_width(&self, op: &Operand) -> u16 {
        match op.kind {
            OperandKind::Register => {
                register_layout(&op.raw, self.arch).map_or(self.bits, |layout| layout.width())
            }
            OperandKind::Memory => memory_operand_width(&op.raw)
                .or_else(|| stack_slot(op).map(|(_, w)| w))
                .unwrap_or(self.bits),
            _ => self.bits,
        }
    }

    /// Read a register operand, returning the matching parent slice
    /// (full register or [`Expr::Extract`] of the parent). Returns
    /// `None` if the operand is not a recognised GPR.
    fn read_register(&self, op: &Operand) -> Option<Expr> {
        let layout = register_layout(&op.raw, self.arch)?;
        // `AArch64` zero registers always read as 0 regardless of
        // which alias is named. Modelling this lets opaque-predicate
        // patterns like `mov x0, xzr; cbz x0, …` resolve to
        // `AlwaysTrue` instead of getting stuck on a free input.
        if layout.parent == "xzr" {
            return Some(Expr::konst(0, layout.width()));
        }
        let parent_bits = self.bits;
        if layout.hi >= parent_bits {
            // Caller is running at a pointer width that cannot fit the
            // register (`rax` in a 32-bit lift). Surface as Unsound at
            // the lifter level rather than fabricate a slice.
            return None;
        }
        let parent = Expr::var(layout.parent, parent_bits);
        if layout.lo == 0 && layout.hi + 1 == parent_bits {
            Some(parent)
        } else {
            Some(Expr::extract(parent, layout.hi, layout.lo))
        }
    }

    /// Read an operand and coerce the result to `width` bits.
    ///
    /// Immediates are constructed directly at `width` (masked to the
    /// requested width), so a small constant alongside a sub-register
    /// read does not need to round-trip through a wider type.
    /// Sub-register and stack-slot reads are produced at their natural
    /// width and zero-extended / truncated to match `width`.
    fn read_operand_at(&self, op: &Operand, width: u16) -> Expr {
        match op.kind {
            OperandKind::Register => match self.read_register(op) {
                Some(natural) => {
                    let nw = register_layout(&op.raw, self.arch)
                        .map_or(self.bits, |layout| layout.width());
                    coerce_to_width(natural, width, nw)
                }
                None => Expr::Unknown(op.raw.clone()),
            },
            OperandKind::Immediate => match parse_immediate(&op.raw) {
                Some(value) => Expr::konst(u128::from(value & width_mask(width)), width),
                None => Expr::Unknown(op.raw.clone()),
            },
            // x86 memory is lowered to a `LoadMem` by
            // [`Self::read_operand_lowered`] before it can reach this
            // pure (`&self`) fallback; anything arriving here is memory
            // whose address cannot be modelled, so it stays opaque.
            OperandKind::Memory => Expr::Unknown(format!("mem {raw}", raw = op.raw)),
            _ => Expr::Unknown(op.raw.clone()),
        }
    }

    /// Read an operand, lowering an x86 memory source through the
    /// byte-granular memory model: it emits a `LoadMem` into a temp and
    /// returns that temp, so `[rax]` / `[rbp + rax*4]` / `[rbp - 8]`
    /// become precise loads instead of the `Expr::Unknown` /
    /// named-slot value `read_operand_at` produced. A load whose
    /// address folds to a constant `rbp`/`rsp` offset names its temp
    /// `stk_<base>_<off>` (the stack-slot canonical name) so the
    /// analyst alias (`var_4h`) still resolves in the pretty-printer;
    /// every other load uses a fresh `t_<addr>_<n>` temp. Registers,
    /// immediates, and x86 memory whose address cannot be modelled
    /// (rip / segment / subtracted-register) fall back to
    /// [`Self::read_operand_at`].
    fn read_operand_lowered(&mut self, op: &Operand, width: u16) -> Expr {
        if matches!(self.arch, Arch::X86 | Arch::X86_64)
            && op.kind == OperandKind::Memory
            && let Some(address) = x86_address_expr(op, self.bits)
        {
            let tmp = match stack_slot(op) {
                Some((name, _)) => Var::new(name, width),
                None => self.new_temp(self.cur_addr, width),
            };
            self.stmts.push(IrStmt::LoadMem {
                dst: tmp.clone(),
                address,
                bits: width,
            });
            return Expr::Var(tmp);
        }
        self.read_operand_at(op, width)
    }

    /// Build the right-hand-side expression that, when assigned to the
    /// parent register, captures writing `value` (already at the
    /// destination's natural width) to the operand `op`.
    fn build_parent_write(&self, op: &Operand, value: Expr) -> Option<(Var, Expr)> {
        let layout = register_layout(&op.raw, self.arch)?;
        let parent_bits = self.bits;
        if layout.hi >= parent_bits {
            return None;
        }
        let parent_var = Var::new(layout.parent, parent_bits);
        let rhs = if layout.lo == 0 && layout.hi + 1 == parent_bits {
            // Full-width write: replaces the parent entirely.
            value
        } else if layout.zero_extends_parent_64 && parent_bits == 64 {
            // x86_64 32-bit write: zero-extends to 64 bits.
            Expr::zero_ext(value, parent_bits)
        } else {
            // Partial write: preserve surrounding bits.
            let parent_read = Expr::var(layout.parent, parent_bits);
            let mut acc = value;
            if layout.lo > 0 {
                let low_preserve = Expr::extract(parent_read.clone(), layout.lo - 1, 0);
                acc = Expr::concat(acc, low_preserve);
            }
            if layout.hi + 1 < parent_bits {
                let high_preserve = Expr::extract(parent_read, parent_bits - 1, layout.hi + 1);
                acc = Expr::concat(high_preserve, acc);
            }
            acc
        };
        Some((parent_var, rhs))
    }

    /// Compute the destination register [`Var`] the operand `op` writes
    /// to. Used by paths that build their own right-hand side (`lea`,
    /// the unsupported-destination fall-throughs); `lea` always targets
    /// a register, so a non-register operand yields `None`.
    fn dst_var(&self, op: &Operand) -> Option<Var> {
        if let Some(layout) = register_layout(&op.raw, self.arch)
            && layout.hi < self.bits
        {
            return Some(Var::new(layout.parent, self.bits));
        }
        None
    }

    fn assign(&mut self, dst: Var, src: Expr) {
        self.stmts.push(IrStmt::Assign { dst, src });
    }

    fn set_flag(&mut self, name: &str, src: Expr) {
        self.assign(Var::new(name, 1), src);
    }

    fn clear_all_flags(&mut self) {
        for f in FLAGS {
            self.set_flag(f, Expr::Unknown(String::new()));
        }
    }

    fn update_logic_flags(&mut self, result: &Expr, width: u16) {
        // After `and / or / xor / test`:
        //   ZF = result == 0
        //   SF = msb(result)
        //   CF = 0, OF = 0
        //   PF = parity (unmodelled).
        self.set_flag("ZF", Expr::eq(result.clone(), Expr::konst(0, width)));
        self.set_flag("SF", Expr::slt(result.clone(), Expr::konst(0, width)));
        self.set_flag("CF", Expr::konst(0, 1));
        self.set_flag("OF", Expr::konst(0, 1));
        self.set_flag("PF", Expr::Unknown(String::new()));
    }

    /// Emit an assignment that writes `value` to register operand
    /// `dst_op`, accounting for sub-register semantics. Returns
    /// `false` if the operand is not a recognised register (caller
    /// should fall back).
    fn write_register_to(&mut self, dst_op: &Operand, value: Expr) -> bool {
        if let Some((dst_var, rhs)) = self.build_parent_write(dst_op, value) {
            self.assign(dst_var, rhs);
            return true;
        }
        false
    }

    /// Emit an assignment for any supported destination. Register
    /// destinations account for sub-register semantics; x86 memory
    /// destinations whose address can be modelled emit a byte-granular
    /// `StoreMem` (including recognised stack slots — the named-slot
    /// scheme no longer represents stack values, only the byte model
    /// does). Unmodellable memory (rip / segment / subtracted-register)
    /// returns `false` so the caller marks the instruction unsupported.
    fn write_dst(&mut self, dst_op: &Operand, value: Expr, dst_width: u16) -> bool {
        if dst_op.kind == OperandKind::Register {
            return self.write_register_to(dst_op, value);
        }
        if matches!(self.arch, Arch::X86 | Arch::X86_64)
            && dst_op.kind == OperandKind::Memory
            && let Some(address) = x86_address_expr(dst_op, self.bits)
        {
            self.stmts.push(IrStmt::StoreMem {
                address,
                value,
                bits: dst_width,
            });
            return true;
        }
        false
    }

    /// The vector-register layout `op` names, if it names one.
    fn simd_layout(&self, op: &Operand) -> Option<RegisterLayout> {
        if op.kind != OperandKind::Register {
            return None;
        }
        let layout = register_layout(&op.raw, self.arch)?;
        is_simd_parent(layout.parent, self.arch).then_some(layout)
    }

    /// Whether `op` names a view of a SIMD vector register.
    fn is_simd_register(&self, op: &Operand) -> bool {
        self.simd_layout(op).is_some()
    }

    /// Architectural width of this ISA's vector parent.
    fn simd_parent_width(&self) -> Option<u16> {
        simd_parent_bits(self.arch)
    }

    /// Whether `op` is a memory operand a SIMD handler may lower through
    /// the byte-granular model: x86 memory whose address expression the
    /// lifter can build. Segment-prefixed, rip-relative and
    /// subtracted-register forms stay out, matching the slicer's own
    /// memory gate — those truncate before ever reaching a handler.
    fn is_modellable_simd_memory(&self, op: &Operand) -> bool {
        matches!(self.arch, Arch::X86 | Arch::X86_64)
            && op.kind == OperandKind::Memory
            && x86_memory_modellable(op)
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
    /// [`Self::read_operand_lowered`], which also supplies the
    /// `stk_<base>_<off>` naming so an analyst alias still resolves.
    /// Reading the whole view once — rather than once per lane — is what
    /// keeps a packed operand to a single load.
    ///
    /// `None` for anything else. The caller marks the instruction
    /// unsupported rather than fabricating a value.
    fn simd_operand_value(&mut self, op: &Operand, view_bits: u16) -> Option<Expr> {
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
    fn read_xmm_operand(&mut self, op: &Operand) -> Option<Expr> {
        let view = self.simd_view_bits(op)?;
        self.simd_operand_value(op, view)
    }

    /// Read the low `lane_bits`-wide lane of a SIMD operand and
    /// reinterpret it as an IEEE float of the matching sort (32→single,
    /// 64→double). Used by the scalar SSE FP handlers.
    fn read_simd_lane_fp(&mut self, op: &Operand, lane_bits: u16) -> Option<Expr> {
        let raw = self.read_simd_lane_bits(op, lane_bits)?;
        let (ebits, sbits) = fp_sort_bits_checked(lane_bits)?;
        Some(Expr::bv_to_fp(raw, ebits, sbits))
    }

    /// The raw bit-vector of the low lane, before any float
    /// reinterpretation. Kept separate so the compare handlers can build
    /// a mask without a pointless round trip through the float sort.
    ///
    /// A scalar memory operand is loaded at the *lane* width, not the
    /// vector view: `movss xmm0, dword [rbp - 8]` reads four bytes.
    fn read_simd_lane_bits(&mut self, op: &Operand, lane_bits: u16) -> Option<Expr> {
        if self.is_modellable_simd_memory(op) {
            return Some(self.read_operand_lowered(op, lane_bits));
        }
        let view = self.simd_view_bits(op)?;
        let value = self.simd_operand_value(op, view)?;
        Self::extract_lane(value, lane_bits, 0)
    }

    /// Extract lane `index` (counting from the least-significant) out of
    /// an already-materialised operand value.
    fn extract_lane(value: Expr, lane_bits: u16, index: u16) -> Option<Expr> {
        let lo = lane_bits.checked_mul(index)?;
        let hi = lo.checked_add(lane_bits)?.checked_sub(1)?;
        Some(extract_collapsing(value, hi, lo))
    }

    /// Write `lane_value` (an IEEE bit-vector of `lane_bits`) to the low
    /// lane of a SIMD destination.
    ///
    /// A register destination keeps the parent bits around the lane
    /// (legacy SSE scalar semantics). A memory destination stores exactly
    /// `lane_bits` — there is nothing around the lane to preserve, which
    /// is why `movss [rbp - 8], xmm0` writes four bytes and leaves the
    /// rest of the slot alone.
    ///
    /// The lane sits at the view's own offset in the parent, so on
    /// `AArch32` a write to `s3` preserves the bits *below* it as well as
    /// above.
    fn write_simd_lane(&mut self, op: &Operand, lane_value: Expr, lane_bits: u16) -> bool {
        if self.is_modellable_simd_memory(op) {
            return self.write_dst(op, lane_value, lane_bits);
        }
        let (Some(layout), Some(parent_bits)) = (self.simd_layout(op), self.simd_parent_width())
        else {
            return false;
        };
        let Some(lane_top) = layout.lo.checked_add(lane_bits) else {
            return false;
        };
        if lane_top > parent_bits {
            return false;
        }
        let parent = Expr::var(layout.parent, parent_bits);
        let value = Self::splice_into_parent(&parent, lane_value, layout.lo, lane_top, parent_bits);
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
    fn packed_lane_count(view_bits: u16, lane_bits: u16) -> Option<u16> {
        if lane_bits == 0 || view_bits % lane_bits != 0 {
            return None;
        }
        Some(view_bits / lane_bits)
    }

    /// Assemble per-lane bit-vectors, least-significant lane first, into
    /// one value of the full view width.
    fn concat_lanes(lanes: Vec<Expr>) -> Option<Expr> {
        let mut iter = lanes.into_iter();
        let mut acc = iter.next()?;
        for lane in iter {
            acc = Expr::concat(lane, acc);
        }
        Some(acc)
    }

    /// Width of a SIMD operand's view: 128 / 256 / 512 for a register,
    /// and for memory whatever size prefix radare2 attached
    /// (`xmmword [rsi]` → 128). `None` if `op` is neither, or is a
    /// memory operand with no size prefix — guessing a width there would
    /// silently load the wrong number of bytes.
    fn simd_view_bits(&self, op: &Operand) -> Option<u16> {
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
    fn simd_instruction_view_bits(&self, ops: &[&Operand]) -> Option<u16> {
        ops.iter()
            .find(|op| self.is_simd_register(op))
            .or_else(|| ops.first())
            .and_then(|op| self.simd_view_bits(op))
    }

    /// Write a `value` (already sized to the operand's view width) to an
    /// x86 SIMD **register** destination, reconstructing the full-width
    /// parent per the write's upper-bits rule: `zero_upper` (VEX-encoded
    /// `v*` forms) zeroes bits above the view; otherwise (legacy SSE)
    /// the bits above the view are preserved from the parent's prior
    /// value. A memory destination stores the view width verbatim —
    /// there are no parent bits to reconstruct.
    fn write_xmm_dst(&mut self, op: &Operand, value: Expr, zero_upper: bool) -> bool {
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

    fn binop_width(&self, lhs: &Operand, rhs: &Operand) -> u16 {
        // For the compare/test family the operation width is the width
        // of the register/memory operand; `read_operand_at` then masks
        // the immediate down to it. An immediate's `operand_width` is
        // the pointer-width pseudo-size (`self.bits`), so taking the
        // max here would inflate `cmp al, 1` to 64 bits and corrupt the
        // sign-dependent flags (SF) for every sub-register compare.
        let lhs_imm = matches!(lhs.kind, OperandKind::Immediate);
        let rhs_imm = matches!(rhs.kind, OperandKind::Immediate);
        match (lhs_imm, rhs_imm) {
            (false, true) => self.operand_width(lhs),
            (true, false) => self.operand_width(rhs),
            _ => {
                let lw = self.operand_width(lhs);
                let rw = self.operand_width(rhs);
                if lw >= rw { lw } else { rw }
            }
        }
    }

    // ---------- AArch64 handlers ------------------------------------
    //
    // AArch64 instructions are 3-operand (`Rd, Rs1, Rs2`) where Rd is
    // write-only — unlike x86's 2-operand RMW shape. Reads / writes
    // route through the same `read_operand_at` / `write_register_to`
    // helpers as the x86 path; the arch-aware `register_layout`
    // resolves `x0` / `w0` correctly because `LiftCtx.arch` is
    // forwarded into every layout query.
    //
    // Flag polarity is x86-style: `cmp_aarch64` emits `CF = (a < b)
    // unsigned`. This is the opposite of ARM's architectural carry
    // (`C = (a >= b) unsigned`), but it lets the existing
    // `lift_branch_condition` stay arch-agnostic — AArch64 condition
    // codes are mapped to x86-equivalent `BranchCondition` variants
    // by `condition::classify`.
}

#[derive(Debug, Clone, Copy)]
enum BitwiseOp {
    And,
    Or,
}

#[derive(Debug, Clone, Copy)]
enum ShiftOp {
    Shl,
    Shr,
    Sar,
}

#[derive(Debug, Clone, Copy)]
enum ExtendKind {
    Zero,
    Sign,
}

/// Generic `AArch64` binary operator used by `lift_aarch64_arith3`.
///
/// `AArch64` instructions are uniformly 3-operand (`Rd, Rs1, Rs2`)
/// without the x86 RMW shape, so a single handler parameterised by
/// this enum covers the whole arithmetic / logical / shift family.
/// A memory access resolved from an operand: the effective address plus
/// an optional base-register writeback (`Xn := Xn + delta`) for the
/// pre/post-index forms. Shared by the `AArch64` and `AArch32` handlers.
pub(super) struct MemAccess {
    pub(super) address: Expr,
    pub(super) writeback: Option<(String, i64)>,
}

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
}

/// Else-branch shape selected by an `AArch64` conditional-select
/// arithmetic instruction (`csinc` / `csinv` / `csneg` and their
/// `cinc` / `cinv` / `cneg` 3-operand aliases).
#[derive(Debug, Clone, Copy)]
enum CsArithOp {
    /// `Rm + 1` (`csinc`, `cinc`).
    Inc,
    /// `~Rm` (`csinv`, `cinv`).
    Inv,
    /// `-Rm` (`csneg`, `cneg`).
    Neg,
}

impl BinOp {
    fn apply(self, lhs: Expr, rhs: Expr) -> Expr {
        match self {
            Self::Add => Expr::add(lhs, rhs),
            Self::Sub => Expr::sub(lhs, rhs),
            Self::Mul => Expr::mul(lhs, rhs),
            Self::UDiv => Expr::udiv(lhs, rhs),
            Self::SDiv => Expr::sdiv(lhs, rhs),
            Self::And => Expr::bv_and(lhs, rhs),
            Self::Or => Expr::bv_or(lhs, rhs),
            Self::Xor => Expr::bv_xor(lhs, rhs),
            Self::Shl => Expr::shl(lhs, rhs),
            Self::Shr => Expr::lshr(lhs, rhs),
            Self::Sar => Expr::ashr(lhs, rhs),
        }
    }
}

/// Recognised `AArch64` / `AArch32` condition-code suffixes, longest
/// first so a greedy `ends_with` walk picks the proper boundary
/// (e.g. `hi` before `s` would split `cmphi` correctly).
const AARCH_COND_SUFFIXES: &[&str] = &[
    "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le",
    "al", "nv",
];

/// `AArch32` base mnemonics that the dispatcher knows how to lift.
/// Used by the predicated-execution wrapper to decide whether to
/// peel the cond suffix or fall through to `Unsupported`.
const AARCH32_BASE_MNEMONICS: &[&str] = &[
    "mov", "mvn", "add", "adds", "sub", "subs", "rsb", "rsbs", "and", "ands", "bic", "bics", "orr",
    "orrs", "eor", "eors", "mul", "muls", "udiv", "sdiv", "lsl", "lsls", "lsr", "lsrs", "asr",
    "asrs", "cmp", "cmn", "tst", "teq",
];

pub(crate) fn is_aarch32_base_supported(base: &str) -> bool {
    AARCH32_BASE_MNEMONICS.contains(&base)
}

/// If `mnem` ends with a recognised condition-code suffix and the
/// remaining prefix is non-empty, return `(base, cond_suffix)`.
/// Otherwise return `None` so the caller dispatches the mnem as-is.
pub(crate) fn strip_aarch32_cond_suffix(mnem: &str) -> Option<(&str, &str)> {
    for cond in AARCH_COND_SUFFIXES {
        let Some(base) = mnem.strip_suffix(cond) else {
            continue;
        };
        if base.is_empty() {
            // Pure conditional mnemonics like `eq` make no sense.
            continue;
        }
        return Some((base, cond));
    }
    None
}

/// Translate an `AArch64` condition suffix (`eq`, `ne`, `cs`, …)
/// into a 1-bit boolean `Expr` over the lifter's flag variables.
/// Mirrors the predicates [`lift_branch_condition`] emits for the
/// equivalent [`BranchCondition`] variants. Used by `csel` / `cset`
/// lifting and any future predicated-execution path.
fn aarch64_cond_suffix_to_predicate(raw: &str) -> Option<Expr> {
    let suffix = raw.trim().to_ascii_lowercase();
    let zf = || Expr::flag("ZF");
    let cf = || Expr::flag("CF");
    let sf = || Expr::flag("SF");
    let of = || Expr::flag("OF");
    let one = || Expr::konst(1, 1);
    let zero = || Expr::konst(0, 1);
    Some(match suffix.as_str() {
        "eq" => Expr::eq(zf(), one()),
        "ne" => Expr::eq(zf(), zero()),
        "cs" | "hs" => Expr::eq(cf(), zero()),
        "cc" | "lo" => Expr::eq(cf(), one()),
        "mi" => Expr::eq(sf(), one()),
        "pl" => Expr::eq(sf(), zero()),
        "vs" => Expr::eq(of(), one()),
        "vc" => Expr::eq(of(), zero()),
        "hi" => Expr::bool_and(Expr::eq(cf(), zero()), Expr::eq(zf(), zero())),
        "ls" => Expr::bool_or(Expr::eq(cf(), one()), Expr::eq(zf(), one())),
        "ge" => Expr::eq(sf(), of()),
        "lt" => Expr::ne(sf(), of()),
        "gt" => Expr::bool_and(Expr::eq(zf(), zero()), Expr::eq(sf(), of())),
        "le" => Expr::bool_or(Expr::eq(zf(), one()), Expr::ne(sf(), of())),
        _ => return None,
    })
}

fn nonzero_width(width: u16) -> Option<u16> {
    if width == 0 { None } else { Some(width) }
}

fn coerce_to_width(value: Expr, target: u16, source: u16) -> Expr {
    match source.cmp(&target) {
        std::cmp::Ordering::Equal => value,
        std::cmp::Ordering::Less => Expr::zero_ext(value, target),
        std::cmp::Ordering::Greater => Expr::extract(value, target - 1, 0),
    }
}

/// Mask covering the low `width` bits of a `u64`. `width == 64` keeps
/// every bit; smaller widths zero the upper bits.
const fn width_mask(width: u16) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

// `parse_immediate` parses signed ARM immediates (`#-1`, `#0x10`) but
// returns the bit-pattern as `u64` so it can flow straight into the IR's
// fixed-width bit-vector constants — the slicer never re-interprets the
// returned value as a signed integer, so the `i64 as u64` reinterpret on
// the negation path is intentional.
#[allow(clippy::cast_sign_loss)]
fn parse_immediate(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    // AArch64 / AArch32 assembly emits immediates with a leading `#`
    // (`#0x10`, `#-1`). Strip it so the rest of the parser only deals
    // with the numeric body.
    let trimmed = trimmed.strip_prefix('#').unwrap_or(trimmed).trim_start();
    let (negative, body) = if let Some(rest) = trimmed.strip_prefix('-') {
        (true, rest.trim_start())
    } else {
        (false, trimmed)
    };
    let value = if let Some(rest) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).ok()?
    } else {
        body.parse::<u64>().ok()?
    };
    if negative {
        Some(value.wrapping_neg())
    } else {
        Some(value)
    }
}

/// Build a symbolic address from an x86 memory operand
/// `[base {+ index*scale} {± disp}]` at the pointer width. Registers
/// resolve through [`register_layout`] to their 64-bit parent (a
/// `rip`-relative base resolves the same way, yielding a free
/// symbolic `rip` — sound-widening); the displacement folds into a
/// single constant. Returns `None` for shapes it cannot model
/// soundly: a subtracted register, or a segment-prefixed operand
/// (`fs:[…]` / `gs:[…]`) whose true address is `segment_base + offset`
/// — modelling it as a bare `offset` would alias with an absolute
/// access at that offset and could *fabricate* a verdict, so the
/// caller declines and the operand stays opaque.
fn x86_address_expr(op: &Operand, ptr_bits: u16) -> Option<Expr> {
    if op.kind != OperandKind::Memory {
        return None;
    }
    if has_segment_prefix(&op.raw) {
        return None;
    }
    let lower = op.raw.to_ascii_lowercase();
    let lb = lower.find('[')?;
    let rb = lower.find(']')?;
    if rb <= lb {
        return None;
    }
    let inner = lower[lb + 1..rb].trim();
    if inner.is_empty() {
        return None;
    }
    let mut acc: Option<Expr> = None;
    let mut disp: i64 = 0;
    for (term, negative) in split_signed_terms(inner) {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        if let Some((reg_s, scale_s)) = term.split_once('*') {
            // index * scale — x86 never subtracts an index.
            if negative {
                return None;
            }
            let parent = register_layout(reg_s.trim(), Arch::X86_64).map(|l| l.parent)?;
            let scale = parse_immediate(scale_s.trim())?;
            let scaled = Expr::mul(
                Expr::var(parent, ptr_bits),
                Expr::konst(u128::from(scale & width_mask(ptr_bits)), ptr_bits),
            );
            acc = Some(match acc.take() {
                None => scaled,
                Some(a) => Expr::add(a, scaled),
            });
        } else if let Some(layout) = register_layout(term, Arch::X86_64) {
            if negative {
                return None;
            }
            let e = Expr::var(layout.parent, ptr_bits);
            acc = Some(match acc.take() {
                None => e,
                Some(a) => Expr::add(a, e),
            });
        } else {
            let mag = parse_immediate(term)?;
            let signed = i64::try_from(mag).ok()?;
            disp = disp.checked_add(if negative { -signed } else { signed })?;
        }
    }
    let base = acc.unwrap_or_else(|| Expr::konst(0, ptr_bits));
    if disp == 0 {
        return Some(base);
    }
    let masked = u64::from_le_bytes(disp.to_le_bytes()) & width_mask(ptr_bits);
    Some(Expr::add(base, Expr::konst(u128::from(masked), ptr_bits)))
}

/// Whether an x86 memory operand's address can be built by the
/// byte-granular model — i.e. [`x86_address_expr`] would return `Some`.
/// The pointer width is irrelevant to the yes/no answer, so a fixed
/// 64-bit probe suffices. Used by the slicer's memory gate
/// ([`crate::effect::has_unmodellable_memory`]) so modellable memory
/// resolves without `--allow-memory` while opaque memory stays gated.
pub(crate) fn x86_memory_modellable(op: &Operand) -> bool {
    x86_address_expr(op, 64).is_some()
}

/// x86 segment-override prefixes. A memory operand carrying one
/// (`fs:[0x30]`, `gs:[0x60]`, …) addresses `segment_base + offset`,
/// not `offset`, so the byte-model address parser must decline it.
const X86_SEGMENT_PREFIXES: [&str; 6] = ["fs:", "gs:", "cs:", "ds:", "es:", "ss:"];

/// Whether a memory operand carries an x86 segment-override prefix.
fn has_segment_prefix(raw: &str) -> bool {
    let lower = raw.to_ascii_lowercase();
    X86_SEGMENT_PREFIXES.iter().any(|seg| lower.contains(seg))
}

/// Split an x86 address body into `(term, is_negative)` pairs on
/// top-level `+` / `-` separators.
fn split_signed_terms(inner: &str) -> Vec<(&str, bool)> {
    let mut terms = Vec::new();
    let mut start = 0;
    let mut negative = false;
    for (i, ch) in inner.char_indices() {
        if (ch == '+' || ch == '-') && i > 0 {
            terms.push((&inner[start..i], negative));
            negative = ch == '-';
            start = i + 1;
        }
    }
    terms.push((&inner[start..], negative));
    terms
}

#[cfg(test)]
mod tests;
