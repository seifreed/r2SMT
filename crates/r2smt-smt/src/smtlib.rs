//! SMT-LIB2 emitter for a [`SsaLiftedSlice`].
//!
//! Produces a self-contained SMT-LIB2 script that any SMT solver
//! understanding the `QF_BV` theory can consume. The script declares
//! every free input, every SSA-assigned variable, asserts each
//! `IrStmt::Assign` as an equality, and pushes / pops a single
//! query for the branch condition's truth value.
//!
//! Two scripts are produced per branch: one assuming the condition
//! is `true`, one assuming `false`. Combining the two `(check-sat)`
//! outcomes gives the same verdict ladder as the Z3 backend
//! ([`crate::SmtResult`]).
//!
//! Output uses the `QF_BV` logic — no quantifiers, only the
//! fixed-width bit-vector theory the slicer's IR maps onto — widening
//! to `QF_BVFP` only for a slice that carries floating point, so
//! float-free scripts are unaffected.
//!
//! [`emit_query`] is lossy: where the renderer cannot encode a shape it
//! substitutes a placeholder to keep the script parseable for
//! render-only consumers. Verdict paths must use [`emit_query_strict`],
//! which fails instead.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use r2smt_common::smt::SolveOptions;
use r2smt_ir::expr::{Expr, RoundingMode};
use r2smt_ir::stmt::IrStmt;
use r2smt_ssa::SsaLiftedSlice;

/// Why the renderer could not produce a faithful script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// A term the textual backend cannot encode portably. Carries an
    /// open-domain description of the offending shape: the set of
    /// rejectable shapes is not a closed enumeration (it grows with the
    /// IR), so a typed kind would be a lie.
    ///
    /// Started as floating-point only and is no longer: a rotate by a
    /// non-constant amount belongs here too, since SMT-LIB spells only
    /// the constant-amount form and the extension is not portable.
    Unrenderable(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unrenderable(detail) => {
                write!(f, "term not renderable in SMT-LIB: {detail}")
            }
        }
    }
}

impl std::error::Error for RenderError {}

/// Prefix for the bit-vector variables minted by [`fp_bridge_to_bv`].
const IEEE_BRIDGE_PREFIX: &str = "__fpbv_";

/// Mutable renderer state shared by every expression rendered into one
/// script: auxiliary declarations / assertions awaiting a flush, the
/// counter minting their names, and the first shape the renderer had to
/// refuse.
///
/// A single instance must span a whole script — preamble *and* branch
/// condition — or the fresh-name counter restarts and mints a duplicate
/// declaration.
#[derive(Debug, Default)]
struct RenderCtx {
    aux: Vec<String>,
    fresh_fp_bv: u32,
    unsupported: Option<RenderError>,
    /// Every symbol declared so far, with the width it was declared at.
    ///
    /// The width is the point. This was a bare name list until
    /// 2026-08-06, so a second `Var` of the same name at a different
    /// width was silently dropped and every *use* rendered at its own
    /// `v.bits` — an ill-sorted script the solver rejects with a parse
    /// error, where the Z3 encoder fails closed to `Unsound`. Two
    /// backends disagreeing by parse error rather than by verdict is
    /// exactly what the strict emitter exists to prevent.
    declared: BTreeMap<String, u16>,
}

impl RenderCtx {
    fn new() -> Self {
        Self::default()
    }

    fn fresh_ieee_name(&mut self) -> String {
        let name = format!("{IEEE_BRIDGE_PREFIX}{n}", n = self.fresh_fp_bv);
        self.fresh_fp_bv = self.fresh_fp_bv.wrapping_add(1);
        name
    }

    /// Record a shape the renderer cannot encode. First one wins: it is
    /// the one closest to the root of the refusal, and the caller only
    /// needs a single reason to decline the whole slice.
    fn note_unsupported(&mut self, detail: impl Into<String>) {
        if self.unsupported.is_none() {
            self.unsupported = Some(RenderError::Unrenderable(detail.into()));
        }
    }

    /// Drain the pending auxiliary lines into `out` in push order, so a
    /// declaration always precedes the assertion referencing it.
    ///
    /// Every site that writes an `(assert …)` line must call this
    /// immediately before doing so: an auxiliary assertion constrains a
    /// name spliced into that very line.
    fn flush_into(&mut self, out: &mut String) {
        for line in self.aux.drain(..) {
            let _ = writeln!(out, "{line}");
        }
    }
}

/// Render the slice's preamble (declarations + statement assertions)
/// into SMT-LIB2 text, leaving the branch-condition query to the
/// caller. The output ends with a blank line so the caller can
/// append a `(push)`/`(assert <cond>)`/`(check-sat)`/`(pop)` block
/// directly.
#[must_use]
pub fn emit_preamble(slice: &SsaLiftedSlice, options: &SolveOptions) -> String {
    emit_preamble_with(slice, options, &mut RenderCtx::new())
}

/// [`emit_preamble`] against a caller-owned [`RenderCtx`], so the
/// preamble and the branch condition appended by [`emit_query`] share
/// one auxiliary-name namespace.
fn emit_preamble_with(
    slice: &SsaLiftedSlice,
    options: &SolveOptions,
    ctx: &mut RenderCtx,
) -> String {
    let mut out = String::new();
    // A float-free slice must keep emitting exactly `QF_BV`: widening
    // the logic unnecessarily changes every existing script.
    let logic = if slice_uses_fp(slice) {
        "QF_BVFP"
    } else {
        "QF_BV"
    };
    let _ = writeln!(out, "(set-logic {logic})");
    let _ = writeln!(out, "(set-option :produce-models false)");
    // Pin the subprocess backends' PRNG (standard SMT-LIB option,
    // honoured by Z3 / CVC5 / Bitwuzla) so a query's verdict is
    // reproducible run-to-run, mirroring the Z3 FFI `random_seed`.
    let _ = writeln!(
        out,
        "(set-option :random-seed {seed})",
        seed = options.random_seed
    );
    let _ = writeln!(out, "(set-info :status unknown)");
    let _ = writeln!(out, "; r2smt slice @ {addr}", addr = slice.branch.address);
    let _ = writeln!(out, "; timeout-ms: {ms}", ms = options.timeout_ms);
    let mut mem = TextMemory::new(ptr_bits_for(slice.arch));
    for var in &slice.inputs {
        declare_bv(&mut out, &var.name, var.bits, ctx);
    }
    for stmt in &slice.statements {
        match stmt {
            IrStmt::Assign { dst, src } => {
                declare_bv(&mut out, &dst.name, dst.bits, ctx);
                let rhs = render_expr_with_width(src, dst.bits, ctx);
                ctx.flush_into(&mut out);
                let _ = writeln!(out, "(assert (= {lhs} {rhs}))", lhs = smt_symbol(&dst.name));
            }
            IrStmt::StoreMem {
                address,
                value,
                bits,
            } => {
                mem.record_store(address, value, *bits, ctx);
                ctx.flush_into(&mut out);
            }
            IrStmt::LoadMem { dst, address, bits } => {
                declare_bv(&mut out, &dst.name, dst.bits, ctx);
                mem.emit_load(&mut out, dst, address, *bits, ctx);
            }
            // Unsupported / Nop emit nothing; the SMT side stays an
            // over-approximation for those (extra freedom can only widen
            // `AlwaysX` to `BothPossible`, never fabricate one).
            IrStmt::Unsupported { .. } | IrStmt::Nop => {}
        }
    }
    let _ = writeln!(out);
    out
}

/// Pointer width in bits for the target architecture, matching the Z3
/// encoder's `ptr_bits`.
fn ptr_bits_for(arch: r2smt_common::Arch) -> u16 {
    match arch {
        r2smt_common::Arch::X86_64 | r2smt_common::Arch::Aarch64 => 64,
        _ => 32,
    }
}

/// Convenience helper combining [`emit_preamble`] with the
/// branch-condition query block. Two scripts are produced; the
/// caller decides which polarity to run first (typical pattern:
/// run "taken" first, then "not-taken").
#[must_use]
pub fn emit_query(slice: &SsaLiftedSlice, options: &SolveOptions, polarity: bool) -> String {
    build_query(slice, options, polarity, &mut RenderCtx::new())
}

/// [`emit_query`], failing instead of emitting a script the renderer
/// could not encode faithfully.
///
/// Verdict paths must use this: [`emit_query`] keeps a lossy contract
/// for render-only consumers, substituting a placeholder where it had
/// to refuse, and a verdict derived from a placeholder would be
/// unsound.
///
/// # Errors
///
/// Returns [`RenderError`] when the slice carries a floating-point
/// shape with no portable SMT-LIB encoding.
pub fn emit_query_strict(
    slice: &SsaLiftedSlice,
    options: &SolveOptions,
    polarity: bool,
) -> Result<String, RenderError> {
    let mut ctx = RenderCtx::new();
    let script = build_query(slice, options, polarity, &mut ctx);
    ctx.unsupported.map_or(Ok(script), Err)
}

fn build_query(
    slice: &SsaLiftedSlice,
    options: &SolveOptions,
    polarity: bool,
    ctx: &mut RenderCtx,
) -> String {
    let mut script = emit_preamble_with(slice, options, ctx);
    let (cond, cond_bits) = render_expr(&slice.condition, ctx);
    let assertion = if cond_bits == 1 {
        // The reachable path: every lifted branch condition is a 1-bit
        // comparison / boolean combination. Byte-identical to the prior
        // output for both polarities.
        if polarity {
            format!("(= {cond} #b1)")
        } else {
            format!("(= {cond} #b0)")
        }
    } else {
        // A wider value in boolean position is true iff nonzero, matching
        // the Z3 encoder's `bv_is_true`. No lifter routes a multi-bit
        // value into a condition today, so this arm is latent — it keeps
        // the two backends from diverging (the coerce-to-bit-0 form used
        // here previously answered `bit0 == 1` where Z3 answered `!= 0`).
        let nonzero = format!("(distinct {cond} (_ bv0 {cond_bits}))");
        if polarity {
            nonzero
        } else {
            format!("(not {nonzero})")
        }
    };
    ctx.flush_into(&mut script);
    let _ = writeln!(&mut script, "(assert {assertion})");
    let _ = writeln!(&mut script, "(check-sat)");
    script
}

/// Byte-store cap for the textual backend, matching the Z3 encoder's
/// `MEM_BYTE_STORE_CAP`. Overflow havocs the store list.
const MEM_BYTE_STORE_CAP: usize = 4096;

/// Byte-granular memory model for the textual SMT-LIB backend, mirroring
/// the Z3 encoder's `Ite`-chain (`encoder.rs`). Each `StoreMem` enqueues
/// its bytes little-endian; each `LoadMem` reads latest-write-wins by
/// folding one `ite` per byte over the store list **oldest → latest**,
/// so the latest write is the outermost `ite`. Bounded by
/// [`MEM_BYTE_STORE_CAP`]: overflow havocs the list and subsequent loads
/// read fresh free bytes — precision lost, soundness preserved.
struct TextMemory {
    stores: Vec<(String, String)>,
    havoced: bool,
    load_counter: u32,
    ptr_bits: u16,
    /// Load memo restoring read-read consistency, mirroring the Z3
    /// encoder's `load_memo`. Keyed by (canonical address rendering,
    /// width) → the destination name of the first identical load;
    /// cleared on every store and on havoc.
    load_memo: HashMap<(String, u16), String>,
}

impl TextMemory {
    fn new(ptr_bits: u16) -> Self {
        Self {
            stores: Vec::new(),
            havoced: false,
            load_counter: 0,
            ptr_bits,
            load_memo: HashMap::new(),
        }
    }

    fn byte_addr(&self, addr: &str, i: u16) -> String {
        if i == 0 {
            addr.to_string()
        } else {
            format!("(bvadd {addr} (_ bv{i} {bits}))", bits = self.ptr_bits)
        }
    }

    fn record_store(&mut self, address: &Expr, value: &Expr, bits: u16, ctx: &mut RenderCtx) {
        if bits == 0 {
            return;
        }
        // Any store may alias a memoized load; drop the memo.
        self.load_memo.clear();
        let nbytes = usize::from(bits.div_ceil(8));
        if self.stores.len().saturating_add(nbytes) > MEM_BYTE_STORE_CAP {
            self.stores.clear();
            self.havoced = true;
            return;
        }
        let addr = render_expr_with_width(address, self.ptr_bits, ctx);
        let value_s = render_expr_with_width(value, bits, ctx);
        for i in 0..u16::try_from(nbytes).unwrap_or(u16::MAX) {
            let byte_addr = self.byte_addr(&addr, i);
            let lo = u32::from(i) * 8;
            let hi = lo + 7;
            let byte = format!("((_ extract {hi} {lo}) {value_s})");
            self.stores.push((byte_addr, byte));
        }
    }

    fn emit_load(
        &mut self,
        out: &mut String,
        dst: &r2smt_ir::expr::Var,
        address: &Expr,
        bits: u16,
        ctx: &mut RenderCtx,
    ) {
        if bits == 0 {
            return;
        }
        // Read-read consistency: reuse a prior identical load's
        // destination when no store has invalidated the memo since.
        let memo_key = (render_expr_with_width(address, self.ptr_bits, ctx), bits);
        if let Some(cached) = self.load_memo.get(&memo_key) {
            ctx.flush_into(out);
            let _ = writeln!(
                out,
                "(assert (= {lhs} {cached}))",
                lhs = smt_symbol(&dst.name)
            );
            return;
        }
        let nbytes = bits.div_ceil(8);
        let addr = render_expr_with_width(address, self.ptr_bits, ctx);
        let load_id = self.load_counter;
        self.load_counter = self.load_counter.wrapping_add(1);
        let mut acc: Option<String> = None;
        for i in 0..nbytes {
            let byte_addr = self.byte_addr(&addr, i);
            let fresh = format!("load_{load_id}_{i}");
            declare_bv(out, &fresh, 8, ctx);
            let mut byte_val = fresh;
            if !self.havoced {
                for (store_addr, store_byte) in &self.stores {
                    byte_val =
                        format!("(ite (= {byte_addr} {store_addr}) {store_byte} {byte_val})");
                }
            }
            acc = Some(match acc.take() {
                None => byte_val,
                Some(prev) => format!("(concat {byte_val} {prev})"),
            });
        }
        let Some(loaded) = acc else {
            return;
        };
        let total_bits = nbytes.saturating_mul(8);
        let coerced = coerce(&loaded, total_bits, bits);
        ctx.flush_into(out);
        let _ = writeln!(
            out,
            "(assert (= {lhs} {coerced}))",
            lhs = smt_symbol(&dst.name)
        );
        self.load_memo
            .insert(memo_key, smt_symbol(&dst.name).into_owned());
    }
}

/// Whether any expression rendered into the script carries a
/// floating-point node, which decides the declared logic.
fn slice_uses_fp(slice: &SsaLiftedSlice) -> bool {
    if expr_uses_fp(&slice.condition) {
        return true;
    }
    slice.statements.iter().any(|stmt| match stmt {
        IrStmt::Assign { src, .. } => expr_uses_fp(src),
        IrStmt::StoreMem { address, value, .. } => expr_uses_fp(address) || expr_uses_fp(value),
        IrStmt::LoadMem { address, .. } => expr_uses_fp(address),
        IrStmt::Unsupported { .. } | IrStmt::Nop => false,
    })
}

/// Whether `expr` contains any floating-point node. Written without a
/// wildcard arm so a new `Expr` variant forces this to be revisited.
// Exhaustive `Expr` dispatch: FP arms mirror the BV arms' shape but stay separate for legibility (CLAUDE.md exhaustive-dispatch exception).
#[allow(clippy::match_same_arms, clippy::too_many_lines)]
fn expr_uses_fp(expr: &Expr) -> bool {
    match expr {
        Expr::FAdd(..)
        | Expr::FSub(..)
        | Expr::FMul(..)
        | Expr::FDiv(..)
        | Expr::FEq(..)
        | Expr::FLt(..)
        | Expr::FLe(..)
        | Expr::FIsNaN(_)
        | Expr::FSqrt(..)
        | Expr::FRoundToIntegral(..)
        | Expr::FpConst { .. }
        | Expr::BvToFp { .. }
        | Expr::FpToIeeeBv(_)
        | Expr::FpToSbv { .. }
        | Expr::SbvToFp { .. }
        | Expr::FpToFp { .. } => true,
        Expr::Var(_) | Expr::Const { .. } | Expr::Unknown(_) => false,
        Expr::BoolNot(a)
        | Expr::Extract { src: a, .. }
        | Expr::ZeroExtend { src: a, .. }
        | Expr::SignExtend { src: a, .. } => expr_uses_fp(a),
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::UDiv(a, b)
        | Expr::URem(a, b)
        | Expr::SDiv(a, b)
        | Expr::SRem(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Shl(a, b)
        | Expr::LShr(a, b)
        | Expr::AShr(a, b)
        | Expr::Ror(a, b)
        | Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Ult(a, b)
        | Expr::Ule(a, b)
        | Expr::Slt(a, b)
        | Expr::Sle(a, b)
        | Expr::BoolAnd(a, b)
        | Expr::BoolOr(a, b)
        | Expr::Concat { high: a, low: b } => expr_uses_fp(a) || expr_uses_fp(b),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => expr_uses_fp(cond) || expr_uses_fp(then_expr) || expr_uses_fp(else_expr),
    }
}

/// Smallest exponent and significand widths SMT-LIB admits for a
/// `FloatingPoint` sort — the theory requires both to exceed 1.
const MIN_FP_SORT_FIELD: u16 = 2;

/// IEEE-754 binary32 `(exponent, significand)` sort.
const IEEE_SINGLE: (u16, u16) = (8, 24);
/// IEEE-754 binary64 `(exponent, significand)` sort.
const IEEE_DOUBLE: (u16, u16) = (11, 53);

/// Whether a floating-point sort is one the renderer can emit for a
/// *verdict*. Two independent constraints, both load-bearing.
///
/// The sort must be well formed — both fields wider than one bit, total
/// width not overflowing — or the emitted text is nonsense.
///
/// It must also be binary32 or binary64. cvc5 1.3.4 rejects any other
/// width outright ("only Float32 (8/24) or Float64 (11/53) types are
/// supported in default mode"), offering `--fp-exp`, which its own
/// message calls experimental and "use at your own risk". A
/// soundness-critical verdict path should not depend on an experimental
/// solver mode, so other sorts decline here and are left to the Z3
/// backend, which handles them natively.
///
/// x87's double-extended sort `(15, 64)` is one of the sorts this
/// declines, which is what makes x87 a Z3-only capability. Note the
/// reason is only cvc5's: the sort itself is well formed, and its
/// pattern width really is `ebits + sbits` — the extra explicit integer
/// bit of the *stored* 80-bit image is bridged in the lifter
/// (`r2smt-slicer`'s `lift/x87.rs`), outside every width computation
/// here and in the Z3 encoder.
///
/// binary128 `(15, 113)` is the second such sort, and the measurement
/// says the same thing as x87's rather than something new: a minimal
/// query naming `(_ FloatingPoint 15 113)` is `sat` under Z3 4.13.0 and
/// under Bitwuzla 0.9.1, and cvc5 1.3.4 answers with the same
/// "only Float32 (8/24) or Float64 (11/53) types are supported in
/// default mode" error. It reaches this pipeline as the *intermediate*
/// a binary64 fused multiply step is computed in — never as a lane — so
/// declining costs the text backends precision on that one family and
/// nothing else. Bitwuzla would take it, but a sort only two of the
/// three portfolio backends parse would make them disagree by parse
/// error rather than by verdict, which is the lesson of the rotate.
///
/// That family stopped being hypothetical when the `AArch64` `.2d` and
/// `AArch32` `vfma.f64` fused steps were opened: any slice carrying one
/// is Z3-only from here on, which the two `cvc5` / `bitwuzla` decline
/// contracts pin. The cost is bounded because the sort stays an
/// intermediate — no *lane* is ever binary128, so a slice declines only
/// when it actually contains a fused step over binary64.
const fn is_renderable_fp_sort(ebits: u16, sbits: u16) -> bool {
    ebits >= MIN_FP_SORT_FIELD
        && sbits >= MIN_FP_SORT_FIELD
        && ebits.checked_add(sbits).is_some()
        && matches!((ebits, sbits), IEEE_SINGLE | IEEE_DOUBLE)
}

/// A rendered floating-point term together with its sort. Mirrors the
/// Z3 encoder's `Encoded::Fp`: the textual renderer's `(String, u16)`
/// pair cannot express an FP sort, and an FP term is never a bit
/// vector of `ebits + sbits` bits — it is a distinct SMT-LIB sort.
struct FpTerm {
    text: String,
    ebits: u16,
    sbits: u16,
}

impl FpTerm {
    /// Total width of the IEEE bit pattern this sort reinterprets.
    fn bv_width(&self) -> u16 {
        self.ebits.saturating_add(self.sbits)
    }
}

/// SMT-LIB spelling of a rounding mode, mirroring the Z3 encoder's
/// `to_z3_rm`.
const fn rm_smtlib(rm: RoundingMode) -> &'static str {
    match rm {
        RoundingMode::NearestTiesEven => "RNE",
        RoundingMode::NearestTiesAway => "RNA",
        RoundingMode::TowardPositive => "RTP",
        RoundingMode::TowardNegative => "RTN",
        RoundingMode::TowardZero => "RTZ",
    }
}

/// Render an FP-sorted expression, or `None` when the expression is not
/// FP-sorted or carries a shape the renderer refuses.
///
/// Written without a wildcard arm so a new `Expr` variant forces this
/// to be revisited rather than silently treated as non-FP.
// Exhaustive `Expr` dispatch: every non-FP-producing variant is listed
// explicitly so the variant set stays in lockstep with the renderer
// (CLAUDE.md exhaustive-dispatch exception).
#[allow(clippy::match_same_arms)]
fn render_fp(expr: &Expr, ctx: &mut RenderCtx) -> Option<FpTerm> {
    match expr {
        Expr::FpConst { bits, ebits, sbits } => {
            let term = fp_sort_checked(*ebits, *sbits, ctx)?;
            let width = term.0;
            Some(FpTerm {
                text: format!("((_ to_fp {ebits} {sbits}) (_ bv{bits} {width}))"),
                ebits: *ebits,
                sbits: *sbits,
            })
        }
        Expr::BvToFp { src, ebits, sbits } => {
            let (width, _) = fp_sort_checked(*ebits, *sbits, ctx)?;
            let (text, bits) = render_expr(src, ctx);
            if bits != width {
                ctx.note_unsupported(format!(
                    "bit-pattern reinterpret of a {bits}-bit value into a {width}-bit sort"
                ));
                return None;
            }
            Some(FpTerm {
                text: format!("((_ to_fp {ebits} {sbits}) {text})"),
                ebits: *ebits,
                sbits: *sbits,
            })
        }
        Expr::SbvToFp {
            src,
            rm,
            ebits,
            sbits,
        } => {
            fp_sort_checked(*ebits, *sbits, ctx)?;
            let (text, _) = render_expr(src, ctx);
            Some(FpTerm {
                text: format!(
                    "((_ to_fp {ebits} {sbits}) {rm} {text})",
                    rm = rm_smtlib(*rm)
                ),
                ebits: *ebits,
                sbits: *sbits,
            })
        }
        Expr::FpToFp {
            src,
            rm,
            ebits,
            sbits,
        } => render_fp_to_fp(src, *rm, (*ebits, *sbits), ctx),
        Expr::FSqrt(a, rm) => render_fp_sqrt(a, *rm, ctx),
        Expr::FRoundToIntegral(a, rm) => render_fp_round_to_integral(a, *rm, ctx),
        Expr::FAdd(a, b, rm) => fp_arith("fp.add", a, b, *rm, ctx),
        Expr::FSub(a, b, rm) => fp_arith("fp.sub", a, b, *rm, ctx),
        Expr::FMul(a, b, rm) => fp_arith("fp.mul", a, b, *rm, ctx),
        Expr::FDiv(a, b, rm) => fp_arith("fp.div", a, b, *rm, ctx),
        // Everything below is bit-vector sorted, including the bridges
        // *out* of FP (`FpToIeeeBv`, `FpToSbv`) and the FP predicates,
        // which render as 1-bit bit vectors.
        Expr::Var(_)
        | Expr::Const { .. }
        | Expr::Unknown(_)
        | Expr::Add(..)
        | Expr::Sub(..)
        | Expr::Mul(..)
        | Expr::UDiv(..)
        | Expr::URem(..)
        | Expr::SDiv(..)
        | Expr::SRem(..)
        | Expr::And(..)
        | Expr::Or(..)
        | Expr::Xor(..)
        | Expr::Shl(..)
        | Expr::LShr(..)
        | Expr::AShr(..)
        | Expr::Ror(..)
        | Expr::Eq(..)
        | Expr::Ne(..)
        | Expr::Ult(..)
        | Expr::Ule(..)
        | Expr::Slt(..)
        | Expr::Sle(..)
        | Expr::BoolAnd(..)
        | Expr::BoolOr(..)
        | Expr::BoolNot(_)
        | Expr::Ite { .. }
        | Expr::Extract { .. }
        | Expr::Concat { .. }
        | Expr::ZeroExtend { .. }
        | Expr::SignExtend { .. }
        | Expr::FEq(..)
        | Expr::FLt(..)
        | Expr::FLe(..)
        | Expr::FIsNaN(_)
        | Expr::FpToIeeeBv(_)
        | Expr::FpToSbv { .. } => None,
    }
}

/// Square root, which stays in its operand's sort.
fn render_fp_sqrt(src: &Expr, rm: RoundingMode, ctx: &mut RenderCtx) -> Option<FpTerm> {
    let inner = render_fp(src, ctx)?;
    Some(FpTerm {
        text: format!(
            "(fp.sqrt {rm} {inner})",
            rm = rm_smtlib(rm),
            inner = inner.text
        ),
        ebits: inner.ebits,
        sbits: inner.sbits,
    })
}

/// Round a float to an integral value, keeping its sort.
///
/// `fp.roundToIntegral` is standard SMT-LIB and needs no decline, which
/// was measured rather than assumed: this function's own output, for
/// every one of the five rounding modes, is `unsat` on the negated
/// polarity in Z3 4.13.0, CVC5 1.3.4 and Bitwuzla 0.9.1 alike — so all
/// three parse the operator *and* agree on its value. That is the
/// opposite of the rotate, whose `bvror` spelling only one of the three
/// accepted, and why this renders where [`render_rotate`] declines.
fn render_fp_round_to_integral(
    src: &Expr,
    rm: RoundingMode,
    ctx: &mut RenderCtx,
) -> Option<FpTerm> {
    let inner = render_fp(src, ctx)?;
    Some(FpTerm {
        text: format!(
            "(fp.roundToIntegral {rm} {inner})",
            rm = rm_smtlib(rm),
            inner = inner.text
        ),
        ebits: inner.ebits,
        sbits: inner.sbits,
    })
}

/// Convert a float to another float sort. Shares SMT-LIB's `to_fp`
/// spelling with the signed-integer conversion; the argument's sort
/// disambiguates them.
fn render_fp_to_fp(
    src: &Expr,
    rm: RoundingMode,
    sort: (u16, u16),
    ctx: &mut RenderCtx,
) -> Option<FpTerm> {
    let (ebits, sbits) = sort;
    fp_sort_checked(ebits, sbits, ctx)?;
    let inner = render_fp(src, ctx)?;
    Some(FpTerm {
        text: format!(
            "((_ to_fp {ebits} {sbits}) {rm} {inner})",
            rm = rm_smtlib(rm),
            inner = inner.text
        ),
        ebits,
        sbits,
    })
}

/// Validate an FP sort, returning its IEEE bit-pattern width. Declines
/// (recording the reason) for sorts outside [`is_renderable_fp_sort`].
fn fp_sort_checked(ebits: u16, sbits: u16, ctx: &mut RenderCtx) -> Option<(u16, u16)> {
    if !is_renderable_fp_sort(ebits, sbits) {
        ctx.note_unsupported(format!("invalid sort (_ FloatingPoint {ebits} {sbits})"));
        return None;
    }
    Some((ebits.saturating_add(sbits), sbits))
}

fn fp_arith(
    name: &str,
    a: &Expr,
    b: &Expr,
    rm: RoundingMode,
    ctx: &mut RenderCtx,
) -> Option<FpTerm> {
    let lhs = render_fp(a, ctx)?;
    let rhs = render_fp(b, ctx)?;
    if (lhs.ebits, lhs.sbits) != (rhs.ebits, rhs.sbits) {
        ctx.note_unsupported("floating-point arithmetic mixing two sorts");
        return None;
    }
    Some(FpTerm {
        text: format!(
            "({name} {rm} {l} {r})",
            rm = rm_smtlib(rm),
            l = lhs.text,
            r = rhs.text
        ),
        ebits: lhs.ebits,
        sbits: lhs.sbits,
    })
}

/// Reinterpret a float as its IEEE bit pattern.
///
/// `fp.to_ieee_bv` is a Z3-only extension — CVC5 and Bitwuzla reject it
/// — so the direction is inverted instead: mint a fresh bit vector and
/// constrain the float to be *its* reinterpretation, using only the
/// standard `(_ to_fp e s)` form.
///
/// The equality must be structural `=`, not `fp.eq`, which conflates
/// `+0` with `-0` and makes `NaN` differ from itself. Under `=` the
/// inversion is exact except for `NaN`: SMT-LIB has a single NaN value,
/// so every NaN bit pattern satisfies the constraint. That is an
/// over-approximation, which can only widen an always-true / always-
/// false verdict to both-possible, never fabricate one.
fn fp_bridge_to_bv(src: &Expr, ctx: &mut RenderCtx) -> (String, u16) {
    let Some(term) = render_fp(src, ctx) else {
        ctx.note_unsupported("bit-pattern reinterpret of a term with no floating-point sort");
        return (UNRENDERABLE_PLACEHOLDER.to_string(), 1);
    };
    let width = term.bv_width();
    let name = ctx.fresh_ieee_name();
    ctx.aux
        .push(format!("(declare-fun {name} () (_ BitVec {width}))"));
    ctx.aux.push(format!(
        "(assert (= {term} ((_ to_fp {e} {s}) {name})))",
        term = term.text,
        e = term.ebits,
        s = term.sbits
    ));
    (name, width)
}

/// Render an FP predicate as the 1-bit bit vector the rest of the
/// renderer expects, matching `bool_op`'s convention.
fn fp_predicate(name: &str, operands: [&Expr; 2], ctx: &mut RenderCtx) -> (String, u16) {
    let (Some(lhs), Some(rhs)) = (render_fp(operands[0], ctx), render_fp(operands[1], ctx)) else {
        ctx.note_unsupported(format!("{name} over a term with no floating-point sort"));
        return (UNRENDERABLE_PLACEHOLDER.to_string(), 1);
    };
    if (lhs.ebits, lhs.sbits) != (rhs.ebits, rhs.sbits) {
        ctx.note_unsupported(format!("{name} comparing two floating-point sorts"));
        return (UNRENDERABLE_PLACEHOLDER.to_string(), 1);
    }
    (
        format!("(ite ({name} {l} {r}) #b1 #b0)", l = lhs.text, r = rhs.text),
        1,
    )
}

/// Stand-in emitted where the renderer had to refuse. It only keeps the
/// script parseable for render-only consumers; a refusal is recorded on
/// the context, and the verdict paths decline before solving.
const UNRENDERABLE_PLACEHOLDER: &str = "(_ bv0 1)";

/// Characters an SMT-LIB *simple* symbol may contain besides letters
/// and digits (SMT-LIB 2.6 §3.1).
const SYMBOL_EXTRA_CHARS: [char; 17] = [
    '~', '!', '@', '$', '%', '^', '&', '*', '_', '-', '+', '=', '<', '>', '.', '?', '/',
];

/// Render `name` as an SMT-LIB symbol, quoting it when it is not a
/// legal simple symbol.
///
/// Load-bearing for SSA names: the rename pass suffixes every
/// definition with `#N`, and `#` is not a simple-symbol character — it
/// opens a bit-vector literal, so an unquoted `zmm0#0` is a parse error
/// in every solver.
fn smt_symbol(name: &str) -> Cow<'_, str> {
    let simple = !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || SYMBOL_EXTRA_CHARS.contains(&c));
    if simple {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("|{name}|"))
    }
}

/// Declare a bit-vector symbol once, refusing a second declaration at a
/// different width.
///
/// SMT-LIB has no sound answer for one name at two sorts, so the slice is
/// declined rather than rendered — [`emit_query_strict`] turns the note
/// into an `Err` and the verdict path falls back to `Inconclusive`,
/// matching the Z3 encoder's `width_conflict`.
fn declare_bv(out: &mut String, name: &str, bits: u16, ctx: &mut RenderCtx) {
    if let Some(&have) = ctx.declared.get(name) {
        if have != bits {
            ctx.note_unsupported(format!("symbol {name} declared at {have} and {bits} bits"));
        }
        return;
    }
    ctx.declared.insert(name.to_string(), bits);
    let _ = writeln!(
        out,
        "(declare-fun {name} () (_ BitVec {bits}))",
        name = smt_symbol(name)
    );
}

/// Render an expression to SMT-LIB2 text, widening / narrowing to
/// the target bit width before returning. Used at every assertion
/// boundary so the produced SMT-LIB stays well-typed.
fn render_expr_with_width(expr: &Expr, target_bits: u16, ctx: &mut RenderCtx) -> String {
    let (rendered, bits) = render_expr(expr, ctx);
    coerce(&rendered, bits, target_bits)
}

// `clippy::too_many_lines`: exhaustive dispatch over `Expr`'s variants —
// each arm is a 1-3 line emitter; splitting per-arm would hide the parity
// between the variant set and the SMT-LIB renderer without removing any
// logic.
// FP arms mirror the BV arms' shape but stay separate for legibility.
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
fn render_expr(expr: &Expr, ctx: &mut RenderCtx) -> (String, u16) {
    match expr {
        // A use at a width other than the declaration's would render the
        // symbol at `v.bits` while the script declares another sort — the
        // mismatch the declaration side cannot see, because a use need
        // not be preceded by its own `declare_bv` call.
        Expr::Var(v) => {
            if let Some(&have) = ctx.declared.get(&v.name)
                && have != v.bits
            {
                ctx.note_unsupported(format!(
                    "symbol {name} declared at {have} bits and read at {read}",
                    name = v.name,
                    read = v.bits
                ));
            }
            (smt_symbol(&v.name).into_owned(), v.bits)
        }
        Expr::Const { value, bits } => (format!("(_ bv{value} {bits})"), *bits),
        Expr::Add(a, b) => bin_op("bvadd", a, b, Signedness::Unsigned, ctx),
        Expr::Sub(a, b) => bin_op("bvsub", a, b, Signedness::Unsigned, ctx),
        Expr::Mul(a, b) => bin_op("bvmul", a, b, Signedness::Unsigned, ctx),
        Expr::UDiv(a, b) => bin_op("bvudiv", a, b, Signedness::Unsigned, ctx),
        Expr::URem(a, b) => bin_op("bvurem", a, b, Signedness::Unsigned, ctx),
        Expr::SDiv(a, b) => bin_op("bvsdiv", a, b, Signedness::Signed, ctx),
        Expr::SRem(a, b) => bin_op("bvsrem", a, b, Signedness::Signed, ctx),
        Expr::Ror(a, b) => render_rotate(a, b, ctx),
        Expr::And(a, b) => bin_op("bvand", a, b, Signedness::Unsigned, ctx),
        Expr::Or(a, b) => bin_op("bvor", a, b, Signedness::Unsigned, ctx),
        Expr::Xor(a, b) => bin_op("bvxor", a, b, Signedness::Unsigned, ctx),
        Expr::Shl(a, b) => bin_op("bvshl", a, b, Signedness::Unsigned, ctx),
        Expr::LShr(a, b) => bin_op("bvlshr", a, b, Signedness::Unsigned, ctx),
        Expr::AShr(a, b) => bin_op("bvashr", a, b, Signedness::Unsigned, ctx),
        Expr::Eq(a, b) => bool_op("=", a, b, Signedness::Unsigned, ctx),
        Expr::Ne(a, b) => bool_op("distinct", a, b, Signedness::Unsigned, ctx),
        Expr::Ult(a, b) => bool_op("bvult", a, b, Signedness::Unsigned, ctx),
        Expr::Ule(a, b) => bool_op("bvule", a, b, Signedness::Unsigned, ctx),
        Expr::Slt(a, b) => bool_op("bvslt", a, b, Signedness::Signed, ctx),
        Expr::Sle(a, b) => bool_op("bvsle", a, b, Signedness::Signed, ctx),
        Expr::BoolAnd(a, b) => bool_combiner("and", a, b, ctx),
        Expr::BoolOr(a, b) => bool_combiner("or", a, b, ctx),
        Expr::BoolNot(inner) => {
            let r = bool_of(inner, ctx);
            (format!("(ite (not {r}) #b1 #b0)"), 1)
        }
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond_str = bool_of(cond, ctx);
            let (then_str, then_bits) = render_expr(then_expr, ctx);
            let (else_str, else_bits) = render_expr(else_expr, ctx);
            let target = then_bits.max(else_bits);
            let then_coerced = coerce(&then_str, then_bits, target);
            let else_coerced = coerce(&else_str, else_bits, target);
            (
                format!("(ite {cond_str} {then_coerced} {else_coerced})"),
                target,
            )
        }
        Expr::Extract { src, hi, lo } => {
            let (src_str, src_bits) = render_expr(src, ctx);
            let width = hi.saturating_sub(*lo).saturating_add(1);
            if *hi >= src_bits || lo > hi {
                // Mirror the Z3 encoder's `width_conflict`: an out-of-range
                // slice is a contract violation, not a renderable term. It
                // is reachable — the differential harness pairs two
                // lowerings of one instruction that need not agree about a
                // value's width, so a 64-bit slice can land on a 16-bit
                // operand. Note it so `emit_query_strict` declines (a sound
                // refusal), and emit a well-typed placeholder so the lossy
                // render-only path stays parseable.
                ctx.note_unsupported(format!(
                    "extract [{hi}:{lo}] out of range for a {src_bits}-bit term"
                ));
                return (format!("(_ bv0 {width})"), width);
            }
            (format!("((_ extract {hi} {lo}) {src_str})"), width)
        }
        Expr::Concat { high, low } => {
            let (high_str, hb) = render_expr(high, ctx);
            let (low_str, lb) = render_expr(low, ctx);
            (format!("(concat {high_str} {low_str})"), hb + lb)
        }
        Expr::ZeroExtend { src, to_bits } => {
            let (src_str, cur) = render_expr(src, ctx);
            (coerce(&src_str, cur, *to_bits), *to_bits)
        }
        Expr::SignExtend { src, to_bits } => {
            let (src_str, cur) = render_expr(src, ctx);
            match cur.cmp(to_bits) {
                std::cmp::Ordering::Equal => (src_str, *to_bits),
                std::cmp::Ordering::Less => (
                    format!("((_ sign_extend {n}) {src_str})", n = to_bits - cur),
                    *to_bits,
                ),
                std::cmp::Ordering::Greater => {
                    let hi = to_bits - 1;
                    (format!("((_ extract {hi} 0) {src_str})"), *to_bits)
                }
            }
        }
        // `Unknown` has no sound text encoding here: a constant under-
        // approximates the value set (unlike the Z3 backend's fresh
        // free var). This constant placeholder only keeps the script
        // parseable for render-only consumers; the CVC5 *verdict* path
        // (`crate::cvc5::solve_branch_cvc5`) declines any slice that
        // contains an `Unknown` before this is ever solved, so no
        // verdict is derived from this placeholder.
        Expr::Unknown(_) => (UNRENDERABLE_PLACEHOLDER.to_string(), 1),
        Expr::FpToIeeeBv(src) => fp_bridge_to_bv(src, ctx),
        // An FP-sorted node reached directly in bit-vector position.
        // The lifter always wraps its results in `FpToIeeeBv`, but
        // hand-built IR need not, and the bridge is precise either way.
        Expr::FAdd(..)
        | Expr::FSub(..)
        | Expr::FMul(..)
        | Expr::FDiv(..)
        | Expr::FpConst { .. }
        | Expr::BvToFp { .. }
        | Expr::SbvToFp { .. }
        | Expr::FpToFp { .. }
        | Expr::FSqrt(..)
        | Expr::FRoundToIntegral(..) => fp_bridge_to_bv(expr, ctx),
        Expr::FpToSbv { src, rm, bits } => {
            let Some(f) = render_fp(src, ctx) else {
                ctx.note_unsupported("float-to-signed conversion of a term with no float sort");
                return (format!("(_ bv0 {bits})"), *bits);
            };
            (
                format!(
                    "((_ fp.to_sbv {bits}) {rm} {f})",
                    rm = rm_smtlib(*rm),
                    f = f.text
                ),
                *bits,
            )
        }
        Expr::FEq(a, b) => fp_predicate("fp.eq", [a, b], ctx),
        Expr::FLt(a, b) => fp_predicate("fp.lt", [a, b], ctx),
        Expr::FLe(a, b) => fp_predicate("fp.leq", [a, b], ctx),
        Expr::FIsNaN(a) => {
            let Some(f) = render_fp(a, ctx) else {
                ctx.note_unsupported("fp.isNaN over a term with no floating-point sort");
                return (UNRENDERABLE_PLACEHOLDER.to_string(), 1);
            };
            (format!("(ite (fp.isNaN {f}) #b1 #b0)", f = f.text), 1)
        }
    }
}

fn bin_op(name: &str, a: &Expr, b: &Expr, sign: Signedness, ctx: &mut RenderCtx) -> (String, u16) {
    let (a_str, a_bits) = render_expr(a, ctx);
    let (b_str, b_bits) = render_expr(b, ctx);
    let target = a_bits.max(b_bits);
    let lhs = coerce_with_sign(&a_str, a_bits, target, sign);
    let rhs = coerce_with_sign(&b_str, b_bits, target, sign);
    (format!("({name} {lhs} {rhs})"), target)
}

/// Render a rotate-right.
///
/// SMT-LIB spells only the **constant**-amount form,
/// `((_ rotate_right n) x)`. The variable-amount extension is not
/// portable and this was checked rather than assumed: `bvror` over two
/// terms is accepted by Bitwuzla, and rejected as an unknown constant
/// by both CVC5 and Z3's own parser. So a non-constant amount declines
/// here, which `emit_query_strict` turns into a sound refusal rather
/// than a wrong answer.
///
/// The declined case *is* reachable from the lifter: a register-controlled
/// rotate (`ror r0, r1, r2`, `rors`, and the `..., ror rN` shifted-operand
/// form) lowers to an `Expr::Ror` whose amount is a masked register read,
/// not a constant. Z3 handles it (`Z3_mk_ext_rotate_right` takes a variable
/// amount), so the text backends' soundness here rests on this decline
/// being load-bearing — not on the lifter never emitting one. Relaxing it
/// on the latter (false) assumption would make CVC5 / Bitwuzla mis-render a
/// variable rotate.
fn render_rotate(a: &Expr, b: &Expr, ctx: &mut RenderCtx) -> (String, u16) {
    let (a_str, a_bits) = render_expr(a, ctx);
    let Expr::Const { value, .. } = b else {
        ctx.note_unsupported("rotate by a non-constant amount");
        return (a_str, a_bits);
    };
    // The rotate is modulo the width, which is what the architecture
    // means and what keeps a large immediate from being a parse error.
    let amount = if a_bits == 0 {
        0
    } else {
        value % u128::from(a_bits)
    };
    (format!("((_ rotate_right {amount}) {a_str})"), a_bits)
}

fn bool_op(name: &str, a: &Expr, b: &Expr, sign: Signedness, ctx: &mut RenderCtx) -> (String, u16) {
    let (a_str, a_bits) = render_expr(a, ctx);
    let (b_str, b_bits) = render_expr(b, ctx);
    let target = a_bits.max(b_bits);
    let lhs = coerce_with_sign(&a_str, a_bits, target, sign);
    let rhs = coerce_with_sign(&b_str, b_bits, target, sign);
    // Wrap the SMT boolean back into a 1-bit BV so the encoder can
    // splice the result wherever a bit-vector is expected.
    (format!("(ite ({name} {lhs} {rhs}) #b1 #b0)"), 1)
}

/// Whether a binary operation interprets its operands as signed or
/// unsigned when one of them needs to be widened to match the other.
#[derive(Debug, Clone, Copy)]
enum Signedness {
    Signed,
    Unsigned,
}

fn coerce_with_sign(rendered: &str, cur: u16, target: u16, sign: Signedness) -> String {
    if cur >= target {
        return coerce(rendered, cur, target);
    }
    let n = target - cur;
    match sign {
        Signedness::Signed => format!("((_ sign_extend {n}) {rendered})"),
        Signedness::Unsigned => format!("((_ zero_extend {n}) {rendered})"),
    }
}

fn bool_combiner(name: &str, a: &Expr, b: &Expr, ctx: &mut RenderCtx) -> (String, u16) {
    let a_bool = bool_of(a, ctx);
    let b_bool = bool_of(b, ctx);
    (format!("(ite ({name} {a_bool} {b_bool}) #b1 #b0)"), 1)
}

fn bool_of(expr: &Expr, ctx: &mut RenderCtx) -> String {
    let (rendered, bits) = render_expr(expr, ctx);
    if bits == 1 {
        format!("(= {rendered} #b1)")
    } else {
        format!("(distinct {rendered} (_ bv0 {bits}))")
    }
}

fn coerce(rendered: &str, cur: u16, target: u16) -> String {
    if cur == target {
        return rendered.to_string();
    }
    if cur < target {
        let n = target - cur;
        return format!("((_ zero_extend {n}) {rendered})");
    }
    let hi = target - 1;
    format!("((_ extract {hi} 0) {rendered})")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use r2smt_common::smt::SolveOptions;
    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
    use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
    use r2smt_ssa::ssa_convert;

    use super::*;

    fn op(raw: &str, kind: OperandKind) -> Operand {
        Operand {
            raw: raw.into(),
            kind,
        }
    }

    fn insn(addr: u64, size: u8, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            address: Address(addr),
            size,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands,
            esil: None,
            pcode: None,
            is_thumb: false,
        }
    }

    fn build_ssa(program: &Program) -> SsaLiftedSlice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("branch");
        let slice = slice_branch(
            cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        );
        let lifted = lift_slice(&slice, program.arch);
        ssa_convert(&lifted)
    }

    #[test]
    fn smtlib_query_for_constant_propagation_has_check_sat_line() {
        // mov eax, 1 ; cmp eax, 1 ; je dest — same fixture used by
        // the Z3 backend tests, but here we only verify the SMT-LIB
        // script structure (logic, declarations, check-sat).
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        insn(
                            0x40_1000,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1005,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                        ),
                        insn(
                            0x40_1008,
                            6,
                            "je",
                            vec![op("0x401080", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        };
        let ssa = build_ssa(&program);
        let script = emit_query(&ssa, &SolveOptions::default(), true);
        assert!(script.starts_with("(set-logic QF_BV)"));
        assert!(script.contains("(check-sat)"));
        assert!(script.contains("(declare-fun "));
        assert!(script.contains("(assert ("));
    }

    #[test]
    fn smtlib_slice_without_float_declares_no_auxiliary_lines() {
        // A bit-vector-only slice must round-trip through the renderer
        // leaving no auxiliary declaration behind: the FP inversion is
        // the only producer of those, so an empty `aux` here is what
        // keeps non-FP output byte-identical.
        let statements = vec![IrStmt::Assign {
            dst: r2smt_ir::expr::Var::new("t0", 32),
            src: Expr::add(Expr::var("x", 32), Expr::konst(1, 32)),
        }];
        let mut ctx = RenderCtx::new();
        let _ = emit_preamble_with(&mem_slice(statements), &SolveOptions::default(), &mut ctx);
        assert!(ctx.aux.is_empty());
    }

    /// IEEE-754 binary32 bit patterns used by the floating-point tests.
    const F32_ONE: u128 = 0x3f80_0000;
    const F32_TWO: u128 = 0x4000_0000;
    const F32_TWO_AND_A_HALF: u128 = 0x4020_0000;
    const F32_THREE: u128 = 0x4040_0000;
    const F32_MINUS_TWO: u128 = 0xc000_0000;
    const F32_MINUS_TWO_AND_A_HALF: u128 = 0xc020_0000;
    const F32_MINUS_THREE: u128 = 0xc040_0000;
    const F32_QUIET_NAN: u128 = 0x7fc0_0000;

    fn f32_const(bits: u128) -> Expr {
        Expr::FpConst {
            bits,
            ebits: 8,
            sbits: 24,
        }
    }

    /// A slice whose single statement defines `t0` from `src`, with
    /// `condition` as the branch predicate.
    fn fp_slice(src: Expr, condition: Expr) -> SsaLiftedSlice {
        let mut slice = mem_slice(vec![IrStmt::Assign {
            dst: r2smt_ir::expr::Var::new("t0", 32),
            src,
        }]);
        slice.condition = condition;
        slice
    }

    /// Solve an emitted script with the already-linked Z3, so the text
    /// backend's *own output* is checked rather than only its shape.
    fn z3_check(script: &str) -> z3::SatResult {
        let solver = z3::Solver::new();
        solver.from_string(script);
        solver.check()
    }

    /// Solve a round-to-integral of one binary32 constant against the
    /// emitted text, on the *negated* polarity: `Unsat` means the
    /// rendering forces exactly `expected`, so the value is checked and
    /// not only the shape.
    fn round_to_integral_check(input: u128, rm: RoundingMode, expected: u128) -> z3::SatResult {
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(Expr::fround_to_integral(f32_const(input), rm)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(expected, 32)),
        );
        let script = emit_query_strict(&slice, &SolveOptions::default(), false)
            .expect("round-to-integral of a binary32 constant must render");
        z3_check(&script)
    }

    #[test]
    fn smtlib_scalar_fp_add_emits_qf_bvfp_logic_and_fp_add() {
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(Expr::fadd(
                f32_const(F32_ONE),
                f32_const(F32_ONE),
                RoundingMode::NearestTiesEven,
            )),
            Expr::eq(Expr::var("t0", 32), Expr::konst(F32_TWO, 32)),
        );
        let script = emit_query(&slice, &SolveOptions::default(), true);
        assert!(script.starts_with("(set-logic QF_BVFP)\n"), "{script}");
        assert!(script.contains("(fp.add RNE "), "{script}");
    }

    #[test]
    fn smtlib_fp_to_ieee_bv_inverts_through_standard_to_fp() {
        // `fp.to_ieee_bv` is a Z3-only extension that CVC5 and Bitwuzla
        // reject, so the reinterpret must invert through the standard
        // one-argument `to_fp` against a freshly declared bit vector.
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(f32_const(F32_ONE)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(F32_ONE, 32)),
        );
        let script = emit_query(&slice, &SolveOptions::default(), true);
        assert!(!script.contains("fp.to_ieee_bv"), "{script}");
        assert!(
            script.contains("(declare-fun __fpbv_0 () (_ BitVec 32))"),
            "{script}"
        );
        assert!(script.contains("((_ to_fp 8 24) __fpbv_0)))"), "{script}");
    }

    #[test]
    fn smtlib_fp_inversion_declares_bridge_before_the_assert_using_it() {
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(f32_const(F32_ONE)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(F32_ONE, 32)),
        );
        let script = emit_query(&slice, &SolveOptions::default(), true);
        let declared = script
            .find("(declare-fun __fpbv_0")
            .expect("bridge declaration");
        let used = script.find("(assert (= t0 __fpbv_0))").expect("bridge use");
        assert!(declared < used, "{script}");
    }

    #[test]
    fn smtlib_fp_inversion_on_nan_keeps_both_polarities_satisfiable() {
        // The soundness contract for the inversion. SMT-LIB collapses
        // every NaN bit pattern onto one NaN value, so constraining the
        // float leaves the bridge variable free across all of them —
        // an over-approximation. Both polarities must stay satisfiable:
        // the earlier constant placeholder pinned the value and would
        // fabricate an always-false verdict here.
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(f32_const(F32_QUIET_NAN)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(F32_QUIET_NAN, 32)),
        );
        let taken = z3_check(&emit_query(&slice, &SolveOptions::default(), true));
        let not_taken = z3_check(&emit_query(&slice, &SolveOptions::default(), false));
        assert_eq!((taken, not_taken), (z3::SatResult::Sat, z3::SatResult::Sat));
    }

    #[test]
    fn smtlib_fp_inversion_still_proves_an_exact_sum_always_true() {
        // The companion to the NaN test: over-approximating NaN must
        // not cost precision on ordinary values, or the renderer would
        // be sound but useless. `1.0f + 1.0f == 2.0f` stays provable.
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(Expr::fadd(
                f32_const(F32_ONE),
                f32_const(F32_ONE),
                RoundingMode::NearestTiesEven,
            )),
            Expr::eq(Expr::var("t0", 32), Expr::konst(F32_TWO, 32)),
        );
        let taken = z3_check(&emit_query(&slice, &SolveOptions::default(), true));
        let not_taken = z3_check(&emit_query(&slice, &SolveOptions::default(), false));
        assert_eq!(
            (taken, not_taken),
            (z3::SatResult::Sat, z3::SatResult::Unsat)
        );
    }

    /// `pxor xmm0, xmm0` then a self-compare then `branch`: every
    /// variable in the resulting slice carries an SSA `#N` suffix, and
    /// the flags are fully determined without a memory operand.
    fn zeroed_self_compare_program(branch: &str) -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        insn(
                            0x40_1000,
                            4,
                            "pxor",
                            vec![
                                op("xmm0", OperandKind::Register),
                                op("xmm0", OperandKind::Register),
                            ],
                        ),
                        insn(
                            0x40_1004,
                            4,
                            "ucomiss",
                            vec![
                                op("xmm0", OperandKind::Register),
                                op("xmm0", OperandKind::Register),
                            ],
                        ),
                        insn(
                            0x40_1008,
                            6,
                            branch,
                            vec![op("0x401080", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    #[test]
    fn smtlib_quotes_ssa_symbols_carrying_a_version_suffix() {
        // The SSA rename suffixes definitions with `#N`, and `#` opens a
        // bit-vector literal in SMT-LIB rather than continuing a symbol.
        // Unquoted, every solver rejects the script at parse time.
        let script = emit_query(
            &build_ssa(&zeroed_self_compare_program("jp")),
            &SolveOptions::default(),
            true,
        );
        assert!(script.contains("(declare-fun |zmm0#0| ()"), "{script}");
    }

    #[test]
    fn smtlib_ssa_renamed_slice_solves_to_the_same_verdict_as_the_z3_backend() {
        // Parses *and* decides: a script whose symbols were rejected
        // would leave the solver with no assertions, making both
        // polarities satisfiable instead of proving the branch dead.
        let slice = build_ssa(&zeroed_self_compare_program("jp"));
        let taken = z3_check(&emit_query(&slice, &SolveOptions::default(), true));
        let not_taken = z3_check(&emit_query(&slice, &SolveOptions::default(), false));
        assert_eq!(
            (taken, not_taken),
            (z3::SatResult::Unsat, z3::SatResult::Sat)
        );
    }

    #[test]
    fn smtlib_declines_a_sort_with_an_out_of_range_field_width() {
        // The decline path had no coverage at all before this. An
        // exponent narrower than the theory allows must fail the strict
        // emitter rather than render an unparseable sort.
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(Expr::FpConst {
                bits: 0,
                ebits: 1,
                sbits: 24,
            }),
            Expr::eq(Expr::var("t0", 32), Expr::konst(0, 32)),
        );
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    fn smtlib_declines_half_precision_rather_than_needing_an_experimental_solver() {
        // Measured against cvc5 1.3.4: a real binary16 query is
        // rejected unless `--fp-exp` is passed, and cvc5 documents that
        // mode as experimental. The Z3 backend handles the sort
        // natively, so the text path declines instead.
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(Expr::FpConst {
                bits: 0x3c00,
                ebits: 5,
                sbits: 11,
            }),
            Expr::eq(Expr::var("t0", 32), Expr::konst(0x3c00, 32)),
        );
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    fn smtlib_declines_an_out_of_range_extract() {
        // An extract whose high index exceeds the source width is a
        // contract violation, reachable via the differential harness
        // pairing two lowerings that disagree on a value's width. The
        // strict emitter must decline it — matching the Z3 encoder's
        // `width_conflict` — rather than emit an ill-typed
        // `((_ extract 63 0) <32-bit term>)` the solver rejects as a
        // parse error.
        let slice = fp_slice(
            Expr::konst(0, 32),
            Expr::eq(
                Expr::extract(Expr::var("t0", 32), 63, 0),
                Expr::konst(0, 64),
            ),
        );
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    #[ignore = "writes a script for manual replay against cvc5 / bitwuzla"]
    fn smtlib_writes_fp_script_for_external_solver_replay() {
        // Opt-in: the default suite must not depend on a solver binary
        // being installed, but the emitted text is what CVC5 and
        // Bitwuzla actually parse, so keep a way to hand it to them:
        //   cargo test -p r2smt-smt --lib -- --ignored \
        //     smtlib_writes_fp_script_for_external_solver_replay
        //   cvc5 --lang smt2 < target/fp-replay-taken.smt2
        //   bitwuzla < target/fp-replay-not-taken.smt2
        // Built by the real pipeline rather than by hand, so the script
        // exercises what the lifter actually emits — the scalar compare
        // pulls in fp.eq, fp.lt and fp.isNaN alongside the reinterpret,
        // over SSA-renamed symbols.
        let slice = build_ssa(&zeroed_self_compare_program("jp"));
        // A unit test runs with the crate directory as its working
        // directory, so reach the workspace target dir explicitly.
        let target = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
        for (polarity, name) in [(true, "taken"), (false, "not-taken")] {
            let script = emit_query(&slice, &SolveOptions::default(), polarity);
            std::fs::write(target.join(format!("fp-replay-{name}.smt2")), script)
                .expect("write replay script");
        }
    }

    fn mem_slice(statements: Vec<IrStmt>) -> SsaLiftedSlice {
        use r2smt_slicer::{BranchCandidate, BranchCondition, BranchKind, SliceStatus};
        let z = Address::new(0x1000);
        SsaLiftedSlice {
            branch: BranchCandidate {
                address: z,
                function: z,
                block: z,
                kind: BranchKind::Jcc,
                mnemonic: "memtest".into(),
                condition: BranchCondition::NotEqual,
                formula: "memtest".into(),
                taken_target: None,
                fallthrough_target: None,
                compare_register: None,
                bit_index: None,
                upstream_resolved: None,
                operand_raws: Vec::new(),
                is_thumb: false,
            },
            statements,
            condition: Expr::konst(0, 1),
            status: SliceStatus::Complete,
            treat_truncation_as_inputs: false,
            inputs: Vec::new(),
            defs: Vec::new(),
            arch: Arch::Aarch64,
        }
    }

    #[test]
    fn smtlib_multibit_condition_tests_nonzero_matching_the_z3_encoder() {
        // A wider value in boolean position is true iff nonzero, which is
        // what the Z3 encoder's `bv_is_true` does. The prior condition
        // path coerced to bit 0 and compared `#b1`, so the two backends
        // could return opposite verdicts on such a value. No lifter
        // produces one today (conditions are 1-bit comparisons), so this
        // locks the alignment against a future one.
        use r2smt_ir::expr::Var;
        let mut ssa = mem_slice(vec![]);
        ssa.condition = Expr::Var(Var::new("t0", 32));
        let script = emit_query(&ssa, &SolveOptions::default(), true);
        assert!(
            script.contains("(distinct t0 (_ bv0 32))"),
            "a multi-bit condition must test nonzero, not bit 0: {script}"
        );
        assert!(
            !script.contains("(_ extract 0 0)"),
            "the condition must not be coerced to its low bit: {script}"
        );
    }

    #[test]
    fn smtlib_store_then_load_emits_byte_granular_ite_chain() {
        // A store to `sp` followed by a load from `sp` must constrain
        // the loaded value via the alias `ite` chain — the same
        // precision the Z3 backend has, not a free variable.
        use r2smt_ir::expr::Var;
        let sp = Expr::Var(Var::new("sp", 64));
        let ssa = mem_slice(vec![
            IrStmt::StoreMem {
                address: sp.clone(),
                value: Expr::konst(5, 64),
                bits: 64,
            },
            IrStmt::LoadMem {
                dst: Var::new("t0", 64),
                address: sp,
                bits: 64,
            },
        ]);
        let script = emit_preamble(&ssa, &SolveOptions::default());
        assert!(
            script.contains("(ite "),
            "expected alias ite chain: {script}"
        );
        assert!(
            script.contains("(concat "),
            "expected byte concat: {script}"
        );
    }

    #[test]
    fn smtlib_load_without_prior_store_stays_a_free_value() {
        // No store → the load reads a fresh free byte (no ite chain).
        use r2smt_ir::expr::Var;
        let ssa = mem_slice(vec![IrStmt::LoadMem {
            dst: Var::new("t0", 64),
            address: Expr::Var(Var::new("sp", 64)),
            bits: 64,
        }]);
        let script = emit_preamble(&ssa, &SolveOptions::default());
        assert!(
            !script.contains("(ite "),
            "a load with no prior store must not build an ite chain: {script}"
        );
    }

    #[test]
    fn smtlib_renders_a_constant_rotate_as_the_portable_indexed_form() {
        // SMT-LIB spells only `((_ rotate_right n) x)`. Checked against
        // the real solvers rather than assumed: `bvror` over two terms
        // is accepted by Bitwuzla and rejected as an unknown constant by
        // both CVC5 and Z3, so emitting it would have made the portfolio
        // backends disagree by parse error.
        let slice = fp_slice(
            Expr::ror(Expr::var("a", 32), Expr::konst(4, 32)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(0, 32)),
        );
        let script = emit_query_strict(&slice, &SolveOptions::default(), true)
            .expect("a constant rotate must render");
        assert!(
            script.contains("(_ rotate_right 4)"),
            "expected the indexed form: {script}"
        );
        assert!(!script.contains("bvror"), "{script}");
    }

    #[test]
    fn smtlib_rotate_amount_is_taken_modulo_the_width() {
        // A rotate by the width is the identity, and `(_ rotate_right
        // 32)` on a 32-bit term is not something every parser accepts.
        let slice = fp_slice(
            Expr::ror(Expr::var("a", 32), Expr::konst(36, 32)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(0, 32)),
        );
        let script = emit_query_strict(&slice, &SolveOptions::default(), true)
            .expect("a constant rotate must render");
        assert!(
            script.contains("(_ rotate_right 4)"),
            "36 mod 32 is 4: {script}"
        );
    }

    #[test]
    fn smtlib_declines_a_rotate_by_a_non_constant_amount() {
        // The variable-amount rotate has no portable spelling, so the
        // strict emitter refuses rather than emitting something one
        // backend parses and two do not.
        let slice = fp_slice(
            Expr::ror(Expr::var("a", 32), Expr::var("n", 32)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(0, 32)),
        );
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    fn smtlib_constant_rotate_solves_to_the_architectural_value() {
        // Solved with the linked Z3 against the emitted text, so the
        // rendering is checked by value and not only by shape:
        // `0x12345678` rotated right by 8 is `0x78123456`.
        let slice = fp_slice(
            Expr::ror(Expr::konst(0x1234_5678, 32), Expr::konst(8, 32)),
            Expr::eq(Expr::var("t0", 32), Expr::konst(0x7812_3456, 32)),
        );
        let script = emit_query_strict(&slice, &SolveOptions::default(), false)
            .expect("a constant rotate must render");
        // Unsat on the negated polarity means the equality is necessary.
        assert_eq!(z3_check(&script), z3::SatResult::Unsat, "{script}");
    }

    #[test]
    fn smtlib_renders_round_to_integral_as_the_standard_fp_operator() {
        // Unlike the rotate, this one needs no decline: the scripts this
        // renderer emits were replayed through Z3 4.13.0, CVC5 1.3.4 and
        // Bitwuzla 0.9.1, and all three returned `unsat` on the negated
        // polarity for all five rounding modes.
        let slice = fp_slice(
            Expr::fp_to_ieee_bv(Expr::fround_to_integral(
                f32_const(F32_TWO_AND_A_HALF),
                RoundingMode::TowardZero,
            )),
            Expr::eq(Expr::var("t0", 32), Expr::konst(F32_TWO, 32)),
        );
        let script = emit_query_strict(&slice, &SolveOptions::default(), true)
            .expect("round-to-integral of a binary32 constant must render");
        assert!(
            script.contains("(fp.roundToIntegral RTZ "),
            "expected the standard operator carrying its mode: {script}"
        );
    }

    #[test]
    fn smtlib_round_to_integral_nearest_ties_even_takes_the_even_neighbour() {
        // 2.5 → 2.0, where ties-away would give 3.0.
        assert_eq!(
            round_to_integral_check(F32_TWO_AND_A_HALF, RoundingMode::NearestTiesEven, F32_TWO),
            z3::SatResult::Unsat
        );
    }

    #[test]
    fn smtlib_round_to_integral_nearest_ties_away_leaves_the_even_neighbour() {
        // 2.5 → 3.0, where ties-even would give 2.0.
        assert_eq!(
            round_to_integral_check(F32_TWO_AND_A_HALF, RoundingMode::NearestTiesAway, F32_THREE),
            z3::SatResult::Unsat
        );
    }

    #[test]
    fn smtlib_round_to_integral_toward_positive_rounds_up() {
        // 2.5 → 3.0, which no nearest mode reaches from below.
        assert_eq!(
            round_to_integral_check(F32_TWO_AND_A_HALF, RoundingMode::TowardPositive, F32_THREE),
            z3::SatResult::Unsat
        );
    }

    #[test]
    fn smtlib_round_to_integral_toward_negative_rounds_down_below_zero() {
        // -2.5 → -3.0, where toward-zero would give -2.0. Signed so the
        // two directional modes cannot be confused for each other.
        assert_eq!(
            round_to_integral_check(
                F32_MINUS_TWO_AND_A_HALF,
                RoundingMode::TowardNegative,
                F32_MINUS_THREE
            ),
            z3::SatResult::Unsat
        );
    }

    #[test]
    fn smtlib_round_to_integral_toward_zero_truncates_below_zero() {
        // -2.5 → -2.0, where toward-negative would give -3.0.
        assert_eq!(
            round_to_integral_check(
                F32_MINUS_TWO_AND_A_HALF,
                RoundingMode::TowardZero,
                F32_MINUS_TWO
            ),
            z3::SatResult::Unsat
        );
    }

    /// IEEE-754 binary128, the format a binary64 fused multiply step is
    /// computed in so that it rounds once.
    const IEEE_QUAD: (u16, u16) = (15, 113);

    /// The binary64 fixture that separates one rounding from two.
    ///
    /// `(1 + 2⁻⁵²)²` is exactly `1 + 2⁻⁵¹ + 2⁻¹⁰⁴`. That low term sits
    /// far below half an ulp of binary64, so a product rounded on its
    /// own discards it; adding `−(1 + 2⁻⁵¹)` then leaves exactly
    /// `2⁻¹⁰⁴` when the whole step rounds once, and exactly zero when it
    /// rounds twice.
    const F64_ONE_PLUS_ULP: u128 = 0x3ff0_0000_0000_0001;
    const F64_MINUS_ONE_PLUS_TWO_ULP: u128 = 0xbff0_0000_0000_0002;
    const F64_TWO_POW_MINUS_104: u128 = 0x3970_0000_0000_0000;
    const F64_ZERO: u128 = 0;

    /// The lowering `r2smt-slicer`'s `fused_multiply_lane` builds for one
    /// lane: both sources and the accumulator widened to `wide`, one
    /// multiply, one add, and a single rounding back to binary64.
    ///
    /// Restated here rather than called because that helper is private
    /// to the lifter, and what is under test is what a solver does with
    /// the sorts rather than the helper's plumbing. Passing `IEEE_DOUBLE`
    /// as the intermediate degenerates the widening to the identity and
    /// so yields the *double-rounded* lowering, which is what makes the
    /// two spellings comparable on one fixture.
    fn multiply_add_via(wide: (u16, u16), a: u128, b: u128, acc: u128) -> Expr {
        let rm = RoundingMode::NearestTiesEven;
        let widen = |bits: u128| {
            Expr::fp_to_fp(
                Expr::bv_to_fp(Expr::konst(bits, 64), IEEE_DOUBLE.0, IEEE_DOUBLE.1),
                rm,
                wide.0,
                wide.1,
            )
        };
        let product = Expr::fmul(widen(a), widen(b), rm);
        let sum = Expr::fadd(widen(acc), product, rm);
        Expr::fp_to_ieee_bv(Expr::fp_to_fp(sum, rm, IEEE_DOUBLE.0, IEEE_DOUBLE.1))
    }

    /// The fused step of the fixture above, computed in `wide`.
    fn fused_fixture(wide: (u16, u16)) -> Expr {
        multiply_add_via(
            wide,
            F64_ONE_PLUS_ULP,
            F64_ONE_PLUS_ULP,
            F64_MINUS_ONE_PLUS_TWO_ULP,
        )
    }

    /// A slice whose single statement defines a 64-bit `t0` from `src`,
    /// with `t0 == expected` as the branch predicate.
    fn f64_slice(src: Expr, expected: u128) -> SsaLiftedSlice {
        let mut slice = mem_slice(vec![IrStmt::Assign {
            dst: r2smt_ir::expr::Var::new("t0", 64),
            src,
        }]);
        slice.condition = Expr::eq(Expr::var("t0", 64), Expr::konst(expected, 64));
        slice
    }

    /// Budget for the floating-point solves below. Well above the
    /// production default, because `cargo test --all` runs every crate's
    /// binary concurrently and a 113-bit significand is not free.
    fn quad_solve_options() -> SolveOptions {
        SolveOptions {
            timeout_ms: 120_000,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn binary128_intermediate_makes_a_binary64_fused_step_round_once() {
        // `AlwaysTrue` means the encoding forces exactly 2⁻¹⁰⁴, which is
        // the architecture's answer for a single rounding.
        let slice = f64_slice(fused_fixture(IEEE_QUAD), F64_TWO_POW_MINUS_104);
        assert_eq!(
            crate::solver::solve_branch(&slice, quad_solve_options()),
            r2smt_common::smt::SmtResult::AlwaysTrue
        );
    }

    #[test]
    fn binary64_intermediate_rounds_the_same_step_twice_and_gives_zero() {
        // The teeth of the test above: on the same inputs the
        // double-rounded lowering is forced to zero, so a test that
        // passed for both spellings would be measuring nothing.
        let slice = f64_slice(fused_fixture(IEEE_DOUBLE), F64_ZERO);
        assert_eq!(
            crate::solver::solve_branch(&slice, quad_solve_options()),
            r2smt_common::smt::SmtResult::AlwaysTrue
        );
    }

    #[test]
    fn smtlib_declines_the_binary128_intermediate_of_a_fused_step() {
        // Measured against the three installed binaries, not assumed:
        // `(_ FloatingPoint 15 113)` is accepted by Z3 4.13.0 and by
        // Bitwuzla 0.9.1, and rejected by CVC5 1.3.4 — "only Float32
        // (8/24) or Float64 (11/53) types are supported in default
        // mode". So the portable text path refuses the sort exactly as
        // it refuses x87's `(15, 64)`, and `emit_query_strict` turns
        // that into a decline rather than a wrong verdict.
        let slice = f64_slice(fused_fixture(IEEE_QUAD), F64_TWO_POW_MINUS_104);
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    fn smtlib_renders_the_same_fused_step_shape_at_a_binary64_intermediate() {
        // Teeth for the decline above: the identical expression shape
        // renders once its sorts are ones cvc5 admits, so the refusal is
        // about the sort and not about the widen-multiply-add shape.
        let slice = f64_slice(fused_fixture(IEEE_DOUBLE), F64_ZERO);
        assert!(emit_query_strict(&slice, &SolveOptions::default(), true).is_ok());
    }

    /// A slice whose only free input declares `sp` at 64 bits.
    fn multi_width_slice(statements: Vec<IrStmt>) -> SsaLiftedSlice {
        let mut slice = mem_slice(statements);
        slice.inputs = vec![r2smt_ir::expr::Var::new("sp", 64)];
        slice
    }

    #[test]
    fn smtlib_refuses_a_symbol_defined_at_a_second_width() {
        // SMT-LIB has no sound answer for one name at two sorts. The
        // renderer used to dedup declarations by name alone and drop the
        // second silently, emitting an ill-sorted script the solver
        // rejected as a parse error — so the text backends and Z3
        // disagreed by *how they failed* rather than by verdict.
        let slice = multi_width_slice(vec![IrStmt::Assign {
            dst: r2smt_ir::expr::Var::new("sp", 16),
            src: Expr::konst(1, 16),
        }]);
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    fn smtlib_refuses_a_symbol_read_at_a_width_other_than_its_declaration() {
        // The half the declaration site cannot see: a *use* need not be
        // preceded by its own `declare_bv`, so the mismatch only shows up
        // when the expression renderer reports the symbol at `v.bits`.
        let slice = multi_width_slice(vec![IrStmt::Assign {
            dst: r2smt_ir::expr::Var::new("t", 16),
            src: Expr::var("sp", 16),
        }]);
        assert!(matches!(
            emit_query_strict(&slice, &SolveOptions::default(), true),
            Err(RenderError::Unrenderable(_))
        ));
    }

    #[test]
    fn smtlib_renders_a_single_width_symbol_unchanged() {
        // Teeth for both refusals: the same shapes render once every
        // occurrence of the name agrees on one width, so the refusal is
        // about the collision and not about slicing a register at all.
        let slice = multi_width_slice(vec![IrStmt::Assign {
            dst: r2smt_ir::expr::Var::new("t", 16),
            src: Expr::extract(Expr::var("sp", 64), 15, 0),
        }]);
        assert!(emit_query_strict(&slice, &SolveOptions::default(), true).is_ok());
    }
}
