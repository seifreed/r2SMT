//! Audited FFI shim for the two Z3 floating-point *reinterpret*
//! constructors that the safe `z3` 0.20 API does not expose:
//! bit-vector → float (`Z3_mk_fpa_to_fp_bv`) and signed-integer → float
//! (`Z3_mk_fpa_to_fp_signed`). Both are needed to bridge x86 SSE — where
//! an `xmm` lane is a raw bit-vector — into Z3's floating-point theory.
//!
//! The rest of the FP API (`add_with_rounding_mode`, `eq_fpa`,
//! `to_ieee_bv`, …) is already safe in `z3` 0.20; only these two
//! reinterprets require `unsafe`, and they are isolated here behind a
//! safe, `Option`-returning surface so callers never touch a raw handle
//! and degrade gracefully (a `None` becomes a sound free value upstream).

use z3::Sort;
use z3::ast::{Ast, BV, Float, RoundingMode};
use z3_sys::{Z3_mk_fpa_to_fp_bv, Z3_mk_fpa_to_fp_signed};

/// Reinterpret the bits of `bv` as an IEEE-754 float of sort
/// `(ebits, sbits)` — SMT-LIB `((_ to_fp ebits sbits) bv)`. The
/// bit-vector width must equal `ebits + sbits`. Returns `None` if Z3
/// rejects the construction (e.g. a width mismatch).
#[must_use]
pub fn bv_to_fp(bv: &BV, ebits: u32, sbits: u32) -> Option<Float> {
    let ctx = bv.get_ctx();
    let sort = Sort::float(ebits, sbits);
    // SAFETY: `ctx` is the context that owns `bv` (obtained from
    // `bv.get_ctx()`), and `Sort::float` builds `sort` in that same
    // thread-local context. `bv.get_z3_ast()` and `sort.get_z3_sort()`
    // are live, correctly-typed handles into `ctx`. `Z3_mk_fpa_to_fp_bv`
    // returns a fresh, Z3-ref-counted ast (or null → `None`), which
    // `Float::wrap` adopts under the same context.
    #[allow(unsafe_code)]
    let raw =
        unsafe { Z3_mk_fpa_to_fp_bv(ctx.get_z3_context(), bv.get_z3_ast(), sort.get_z3_sort()) }?;
    // SAFETY: `raw` is a valid floating-point ast created in `ctx`.
    #[allow(unsafe_code)]
    let fp = unsafe { Float::wrap(ctx, raw) };
    Some(fp)
}

/// Convert a two's-complement signed integer bit-vector `bv` to an
/// IEEE-754 float of sort `(ebits, sbits)` under rounding mode `rm` —
/// SMT-LIB `((_ to_fp ebits sbits) rm bv)`. Returns `None` if Z3 rejects
/// the construction.
#[must_use]
pub fn sbv_to_fp(bv: &BV, rm: &RoundingMode, ebits: u32, sbits: u32) -> Option<Float> {
    let ctx = bv.get_ctx();
    let sort = Sort::float(ebits, sbits);
    // SAFETY: `ctx` owns `bv` and `rm`; `sort` is built in the same
    // thread-local context. All three handles are live and correctly
    // typed. `Z3_mk_fpa_to_fp_signed` returns a fresh ref-counted ast
    // (or null → `None`) adopted by `Float::wrap` under `ctx`.
    #[allow(unsafe_code)]
    let raw = unsafe {
        Z3_mk_fpa_to_fp_signed(
            ctx.get_z3_context(),
            rm.get_z3_ast(),
            bv.get_z3_ast(),
            sort.get_z3_sort(),
        )
    }?;
    // SAFETY: `raw` is a valid floating-point ast created in `ctx`.
    #[allow(unsafe_code)]
    let fp = unsafe { Float::wrap(ctx, raw) };
    Some(fp)
}
