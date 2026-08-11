//! Operand-aware helpers for the `AArch64` `cset` / `csel` / `csinc` /
//! `csinv` / `csneg` family.
//!
//! These pseudo-instructions all derive their result from a flag
//! predicate plus one or two general-purpose registers. When r2SMT
//! proves the predicate constant, the `cs*` instruction collapses to
//! a deterministic data-movement form. The helpers in this module
//! synthesize the equivalent textual assembly so [`BytePatcher::assemble`]
//! can encode it (see ARM ARM Vol. C §C6.2 for the canonical
//! reference encodings):
//!
//! - `cset`    → `mov  Rd, #imm`          (`csinc Rd, RZR, RZR, !cond`)
//! - `csetm`   → `mov  Rd, #imm`          (`csinv Rd, RZR, RZR, !cond`)
//! - `csel`    → `mov  Rd, Rn` or `Rm`
//! - `csinc`   → `mov  Rd, Rn` or `add Rd, Rm, #1`
//! - `csinv`   → `mov  Rd, Rn` or `mvn Rd, Rm`
//! - `csneg`   → `mov  Rd, Rn` or `neg Rd, Rm`
//!
//! The functions intentionally avoid hand-encoding instruction bytes:
//! delegating to `r2`'s assembler (or to the in-memory test double's
//! `add_assemble`) keeps the encoding path identical to the
//! `replace_jcc_with_jmp` strategy and avoids reimplementing the
//! Armv8 `MOV (register)` / `ADD (immediate)` aliases by hand.
//!
//! [`BytePatcher::assemble`]: r2smt_ir::byte_patcher::BytePatcher::assemble

/// Canonical "wide" zero register name. Returned by [`parse_xreg`]
/// when the operand text is one of `xzr` / `wzr`.
pub const ZERO_REGISTER_X: &str = "xzr";

/// Canonical "wide" zero register name for 32-bit views.
pub const ZERO_REGISTER_W: &str = "wzr";

/// Parse a raw operand string into a canonical `AArch64` GPR name.
///
/// Accepts the textual forms r2 emits (`x0`, `w0`, `xzr`, `wzr`,
/// optional whitespace, optional `,` from the operand splitter) and
/// returns the lower-cased canonical name. Returns `None` for the
/// stack-pointer aliases (`sp`, `wsp`) — patching them is unsafe in
/// the general case (function epilogues, frame setup), and for any
/// other shape (SIMD registers, memory operands, immediates).
#[must_use]
pub fn parse_xreg(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches(',').trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == ZERO_REGISTER_X || lower == ZERO_REGISTER_W {
        return Some(lower);
    }
    if let Some(rest) = lower.strip_prefix('x').or_else(|| lower.strip_prefix('w'))
        && let Ok(n) = rest.parse::<u8>()
        && n <= 30
    {
        return Some(lower);
    }
    None
}

/// Assemble syntax for `mov Rd, Rs` (the `AArch64` register-move alias).
#[must_use]
pub fn mov_reg(dst: &str, src: &str) -> String {
    format!("mov {dst}, {src}")
}

/// Assemble syntax for `mov Rd, #imm` (a `MOVZ` / `MOVN` alias).
#[must_use]
pub fn mov_imm(dst: &str, imm: i64) -> String {
    if imm < 0 {
        format!("mov {dst}, #-{abs}", abs = imm.unsigned_abs())
    } else {
        format!("mov {dst}, #{imm}")
    }
}

/// Assemble syntax for `mvn Rd, Rs` (bitwise NOT — `ORN Rd, RZR, Rs`).
#[must_use]
pub fn mvn_reg(dst: &str, src: &str) -> String {
    format!("mvn {dst}, {src}")
}

/// Assemble syntax for `neg Rd, Rs` (two's-complement negation —
/// `SUB Rd, RZR, Rs`).
#[must_use]
pub fn neg_reg(dst: &str, src: &str) -> String {
    format!("neg {dst}, {src}")
}

/// Assemble syntax for `add Rd, Rs, #imm`.
#[must_use]
pub fn add_imm(dst: &str, src: &str, imm: i64) -> String {
    if imm < 0 {
        format!("sub {dst}, {src}, #{abs}", abs = imm.unsigned_abs())
    } else {
        format!("add {dst}, {src}, #{imm}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_xreg_accepts_x_and_w_views() {
        assert_eq!(parse_xreg("x0").as_deref(), Some("x0"));
        assert_eq!(parse_xreg("w15").as_deref(), Some("w15"));
        assert_eq!(parse_xreg("x30").as_deref(), Some("x30"));
    }

    #[test]
    fn parse_xreg_accepts_zero_registers() {
        assert_eq!(parse_xreg("xzr").as_deref(), Some("xzr"));
        assert_eq!(parse_xreg("WZR").as_deref(), Some("wzr"));
    }

    #[test]
    fn parse_xreg_strips_trailing_comma_and_whitespace() {
        assert_eq!(parse_xreg(" x0 , ").as_deref(), Some("x0"));
    }

    #[test]
    fn parse_xreg_rejects_stack_pointer() {
        assert!(parse_xreg("sp").is_none());
        assert!(parse_xreg("wsp").is_none());
    }

    #[test]
    fn parse_xreg_rejects_out_of_range_indices() {
        assert!(parse_xreg("x31").is_none());
        assert!(parse_xreg("w99").is_none());
    }

    #[test]
    fn parse_xreg_rejects_simd_and_memory_operands() {
        assert!(parse_xreg("v0").is_none());
        assert!(parse_xreg("q3").is_none());
        assert!(parse_xreg("[x0]").is_none());
        assert!(parse_xreg("#1").is_none());
    }

    #[test]
    fn mov_imm_handles_positive_zero_and_negative() {
        assert_eq!(mov_imm("x0", 0), "mov x0, #0");
        assert_eq!(mov_imm("x0", 1), "mov x0, #1");
        assert_eq!(mov_imm("x0", -1), "mov x0, #-1");
    }

    #[test]
    fn add_imm_falls_through_to_sub_on_negative() {
        assert_eq!(add_imm("x0", "x1", 4), "add x0, x1, #4");
        assert_eq!(add_imm("x0", "x1", -4), "sub x0, x1, #4");
    }

    #[test]
    fn register_alias_helpers_emit_canonical_aliases() {
        assert_eq!(mov_reg("x0", "x1"), "mov x0, x1");
        assert_eq!(mvn_reg("x0", "x1"), "mvn x0, x1");
        assert_eq!(neg_reg("x0", "x1"), "neg x0, x1");
    }
}
