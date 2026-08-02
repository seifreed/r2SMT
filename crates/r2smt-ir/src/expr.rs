//! Symbolic expression language consumed by the lifter, SSA pass, and
//! SMT backend.
//!
//! `Expr` is a small bit-vector and boolean algebra:
//!
//! - bit-vector arithmetic (`Add`, `Sub`, `Mul`),
//! - bit-vector logic (`And`, `Or`, `Xor`, `Shl`, `LShr`, `AShr`),
//! - bit-slice operators (`Extract`, `Concat`, `ZeroExtend`,
//!   `SignExtend`) used by the lifter to model sub-register reads and
//!   writes (`al`, `ah`, `ax`, `eax` on top of `rax`),
//! - comparisons that yield a 1-bit boolean (`Eq`, `Ne`, `Ult`, `Ule`,
//!   `Slt`, `Sle`),
//! - boolean connectives (`BoolAnd`, `BoolOr`, `BoolNot`),
//! - the conditional `Ite`,
//! - the escape hatch `Unknown` for behaviour the lifter cannot model
//!   yet.
//!
//! Variables carry a name and a bit width. Flags (`ZF`, `CF`, …) are
//! represented as variables of width 1.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A typed bit-vector variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Var {
    /// Variable name (e.g. `"rax"`, `"ZF"`, `"t0"`).
    pub name: String,
    /// Width in bits (`1` for flags, `8/16/32/64` for register slices,
    /// up to `512` for `zmm` vector registers).
    pub bits: u16,
}

impl Var {
    /// Construct a variable with a borrowed name.
    #[must_use]
    pub fn new(name: impl Into<String>, bits: u16) -> Self {
        Self {
            name: name.into(),
            bits,
        }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Bit-vector / boolean expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expr {
    /// Reference to a typed variable.
    Var(Var),
    /// Constant bit-vector value.
    Const {
        /// Unsigned representation (two's-complement for negatives).
        value: u128,
        /// Width in bits.
        bits: u16,
    },
    /// Bit-vector addition.
    Add(Box<Expr>, Box<Expr>),
    /// Bit-vector subtraction.
    Sub(Box<Expr>, Box<Expr>),
    /// Bit-vector multiplication.
    Mul(Box<Expr>, Box<Expr>),
    /// Bit-vector unsigned division. Operand width must match. Result
    /// width equals the operand width. Division by zero follows SMT-LIB
    /// bit-vector semantics (returns all-ones at the operand width).
    UDiv(Box<Expr>, Box<Expr>),
    /// Bit-vector unsigned remainder. Mirrors [`UDiv`] for operand
    /// width and divide-by-zero semantics.
    ///
    /// [`UDiv`]: Expr::UDiv
    URem(Box<Expr>, Box<Expr>),
    /// Bit-vector signed division (truncated towards zero).
    SDiv(Box<Expr>, Box<Expr>),
    /// Bit-vector signed remainder. Sign of the result matches the
    /// dividend.
    SRem(Box<Expr>, Box<Expr>),
    /// Bitwise AND.
    And(Box<Expr>, Box<Expr>),
    /// Bitwise OR.
    Or(Box<Expr>, Box<Expr>),
    /// Bitwise XOR.
    Xor(Box<Expr>, Box<Expr>),
    /// Logical left shift.
    Shl(Box<Expr>, Box<Expr>),
    /// Logical right shift.
    LShr(Box<Expr>, Box<Expr>),
    /// Arithmetic right shift.
    AShr(Box<Expr>, Box<Expr>),
    /// Bit-vector equality, yields a 1-bit value.
    Eq(Box<Expr>, Box<Expr>),
    /// Bit-vector inequality, yields a 1-bit value.
    Ne(Box<Expr>, Box<Expr>),
    /// Unsigned less-than.
    Ult(Box<Expr>, Box<Expr>),
    /// Unsigned less-than-or-equal.
    Ule(Box<Expr>, Box<Expr>),
    /// Signed less-than.
    Slt(Box<Expr>, Box<Expr>),
    /// Signed less-than-or-equal.
    Sle(Box<Expr>, Box<Expr>),
    /// Boolean conjunction (both operands must be 1-bit).
    BoolAnd(Box<Expr>, Box<Expr>),
    /// Boolean disjunction (both operands must be 1-bit).
    BoolOr(Box<Expr>, Box<Expr>),
    /// Boolean negation (operand must be 1-bit).
    BoolNot(Box<Expr>),
    /// If-then-else (`cond` is 1-bit; both branches share `bits`).
    Ite {
        /// 1-bit selector.
        cond: Box<Expr>,
        /// Value when `cond` is `1`.
        then_expr: Box<Expr>,
        /// Value when `cond` is `0`.
        else_expr: Box<Expr>,
    },
    /// Bit-slice extraction. Result width is `hi - lo + 1`. Bit indices
    /// are inclusive and zero-based from the least-significant bit, so
    /// `Extract(rax, 7, 0)` is the `al` byte and `Extract(rax, 15, 8)`
    /// is the `ah` byte.
    Extract {
        /// Source bit-vector.
        src: Box<Expr>,
        /// Inclusive high bit.
        hi: u16,
        /// Inclusive low bit.
        lo: u16,
    },
    /// Bit-vector concatenation: `high` placed above `low`. Result
    /// width is the sum of the operand widths.
    Concat {
        /// Bits placed at the most-significant positions.
        high: Box<Expr>,
        /// Bits placed at the least-significant positions.
        low: Box<Expr>,
    },
    /// Zero-extend `src` to `to_bits` total bits. `to_bits` must be
    /// strictly greater than the natural width of `src`.
    ZeroExtend {
        /// Source bit-vector.
        src: Box<Expr>,
        /// Target width.
        to_bits: u16,
    },
    /// Sign-extend `src` to `to_bits` total bits.
    SignExtend {
        /// Source bit-vector.
        src: Box<Expr>,
        /// Target width.
        to_bits: u16,
    },
    /// Something the lifter could not translate (operand we cannot
    /// parse, instruction with side effects we have not modelled, …).
    /// The string is a short, human-readable hint.
    Unknown(String),

    // --- Floating-point (IEEE-754) theory ---
    // FP values carry a `(ebits, sbits)` sort: `ebits` exponent bits and
    // `sbits` significand bits including the implicit leading bit. IEEE
    // single is `(8, 24)`, double `(11, 53)`. An FP expression's total
    // bit width (for coercion) is `ebits + sbits`.
    /// Floating-point addition under a rounding mode.
    FAdd(Box<Expr>, Box<Expr>, RoundingMode),
    /// Floating-point subtraction under a rounding mode.
    FSub(Box<Expr>, Box<Expr>, RoundingMode),
    /// Floating-point multiplication under a rounding mode.
    FMul(Box<Expr>, Box<Expr>, RoundingMode),
    /// Floating-point division under a rounding mode.
    FDiv(Box<Expr>, Box<Expr>, RoundingMode),
    /// IEEE floating-point equality (`NaN` ≠ `NaN`), yields a 1-bit value.
    FEq(Box<Expr>, Box<Expr>),
    /// IEEE floating-point less-than, yields a 1-bit value.
    FLt(Box<Expr>, Box<Expr>),
    /// IEEE floating-point less-than-or-equal, yields a 1-bit value.
    FLe(Box<Expr>, Box<Expr>),
    /// `true` (1-bit) when the operand is a `NaN`.
    FIsNaN(Box<Expr>),
    /// Floating-point square root under a rounding mode.
    FSqrt(Box<Expr>, RoundingMode),
    /// Floating-point constant, as its IEEE bit pattern under the sort.
    FpConst {
        /// IEEE bit pattern (width `ebits + sbits`).
        bits: u128,
        /// Exponent bit width.
        ebits: u16,
        /// Significand bit width (incl. implicit bit).
        sbits: u16,
    },
    /// Reinterpret a raw bit-vector as a float of the given sort
    /// (SMT-LIB `(_ to_fp e s)` applied to a bit-vector — a bit-pattern
    /// reinterpret, no rounding).
    BvToFp {
        /// Source bit-vector (width must equal `ebits + sbits`).
        src: Box<Expr>,
        /// Exponent bit width.
        ebits: u16,
        /// Significand bit width.
        sbits: u16,
    },
    /// Reinterpret a float as its IEEE bit-vector (width `ebits + sbits`).
    FpToIeeeBv(Box<Expr>),
    /// Convert a float to a signed integer bit-vector of `bits` width.
    FpToSbv {
        /// Source float.
        src: Box<Expr>,
        /// Rounding mode for the conversion.
        rm: RoundingMode,
        /// Result bit width.
        bits: u16,
    },
    /// Convert a signed integer bit-vector to a float of the given sort.
    SbvToFp {
        /// Source signed integer bit-vector.
        src: Box<Expr>,
        /// Rounding mode for the conversion.
        rm: RoundingMode,
        /// Exponent bit width.
        ebits: u16,
        /// Significand bit width.
        sbits: u16,
    },
    /// Convert a float to a different float sort, rounding when the
    /// target significand is narrower.
    ///
    /// Distinct from [`Expr::BvToFp`], which reinterprets a bit pattern
    /// and never rounds.
    FpToFp {
        /// Source float.
        src: Box<Expr>,
        /// Rounding mode for the conversion.
        rm: RoundingMode,
        /// Target exponent bit width.
        ebits: u16,
        /// Target significand bit width.
        sbits: u16,
    },
}

/// IEEE-754 rounding mode for floating-point operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundingMode {
    /// Round to nearest, ties to even — the IEEE default (`RNE`).
    NearestTiesEven,
    /// Round to nearest, ties away from zero (`RNA`).
    NearestTiesAway,
    /// Round toward positive infinity (`RTP`).
    TowardPositive,
    /// Round toward negative infinity (`RTN`).
    TowardNegative,
    /// Round toward zero (`RTZ`).
    TowardZero,
}

impl Expr {
    /// Construct a [`Expr::Var`].
    #[must_use]
    pub fn var(name: impl Into<String>, bits: u16) -> Self {
        Self::Var(Var::new(name, bits))
    }

    /// Construct a 1-bit flag variable.
    #[must_use]
    pub fn flag(name: impl Into<String>) -> Self {
        Self::var(name, 1)
    }

    /// Construct a typed constant.
    #[must_use]
    pub const fn konst(value: u128, bits: u16) -> Self {
        Self::Const { value, bits }
    }

    /// `Unknown` with no commentary.
    #[must_use]
    pub fn unknown() -> Self {
        Self::Unknown(String::new())
    }
}

/// Convenience binary-expression constructors.
///
/// The methods `add`, `sub`, `mul`, `shl` shadow names from
/// `std::ops::*`; we intentionally use the same names because they
/// produce IR nodes, not native integer arithmetic. Clippy's
/// `should_implement_trait` lint is suppressed at the impl level.
#[allow(clippy::should_implement_trait)]
impl Expr {
    /// `lhs + rhs` as a bit-vector.
    #[must_use]
    pub fn add(lhs: Self, rhs: Self) -> Self {
        Self::Add(Box::new(lhs), Box::new(rhs))
    }
    /// `lhs - rhs` as a bit-vector.
    #[must_use]
    pub fn sub(lhs: Self, rhs: Self) -> Self {
        Self::Sub(Box::new(lhs), Box::new(rhs))
    }
    /// `lhs * rhs` as a bit-vector.
    #[must_use]
    pub fn mul(lhs: Self, rhs: Self) -> Self {
        Self::Mul(Box::new(lhs), Box::new(rhs))
    }
    /// Unsigned bit-vector division (`lhs udiv rhs`).
    #[must_use]
    pub fn udiv(lhs: Self, rhs: Self) -> Self {
        Self::UDiv(Box::new(lhs), Box::new(rhs))
    }
    /// Unsigned bit-vector remainder (`lhs urem rhs`).
    #[must_use]
    pub fn urem(lhs: Self, rhs: Self) -> Self {
        Self::URem(Box::new(lhs), Box::new(rhs))
    }
    /// Signed bit-vector division (`lhs sdiv rhs`).
    #[must_use]
    pub fn sdiv(lhs: Self, rhs: Self) -> Self {
        Self::SDiv(Box::new(lhs), Box::new(rhs))
    }
    /// Signed bit-vector remainder (`lhs srem rhs`).
    #[must_use]
    pub fn srem(lhs: Self, rhs: Self) -> Self {
        Self::SRem(Box::new(lhs), Box::new(rhs))
    }
    /// Bitwise AND.
    #[must_use]
    pub fn bv_and(lhs: Self, rhs: Self) -> Self {
        Self::And(Box::new(lhs), Box::new(rhs))
    }
    /// Bitwise OR.
    #[must_use]
    pub fn bv_or(lhs: Self, rhs: Self) -> Self {
        Self::Or(Box::new(lhs), Box::new(rhs))
    }
    /// Bitwise XOR.
    #[must_use]
    pub fn bv_xor(lhs: Self, rhs: Self) -> Self {
        Self::Xor(Box::new(lhs), Box::new(rhs))
    }
    /// Logical left shift.
    #[must_use]
    pub fn shl(lhs: Self, rhs: Self) -> Self {
        Self::Shl(Box::new(lhs), Box::new(rhs))
    }
    /// Logical right shift.
    #[must_use]
    pub fn lshr(lhs: Self, rhs: Self) -> Self {
        Self::LShr(Box::new(lhs), Box::new(rhs))
    }
    /// Arithmetic right shift.
    #[must_use]
    pub fn ashr(lhs: Self, rhs: Self) -> Self {
        Self::AShr(Box::new(lhs), Box::new(rhs))
    }
    /// `lhs == rhs` as a 1-bit value.
    #[must_use]
    pub fn eq(lhs: Self, rhs: Self) -> Self {
        Self::Eq(Box::new(lhs), Box::new(rhs))
    }
    /// `lhs != rhs` as a 1-bit value.
    #[must_use]
    pub fn ne(lhs: Self, rhs: Self) -> Self {
        Self::Ne(Box::new(lhs), Box::new(rhs))
    }
    /// Unsigned less-than.
    #[must_use]
    pub fn ult(lhs: Self, rhs: Self) -> Self {
        Self::Ult(Box::new(lhs), Box::new(rhs))
    }
    /// Unsigned less-than-or-equal.
    #[must_use]
    pub fn ule(lhs: Self, rhs: Self) -> Self {
        Self::Ule(Box::new(lhs), Box::new(rhs))
    }
    /// Signed less-than.
    #[must_use]
    pub fn slt(lhs: Self, rhs: Self) -> Self {
        Self::Slt(Box::new(lhs), Box::new(rhs))
    }
    /// Signed less-than-or-equal.
    #[must_use]
    pub fn sle(lhs: Self, rhs: Self) -> Self {
        Self::Sle(Box::new(lhs), Box::new(rhs))
    }
    /// Boolean AND of 1-bit operands.
    #[must_use]
    pub fn bool_and(lhs: Self, rhs: Self) -> Self {
        Self::BoolAnd(Box::new(lhs), Box::new(rhs))
    }
    /// Boolean OR of 1-bit operands.
    #[must_use]
    pub fn bool_or(lhs: Self, rhs: Self) -> Self {
        Self::BoolOr(Box::new(lhs), Box::new(rhs))
    }
    /// Boolean NOT of a 1-bit operand.
    #[must_use]
    pub fn bool_not(operand: Self) -> Self {
        Self::BoolNot(Box::new(operand))
    }
    /// Extract bits `[hi:lo]` (inclusive) from `src`.
    #[must_use]
    pub fn extract(src: Self, hi: u16, lo: u16) -> Self {
        Self::Extract {
            src: Box::new(src),
            hi,
            lo,
        }
    }
    /// Concatenate two bit-vectors: `high` is placed above `low`.
    #[must_use]
    pub fn concat(high: Self, low: Self) -> Self {
        Self::Concat {
            high: Box::new(high),
            low: Box::new(low),
        }
    }
    /// Zero-extend `src` to `to_bits` total bits.
    #[must_use]
    pub fn zero_ext(src: Self, to_bits: u16) -> Self {
        Self::ZeroExtend {
            src: Box::new(src),
            to_bits,
        }
    }
    /// Sign-extend `src` to `to_bits` total bits.
    #[must_use]
    pub fn sign_ext(src: Self, to_bits: u16) -> Self {
        Self::SignExtend {
            src: Box::new(src),
            to_bits,
        }
    }

    /// Floating-point addition under `rm`.
    #[must_use]
    pub fn fadd(a: Self, b: Self, rm: RoundingMode) -> Self {
        Self::FAdd(Box::new(a), Box::new(b), rm)
    }
    /// Floating-point subtraction under `rm`.
    #[must_use]
    pub fn fsub(a: Self, b: Self, rm: RoundingMode) -> Self {
        Self::FSub(Box::new(a), Box::new(b), rm)
    }
    /// Floating-point multiplication under `rm`.
    #[must_use]
    pub fn fmul(a: Self, b: Self, rm: RoundingMode) -> Self {
        Self::FMul(Box::new(a), Box::new(b), rm)
    }
    /// Floating-point division under `rm`.
    #[must_use]
    pub fn fdiv(a: Self, b: Self, rm: RoundingMode) -> Self {
        Self::FDiv(Box::new(a), Box::new(b), rm)
    }
    /// IEEE floating-point equality (1-bit result).
    #[must_use]
    pub fn feq(a: Self, b: Self) -> Self {
        Self::FEq(Box::new(a), Box::new(b))
    }
    /// IEEE floating-point less-than (1-bit result).
    #[must_use]
    pub fn flt(a: Self, b: Self) -> Self {
        Self::FLt(Box::new(a), Box::new(b))
    }
    /// IEEE floating-point less-than-or-equal (1-bit result).
    #[must_use]
    pub fn fle(a: Self, b: Self) -> Self {
        Self::FLe(Box::new(a), Box::new(b))
    }
    /// Floating-point is-NaN predicate (1-bit result).
    #[must_use]
    pub fn fisnan(a: Self) -> Self {
        Self::FIsNaN(Box::new(a))
    }
    /// Reinterpret a bit-vector as a float of sort `(ebits, sbits)`.
    #[must_use]
    pub fn bv_to_fp(src: Self, ebits: u16, sbits: u16) -> Self {
        Self::BvToFp {
            src: Box::new(src),
            ebits,
            sbits,
        }
    }
    /// Reinterpret a float as its IEEE bit-vector.
    #[must_use]
    pub fn fp_to_ieee_bv(src: Self) -> Self {
        Self::FpToIeeeBv(Box::new(src))
    }
    /// Convert a float to a signed `bits`-wide integer under `rm`.
    #[must_use]
    pub fn fp_to_sbv(src: Self, rm: RoundingMode, bits: u16) -> Self {
        Self::FpToSbv {
            src: Box::new(src),
            rm,
            bits,
        }
    }
    /// Convert a signed integer to a float of sort `(ebits, sbits)`.
    #[must_use]
    pub fn sbv_to_fp(src: Self, rm: RoundingMode, ebits: u16, sbits: u16) -> Self {
        Self::SbvToFp {
            src: Box::new(src),
            rm,
            ebits,
            sbits,
        }
    }

    /// Construct a [`Expr::FSqrt`].
    #[must_use]
    pub fn fsqrt(a: Self, rm: RoundingMode) -> Self {
        Self::FSqrt(Box::new(a), rm)
    }

    /// Construct a [`Expr::FpToFp`].
    #[must_use]
    pub fn fp_to_fp(src: Self, rm: RoundingMode, ebits: u16, sbits: u16) -> Self {
        Self::FpToFp {
            src: Box::new(src),
            rm,
            ebits,
            sbits,
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(v) => write!(f, "{}", v.name),
            Self::Const { value, bits } => write!(f, "{value:#x}:{bits}"),
            Self::Add(a, b) => write!(f, "({a} + {b})"),
            Self::Sub(a, b) => write!(f, "({a} - {b})"),
            Self::Mul(a, b) => write!(f, "({a} * {b})"),
            Self::UDiv(a, b) => write!(f, "({a} /u {b})"),
            Self::URem(a, b) => write!(f, "({a} %u {b})"),
            Self::SDiv(a, b) => write!(f, "({a} /s {b})"),
            Self::SRem(a, b) => write!(f, "({a} %s {b})"),
            Self::And(a, b) => write!(f, "({a} & {b})"),
            Self::Or(a, b) => write!(f, "({a} | {b})"),
            Self::Xor(a, b) => write!(f, "({a} ^ {b})"),
            Self::Shl(a, b) => write!(f, "({a} << {b})"),
            Self::LShr(a, b) => write!(f, "({a} >>u {b})"),
            Self::AShr(a, b) => write!(f, "({a} >>s {b})"),
            Self::Eq(a, b) => write!(f, "({a} == {b})"),
            Self::Ne(a, b) => write!(f, "({a} != {b})"),
            Self::Ult(a, b) => write!(f, "({a} <u {b})"),
            Self::Ule(a, b) => write!(f, "({a} <=u {b})"),
            Self::Slt(a, b) => write!(f, "({a} <s {b})"),
            Self::Sle(a, b) => write!(f, "({a} <=s {b})"),
            Self::BoolAnd(a, b) => write!(f, "({a} && {b})"),
            Self::BoolOr(a, b) => write!(f, "({a} || {b})"),
            Self::BoolNot(e) => write!(f, "!({e})"),
            Self::Ite {
                cond,
                then_expr,
                else_expr,
            } => {
                write!(f, "ite({cond}, {then_expr}, {else_expr})")
            }
            Self::Extract { src, hi, lo } => write!(f, "{src}[{hi}:{lo}]"),
            Self::Concat { high, low } => write!(f, "concat({high}, {low})"),
            Self::ZeroExtend { src, to_bits } => write!(f, "zext({src}, {to_bits})"),
            Self::SignExtend { src, to_bits } => write!(f, "sext({src}, {to_bits})"),
            Self::Unknown(reason) if reason.is_empty() => write!(f, "?"),
            Self::Unknown(reason) => write!(f, "?({reason})"),
            Self::FAdd(a, b, rm) => write!(f, "fadd({a}, {b}, {rm:?})"),
            Self::FSub(a, b, rm) => write!(f, "fsub({a}, {b}, {rm:?})"),
            Self::FMul(a, b, rm) => write!(f, "fmul({a}, {b}, {rm:?})"),
            Self::FDiv(a, b, rm) => write!(f, "fdiv({a}, {b}, {rm:?})"),
            Self::FEq(a, b) => write!(f, "feq({a}, {b})"),
            Self::FLt(a, b) => write!(f, "flt({a}, {b})"),
            Self::FLe(a, b) => write!(f, "fle({a}, {b})"),
            Self::FIsNaN(a) => write!(f, "fisnan({a})"),
            Self::FSqrt(a, rm) => write!(f, "fsqrt({a}, {rm:?})"),
            Self::FpConst { bits, ebits, sbits } => write!(f, "fp{ebits}_{sbits}({bits:#x})"),
            Self::BvToFp { src, ebits, sbits } => write!(f, "bv_to_fp({src}, {ebits}, {sbits})"),
            Self::FpToIeeeBv(src) => write!(f, "fp_to_bv({src})"),
            Self::FpToSbv { src, rm, bits } => write!(f, "fp_to_sbv({src}, {rm:?}, {bits})"),
            Self::SbvToFp {
                src,
                rm,
                ebits,
                sbits,
            } => write!(f, "sbv_to_fp({src}, {rm:?}, {ebits}, {sbits})"),
            Self::FpToFp {
                src,
                rm,
                ebits,
                sbits,
            } => write!(f, "fp_to_fp({src}, {rm:?}, {ebits}, {sbits})"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn display_renders_infix() {
        let expr = Expr::eq(
            Expr::sub(Expr::var("eax", 32), Expr::konst(2, 32)),
            Expr::konst(0, 32),
        );
        assert_eq!(expr.to_string(), "((eax - 0x2:32) == 0x0:32)");
    }

    #[test]
    fn display_renders_boolean_combinations() {
        let expr = Expr::bool_and(
            Expr::eq(Expr::flag("CF"), Expr::konst(0, 1)),
            Expr::eq(Expr::flag("ZF"), Expr::konst(0, 1)),
        );
        assert_eq!(expr.to_string(), "((CF == 0x0:1) && (ZF == 0x0:1))");
    }

    #[test]
    fn json_round_trip_preserves_shape() {
        let expr = Expr::Ite {
            cond: Box::new(Expr::flag("ZF")),
            then_expr: Box::new(Expr::var("eax", 32)),
            else_expr: Box::new(Expr::konst(0, 32)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let back: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, expr);
    }

    #[test]
    fn unknown_carries_optional_hint() {
        let e = Expr::Unknown("unmodeled flag".into());
        assert_eq!(e.to_string(), "?(unmodeled flag)");
        assert_eq!(Expr::unknown().to_string(), "?");
    }

    #[test]
    fn wide_vector_widths_round_trip_through_json() {
        // zmm-width var, a high-half extract, and a 128-bit all-ones
        // constant exceeding u64 — all representable after the u16/u128
        // width migration and byte-identical on the JSON wire.
        let expr = Expr::Concat {
            high: Box::new(Expr::Extract {
                src: Box::new(Expr::var("zmm0", 512)),
                hi: 511,
                lo: 256,
            }),
            low: Box::new(Expr::konst(u128::MAX >> 1, 128)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let back: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, expr);
    }

    #[test]
    fn floating_point_nodes_round_trip_through_json() {
        // A single-precision `(x + 1.0) == 2.0` predicate over a
        // bit-vector reinterpreted as float, exercising the FP sorts,
        // a rounding mode, an FP constant and the BV↔FP reinterprets.
        let x = Expr::bv_to_fp(Expr::var("xmm0_lo", 32), 8, 24);
        let one = Expr::FpConst {
            bits: 0x3f80_0000,
            ebits: 8,
            sbits: 24,
        };
        let two = Expr::FpConst {
            bits: 0x4000_0000,
            ebits: 8,
            sbits: 24,
        };
        let expr = Expr::feq(Expr::fadd(x, one, RoundingMode::NearestTiesEven), two);
        let json = serde_json::to_string(&expr).unwrap();
        let back: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, expr);
    }

    #[test]
    fn float_to_float_conversion_round_trips_through_json() {
        let expr = Expr::fp_to_fp(
            Expr::bv_to_fp(Expr::var("xmm0_lo", 32), 8, 24),
            RoundingMode::NearestTiesEven,
            11,
            53,
        );
        let json = serde_json::to_string(&expr).unwrap();
        let back: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(back, expr);
    }
}
