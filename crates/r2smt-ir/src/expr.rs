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
    /// Width in bits (`1` for flags, `8/16/32/64` for register slices).
    pub bits: u8,
}

impl Var {
    /// Construct a variable with a borrowed name.
    #[must_use]
    pub fn new(name: impl Into<String>, bits: u8) -> Self {
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
        value: u64,
        /// Width in bits.
        bits: u8,
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
        hi: u8,
        /// Inclusive low bit.
        lo: u8,
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
        to_bits: u8,
    },
    /// Sign-extend `src` to `to_bits` total bits.
    SignExtend {
        /// Source bit-vector.
        src: Box<Expr>,
        /// Target width.
        to_bits: u8,
    },
    /// Something the lifter could not translate (operand we cannot
    /// parse, instruction with side effects we have not modelled, …).
    /// The string is a short, human-readable hint.
    Unknown(String),
}

impl Expr {
    /// Construct a [`Expr::Var`].
    #[must_use]
    pub fn var(name: impl Into<String>, bits: u8) -> Self {
        Self::Var(Var::new(name, bits))
    }

    /// Construct a 1-bit flag variable.
    #[must_use]
    pub fn flag(name: impl Into<String>) -> Self {
        Self::var(name, 1)
    }

    /// Construct a typed constant.
    #[must_use]
    pub const fn konst(value: u64, bits: u8) -> Self {
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
    pub fn extract(src: Self, hi: u8, lo: u8) -> Self {
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
    pub fn zero_ext(src: Self, to_bits: u8) -> Self {
        Self::ZeroExtend {
            src: Box::new(src),
            to_bits,
        }
    }
    /// Sign-extend `src` to `to_bits` total bits.
    #[must_use]
    pub fn sign_ext(src: Self, to_bits: u8) -> Self {
        Self::SignExtend {
            src: Box::new(src),
            to_bits,
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
}
