//! IR statement form produced by the lifter.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::expr::{Expr, Var};

/// A single IR statement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IrStmt {
    /// `dst := src`. Used for both register assignments and flag
    /// updates (flags are 1-bit [`Var`]s).
    Assign {
        /// Destination variable.
        dst: Var,
        /// Right-hand-side expression.
        src: Expr,
    },
    /// `dst := *address` reading `bits` bits.
    LoadMem {
        /// Destination variable.
        dst: Var,
        /// Address expression.
        address: Expr,
        /// Width in bits being read.
        bits: u8,
    },
    /// `*address := value` writing `bits` bits.
    StoreMem {
        /// Address expression.
        address: Expr,
        /// Value being written.
        value: Expr,
        /// Width in bits being written.
        bits: u8,
    },
    /// Marker for an instruction the lifter could not translate; the
    /// payload carries the original mnemonic and a hint.
    Unsupported {
        /// Original mnemonic.
        mnemonic: String,
        /// Short reason explaining the failure.
        comment: String,
    },
    /// No-op (used for instructions whose effect is fully captured by
    /// previous statements or that are intentionally elided).
    Nop,
}

impl fmt::Display for IrStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Assign { dst, src } => write!(f, "{dst} := {src}"),
            Self::LoadMem { dst, address, bits } => write!(f, "{dst} := load{bits}({address})"),
            Self::StoreMem {
                address,
                value,
                bits,
            } => write!(f, "store{bits}({address}) := {value}"),
            Self::Unsupported { mnemonic, comment } => {
                if comment.is_empty() {
                    write!(f, "// unsupported: {mnemonic}")
                } else {
                    write!(f, "// unsupported: {mnemonic} ({comment})")
                }
            }
            Self::Nop => write!(f, "nop"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn display_renders_assignment() {
        let stmt = IrStmt::Assign {
            dst: Var::new("eax", 32),
            src: Expr::konst(0x10, 32),
        };
        assert_eq!(stmt.to_string(), "eax := 0x10:32");
    }

    #[test]
    fn display_renders_unsupported_with_reason() {
        let stmt = IrStmt::Unsupported {
            mnemonic: "vpxor".into(),
            comment: "SIMD".into(),
        };
        assert_eq!(stmt.to_string(), "// unsupported: vpxor (SIMD)");
    }

    #[test]
    fn json_round_trip() {
        let stmt = IrStmt::Assign {
            dst: Var::new("ZF", 1),
            src: Expr::eq(Expr::var("t0", 32), Expr::konst(0, 32)),
        };
        let json = serde_json::to_string(&stmt).unwrap();
        let back: IrStmt = serde_json::from_str(&json).unwrap();
        assert_eq!(back, stmt);
    }
}
