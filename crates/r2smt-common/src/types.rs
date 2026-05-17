//! Cross-crate primitive types: machine addresses and target architectures.

use std::fmt;
use std::num::ParseIntError;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::Error;

/// A guest machine address.
///
/// Serializes as a lowercase hexadecimal string with a `0x` prefix
/// (e.g. `"0x401050"`) so JSON output stays human-readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Address(pub u64);

impl Address {
    /// Construct from a raw `u64`.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Underlying integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x}", self.0)
    }
}

impl From<u64> for Address {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl FromStr for Address {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        let (radix, body) = if let Some(rest) = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
        {
            (16, rest)
        } else {
            (10, trimmed)
        };
        let parsed: Result<u64, ParseIntError> = u64::from_str_radix(body, radix);
        parsed
            .map(Self)
            .map_err(|e| Error::parse("address", e.to_string()))
    }
}

impl Serialize for Address {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Address {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Address::from_str(&raw).map_err(serde::de::Error::custom)
    }
}

/// Target instruction set.
///
/// v0 supported `x86` and `x86_64`. v0.6 adds `arm` (32-bit `AArch32` /
/// Thumb) and `aarch64` (64-bit `ARMv8-A`) as recognised values so
/// the register-layout table, the parser, and downstream consumers
/// can distinguish them. The slicer / lifter still only handle x86
/// mnemonics today; the ARM variants exist so future lifter work can
/// plug in without re-plumbing the data flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Arch {
    /// 32-bit Intel/AMD.
    X86,
    /// 64-bit Intel/AMD.
    X86_64,
    /// 32-bit ARM (`AArch32` / Thumb / `ARMv7`).
    Arm,
    /// 64-bit ARM (`AArch64` / `ARMv8-A`).
    Aarch64,
}

impl Arch {
    /// Default register width in bits.
    #[must_use]
    pub const fn pointer_bits(self) -> u8 {
        match self {
            Self::X86 | Self::Arm => 32,
            Self::X86_64 | Self::Aarch64 => 64,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn address_displays_lowercase_hex_with_prefix() {
        assert_eq!(Address(0x40_1050).to_string(), "0x401050");
        assert_eq!(Address(0).to_string(), "0x0");
    }

    #[test]
    fn address_parses_hex_and_decimal() {
        assert_eq!(Address::from_str("0x401050").unwrap(), Address(0x40_1050));
        assert_eq!(Address::from_str("0X401050").unwrap(), Address(0x40_1050));
        assert_eq!(Address::from_str("1024").unwrap(), Address(1024));
    }

    #[test]
    fn address_rejects_garbage() {
        assert!(Address::from_str("not-an-address").is_err());
        assert!(Address::from_str("0xZZZ").is_err());
    }

    #[test]
    fn address_json_round_trip() {
        let addr = Address(0xdead_beef_u64);
        let json = serde_json::to_string(&addr).unwrap();
        assert_eq!(json, "\"0xdeadbeef\"");
        let back: Address = serde_json::from_str(&json).unwrap();
        assert_eq!(back, addr);
    }

    #[test]
    fn arch_pointer_bits() {
        assert_eq!(Arch::X86.pointer_bits(), 32);
        assert_eq!(Arch::X86_64.pointer_bits(), 64);
        assert_eq!(Arch::Arm.pointer_bits(), 32);
        assert_eq!(Arch::Aarch64.pointer_bits(), 64);
    }

    #[test]
    fn arch_json_round_trip() {
        let json = serde_json::to_string(&Arch::X86_64).unwrap();
        assert_eq!(json, "\"x86_64\"");
        let back: Arch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Arch::X86_64);
    }

    #[test]
    fn arch_arm_variants_json_round_trip() {
        for (arch, wire) in [(Arch::Arm, "\"arm\""), (Arch::Aarch64, "\"aarch64\"")] {
            let json = serde_json::to_string(&arch).unwrap();
            assert_eq!(json, wire);
            let back: Arch = serde_json::from_str(&json).unwrap();
            assert_eq!(back, arch);
        }
    }
}
