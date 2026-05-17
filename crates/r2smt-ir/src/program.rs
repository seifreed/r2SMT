//! Normalized program model: functions, basic blocks, and instructions.

use r2smt_common::{Address, Arch};
use serde::{Deserialize, Serialize};

/// A binary as seen by r2SMT after radare2 analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    /// Target instruction set.
    pub arch: Arch,
    /// Pointer width in bits (32 for x86, 64 for `x86_64`).
    pub bits: u8,
    /// Entry point address, when known.
    pub entry: Option<Address>,
    /// Functions discovered by radare2.
    pub functions: Vec<Function>,
}

/// A function discovered in the binary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Function {
    /// Function start address.
    pub address: Address,
    /// Symbolic name reported by radare2 (e.g. `sym.main`), if any.
    pub name: Option<String>,
    /// Basic blocks belonging to this function.
    pub blocks: Vec<BasicBlock>,
    /// `true` when this function is encoded in `AArch32` Thumb / Thumb-2
    /// mode (2-byte or mixed 2/4-byte instructions). Only meaningful
    /// for `Arch::Arm`; ignored on every other ISA. Defaults to `false`
    /// so legacy fixtures and non-ARM targets need no migration.
    #[serde(default)]
    pub is_thumb: bool,
}

/// A straight-line sequence of instructions ending in a branch / return.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasicBlock {
    /// First instruction address.
    pub address: Address,
    /// Instructions in execution order.
    pub instructions: Vec<Instruction>,
    /// Addresses of successor blocks in the control-flow graph.
    pub successors: Vec<Address>,
}

/// A single decoded instruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instruction {
    /// Instruction address.
    pub address: Address,
    /// Instruction size in bytes.
    pub size: u8,
    /// Raw instruction bytes.
    pub bytes: Vec<u8>,
    /// Mnemonic as reported by radare2 (lowercased).
    pub mnemonic: String,
    /// Decoded operands in source order.
    pub operands: Vec<Operand>,
    /// ESIL expression as reported by radare2, if any.
    pub esil: Option<String>,
    /// SLEIGH P-code text for *this instruction only* (one `pdgsd`
    /// header line plus its op lines), populated by the r2ghidra
    /// adapter only when the P-code IR source is selected. `None`
    /// otherwise so the default (ESIL) path needs no migration.
    #[serde(default)]
    pub pcode: Option<String>,
    /// `true` when this instruction is encoded in `AArch32` Thumb mode.
    /// Per-instruction override of [`Function::is_thumb`] for the rare
    /// case of ARM/Thumb interworking inside a single function. Only
    /// meaningful for `Arch::Arm`. Defaults to `false`.
    #[serde(default)]
    pub is_thumb: bool,
}

/// A single instruction operand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operand {
    /// Operand spelling as it appears in disassembly.
    pub raw: String,
    /// Coarse operand classification.
    pub kind: OperandKind,
}

/// Coarse operand classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OperandKind {
    /// A CPU register (e.g. `eax`, `rcx`).
    Register,
    /// An immediate value (e.g. `0x10`, `2`).
    Immediate,
    /// A memory reference (e.g. `[rbp - 4]`).
    Memory,
    /// Anything r2SMT cannot classify yet.
    Unknown,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn sample_program() -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![Instruction {
                        address: Address(0x40_1000),
                        size: 2,
                        bytes: vec![0x31, 0xc0],
                        mnemonic: "xor".into(),
                        operands: vec![
                            Operand {
                                raw: "eax".into(),
                                kind: OperandKind::Register,
                            },
                            Operand {
                                raw: "eax".into(),
                                kind: OperandKind::Register,
                            },
                        ],
                        esil: Some("0,eax,=".into()),
                        pcode: None,
                        is_thumb: false,
                    }],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    #[test]
    fn program_round_trips_through_json() {
        let prog = sample_program();
        let json = serde_json::to_string(&prog).unwrap();
        let back: Program = serde_json::from_str(&json).unwrap();
        assert_eq!(prog, back);
    }

    #[test]
    fn json_uses_hex_addresses_and_snake_case_arch() {
        let prog = sample_program();
        let json = serde_json::to_string(&prog).unwrap();
        assert!(json.contains("\"arch\":\"x86_64\""));
        assert!(json.contains("\"entry\":\"0x401000\""));
        assert!(json.contains("\"kind\":\"register\""));
    }
}
