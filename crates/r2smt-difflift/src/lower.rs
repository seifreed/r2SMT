//! Independent invocation of the three IR lowerings for one
//! instruction.
//!
//! Each lowering is produced *in isolation* — the production
//! P-code-first → ESIL-first → per-mnemonic dispatch ladder in
//! [`r2smt_slicer::lift_slice`] is **not** used here. That ladder
//! short-circuits on the first lowering that succeeds, which is
//! exactly the behaviour a differential harness must bypass: it needs
//! every engine's opinion on the same bytes, not just the winner.

use r2smt_common::Arch;
use r2smt_ir::program::Instruction;
use r2smt_ir::stmt::IrStmt;

/// Identifies which independent lowering produced a body of IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lowering {
    /// Ghidra SLEIGH P-code ([`r2smt_pcode::lift_pcode`]).
    Pcode,
    /// radare2 ESIL ([`r2smt_esil::lift_esil`]).
    Esil,
    /// Per-mnemonic Fase-C handler ([`r2smt_slicer::lift_per_mnemonic`]).
    Mnemonic,
}

impl Lowering {
    /// Stable lower-case identifier (`"pcode"` / `"esil"` /
    /// `"mnemonic"`), used in diagnostics and metrics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pcode => "pcode",
            Self::Esil => "esil",
            Self::Mnemonic => "mnemonic",
        }
    }
}

/// The independent lowerings of a single instruction.
///
/// `pcode` / `esil` are `None` when radare2 attached no such IR for
/// the instruction or the respective lifter declined it (the construct
/// was outside its sound subset). `mnemonic` is always present — the
/// per-mnemonic dispatch emits an [`IrStmt::Unsupported`] marker rather
/// than declining, so the harness can still observe "this engine
/// modelled nothing".
#[derive(Debug, Clone)]
pub struct Lowerings {
    /// SLEIGH P-code lowering, if available.
    pub pcode: Option<Vec<IrStmt>>,
    /// ESIL lowering, if available.
    pub esil: Option<Vec<IrStmt>>,
    /// Per-mnemonic lowering (always produced).
    pub mnemonic: Vec<IrStmt>,
}

impl Lowerings {
    /// Iterate the available `(Lowering, statements)` pairs in a
    /// stable order: P-code, ESIL, then per-mnemonic.
    pub fn available(&self) -> impl Iterator<Item = (Lowering, &[IrStmt])> {
        self.pcode
            .as_deref()
            .map(|s| (Lowering::Pcode, s))
            .into_iter()
            .chain(self.esil.as_deref().map(|s| (Lowering::Esil, s)))
            .chain(std::iter::once((
                Lowering::Mnemonic,
                self.mnemonic.as_slice(),
            )))
    }
}

/// Lower `insn` through every independent IR pipeline under `arch`.
///
/// Declined ESIL / P-code lifts collapse to `None` (the harness simply
/// has fewer engines to cross-check for that instruction); a declined
/// lift never partially populates a statement list.
#[must_use]
pub fn lower_all(insn: &Instruction, arch: Arch) -> Lowerings {
    let pcode = insn
        .pcode
        .as_deref()
        .and_then(|text| r2smt_pcode::lift_pcode(text, arch).ok())
        .map(|lift| lift.statements);
    let esil = insn
        .esil
        .as_deref()
        .and_then(|text| r2smt_esil::lift_esil(text, arch).ok())
        .map(|lift| lift.statements);
    let mnemonic = r2smt_slicer::lift_per_mnemonic(insn, arch);
    Lowerings {
        pcode,
        esil,
        mnemonic,
    }
}
