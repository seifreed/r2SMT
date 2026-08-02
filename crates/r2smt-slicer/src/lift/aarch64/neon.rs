//! `AArch64` NEON (Advanced SIMD) shape resolution and lowering.
//!
//! Every NEON instruction the lifter models is resolved here, by one
//! function, into a [`NeonShape`] that fully describes the lowering. The
//! effect table and the per-mnemonic dispatcher both consult that
//! resolver, which is what keeps them from disagreeing about which
//! instructions the slicer may retain: an instruction kept because its
//! destination is a definition, but whose lowering is then dropped,
//! would leave a later read bound to a stale value.
//!
//! Resolution is deliberately strict about operand shape. NEON reuses
//! the scalar mnemonics (`add`, `fadd`, `mov`), so a shape that does not
//! match the family exactly must fall through to the next family and, in
//! the end, decline — never be guessed at.

use r2smt_common::Arch;
use r2smt_ir::expr::Expr;
use r2smt_ir::program::{Instruction, Operand, OperandKind};

use crate::registers::{
    Arrangement, element_type_bits, is_simd_parent, parse_arrangement, parse_lane_index,
    register_layout,
};

use super::super::{BinOp, FpArithOp, LiftCtx, PackedIntOp, PackedOp, parse_immediate};

/// Number of bits in a byte, used wherever an operand counts bytes
/// (`ext`'s index) but the model counts bits.
const BITS_PER_BYTE: u16 = 8;

/// Which of an instruction's two vector sources a permuted lane comes
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermuteSource {
    First,
    Second,
}

/// A lane permutation, expressed as the source each destination lane
/// draws from. Every member of the family is a pure rearrangement of
/// existing lanes, so one representation covers them all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermuteKind {
    /// `zip1` / `zip2` — interleave the lower or upper halves.
    Zip { upper: bool },
    /// `uzp1` / `uzp2` — deinterleave the even or odd lanes.
    Uzp { odd: bool },
    /// `trn1` / `trn2` — transpose the even or odd lanes.
    Trn { odd: bool },
    /// `rev16` / `rev32` / `rev64` — reverse the element order within
    /// each container of this many bits.
    Reverse { container_bits: u16 },
}

/// What a resolved NEON instruction computes.
#[derive(Debug, Clone, Copy)]
enum NeonOp {
    /// Lane-wise arithmetic or logic over identically shaped operands.
    Packed(PackedOp),
    /// `movi` / `mvni` — replicate an immediate to every lane,
    /// optionally inverted.
    Immediate { value: u64, invert: bool },
    /// `dup` — replicate one source value to every lane. The source is
    /// a general-purpose register, or an element named by
    /// [`NeonShape::source_index`].
    Duplicate { from_element: bool },
    /// `ext` — extract a window from the two sources concatenated, at a
    /// byte offset.
    Extract { byte_offset: u16 },
    /// A pure lane permutation over one or two sources.
    Permute(PermuteKind),
    /// `umov` / `smov` — move one element into a general-purpose
    /// register, zero- or sign-extended.
    ElementToGpr { signed: bool },
    /// `ins` — insert a general-purpose register or another vector's
    /// element into one lane of the destination.
    Insert { from_element: bool },
    /// A widening or narrowing element operation. `signed` selects the
    /// extension; `upper` is the `2` suffix, which reads the top half of
    /// its sources and, when narrowing, writes the top half of its
    /// destination.
    Widen {
        kind: WidenKind,
        signed: bool,
        upper: bool,
    },
    /// `mla` / `mls` and the long `umlal` / `smlal` / `umlsl` / `smlsl`:
    /// multiply the two sources and accumulate the product into the
    /// destination.
    MultiplyAccumulate(AccumulateKind),
}

/// How a multiply-accumulate reads its source elements.
///
/// A separate type rather than a `widen` flag beside a `signed` one,
/// because signedness is meaningless without widening: `mla` extends
/// nothing, so there is no extension to choose.
#[derive(Debug, Clone, Copy)]
enum AccumulateSources {
    /// `mla` / `mls` — sources are already the destination's element
    /// width.
    SameWidth,
    /// `umlal` / `smlal` / `umlsl` / `smlsl` — sources are half the
    /// destination's element width and are extended before the multiply.
    Long { signed: bool },
}

/// How a multiply-accumulate reads its sources and combines the product.
#[derive(Debug, Clone, Copy)]
struct AccumulateKind {
    /// Whether the product is added to or subtracted from the
    /// accumulator.
    combine: BinOp,
    sources: AccumulateSources,
    /// The `2` suffix — read the sources' upper half.
    upper: bool,
}

/// The widening and narrowing element operations.
#[derive(Debug, Clone, Copy)]
enum WidenKind {
    /// `ushll` / `sshll` and their `uxtl` / `sxtl` zero-shift aliases:
    /// extend each element to double width, then shift left.
    ShiftLong { shift: u16 },
    /// `xtn` — truncate each element to half its width.
    Narrow,
    /// The long and wide arithmetic: extend the narrow sources to the
    /// destination's element width and operate there. `wide_first` marks
    /// the `w`-suffixed forms, whose first source is already wide.
    Arith { op: BinOp, wide_first: bool },
}

/// A resolved NEON instruction: what to compute, and at what geometry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NeonShape {
    op: NeonOp,
    /// Element width of the destination, in bits.
    lane_bits: u16,
    /// Number of destination lanes. One for the forms whose destination
    /// is a single element or a general-purpose register.
    lanes: u16,
    /// Element index addressed in the destination, for the inserts.
    dest_index: u16,
    /// Element index addressed in the source, for the element-reading
    /// forms.
    source_index: u16,
}

impl NeonShape {
    /// Whether the lowering reads the destination register as well as
    /// writing it.
    ///
    /// True for the element inserts, which preserve every lane they do
    /// not write, and for the narrowing `2` forms, which write only the
    /// destination's upper half. Everything else here writes the
    /// destination whole — an `AArch64` SIMD write has no merging form,
    /// so even a 64-bit arrangement replaces the register by zeroing its
    /// upper half.
    pub(crate) const fn reads_destination(&self) -> bool {
        matches!(
            self.op,
            NeonOp::Insert { .. }
                | NeonOp::MultiplyAccumulate { .. }
                | NeonOp::Widen {
                    kind: WidenKind::Narrow,
                    upper: true,
                    ..
                }
        )
    }

    /// Whether the destination is a general-purpose register rather than
    /// a vector one.
    const fn writes_gpr(&self) -> bool {
        matches!(self.op, NeonOp::ElementToGpr { .. })
    }
}

/// Resolve `insn` into the NEON lowering that models it, or `None` when
/// none does.
///
/// The families are tried in order of how tightly they constrain their
/// operands, so a mnemonic shared between two of them (`mov` is both a
/// lane-wise copy and an element insert) reaches the one whose operand
/// shape it actually matches.
pub(crate) fn shape(insn: &Instruction) -> Option<NeonShape> {
    let mnemonic = insn.mnemonic.trim().to_ascii_lowercase();
    packed_shape_of(insn, &mnemonic)
        .or_else(|| immediate_shape(insn, &mnemonic))
        .or_else(|| duplicate_shape(insn, &mnemonic))
        .or_else(|| extract_shape(insn, &mnemonic))
        .or_else(|| permute_shape(insn, &mnemonic))
        .or_else(|| element_to_gpr_shape(insn, &mnemonic))
        .or_else(|| insert_shape(insn, &mnemonic))
        .or_else(|| widen_shape(insn, &mnemonic))
        .or_else(|| multiply_accumulate_shape(insn, &mnemonic))
}

// ===================== multiply-accumulate =====================

/// The multiply-accumulate family.
///
/// Same-width (`mla` / `mls`) and long (`umlal` / `smlal` / `umlsl` /
/// `smlsl`) share one lowering: multiply the sources, then add or
/// subtract the product against the destination's existing lane. The
/// long forms multiply at the destination's width, so the product cannot
/// overflow the element the way a same-width multiply can.
///
/// Every member reads its destination. That is the whole point of an
/// accumulator, and it is what the effect table has to be told, or the
/// slicer will drop the definition the accumulation builds on.
fn multiply_accumulate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let long = |signed| AccumulateSources::Long { signed };
    let (combine, sources) = match base {
        "mla" => (BinOp::Add, AccumulateSources::SameWidth),
        "mls" => (BinOp::Sub, AccumulateSources::SameWidth),
        "umlal" => (BinOp::Add, long(false)),
        "smlal" => (BinOp::Add, long(true)),
        "umlsl" => (BinOp::Sub, long(false)),
        "smlsl" => (BinOp::Sub, long(true)),
        _ => return None,
    };
    let widen = matches!(sources, AccumulateSources::Long { .. });
    // The same-width forms have no `2` variant.
    if upper && !widen {
        return None;
    }
    if insn.operands.len() != 3 {
        return None;
    }
    let destination = operand_arrangement(insn.operands.first()?)?;
    let source_bits = if widen {
        destination.lane_bits / 2
    } else {
        destination.lane_bits
    };
    if source_bits == 0 {
        return None;
    }
    let expected_lanes = if upper {
        destination.lanes.checked_mul(2)?
    } else {
        destination.lanes
    };
    for operand in insn.operands.iter().skip(1) {
        let arrangement = operand_arrangement(operand)?;
        if arrangement.lane_bits != source_bits || arrangement.lanes != expected_lanes {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::MultiplyAccumulate(AccumulateKind {
            combine,
            sources,
            upper,
        }),
        lane_bits: destination.lane_bits,
        lanes: destination.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== widening and narrowing =====================

/// Peel the `2` suffix that marks the half-register forms.
fn peel_upper(mnemonic: &str) -> (&str, bool) {
    mnemonic
        .strip_suffix('2')
        .map_or((mnemonic, false), |base| (base, true))
}

/// The widening / narrowing operation a mnemonic names, with the
/// signedness its leading letter spells.
fn widen_kind(base: &str) -> Option<(WidenKind, bool)> {
    // The zero-shift aliases carry no immediate operand of their own.
    let arith = |op, wide_first| WidenKind::Arith { op, wide_first };
    Some(match base {
        "xtn" => (WidenKind::Narrow, false),
        "uaddl" => (arith(BinOp::Add, false), false),
        "saddl" => (arith(BinOp::Add, false), true),
        "usubl" => (arith(BinOp::Sub, false), false),
        "ssubl" => (arith(BinOp::Sub, false), true),
        "umull" => (arith(BinOp::Mul, false), false),
        "smull" => (arith(BinOp::Mul, false), true),
        "uaddw" => (arith(BinOp::Add, true), false),
        "saddw" => (arith(BinOp::Add, true), true),
        "usubw" => (arith(BinOp::Sub, true), false),
        "ssubw" => (arith(BinOp::Sub, true), true),
        _ => return None,
    })
}

/// `ushll` / `sshll` and their `uxtl` / `sxtl` aliases, whose shift is
/// an operand in the first spelling and zero in the second.
fn shift_long_kind(base: &str, insn: &Instruction) -> Option<(WidenKind, bool)> {
    let (signed, has_immediate) = match base {
        "ushll" => (false, true),
        "sshll" => (true, true),
        "uxtl" => (false, false),
        "sxtl" => (true, false),
        _ => return None,
    };
    let shift = if has_immediate {
        u16::try_from(parse_immediate(&insn.operands.get(2)?.raw)?).ok()?
    } else {
        if insn.operands.len() != 2 {
            return None;
        }
        0
    };
    Some((WidenKind::ShiftLong { shift }, signed))
}

/// The widening and narrowing family.
///
/// The destination's arrangement gives the geometry; the sources are
/// checked against it rather than trusted, because the whole family is
/// distinguished from the lane-wise one precisely by its operands having
/// *different* arrangements. A `2` form reads sources that span the full
/// register and takes their upper half.
fn widen_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let (base, upper) = peel_upper(mnemonic);
    let (kind, signed) = widen_kind(base).or_else(|| shift_long_kind(base, insn))?;
    let destination = operand_arrangement(insn.operands.first()?)?;
    let expected_operands = match kind {
        WidenKind::Narrow => 2,
        WidenKind::ShiftLong { .. } => {
            if matches!(base, "uxtl" | "sxtl") {
                2
            } else {
                3
            }
        }
        WidenKind::Arith { .. } => 3,
    };
    if insn.operands.len() != expected_operands {
        return None;
    }
    let narrow_bits = match kind {
        // The destination is the narrow side; the source is wide.
        WidenKind::Narrow => destination.lane_bits.checked_mul(2)?,
        _ => destination.lane_bits / 2,
    };
    if narrow_bits == 0 || !matches!(narrow_bits, 8 | 16 | 32 | 64) {
        return None;
    }
    // Written lanes: for a narrowing `2` form the destination holds
    // twice as many as the instruction produces, the low half surviving.
    let written = match kind {
        WidenKind::Narrow if upper => destination.lanes / 2,
        _ => destination.lanes,
    };
    if written == 0 {
        return None;
    }
    for (index, operand) in insn.operands.iter().enumerate().skip(1) {
        let Some(arrangement) = operand_arrangement(operand) else {
            // The shift is an immediate, not an arrangement.
            if matches!(kind, WidenKind::ShiftLong { .. }) && index == 2 {
                continue;
            }
            return None;
        };
        let wide_operand = matches!(kind, WidenKind::Narrow)
            || matches!(kind, WidenKind::Arith { wide_first: true, .. } if index == 1);
        let expected_bits = if wide_operand {
            match kind {
                WidenKind::Narrow => narrow_bits,
                _ => destination.lane_bits,
            }
        } else {
            narrow_bits
        };
        if arrangement.lane_bits != expected_bits {
            return None;
        }
        // A `2` form's narrow operands span the whole register, so they
        // carry twice the lanes the instruction consumes; a wide operand
        // is never halved.
        let expected_lanes = if upper && !wide_operand {
            written.checked_mul(2)?
        } else {
            written
        };
        if arrangement.lanes != expected_lanes {
            return None;
        }
    }
    Some(NeonShape {
        op: NeonOp::Widen {
            kind,
            signed,
            upper,
        },
        lane_bits: destination.lane_bits,
        lanes: written,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== lane-wise arithmetic and logic =====================

/// The packed operation an `AArch64` NEON data-processing mnemonic
/// computes, or `None` for a mnemonic no packed handler models.
fn packed_op(mnemonic: &str) -> Option<PackedOp> {
    Some(match mnemonic {
        "add" => PackedOp::Int(PackedIntOp::Bin(BinOp::Add)),
        "sub" => PackedOp::Int(PackedIntOp::Bin(BinOp::Sub)),
        "mul" => PackedOp::Int(PackedIntOp::Bin(BinOp::Mul)),
        "and" => PackedOp::Int(PackedIntOp::Bin(BinOp::And)),
        "orr" => PackedOp::Int(PackedIntOp::Bin(BinOp::Or)),
        "eor" => PackedOp::Int(PackedIntOp::Bin(BinOp::Xor)),
        "bic" => PackedOp::Int(PackedIntOp::BitClear),
        "mvn" | "not" => PackedOp::Int(PackedIntOp::Not),
        "mov" => PackedOp::Int(PackedIntOp::Copy),
        "fadd" => PackedOp::Fp(FpArithOp::Add),
        "fsub" => PackedOp::Fp(FpArithOp::Sub),
        "fmul" => PackedOp::Fp(FpArithOp::Mul),
        "fdiv" => PackedOp::Fp(FpArithOp::Div),
        _ => return None,
    })
}

/// The lane-wise family: every operand a vector register carrying the
/// *same* arrangement.
///
/// That is what the architecture spells for these mnemonics, and
/// requiring it is what rejects the widening forms
/// (`umlal v0.4s, v1.4h, v2.4h`), the by-element forms
/// (`mul v0.4s, v1.4s, v2.s[0]`) and the immediate ones
/// (`bic v0.4h, #0x10`) without listing any of them.
fn packed_shape_of(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let op = packed_op(mnemonic)?;
    if insn.operands.len() != op.operand_count() {
        return None;
    }
    let mut shared: Option<Arrangement> = None;
    for operand in &insn.operands {
        let arrangement = operand_arrangement(operand)?;
        if *shared.get_or_insert(arrangement) != arrangement {
            return None;
        }
    }
    let arrangement = shared?;
    // A floating-point lane has to name a float sort; `.16b` does not.
    if matches!(op, PackedOp::Fp(_)) && !matches!(arrangement.lane_bits, 16 | 32 | 64) {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Packed(op),
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

// ===================== broadcast and permutation =====================

/// `movi vd.<T>, #imm{, lsl #shift}` and `mvni`, which replicate an
/// immediate across every lane.
///
/// The disassembler prints the *decoded* immediate, so the printed value
/// is the per-lane one and needs no re-expansion — including for `.2d`,
/// whose encoded form is a byte mask. An `msl` shift is a different
/// operation (it shifts ones in, not zeroes) and declines rather than
/// being treated as `lsl`.
fn immediate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let invert = match mnemonic {
        "movi" => false,
        "mvni" => true,
        _ => return None,
    };
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    let raw = parse_immediate(&insn.operands.get(1)?.raw)?;
    let shift = match insn.operands.len() {
        2 => 0,
        3 => shift_amount(insn.operands.get(2)?)?,
        _ => return None,
    };
    let value = raw.checked_shl(u32::from(shift))?;
    Some(NeonShape {
        op: NeonOp::Immediate { value, invert },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The `lsl #n` shift operand of a `movi`. Radare2 renders the shifting
/// modifier as one operand, so the whole thing is parsed here.
fn shift_amount(op: &Operand) -> Option<u16> {
    let raw = op.raw.trim().to_ascii_lowercase();
    let body = raw.strip_prefix("lsl")?.trim();
    u16::try_from(parse_immediate(body)?).ok()
}

/// `dup vd.<T>, rn` and `dup vd.<T>, vn.<Ts>[index]`.
///
/// The scalar-destination form (`dup d0, v1.d[1]`) declines: its
/// destination carries no arrangement, so the lane count is not spelled.
fn duplicate_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if mnemonic != "dup" || insn.operands.len() != 2 {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    let source = insn.operands.get(1)?;
    let (from_element, source_index) = if let Some((element_bits, index)) = indexed_element(source)
    {
        if element_bits != arrangement.lane_bits {
            return None;
        }
        (true, index)
    } else {
        // A general-purpose source supplies the low `lane_bits`.
        if !is_general_register(source) {
            return None;
        }
        (false, 0)
    };
    Some(NeonShape {
        op: NeonOp::Duplicate { from_element },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index,
    })
}

/// `ext vd.<T>, vn.<T>, vm.<T>, #index` — a byte-granular window over
/// the two sources concatenated, with `vn` at the low end.
///
/// Only the byte arrangements are architecturally valid, and the index
/// must leave a whole view inside the concatenation.
fn extract_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    if mnemonic != "ext" || insn.operands.len() != 4 {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    if arrangement.lane_bits != BITS_PER_BYTE {
        return None;
    }
    for index in 1..=2 {
        if operand_arrangement(insn.operands.get(index)?)? != arrangement {
            return None;
        }
    }
    let byte_offset = u16::try_from(parse_immediate(&insn.operands.get(3)?.raw)?).ok()?;
    if byte_offset >= arrangement.lanes {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Extract { byte_offset },
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// The permutation family: `zip`, `uzp`, `trn` over two sources, and
/// `rev` over one.
fn permute_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let kind = match mnemonic {
        "zip1" => PermuteKind::Zip { upper: false },
        "zip2" => PermuteKind::Zip { upper: true },
        "uzp1" => PermuteKind::Uzp { odd: false },
        "uzp2" => PermuteKind::Uzp { odd: true },
        "trn1" => PermuteKind::Trn { odd: false },
        "trn2" => PermuteKind::Trn { odd: true },
        "rev16" => PermuteKind::Reverse { container_bits: 16 },
        "rev32" => PermuteKind::Reverse { container_bits: 32 },
        "rev64" => PermuteKind::Reverse { container_bits: 64 },
        _ => return None,
    };
    let operand_count = if matches!(kind, PermuteKind::Reverse { .. }) {
        2
    } else {
        3
    };
    if insn.operands.len() != operand_count {
        return None;
    }
    let arrangement = operand_arrangement(insn.operands.first()?)?;
    for index in 1..operand_count {
        if operand_arrangement(insn.operands.get(index)?)? != arrangement {
            return None;
        }
    }
    // A reversal needs whole elements inside each container, and at
    // least two of them for the reversal to mean anything.
    if let PermuteKind::Reverse { container_bits } = kind {
        if container_bits <= arrangement.lane_bits
            || container_bits % arrangement.lane_bits != 0
            || arrangement.view_bits() % container_bits != 0
        {
            return None;
        }
    } else if arrangement.lanes % 2 != 0 {
        return None;
    }
    Some(NeonShape {
        op: NeonOp::Permute(kind),
        lane_bits: arrangement.lane_bits,
        lanes: arrangement.lanes,
        dest_index: 0,
        source_index: 0,
    })
}

/// `umov wd, vn.<Ts>[index]` and `smov`, which move one element into a
/// general-purpose register.
fn element_to_gpr_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    let signed = match mnemonic {
        // `mov wd, vn.s[0]` is the assembler alias for `umov`.
        "umov" | "mov" => false,
        "smov" => true,
        _ => return None,
    };
    if insn.operands.len() != 2 {
        return None;
    }
    let destination = insn.operands.first()?;
    if !is_general_register(destination) {
        return None;
    }
    let (lane_bits, source_index) = indexed_element(insn.operands.get(1)?)?;
    Some(NeonShape {
        op: NeonOp::ElementToGpr { signed },
        lane_bits,
        lanes: 1,
        dest_index: 0,
        source_index,
    })
}

/// `ins vd.<Ts>[index], rn` and `ins vd.<Ts>[index], vn.<Ts>[index2]`,
/// which write one lane and preserve the rest.
fn insert_shape(insn: &Instruction, mnemonic: &str) -> Option<NeonShape> {
    // `mov v0.s[1], w0` is the assembler alias for `ins`.
    if !matches!(mnemonic, "ins" | "mov") || insn.operands.len() != 2 {
        return None;
    }
    let (lane_bits, dest_index) = indexed_element(insn.operands.first()?)?;
    let source = insn.operands.get(1)?;
    let (from_element, source_index) = if let Some((source_bits, index)) = indexed_element(source) {
        if source_bits != lane_bits {
            return None;
        }
        (true, index)
    } else {
        if !is_general_register(source) {
            return None;
        }
        (false, 0)
    };
    Some(NeonShape {
        op: NeonOp::Insert { from_element },
        lane_bits,
        lanes: 1,
        dest_index,
        source_index,
    })
}

// ===================== operand shape helpers =====================

/// The arrangement an operand carries, if it is a vector register that
/// carries one.
fn operand_arrangement(op: &Operand) -> Option<Arrangement> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if !names_vector_register(&raw) {
        return None;
    }
    let (_, body) = raw.split_once('.')?;
    parse_arrangement(body)
}

/// The `(element_bits, index)` an indexed operand names: `v0.s[1]` is a
/// 32-bit element at index 1.
fn indexed_element(op: &Operand) -> Option<(u16, u16)> {
    if op.kind != OperandKind::Register {
        return None;
    }
    let raw = op.raw.trim().to_ascii_lowercase();
    if !names_vector_register(&raw) {
        return None;
    }
    let (_, body) = raw.split_once('.')?;
    let mut chars = body.chars();
    let element_bits = element_type_bits(chars.next()?)?;
    let index = parse_lane_index(chars.as_str())?;
    // The element has to fit inside the 128-bit register.
    let lanes = crate::registers::simd_parent_bits(Arch::Aarch64)? / element_bits;
    (index < lanes).then_some((element_bits, index))
}

/// Whether an operand names a general-purpose register.
fn is_general_register(op: &Operand) -> bool {
    op.kind == OperandKind::Register
        && register_layout(&op.raw, Arch::Aarch64)
            .is_some_and(|layout| !is_simd_parent(layout.parent, Arch::Aarch64))
}

/// Whether the operand text names a vector register (before any shape
/// suffix is considered).
fn names_vector_register(raw: &str) -> bool {
    register_layout(raw, Arch::Aarch64)
        .is_some_and(|layout| is_simd_parent(layout.parent, Arch::Aarch64))
}

// ===================== lowering =====================

impl LiftCtx {
    /// Lower a resolved NEON instruction.
    ///
    /// A decline here is sound by the free-input boundary: the slicer
    /// consumed the destination as a definition and therefore stopped
    /// tracking its upstream definitions, so emitting no assignment
    /// leaves the register a free SSA input rather than bound to a stale
    /// value.
    pub(super) fn lift_neon(&mut self, insn: &Instruction, shape: NeonShape) {
        if let NeonOp::Packed(op) = shape.op {
            // An `AArch64` SIMD write has no merging form, so the write
            // zeroes the register above the arrangement's view.
            self.lift_packed_vector(insn, op, shape.lane_bits, true);
            return;
        }
        let Some(destination) = insn.operands.first().cloned() else {
            self.push_neon_unsupported(insn);
            return;
        };
        let Some(value) = self.neon_value(insn, shape) else {
            self.push_neon_unsupported(insn);
            return;
        };
        let written = if shape.writes_gpr() {
            self.write_register_to(&destination, value)
        } else if matches!(shape.op, NeonOp::Insert { .. }) {
            self.write_simd_lane(&destination, value, shape.lane_bits, shape.dest_index)
        } else {
            self.write_xmm_dst(&destination, value, true)
        };
        if !written {
            self.push_neon_unsupported(insn);
        }
    }

    /// Build the value a resolved NEON instruction writes.
    fn neon_value(&mut self, insn: &Instruction, shape: NeonShape) -> Option<Expr> {
        match shape.op {
            // Handled by the caller; never reaches here.
            NeonOp::Packed(_) => None,
            NeonOp::Immediate { value, invert } => Some(immediate_lanes(&shape, value, invert)),
            NeonOp::Duplicate { from_element } => self.duplicate_lanes(insn, shape, from_element),
            NeonOp::Extract { byte_offset } => self.extract_window(insn, shape, byte_offset),
            NeonOp::Permute(kind) => self.permute_lanes(insn, shape, kind),
            NeonOp::ElementToGpr { signed } => self.element_to_gpr(insn, shape, signed),
            NeonOp::Insert { from_element } => self.insert_source(insn, shape, from_element),
            NeonOp::Widen {
                kind,
                signed,
                upper,
            } => self.widen_lanes(insn, shape, kind, signed, upper),
            NeonOp::MultiplyAccumulate(kind) => self.multiply_accumulate_lanes(insn, shape, kind),
        }
    }

    /// The multiply-accumulate family: each destination lane is its own
    /// prior value plus or minus the product of the two source lanes.
    fn multiply_accumulate_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: AccumulateKind,
    ) -> Option<Expr> {
        let AccumulateKind {
            combine,
            sources,
            upper,
        } = kind;
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let accumulator = self.simd_operand_value(&insn.operands.first()?.clone(), view)?;
        let first = self.widen_source(insn, 1)?;
        let second = self.widen_source(insn, 2)?;
        let source_bits = match sources {
            AccumulateSources::SameWidth => shape.lane_bits,
            AccumulateSources::Long { .. } => shape.lane_bits / 2,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let source_lane = if upper {
                index.checked_add(shape.lanes)?
            } else {
                index
            };
            let read = |value: &Expr| -> Option<Expr> {
                let element = LiftCtx::extract_lane(value.clone(), source_bits, source_lane)?;
                Some(match sources {
                    AccumulateSources::SameWidth => element,
                    AccumulateSources::Long { signed: true } => {
                        Expr::sign_ext(element, shape.lane_bits)
                    }
                    AccumulateSources::Long { signed: false } => {
                        Expr::zero_ext(element, shape.lane_bits)
                    }
                })
            };
            let product = Expr::mul(read(&first)?, read(&second)?);
            let previous = LiftCtx::extract_lane(accumulator.clone(), shape.lane_bits, index)?;
            lanes.push(combine.apply(previous, product));
        }
        Self::concat_lanes(lanes)
    }

    /// The widening and narrowing family: read each source element,
    /// extend or truncate it to the destination's width, and operate
    /// there.
    ///
    /// Extending *before* operating is the whole point of the family —
    /// it is what stops the result overflowing the element, which is
    /// exactly what the same-width lane-wise form would do.
    fn widen_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: WidenKind,
        signed: bool,
        upper: bool,
    ) -> Option<Expr> {
        let first = self.widen_source(insn, 1)?;
        let second = match insn.operands.get(2) {
            Some(operand) if operand_arrangement(operand).is_some() => {
                Some(self.widen_source(insn, 2)?)
            }
            _ => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            lanes.push(Self::widen_lane(
                &first,
                second.as_ref(),
                shape,
                kind,
                signed,
                index,
                upper,
            )?);
        }
        let narrowed = Self::concat_lanes(lanes)?;
        if !matches!(kind, WidenKind::Narrow) || !upper {
            return Some(narrowed);
        }
        // `xtn2` writes the destination's upper half and preserves the
        // lower one.
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let destination = self.simd_operand_value(&insn.operands.first()?.clone(), view * 2)?;
        Some(Expr::concat(
            narrowed,
            Expr::extract(destination, view - 1, 0),
        ))
    }

    /// One source operand, materialised once at its own view width.
    fn widen_source(&mut self, insn: &Instruction, position: usize) -> Option<Expr> {
        let operand = insn.operands.get(position)?.clone();
        let arrangement = operand_arrangement(&operand)?;
        self.simd_operand_value(&operand, arrangement.view_bits())
    }

    /// One destination lane of a widening or narrowing operation.
    ///
    /// `upper` shifts the lane read out of a *narrow* source by half a
    /// register, which is what the `2` suffix means. A wide source is
    /// never halved — `uaddw2 v0.4s, v1.4s, v2.8h` takes `v1`'s lane `i`
    /// and `v2`'s lane `i + 4`.
    fn widen_lane(
        first: &Expr,
        second: Option<&Expr>,
        shape: NeonShape,
        kind: WidenKind,
        signed: bool,
        index: u16,
        upper: bool,
    ) -> Option<Expr> {
        let extend = |value: Expr| {
            if signed {
                Expr::sign_ext(value, shape.lane_bits)
            } else {
                Expr::zero_ext(value, shape.lane_bits)
            }
        };
        let narrow_lane = if upper {
            index.checked_add(shape.lanes)?
        } else {
            index
        };
        let narrow = shape.lane_bits / 2;
        match kind {
            WidenKind::Narrow => {
                // Truncation needs no signedness: the low half of a
                // two's-complement value is the same bits either way.
                // The source is wide, so it is never halved — `xtn2`
                // narrows the same lanes `xtn` does and only the
                // *destination* half differs.
                let wide = shape.lane_bits.checked_mul(2)?;
                let element = LiftCtx::extract_lane(first.clone(), wide, index)?;
                Some(Expr::extract(element, shape.lane_bits - 1, 0))
            }
            WidenKind::ShiftLong { shift } => {
                let element = LiftCtx::extract_lane(first.clone(), narrow, narrow_lane)?;
                let widened = extend(element);
                if shift == 0 {
                    return Some(widened);
                }
                Some(Expr::shl(
                    widened,
                    Expr::konst(u128::from(shift), shape.lane_bits),
                ))
            }
            WidenKind::Arith { op, wide_first } => {
                // A `w`-form's first source is already at the
                // destination's width and is read at its own lane index.
                let lhs = if wide_first {
                    LiftCtx::extract_lane(first.clone(), shape.lane_bits, index)?
                } else {
                    extend(LiftCtx::extract_lane(first.clone(), narrow, narrow_lane)?)
                };
                let rhs = extend(LiftCtx::extract_lane(second?.clone(), narrow, narrow_lane)?);
                Some(op.apply(lhs, rhs))
            }
        }
    }

    /// `dup` — one value replicated to every destination lane.
    fn duplicate_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        from_element: bool,
    ) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        let element = if from_element {
            self.read_simd_lane_bits(&source, shape.lane_bits, shape.source_index)?
        } else {
            // The low `lane_bits` of a general-purpose register.
            self.read_general_element(&source, shape.lane_bits)?
        };
        Self::concat_lanes(vec![element; usize::from(shape.lanes)])
    }

    /// `ext` — a window over `vm:vn` starting `byte_offset` bytes up,
    /// with `vn` at the low end of the concatenation.
    fn extract_window(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        byte_offset: u16,
    ) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let low = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let high = self.simd_operand_value(&insn.operands.get(2)?.clone(), view)?;
        let pair = Expr::concat(high, low);
        let lo = byte_offset.checked_mul(BITS_PER_BYTE)?;
        let hi = lo.checked_add(view)?.checked_sub(1)?;
        Some(Expr::extract(pair, hi, lo))
    }

    /// The permutation family — every destination lane is some source
    /// lane, so one loop over [`permuted_source`] covers all of them.
    fn permute_lanes(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        kind: PermuteKind,
    ) -> Option<Expr> {
        let view = shape.lane_bits.checked_mul(shape.lanes)?;
        let first = self.simd_operand_value(&insn.operands.get(1)?.clone(), view)?;
        let second = match insn.operands.get(2) {
            Some(operand) => Some(self.simd_operand_value(&operand.clone(), view)?),
            None => None,
        };
        let mut lanes = Vec::with_capacity(usize::from(shape.lanes));
        for index in 0..shape.lanes {
            let (which, source_lane) = permuted_source(kind, index, shape)?;
            let value = match which {
                PermuteSource::First => first.clone(),
                PermuteSource::Second => second.clone()?,
            };
            lanes.push(Self::extract_lane(value, shape.lane_bits, source_lane)?);
        }
        Self::concat_lanes(lanes)
    }

    /// `umov` / `smov` — one element widened into a general-purpose
    /// register.
    fn element_to_gpr(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        signed: bool,
    ) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        let element = self.read_simd_lane_bits(&source, shape.lane_bits, shape.source_index)?;
        let destination_bits = self.operand_width(insn.operands.first()?);
        Some(match destination_bits.cmp(&shape.lane_bits) {
            std::cmp::Ordering::Equal => element,
            std::cmp::Ordering::Greater if signed => Expr::sign_ext(element, destination_bits),
            std::cmp::Ordering::Greater => Expr::zero_ext(element, destination_bits),
            std::cmp::Ordering::Less => return None,
        })
    }

    /// `ins` — the value written into one destination lane.
    fn insert_source(
        &mut self,
        insn: &Instruction,
        shape: NeonShape,
        from_element: bool,
    ) -> Option<Expr> {
        let source = insn.operands.get(1)?.clone();
        if from_element {
            return self.read_simd_lane_bits(&source, shape.lane_bits, shape.source_index);
        }
        self.read_general_element(&source, shape.lane_bits)
    }

    /// The low `lane_bits` of a general-purpose register operand, as the
    /// element `dup` replicates and `ins` writes.
    fn read_general_element(&self, op: &Operand, lane_bits: u16) -> Option<Expr> {
        let whole = self.read_register(op)?;
        let natural = self.operand_width(op);
        Some(match natural.cmp(&lane_bits) {
            std::cmp::Ordering::Equal => whole,
            std::cmp::Ordering::Greater => Expr::extract(whole, lane_bits - 1, 0),
            std::cmp::Ordering::Less => return None,
        })
    }

    fn push_neon_unsupported(&mut self, insn: &Instruction) {
        self.stmts.push(r2smt_ir::stmt::IrStmt::Unsupported {
            mnemonic: insn.mnemonic.clone(),
            comment: format!("unmodellable NEON operand at {addr}", addr = insn.address),
        });
    }
}

/// The replicated lane value of a `movi` / `mvni`.
fn immediate_lanes(shape: &NeonShape, value: u64, invert: bool) -> Expr {
    let mask = if shape.lane_bits >= u16::try_from(u64::BITS).unwrap_or(u16::MAX) {
        u64::MAX
    } else {
        (1u64 << shape.lane_bits) - 1
    };
    let lane = if invert { !value } else { value } & mask;
    let element = Expr::konst(u128::from(lane), shape.lane_bits);
    LiftCtx::concat_lanes(vec![element; usize::from(shape.lanes)])
        .unwrap_or_else(|| Expr::konst(u128::from(lane), shape.lane_bits))
}

/// Which source lane feeds destination lane `index`.
///
/// - `zip1` / `zip2` interleave one half of each source: destination
///   lane `2i` takes source-one lane `i` (plus half the lane count for
///   `zip2`), lane `2i+1` takes source two's.
/// - `uzp1` / `uzp2` deinterleave: the low half of the destination walks
///   the even (or odd) lanes of source one, the high half those of
///   source two.
/// - `trn1` / `trn2` transpose: even destination lanes take even (or
///   odd) lanes of source one, odd ones the same lanes of source two.
/// - `rev` reverses element order inside each container, which is a
///   single-source permutation.
fn permuted_source(
    kind: PermuteKind,
    index: u16,
    shape: NeonShape,
) -> Option<(PermuteSource, u16)> {
    let lanes = shape.lanes;
    let half = lanes / 2;
    Some(match kind {
        PermuteKind::Zip { upper } => {
            let base = if upper { half } else { 0 };
            let pair = index / 2;
            if index % 2 == 0 {
                (PermuteSource::First, base.checked_add(pair)?)
            } else {
                (PermuteSource::Second, base.checked_add(pair)?)
            }
        }
        PermuteKind::Uzp { odd } => {
            let offset = u16::from(odd);
            if index < half {
                (
                    PermuteSource::First,
                    index.checked_mul(2)?.checked_add(offset)?,
                )
            } else {
                let local = index - half;
                (
                    PermuteSource::Second,
                    local.checked_mul(2)?.checked_add(offset)?,
                )
            }
        }
        PermuteKind::Trn { odd } => {
            let pair = index / 2;
            let source_lane = pair.checked_mul(2)?.checked_add(u16::from(odd))?;
            if index % 2 == 0 {
                (PermuteSource::First, source_lane)
            } else {
                (PermuteSource::Second, source_lane)
            }
        }
        PermuteKind::Reverse { container_bits } => {
            let per_container = container_bits / shape.lane_bits;
            let container = index / per_container;
            let within = index % per_container;
            let source_lane = container
                .checked_mul(per_container)?
                .checked_add(per_container - 1 - within)?;
            (PermuteSource::First, source_lane)
        }
    })
}
