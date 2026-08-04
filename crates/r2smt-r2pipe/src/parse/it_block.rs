//! Fold Thumb-2 `IT` block state into per-instruction condition
//! suffixes.
//!
//! radare2 decodes each instruction independently and loses ITSTATE
//! before the block ends: on r2 6.1.8 an `itt` suffixes 1 of its 2
//! instructions and an `ittt` 2 of its 3 — the *last* instruction of
//! every block is printed unconditionally. Lifting that instruction as
//! unconditional is a wrong value, not a decline, and the backward
//! slicer can finish satisfying its live set before it ever reaches the
//! `it` that would have truncated the slice. The result is a fabricated
//! verdict.
//!
//! The mask is fully recoverable from the mnemonic radare2 already
//! prints (`ittt ne` covers three `ne` instructions; `ite eq` covers
//! `eq` then `ne`), and the ARM ARM defines it, so the mask — not the
//! disassembler's suffixes — is the authority here.
//!
//! The pass fails closed: anything it cannot resolve exactly leaves the
//! `it` mnemonic intact, and the slicer then truncates on it as before.

use r2smt_ir::program::{BasicBlock, Instruction};

/// The `AArch32` condition-code suffixes, paired with their inverses.
/// `hs`/`cs` and `lo`/`cc` are the same two encodings under both
/// spellings, so each maps back to the spelling it was written with.
const COND_INVERSES: &[(&str, &str)] = &[
    ("eq", "ne"),
    ("ne", "eq"),
    ("cs", "cc"),
    ("cc", "cs"),
    ("hs", "lo"),
    ("lo", "hs"),
    ("mi", "pl"),
    ("pl", "mi"),
    ("vs", "vc"),
    ("vc", "vs"),
    ("hi", "ls"),
    ("ls", "hi"),
    ("ge", "lt"),
    ("lt", "ge"),
    ("gt", "le"),
    ("le", "gt"),
];

/// Mnemonics whose own spelling ends in a condition-code sequence, so
/// "does this already carry suffix `c`?" cannot be answered by looking
/// at the trailing characters. `teq` ends in `eq`, `movs` in `vs`,
/// `bics`/`adcs`/`sbcs` in `cs`, `lsls`/`muls`/`mls`/`vmls` in `ls`,
/// `vcge`/`vcgt`/`vcle` in `ge`/`gt`/`le`. Rather than guess, the fold
/// is abandoned when one is covered by an IT block.
const COND_AMBIGUOUS_MNEMONICS: &[&str] = &[
    "teq", "movs", "bics", "adcs", "sbcs", "lsls", "muls", "mls", "vmls", "vmlsl", "smlals",
    "umlals", "vcge", "vcgt", "vcle",
];

/// The condition each instruction covered by an `it` block executes
/// under, in order, or `None` when `insn` is not a Thumb `IT`.
///
/// `it<x><y><z> <cond>` covers `1 + letters` instructions: the first
/// always takes `cond`, and each later one takes `cond` for a `t` and
/// its inverse for an `e`.
fn it_block_conditions(insn: &Instruction) -> Option<Vec<&'static str>> {
    if !insn.is_thumb {
        return None;
    }
    let letters = insn.mnemonic.strip_prefix("it")?;
    if letters.len() > 3 || !letters.chars().all(|c| c == 't' || c == 'e') {
        return None;
    }
    let spelled = insn.operands.first()?.raw.trim().to_ascii_lowercase();
    let (base, inverse) = COND_INVERSES
        .iter()
        .find(|(cond, _)| *cond == spelled)
        .copied()?;
    let mut conditions = vec![base];
    for letter in letters.chars() {
        conditions.push(if letter == 't' { base } else { inverse });
    }
    Some(conditions)
}

/// Split a mnemonic into the part a condition suffix attaches to and
/// any trailing qualifier. ARM spells the encoding-width hint and the
/// data type *after* the condition (`addne.w`, `vaddeq.f32`), so a
/// suffix appended to the whole string would land in the wrong place.
fn split_qualifier(mnemonic: &str) -> (&str, &str) {
    mnemonic
        .find('.')
        .map_or((mnemonic, ""), |at| mnemonic.split_at(at))
}

/// The mnemonic `base` executes under condition `cond`, or `None` when
/// that cannot be decided — the caller abandons the whole fold.
fn predicated_mnemonic(mnemonic: &str, cond: &str) -> Option<String> {
    let (base, qualifier) = split_qualifier(mnemonic);
    if COND_AMBIGUOUS_MNEMONICS.contains(&base) {
        return None;
    }
    if base.ends_with(cond) && base.len() > cond.len() {
        return Some(mnemonic.to_string());
    }
    // A suffix that is not the one the mask calls for means the mask
    // was misread; refuse rather than stack a second condition on it.
    if COND_INVERSES
        .iter()
        .any(|(other, _)| base.len() > other.len() && base.ends_with(other))
    {
        return None;
    }
    Some(format!("{base}{cond}{qualifier}"))
}

/// Rewrite every instruction covered by a Thumb `IT` block so it
/// carries the condition the block's mask assigns it, and reduce the
/// `it` itself to a `nop`.
///
/// The `nop` rewrite is the reason this is safe for the slicer to trust
/// without a second signal: it happens only on a fold that resolved
/// every covered instruction, so an `it` mnemonic surviving into the
/// slicer still means "not modelled" and still truncates.
pub(crate) fn fold_it_blocks(blocks: &mut [BasicBlock]) {
    let mut order: Vec<(usize, usize)> = blocks
        .iter()
        .enumerate()
        .flat_map(|(block, b)| (0..b.instructions.len()).map(move |insn| (block, insn)))
        .collect();
    order.sort_by_key(|&(block, insn)| blocks[block].instructions[insn].address);

    let mut at = 0;
    while at < order.len() {
        let (block, insn) = order[at];
        let Some(conditions) = it_block_conditions(&blocks[block].instructions[insn]) else {
            at += 1;
            continue;
        };
        // An IT block truncated by the end of the function is malformed;
        // leave it for the slicer to truncate on.
        let Some(covered) = order.get(at + 1..=at + conditions.len()) else {
            at += 1;
            continue;
        };
        if fold_one(blocks, covered, &conditions) {
            let it = &mut blocks[block].instructions[insn];
            it.mnemonic = "nop".to_string();
            it.operands.clear();
        }
        at += 1 + conditions.len();
    }
}

/// Apply `conditions` to `covered`, or leave every one of them
/// untouched. Resolving the whole block before writing any of it is
/// what makes an abandoned fold indistinguishable from no fold.
fn fold_one(
    blocks: &mut [BasicBlock],
    covered: &[(usize, usize)],
    conditions: &[&'static str],
) -> bool {
    let mut rewritten = Vec::with_capacity(covered.len());
    for (&(block, insn), cond) in covered.iter().zip(conditions) {
        let mnemonic = &blocks[block].instructions[insn].mnemonic;
        let Some(predicated) = predicated_mnemonic(mnemonic, cond) else {
            return false;
        };
        rewritten.push(predicated);
    }
    for (&(block, insn), mnemonic) in covered.iter().zip(rewritten) {
        blocks[block].instructions[insn].mnemonic = mnemonic;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use r2smt_common::Address;
    use r2smt_ir::program::{Operand, OperandKind};

    fn insn(address: u64, mnemonic: &str, operands: &[&str]) -> Instruction {
        Instruction {
            address: Address(address),
            size: 2,
            bytes: Vec::new(),
            mnemonic: mnemonic.to_string(),
            operands: operands
                .iter()
                .map(|raw| Operand {
                    raw: (*raw).to_string(),
                    kind: OperandKind::Register,
                })
                .collect(),
            esil: None,
            pcode: None,
            is_thumb: true,
        }
    }

    fn block(instructions: Vec<Instruction>) -> Vec<BasicBlock> {
        vec![BasicBlock {
            address: Address(0),
            instructions,
            successors: Vec::new(),
        }]
    }

    fn mnemonics(blocks: &[BasicBlock]) -> Vec<String> {
        blocks[0]
            .instructions
            .iter()
            .map(|i| i.mnemonic.clone())
            .collect()
    }

    #[test]
    fn test_it_block_suffixes_the_last_instruction_radare2_left_bare() {
        // The defect this pass exists for: r2 prints the third
        // instruction of an `ittt` with no suffix at all.
        let mut blocks = block(vec![
            insn(0, "ittt", &["ne"]),
            insn(2, "addne", &["r0", "r0", "r1"]),
            insn(4, "subne", &["r2", "r2", "r3"]),
            insn(6, "mov", &["r4", "r5"]),
        ]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks)[3], "movne");
    }

    #[test]
    fn test_it_block_reduces_the_it_instruction_to_a_nop() {
        let mut blocks = block(vec![
            insn(0, "it", &["gt"]),
            insn(2, "movgt", &["r0", "r1"]),
        ]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks)[0], "nop");
    }

    #[test]
    fn test_it_else_branch_takes_the_inverted_condition() {
        let mut blocks = block(vec![
            insn(0, "ite", &["eq"]),
            insn(2, "moveq", &["r0", "r1"]),
            insn(4, "mov", &["r2", "r3"]),
        ]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks)[2], "movne");
    }

    #[test]
    fn test_it_block_inserts_the_condition_before_an_encoding_width_suffix() {
        let mut blocks = block(vec![
            insn(0, "it", &["eq"]),
            insn(2, "add.w", &["r0", "r1"]),
        ]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks)[1], "addeq.w");
    }

    #[test]
    fn test_it_block_leaves_an_already_suffixed_mnemonic_alone() {
        let mut blocks = block(vec![
            insn(0, "it", &["eq"]),
            insn(2, "moveq", &["r0", "r1"]),
        ]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks)[1], "moveq");
    }

    #[test]
    fn test_it_block_abandons_the_fold_on_a_mnemonic_ending_in_a_condition_code() {
        // `teq` ends in `eq`; treating that as an existing suffix would
        // leave a predicated instruction unconditional.
        let mut blocks = block(vec![insn(0, "it", &["eq"]), insn(2, "teq", &["r0", "r1"])]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks), vec!["it", "teq"]);
    }

    #[test]
    fn test_it_block_abandons_the_fold_when_the_function_ends_early() {
        let mut blocks = block(vec![
            insn(0, "itt", &["eq"]),
            insn(2, "moveq", &["r0", "r1"]),
        ]);
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks), vec!["itt", "moveq"]);
    }

    #[test]
    fn test_it_block_spanning_two_basic_blocks_folds_in_address_order() {
        let mut blocks = vec![
            BasicBlock {
                address: Address(0),
                instructions: vec![insn(0, "itt", &["cs"]), insn(2, "movcs", &["r0", "r1"])],
                successors: Vec::new(),
            },
            BasicBlock {
                address: Address(4),
                instructions: vec![insn(4, "mov", &["r2", "r3"])],
                successors: Vec::new(),
            },
        ];
        fold_it_blocks(&mut blocks);
        assert_eq!(blocks[1].instructions[0].mnemonic, "movcs");
    }

    #[test]
    fn test_arm_mode_instruction_is_never_treated_as_an_it_block() {
        let mut blocks = block(vec![insn(0, "it", &["eq"]), insn(4, "mov", &["r0", "r1"])]);
        blocks[0].instructions[0].is_thumb = false;
        fold_it_blocks(&mut blocks);
        assert_eq!(mnemonics(&blocks), vec!["it", "mov"]);
    }
}
