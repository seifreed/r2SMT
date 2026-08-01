//! Bounded simple-diamond Φ-merge *detection*, extracted from
//! `slice.rs`. Recognises a `head → {taken, fallthrough} → join`
//! shape whose arms are each ≤1 block, fully sliceable and call /
//! memory / unsupported-free. Pairs with `lift/merge.rs` (the
//! lowering side). The backward-walk slicer in the parent module
//! calls `try_build_diamond_merge`; this module calls back into the
//! parent's `slice_branch` / `single_pred` / `predecessors_of`
//! (ancestor-private, reachable from a child module).

use std::collections::BTreeSet;

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction};
use tracing::debug;

use crate::collector::BranchCandidate;
use crate::effect::{InstructionKind, analyze};

use super::{
    MERGE_ARM_MAX_INSTRUCTIONS, MergedVar, SliceLimits, SliceMerge, WalkState, single_pred,
    slice_branch,
};

/// One resolved diamond edge: the arm body to lower (empty for an
/// `if`-with-no-`else` edge) and the predecessor of the join that this
/// edge accounts for (the arm block for a non-empty arm, the head for
/// an empty arm).
struct ArmResolution {
    instructions: Vec<Instruction>,
    covered_pred: Address,
}

/// Resolve one head edge (`edge_target`) into an [`ArmResolution`].
///
/// * `edge_target == join` → empty arm; the head itself is the join
///   predecessor this edge covers.
/// * otherwise `edge_target` starts a **straight-line chain** of blocks
///   `edge_target → … → last`, where every block has a single successor
///   (the next block in the chain, or the join for `last`), every
///   non-first block's only predecessor is the previous block, the
///   first block's only predecessor is the head, every instruction is
///   call / memory / unsupported-free, and the whole chain fits the
///   instruction budget. `last` is the join predecessor this edge
///   covers.
///
/// Any other shape returns `None`, forcing the caller to fall back to
/// the sound free-input boundary. A straight-line chain is safe to fold
/// linearly because it has no internal branches — it is just a longer
/// single-path arm.
fn resolve_arm_edge(
    function: &Function,
    head_addr: Address,
    join: Address,
    edge_target: Address,
    arch: Arch,
) -> Option<ArmResolution> {
    if edge_target == join {
        return Some(ArmResolution {
            instructions: Vec::new(),
            covered_pred: head_addr,
        });
    }
    if edge_target == head_addr {
        return None;
    }
    let mut instructions: Vec<Instruction> = Vec::new();
    let mut block_addr = edge_target;
    let mut chain: BTreeSet<Address> = BTreeSet::new();
    loop {
        if !chain.insert(block_addr) {
            return None; // cycle inside the arm — not a straight-line chain
        }
        let blk = function.blocks.iter().find(|b| b.address == block_addr)?;
        // The chain's first block must descend directly from the head;
        // interior links are verified before advancing (next block's
        // sole predecessor is the current one).
        if block_addr == edge_target && single_pred(function, block_addr) != Some(head_addr) {
            return None;
        }
        if !blk
            .instructions
            .iter()
            .all(|insn| mergeable_arm_insn(insn, arch))
        {
            return None;
        }
        instructions.extend(blk.instructions.iter().cloned());
        if instructions.len() > MERGE_ARM_MAX_INSTRUCTIONS {
            return None;
        }
        match blk.successors.as_slice() {
            [succ] if *succ == join => {
                return Some(ArmResolution {
                    instructions,
                    covered_pred: block_addr,
                });
            }
            [succ] => {
                // Continue only through a genuine single-entry chain:
                // the next block's sole predecessor must be this block.
                if single_pred(function, *succ) != Some(block_addr) {
                    return None;
                }
                block_addr = *succ;
            }
            _ => return None,
        }
    }
}

/// An instruction is safe to fold into a Φ-merge arm only if it is
/// inside the lifter's supported, side-effect-free subset — the same
/// gate [`walk_block`] applies. The lift stage applies a second guard
/// (any non-`Assign`/`Nop` lowering aborts the merge), so an
/// over-permissive answer here can never produce an unsound verdict.
fn mergeable_arm_insn(insn: &Instruction, arch: Arch) -> bool {
    let effect = analyze(insn, arch);
    !effect.is_call && !effect.has_memory_access && effect.kind != InstructionKind::Other
}

/// Locate the head block's still-two-way conditional terminator.
fn head_terminator(function: &Function, head_addr: Address, arch: Arch) -> Option<BranchCandidate> {
    crate::collector::collect_function_branches(function, arch)
        .into_iter()
        .rfind(|c| {
            c.block == head_addr
                && matches!(c.kind, crate::condition::BranchKind::Jcc)
                && c.taken_target.is_some()
                && c.fallthrough_target.is_some()
                && c.upstream_resolved.is_none()
        })
}

/// Walk up the unique-predecessor chain from `from` to the nearest
/// enclosing two-way conditional head, bounded by the arm budget. A
/// join predecessor may be the last block of a *multi-block* arm, so the
/// head is not necessarily its direct predecessor — this climbs the
/// single-entry chain until it reaches a still-two-way `Jcc` head. The
/// starting block itself counts (the `if`-with-no-`else` shape, where a
/// join predecessor *is* the head).
fn head_above(function: &Function, from: Address, arch: Arch) -> Option<Address> {
    let mut addr = from;
    let mut steps: usize = 0;
    let mut seen: BTreeSet<Address> = BTreeSet::new();
    loop {
        if !seen.insert(addr) {
            return None; // cycle
        }
        if head_terminator(function, addr, arch).is_some() {
            return Some(addr);
        }
        if steps >= MERGE_ARM_MAX_INSTRUCTIONS {
            return None;
        }
        steps += 1;
        addr = single_pred(function, addr)?;
    }
}

/// Identify the enclosing two-way head of a 2-predecessor join. Each
/// predecessor may be the tail of a multi-block straight-line arm, so
/// the head is found by climbing the unique-predecessor chain above
/// either predecessor to the nearest conditional head; the arm
/// resolution + covered-predecessor check then verifies the two edges
/// actually reconverge at this join.
fn diamond_head(function: &Function, a: Address, b: Address, arch: Arch) -> Option<Address> {
    head_above(function, a, arch).or_else(|| head_above(function, b, arch))
}

/// Slice the head block's condition within the head block only, so its
/// flag / register definitions are lowered ahead of the `Ite`
/// selector. Returns `None` when the dependency chain exceeds the arm
/// budget or contains a non-mergeable instruction.
fn head_condition_instructions(
    function: &Function,
    head: &BranchCandidate,
    arch: Arch,
) -> Option<Vec<Instruction>> {
    let limits = SliceLimits {
        max_basic_blocks: 1,
        ..SliceLimits::default()
    };
    let slice = slice_branch(head, function, &limits, arch);
    if slice.instructions.len() > MERGE_ARM_MAX_INSTRUCTIONS {
        return None;
    }
    if !slice
        .instructions
        .iter()
        .all(|insn| mergeable_arm_insn(insn, arch))
    {
        return None;
    }
    Some(slice.instructions)
}

/// Flags merged across both arms when the join branch still needs its
/// NZCV defined. The full x86-polarity set; unread flags collapse to
/// their pre-diamond value and are dead-flag-eliminated downstream.
const MERGED_FLAGS: &[&str] = &["ZF", "CF", "SF", "OF", "PF"];

/// External register dependencies of a merge's head condition: the
/// canonical roots of the head-condition slice, taken within the head
/// block only. Returns `None` when the head condition is *not* cleanly
/// resolvable inside its own block (truncated, or a non-register root),
/// in which case the caller must not chain into an upstream diamond and
/// should finalize at the sound free-input boundary instead. When
/// `Some`, chasing these roots into an upstream diamond is the
/// chained-diamond continuation.
pub(super) fn head_condition_roots(
    function: &Function,
    head: &BranchCandidate,
    arch: Arch,
) -> Option<Vec<&'static str>> {
    let limits = SliceLimits {
        max_basic_blocks: 1,
        ..SliceLimits::default()
    };
    let slice = slice_branch(head, function, &limits, arch);
    if !matches!(slice.status, crate::slice::SliceStatus::Complete) {
        return None;
    }
    let mut roots: Vec<&'static str> = Vec::new();
    for r in &slice.roots {
        // A non-register root (e.g. a stack slot) cannot be canonicalised
        // to a `'static` live-set entry — bail rather than guess.
        let layout = crate::registers::register_layout(r, arch)?;
        if !roots.contains(&layout.parent) {
            roots.push(layout.parent);
        }
    }
    Some(roots)
}

/// Attempt to recover a bounded simple-diamond Φ-merge at `join`.
///
/// Returns `Some` only when the join is a fully-resolvable diamond:
/// exactly two predecessors forming a `head → {taken, fallthrough} →
/// join` shape, each arm a straight-line call / memory / unsupported-free
/// chain within budget, the head ending in a still-two-way conditional
/// branch, and something pending (a live register or an outstanding flag
/// obligation). Registers are merged from the live set; when the branch
/// still needs flags, the NZCV set is merged from the arms too. Polarity
/// is taken verbatim from the
/// head [`BranchCandidate`] (`taken_target` → `taken_arm`). Any
/// deviation returns `None`, so the caller falls back to the existing
/// sound free-input boundary — the recovery only ever *adds*
/// precision, never widens the soundness envelope.
pub(super) fn try_build_diamond_merge(
    function: &Function,
    join: &BasicBlock,
    preds: &[Address],
    state: &WalkState,
    arch: Arch,
) -> Option<SliceMerge> {
    if !state.needs_flags && state.live.is_empty() {
        return None;
    }
    if preds.len() != 2 {
        return None;
    }
    let j = join.address;
    let (a, b) = (preds[0], preds[1]);
    let head_addr = diamond_head(function, a, b, arch)?;
    let head = head_terminator(function, head_addr, arch)?;
    let taken_target = head.taken_target?;
    let fallthrough_target = head.fallthrough_target?;
    if taken_target == fallthrough_target {
        return None;
    }

    let taken = resolve_arm_edge(function, head_addr, j, taken_target, arch)?;
    let fallthrough = resolve_arm_edge(function, head_addr, j, fallthrough_target, arch)?;

    // Both edges together must account for exactly the join's two
    // predecessors — otherwise the structure is not the simple
    // diamond we are allowed to merge soundly.
    let mut covered = [taken.covered_pred, fallthrough.covered_pred];
    covered.sort_unstable();
    let mut expected = [a, b];
    expected.sort_unstable();
    if covered != expected {
        return None;
    }

    let head_instructions = head_condition_instructions(function, &head, arch)?;

    let bits = arch.pointer_bits();
    let mut merged: Vec<MergedVar> = state
        .live
        .iter()
        .map(|name| MergedVar {
            name: (*name).to_string(),
            bits,
        })
        .collect();
    // When the join branch still needs its flags defined, merge them
    // across the arms too: each arm's fold yields its own flag values,
    // so `flag := Ite(head_cond, taken_flag, fallthrough_flag)` models
    // the branch's real NZCV precisely. An arm that does not set a flag
    // falls back to the pre-diamond value (the arm-env miss in
    // `lower_merge`), which is exactly its value on that path — never
    // worse than the free-input boundary this replaces.
    if state.needs_flags {
        for flag in MERGED_FLAGS {
            merged.push(MergedVar {
                name: (*flag).to_string(),
                bits: 1,
            });
        }
    }
    merged.sort_by(|x, y| x.name.cmp(&y.name));
    if merged.is_empty() {
        return None;
    }

    debug!(
        target: "r2smt::slicer",
        join = %j,
        head = %head_addr,
        merged = merged.len(),
        "bounded diamond Φ-merge recovered"
    );

    Some(SliceMerge {
        head,
        head_instructions,
        taken_arm: taken.instructions,
        fallthrough_arm: fallthrough.instructions,
        merged,
        // Filled in by the walker from `WalkState::kept.len()` when the
        // merge is recorded (this detector has no view of that count).
        tail_instructions: 0,
    })
}
