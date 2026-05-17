//! Bounded simple-diamond Φ-merge *detection*, extracted from
//! `slice.rs`. Recognises a `head → {taken, fallthrough} → join`
//! shape whose arms are each ≤1 block, fully sliceable and call /
//! memory / unsupported-free. Pairs with `lift/merge.rs` (the
//! lowering side). The backward-walk slicer in the parent module
//! calls `try_build_diamond_merge`; this module calls back into the
//! parent's `slice_branch` / `single_pred` / `predecessors_of`
//! (ancestor-private, reachable from a child module).

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction};
use tracing::debug;

use crate::collector::BranchCandidate;
use crate::effect::{InstructionKind, analyze};

use super::{
    MERGE_ARM_MAX_INSTRUCTIONS, MergedVar, SliceLimits, SliceMerge, WalkState, predecessors_of,
    single_pred, slice_branch,
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
/// * otherwise `edge_target` must be a single block whose only
///   predecessor is the head and whose only successor is the join,
///   call / memory / unsupported-free and within the instruction
///   budget — a clean `≤1`-block arm.
///
/// Any other shape returns `None`, forcing the caller to fall back to
/// the sound free-input boundary.
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
    let arm = function
        .blocks
        .iter()
        .find(|blk| blk.address == edge_target)?;
    if arm.successors.as_slice() != [join] {
        return None;
    }
    if single_pred(function, arm.address) != Some(head_addr) {
        return None;
    }
    if arm.instructions.len() > MERGE_ARM_MAX_INSTRUCTIONS {
        return None;
    }
    if !arm
        .instructions
        .iter()
        .all(|insn| mergeable_arm_insn(insn, arch))
    {
        return None;
    }
    Some(ArmResolution {
        instructions: arm.instructions.clone(),
        covered_pred: arm.address,
    })
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

/// Identify the head block of a 2-predecessor join, covering the full
/// diamond (`sp(a) == sp(b)`) and the `if`-with-no-`else` shape (one
/// predecessor *is* the head).
fn diamond_head(function: &Function, a: Address, b: Address, join: Address) -> Option<Address> {
    if let (Some(ha), Some(hb)) = (single_pred(function, a), single_pred(function, b))
        && ha == hb
    {
        return Some(ha);
    }
    if single_pred(function, b) == Some(a) && predecessors_of(function, join).contains(&a) {
        return Some(a);
    }
    if single_pred(function, a) == Some(b) && predecessors_of(function, join).contains(&b) {
        return Some(b);
    }
    None
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

/// Attempt to recover a bounded simple-diamond Φ-merge at `join`.
///
/// Returns `Some` only when the join is a fully-resolvable diamond:
/// exactly two predecessors forming a `head → {taken, fallthrough} →
/// join` shape, each arm a single call / memory / unsupported-free
/// block within budget, the head ending in a still-two-way
/// conditional branch, the pending live set register-only, and no
/// outstanding flag obligation. Polarity is taken verbatim from the
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
    if state.needs_flags || !state.live_stack.is_empty() || state.live.is_empty() {
        return None;
    }
    if preds.len() != 2 {
        return None;
    }
    let j = join.address;
    let (a, b) = (preds[0], preds[1]);
    let head_addr = diamond_head(function, a, b, j)?;
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
    })
}
