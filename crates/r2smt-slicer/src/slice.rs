//! Bounded backward data-flow slicer.
//!
//! Given a [`BranchCandidate`] and the [`Function`] it belongs to, this
//! module walks the candidate's basic block in reverse and collects the
//! minimal set of instructions whose output reaches the candidate's
//! flag predicate. The walk is bounded by [`SliceLimits`] and stops
//! cleanly on unsupported instructions or memory / call boundaries.
//!
//! Multi-block walking is supported through unique-predecessor chains:
//! when the live set still holds registers (or `needs_flags` is true)
//! at the entry of the current block, the slicer continues into the
//! block's predecessor *if and only if* the current block has exactly
//! one predecessor in the function's CFG. Any join point (≥2
//! predecessors) terminates the walk as [`SliceStatus::Truncated`] —
//! mixing definitions from divergent paths without Φ-merging would
//! turn a sound "always-X" verdict into garbage. Cycles (loops) and
//! block-budget exhaustion (`SliceLimits::max_basic_blocks`) are
//! detected and truncate cleanly. The default budget remains `1`, so
//! callers opt in to multi-block walks by raising the limit.
//!
//! [`SliceLimits::allow_join_merge`] (default `false`) first attempts
//! a *bounded simple-diamond Φ-merge* (see [`SliceMerge`]): a
//! `head → {taken, fallthrough} → join` shape with ≤1-block,
//! fully-sliceable, side-effect-free arms is recovered precisely by
//! lowering each pending register to `Ite(head_condition, taken,
//! fallthrough)` ahead of the join body. When the join is *not* that
//! clean diamond it falls back to the *sound free-input boundary*:
//! the join-pending live set is promoted to free symbolic inputs
//! (`treat_truncation_as_inputs`), scoped to joins only — calls /
//! memory / unsupported still hard-truncate. Both behaviours are
//! sound: a path-insensitive predicate keeps its high-confidence
//! verdict, a path-sensitive one only ever widens an `AlwaysX` to
//! `BothPossible` — never a fabricated verdict. The *general* DAG
//! Φ-merge (nested / chained diamonds, merges inside loops) remains
//! future work — see CLAUDE.md "Still outside scope".

use std::collections::BTreeSet;

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::collector::BranchCandidate;
use crate::condition::BranchCondition;
use crate::effect::{InstructionKind, analyze};

mod merge_detect;
use merge_detect::try_build_diamond_merge;

/// Limits applied while slicing.
///
/// Defaults follow `SPEC.md` §5.4. `max_basic_blocks = 1` keeps the
/// historical single-block behavior; raising it enables the
/// multi-block walk through unique-predecessor chains (see the
/// module docs).
// `clippy::struct_excessive_bools`: justified, reviewed exception.
// `SliceLimits` is a stable serde DTO and a regression contract:
// it round-trips through `--json` output and is consumed
// cross-crate (slicer / core / cli) via individually-named
// `limits.allow_*` accesses, each mapped 1:1 to a documented
// `#[arg(long)]` CLI toggle. Collapsing the four orthogonal
// analysis toggles into a bit-flag/enum would break the public
// JSON schema and every call site for zero behavioral gain — the
// boolean-blindness the lint guards against does not apply to a
// named, individually-documented config record.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceLimits {
    /// Maximum number of instructions to keep in the slice.
    pub max_instructions: usize,
    /// Maximum number of basic blocks the slice may traverse,
    /// including the block containing the branch. `1` confines the
    /// walk to the branch's own block. Multi-block walks stop on
    /// joins (≥2 predecessors), cycles, function entry, or this
    /// budget — whichever comes first.
    pub max_basic_blocks: u32,
    /// Whether memory loads / stores may appear in the slice.
    pub allow_memory: bool,
    /// Whether `call` instructions may appear in the slice.
    pub allow_calls: bool,
    /// When set, a [`SliceStatus::Truncated`] result still drives the
    /// SMT pipeline: the remaining `roots` (registers / stack slots
    /// the slicer could not resolve) are propagated as free symbolic
    /// inputs through SSA. The verdict that comes back is sound —
    /// introducing free symbolic variables can only widen an
    /// `AlwaysX` to `BothPossible`, never fabricate a verdict — but
    /// the classifier downgrades its confidence by one notch to
    /// reflect that some inputs lived outside the analysed scope.
    #[serde(default)]
    pub unknowns_on_truncation: bool,
    /// When set, a CFG join (≥2 predecessors) no longer abandons the
    /// analysis: the join-pending live set is promoted to free
    /// symbolic inputs (sets `treat_truncation_as_inputs`), scoped to
    /// joins only — call / memory / unsupported truncations are
    /// unaffected. Sound: free inputs can only widen `AlwaysX` to
    /// `BothPossible`, never fabricate a verdict; the classifier
    /// downgrades confidence accordingly. Default `false` keeps the
    /// historical hard-truncate-at-join behavior byte-identical.
    #[serde(default)]
    pub allow_join_merge: bool,
}

impl Default for SliceLimits {
    fn default() -> Self {
        Self {
            max_instructions: 32,
            max_basic_blocks: 1,
            allow_memory: false,
            allow_calls: false,
            unknowns_on_truncation: false,
            allow_join_merge: false,
        }
    }
}

/// Result of slicing a single branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slice {
    /// The branch the slice was built for.
    pub branch: BranchCandidate,
    /// Instructions contributing to the branch condition, in execution
    /// (forward) order. Excludes the branch itself.
    pub instructions: Vec<Instruction>,
    /// Whether the slice is complete or why it was truncated.
    pub status: SliceStatus,
    /// Canonical names of registers that are *inputs* to the slice
    /// (defined outside the basic block).
    pub roots: Vec<String>,
    /// When `true`, downstream consumers (SSA, solver, classifier)
    /// must treat a [`SliceStatus::Truncated`] slice as if it were
    /// `Complete` with free symbolic inputs covering its `roots`.
    /// Only set when [`SliceLimits::unknowns_on_truncation`] was true
    /// and the walk produced a truncated result — preserves backward
    /// compatibility for every caller that does not opt in.
    #[serde(default)]
    pub treat_truncation_as_inputs: bool,
    /// Path-conditioned Φ-merges recovered at CFG joins (bounded
    /// simple-diamond shape only — see [`SliceMerge`]). Empty for
    /// every walk that did not recognise a fully-resolvable diamond,
    /// so the JSON contract is byte-identical for legacy callers and
    /// for the default (`allow_join_merge = false`) path. The lifter
    /// lowers each entry into a single `Ite` assignment ahead of the
    /// join-block body; SSA and the encoder need no changes because
    /// `Expr::Ite` is already pipeline-native.
    #[serde(default)]
    pub merges: Vec<SliceMerge>,
}

/// One bounded path-conditioned Φ-merge recovered at a CFG join.
///
/// Models the *simple diamond* only:
///
/// ```text
///         head        ; conditional branch, selector = head.condition
///        /    \
///   taken_arm  fallthrough_arm   ; each ≤ 1 block, fully sliceable,
///        \    /                  ; no call / memory / unsupported
///         join         ; backward walk reached here with `merged`
///                      ; registers still pending
/// ```
///
/// Polarity is taken verbatim from the head [`BranchCandidate`]:
/// `taken_arm` is the block on `head.taken_target` (reached when the
/// branch condition is **true**), `fallthrough_arm` the block on
/// `head.fallthrough_target`. The lifter emits, for every
/// [`MergedVar`], `var := Ite(head_condition, taken_value,
/// fallthrough_value)`. Recovering this only ever *adds* precision:
/// when both arms make the downstream predicate path-insensitive the
/// verdict stays high-confidence; when the arms genuinely differ the
/// head dependencies become free inputs and the verdict can only
/// widen to `BothPossible` — never a fabricated `AlwaysX`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceMerge {
    /// Head conditional branch whose two edges form the diamond. Its
    /// [`BranchCandidate::condition`] is lowered into the `Ite`
    /// selector via the shared `lift_branch_condition`.
    pub head: BranchCandidate,
    /// Head-block instructions defining what `head.condition` reads
    /// (sliced within the head block only), in execution order.
    pub head_instructions: Vec<Instruction>,
    /// Arm reached when `head.condition` is **true**
    /// (`head.taken_target`), in execution order. May be empty for an
    /// `if`-with-no-`else` diamond.
    pub taken_arm: Vec<Instruction>,
    /// Arm reached when `head.condition` is **false**
    /// (`head.fallthrough_target`), in execution order. May be empty
    /// for an `if`-with-no-`else` diamond.
    pub fallthrough_arm: Vec<Instruction>,
    /// Canonical registers merged at the join, with bit width.
    pub merged: Vec<MergedVar>,
}

/// A register merged across the two arms of a [`SliceMerge`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedVar {
    /// Canonical (parent) register name, e.g. `"rax"`.
    pub name: String,
    /// Bit width of the merged value.
    pub bits: u8,
}

/// Whether the slice represents the full data-flow chain or was cut
/// short by a limit / unsupported construct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SliceStatus {
    /// Every dependency was resolved within the basic block (the
    /// remaining `roots` are by definition external inputs).
    Complete,
    /// The slicer hit an unsupported construct or a limit.
    Truncated {
        /// Human-readable explanation, e.g. `"call encountered at 0x401050"`.
        reason: String,
    },
}

impl SliceStatus {
    /// Build a [`SliceStatus::Truncated`] from a free-form reason.
    #[must_use]
    pub fn truncated(reason: impl Into<String>) -> Self {
        Self::Truncated {
            reason: reason.into(),
        }
    }
}

/// Slice the data-flow leading to `candidate` inside `function` under
/// `arch`.
///
/// `function` must own a block whose `address` matches
/// `candidate.block`; otherwise the result is truncated with reason
/// `"block not found"`. The `arch` argument is forwarded to
/// [`crate::effect::analyze`] so the slicer recognises x86 or
/// `AArch64` mnemonics correctly.
///
/// When `limits.max_basic_blocks > 1`, the walk continues past the
/// candidate's block into a unique-predecessor chain. Joins (≥2
/// predecessors) terminate the walk as [`SliceStatus::Truncated`]
/// — see the module docs.
#[must_use]
pub fn slice_branch(
    candidate: &BranchCandidate,
    function: &Function,
    limits: &SliceLimits,
    arch: Arch,
) -> Slice {
    let Some(block) = function
        .blocks
        .iter()
        .find(|b| b.address == candidate.block)
    else {
        return Slice {
            branch: candidate.clone(),
            instructions: Vec::new(),
            status: SliceStatus::truncated("block not found in function"),
            roots: Vec::new(),
            treat_truncation_as_inputs: false,
            merges: Vec::new(),
        };
    };

    let Some(jcc_idx) = block
        .instructions
        .iter()
        .position(|i| i.address == candidate.address)
    else {
        return Slice {
            branch: candidate.clone(),
            instructions: Vec::new(),
            status: SliceStatus::truncated("candidate instruction not found in block"),
            roots: Vec::new(),
            treat_truncation_as_inputs: false,
            merges: Vec::new(),
        };
    };

    walk_backwards(candidate, function, block, jcc_idx, limits, arch)
}

/// Mutable state shared across single- and multi-block walks.
struct WalkState {
    /// Instructions kept by the slicer, accumulated in reverse-walk
    /// order. Reversed once at the end to produce execution order.
    kept: Vec<Instruction>,
    /// Canonical register names still needed by the slice (their
    /// definition site has not yet been found).
    live: BTreeSet<&'static str>,
    /// Stack slots still needed by the slice.
    live_stack: BTreeSet<String>,
    /// `true` while the slice still needs a flag-defining instruction
    /// (`cmp`, `test`, flag-setting arithmetic, …). Always `false`
    /// for compare-and-branch families that bypass NZCV.
    needs_flags: bool,
    /// Number of basic blocks the walk has entered so far.
    blocks_visited: u32,
    /// Block addresses already visited — used for cycle detection in
    /// multi-block walks.
    visited: BTreeSet<Address>,
    /// Forwarded copy of [`SliceLimits::unknowns_on_truncation`].
    /// Stamped onto the resulting [`Slice`] only when the walk ends
    /// in a truncated state — a complete slice never needs the
    /// downstream "treat roots as inputs" handling.
    unknowns_on_truncation: bool,
    /// Bounded simple-diamond Φ-merges recovered while walking. Moved
    /// verbatim onto the resulting [`Slice::merges`].
    merges: Vec<SliceMerge>,
}

impl WalkState {
    fn new(candidate: &BranchCandidate, arch: Arch, unknowns_on_truncation: bool) -> Self {
        // Compare-and-branch (`cbz`/`cbnz`/`tbz`/`tbnz`) doesn't read
        // NZCV — its data dependency is the register named in
        // `compare_register`. Seed the live set with that register's
        // canonical parent and disable flag tracking, otherwise the
        // slicer would walk past the `mov` / `add` that wrote the
        // register looking for a `cmp` that doesn't exist.
        let needs_flags = !matches!(
            candidate.condition,
            BranchCondition::RegisterZero
                | BranchCondition::RegisterNotZero
                | BranchCondition::BitZero
                | BranchCondition::BitNotZero
        );
        let mut live: BTreeSet<&'static str> = BTreeSet::new();
        if let Some(raw) = &candidate.compare_register
            && let Some(layout) = crate::registers::register_layout(raw, arch)
        {
            live.insert(layout.parent);
        }
        Self {
            kept: Vec::new(),
            live,
            live_stack: BTreeSet::new(),
            needs_flags,
            blocks_visited: 0,
            visited: BTreeSet::new(),
            unknowns_on_truncation,
            merges: Vec::new(),
        }
    }

    fn into_complete_slice(self, candidate: &BranchCandidate) -> Slice {
        let mut instructions = self.kept;
        instructions.reverse();
        let mut roots: Vec<String> = self.live.iter().map(|r| (*r).to_string()).collect();
        roots.extend(self.live_stack.iter().cloned());
        Slice {
            branch: candidate.clone(),
            instructions,
            status: SliceStatus::Complete,
            roots,
            treat_truncation_as_inputs: false,
            merges: self.merges,
        }
    }

    fn into_truncated_slice(self, candidate: &BranchCandidate, reason: String) -> Slice {
        let mut instructions = self.kept;
        instructions.reverse();
        let mut roots: Vec<String> = self.live.iter().map(|r| (*r).to_string()).collect();
        roots.extend(self.live_stack.iter().cloned());
        let treat_truncation_as_inputs = self.unknowns_on_truncation;
        Slice {
            branch: candidate.clone(),
            instructions,
            status: SliceStatus::truncated(reason),
            roots,
            treat_truncation_as_inputs,
            merges: self.merges,
        }
    }

    /// Decide whether the slice is [`SliceStatus::Complete`] or
    /// [`SliceStatus::Truncated`] after exhausting the predecessor
    /// chain at a block entry.
    ///
    /// A flag-dependent branch (`jcc`/`b.<cond>`) needs a definite
    /// flag-defining instruction; if `needs_flags` is still true at
    /// this point, the slice is unsound and gets truncated. Otherwise
    /// the remaining `live` and `live_stack` entries are treated as
    /// external inputs (roots) and the slice is complete — same
    /// semantic the single-block walker had since Phase 3.
    fn finalize_at_block_entry(self, candidate: &BranchCandidate, structural: &str) -> Slice {
        if self.needs_flags {
            let pending = pending_summary(&self);
            let reason = format!(
                "no flag-defining instruction found in slice ({structural}; pending: {pending})"
            );
            return self.into_truncated_slice(candidate, reason);
        }
        self.into_complete_slice(candidate)
    }
}

/// What happened while walking a single block in reverse.
enum BlockWalkOutcome {
    /// The live set was satisfied — the slice is [`SliceStatus::Complete`].
    Done,
    /// Reached the block entry with live entries still pending. The
    /// caller decides whether to follow a predecessor edge.
    BlockEntry,
    /// Hit a hard stop inside the block (call, memory, unsupported
    /// mnemonic, instruction limit). The slice is truncated.
    Truncated(String),
}

fn walk_backwards(
    candidate: &BranchCandidate,
    function: &Function,
    start_block: &BasicBlock,
    jcc_idx: usize,
    limits: &SliceLimits,
    arch: Arch,
) -> Slice {
    let mut state = WalkState::new(candidate, arch, limits.unknowns_on_truncation);
    let mut current_block = start_block;
    let mut start_idx = jcc_idx;

    loop {
        state.visited.insert(current_block.address);
        state.blocks_visited = state.blocks_visited.saturating_add(1);

        match walk_block(&mut state, current_block, start_idx, limits, arch) {
            BlockWalkOutcome::Done => {
                debug!(
                    target: "r2smt::slicer",
                    at = %candidate.address,
                    kept = state.kept.len(),
                    blocks = state.blocks_visited,
                    "slice complete"
                );
                return state.into_complete_slice(candidate);
            }
            BlockWalkOutcome::Truncated(reason) => {
                return state.into_truncated_slice(candidate, reason);
            }
            BlockWalkOutcome::BlockEntry => {
                let preds = predecessors_of(function, current_block.address);
                match preds.len() {
                    0 => {
                        return state.finalize_at_block_entry(
                            candidate,
                            "no predecessor block (function entry)",
                        );
                    }
                    1 => {
                        let pred_addr = preds[0];
                        if state.visited.contains(&pred_addr) {
                            let structural = format!("cycle back to {pred_addr}");
                            return state.finalize_at_block_entry(candidate, &structural);
                        }
                        if state.blocks_visited >= limits.max_basic_blocks {
                            let structural = format!(
                                "block budget {limit} exhausted",
                                limit = limits.max_basic_blocks
                            );
                            return state.finalize_at_block_entry(candidate, &structural);
                        }
                        let Some(pred_block) =
                            function.blocks.iter().find(|b| b.address == pred_addr)
                        else {
                            let reason =
                                format!("predecessor block {pred_addr} not present in function");
                            return state.into_truncated_slice(candidate, reason);
                        };
                        current_block = pred_block;
                        start_idx = pred_block.instructions.len();
                    }
                    n => {
                        if limits.allow_join_merge
                            && let Some(merge) = try_build_diamond_merge(
                                function,
                                current_block,
                                &preds,
                                &state,
                                arch,
                            )
                        {
                            // Bounded simple-diamond Φ-merge: the join
                            // is fully resolved by lowering each
                            // pending register to `Ite(head_cond,
                            // taken, fallthrough)` ahead of the join
                            // body. The merged registers are no longer
                            // roots; head / arm-external reads become
                            // ordinary free inputs at SSA. Sound: a
                            // path-insensitive predicate keeps its
                            // high-confidence verdict, a genuinely
                            // path-sensitive one widens to
                            // `BothPossible` — never a fabricated
                            // `AlwaysX`.
                            for mv in &merge.merged {
                                if let Some(layout) =
                                    crate::registers::register_layout(&mv.name, arch)
                                {
                                    state.live.remove(layout.parent);
                                } else {
                                    state.live.remove(mv.name.as_str());
                                }
                            }
                            state.merges.push(merge);
                            let structural = format!(
                                "join at {addr} resolved by bounded Φ-merge",
                                addr = current_block.address
                            );
                            return state.finalize_at_block_entry(candidate, &structural);
                        }
                        let structural = if limits.allow_join_merge {
                            // Sound free-input boundary: promote the
                            // join-pending live set to free symbolic
                            // inputs (scoped to joins only). This only
                            // widens an `AlwaysX` to `BothPossible`,
                            // never fabricates a verdict.
                            state.unknowns_on_truncation = true;
                            format!(
                                "join at {addr} ({n} predecessors) — merged as free inputs (allow_join_merge)",
                                addr = current_block.address
                            )
                        } else {
                            format!(
                                "join at {addr} ({n} predecessors) — slicer cannot Φ-merge",
                                addr = current_block.address
                            )
                        };
                        return state.finalize_at_block_entry(candidate, &structural);
                    }
                }
            }
        }
    }
}

/// Unmodeled mnemonics that are architecturally side-effect-free with
/// respect to a data-flow slice: they define no register, no flag, and
/// access no memory the slicer tracks. Stepping over them is sound; every
/// other [`InstructionKind::Other`] instruction has unknown effects and
/// must truncate a still-pending slice rather than be silently skipped.
const SIDE_EFFECT_FREE_OTHER: &[&str] = &[
    "nop", "endbr64", "endbr32", "fnop", "pause", "yield", "lfence", "sfence", "mfence", "dmb",
    "dsb", "isb",
];

fn walk_block(
    state: &mut WalkState,
    block: &BasicBlock,
    start_idx: usize,
    limits: &SliceLimits,
    arch: Arch,
) -> BlockWalkOutcome {
    for i in (0..start_idx).rev() {
        if state.kept.len() >= limits.max_instructions {
            return BlockWalkOutcome::Truncated("instruction limit reached".into());
        }

        let insn = &block.instructions[i];
        let effect = analyze(insn, arch);

        if !limits.allow_calls && effect.is_call {
            return BlockWalkOutcome::Truncated(format!("call at {addr}", addr = insn.address));
        }
        if !limits.allow_memory && effect.has_memory_access {
            return BlockWalkOutcome::Truncated(format!(
                "memory access at {addr}",
                addr = insn.address
            ));
        }

        let touches_flags = state.needs_flags && effect.defines_flags;
        let touches_live: Vec<&'static str> = effect
            .defs
            .iter()
            .filter(|d| state.live.contains(*d))
            .copied()
            .collect();
        let touches_live_stack: Vec<String> = effect
            .stack_defs
            .iter()
            .filter(|d| state.live_stack.contains(*d))
            .cloned()
            .collect();

        if effect.kind == InstructionKind::Other {
            if SIDE_EFFECT_FREE_OTHER.contains(&insn.mnemonic.as_str()) {
                continue;
            }
            let slice_pending =
                state.needs_flags || !state.live.is_empty() || !state.live_stack.is_empty();
            if slice_pending {
                return BlockWalkOutcome::Truncated(format!(
                    "unsupported '{mnem}' at {addr} may redefine slice state",
                    mnem = insn.mnemonic,
                    addr = insn.address
                ));
            }
            continue;
        }

        if !touches_flags && touches_live.is_empty() && touches_live_stack.is_empty() {
            // P26: keep instructions that touch memory even when they
            // don't define a currently-live register/flag. A `str`
            // doesn't satisfy any pending live name, but its byte-
            // granular effect on memory state IS observed by any
            // downstream `ldr` that the encoder will lower through
            // its `Ite`-chain memory model. Skipping it here would
            // silently drop the write, unsoundly making the later
            // load free. Gated on `allow_memory` so the default
            // (no `--allow-memory`) path is byte-identical to pre-P26.
            if !(limits.allow_memory && effect.has_memory_access) {
                continue;
            }
        }

        state.kept.push(insn.clone());
        if touches_flags {
            state.needs_flags = false;
        }
        for def in &touches_live {
            state.live.remove(def);
        }
        for def in &touches_live_stack {
            state.live_stack.remove(def);
        }
        for u in &effect.uses {
            state.live.insert(*u);
        }
        for u in &effect.stack_uses {
            state.live_stack.insert(u.clone());
        }
        // If the kept instruction reads NZCV (AArch64 conditional-
        // select family, AArch32 predicated execution, …) we must
        // keep walking backwards until we find a flag-defining
        // instruction so the upstream `cmp` ends up in the slice.
        // This is sound regardless of whether the original branch
        // was flag-based: introducing an extra flag definer can only
        // *narrow* the verdict.
        if effect.reads_flags {
            state.needs_flags = true;
        }

        if !state.needs_flags && state.live.is_empty() && state.live_stack.is_empty() {
            return BlockWalkOutcome::Done;
        }
    }
    BlockWalkOutcome::BlockEntry
}

/// Collect the addresses of every block in `function` whose
/// successor list contains `target`.
fn predecessors_of(function: &Function, target: Address) -> Vec<Address> {
    function
        .blocks
        .iter()
        .filter(|b| b.successors.contains(&target))
        .map(|b| b.address)
        .collect()
}

/// The unique predecessor of `target`, or `None` when it has zero or
/// more than one predecessor.
fn single_pred(function: &Function, target: Address) -> Option<Address> {
    let preds = predecessors_of(function, target);
    if preds.len() == 1 {
        Some(preds[0])
    } else {
        None
    }
}

/// Maximum instructions allowed in a single Φ-merge arm or in the
/// head-condition slice. Keeps bounded-diamond recovery cheap and its
/// soundness obvious; a longer arm falls back to the sound free-input
/// boundary instead of being merged.
const MERGE_ARM_MAX_INSTRUCTIONS: usize = 16;

/// Human-readable summary of what the slicer still needs at a
/// truncation point — surfaced inside `SliceStatus::Truncated::reason`
/// so failure modes are self-describing.
fn pending_summary(state: &WalkState) -> String {
    let mut parts: Vec<String> = Vec::new();
    if state.needs_flags {
        parts.push("flag-defining instruction".into());
    }
    if !state.live.is_empty() {
        let regs: Vec<&'static str> = state.live.iter().copied().collect();
        parts.push(format!("registers {regs:?}"));
    }
    if !state.live_stack.is_empty() {
        let slots: Vec<&str> = state.live_stack.iter().map(String::as_str).collect();
        parts.push(format!("stack slots {slots:?}"));
    }
    if parts.is_empty() {
        // Should not happen — `BlockEntry` only fires while something
        // is unresolved. Surface a sentinel rather than panic.
        "<nothing>".into()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests;
