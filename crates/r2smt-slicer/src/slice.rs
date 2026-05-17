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
            if touches_flags || !touches_live.is_empty() || !touches_live_stack.is_empty() {
                return BlockWalkOutcome::Truncated(format!(
                    "unsupported '{mnem}' at {addr} touches slice",
                    mnem = insn.mnemonic,
                    addr = insn.address
                ));
            }
            continue;
        }

        if !touches_flags && touches_live.is_empty() && touches_live_stack.is_empty() {
            continue;
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
fn try_build_diamond_merge(
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
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};

    use super::*;
    use crate::collector::collect_branches;

    fn op(raw: &str, kind: OperandKind) -> Operand {
        Operand {
            raw: raw.into(),
            kind,
        }
    }

    fn insn(addr: u64, size: u8, mnemonic: &str, operands: Vec<Operand>) -> Instruction {
        Instruction {
            address: Address(addr),
            size,
            bytes: vec![],
            mnemonic: mnemonic.into(),
            operands,
            esil: None,
            pcode: None,
            is_thumb: false,
        }
    }

    fn one_block_program(insns: Vec<Instruction>) -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: insns,
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    fn slice_first(program: &Program) -> Slice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        slice_branch(
            cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        )
    }

    #[test]
    fn canonical_opaque_predicate_yields_complete_slice() {
        // The SCC example: ((ecx * ecx) & 1) == 2 — always false.
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                3,
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "and",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_100b,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        assert_eq!(slice.status, SliceStatus::Complete);
        let mnemonics: Vec<_> = slice
            .instructions
            .iter()
            .map(|i| i.mnemonic.as_str())
            .collect();
        assert_eq!(mnemonics, vec!["mov", "imul", "and", "cmp"]);
        assert_eq!(slice.roots, vec!["rcx".to_string()]);
    }

    #[test]
    fn constant_propagation_has_no_roots() {
        // mov eax, 1; cmp eax, 1; jne junk → always false (constant).
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                5,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        assert_eq!(slice.status, SliceStatus::Complete);
        assert_eq!(slice.instructions.len(), 2);
        assert!(slice.roots.is_empty(), "roots: {:?}", slice.roots);
    }

    #[test]
    fn xor_zero_idiom_terminates_slice_without_roots() {
        // xor eax,eax; test eax,eax; jnz junk
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                2,
                "xor",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                6,
                "jnz",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        assert_eq!(slice.status, SliceStatus::Complete);
        let mnemonics: Vec<_> = slice
            .instructions
            .iter()
            .map(|i| i.mnemonic.as_str())
            .collect();
        assert_eq!(mnemonics, vec!["xor", "test"]);
        assert!(slice.roots.is_empty());
    }

    #[test]
    fn call_before_flag_producer_truncates() {
        // call f; cmp eax, 0; je → truncated because the call destroys
        // every assumption we have about eax.
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1008,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        assert!(matches!(slice.status, SliceStatus::Truncated { .. }));
        // The flag producer (`cmp`) is still in the slice; the truncation
        // happens when we walk into the `call`.
        assert_eq!(slice.instructions.len(), 1);
        assert_eq!(slice.instructions[0].mnemonic, "cmp");
    }

    #[test]
    fn memory_load_truncates_when_not_allowed() {
        // mov eax, [rax]; cmp eax, 5; je — `[rax]` is unresolved memory
        // (not a stack slot) so the slicer must still truncate.
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                3,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("[rax]", OperandKind::Memory),
                ],
            ),
            insn(
                0x40_1003,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("5", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_1006,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        assert!(matches!(slice.status, SliceStatus::Truncated { .. }));
    }

    #[test]
    fn missing_flag_producer_truncates() {
        // mov eax, ebx; jne junk — no cmp/test ever sets ZF.
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("ebx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("flag-defining"));
    }

    #[test]
    fn instruction_limit_truncates() {
        let mut insns: Vec<Instruction> = Vec::new();
        // 50 nops worth of dependency chain.
        for i in 0_u32..50 {
            insns.push(insn(
                0x40_1000 + u64::from(i),
                1,
                "add",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ));
        }
        insns.push(insn(
            0x40_1100,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("100", OperandKind::Immediate),
            ],
        ));
        insns.push(insn(
            0x40_1103,
            6,
            "jne",
            vec![op("0x401080", OperandKind::Immediate)],
        ));
        let program = one_block_program(insns);
        let cand = collect_branches(&program).into_iter().next().unwrap();
        let limits = SliceLimits {
            max_instructions: 8,
            ..SliceLimits::default()
        };
        let slice = slice_branch(&cand, &program.functions[0], &limits, program.arch);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected truncated");
        };
        assert!(reason.contains("instruction limit"));
        assert!(slice.instructions.len() <= 8);
    }

    #[test]
    fn json_round_trips() {
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                2,
                "xor",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1002,
                2,
                "test",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_1004,
                6,
                "jnz",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let slice = slice_first(&program);
        let json = serde_json::to_string(&slice).unwrap();
        let back: Slice = serde_json::from_str(&json).unwrap();
        assert_eq!(back, slice);
    }

    // --- Multi-block slicing ---

    /// Build a function with two linear blocks: `A` falls through to
    /// `B`. The branch lives at the end of `B`.
    fn two_block_function(a_insns: Vec<Instruction>, b_insns: Vec<Instruction>) -> Program {
        let b_addr = b_insns.first().map_or(Address(0x40_2000), |i| i.address);
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![
                    BasicBlock {
                        address: Address(0x40_1000),
                        instructions: a_insns,
                        successors: vec![b_addr],
                    },
                    BasicBlock {
                        address: b_addr,
                        instructions: b_insns,
                        successors: vec![],
                    },
                ],
                is_thumb: false,
            }],
        }
    }

    fn slice_first_with(program: &Program, limits: &SliceLimits) -> Slice {
        let candidates = collect_branches(program);
        let cand = candidates.first().expect("at least one branch");
        slice_branch(cand, &program.functions[0], limits, program.arch)
    }

    #[test]
    fn multi_block_resolves_definition_in_predecessor() {
        // Block A: `mov ecx, 5`
        // Block B: `imul eax, ecx, ecx ; and eax, 1 ; cmp eax, 2 ; jne junk`
        //   → ((5 * 5) & 1) == 2 → false, but the slicer only needs the
        //     mov+imul+and+cmp chain to see that. With max_blocks=2 the
        //     slicer pulls in the `mov ecx, 5` and rcx drops out of roots.
        let a = vec![insn(
            0x40_1000,
            5,
            "mov",
            vec![
                op("ecx", OperandKind::Register),
                op("5", OperandKind::Immediate),
            ],
        )];
        let b = vec![
            insn(
                0x40_2000,
                2,
                "mov",
                vec![
                    op("eax", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_2002,
                3,
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("eax", OperandKind::Register),
                ],
            ),
            insn(
                0x40_2005,
                3,
                "and",
                vec![
                    op("eax", OperandKind::Register),
                    op("1", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_2008,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_200b,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ];
        let program = two_block_function(a, b);
        let limits = SliceLimits {
            max_basic_blocks: 2,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        assert_eq!(slice.status, SliceStatus::Complete);
        let mnemonics: Vec<_> = slice
            .instructions
            .iter()
            .map(|i| i.mnemonic.as_str())
            .collect();
        // Execution order: A's mov ecx,5 comes first, then B's chain.
        assert_eq!(mnemonics, vec!["mov", "mov", "imul", "and", "cmp"]);
        assert!(
            slice.roots.is_empty(),
            "multi-block walk should drop rcx root, got {:?}",
            slice.roots
        );
    }

    #[test]
    fn single_block_default_keeps_root_when_definition_lives_upstream() {
        // Same fixture as the multi-block test, but with the default
        // limits (max_basic_blocks=1). The slicer stops in B with rcx
        // as an external input.
        let a = vec![insn(
            0x40_1000,
            5,
            "mov",
            vec![
                op("ecx", OperandKind::Register),
                op("5", OperandKind::Immediate),
            ],
        )];
        let b = vec![
            insn(
                0x40_2000,
                3,
                "imul",
                vec![
                    op("eax", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                    op("ecx", OperandKind::Register),
                ],
            ),
            insn(
                0x40_2003,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("2", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_2006,
                6,
                "jne",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ];
        let program = two_block_function(a, b);
        let slice = slice_first_with(&program, &SliceLimits::default());
        // Default (max_basic_blocks=1) — the slicer stays in block B
        // and treats rcx as external. Same Complete-with-roots
        // semantic as Phase 3.
        assert_eq!(slice.status, SliceStatus::Complete);
        assert_eq!(slice.roots, vec!["rcx".to_string()]);
    }

    #[test]
    fn multi_block_resolves_flag_producer_in_predecessor() {
        // Block A: `cmp eax, 0` (sets ZF).
        // Block B: `je junk` — needs ZF.
        // Without multi-block this truncates with "no flag-defining
        // instruction found". With max_blocks=2 the slicer pulls in
        // the cmp and the slice is Complete.
        let a = vec![insn(
            0x40_1000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        )];
        let b = vec![insn(
            0x40_2000,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        )];
        let program = two_block_function(a, b);
        let limits = SliceLimits {
            max_basic_blocks: 2,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        assert_eq!(slice.status, SliceStatus::Complete);
        let mnemonics: Vec<_> = slice
            .instructions
            .iter()
            .map(|i| i.mnemonic.as_str())
            .collect();
        assert_eq!(mnemonics, vec!["cmp"]);
    }

    #[test]
    fn multi_block_join_truncates_with_phi_reason() {
        // CFG: A and C both fall through to B. B ends with `je junk`.
        // The slicer walks B in reverse, reaches block entry while
        // still needing flag info, sees TWO predecessors (A and C),
        // and truncates with a join reason.
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![
                    BasicBlock {
                        address: Address(0x40_1000),
                        instructions: vec![insn(
                            0x40_1000,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("0", OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_2000)],
                    },
                    BasicBlock {
                        address: Address(0x40_1100),
                        instructions: vec![insn(
                            0x40_1100,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_2000)],
                    },
                    BasicBlock {
                        address: Address(0x40_2000),
                        instructions: vec![insn(
                            0x40_2000,
                            6,
                            "je",
                            vec![op("0x401080", OperandKind::Immediate)],
                        )],
                        successors: vec![],
                    },
                ],
                is_thumb: false,
            }],
        };
        let limits = SliceLimits {
            max_basic_blocks: 4,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("join"), "reason was: {reason}");
        assert!(reason.contains("predecessors"), "reason was: {reason}");
        assert!(reason.contains("flag-defining"), "reason was: {reason}");
    }

    // --- Phase 6: opt-in sound join → free-input boundary ---

    fn join_program() -> Program {
        // A (cmp eax,0) and C (cmp eax,1) both fall through to B
        // (`je junk`). Walking B in reverse hits a 2-predecessor join.
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![
                    BasicBlock {
                        address: Address(0x40_1000),
                        instructions: vec![insn(
                            0x40_1000,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("0", OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_2000)],
                    },
                    BasicBlock {
                        address: Address(0x40_1100),
                        instructions: vec![insn(
                            0x40_1100,
                            3,
                            "cmp",
                            vec![
                                op("eax", OperandKind::Register),
                                op("1", OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_2000)],
                    },
                    BasicBlock {
                        address: Address(0x40_2000),
                        instructions: vec![insn(
                            0x40_2000,
                            6,
                            "je",
                            vec![op("0x401080", OperandKind::Immediate)],
                        )],
                        successors: vec![],
                    },
                ],
                is_thumb: false,
            }],
        }
    }

    #[test]
    fn test_join_default_off_truncates_byte_identical() {
        let program = join_program();
        let limits = SliceLimits {
            max_basic_blocks: 4,
            ..SliceLimits::default()
        };
        assert!(!limits.allow_join_merge, "default must be off");
        let slice = slice_first_with(&program, &limits);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("cannot Φ-merge"), "reason: {reason}");
        assert!(
            !slice.treat_truncation_as_inputs,
            "default off must not promote join-live to free inputs"
        );
    }

    #[test]
    fn test_join_merge_promotes_live_to_free_inputs_soundly() {
        let program = join_program();
        let limits = SliceLimits {
            max_basic_blocks: 4,
            allow_join_merge: true,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        // Soundness guard: a join is never reported `Complete` — the
        // verdict must stay derivable only via free inputs (widen-only,
        // confidence-downgraded), never claimed as a resolved slice.
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("merged as free inputs"), "reason: {reason}");
        assert!(
            slice.treat_truncation_as_inputs,
            "allow_join_merge must promote the join-live set to free inputs"
        );
    }

    #[test]
    fn test_allow_join_merge_does_not_promote_non_join_truncations() {
        // A `call` truncation in a single block (no join). The
        // join-scoped promotion must not leak to call / memory /
        // unsupported truncations — only the global
        // `unknowns_on_truncation` does that.
        let program = one_block_program(vec![
            insn(
                0x40_1000,
                5,
                "call",
                vec![op("0x402000", OperandKind::Immediate)],
            ),
            insn(
                0x40_1005,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ]);
        let limits = SliceLimits {
            allow_join_merge: true,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("call"), "reason: {reason}");
        assert!(
            !slice.treat_truncation_as_inputs,
            "join-merge must be scoped to joins; call truncation unaffected"
        );
    }

    #[test]
    fn multi_block_budget_exhausted_with_needs_flags_truncates() {
        // Three linear blocks A → B → C; C ends with `je junk`. With
        // max_blocks=2 the slicer can visit only C and B before the
        // budget runs out. Since B does not set ZF, needs_flags is
        // still true — slice is Truncated.
        let a = vec![insn(
            0x40_1000,
            3,
            "cmp",
            vec![
                op("eax", OperandKind::Register),
                op("0", OperandKind::Immediate),
            ],
        )];
        let b = vec![insn(
            0x40_1100,
            2,
            "mov",
            vec![
                op("ebx", OperandKind::Register),
                op("ecx", OperandKind::Register),
            ],
        )];
        let c = vec![insn(
            0x40_1200,
            6,
            "je",
            vec![op("0x401080", OperandKind::Immediate)],
        )];
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![
                    BasicBlock {
                        address: Address(0x40_1000),
                        instructions: a,
                        successors: vec![Address(0x40_1100)],
                    },
                    BasicBlock {
                        address: Address(0x40_1100),
                        instructions: b,
                        successors: vec![Address(0x40_1200)],
                    },
                    BasicBlock {
                        address: Address(0x40_1200),
                        instructions: c,
                        successors: vec![],
                    },
                ],
                is_thumb: false,
            }],
        };
        let limits = SliceLimits {
            max_basic_blocks: 2,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("budget"), "reason was: {reason}");
        assert!(reason.contains("flag-defining"), "reason was: {reason}");
    }

    #[test]
    fn multi_block_cycle_back_to_self_truncates() {
        // Self-loop: block A ends with `je 0x401000` (target is A
        // itself). The branch's basic block is A. Walking backward in
        // A doesn't find ZF (no cmp), so we look at A's predecessors
        // — A is its own predecessor, and visited contains A, so we
        // report a cycle.
        let program = Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![
                        insn(
                            0x40_1000,
                            2,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op("ebx", OperandKind::Register),
                            ],
                        ),
                        insn(
                            0x40_1002,
                            6,
                            "je",
                            vec![op("0x401000", OperandKind::Immediate)],
                        ),
                    ],
                    successors: vec![Address(0x40_1000)],
                }],
                is_thumb: false,
            }],
        };
        let limits = SliceLimits {
            max_basic_blocks: 4,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.contains("cycle"), "reason was: {reason}");
        assert!(reason.contains("flag-defining"), "reason was: {reason}");
    }

    #[test]
    fn multi_block_call_in_predecessor_truncates_inside_walk() {
        // Block A: `call f ; ret_addr_unused`. Block B: `cmp eax, 0 ;
        // je junk`. With multi-block walk enabled and `allow_calls`
        // off, walking back into A hits the `call` and truncates with
        // the normal "call at <addr>" reason — same code path as the
        // single-block walker, just exercised across a block edge.
        let a = vec![insn(
            0x40_1000,
            5,
            "call",
            vec![op("0x402000", OperandKind::Immediate)],
        )];
        let b = vec![
            insn(
                0x40_2000,
                3,
                "cmp",
                vec![
                    op("eax", OperandKind::Register),
                    op("0", OperandKind::Immediate),
                ],
            ),
            insn(
                0x40_2003,
                6,
                "je",
                vec![op("0x401080", OperandKind::Immediate)],
            ),
        ];
        let program = two_block_function(a, b);
        let limits = SliceLimits {
            max_basic_blocks: 4,
            ..SliceLimits::default()
        };
        let slice = slice_first_with(&program, &limits);
        // The cmp resolves needs_flags in B; the eax live entry then
        // drives us into A, which truncates on the `call`.
        let SliceStatus::Truncated { reason } = &slice.status else {
            panic!("expected Truncated, got {:?}", slice.status);
        };
        assert!(reason.starts_with("call at "), "reason was: {reason}");
    }

    // --- Bounded simple-diamond Φ-merge ---

    /// `H: cmp ecx,0; je THEN` — a full diamond whose two arms each set
    /// `eax`, reconverging at `JOIN: cmp eax,7; je analysed`. The
    /// taken edge (`ZF==1` ⇒ `ecx==0`) reaches `THEN`.
    fn diamond_program(then_imm: &str, else_imm: &str) -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![
                    BasicBlock {
                        address: Address(0x40_1000),
                        instructions: vec![
                            insn(
                                0x40_1000,
                                3,
                                "cmp",
                                vec![
                                    op("ecx", OperandKind::Register),
                                    op("0", OperandKind::Immediate),
                                ],
                            ),
                            insn(
                                0x40_1003,
                                6,
                                "je",
                                vec![op("0x401100", OperandKind::Immediate)],
                            ),
                        ],
                        successors: vec![Address(0x40_1100), Address(0x40_1009)],
                    },
                    BasicBlock {
                        address: Address(0x40_1009),
                        instructions: vec![insn(
                            0x40_1009,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op(else_imm, OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_1200)],
                    },
                    BasicBlock {
                        address: Address(0x40_1100),
                        instructions: vec![insn(
                            0x40_1100,
                            5,
                            "mov",
                            vec![
                                op("eax", OperandKind::Register),
                                op(then_imm, OperandKind::Immediate),
                            ],
                        )],
                        successors: vec![Address(0x40_1200)],
                    },
                    BasicBlock {
                        address: Address(0x40_1200),
                        instructions: vec![
                            insn(
                                0x40_1200,
                                3,
                                "cmp",
                                vec![
                                    op("eax", OperandKind::Register),
                                    op("7", OperandKind::Immediate),
                                ],
                            ),
                            insn(
                                0x40_1203,
                                6,
                                "je",
                                vec![op("0x401300", OperandKind::Immediate)],
                            ),
                        ],
                        successors: vec![],
                    },
                ],
                is_thumb: false,
            }],
        }
    }

    fn slice_join(program: &Program, limits: &SliceLimits) -> Slice {
        let cands = collect_branches(program);
        let join = cands
            .iter()
            .find(|c| c.address == Address(0x40_1203))
            .expect("join branch present");
        slice_branch(join, &program.functions[0], limits, program.arch)
    }

    #[test]
    fn test_bounded_diamond_merge_recovered_as_complete() {
        let program = diamond_program("5", "5");
        let limits = SliceLimits {
            max_basic_blocks: 8,
            allow_join_merge: true,
            ..SliceLimits::default()
        };
        let slice = slice_join(&program, &limits);
        assert_eq!(
            slice.status,
            SliceStatus::Complete,
            "a fully-resolved diamond is a sound complete slice"
        );
        assert!(
            !slice.treat_truncation_as_inputs,
            "a recovered diamond is resolved, not promoted to free inputs"
        );
        assert_eq!(slice.merges.len(), 1);
        let merge = &slice.merges[0];
        assert_eq!(
            merge
                .merged
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            vec!["rax"]
        );
        assert_eq!(merge.head.taken_target, Some(Address(0x40_1100)));
        assert_eq!(merge.taken_arm.len(), 1);
        assert_eq!(merge.fallthrough_arm.len(), 1);
        assert_eq!(
            merge
                .head_instructions
                .iter()
                .map(|i| i.mnemonic.as_str())
                .collect::<Vec<_>>(),
            vec!["cmp"]
        );
        assert!(
            !slice.roots.contains(&"rax".to_string()),
            "merged register is resolved by the Ite, not a root"
        );
    }

    #[test]
    fn test_bounded_diamond_default_off_byte_identical() {
        // With `allow_join_merge` off the join handling is unchanged:
        // `needs_flags` is already satisfied by `cmp eax,7` in the
        // join block, so the pre-existing finaliser returns a
        // `Complete` slice with the still-pending `rax` carried as an
        // *unresolved root* (→ free SSA input → sound, imprecise).
        // The contract this test pins: off recovers no merge and the
        // merged register stays a root exactly as before.
        let program = diamond_program("5", "5");
        let limits = SliceLimits {
            max_basic_blocks: 8,
            ..SliceLimits::default()
        };
        assert!(!limits.allow_join_merge);
        let slice = slice_join(&program, &limits);
        assert_eq!(slice.status, SliceStatus::Complete);
        assert!(
            slice.merges.is_empty(),
            "default off must recover no Φ-merge"
        );
        assert!(
            slice.roots.contains(&"rax".to_string()),
            "off: merged register stays an unresolved root (byte-identical)"
        );
        assert!(!slice.treat_truncation_as_inputs);
    }

    #[test]
    fn test_bounded_diamond_lift_polarity_taken_edge_is_then_branch() {
        use r2smt_ir::expr::Expr;
        use r2smt_ir::stmt::IrStmt;

        // THEN sets eax=0xb (11), ELSE sets eax=0x16 (22). The head
        // `je` is taken (→ THEN) exactly when its condition is true,
        // so the lowered `Ite` must put THEN's value in `then_expr`.
        let program = diamond_program("11", "22");
        let limits = SliceLimits {
            max_basic_blocks: 8,
            allow_join_merge: true,
            ..SliceLimits::default()
        };
        let slice = slice_join(&program, &limits);
        let lifted = crate::lift::lift_slice(&slice, program.arch);
        let (then_expr, else_expr) = lifted
            .statements
            .iter()
            .find_map(|s| match s {
                IrStmt::Assign {
                    dst,
                    src:
                        Expr::Ite {
                            then_expr,
                            else_expr,
                            ..
                        },
                } if dst.name == "rax" => Some(((**then_expr).clone(), (**else_expr).clone())),
                _ => None,
            })
            .expect("rax := Ite(...) assignment lowered from the merge");
        assert!(
            then_expr.to_string().contains("0xb"),
            "taken (THEN) value must drive then_expr, got: {then_expr}"
        );
        assert!(
            else_expr.to_string().contains("0x16"),
            "fallthrough (ELSE) value must drive else_expr, got: {else_expr}"
        );
    }
}
