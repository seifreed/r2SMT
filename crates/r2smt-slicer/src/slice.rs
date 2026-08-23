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
//! `BothPossible` — never a fabricated verdict.
//!
//! The merge is now a general DAG recovery, always fail-closing to the
//! sound free-input boundary: multi-block straight-line arms, flag
//! merging across arms, chained / nested diamonds (the walk resumes
//! after each merge), and merged memory state (per-arm conditional
//! stores). Concrete counted self-loops are resolved by a bounded
//! forward unroll to the counter's exit constant (see `loop_unroll`);
//! non-concrete or non-convergent loops decline and stay free.

use std::collections::BTreeSet;

use r2smt_common::{Address, Arch};
use r2smt_ir::program::{BasicBlock, Function, Instruction};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::collector::BranchCandidate;
use crate::condition::BranchCondition;
use crate::effect::{InstructionKind, analyze, has_unmodellable_memory};

mod merge_detect;
use merge_detect::{head_condition_roots, try_build_diamond_merge};

mod loop_unroll;
use loop_unroll::try_unroll_counted_loop;

/// Limits applied while slicing.
///
/// `max_basic_blocks = 1` keeps the historical single-block behavior;
/// raising it enables the
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
    /// When set, an instruction whose ESIL writes flags (any string
    /// carrying `:=`) may take the ESIL rung instead of being sent to
    /// its per-mnemonic handler.
    ///
    /// Supported on x86, x86-64, `AArch32`, and `AArch64`. The CLI disables
    /// this rung automatically before radare2 6.2.0, where the required
    /// ARM `subs` ESIL fix is unavailable. Turn it off explicitly with
    /// `--no-esil-flags` to A/B the two lowerings.
    #[serde(default = "default_true")]
    pub esil_flags: bool,
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
            esil_flags: true,
        }
    }
}

/// Serde default for [`SliceLimits::esil_flags`], which is on.
const fn default_true() -> bool {
    true
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
    /// Number of kept slice instructions that execute *after* this
    /// merge (i.e. below the join in execution order). Set by the
    /// walker from `WalkState::kept.len()` at the moment the merge is
    /// recorded. The lifter uses it to interleave multiple merges with
    /// the kept-instruction stream in execution order: a merge with a
    /// larger tail executes earlier. A single merge captures the whole
    /// tail, reproducing the historical "lower merges first" order.
    #[serde(default)]
    pub tail_instructions: usize,
}

/// A register merged across the two arms of a [`SliceMerge`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedVar {
    /// Canonical (parent) register name, e.g. `"rax"`.
    pub name: String,
    /// Bit width of the merged value.
    pub bits: u16,
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

    let slice = walk_backwards(candidate, function, block, jcc_idx, limits, arch);
    reject_unpinned_rounding_mode(slice, function, candidate, arch)
}

/// Truncate a slice whose floating-point arithmetic assumes a control
/// field holds its architectural default while the containing function
/// reprograms it.
///
/// This cannot be caught by the backward walk. A control word has no
/// register name, so `ldmxcsr` / `fldcw` define nothing the live set can
/// require, and the walk stops as soon as the live set is satisfied — a
/// write ahead of the arithmetic is simply never visited. Without this
/// guard such a slice reports `Complete` at high confidence while the
/// hardware computed some other way.
///
/// x87 makes the hazard strictly worse than SSE's. Its control word
/// carries a *precision-control* field alongside the rounding mode, and
/// narrowing it changes the significand width of every subsequent
/// arithmetic result — which a fixed floating-point sort cannot express
/// at all, only decline. So the truncation here is the whole remedy
/// rather than a fallback for one that models it.
///
/// Function-scoped, so a control word installed by a *caller* remains
/// invisible. A slice reaching a call now ends there (the call is a
/// value barrier), which bounds the residue to the state the function
/// was entered with.
///
/// Arch-aware on both halves: x86 pins MXCSR and the x87 control word,
/// reprogrammed by `ldmxcsr` / `fldcw` and the state restores; ARM pins
/// FPCR / FPSCR and reprograms them with `msr` / `vmsr`.
fn reject_unpinned_rounding_mode(
    slice: Slice,
    function: &Function,
    candidate: &BranchCandidate,
    arch: Arch,
) -> Slice {
    if !matches!(slice.status, SliceStatus::Complete) {
        return slice;
    }
    if !slice
        .instructions
        .iter()
        .any(|i| crate::lift::pins_rounding_mode(i, arch))
    {
        return slice;
    }
    let Some(writer) = function
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .find(|i| crate::lift::writes_rounding_control(i, arch))
    else {
        return slice;
    };
    Slice {
        branch: candidate.clone(),
        instructions: slice.instructions,
        status: SliceStatus::truncated(format!(
            "'{mnem}' at {addr} reprograms an FPU control word; the rounding mode and precision this slice's floating-point arithmetic assumes are not sound",
            mnem = writer.mnemonic,
            addr = writer.address
        )),
        roots: slice.roots,
        treat_truncation_as_inputs: slice.treat_truncation_as_inputs,
        merges: slice.merges,
    }
}

/// Mutable state shared across single- and multi-block walks.
struct WalkState {
    /// Instructions kept by the slicer, accumulated in reverse-walk
    /// order. Reversed once at the end to produce execution order.
    kept: Vec<Instruction>,
    /// Canonical register names still needed by the slice (their
    /// definition site has not yet been found).
    live: BTreeSet<&'static str>,
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
        let roots: Vec<String> = self.live.iter().map(|r| (*r).to_string()).collect();
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
        let roots: Vec<String> = self.live.iter().map(|r| (*r).to_string()).collect();
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
    /// the remaining `live` entries are treated as external inputs
    /// (roots) and the slice is complete — same semantic the
    /// single-block walker had since Phase 3.
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

    /// Resolve a block entry reached with the live set still pending:
    /// follow a lone predecessor, recover a bounded simple-diamond
    /// Φ-merge, or stop. Extracted from [`walk_backwards`] so its join
    /// handling is a flat dispatch rather than nested inside the walk
    /// loop.
    fn resolve_block_entry<'f>(
        &mut self,
        function: &'f Function,
        current_block: &'f BasicBlock,
        preds: &[Address],
        limits: &SliceLimits,
        arch: Arch,
    ) -> WalkStep<'f> {
        match preds.len() {
            0 => WalkStep::FinalizeAtEntry("no predecessor block (function entry)".to_string()),
            1 => self.resolve_single_predecessor(function, preds[0], limits, arch),
            n => self.resolve_join(function, current_block, preds, n, limits, arch),
        }
    }

    /// Follow the unique predecessor of a block, first attempting a
    /// bounded counted-loop unroll: if the predecessor is a concrete
    /// counted self-loop, resolve the counter to its exit constant
    /// rather than walking (and truncating on) the loop. A non-concrete
    /// or non-convergent loop declines and the walk continues into the
    /// predecessor (or stops on a cycle / block-budget exhaustion).
    fn resolve_single_predecessor<'f>(
        &mut self,
        function: &'f Function,
        pred_addr: Address,
        limits: &SliceLimits,
        arch: Arch,
    ) -> WalkStep<'f> {
        if limits.allow_join_merge
            && let Some(pred_block) = function.blocks.iter().find(|b| b.address == pred_addr)
            && let Some(synth) =
                try_unroll_counted_loop(function, pred_block, &self.live, self.needs_flags, arch)
        {
            self.kept.push(synth);
            // The synthetic `mov counter, const` resolves the sole live
            // counter; nothing else pends.
            self.live.clear();
            return WalkStep::FinalizeAtEntry("counted loop unrolled to constant".to_string());
        }
        if self.visited.contains(&pred_addr) {
            return WalkStep::FinalizeAtEntry(format!("cycle back to {pred_addr}"));
        }
        if self.blocks_visited >= limits.max_basic_blocks {
            return WalkStep::FinalizeAtEntry(format!(
                "block budget {limit} exhausted",
                limit = limits.max_basic_blocks
            ));
        }
        let Some(pred_block) = function.blocks.iter().find(|b| b.address == pred_addr) else {
            return WalkStep::Truncate(format!(
                "predecessor block {pred_addr} not present in function"
            ));
        };
        WalkStep::Continue {
            block: pred_block,
            start_idx: pred_block.instructions.len(),
        }
    }

    /// Resolve a join (≥2 predecessors). Recover a bounded simple-diamond
    /// Φ-merge when enabled, otherwise stop at the sound free-input
    /// boundary.
    fn resolve_join<'f>(
        &mut self,
        function: &'f Function,
        current_block: &'f BasicBlock,
        preds: &[Address],
        n: usize,
        limits: &SliceLimits,
        arch: Arch,
    ) -> WalkStep<'f> {
        if limits.allow_join_merge
            && let Some(mut merge) =
                try_build_diamond_merge(function, current_block, preds, self, arch)
        {
            // Bounded simple-diamond Φ-merge: the join is fully resolved
            // by lowering each pending register to `Ite(head_cond, taken,
            // fallthrough)` ahead of the join body. The merged registers
            // are no longer roots; head / arm-external reads become
            // ordinary free inputs at SSA. Sound: a path-insensitive
            // predicate keeps its high-confidence verdict, a genuinely
            // path-sensitive one widens to `BothPossible` — never a
            // fabricated `AlwaysX`.
            for mv in &merge.merged {
                if let Some(layout) = crate::registers::register_layout(&mv.name, arch) {
                    self.live.remove(layout.parent);
                } else {
                    self.live.remove(mv.name.as_str());
                }
            }
            // The merge defines the branch's flags via per-arm `Ite`s, so
            // the outstanding flag obligation is discharged.
            self.needs_flags = false;
            // Anchor the merge in execution order: every instruction kept
            // so far executes after it (the walk is backward), so its
            // `Ite`s must precede them in the lowered stream.
            merge.tail_instructions = self.kept.len();
            // Chained diamonds: if the head condition's selector reads
            // registers defined upstream, resume the walk from the head
            // block to resolve them — possibly through another diamond.
            // The head block's own body is already captured in
            // `head_instructions`, so we re-enter at its predecessors
            // (start_idx 0) and never double-count it.
            let head_addr = merge.head.block;
            let continuation = head_condition_roots(function, &merge.head, arch);
            self.merges.push(merge);
            if let Some(roots) = continuation
                && !roots.is_empty()
                && self.blocks_visited < limits.max_basic_blocks
                && !self.visited.contains(&head_addr)
                && let Some(head_block) = function.blocks.iter().find(|b| b.address == head_addr)
            {
                self.live.clear();
                for r in roots {
                    self.live.insert(r);
                }
                return WalkStep::Continue {
                    block: head_block,
                    start_idx: 0,
                };
            }
            return WalkStep::FinalizeAtEntry(format!(
                "join at {addr} resolved by bounded Φ-merge",
                addr = current_block.address
            ));
        }
        if limits.allow_join_merge {
            // Sound free-input boundary: promote the join-pending live
            // set to free symbolic inputs (scoped to joins only). This
            // only widens an `AlwaysX` to `BothPossible`, never fabricates
            // a verdict.
            self.unknowns_on_truncation = true;
            WalkStep::FinalizeAtEntry(format!(
                "join at {addr} ({n} predecessors) — merged as free inputs (allow_join_merge)",
                addr = current_block.address
            ))
        } else {
            WalkStep::FinalizeAtEntry(format!(
                "join at {addr} ({n} predecessors) — slicer cannot Φ-merge",
                addr = current_block.address
            ))
        }
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

/// The next step after a block-entry join, so the deep join handling
/// lives in [`WalkState::resolve_block_entry`] and the walk loop stays a
/// flat dispatch.
enum WalkStep<'f> {
    /// Continue the backward walk from `block`, below `start_idx`.
    Continue {
        block: &'f BasicBlock,
        start_idx: usize,
    },
    /// Finalise the slice as complete at this entry, with this reason.
    FinalizeAtEntry(String),
    /// Truncate the slice with this reason.
    Truncate(String),
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
                match state.resolve_block_entry(function, current_block, &preds, limits, arch) {
                    WalkStep::Continue {
                        block,
                        start_idx: next,
                    } => {
                        current_block = block;
                        start_idx = next;
                    }
                    WalkStep::FinalizeAtEntry(reason) => {
                        return state.finalize_at_block_entry(candidate, &reason);
                    }
                    WalkStep::Truncate(reason) => {
                        return state.into_truncated_slice(candidate, reason);
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

        if effect.is_call {
            if !limits.allow_calls {
                return BlockWalkOutcome::Truncated(format!("call at {addr}", addr = insn.address));
            }
            // A permitted call is a *value barrier*, not a no-op. The
            // callee may write any register the ABI lets it clobber —
            // the return value above all — so a definition found
            // upstream of the call does not describe the value the
            // branch actually reads. Stepping over it and continuing
            // the walk would bind live registers to pre-call
            // definitions and fabricate verdicts.
            //
            // Modelling the clobber exactly needs an ABI table and
            // interprocedural analysis; neither exists here. The sound
            // alternative is to end the walk at the call, which hands
            // the still-live registers to the SMT layer as free inputs
            // via `roots` — a widening, never a fabrication.
            //
            // A pending flag obligation cannot be discharged that way:
            // the callee clobbers the flags too, so a `cmp` upstream of
            // the call is not the definition the branch reads.
            if state.needs_flags {
                return BlockWalkOutcome::Truncated(format!(
                    "call at {addr} clobbers the flags the branch reads",
                    addr = insn.address
                ));
            }
            return BlockWalkOutcome::Done;
        }
        // Truncate only on memory the byte-granular model cannot build
        // (rip / segment / subtracted-register on x86; any memory on
        // ARM). Modellable memory — stack slots, globals, register-
        // indirect — resolves through the byte model and needs no
        // `--allow-memory`.
        if !limits.allow_memory
            && effect.has_memory_access
            && has_unmodellable_memory(&insn.operands, arch)
        {
            return BlockWalkOutcome::Truncated(format!(
                "unmodellable memory access at {addr}",
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

        if effect.kind == InstructionKind::Other {
            if SIDE_EFFECT_FREE_OTHER.contains(&insn.mnemonic.as_str()) {
                continue;
            }
            let slice_pending = state.needs_flags || !state.live.is_empty();
            if slice_pending {
                return BlockWalkOutcome::Truncated(format!(
                    "unsupported '{mnem}' at {addr} may redefine slice state",
                    mnem = insn.mnemonic,
                    addr = insn.address
                ));
            }
            continue;
        }

        if !touches_flags && touches_live.is_empty() {
            // Keep instructions that touch memory even when they don't
            // define a currently-live register/flag. A store doesn't
            // satisfy any pending live name, but its byte-granular
            // effect on memory state IS observed by any downstream load
            // that the encoder lowers through its `Ite`-chain memory
            // model. Skipping it would silently drop the write,
            // unsoundly making the later load free. Kept regardless of
            // `--allow-memory`: unmodellable memory already truncated
            // above, so anything reaching here is modellable and its
            // store/load must stay in the slice.
            if !effect.has_memory_access {
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
        for u in &effect.uses {
            state.live.insert(*u);
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

        if !state.needs_flags && state.live.is_empty() {
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
