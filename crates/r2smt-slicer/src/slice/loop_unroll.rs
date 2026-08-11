//! Bounded concrete counted-loop unrolling.
//!
//! Recognises a single-block self-loop with one integer counter that is
//! initialised to a constant, stepped by a constant, and exited by a
//! compare against a constant — the shape a compiler emits for
//! `for (i = A; i <cc> N; i += S)`. When every ingredient is concrete,
//! the counter's value at loop exit is a single constant, computed by
//! forward-simulating the loop up to [`MAX_UNROLL`] iterations. The
//! caller injects that constant as a synthetic `mov counter, value`, so
//! a downstream branch on the counter resolves precisely.
//!
//! This only ever *adds* precision. Any deviation from the concrete
//! counted shape — a non-constant init/step/bound, a second counter, a
//! memory/call side effect, an unsupported exit condition, or a loop
//! that does not converge within the bound — returns `None`, and the
//! caller falls back to the sound free-input boundary (the loop counter
//! becomes a free symbolic input, widening `AlwaysX` to `BothPossible`,
//! never a fabricated verdict).

use std::collections::BTreeSet;

use r2smt_common::Arch;
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind};
use tracing::debug;

use crate::collector::collect_function_branches;
use crate::condition::{BranchCondition, BranchKind};
use crate::effect::{InstructionKind, analyze};
use crate::registers::register_layout;

use super::predecessors_of;

/// Hard cap on simulated iterations. A loop that has not exited by here
/// is treated as non-convergent and declined (sound free-input
/// fallback). Bounds host work to a small constant per attempt.
const MAX_UNROLL: u64 = 4096;

/// Attempt to resolve a concrete counted self-loop into the counter's
/// exit constant. Returns a synthetic `mov counter, value` instruction
/// on success, or `None` to fall back to the sound free-input boundary.
pub(super) fn try_unroll_counted_loop(
    function: &Function,
    loop_block: &BasicBlock,
    live: &BTreeSet<&'static str>,
    needs_flags: bool,
    arch: Arch,
) -> Option<Instruction> {
    // The branch must depend on exactly the one loop counter, with no
    // outstanding flag obligation the loop body would have to satisfy.
    if needs_flags || live.len() != 1 {
        return None;
    }
    let counter_parent = *live.iter().next()?;

    // Single-block self-loop: `loop_block` is its own successor and has
    // exactly two successors (itself + the exit).
    let self_loop = loop_block.successors.contains(&loop_block.address);
    if !self_loop || loop_block.successors.len() != 2 {
        return None;
    }

    let shape = analyse_body(loop_block, counter_parent, arch)?;
    let init = concrete_init(function, loop_block, counter_parent, shape.width, arch)?;
    let exit_value = simulate(init, &shape)?;

    debug!(
        target: "r2smt::slicer",
        loop_block = %loop_block.address,
        counter = counter_parent,
        exit_value,
        "counted loop unrolled to constant"
    );

    Some(synthetic_mov(
        loop_block,
        &shape.counter_operand,
        exit_value,
        shape.width,
    ))
}

/// The concrete ingredients of a counted loop body.
struct LoopShape {
    /// Register spelling of the counter as it appears in the body
    /// (e.g. `"eax"`), used for the synthetic assignment.
    counter_operand: String,
    /// Operand width in bits.
    width: u16,
    /// Signed per-iteration step, in the counter's bit pattern.
    step: i128,
    /// Loop-continue condition (the loop-back branch's condition,
    /// comparing the counter to `bound`).
    condition: BranchCondition,
    /// Constant compared against by the exit test.
    bound: u128,
}

/// Verify the loop body is a concrete counted step + compare + loop-back
/// and nothing else, returning its ingredients.
fn analyse_body(loop_block: &BasicBlock, counter_parent: &str, arch: Arch) -> Option<LoopShape> {
    let mut counter_operand: Option<String> = None;
    let mut step: Option<i128> = None;
    let mut bound: Option<u128> = None;
    // Positions and widths of the step and the compare, checked below:
    // `simulate` models the do-while shape *step, then test the stepped
    // value against `bound`*, so the compare must come after the step (or
    // the loop-back branch reads the step's flags / tests the pre-step
    // value) and must test the counter at the width it is stepped (a
    // `cmp al, N` against an `eax` step compares only the low byte). Both
    // pass `canonical` because it folds sub-registers to one parent.
    let mut step_index: Option<usize> = None;
    let mut step_width: Option<u16> = None;
    let mut cmp_index: Option<usize> = None;
    let mut cmp_width: Option<u16> = None;

    for (index, insn) in loop_block.instructions.iter().enumerate() {
        let mnem = insn.mnemonic.trim().to_ascii_lowercase();
        let effect = analyze(insn, arch);
        if effect.is_call || effect.has_memory_access {
            return None;
        }
        match mnem.as_str() {
            "add" | "sub" => {
                let (reg, imm) = reg_imm_operands(insn)?;
                if canonical(&reg, arch)? != counter_parent {
                    return None;
                }
                let magnitude = i128::from(parse_u64(&imm)?);
                let delta = if mnem == "sub" { -magnitude } else { magnitude };
                if step.replace(delta).is_some() {
                    return None; // counter stepped twice — not a simple loop
                }
                step_index = Some(index);
                step_width = Some(register_layout(&reg, arch)?.width());
                counter_operand.get_or_insert(reg);
            }
            "inc" | "dec" => {
                let reg = single_reg_operand(insn)?;
                if canonical(&reg, arch)? != counter_parent {
                    return None;
                }
                let delta = if mnem == "dec" { -1 } else { 1 };
                if step.replace(delta).is_some() {
                    return None;
                }
                step_index = Some(index);
                step_width = Some(register_layout(&reg, arch)?.width());
                counter_operand.get_or_insert(reg);
            }
            "cmp" => {
                let (reg, imm) = reg_imm_operands(insn)?;
                if canonical(&reg, arch)? != counter_parent {
                    return None;
                }
                if bound.replace(u128::from(parse_u64(&imm)?)).is_some() {
                    return None;
                }
                cmp_index = Some(index);
                cmp_width = Some(register_layout(&reg, arch)?.width());
                counter_operand.get_or_insert(reg);
            }
            _ => {
                // Only the loop-back conditional branch and pure no-ops
                // are tolerated; anything that defines a register or has
                // an unmodelled effect declines the unroll.
                if effect.kind == InstructionKind::Jmp {
                    continue;
                }
                let is_branch = mnem.starts_with('j');
                if !is_branch && !effect.defs.is_empty() {
                    return None;
                }
                if !is_branch && effect.kind == InstructionKind::Other && mnem != "nop" {
                    return None;
                }
            }
        }
    }

    let condition = loop_back_condition(loop_block, arch)?;
    let counter_operand = counter_operand?;
    let width = register_layout(&counter_operand, arch)?.width();
    // The compare must test the counter at the step's width, and must be
    // the flag-defining instruction the loop-back branch actually reads —
    // i.e. it must come after the step. A reordered (`cmp; step; jcc`) or
    // width-mismatched (`cmp al` while stepping `eax`) shape would make
    // `simulate`'s step-then-test model fabricate a wrong exit constant,
    // so it declines to the sound free-input boundary instead.
    if step_width? != cmp_width? || step_index? >= cmp_index? {
        return None;
    }
    Some(LoopShape {
        counter_operand,
        width,
        step: step?,
        condition,
        bound: bound?,
    })
}

/// The condition under which the loop body branches back to itself.
fn loop_back_condition(loop_block: &BasicBlock, arch: Arch) -> Option<BranchCondition> {
    collect_function_branches_in_block(loop_block, arch)
        .into_iter()
        .find(|c| c.taken_target == Some(loop_block.address) && matches!(c.kind, BranchKind::Jcc))
        .map(|c| c.condition)
}

fn collect_function_branches_in_block(
    loop_block: &BasicBlock,
    arch: Arch,
) -> Vec<crate::collector::BranchCandidate> {
    // A single self-looping block: build a throwaway one-block function
    // so the collector resolves its terminator's targets.
    let func = Function {
        address: loop_block.address,
        name: None,
        blocks: vec![loop_block.clone()],
        is_thumb: false,
    };
    collect_function_branches(&func, arch)
}

/// The counter's constant value on entry to the loop, taken from the
/// loop's single non-self predecessor. Requires a `mov counter, imm`
/// there and no later non-constant write.
fn concrete_init(
    function: &Function,
    loop_block: &BasicBlock,
    counter_parent: &str,
    width: u16,
    arch: Arch,
) -> Option<u128> {
    let preds = predecessors_of(function, loop_block.address);
    let mut non_self = preds.into_iter().filter(|p| *p != loop_block.address);
    let entry = non_self.next()?;
    // The counter's entry value is a single constant only when the loop is
    // reached from one place. A header with two entry edges can be reached
    // with different inits (or via a path that never enters the body), so
    // one arbitrary entry's `mov` does not describe every execution —
    // decline to the sound free-input boundary rather than fabricate.
    if non_self.next().is_some() {
        return None;
    }
    let entry_block = function.blocks.iter().find(|b| b.address == entry)?;

    let mut init: Option<u128> = None;
    for insn in &entry_block.instructions {
        let effect = analyze(insn, arch);
        let writes_counter = effect.defs.contains(&counter_parent);
        if !writes_counter {
            continue;
        }
        // Only a `mov counter, imm` gives a concrete init; any other
        // writer (add, load, call result, …) makes the entry value
        // non-constant, so decline.
        if !insn.mnemonic.trim().eq_ignore_ascii_case("mov") {
            return None;
        }
        let (reg, imm) = reg_imm_operands(insn)?;
        if canonical(&reg, arch)? != counter_parent {
            return None;
        }
        // The init must set the counter's full width. A sub-register
        // `mov` — `mov al, 5` when the loop steps `eax` — leaves the
        // upper bits at their prior (unknown) contents, so treating its
        // immediate as the whole counter value would simulate from a
        // wrong start and inject a definite wrong exit constant. Decline
        // to the sound free-input boundary instead.
        let layout = register_layout(&reg, arch)?;
        if layout.lo != 0 || layout.hi.saturating_add(1) < width {
            return None;
        }
        init = Some(u128::from(parse_u64(&imm)?));
    }
    init
}

/// Forward-simulate the loop; return the counter's value at exit, or
/// `None` if the condition is unsupported or it does not converge.
// The `step as u128` reinterprets a signed delta as its two's-complement
// bit pattern for `wrapping_add` at the counter width — intentional.
#[allow(clippy::cast_sign_loss)]
fn simulate(init: u128, shape: &LoopShape) -> Option<u128> {
    let mask = width_mask(shape.width);
    let mut v = init & mask;
    let mut iters: u64 = 0;
    loop {
        v = v.wrapping_add(shape.step as u128) & mask;
        iters += 1;
        if !continue_loop(shape.condition, v, shape.bound & mask, shape.width)? {
            return Some(v);
        }
        if iters >= MAX_UNROLL {
            return None;
        }
    }
}

/// Evaluate the loop-continue condition on concrete `counter`/`bound`.
/// `None` for conditions this unroller does not model.
fn continue_loop(cond: BranchCondition, counter: u128, bound: u128, width: u16) -> Option<bool> {
    let sc = as_signed(counter, width);
    let sb = as_signed(bound, width);
    Some(match cond {
        BranchCondition::Equal => counter == bound,
        BranchCondition::NotEqual => counter != bound,
        BranchCondition::Less => sc < sb,
        BranchCondition::LessOrEqual => sc <= sb,
        BranchCondition::Greater => sc > sb,
        BranchCondition::GreaterOrEqual => sc >= sb,
        BranchCondition::Below => counter < bound,
        BranchCondition::BelowOrEqual => counter <= bound,
        BranchCondition::Above => counter > bound,
        BranchCondition::AboveOrEqual => counter >= bound,
        _ => return None,
    })
}

// Two's-complement reinterpretation of a `width`-bit value as `i128`;
// the `u128 as i128` casts are the intended sign reinterpret.
#[allow(clippy::cast_possible_wrap)]
fn as_signed(value: u128, width: u16) -> i128 {
    if width == 0 || width >= 128 {
        return value as i128;
    }
    let sign_bit = 1u128 << (width - 1);
    if value & sign_bit != 0 {
        (value as i128) - (1i128 << width)
    } else {
        value as i128
    }
}

fn width_mask(width: u16) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn synthetic_mov(
    loop_block: &BasicBlock,
    counter_operand: &str,
    value: u128,
    width: u16,
) -> Instruction {
    Instruction {
        address: loop_block.address,
        size: 0,
        bytes: Vec::new(),
        mnemonic: "mov".into(),
        operands: vec![
            Operand {
                raw: counter_operand.to_string(),
                kind: OperandKind::Register,
            },
            Operand {
                raw: format!("0x{:x}", value & width_mask(width)),
                kind: OperandKind::Immediate,
            },
        ],
        esil: None,
        pcode: None,
        is_thumb: false,
    }
}

fn reg_imm_operands(insn: &Instruction) -> Option<(String, String)> {
    let dst = insn.operands.first()?;
    let src = insn.operands.get(1)?;
    if dst.kind != OperandKind::Register || src.kind != OperandKind::Immediate {
        return None;
    }
    Some((dst.raw.clone(), src.raw.clone()))
}

fn single_reg_operand(insn: &Instruction) -> Option<String> {
    let dst = insn.operands.first()?;
    (dst.kind == OperandKind::Register).then(|| dst.raw.clone())
}

fn canonical(raw: &str, arch: Arch) -> Option<&'static str> {
    register_layout(raw, arch).map(|l| l.parent)
}

fn parse_u64(raw: &str) -> Option<u64> {
    let t = raw.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        t.parse().ok()
    }
}
