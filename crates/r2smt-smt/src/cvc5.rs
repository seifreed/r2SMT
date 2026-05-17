//! CVC5 backend driven by SMT-LIB2 via subprocess.
//!
//! Spawns `cvc5` for each branch query (twice — once for the taken
//! polarity, once for the not-taken). The SMT-LIB2 script comes
//! straight from [`crate::smtlib::emit_query`] so the verdict ladder
//! matches the Z3 backend's combine table.
//!
//! Requires `cvc5` to be available on `$PATH`. Distribution
//! recommendation: `brew install cvc5` on macOS,
//! `apt install cvc5` on recent Debian / Ubuntu.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::time::Duration;

use r2smt_common::smt::{SmtResult, SolveOptions};
use r2smt_ir::{Expr, IrStmt};
use r2smt_slicer::SliceStatus;
use r2smt_ssa::SsaLiftedSlice;
use tracing::debug;

use crate::smtlib::emit_query;

/// Failure modes specific to the CVC5 subprocess backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cvc5Error {
    /// `cvc5` was not found on `$PATH`. Carries the io error message
    /// from the spawn attempt.
    NotFound(String),
    /// The subprocess exited but its stdout did not contain a
    /// recognisable SAT verdict (`sat` / `unsat` / `unknown`).
    UnrecognisedVerdict(String),
    /// Spawning or communicating with the subprocess failed.
    SubprocessError(String),
}

/// Solve a branch via the CVC5 subprocess. Mirrors the contract of
/// [`crate::solve_branch`]: truncated slices that did not opt into
/// `unknowns_on_truncation` are reported as
/// [`SmtResult::Unsound`] without invoking the subprocess.
///
/// # Errors
///
/// Returns [`Cvc5Error`] when the subprocess cannot be spawned or
/// its output is malformed. Caller may fall back to another solver
/// or surface the failure in the verdict.
pub fn solve_branch_cvc5(
    slice: &SsaLiftedSlice,
    options: SolveOptions,
) -> Result<SmtResult, Cvc5Error> {
    let is_complete = matches!(slice.status, SliceStatus::Complete);
    if !is_complete && !slice.treat_truncation_as_inputs {
        return Ok(SmtResult::Unsound);
    }
    // The SMT-LIB renderer cannot model `Expr::Unknown` as a sound free
    // variable — it emits a constant placeholder, which *under*-
    // approximates the value set and can fabricate an `AlwaysX` verdict
    // (a constant has less freedom than the free var the Z3 backend
    // mints). Decline rather than answer unsoundly, mirroring the
    // truncated-slice guard above. The Z3 backend stays precise on
    // `Unknown`; CVC5 precision here is intentionally deferred.
    if slice_contains_unknown(slice) {
        return Ok(SmtResult::Unsound);
    }
    let script_taken = emit_query(slice, &options, true);
    let script_not_taken = emit_query(slice, &options, false);
    let taken = run_cvc5(&script_taken, options.timeout_ms)?;
    let not_taken = run_cvc5(&script_not_taken, options.timeout_ms)?;
    let verdict = combine(taken, not_taken);
    debug!(
        target: "r2smt::cvc5",
        at = %slice.branch.address,
        ?taken,
        ?not_taken,
        ?verdict,
        "cvc5 verdict"
    );
    Ok(verdict)
}

/// Whether any expression rendered into the SMT-LIB query carries an
/// [`Expr::Unknown`] node. The text backend has no sound encoding for
/// it (see [`solve_branch_cvc5`]), so such slices are declined.
fn slice_contains_unknown(slice: &SsaLiftedSlice) -> bool {
    if expr_has_unknown(&slice.condition) {
        return true;
    }
    slice.statements.iter().any(|stmt| match stmt {
        IrStmt::Assign { src, .. } => expr_has_unknown(src),
        IrStmt::StoreMem { address, value, .. } => {
            expr_has_unknown(address) || expr_has_unknown(value)
        }
        IrStmt::LoadMem { address, .. } => expr_has_unknown(address),
        IrStmt::Unsupported { .. } | IrStmt::Nop => false,
    })
}

/// Exhaustive recursive check for an [`Expr::Unknown`] anywhere in the
/// tree. Written without a wildcard arm so a new `Expr` variant forces
/// this to be revisited rather than silently treated as Unknown-free.
fn expr_has_unknown(expr: &Expr) -> bool {
    match expr {
        Expr::Unknown(_) => true,
        Expr::Var(_) | Expr::Const { .. } => false,
        Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::UDiv(a, b)
        | Expr::URem(a, b)
        | Expr::SDiv(a, b)
        | Expr::SRem(a, b)
        | Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Shl(a, b)
        | Expr::LShr(a, b)
        | Expr::AShr(a, b)
        | Expr::Eq(a, b)
        | Expr::Ne(a, b)
        | Expr::Ult(a, b)
        | Expr::Ule(a, b)
        | Expr::Slt(a, b)
        | Expr::Sle(a, b)
        | Expr::BoolAnd(a, b)
        | Expr::BoolOr(a, b)
        | Expr::Concat { high: a, low: b } => expr_has_unknown(a) || expr_has_unknown(b),
        Expr::BoolNot(inner)
        | Expr::Extract { src: inner, .. }
        | Expr::ZeroExtend { src: inner, .. }
        | Expr::SignExtend { src: inner, .. } => expr_has_unknown(inner),
        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => expr_has_unknown(cond) || expr_has_unknown(then_expr) || expr_has_unknown(else_expr),
    }
}

/// `(check-sat)` outcome from a single CVC5 run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SatOutcome {
    Sat,
    Unsat,
    Unknown,
}

fn combine(taken: SatOutcome, not_taken: SatOutcome) -> SmtResult {
    match (taken, not_taken) {
        (SatOutcome::Sat, SatOutcome::Unsat) => SmtResult::AlwaysTrue,
        (SatOutcome::Unsat, SatOutcome::Sat) => SmtResult::AlwaysFalse,
        (SatOutcome::Sat, SatOutcome::Sat) => SmtResult::BothPossible,
        (SatOutcome::Unsat, SatOutcome::Unsat) => SmtResult::Unsound,
        (SatOutcome::Unknown, _) | (_, SatOutcome::Unknown) => SmtResult::Timeout,
    }
}

fn run_cvc5(script: &str, timeout_ms: u32) -> Result<SatOutcome, Cvc5Error> {
    let timeout_ms_str = timeout_ms.to_string();
    let mut child = Command::new("cvc5")
        .args(["--lang", "smt2", "--tlimit-per", &timeout_ms_str])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                Cvc5Error::NotFound(err.to_string())
            } else {
                Cvc5Error::SubprocessError(err.to_string())
            }
        })?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(script.as_bytes())
            .map_err(|err| Cvc5Error::SubprocessError(err.to_string()))?;
    }
    let _ = Duration::from_millis(u64::from(timeout_ms));
    let output = child
        .wait_with_output()
        .map_err(|err| Cvc5Error::SubprocessError(err.to_string()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_verdict(&stdout)
}

fn parse_verdict(stdout: &str) -> Result<SatOutcome, Cvc5Error> {
    for line in stdout.lines().rev() {
        match line.trim() {
            "sat" => return Ok(SatOutcome::Sat),
            "unsat" => return Ok(SatOutcome::Unsat),
            "unknown" => return Ok(SatOutcome::Unknown),
            _ => {}
        }
    }
    Err(Cvc5Error::UnrecognisedVerdict(stdout.to_string()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn parse_verdict_recognises_three_outcomes() {
        assert_eq!(parse_verdict("sat\n"), Ok(SatOutcome::Sat));
        assert_eq!(parse_verdict("unsat\n"), Ok(SatOutcome::Unsat));
        assert_eq!(parse_verdict("unknown\n"), Ok(SatOutcome::Unknown));
    }

    #[test]
    fn parse_verdict_skips_leading_diagnostics() {
        // CVC5 sometimes prints version banners or comments before
        // the verdict. The parser walks from the last line backwards.
        let output = "; comment\n; another line\nsat\n";
        assert_eq!(parse_verdict(output), Ok(SatOutcome::Sat));
    }

    #[test]
    fn parse_verdict_reports_unrecognised_output() {
        let err = parse_verdict("garbage output").expect_err("should fail");
        assert!(matches!(err, Cvc5Error::UnrecognisedVerdict(_)));
    }

    #[test]
    fn slice_with_unknown_is_declined_without_spawning_cvc5() {
        use r2smt_common::{Address, Arch};
        use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind, Program};
        use r2smt_slicer::{SliceLimits, collect_branches, lift_slice, slice_branch};
        use r2smt_ssa::ssa_convert;

        // `cmp eax, <unmodeled>` makes ZF carry an `Expr::Unknown`. The
        // text backend cannot render it as a sound free var, so the
        // CVC5 path must decline (return Unsound) *before* spawning the
        // subprocess — this also makes the test deterministic on hosts
        // without `cvc5` installed.
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
                        Instruction {
                            address: Address(0x40_1000),
                            size: 3,
                            bytes: vec![],
                            mnemonic: "cmp".into(),
                            operands: vec![
                                Operand {
                                    raw: "eax".into(),
                                    kind: OperandKind::Register,
                                },
                                Operand {
                                    raw: "junk".into(),
                                    kind: OperandKind::Unknown,
                                },
                            ],
                            esil: None,
                            pcode: None,
                            is_thumb: false,
                        },
                        Instruction {
                            address: Address(0x40_1003),
                            size: 6,
                            bytes: vec![],
                            mnemonic: "jne".into(),
                            operands: vec![Operand {
                                raw: "0x401080".into(),
                                kind: OperandKind::Immediate,
                            }],
                            esil: None,
                            pcode: None,
                            is_thumb: false,
                        },
                    ],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        };
        let cand = collect_branches(&program)
            .into_iter()
            .next()
            .expect("a branch");
        let slice = slice_branch(
            &cand,
            &program.functions[0],
            &SliceLimits::default(),
            program.arch,
        );
        let ssa = ssa_convert(&lift_slice(&slice, program.arch));
        assert!(
            slice_contains_unknown(&ssa),
            "the lifted slice should carry an Expr::Unknown"
        );
        assert_eq!(
            solve_branch_cvc5(&ssa, SolveOptions::default()),
            Ok(SmtResult::Unsound)
        );
    }
}
