//! ESIL token parser.
//!
//! ESIL strings are comma-separated postfix expressions. Tokens are
//! either operators (`+`, `-`, `==`, `=`, `[4]`, `=[8]`, `?{`, `}`,
//! `GOTO`, …) or operands (decimal / hex integers, register names,
//! flag tokens like `$z`). This module turns the raw string into a
//! `Vec<EsilToken>`; the stack machine in [`crate::machine`] consumes
//! that vector.
//!
//! The lexer is deliberately conservative: any token it does not
//! recognise becomes [`EsilToken::Unknown`] so the stack-machine
//! layer can bail out cleanly without misclassifying control-flow
//! markers as arithmetic.

/// One ESIL token after lexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EsilToken {
    /// Unsigned integer literal (`0x10`, `42`).
    Integer(u64),
    /// Register / variable name (`rax`, `eax`, `rsp`).
    Register(String),
    /// Pseudo-flag token (`$z`, `$c`, `$s`, `$o`, `$p`, `$0`, …). The
    /// payload is the suffix after the `$` so the evaluator can
    /// dispatch on it.
    Flag(String),
    /// Arithmetic / logical operator that pops two operands and
    /// pushes one. The payload is the canonical operator string
    /// (`"+"`, `"-"`, `"*"`, `"&"`, `"|"`, `"^"`, `"<<"`, `">>"`,
    /// `">>>>"`, `"<"`, `"<="`, `">"`, `">="`, `"=="`, `"/"`,
    /// `"%"`).
    Binary(&'static str),
    /// Unary operator that pops one operand and pushes one. The
    /// payload is `"!"` for logical NOT (currently the only modelled
    /// case).
    Unary(&'static str),
    /// `=` token: pops `value`, then `target`. The slicer assigns
    /// `target = value`, and the write seeds the flag context.
    Assign,
    /// `:=` — the same assignment **without** seeding the flag context.
    /// A distinct variant rather than a payload on [`EsilToken::Assign`]
    /// so the machine's dispatch stays a flat match.
    AssignNoFlags,
    /// `==` — pops both operands, seeds the flag context, pushes
    /// nothing.
    Compare,
    /// `!=` — pops one register and writes its logical NOT back.
    NegAssign,
    /// Compound assignment such as `+=`, `-=`, `&=`, etc. The
    /// payload is the operator part (`"+"`, `"-"`, …).
    CompoundAssign(&'static str),
    /// Memory load `[N]` — pops an address, pushes the loaded value.
    /// `N` is the access size in bytes.
    Load(u8),
    /// Memory store `=[N]` — pops `value`, then `address`. Stores
    /// `*address = value` at `N` bytes.
    Store(u8),
    /// `?{` — pop a condition; the block until matching `}` only
    /// executes when the condition is non-zero. The evaluator
    /// currently bails on this token (predicated bodies are out of
    /// the MVP scope).
    BlockOpen,
    /// Closing `}` of a `?{ … }` block.
    BlockClose,
    /// Anything the lexer cannot place. The evaluator treats this
    /// as a hard error.
    Unknown(String),
}

/// Lex an ESIL string into a token sequence. Empty / whitespace-only
/// tokens are skipped.
#[must_use]
pub fn tokenize(esil: &str) -> Vec<EsilToken> {
    esil.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(classify)
        .collect()
}

fn classify(tok: &str) -> EsilToken {
    if let Some(rest) = tok.strip_prefix('$') {
        return EsilToken::Flag(rest.to_string());
    }
    if let Some(rest) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X"))
        && let Ok(value) = u64::from_str_radix(rest, 16)
    {
        return EsilToken::Integer(value);
    }
    if tok.chars().all(|c| c.is_ascii_digit())
        && let Ok(value) = tok.parse::<u64>()
    {
        return EsilToken::Integer(value);
    }
    match tok {
        "+" => EsilToken::Binary("+"),
        "-" => EsilToken::Binary("-"),
        "*" => EsilToken::Binary("*"),
        "/" => EsilToken::Binary("/"),
        "%" => EsilToken::Binary("%"),
        "&" => EsilToken::Binary("&"),
        "|" => EsilToken::Binary("|"),
        "^" => EsilToken::Binary("^"),
        "<<" => EsilToken::Binary("<<"),
        ">>" => EsilToken::Binary(">>"),
        "<" => EsilToken::Binary("<"),
        "<=" => EsilToken::Binary("<="),
        ">" => EsilToken::Binary(">"),
        ">=" => EsilToken::Binary(">="),
        // Not a binary operator: `ae 5,3,==` leaves the stack **empty**,
        // so it pops two and pushes nothing, seeding the flag context
        // instead. Modelling it as one left a boolean the next token
        // would consume as an operand.
        "==" => EsilToken::Compare,
        // radare2's own help calls this "negate all bits" and that is
        // wrong: `ar rax=0x0f; ae rax,!=` leaves **0**, not
        // `0xfffffffffffffff0`. It is a logical NOT written back to the
        // popped register, `(1 -- 0)` tagged `math+regw`. The operator
        // was withdrawn from the model in `f45b3cdd` rather than fixed,
        // because two tests pinned the complement reading; this is the
        // measured one.
        "!=" => EsilToken::NegAssign,
        "!" => EsilToken::Unary("!"),
        "=" => EsilToken::Assign,
        // `:=` is **deliberately** still unlexed, so it stays an
        // `Unknown` and the whole lift fails into the per-mnemonic
        // handler. The machine models it already
        // ([`EsilToken::AssignNoFlags`], `apply_assign(_, false)`) — what
        // is missing is the *gate*, not the semantics.
        //
        // radare2 writes every flag of every ISA with `:=`, so lexing it
        // moves the entire flag-setting population off the per-mnemonic
        // handlers, which carry ~750 solver-backed contracts, and onto
        // this machine, which has ~50 unit tests. Two measured facts make
        // that unsafe on ARM specifically: r2's `a64 cmp` emits
        // `64,$b,!,cf,:=`, i.e. ARM's architectural C, where this
        // pipeline stores the *inverse* borrow polarity so one
        // `lift_branch_condition` can serve both ISAs — every unsigned
        // ARM branch would resolve to the other arm. And r2's own a64
        // `subs` seeds through `x0,=`, so its carry is a function of
        // whatever the destination held *before* the instruction; that
        // is a bug in radare2 that a faithful implementation imports.
        //
        // The unblock therefore needs the override seam in
        // `r2smt_slicer::lift::lift_instruction` first.
        "+=" => EsilToken::CompoundAssign("+"),
        "-=" => EsilToken::CompoundAssign("-"),
        "*=" => EsilToken::CompoundAssign("*"),
        "&=" => EsilToken::CompoundAssign("&"),
        "|=" => EsilToken::CompoundAssign("|"),
        "^=" => EsilToken::CompoundAssign("^"),
        "<<=" => EsilToken::CompoundAssign("<<"),
        ">>=" => EsilToken::CompoundAssign(">>"),
        "?{" => EsilToken::BlockOpen,
        "}" => EsilToken::BlockClose,
        _ => parse_memory_or_register(tok),
    }
}

fn parse_memory_or_register(tok: &str) -> EsilToken {
    if let Some(size_str) = tok.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
        && let Ok(n) = size_str.parse::<u8>()
    {
        return EsilToken::Load(n);
    }
    if let Some(size_str) = tok.strip_prefix("=[").and_then(|s| s.strip_suffix(']'))
        && let Ok(n) = size_str.parse::<u8>()
    {
        return EsilToken::Store(n);
    }
    if is_identifier(tok) && !is_esil_keyword(tok) {
        EsilToken::Register(tok.to_ascii_lowercase())
    } else {
        EsilToken::Unknown(tok.to_string())
    }
}

/// Whether an otherwise register-shaped token is really one of ESIL's
/// alphabetic operators.
///
/// Without this every one of them — `DUP`, `SWAP`, `POP`, `ASR`, `ROR`,
/// `GOTO`, `BREAK`, `TODO`, `NAN`, … — parses as a *register*, and the
/// damage is silent and in the unsound direction: `lift_esil` returns
/// `Ok` holding a free variable (or, for the zero-effect ones like
/// `TODO`, no statements at all), so [`crate::lift_esil`]'s caller takes
/// the result and never falls through to the per-mnemonic handler. An
/// ARM `r2,r1,ROR,r0,=` became `r0 = <free>`: a fabricated value, not a
/// lost one. Measured in real output from one `AArch64` sample: `DUP`
/// 1 121, `ROR` 641, `ASR` 130, `SWAP` 57.
///
/// The test is *case*, not a name list, and that is deliberate. Every
/// alphabetic operator radare2 lists in `ae???` is upper-case, while
/// register names arrive lower-case — an assumption this module already
/// relies on, since it lower-cases them on the way in. So the rule
/// covers operators added by later radare2 versions, which a fixed list
/// would silently start misparsing again. Measured across `AArch64`,
/// x86-64 and `AArch32` samples, the only upper-case ESIL tokens in real
/// output are operators and hex literals, and the literals never reach
/// here — [`classify`] claims them first.
///
/// A token wrongly rejected costs precision: the lift fails and the
/// per-mnemonic handler runs instead. A token wrongly accepted costs
/// soundness. The rule errs toward the first.
fn is_esil_keyword(tok: &str) -> bool {
    tok.chars().any(|c| c.is_ascii_uppercase())
}

fn is_identifier(tok: &str) -> bool {
    let mut chars = tok.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_literals_in_hex_and_decimal() {
        assert_eq!(tokenize("0x10"), vec![EsilToken::Integer(0x10)]);
        assert_eq!(tokenize("42"), vec![EsilToken::Integer(42)]);
    }

    #[test]
    fn an_upper_case_token_is_an_operator_and_not_a_register() {
        // This used to assert `RAX` lowercases into `Register("rax")`,
        // which read as harmless normalisation and was the whole bug:
        // radare2 spells register names lower-case and its alphabetic
        // *operators* upper-case, so accepting an upper-case identifier
        // as a register is what turned `DUP` / `ROR` / `ASR` into free
        // variables. Nothing measured emits an upper-case register.
        assert_eq!(tokenize("DUP"), vec![EsilToken::Unknown("DUP".to_string())]);
        assert_eq!(
            tokenize("rax"),
            vec![EsilToken::Register("rax".to_string())]
        );
    }

    #[test]
    fn flag_token_keeps_suffix() {
        assert_eq!(tokenize("$z"), vec![EsilToken::Flag("z".to_string())]);
    }

    #[test]
    fn assignment_and_compound_assignment() {
        assert_eq!(
            tokenize("rax,="),
            vec![EsilToken::Register("rax".to_string()), EsilToken::Assign,]
        );
        assert_eq!(
            tokenize("1,rax,+=").last(),
            Some(&EsilToken::CompoundAssign("+"))
        );
    }

    #[test]
    fn binary_operators() {
        for op in ["+", "-", "*", "&", "|", "^", "<<", ">>"] {
            let toks = tokenize(op);
            assert_eq!(toks, vec![EsilToken::Binary(op)], "for {op}");
        }
    }

    #[test]
    fn memory_load_and_store_sizes() {
        assert_eq!(tokenize("[4]"), vec![EsilToken::Load(4)]);
        assert_eq!(tokenize("=[8]"), vec![EsilToken::Store(8)]);
    }

    #[test]
    fn unknown_token_preserves_raw_text() {
        assert_eq!(
            tokenize("XYZZY!"),
            vec![EsilToken::Unknown("XYZZY!".to_string())]
        );
    }

    #[test]
    fn complete_program_parses_into_expected_sequence() {
        // Real ESIL string equivalent to `mov eax, 1; cmp eax, 1; je`:
        //   `1,eax,=,1,eax,==,$z,zf,=`
        let toks = tokenize("1,eax,=,1,eax,==,$z,zf,=");
        assert_eq!(toks.len(), 9);
        assert!(matches!(toks[0], EsilToken::Integer(1)));
        assert!(matches!(toks[1], EsilToken::Register(_)));
        assert!(matches!(toks[2], EsilToken::Assign));
        assert!(matches!(toks[5], EsilToken::Compare));
        assert!(matches!(toks[6], EsilToken::Flag(_)));
        assert!(matches!(toks[7], EsilToken::Register(_)));
        assert!(matches!(toks[8], EsilToken::Assign));
    }
}
