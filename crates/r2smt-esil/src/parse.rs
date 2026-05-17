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
    /// `">>>>"`, `"<"`, `"<="`, `">"`, `">="`, `"=="`, `"!="`, `"/"`,
    /// `"%"`).
    Binary(&'static str),
    /// Unary operator that pops one operand and pushes one. The
    /// payload is `"!"` for logical NOT (currently the only modelled
    /// case).
    Unary(&'static str),
    /// `=` token: pops `value`, then `target`. The slicer assigns
    /// `target = value`.
    Assign,
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
        "==" => EsilToken::Binary("=="),
        "!=" => EsilToken::Binary("!="),
        "!" => EsilToken::Unary("!"),
        "=" => EsilToken::Assign,
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
    if is_identifier(tok) {
        EsilToken::Register(tok.to_ascii_lowercase())
    } else {
        EsilToken::Unknown(tok.to_string())
    }
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
    fn register_name_lowercases() {
        assert_eq!(
            tokenize("RAX"),
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
        for op in ["+", "-", "*", "&", "|", "^", "<<", ">>", "==", "!="] {
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
        assert!(matches!(toks[5], EsilToken::Binary("==")));
        assert!(matches!(toks[6], EsilToken::Flag(_)));
        assert!(matches!(toks[7], EsilToken::Register(_)));
        assert!(matches!(toks[8], EsilToken::Assign));
    }
}
