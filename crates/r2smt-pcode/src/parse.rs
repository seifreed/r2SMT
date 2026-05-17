//! Pure parser for r2ghidra `pdgsd` P-code text.
//!
//! Independent of any live r2 process so it can be exercised with
//! fixed fixtures and without an r2ghidra install. The grammar
//! (observed from `pdgsd N`, stable SLEIGH textual P-code):
//!
//! ```text
//! 0x<hex>: <asm text>                 # instruction header (not indented)
//!     <out> = <OPCODE> <in0>[, <in1>] # data-flow op (indented)
//!     <out> = LOAD ram[<addr>]        # memory load
//!     STORE ram[<addr>] = <value>     # memory store
//!     BRANCH qword_ptr(0x<hex>)       # control flow
//!     CBRANCH qword_ptr(0x<hex>), <cond>
//! ```
//!
//! Varnode tokens: `(unique,0x<hex>,<size>)`, a bare register name
//! (`sp`, `w0`, `x8`, `ZR`, `tmpCY`, …), a hex constant (`0x10`),
//! `ram[<vn>]`, or `qword_ptr(0x<hex>)`.

/// Structural parse failure. Open-domain (free-form reason) per the
/// project error policy — parser reasons are inherently open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

/// A P-code varnode operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Varnode {
    /// A named register / flag varnode (`sp`, `w8`, `ZR`, `tmpCY`, …).
    Register(String),
    /// A SLEIGH `unique` temporary: `(unique,offset,size)`.
    Unique {
        /// Byte offset within the unique space.
        offset: u64,
        /// Width in bytes.
        size: u8,
    },
    /// An integer constant. Width is often implicit in the textual
    /// form, so it is inferred at lift time from the operation.
    Const {
        /// The literal value.
        value: u64,
        /// Width in bytes when the form carried one.
        size: Option<u8>,
    },
    /// A memory reference `ram[<addr>]` — the inner varnode is the
    /// address expression.
    Ram(Box<Varnode>),
    /// A code address branch target (`qword_ptr(0x..)`).
    CodeAddr(u64),
}

/// One P-code micro-operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcodeOp {
    /// Output varnode, or `None` for `STORE` / control-flow ops.
    pub out: Option<Varnode>,
    /// SLEIGH opcode mnemonic, verbatim (`INT_ADD`, `COPY`, …).
    pub opcode: String,
    /// Input varnodes in source order.
    pub inputs: Vec<Varnode>,
}

/// All P-code ops emitted for one machine instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcodeInsn {
    /// Machine address of the instruction.
    pub address: u64,
    /// P-code ops, in emission order.
    pub ops: Vec<PcodeOp>,
}

/// Parse a `pdgsd` dump into structured instructions.
///
/// # Errors
///
/// Returns [`ParseError`] when an indented op line cannot be
/// structurally decoded — dropping a data-flow op silently would be
/// unsound, so the caller must fall back to another IR source.
pub fn parse_pcode(text: &str) -> Result<Vec<PcodeInsn>, ParseError> {
    let mut out: Vec<PcodeInsn> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with(char::is_whitespace) {
            let op = parse_op_line(line.trim())?;
            let Some(current) = out.last_mut() else {
                return Err(ParseError(format!(
                    "op line before any instruction header: {line:?}"
                )));
            };
            current.ops.push(op);
        } else {
            // Instruction header: `0x<hex>: <asm>`.
            let Some(addr) = parse_header_address(line) else {
                // Non-header, non-indented line (e.g. a stray r2 log
                // that slipped through). Skip — only op lines are
                // load-bearing.
                continue;
            };
            out.push(PcodeInsn {
                address: addr,
                ops: Vec::new(),
            });
        }
    }
    Ok(out)
}

fn parse_header_address(line: &str) -> Option<u64> {
    let (addr_tok, _rest) = line.split_once(':')?;
    let hex = addr_tok.trim().strip_prefix("0x")?;
    u64::from_str_radix(hex, 16).ok()
}

fn parse_op_line(line: &str) -> Result<PcodeOp, ParseError> {
    // `STORE ram[<addr>] = <value>` — no varnode on the left of `=`.
    if let Some(rest) = line.strip_prefix("STORE ") {
        let (addr_part, value_part) = rest
            .split_once(" = ")
            .ok_or_else(|| ParseError(format!("malformed STORE: {line:?}")))?;
        return Ok(PcodeOp {
            out: None,
            opcode: "STORE".to_string(),
            inputs: vec![
                parse_varnode(addr_part.trim())?,
                parse_varnode(value_part.trim())?,
            ],
        });
    }

    // `OUT = OPCODE [IN, IN, ...]`  or  `OUT = LOAD ram[ADDR]`.
    if let Some((lhs, rhs)) = split_assignment(line) {
        let out = parse_varnode(lhs.trim())?;
        let mut toks = rhs.trim().splitn(2, char::is_whitespace);
        let opcode = toks
            .next()
            .ok_or_else(|| ParseError(format!("missing opcode: {line:?}")))?
            .to_string();
        let inputs = match toks.next() {
            Some(args) => parse_input_list(args.trim())?,
            None => Vec::new(),
        };
        return Ok(PcodeOp {
            out: Some(out),
            opcode,
            inputs,
        });
    }

    // Control-flow ops with no output: `BRANCH ...`, `CBRANCH ...`,
    // `RETURN ...`, `CALL ...`, `BRANCHIND ...`, `CALLIND ...`.
    let mut toks = line.splitn(2, char::is_whitespace);
    let opcode = toks
        .next()
        .ok_or_else(|| ParseError(format!("empty op line: {line:?}")))?
        .to_string();
    if !opcode.chars().all(|c| c.is_ascii_uppercase() || c == '_') {
        return Err(ParseError(format!("unrecognised op line: {line:?}")));
    }
    let inputs = match toks.next() {
        Some(args) => parse_input_list(args.trim())?,
        None => Vec::new(),
    };
    Ok(PcodeOp {
        out: None,
        opcode,
        inputs,
    })
}

/// Split `A = B` on the *first* ` = ` that is not inside parentheses
/// or brackets (so `(unique,0x6000,8) = INT_ADD sp, 0x8` splits at the
/// right place and `ram[..]` indices are not mis-split).
fn split_assignment(line: &str) -> Option<(&str, &str)> {
    let bytes = line.as_bytes();
    let mut depth: i32 = 0;
    let mut i = 0;
    while i + 2 < bytes.len() {
        match bytes[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b' ' if depth == 0 && bytes[i + 1] == b'=' && bytes[i + 2] == b' ' => {
                return Some((&line[..i], &line[i + 3..]));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Split a top-level comma list, respecting `(...)` / `[...]` nesting.
fn parse_input_list(s: &str) -> Result<Vec<Varnode>, ParseError> {
    let mut parts: Vec<&str> = Vec::new();
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(parse_varnode)
        .collect()
}

fn parse_varnode(tok: &str) -> Result<Varnode, ParseError> {
    let tok = tok.trim();
    if let Some(inner) = tok.strip_prefix("ram[").and_then(|s| s.strip_suffix(']')) {
        return Ok(Varnode::Ram(Box::new(parse_varnode(inner)?)));
    }
    if let Some(inner) = tok
        .strip_prefix("qword_ptr(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let hex = inner.trim().strip_prefix("0x").unwrap_or(inner.trim());
        let v = u64::from_str_radix(hex, 16)
            .map_err(|_| ParseError(format!("bad code addr: {tok:?}")))?;
        return Ok(Varnode::CodeAddr(v));
    }
    if let Some(inner) = tok.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        // (space,0xoffset,size)
        let mut f = inner.split(',');
        let space = f.next().unwrap_or("").trim();
        let off_tok = f.next().unwrap_or("").trim();
        let size_tok = f.next().unwrap_or("").trim();
        let offset =
            parse_u64(off_tok).ok_or_else(|| ParseError(format!("bad varnode offset: {tok:?}")))?;
        let size: u8 = size_tok
            .parse()
            .map_err(|_| ParseError(format!("bad varnode size: {tok:?}")))?;
        return match space {
            "unique" => Ok(Varnode::Unique { offset, size }),
            "const" => Ok(Varnode::Const {
                value: offset,
                size: Some(size),
            }),
            _ => Ok(Varnode::Register(format!("{space}_{offset:#x}"))),
        };
    }
    if let Some(v) = parse_u64(tok) {
        return Ok(Varnode::Const {
            value: v,
            size: None,
        });
    }
    if !tok.is_empty()
        && tok
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return Ok(Varnode::Register(tok.to_string()));
    }
    Err(ParseError(format!("unrecognised varnode: {tok:?}")))
}

fn parse_u64(tok: &str) -> Option<u64> {
    if let Some(hex) = tok.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        tok.parse::<u64>().ok()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    const FIXTURE: &str = "\
0x00000328: sub sp, sp, #0x10
    sp = INT_SUB sp, 0x10
0x00000338: mul w8, w8, w9
    (unique,0x2ae80,4) = INT_MULT w8, w9
    x8 = INT_ZEXT (unique,0x2ae80,4)
0x00000344: and w8, w8, #0x1
    (unique,0x12180,4) = INT_AND w8, 0x1
    x8 = INT_ZEXT (unique,0x12180,4)
0x0000034c: b.ne 0x360
    (unique,0xa00,1) = BOOL_NEGATE ZR
    CBRANCH qword_ptr(0x00000360), (unique,0xa00,1)
0x00000360: ldr w8, [sp, #0x8]
    (unique,0x6000,8) = INT_ADD sp, 0x8
    (unique,0x24700,4) = LOAD ram[(unique,0x6000,8)]
    x8 = INT_ZEXT (unique,0x24700,4)
0x00000370: str w8, [sp, #0xc]
    STORE ram[(unique,0x6000,8)] = w8";

    #[test]
    fn test_parse_groups_ops_under_instruction_headers() {
        let p = parse_pcode(FIXTURE).unwrap();
        assert_eq!(p.len(), 6);
        assert_eq!(p[0].address, 0x328);
        assert_eq!(p[0].ops.len(), 1);
        assert_eq!(p[2].address, 0x344);
        assert_eq!(p[2].ops.len(), 2);
    }

    #[test]
    fn test_parse_assignment_op_varnodes() {
        let p = parse_pcode(FIXTURE).unwrap();
        let op = &p[1].ops[0]; // (unique,0x2ae80,4) = INT_MULT w8, w9
        assert_eq!(
            op.out,
            Some(Varnode::Unique {
                offset: 0x2ae80,
                size: 4
            })
        );
        assert_eq!(op.opcode, "INT_MULT");
        assert_eq!(
            op.inputs,
            vec![
                Varnode::Register("w8".into()),
                Varnode::Register("w9".into())
            ]
        );
    }

    #[test]
    fn test_parse_const_input() {
        let p = parse_pcode(FIXTURE).unwrap();
        let op = &p[2].ops[0]; // (unique,0x12180,4) = INT_AND w8, 0x1
        assert_eq!(
            op.inputs[1],
            Varnode::Const {
                value: 1,
                size: None
            }
        );
    }

    #[test]
    fn test_parse_cbranch_and_store_have_no_output() {
        let p = parse_pcode(FIXTURE).unwrap();
        let cbr = &p[3].ops[1];
        assert_eq!(cbr.opcode, "CBRANCH");
        assert_eq!(cbr.out, None);
        assert_eq!(cbr.inputs[0], Varnode::CodeAddr(0x360));

        let store = p[5].ops.last().unwrap();
        assert_eq!(store.opcode, "STORE");
        assert_eq!(store.out, None);
        assert!(matches!(store.inputs[0], Varnode::Ram(_)));
        assert_eq!(store.inputs[1], Varnode::Register("w8".into()));
    }

    #[test]
    fn test_parse_load_ram_address() {
        let p = parse_pcode(FIXTURE).unwrap();
        let ld = &p[4].ops[1]; // (unique,0x24700,4) = LOAD ram[(unique,0x6000,8)]
        assert_eq!(ld.opcode, "LOAD");
        assert_eq!(
            ld.inputs[0],
            Varnode::Ram(Box::new(Varnode::Unique {
                offset: 0x6000,
                size: 8
            }))
        );
    }

    #[test]
    fn test_parse_rejects_orphan_op_line() {
        let err = parse_pcode("    sp = INT_SUB sp, 0x10").unwrap_err();
        assert!(err.0.contains("before any instruction header"));
    }
}
