//! Pure parsers for radare2 JSON output.
//!
//! Kept independent of the live r2 process so they can be exercised with
//! fixed fixtures and without a radare2 installation. Each parser maps
//! one r2 command output to a fragment of the normalized program model.

use r2smt_common::{Address, Arch, Error, Result};
use r2smt_ir::program::{BasicBlock, Function, Instruction, Operand, OperandKind};
use serde::Deserialize;

/// Architecture metadata extracted from `ij`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryInfo {
    /// Target instruction set.
    pub arch: Arch,
    /// Pointer width in bits.
    pub bits: u8,
    /// Entry point, when reported by r2.
    pub entry: Option<Address>,
}

#[derive(Debug, Deserialize)]
struct IjBin {
    arch: String,
    bits: u8,
}

#[derive(Debug, Deserialize)]
struct IjRoot {
    bin: IjBin,
}

/// Parse the response of `ij` into [`BinaryInfo`].
///
/// Sets `entry` to `None`; the dedicated entry-point query (`iej`) feeds
/// it via [`parse_entry`].
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed or the architecture
/// is unsupported.
pub fn parse_info(json: &str) -> Result<BinaryInfo> {
    let root: IjRoot = serde_json::from_str(json).map_err(|e| Error::parse("ij", e.to_string()))?;
    let arch = arch_from_str(&root.bin.arch, root.bin.bits)?;
    Ok(BinaryInfo {
        arch,
        bits: root.bin.bits,
        entry: None,
    })
}

fn arch_from_str(name: &str, bits: u8) -> Result<Arch> {
    match (name, bits) {
        ("x86", 32) => Ok(Arch::X86),
        ("x86", 64) => Ok(Arch::X86_64),
        // radare2 reports both AArch32 and AArch64 with arch="arm" and
        // discriminates via the bits field.
        ("arm", 32) => Ok(Arch::Arm),
        ("arm", 64) => Ok(Arch::Aarch64),
        _ => Err(Error::Unsupported(format!(
            "unsupported arch '{name}' ({bits} bits)"
        ))),
    }
}

#[derive(Debug, Deserialize)]
struct IjEntry {
    vaddr: u64,
}

/// Parse the response of `iej` and return the first entry's virtual
/// address, if any.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed.
pub fn parse_entry(json: &str) -> Result<Option<Address>> {
    let entries: Vec<IjEntry> =
        serde_json::from_str(json).map_err(|e| Error::parse("iej", e.to_string()))?;
    Ok(entries.first().map(|e| Address(e.vaddr)))
}

#[derive(Debug, Deserialize)]
struct AfljEntry {
    #[serde(alias = "offset")]
    addr: u64,
    name: String,
}

/// A reference to a discovered function returned by `aflj`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionRef {
    /// Function start address.
    pub address: Address,
    /// Symbolic name reported by radare2.
    pub name: String,
}

/// Parse the response of `aflj` into the list of function references.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed.
pub fn parse_function_list(json: &str) -> Result<Vec<FunctionRef>> {
    let raw: Vec<AfljEntry> =
        serde_json::from_str(json).map_err(|e| Error::parse("aflj", e.to_string()))?;
    Ok(raw
        .into_iter()
        .map(|f| FunctionRef {
            address: Address(f.addr),
            name: f.name,
        })
        .collect())
}

#[derive(Debug, Deserialize)]
struct ISjSection {
    vaddr: u64,
    #[serde(default)]
    vsize: u64,
    #[serde(default)]
    perm: String,
}

/// A half-open virtual-address interval `[start, end)` of a section
/// the loader maps as executable.
pub type ExecRange = (u64, u64);

/// Parse the response of `iSj` and return the half-open virtual
/// address intervals of every section the loader maps executable
/// (permission string contains `x`).
///
/// Used to enforce the format-grounded invariant that an instruction
/// must live in an executable mapping: radare2's analysis can
/// over-extend a function's CFG into data sections (string tables,
/// rodata) and decode that data as garbage instructions. Those blocks
/// are filtered out by [`address_in_ranges`] in the adapter. The set
/// is intentionally derived from r2's own permission model so it works
/// for ELF / PE / Mach-O without hardcoding section names.
///
/// Sections with zero `vsize` are skipped (they map no bytes).
/// Returns an empty vector when r2 reports no section view (e.g. a
/// fully stripped binary); callers must treat "no ranges known" as
/// "do not filter" rather than "filter everything".
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed.
pub fn parse_executable_ranges(json: &str) -> Result<Vec<ExecRange>> {
    let sections: Vec<ISjSection> =
        serde_json::from_str(json).map_err(|e| Error::parse("iSj", e.to_string()))?;
    let mut ranges: Vec<ExecRange> = sections
        .into_iter()
        .filter(|s| s.perm.contains('x') && s.vsize > 0)
        .filter_map(|s| s.vaddr.checked_add(s.vsize).map(|end| (s.vaddr, end)))
        .collect();
    ranges.sort_unstable();
    Ok(ranges)
}

/// `true` when `addr` falls inside any executable range. `ranges` is
/// the output of [`parse_executable_ranges`]; each entry is half-open
/// (`start` inclusive, `end` exclusive). The list is small (a handful
/// of code sections), so a linear scan is both clearest and fast
/// enough on the hot load path.
#[must_use]
pub fn address_in_ranges(ranges: &[ExecRange], addr: u64) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| start <= addr && addr < end)
}

/// Drop every block of `func` whose start address is not inside an
/// executable range, returning the number of blocks removed.
///
/// `ranges` must be the executable mapping from
/// [`parse_executable_ranges`]. The caller is responsible for the
/// "no ranges known" policy: an empty `ranges` would strip every
/// block, so callers must skip this call entirely when the section
/// view is unavailable (stripped binary) rather than pass `&[]`.
/// A debug assertion guards that contract.
pub fn retain_executable_blocks(func: &mut Function, ranges: &[ExecRange]) -> usize {
    debug_assert!(
        !ranges.is_empty(),
        "retain_executable_blocks called with no executable ranges — \
         caller must skip filtering when the section view is unknown"
    );
    let before = func.blocks.len();
    func.blocks
        .retain(|b| address_in_ranges(ranges, b.address.get()));
    before - func.blocks.len()
}

#[derive(Debug, Deserialize)]
struct AgfjOp {
    #[serde(alias = "offset")]
    addr: u64,
    size: u8,
    #[serde(default)]
    bytes: Option<String>,
    #[serde(default)]
    opcode: Option<String>,
    #[serde(default)]
    disasm: Option<String>,
    #[serde(default)]
    esil: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AgfjBlock {
    #[serde(alias = "offset")]
    addr: u64,
    #[serde(default)]
    jump: Option<u64>,
    #[serde(default)]
    fail: Option<u64>,
    #[serde(default)]
    ops: Vec<AgfjOp>,
}

#[derive(Debug, Deserialize)]
struct AgfjFunc {
    #[serde(alias = "offset")]
    addr: u64,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    blocks: Vec<AgfjBlock>,
    /// Per-function instruction-width hint from r2. `Some(16)` means
    /// the function decodes as Thumb / Thumb-2 on `Arch::Arm`; any
    /// other value (or absence) means 32-bit instructions.
    #[serde(default)]
    bits: Option<u8>,
}

/// Parse the response of `agfj @ <addr>` into a fully-populated
/// [`Function`], or `None` when r2 has no control-flow graph for the
/// queried address.
///
/// `agfj` returns an empty JSON array (`[]`) whenever the address is
/// not the head of a function r2 could decode into basic blocks —
/// import thunks, PLT stubs, data misclassified as code, or functions
/// r2 created a placeholder for but could not lift. That is an
/// expected, format-grounded outcome for *any* binary, not a corrupt
/// response, so it is surfaced as `Ok(None)` for callers that walk a
/// whole function list and can simply skip it. Genuinely malformed
/// JSON or an undecodable instruction byte string is still a contract
/// violation and propagates as [`Error::Parse`].
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed or if any
/// instruction byte string fails to decode as hex.
pub fn parse_function_blocks_opt(json: &str) -> Result<Option<Function>> {
    let funcs: Vec<AgfjFunc> =
        serde_json::from_str(json).map_err(|e| Error::parse("agfj", e.to_string()))?;
    let Some(func) = funcs.into_iter().next() else {
        return Ok(None);
    };

    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(func.blocks.len());
    for block in func.blocks {
        let mut successors: Vec<Address> = Vec::new();
        if let Some(j) = block.jump {
            successors.push(Address(j));
        }
        if let Some(f) = block.fail {
            successors.push(Address(f));
        }
        let mut instructions: Vec<Instruction> = Vec::with_capacity(block.ops.len());
        for op in block.ops {
            instructions.push(op_to_instruction(op, func.bits == Some(16))?);
        }
        blocks.push(BasicBlock {
            address: Address(block.addr),
            instructions,
            successors,
        });
    }

    Ok(Some(Function {
        address: Address(func.addr),
        name: func.name,
        blocks,
        is_thumb: func.bits == Some(16),
    }))
}

/// Strict variant of [`parse_function_blocks_opt`] for callers that
/// requested a *specific* function and therefore treat "r2 has no CFG
/// here" as a hard error (single-function load, block-at probes).
///
/// # Errors
///
/// Returns [`Error::Parse`] for malformed JSON, an undecodable
/// instruction byte string, or an empty `agfj` array (no function at
/// the queried address).
pub fn parse_function_blocks(json: &str) -> Result<Function> {
    parse_function_blocks_opt(json)?.ok_or_else(|| Error::parse("agfj", "empty function array"))
}

fn op_to_instruction(op: AgfjOp, function_is_thumb: bool) -> Result<Instruction> {
    let bytes = match op.bytes.as_deref() {
        Some(s) if !s.is_empty() => decode_hex_bytes(s)?,
        _ => Vec::new(),
    };
    let mnemonic_source = op
        .opcode
        .as_deref()
        .or(op.disasm.as_deref())
        .unwrap_or("")
        .trim();
    let (mnemonic, operands) = split_disasm(mnemonic_source);
    Ok(Instruction {
        address: Address(op.addr),
        size: op.size,
        bytes,
        mnemonic,
        operands,
        esil: op.esil,
        pcode: None,
        is_thumb: function_is_thumb,
    })
}

fn decode_hex_bytes(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(Error::parse("agfj", "odd-length byte string"));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let pair = s
            .get(i..i + 2)
            .ok_or_else(|| Error::parse("agfj", "byte string not ASCII-aligned"))?;
        let byte = u8::from_str_radix(pair, 16).map_err(|e| Error::parse("agfj", e.to_string()))?;
        out.push(byte);
    }
    Ok(out)
}

fn split_disasm(text: &str) -> (String, Vec<Operand>) {
    let mut parts = text.splitn(2, char::is_whitespace);
    let mnemonic = parts.next().map(str::to_lowercase).unwrap_or_default();
    let operand_str = parts.next().unwrap_or("").trim();
    if operand_str.is_empty() {
        return (mnemonic, Vec::new());
    }
    let operands = operand_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|raw| Operand {
            raw: raw.to_string(),
            kind: classify_operand(raw),
        })
        .collect();
    (mnemonic, operands)
}

/// Coarse control-flow classification of a single instruction, as
/// returned by `aoj 1 @ addr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsnFlow {
    /// Normal sequential instruction (mov, arith, …).
    Linear,
    /// Conditional branch (`jcc`).
    ConditionalBranch,
    /// Unconditional jump (`jmp`).
    UnconditionalJump,
    /// `call` — falls through after the callee returns.
    Call,
    /// Return-type terminator (`ret`).
    Return,
    /// Anything else (privileged, illegal, …) — treated as a
    /// terminator by the shellcode block finder.
    Other,
}

impl InsnFlow {
    /// `true` if the instruction terminates the current basic block
    /// (jumps unconditionally, returns, or otherwise stops fall-through).
    #[must_use]
    pub const fn ends_block(self) -> bool {
        matches!(
            self,
            Self::UnconditionalJump | Self::ConditionalBranch | Self::Return | Self::Other
        )
    }
}

/// One instruction as decoded by `aoj 1 @ addr`. The `mnemonic`,
/// `bytes`, and `size` carry enough to reconstruct an
/// [`Instruction`]; the `flow` field is what the shellcode finder
/// uses to detect block boundaries.
#[derive(Debug, Clone)]
pub struct AojInstruction {
    /// Instruction address.
    pub address: Address,
    /// Encoded instruction size in bytes.
    pub size: u8,
    /// Lower-case mnemonic.
    pub mnemonic: String,
    /// Operand list (raw text per token).
    pub operands: Vec<Operand>,
    /// Raw bytes as decoded from the `bytes` field, if r2 emitted it.
    pub bytes: Vec<u8>,
    /// ESIL string when available.
    pub esil: Option<String>,
    /// Control-flow classification.
    pub flow: InsnFlow,
}

#[derive(Debug, Deserialize)]
struct AojEntry {
    #[serde(alias = "offset")]
    addr: u64,
    size: u8,
    #[serde(default)]
    bytes: Option<String>,
    #[serde(default)]
    opcode: Option<String>,
    #[serde(default)]
    disasm: Option<String>,
    #[serde(default)]
    esil: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

/// Parse the response of `aoj N @ addr` into a list of
/// [`AojInstruction`]s.
///
/// # Errors
///
/// Returns [`Error::Parse`] if the JSON is malformed or any byte
/// string fails to decode as hex.
pub fn parse_aoj(json: &str) -> Result<Vec<AojInstruction>> {
    let entries: Vec<AojEntry> =
        serde_json::from_str(json).map_err(|e| Error::parse("aoj", e.to_string()))?;
    entries.into_iter().map(parse_aoj_entry).collect()
}

fn parse_aoj_entry(entry: AojEntry) -> Result<AojInstruction> {
    let bytes = match entry.bytes.as_deref() {
        Some(s) if !s.is_empty() => decode_hex_bytes(s)?,
        _ => Vec::new(),
    };
    let mnemonic_source = entry
        .opcode
        .as_deref()
        .or(entry.disasm.as_deref())
        .unwrap_or("")
        .trim();
    let (mnemonic, operands) = split_disasm(mnemonic_source);
    let flow = classify_aoj_kind(entry.kind.as_deref(), &mnemonic);
    Ok(AojInstruction {
        address: Address(entry.addr),
        size: entry.size,
        mnemonic,
        operands,
        bytes,
        esil: entry.esil,
        flow,
    })
}

fn classify_aoj_kind(kind: Option<&str>, mnemonic: &str) -> InsnFlow {
    // r2 returns `type` strings such as "cjmp", "ujmp", "jmp", "call",
    // "ret", "ret_far", "ill", "trap", "swi". Normalise to a small
    // enum so callers do not pattern-match on free-form strings.
    if let Some(t) = kind {
        let lower = t.trim().to_ascii_lowercase();
        if lower == "cjmp" || lower == "cond" {
            return InsnFlow::ConditionalBranch;
        }
        if lower == "jmp" || lower == "ujmp" || lower == "ijmp" || lower == "rjmp" {
            return InsnFlow::UnconditionalJump;
        }
        if lower == "call" || lower == "ucall" || lower == "icall" || lower == "rcall" {
            return InsnFlow::Call;
        }
        if lower.starts_with("ret") {
            return InsnFlow::Return;
        }
        if lower == "ill" || lower == "trap" || lower == "swi" {
            return InsnFlow::Other;
        }
        if !lower.is_empty() {
            return InsnFlow::Linear;
        }
    }
    // Fallback: classify by mnemonic prefix. This is approximate but
    // good enough for the shellcode finder when r2 has no `type`.
    let m = mnemonic.trim().to_ascii_lowercase();
    if m == "ret" || m.starts_with("ret") {
        return InsnFlow::Return;
    }
    if m == "jmp" {
        return InsnFlow::UnconditionalJump;
    }
    if m.starts_with('j') && m != "jmp" {
        return InsnFlow::ConditionalBranch;
    }
    if m.starts_with("call") {
        return InsnFlow::Call;
    }
    InsnFlow::Linear
}

#[derive(Debug, Deserialize)]
struct AxtjEntry {
    from: u64,
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

/// One inbound xref reported by `axtj @ addr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InboundXref {
    /// Address of the referring instruction.
    pub from: Address,
    /// r2 xref type — `"code"`, `"call"`, `"data"`, …
    pub kind: Option<String>,
}

/// Parse `axtj @ addr` into the inbound xrefs.
///
/// # Errors
///
/// Returns [`Error::Parse`] when the JSON is malformed.
pub fn parse_xrefs(json: &str) -> Result<Vec<InboundXref>> {
    let entries: Vec<AxtjEntry> =
        serde_json::from_str(json).map_err(|e| Error::parse("axtj", e.to_string()))?;
    Ok(entries
        .into_iter()
        .map(|e| InboundXref {
            from: Address(e.from),
            kind: e.kind,
        })
        .collect())
}

/// An `afvj` entry's `"ref"` payload, which differs by `kind`:
///
/// - Stack-relative locals (`bp` / `sp`) carry an object `{base,
///   offset}` describing the slot.
/// - Register-typed locals (`reg`) carry a bare string with the
///   register name (e.g. `"rdi"`).
///
/// The enum lets serde tolerate either shape without losing the
/// payload.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AfvjRef {
    Stack {
        #[serde(default)]
        base: Option<String>,
        #[serde(default)]
        offset: Option<i64>,
    },
    Register(String),
}

#[derive(Debug, Deserialize)]
struct AfvjEntry {
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "ref")]
    location: Option<AfvjRef>,
}

#[derive(Debug, Deserialize)]
struct AfvjRoot {
    #[serde(default)]
    bp: Vec<AfvjEntry>,
    #[serde(default)]
    sp: Vec<AfvjEntry>,
    #[serde(default)]
    reg: Vec<AfvjEntry>,
}

/// One named stack-slot local surfaced by `afvj @ fn_addr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalVariable {
    /// Analyst-facing name r2 assigned (`var_4h`, `arg_8h`, …).
    pub name: String,
    /// Canonical stack-slot key matching the lifter's naming
    /// (`stk_rbp_-4`, `stk_rsp_+8`).
    pub stack_slot: String,
}

/// One register-typed argument / local surfaced by `afvj @ fn_addr`.
///
/// Populated when the analyst (or the calling-convention pass) has
/// named a value that lives in a register — e.g. `arg1` →
/// `register = "rdi"` on the x86-64 System V ABI. The pretty-printer
/// uses this to render the analyst-supplied name instead of the bare
/// register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterRename {
    /// Analyst-facing name r2 assigned (`arg1`, `userInput`, …).
    pub name: String,
    /// Register the value lives in, as r2 reports it (typically the
    /// parent name: `"rdi"`, `"rsi"`, `"x0"`, …). The provider stores
    /// this verbatim against [`NameHints::registers`] — analysts who
    /// rename a sub-register alias (`edi`) will get the alias only
    /// when the lifter's free input matches that exact spelling.
    ///
    /// [`NameHints::registers`]: r2smt_ir::name_hints::NameHints::registers
    pub register: String,
}

/// Locals surfaced by `afvj @ fn_addr`, split by storage class.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Locals {
    /// BP- / SP-relative stack slots (the entries r2 returns under
    /// the `bp` and `sp` keys with a base register r2SMT recognises).
    pub stack_slots: Vec<LocalVariable>,
    /// Register-typed entries (the `reg` key) — function arguments
    /// or analyst-named values that live in a register.
    pub registers: Vec<RegisterRename>,
}

/// Parse `afvj @ fn` into the named locals it exposes.
///
/// Stack-relative entries with a base register r2SMT recognises
/// (`rbp` / `rsp` and their 32-bit aliases) land in
/// [`Locals::stack_slots`]; entries with any other base are dropped.
/// Register-typed entries land in [`Locals::registers`] verbatim
/// (provider canonicalisation is intentionally not done here so the
/// parser stays arch-agnostic).
///
/// # Errors
///
/// Returns [`Error::Parse`] when the JSON is malformed.
pub fn parse_locals(json: &str) -> Result<Locals> {
    // r2 6.x emits an object; older builds emitted an array. Try the
    // object first, fall back to an array of entries.
    let root: AfvjRoot = if let Ok(r) = serde_json::from_str::<AfvjRoot>(json) {
        r
    } else {
        let flat: Vec<AfvjEntry> =
            serde_json::from_str(json).map_err(|e| Error::parse("afvj", e.to_string()))?;
        AfvjRoot {
            bp: flat,
            sp: Vec::new(),
            reg: Vec::new(),
        }
    };
    let mut out = Locals::default();
    for entry in root.bp.into_iter().chain(root.sp) {
        let Some(name) = entry.name.filter(|n| !n.is_empty()) else {
            continue;
        };
        let Some(AfvjRef::Stack { base, offset }) = entry.location else {
            // `bp` / `sp` arrays should only contain stack refs; a
            // stray string-shaped ref is dropped rather than reported
            // as a register, since the storage class metadata is more
            // authoritative.
            continue;
        };
        let base = base.as_deref().unwrap_or("").to_ascii_lowercase();
        let canonical_base = match base.as_str() {
            "rbp" | "ebp" | "bp" => "rbp",
            "rsp" | "esp" | "sp" => "rsp",
            _ => continue,
        };
        let offset = offset.unwrap_or(0);
        out.stack_slots.push(LocalVariable {
            name,
            stack_slot: format!("stk_{canonical_base}_{offset}"),
        });
    }
    for entry in root.reg {
        let Some(name) = entry.name.filter(|n| !n.is_empty()) else {
            continue;
        };
        let register = match entry.location {
            Some(AfvjRef::Register(r)) => r,
            // Some r2 builds repeat the storage in object form
            // (`{"base": "rdi", "offset": 0}` for a register). Accept
            // both for resilience.
            Some(AfvjRef::Stack {
                base: Some(b),
                offset: _,
            }) => b,
            _ => continue,
        };
        let register = register.trim().to_ascii_lowercase();
        if register.is_empty() {
            continue;
        }
        out.registers.push(RegisterRename { name, register });
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
struct FdjEntry {
    name: String,
}

/// Resolve `fdj @ addr` (nearest flag / symbol). r2 may emit either a
/// single object or an array; both shapes are accepted. Returns the
/// first flag name r2 reported, if any.
///
/// # Errors
///
/// Returns [`Error::Parse`] when the JSON is malformed.
pub fn parse_flag(json: &str) -> Result<Option<String>> {
    let trimmed = json.trim();
    if trimmed.is_empty() || trimmed == "null" || trimmed == "{}" || trimmed == "[]" {
        return Ok(None);
    }
    if trimmed.starts_with('[') {
        let list: Vec<FdjEntry> =
            serde_json::from_str(trimmed).map_err(|e| Error::parse("fdj", e.to_string()))?;
        Ok(list.into_iter().next().map(|e| e.name))
    } else {
        let entry: FdjEntry =
            serde_json::from_str(trimmed).map_err(|e| Error::parse("fdj", e.to_string()))?;
        Ok(Some(entry.name))
    }
}

fn classify_operand(raw: &str) -> OperandKind {
    if raw.starts_with('[') || raw.contains("ptr ") || raw.contains('[') {
        OperandKind::Memory
    } else if raw.starts_with("0x")
        || raw.starts_with("-0x")
        || raw.starts_with('#')
        || raw.chars().all(|c| c.is_ascii_digit() || c == '-')
    {
        // `#`-prefixed immediates come from AArch64 / AArch32
        // disassembly (`mov x0, #0x10`). r2 emits them verbatim;
        // classify them as `Immediate` so the lifter's
        // `parse_immediate` (which strips the `#`) takes the right
        // path.
        OperandKind::Immediate
    } else if raw.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        OperandKind::Register
    } else {
        OperandKind::Unknown
    }
}

/// Group a `pdgsd` dump into `(address, text)` per instruction, where
/// `text` is the instruction's `0x…:` header line plus its indented
/// P-code op lines. Lets the r2ghidra adapter attach per-instruction
/// P-code to [`r2smt_ir::program::Instruction::pcode`] so the pure
/// [`r2smt_pcode`] lifter never touches a live r2 process.
///
/// Pure and fixture-testable; non-header, non-indented noise (stray r2
/// log lines) is ignored.
#[must_use]
pub fn split_pdgsd_by_instruction(dump: &str) -> Vec<(u64, String)> {
    let mut out: Vec<(u64, String)> = Vec::new();
    for line in dump.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with(char::is_whitespace) {
            if let Some((_, text)) = out.last_mut() {
                text.push('\n');
                text.push_str(line);
            }
            continue;
        }
        if let Some(addr) = line
            .split_once(':')
            .and_then(|(a, _)| a.trim().strip_prefix("0x"))
            .and_then(|h| u64::from_str_radix(h, 16).ok())
        {
            out.push((addr, line.to_string()));
        }
    }
    out
}

/// r2 sentinels printed when no decompiler plugin is loaded or the
/// command is unknown. Generic to the tool, not to any sample.
const DECOMPILE_SENTINELS: [&str; 4] = [
    "cannot find decompiler",
    "unknown command",
    "no function",
    "command not found",
];

/// Extract decompiled C from an r2ghidra `pdgj` response.
///
/// Returns `None` when the JSON is unparseable, carries no `code`
/// field, or the code is blank — i.e. when r2ghidra is absent or
/// produced nothing for the address.
#[must_use]
pub fn parse_pdgj(json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json.trim()).ok()?;
    let code = v.get("code")?.as_str()?.trim();
    if code.is_empty() {
        None
    } else {
        Some(code.to_string())
    }
}

/// Extract decompiled C from an r2dec `pddj` response by joining its
/// `lines[].str` fragments. Returns `None` when the JSON is
/// unparseable or carries no renderable lines.
#[must_use]
pub fn parse_pddj(json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json.trim()).ok()?;
    let lines = v.get("lines")?.as_array()?;
    let mut out = String::new();
    for line in lines {
        if let Some(s) = line.get("str").and_then(serde_json::Value::as_str) {
            out.push_str(s);
            out.push('\n');
        }
    }
    let trimmed = out.trim_end();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Accept plain `pdg` text only when it is non-empty and not one of
/// r2's "no decompiler / unknown command" sentinels. Structured
/// `pdgj` / `pddj` are preferred; this is the last-resort path.
#[must_use]
pub fn clean_plain_decompile(raw: &str) -> Option<String> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    if DECOMPILE_SENTINELS.iter().any(|s| lower.contains(s)) {
        return None;
    }
    Some(t.to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const IJ_FIXTURE: &str = r#"{
        "core": {},
        "bin": {"arch": "x86", "bits": 64, "baddr": "0x100000000"}
    }"#;

    const IEJ_FIXTURE: &str = r#"[
        {"vaddr": 4198720, "paddr": 4096, "haddr": 64, "type": "program"}
    ]"#;

    const AFLJ_FIXTURE: &str = r#"[
        {"offset": 4198720, "name": "sym.main", "size": 24},
        {"offset": 4198912, "name": "sym.helper", "size": 16}
    ]"#;

    const AGFJ_FIXTURE: &str = r#"[
        {
            "name": "sym.main",
            "offset": 4198720,
            "blocks": [
                {
                    "offset": 4198720,
                    "jump": 4198736,
                    "fail": 4198740,
                    "ops": [
                        {
                            "offset": 4198720,
                            "size": 2,
                            "bytes": "31c0",
                            "opcode": "xor eax, eax",
                            "esil": "0,eax,="
                        },
                        {
                            "offset": 4198722,
                            "size": 5,
                            "bytes": "b810000000",
                            "opcode": "mov eax, 0x10"
                        }
                    ]
                }
            ]
        }
    ]"#;

    #[test]
    fn parse_info_extracts_arch_and_bits() {
        let info = parse_info(IJ_FIXTURE).unwrap();
        assert_eq!(info.arch, Arch::X86_64);
        assert_eq!(info.bits, 64);
        assert_eq!(info.entry, None);
    }

    #[test]
    fn parse_info_rejects_unsupported_arch() {
        let bad = r#"{"bin": {"arch": "mips", "bits": 32}}"#;
        assert!(parse_info(bad).is_err());
        let bad_bits = r#"{"bin": {"arch": "arm", "bits": 16}}"#;
        assert!(parse_info(bad_bits).is_err());
    }

    #[test]
    fn parse_info_accepts_arm_32_and_aarch64() {
        let arm32 = r#"{"bin": {"arch": "arm", "bits": 32}}"#;
        let info = parse_info(arm32).unwrap();
        assert_eq!(info.arch, Arch::Arm);
        assert_eq!(info.bits, 32);
        let aarch64 = r#"{"bin": {"arch": "arm", "bits": 64}}"#;
        let info = parse_info(aarch64).unwrap();
        assert_eq!(info.arch, Arch::Aarch64);
        assert_eq!(info.bits, 64);
    }

    #[test]
    fn parse_entry_returns_first_vaddr() {
        let entry = parse_entry(IEJ_FIXTURE).unwrap();
        assert_eq!(entry, Some(Address(4_198_720)));
    }

    #[test]
    fn parse_entry_empty_list_returns_none() {
        let entry = parse_entry("[]").unwrap();
        assert_eq!(entry, None);
    }

    #[test]
    fn parse_function_list_returns_two_functions() {
        let funcs = parse_function_list(AFLJ_FIXTURE).unwrap();
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].address, Address(4_198_720));
        assert_eq!(funcs[0].name, "sym.main");
        assert_eq!(funcs[1].name, "sym.helper");
    }

    #[test]
    fn parse_function_blocks_recovers_instructions_and_successors() {
        let func = parse_function_blocks(AGFJ_FIXTURE).unwrap();
        assert_eq!(func.address, Address(4_198_720));
        assert_eq!(func.name.as_deref(), Some("sym.main"));
        assert_eq!(func.blocks.len(), 1);
        let block = &func.blocks[0];
        assert_eq!(
            block.successors,
            vec![Address(4_198_736), Address(4_198_740)]
        );
        assert_eq!(block.instructions.len(), 2);
        assert_eq!(block.instructions[0].mnemonic, "xor");
        assert_eq!(block.instructions[0].bytes, vec![0x31, 0xc0]);
        assert_eq!(block.instructions[1].mnemonic, "mov");
        assert_eq!(
            block.instructions[1].bytes,
            vec![0xb8, 0x10, 0x00, 0x00, 0x00]
        );
        assert_eq!(block.instructions[0].operands.len(), 2);
        assert_eq!(
            block.instructions[0].operands[0].kind,
            OperandKind::Register
        );
        assert_eq!(
            block.instructions[1].operands[1].kind,
            OperandKind::Immediate
        );
    }

    #[test]
    fn parse_function_blocks_rejects_empty_array() {
        // Strict wrapper still treats "no function here" as an error
        // for single-function callers (load_function / block-at).
        assert!(parse_function_blocks("[]").is_err());
    }

    const ISJ_FIXTURE: &str = r#"[
        {"name":".text","vaddr":4096,"vsize":256,"perm":"-r-x"},
        {"name":".plt","vaddr":4352,"vsize":64,"perm":"-r-x"},
        {"name":".rodata","vaddr":8192,"vsize":512,"perm":"-r--"},
        {"name":".data","vaddr":12288,"vsize":128,"perm":"-rw-"},
        {"name":".bss","vaddr":16384,"vsize":0,"perm":"-rwx"}
    ]"#;

    #[test]
    fn parse_executable_ranges_extracts_x_perm_sections() {
        let r = parse_executable_ranges(ISJ_FIXTURE).unwrap();
        // .text [4096,4352) and .plt [4352,4416); .bss has vsize 0 so
        // it maps no bytes and is skipped despite the x permission.
        assert_eq!(r, vec![(4096, 4352), (4352, 4416)]);
    }

    #[test]
    fn parse_executable_ranges_excludes_non_exec_sections() {
        let r = parse_executable_ranges(ISJ_FIXTURE).unwrap();
        // .rodata (8192) and .data (12288) must not appear.
        assert!(!address_in_ranges(&r, 8192));
        assert!(!address_in_ranges(&r, 12288));
    }

    #[test]
    fn parse_executable_ranges_empty_when_no_sections() {
        // Stripped binary: r2 reports an empty section array. Callers
        // treat this as "do not filter", never "filter everything".
        assert!(parse_executable_ranges("[]").unwrap().is_empty());
    }

    #[test]
    fn parse_executable_ranges_errors_on_malformed_json() {
        assert!(parse_executable_ranges("{ not json").is_err());
    }

    #[test]
    fn address_in_ranges_is_half_open() {
        let ranges = vec![(0x1000u64, 0x1100u64)];
        assert!(!address_in_ranges(&ranges, 0x0FFF));
        assert!(address_in_ranges(&ranges, 0x1000)); // start inclusive
        assert!(address_in_ranges(&ranges, 0x10FF));
        assert!(!address_in_ranges(&ranges, 0x1100)); // end exclusive
        assert!(!address_in_ranges(&[], 0x1000)); // empty => never inside
    }

    fn block_at(addr: u64) -> BasicBlock {
        BasicBlock {
            address: Address(addr),
            instructions: Vec::new(),
            successors: Vec::new(),
        }
    }

    fn func_with_blocks(addr: u64, block_addrs: &[u64]) -> Function {
        Function {
            address: Address(addr),
            name: Some("sym.test".into()),
            blocks: block_addrs.iter().copied().map(block_at).collect(),
            is_thumb: false,
        }
    }

    #[test]
    fn retain_executable_blocks_drops_data_blocks_keeps_text() {
        // .text [0x1000,0x2000). One real block at 0x1000, one block
        // r2 over-extended into a string table at 0x9000 (data).
        let ranges = vec![(0x1000u64, 0x2000u64)];
        let mut f = func_with_blocks(0x1000, &[0x1000, 0x9000, 0x1100]);
        let dropped = retain_executable_blocks(&mut f, &ranges);
        assert_eq!(dropped, 1);
        let kept: Vec<u64> = f.blocks.iter().map(|b| b.address.get()).collect();
        assert_eq!(kept, vec![0x1000, 0x1100]);
    }

    #[test]
    fn retain_executable_blocks_empties_a_fully_data_function() {
        // Every block r2 attributed to this "function" is in data —
        // the whole thing is an analysis artefact. The caller skips
        // functions left with zero blocks.
        let ranges = vec![(0x1000u64, 0x2000u64)];
        let mut f = func_with_blocks(0x0014_0710, &[0x0005_5ebc, 0x0005_6018]);
        let dropped = retain_executable_blocks(&mut f, &ranges);
        assert_eq!(dropped, 2);
        assert!(f.blocks.is_empty());
    }

    #[test]
    fn parse_function_blocks_opt_returns_none_for_empty_array() {
        // r2 returns `[]` for an address with no decoded CFG (import
        // thunk / data / placeholder). The list-walking caller must be
        // able to skip it rather than abort the whole program load.
        assert_eq!(parse_function_blocks_opt("[]").unwrap(), None);
    }

    #[test]
    fn parse_function_blocks_opt_returns_some_for_real_function() {
        let func = parse_function_blocks_opt(AGFJ_MODERN).unwrap();
        assert!(func.is_some());
    }

    #[test]
    fn parse_function_blocks_opt_still_errors_on_malformed_json() {
        // A genuinely corrupt response is a contract violation and
        // must NOT be silently swallowed as "no function here".
        assert!(parse_function_blocks_opt("{not json").is_err());
        assert!(parse_function_blocks_opt("[{\"addr\": \"oops\"}]").is_err());
    }

    #[test]
    fn decode_hex_bytes_rejects_odd_length() {
        assert!(decode_hex_bytes("abc").is_err());
    }

    #[test]
    fn classify_operand_known_cases() {
        assert_eq!(classify_operand("eax"), OperandKind::Register);
        assert_eq!(classify_operand("0x10"), OperandKind::Immediate);
        assert_eq!(classify_operand("[rbp - 4]"), OperandKind::Memory);
        assert_eq!(classify_operand("dword ptr [eax]"), OperandKind::Memory);
        // AArch64 / AArch32 immediates carry a `#` prefix.
        assert_eq!(classify_operand("#0x10"), OperandKind::Immediate);
        assert_eq!(classify_operand("#-1"), OperandKind::Immediate);
        assert_eq!(classify_operand("#42"), OperandKind::Immediate);
        // AArch64 register tokens stay register-classified.
        assert_eq!(classify_operand("x0"), OperandKind::Register);
        assert_eq!(classify_operand("w0"), OperandKind::Register);
    }

    // The radare2 schema migrated from `offset` to `addr` between r2 5.x
    // and 6.x. The parsers use `#[serde(alias = "offset")]` so both
    // schemas keep working; these tests pin that behaviour.

    const AFLJ_MODERN: &str = r#"[
        {"addr": 4198720, "name": "sym.main", "size": 24}
    ]"#;

    const AGFJ_MODERN: &str = r#"[
        {
            "name": "sym.main",
            "addr": 4198720,
            "blocks": [
                {
                    "addr": 4198720,
                    "jump": 4198736,
                    "fail": 4198740,
                    "ops": [
                        {
                            "addr": 4198720,
                            "size": 2,
                            "bytes": "31c0",
                            "opcode": "xor eax, eax"
                        }
                    ]
                }
            ]
        }
    ]"#;

    #[test]
    fn parse_function_list_accepts_modern_addr_schema() {
        let funcs = parse_function_list(AFLJ_MODERN).unwrap();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].address, Address(4_198_720));
    }

    #[test]
    fn parse_function_blocks_accepts_modern_addr_schema() {
        let func = parse_function_blocks(AGFJ_MODERN).unwrap();
        assert_eq!(func.address, Address(4_198_720));
        assert_eq!(func.blocks[0].address, Address(4_198_720));
        assert_eq!(func.blocks[0].instructions[0].address, Address(4_198_720));
    }

    #[test]
    fn parse_function_blocks_marks_thumb_when_bits_16() {
        let json = r#"[
            {
                "addr": 32774,
                "name": "sym.thumb_fn",
                "bits": 16,
                "blocks": [
                    {
                        "addr": 32774,
                        "ops": [
                            {
                                "addr": 32774,
                                "size": 2,
                                "bytes": "00bf",
                                "opcode": "nop"
                            }
                        ]
                    }
                ]
            }
        ]"#;
        let func = parse_function_blocks(json).unwrap();
        assert!(func.is_thumb);
        assert!(func.blocks[0].instructions[0].is_thumb);
    }

    #[test]
    fn parse_function_blocks_defaults_to_arm_mode_when_bits_absent() {
        // Most agfj responses omit `bits`. The parser must default to
        // ARM mode (is_thumb == false) so existing fixtures keep
        // working unchanged.
        let func = parse_function_blocks(AGFJ_MODERN).unwrap();
        assert!(!func.is_thumb);
        assert!(!func.blocks[0].instructions[0].is_thumb);
    }

    // ----- aoj / axtj / afvj / fdj ----------------------------------

    #[test]
    fn parse_aoj_classifies_branch_type() {
        let json = r#"[
            {"addr": 4198720, "size": 6, "bytes": "0f8500000000",
             "opcode": "jne 0x401080", "type": "cjmp"},
            {"addr": 4198726, "size": 1, "bytes": "c3",
             "opcode": "ret", "type": "ret"},
            {"addr": 4198727, "size": 5, "bytes": "e900000000",
             "opcode": "jmp 0x401100", "type": "jmp"},
            {"addr": 4198732, "size": 3, "bytes": "488d05",
             "opcode": "mov rax, 1", "type": "mov"}
        ]"#;
        let parsed = parse_aoj(json).unwrap();
        assert_eq!(parsed[0].flow, InsnFlow::ConditionalBranch);
        assert_eq!(parsed[1].flow, InsnFlow::Return);
        assert_eq!(parsed[2].flow, InsnFlow::UnconditionalJump);
        assert_eq!(parsed[3].flow, InsnFlow::Linear);
        assert_eq!(parsed[0].mnemonic, "jne");
        assert_eq!(parsed[1].bytes, vec![0xC3]);
    }

    #[test]
    fn parse_aoj_falls_back_to_mnemonic_when_type_missing() {
        let json = r#"[{"addr": 100, "size": 2, "bytes": "74fe", "opcode": "je 0x60"}]"#;
        let parsed = parse_aoj(json).unwrap();
        assert_eq!(parsed[0].flow, InsnFlow::ConditionalBranch);
    }

    #[test]
    fn insn_flow_ends_block_only_for_terminators() {
        assert!(InsnFlow::ConditionalBranch.ends_block());
        assert!(InsnFlow::UnconditionalJump.ends_block());
        assert!(InsnFlow::Return.ends_block());
        assert!(!InsnFlow::Linear.ends_block());
        assert!(!InsnFlow::Call.ends_block());
    }

    #[test]
    fn parse_xrefs_returns_inbound_refs() {
        let json = r#"[
            {"from": 4198704, "type": "code", "opcode": "call sym.main"},
            {"from": 4198800, "type": "data"}
        ]"#;
        let xrefs = parse_xrefs(json).unwrap();
        assert_eq!(xrefs.len(), 2);
        assert_eq!(xrefs[0].from, Address(4_198_704));
        assert_eq!(xrefs[0].kind.as_deref(), Some("code"));
        assert_eq!(xrefs[1].kind.as_deref(), Some("data"));
    }

    #[test]
    fn parse_xrefs_empty_array_returns_empty() {
        let xrefs = parse_xrefs("[]").unwrap();
        assert!(xrefs.is_empty());
    }

    #[test]
    fn parse_locals_returns_local_names_keyed_by_stack_slot() {
        let json = r#"{
            "bp": [
                {"name": "var_4h", "kind": "v", "type": "int",
                 "ref": {"base": "rbp", "offset": -4}},
                {"name": "arg_8h", "kind": "a", "type": "int",
                 "ref": {"base": "rbp", "offset": 8}}
            ],
            "sp": [
                {"name": "local_10h", "kind": "v", "type": "int",
                 "ref": {"base": "rsp", "offset": 16}}
            ]
        }"#;
        let locals = parse_locals(json).unwrap();
        assert_eq!(locals.stack_slots.len(), 3);
        assert!(locals.registers.is_empty());
        assert!(
            locals
                .stack_slots
                .iter()
                .any(|l| l.name == "var_4h" && l.stack_slot == "stk_rbp_-4")
        );
        assert!(
            locals
                .stack_slots
                .iter()
                .any(|l| l.name == "arg_8h" && l.stack_slot == "stk_rbp_8")
        );
        assert!(
            locals
                .stack_slots
                .iter()
                .any(|l| l.name == "local_10h" && l.stack_slot == "stk_rsp_16")
        );
    }

    #[test]
    fn parse_locals_skips_unsupported_bases() {
        let json = r#"{
            "bp": [{"name": "weird", "kind": "v",
                    "ref": {"base": "rax", "offset": 0}}]
        }"#;
        let locals = parse_locals(json).unwrap();
        assert!(locals.stack_slots.is_empty());
        assert!(locals.registers.is_empty());
    }

    #[test]
    fn parse_locals_handles_empty_object() {
        let locals = parse_locals("{}").unwrap();
        assert!(locals.stack_slots.is_empty());
        assert!(locals.registers.is_empty());
    }

    #[test]
    fn parse_locals_surfaces_register_renames_with_string_ref() {
        // r2 6.x emits register-typed locals as a bare string in
        // `ref`. The parser must surface them under
        // `Locals::registers` without touching `stack_slots`.
        let json = r#"{
            "bp": [],
            "sp": [],
            "reg": [
                {"name": "arg1", "kind": "r", "type": "int", "ref": "rdi"},
                {"name": "userInput", "kind": "r", "type": "char*", "ref": "rsi"}
            ]
        }"#;
        let locals = parse_locals(json).unwrap();
        assert!(locals.stack_slots.is_empty());
        assert_eq!(locals.registers.len(), 2);
        assert!(
            locals
                .registers
                .iter()
                .any(|r| r.name == "arg1" && r.register == "rdi")
        );
        assert!(
            locals
                .registers
                .iter()
                .any(|r| r.name == "userInput" && r.register == "rsi")
        );
    }

    #[test]
    fn parse_locals_accepts_register_ref_in_object_form() {
        // Older r2 builds emitted register entries as
        // `{"base": "rdi", "offset": 0}` instead of a bare string.
        // The parser tolerates this shape too.
        let json = r#"{
            "reg": [
                {"name": "arg1", "kind": "r",
                 "ref": {"base": "rdi", "offset": 0}}
            ]
        }"#;
        let locals = parse_locals(json).unwrap();
        assert_eq!(locals.registers.len(), 1);
        assert_eq!(locals.registers[0].name, "arg1");
        assert_eq!(locals.registers[0].register, "rdi");
    }

    #[test]
    fn parse_locals_normalises_register_name_case() {
        // r2 occasionally emits register names in mixed case
        // (`"RDI"`). The lifter compares against lowercase, so the
        // parser normalises at the boundary.
        let json = r#"{
            "reg": [{"name": "arg1", "ref": "RDI"}]
        }"#;
        let locals = parse_locals(json).unwrap();
        assert_eq!(locals.registers.len(), 1);
        assert_eq!(locals.registers[0].register, "rdi");
    }

    #[test]
    fn parse_locals_drops_register_entries_with_empty_payload() {
        let json = r#"{
            "reg": [
                {"name": "ok", "ref": "rdi"},
                {"name": "missing_ref"},
                {"name": "blank", "ref": ""},
                {"ref": "rsi"}
            ]
        }"#;
        let locals = parse_locals(json).unwrap();
        assert_eq!(locals.registers.len(), 1);
        assert_eq!(locals.registers[0].name, "ok");
    }

    #[test]
    fn parse_flag_returns_name_for_object_form() {
        let json = r#"{"name": "sym.main", "offset": 4198720, "size": 0}"#;
        let flag = parse_flag(json).unwrap();
        assert_eq!(flag.as_deref(), Some("sym.main"));
    }

    #[test]
    fn parse_flag_returns_first_name_for_array_form() {
        let json = r#"[{"name": "sym.first"}, {"name": "sym.second"}]"#;
        let flag = parse_flag(json).unwrap();
        assert_eq!(flag.as_deref(), Some("sym.first"));
    }

    #[test]
    fn parse_flag_returns_none_for_empty_responses() {
        assert_eq!(parse_flag("").unwrap(), None);
        assert_eq!(parse_flag("null").unwrap(), None);
        assert_eq!(parse_flag("[]").unwrap(), None);
        assert_eq!(parse_flag("{}").unwrap(), None);
    }

    #[test]
    fn parse_pdgj_extracts_code_field() {
        let json = r#"{"code": "int main() {\n  return 0;\n}\n", "annotations": []}"#;
        assert_eq!(
            parse_pdgj(json).as_deref(),
            Some("int main() {\n  return 0;\n}")
        );
    }

    #[test]
    fn parse_pdgj_returns_none_when_backend_absent() {
        assert_eq!(parse_pdgj("Cannot find decompiler for current arch"), None);
        assert_eq!(parse_pdgj(""), None);
        assert_eq!(parse_pdgj("{}"), None);
        assert_eq!(parse_pdgj(r#"{"code": "   "}"#), None);
    }

    #[test]
    fn parse_pddj_joins_line_fragments() {
        let json = r#"{"errors":[],"log":[],"lines":[{"str":"int f(void) {"},{"str":"  return 1;"},{"str":"}"}]}"#;
        assert_eq!(
            parse_pddj(json).as_deref(),
            Some("int f(void) {\n  return 1;\n}")
        );
    }

    #[test]
    fn parse_pddj_returns_none_without_lines() {
        assert_eq!(parse_pddj(r#"{"errors":["r2dec failed"]}"#), None);
        assert_eq!(parse_pddj("not json"), None);
        assert_eq!(parse_pddj(r#"{"lines":[]}"#), None);
    }

    #[test]
    fn clean_plain_decompile_filters_sentinels_but_keeps_code() {
        assert_eq!(clean_plain_decompile("Unknown command 'pdg'"), None);
        assert_eq!(clean_plain_decompile("   "), None);
        assert_eq!(
            clean_plain_decompile("undefined4 main(void)\n{\n  return 0;\n}\n").as_deref(),
            Some("undefined4 main(void)\n{\n  return 0;\n}")
        );
    }

    #[test]
    fn split_pdgsd_groups_ops_under_each_instruction() {
        let dump = "\
0x100: sub sp, sp, #0x10
    sp = INT_SUB sp, 0x10
0x104: mul w8, w8, w9
    (unique,0x2ae80,4) = INT_MULT w8, w9
    x8 = INT_ZEXT (unique,0x2ae80,4)";
        let groups = split_pdgsd_by_instruction(dump);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, 0x100);
        assert_eq!(
            groups[0].1,
            "0x100: sub sp, sp, #0x10\n    sp = INT_SUB sp, 0x10"
        );
        assert_eq!(groups[1].0, 0x104);
        assert!(groups[1].1.contains("INT_MULT w8, w9"));
        assert!(groups[1].1.contains("INT_ZEXT"));
    }

    #[test]
    fn split_pdgsd_ignores_stray_log_lines() {
        let dump = "WARN: something\n0x100: nop\n    --- nop has no ops\nERROR: noise";
        let groups = split_pdgsd_by_instruction(dump);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, 0x100);
    }
}
