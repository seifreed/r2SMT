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
mod tests;
