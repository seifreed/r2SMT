//! Live radare2 adapter built on top of `r2pipe`.

use std::path::Path;
use std::thread::sleep;
use std::time::Duration;

use r2pipe::{R2Pipe, R2PipeSpawnOptions};
use r2smt_common::{Address, Error, Result};
use r2smt_ir::annotator::Annotator;
use r2smt_ir::byte_patcher::BytePatcher;
use r2smt_ir::decompiler::Decompiler;
use r2smt_ir::name_hints::NameHints;
use r2smt_ir::program::{BasicBlock, Function, Instruction, Program};
use r2smt_ir::provider::BinaryProvider;
use tracing::{debug, info, warn};

use crate::b64;
use crate::parse;

/// How aggressively radare2 should analyse the binary at open time.
///
/// `Standard` runs the conventional `aaa` pass and matches every
/// previous r2SMT release. `Deep` runs `aaaa`, the experimental pass
/// that adds speculative function discovery, vtable parsing, and more
/// noref-aware heuristics — useful when the default pass misses an
/// opaque-predicate function but at the cost of additional runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnalysisLevel {
    /// Run `aaa` (default).
    Standard,
    /// Run `aaaa` for deeper, slower analysis.
    Deep,
}

impl AnalysisLevel {
    /// The r2 command this level executes after spawning the session.
    #[must_use]
    pub const fn command(self) -> &'static str {
        match self {
            Self::Standard => "aaa",
            Self::Deep => "aaaa",
        }
    }
}

/// Total number of times the adapter attempts the spawn + initial
/// analysis sequence before surfacing the failure.
///
/// radare2 startup is a `fork` + `exec` + bidirectional-pipe
/// handshake. Under concurrent process creation (parallel r2SMT
/// invocations, or r2SMT alongside other r2 sessions) that handshake
/// can transiently fail: `fork` can return `EAGAIN`, and the first
/// pipe read can be interrupted and surface as an empty response.
/// These are not contract violations — a small bounded retry absorbs
/// them. A genuinely missing `r2` binary or an unreadable target
/// produces the *same* error on every attempt and is not in the
/// transient allow-list, so it still fails fast.
const SPAWN_MAX_ATTEMPTS: usize = 4;

/// Base backoff between spawn retries, multiplied by the attempt
/// number (linear backoff). Worst-case cumulative wait with
/// [`SPAWN_MAX_ATTEMPTS`] = 4 is 120 + 240 + 360 ≈ 0.72 s, small
/// enough to stay invisible on success and bounded on failure.
const SPAWN_RETRY_BACKOFF_MS: u64 = 120;

/// `true` when `err` looks like a transient radare2 transport / spawn
/// failure that a retry can plausibly recover from, rather than a
/// stable contract violation (missing binary, unreadable file,
/// non-UTF-8 path). The discriminator is the foreign tool's own
/// message — r2pipe collapses every failure to a string, so this is
/// the only signal available. The allow-list is deliberately narrow
/// and lower-cased so it does not accidentally swallow real errors.
fn is_transient_spawn_error(err: &Error) -> bool {
    let Error::R2Pipe(message) = err else {
        return false;
    };
    let lowered = message.to_ascii_lowercase();
    [
        "i/o error",
        "empty response",
        "broken pipe",
        "resource temporarily unavailable",
        "interrupted",
        "connection reset",
        "would block",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

/// Owns a live radare2 session and answers [`BinaryProvider`] queries
/// against it.
pub struct R2PipeProvider {
    r2: R2Pipe,
    /// When set, [`BinaryProvider::load_program`] additionally attaches
    /// r2ghidra SLEIGH P-code to every instruction (opt-in `--ir
    /// pcode|auto`). Default `false` keeps the ESIL-only path
    /// byte-identical.
    attach_pcode_on_load: bool,
}

impl R2PipeProvider {
    /// Open `path` with radare2 (read-only) and run `aaa`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::R2Pipe`] if r2 cannot be spawned or `aaa` fails.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_analysis(path, false, AnalysisLevel::Standard)
    }

    /// Open `path` with radare2 (read-only) and run `aaaa` (deep
    /// experimental analysis). Useful when the standard pass misses
    /// functions r2SMT cares about — see Phase F roadmap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::R2Pipe`] if r2 cannot be spawned or the
    /// analysis command fails.
    pub fn open_deep(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_analysis(path, false, AnalysisLevel::Deep)
    }

    /// Open `path` with radare2 in write mode (`-w`) and run `aaa`.
    ///
    /// Required for `BytePatcher` writes, which need radare2 to hold
    /// the binary open for in-place modification. Callers must take a
    /// backup of the file *before* invoking this — radare2 may flush
    /// pending writes when the session is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::R2Pipe`] if r2 cannot be spawned or `aaa` fails.
    pub fn open_writable(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_analysis(path, true, AnalysisLevel::Standard)
    }

    /// Write-mode counterpart of [`Self::open_deep`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::R2Pipe`] if r2 cannot be spawned or the
    /// analysis command fails.
    pub fn open_writable_deep(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_analysis(path, true, AnalysisLevel::Deep)
    }

    /// Spawn radare2 and run the requested analysis pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::R2Pipe`] if r2 cannot be spawned or the
    /// analysis command rejects the binary.
    pub fn open_with_analysis(
        path: impl AsRef<Path>,
        writable: bool,
        level: AnalysisLevel,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let path_str = path_ref
            .to_str()
            .ok_or_else(|| Error::r2pipe("non-UTF-8 binary path"))?;
        info!(
            target: "r2smt::r2pipe",
            path = %path_str,
            writable,
            level = level.command(),
            "spawning radare2"
        );
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            match Self::try_spawn(path_str, writable, level) {
                Ok(provider) => return Ok(provider),
                Err(err) if attempt < SPAWN_MAX_ATTEMPTS && is_transient_spawn_error(&err) => {
                    warn!(
                        target: "r2smt::r2pipe",
                        attempt,
                        max_attempts = SPAWN_MAX_ATTEMPTS,
                        error = %err,
                        "transient radare2 spawn failure — retrying"
                    );
                    sleep(Duration::from_millis(
                        SPAWN_RETRY_BACKOFF_MS * attempt as u64,
                    ));
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// One spawn + initial-analysis attempt. Separated from the retry
    /// loop so the transient-failure policy lives in exactly one place
    /// and the happy path stays a straight line.
    fn try_spawn(path_str: &str, writable: bool, level: AnalysisLevel) -> Result<Self> {
        let opts = if writable {
            Some(R2PipeSpawnOptions {
                exepath: "r2".to_string(),
                args: vec!["-w"],
            })
        } else {
            None
        };
        let mut r2 = R2Pipe::spawn(path_str, opts).map_err(Error::r2pipe)?;
        let command = level.command();
        debug!(target: "r2smt::r2pipe", command, "running analysis pass");
        r2.cmd(command).map_err(Error::r2pipe)?;
        Ok(Self {
            r2,
            attach_pcode_on_load: false,
        })
    }

    /// Opt instruction-level SLEIGH P-code attachment in/out for the
    /// next [`BinaryProvider::load_program`] call. Default is off.
    pub fn set_attach_pcode(&mut self, on: bool) {
        self.attach_pcode_on_load = on;
    }

    fn cmd(&mut self, command: &str) -> Result<String> {
        debug!(target: "r2smt::r2pipe", command, "issuing r2 command");
        self.r2.cmd(command).map_err(Error::r2pipe)
    }

    /// Save the current radare2 session as a project so the annotations
    /// applied through [`Annotator::set_comment`] can be reopened later
    /// with `r2 -p <name>`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::R2Pipe`] if the project name is empty or r2
    /// reports a failure.
    pub fn save_project(&mut self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(Error::r2pipe("project name must be non-empty"));
        }
        if name.chars().any(|c| c == '\n' || c == '\r' || c == ';') {
            return Err(Error::r2pipe(
                "project name must not contain newlines or ';'",
            ));
        }
        let response = self.cmd(&format!("Ps {name}"))?;
        let trimmed = response.trim();
        if trimmed.to_ascii_lowercase().contains("error") {
            return Err(Error::r2pipe(format!("r2 refused 'Ps {name}': {trimmed}")));
        }
        info!(target: "r2smt::r2pipe", project = %name, "saved r2 project");
        Ok(())
    }

    /// Attach SLEIGH P-code to every instruction of `program` via
    /// r2ghidra's `pdgsd` (one call per function). Best-effort: a
    /// function whose `pdgsd` fails or is empty (no r2ghidra plugin,
    /// data-as-code, …) simply keeps `pcode = None` for its
    /// instructions, so the lifter falls back to ESIL. Never hard-
    /// fails the run — P-code is an optional, opt-in IR source.
    pub fn attach_pcode(&mut self, program: &mut Program) {
        for func in &mut program.functions {
            let count: usize = func.blocks.iter().map(|b| b.instructions.len()).sum();
            if count == 0 {
                continue;
            }
            let cmd = format!("pdgsd {count} @ {addr}", addr = func.address.get());
            let Ok(dump) = self.cmd(&cmd) else {
                continue;
            };
            let groups = parse::split_pdgsd_by_instruction(&dump);
            if groups.is_empty() {
                continue;
            }
            let by_addr: std::collections::BTreeMap<u64, String> = groups.into_iter().collect();
            for block in &mut func.blocks {
                for insn in &mut block.instructions {
                    if let Some(text) = by_addr.get(&insn.address.get()) {
                        insn.pcode = Some(text.clone());
                    }
                }
            }
        }
    }
}

impl Drop for R2PipeProvider {
    fn drop(&mut self) {
        self.r2.close();
    }
}

/// Maximum number of instructions the shellcode block finder will
/// walk backwards from the target address before bailing out. Matches
/// `MicroSMT`'s heuristic (200) so the surface stays comparable.
const SHELLCODE_BACKWARDS_LIMIT: usize = 200;

/// Maximum number of instructions the forward walk will scan from the
/// chosen block start. A normal basic block fits well under this; the
/// limit only matters when r2's instruction stream is degenerate.
const SHELLCODE_FORWARDS_LIMIT: usize = 256;

impl R2PipeProvider {
    /// Run `af @ addr` to force radare2 to analyse a function at that
    /// address. Returns `true` if r2 acknowledged the request without
    /// reporting an error.
    fn force_analyse_at(&mut self, address: Address) -> Result<bool> {
        let response = self.cmd(&format!("af @ {addr}", addr = address.get()))?;
        let trimmed = response.trim();
        if trimmed.to_ascii_lowercase().contains("error") {
            return Ok(false);
        }
        Ok(true)
    }

    /// Find the entry address of the function r2 has analysed that
    /// contains `address`, if any. Uses the current `aflj` listing
    /// and a single `agfj <func>` probe.
    fn function_containing(&mut self, address: Address) -> Result<Option<Address>> {
        let listing = self.cmd("aflj")?;
        let Ok(func_refs) = parse::parse_function_list(&listing) else {
            return Ok(None);
        };
        for func_ref in func_refs {
            let func_json = self.cmd(&format!("agfj {addr}", addr = func_ref.address.get()))?;
            let Ok(func) = parse::parse_function_blocks(&func_json) else {
                continue;
            };
            for block in &func.blocks {
                let block_start = block.address.get();
                let block_end = block
                    .instructions
                    .last()
                    .map_or(block_start, |i| i.address.get() + u64::from(i.size));
                if block_start <= address.get() && address.get() < block_end {
                    return Ok(Some(func.address));
                }
            }
        }
        Ok(None)
    }

    /// Heuristic block finder for shellcode / unanalysed regions. Walks
    /// backwards until reaching an xref target or a terminator
    /// instruction, then walks forwards until the next terminator,
    /// returning a synthetic [`Function`] with one [`BasicBlock`].
    fn synthesise_block_at(&mut self, address: Address) -> Result<Function> {
        let block_start = self.walk_back_to_block_start(address)?;
        let (instructions, _last_end) = self.walk_forward_until_terminator(block_start)?;
        if instructions.is_empty() {
            return Err(Error::r2pipe(format!(
                "shellcode finder collected no instructions at {address}",
            )));
        }
        Ok(Function {
            address: block_start,
            name: Some(format!("<shellcode @ {block_start}>")),
            blocks: vec![BasicBlock {
                address: block_start,
                instructions,
                successors: Vec::new(),
            }],
            is_thumb: false,
        })
    }

    fn walk_back_to_block_start(&mut self, address: Address) -> Result<Address> {
        let mut current = address;
        for _ in 0..SHELLCODE_BACKWARDS_LIMIT {
            // A branch target starts a new block — stop here.
            let xref_json = self.cmd(&format!("axtj @ {addr}", addr = current.get()))?;
            let xrefs = parse::parse_xrefs(&xref_json).unwrap_or_default();
            if !xrefs.is_empty() && current != address {
                return Ok(current);
            }
            // Look at the previous instruction. If it ends a block,
            // `current` is the start of the next block.
            let prev_json = self.cmd(&format!("pdj -1 @ {addr}", addr = current.get()))?;
            let Ok(prev) = parse::parse_aoj(&prev_json) else {
                return Ok(current);
            };
            let Some(prev_insn) = prev.into_iter().next() else {
                return Ok(current);
            };
            if prev_insn.address.get() + u64::from(prev_insn.size) != current.get() {
                // Non-contiguous (data hole) — stop.
                return Ok(current);
            }
            if prev_insn.flow.ends_block() {
                return Ok(current);
            }
            current = prev_insn.address;
        }
        Ok(current)
    }

    fn walk_forward_until_terminator(
        &mut self,
        start: Address,
    ) -> Result<(Vec<Instruction>, Address)> {
        let mut instructions: Vec<Instruction> = Vec::new();
        let mut cursor = start;
        for _ in 0..SHELLCODE_FORWARDS_LIMIT {
            let json = self.cmd(&format!("aoj 1 @ {addr}", addr = cursor.get()))?;
            let Ok(mut decoded) = parse::parse_aoj(&json) else {
                break;
            };
            let Some(insn) = decoded.pop() else {
                break;
            };
            let size = insn.size;
            let flow = insn.flow;
            instructions.push(Instruction {
                address: insn.address,
                size,
                bytes: insn.bytes,
                mnemonic: insn.mnemonic,
                operands: insn.operands,
                esil: insn.esil,
                pcode: None,
                is_thumb: false,
            });
            cursor = Address(insn.address.get() + u64::from(size));
            if flow.ends_block() {
                break;
            }
        }
        Ok((instructions, cursor))
    }
}

impl BinaryProvider for R2PipeProvider {
    fn load_program(&mut self) -> Result<Program> {
        let info_json = self.cmd("ij")?;
        let info = parse::parse_info(&info_json)?;

        let entry_json = self.cmd("iej")?;
        let entry = parse::parse_entry(&entry_json).unwrap_or(None);

        let listing_json = self.cmd("aflj")?;
        let func_refs = parse::parse_function_list(&listing_json)?;
        // Executable virtual-address ranges, derived from r2's own
        // section permission model. Used to reject blocks r2's
        // analysis over-extended into data sections (string tables,
        // rodata) and decoded as garbage instructions. An empty set
        // means "no section view" (stripped binary) — in that case we
        // do NOT filter, since filtering against an empty range would
        // discard every block.
        let exec_ranges = match self.cmd("iSj") {
            Ok(isj) => parse::parse_executable_ranges(&isj).unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        info!(
            target: "r2smt::r2pipe",
            arch = ?info.arch,
            bits = info.bits,
            functions = func_refs.len(),
            exec_ranges = exec_ranges.len(),
            "loaded program"
        );

        let mut functions: Vec<Function> = Vec::with_capacity(func_refs.len());
        let mut skipped_no_cfg: usize = 0;
        let mut dropped_nonexec_blocks: usize = 0;
        let mut skipped_all_nonexec: usize = 0;
        for func_ref in func_refs {
            let cmd = format!("agfj {addr}", addr = func_ref.address.get());
            let block_json = self.cmd(&cmd)?;
            let Some(mut func) = parse::parse_function_blocks_opt(&block_json)? else {
                // r2 listed this address in `aflj` but `agfj` has no
                // control-flow graph for it (import thunk, PLT stub,
                // data misclassified as code, or an undecoded
                // placeholder). Skipping one such entry must not abort
                // the whole program load — a large stripped binary can
                // have thousands of these alongside real functions.
                skipped_no_cfg += 1;
                continue;
            };
            if !exec_ranges.is_empty() {
                dropped_nonexec_blocks += parse::retain_executable_blocks(&mut func, &exec_ranges);
                if func.blocks.is_empty() {
                    // Every block r2 attributed to this function lay in
                    // a non-executable section — the whole function is
                    // an analysis artefact (data decoded as code), not
                    // a real control-flow graph. Drop it the same way
                    // as a no-CFG entry.
                    skipped_all_nonexec += 1;
                    continue;
                }
            }
            if func.name.is_none() {
                func.name = Some(func_ref.name);
            }
            functions.push(func);
        }
        if skipped_no_cfg > 0 || skipped_all_nonexec > 0 || dropped_nonexec_blocks > 0 {
            warn!(
                target: "r2smt::r2pipe",
                skipped_no_cfg,
                skipped_all_nonexec,
                dropped_nonexec_blocks,
                loaded = functions.len(),
                "filtered analysis artefacts during program load"
            );
        }

        let mut program = Program {
            arch: info.arch,
            bits: info.bits,
            entry,
            functions,
        };
        if self.attach_pcode_on_load {
            self.attach_pcode(&mut program);
        }
        Ok(program)
    }

    fn load_function(&mut self, address: Address) -> Result<Function> {
        let cmd = format!("agfj {addr}", addr = address.get());
        let json = self.cmd(&cmd)?;
        parse::parse_function_blocks(&json)
    }

    fn load_block_at(&mut self, address: Address) -> Result<Function> {
        // Step 1: r2 has already analysed a function containing addr.
        if let Some(func_addr) = self.function_containing(address)? {
            return self.load_function(func_addr);
        }
        // Step 2: force-analyse a function at addr, then retry.
        if self.force_analyse_at(address)?
            && let Some(func_addr) = self.function_containing(address)?
        {
            return self.load_function(func_addr);
        }
        // Step 3: shellcode heuristic.
        self.synthesise_block_at(address)
    }

    fn name_hints(&mut self, function: Address) -> Result<NameHints> {
        let mut hints = NameHints::default();
        let json = self.cmd(&format!("afvj @ {addr}", addr = function.get()))?;
        let locals = parse::parse_locals(&json).unwrap_or_default();
        for local in locals.stack_slots {
            hints.add_stack_slot(local.stack_slot, local.name);
        }
        for rename in locals.registers {
            // r2 reports the parent register name in practice
            // (`"rdi"`, `"x0"`, …), which matches the lifter's
            // canonical SSA variable name. If r2 ever emits a
            // sub-register alias (`"edi"`) the lookup in the pretty-
            // printer simply misses and the canonical name falls
            // through — sound, just no humanisation.
            hints.add_register(rename.register, rename.name);
        }
        Ok(hints)
    }
}

impl Annotator for R2PipeProvider {
    fn set_comment(&mut self, address: Address, comment: &str) -> Result<()> {
        let encoded = b64::encode(comment.as_bytes());
        let cmd = format!("CCu base64:{encoded} @ {addr}", addr = address.get());
        let response = self.cmd(&cmd)?;
        let trimmed = response.trim();
        if !trimmed.is_empty() && trimmed.to_ascii_lowercase().contains("error") {
            return Err(Error::r2pipe(format!(
                "r2 refused 'CCu' at {address}: {trimmed}"
            )));
        }
        Ok(())
    }
}

impl Decompiler for R2PipeProvider {
    /// Best-effort pseudocode fetch: r2ghidra `pdgj` (JSON) →
    /// r2dec `pddj` (JSON) → r2ghidra `pdg` (plain). Each rung is
    /// tried only if the previous yielded nothing. A missing
    /// decompiler plugin (or a per-command transport hiccup on the
    /// optional context path) degrades to `Ok(None)` — it must never
    /// abort the analysis run, per the [`Decompiler`] contract.
    fn pseudocode(&mut self, function: Address) -> Result<Option<String>> {
        let at = function.get();
        if let Ok(resp) = self.cmd(&format!("pdgj @ {at}"))
            && let Some(code) = parse::parse_pdgj(&resp)
        {
            return Ok(Some(code));
        }
        if let Ok(resp) = self.cmd(&format!("pddj @ {at}"))
            && let Some(code) = parse::parse_pddj(&resp)
        {
            return Ok(Some(code));
        }
        if let Ok(resp) = self.cmd(&format!("pdg @ {at}"))
            && let Some(code) = parse::clean_plain_decompile(&resp)
        {
            return Ok(Some(code));
        }
        Ok(None)
    }
}

impl BytePatcher for R2PipeProvider {
    fn read_bytes(&mut self, address: Address, size: usize) -> Result<Vec<u8>> {
        if size == 0 {
            return Ok(Vec::new());
        }
        // `p8 N @ addr` prints N bytes as a contiguous hex string.
        let cmd = format!("p8 {size} @ {addr}", addr = address.get());
        let response = self.cmd(&cmd)?;
        let hex_str: String = response.chars().filter(|c| !c.is_whitespace()).collect();
        if hex_str.len() != size * 2 {
            return Err(Error::r2pipe(format!(
                "p8 returned {got} hex chars at {address}, expected {want}",
                got = hex_str.len(),
                want = size * 2,
            )));
        }
        decode_hex(&hex_str)
    }

    fn write_bytes(&mut self, address: Address, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let hex_str = encode_hex(bytes);
        let cmd = format!("wx {hex_str} @ {addr}", addr = address.get());
        let response = self.cmd(&cmd)?;
        let trimmed = response.trim();
        if !trimmed.is_empty() && trimmed.to_ascii_lowercase().contains("error") {
            return Err(Error::r2pipe(format!(
                "r2 refused 'wx' at {address}: {trimmed}"
            )));
        }
        Ok(())
    }

    fn assemble(&mut self, address: Address, asm: &str) -> Result<Vec<u8>> {
        if asm.chars().any(|c| c == '\n' || c == '\r' || c == ';') {
            return Err(Error::r2pipe(
                "assembly text must not contain newlines or ';'",
            ));
        }
        // `pa <asm> @ <addr>` assembles without writing, returning the
        // hex encoding on stdout.
        let cmd = format!("pa {asm} @ {addr}", addr = address.get());
        let response = self.cmd(&cmd)?;
        let hex_str: String = response.chars().filter(|c| !c.is_whitespace()).collect();
        if hex_str.is_empty() {
            return Err(Error::r2pipe(format!(
                "r2 'pa' returned no bytes for '{asm}' at {address}"
            )));
        }
        decode_hex(&hex_str)
    }
}

fn decode_hex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(Error::r2pipe(format!("odd-length hex from r2: '{s}'")));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let high = hex_nibble(bytes[i])?;
        let low = hex_nibble(bytes[i + 1])?;
        out.push((high << 4) | low);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(Error::r2pipe(format!(
            "non-hex char '{ch}' in r2 output",
            ch = char::from(c)
        ))),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(hex_char(byte >> 4));
        out.push(hex_char(byte & 0x0F));
    }
    out
}

fn hex_char(nibble: u8) -> char {
    match nibble {
        0..=9 => char::from(b'0' + nibble),
        10..=15 => char::from(b'a' + nibble - 10),
        _ => '?',
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::Error;

    use super::{decode_hex, encode_hex, is_transient_spawn_error};

    #[test]
    fn encode_hex_lowercase_pairs() {
        assert_eq!(encode_hex(&[0x00, 0x0f, 0x90, 0xff]), "000f90ff");
    }

    #[test]
    fn transient_spawn_errors_are_retried() {
        // The two failure modes observed under concurrent spawn storms.
        assert!(is_transient_spawn_error(&Error::r2pipe("I/O error")));
        assert!(is_transient_spawn_error(&Error::r2pipe(
            "Empty response from JSON"
        )));
        // Case-insensitive and substring-anchored.
        assert!(is_transient_spawn_error(&Error::r2pipe(
            "pipe: Resource temporarily unavailable (os error 35)"
        )));
    }

    #[test]
    fn permanent_spawn_errors_fail_fast() {
        // A missing r2 binary or unreadable target is stable across
        // attempts — retrying only wastes time and masks the cause.
        assert!(!is_transient_spawn_error(&Error::r2pipe(
            "No such file or directory (os error 2)"
        )));
        assert!(!is_transient_spawn_error(&Error::r2pipe(
            "non-UTF-8 binary path"
        )));
        // Non-r2pipe error variants are never treated as transient.
        assert!(!is_transient_spawn_error(&Error::parse("x", "y")));
    }

    #[test]
    fn decode_hex_accepts_both_cases() {
        assert_eq!(decode_hex("AB").unwrap(), vec![0xab]);
        assert_eq!(decode_hex("ab").unwrap(), vec![0xab]);
    }

    #[test]
    fn decode_hex_rejects_odd_length() {
        assert!(decode_hex("abc").is_err());
    }

    #[test]
    fn decode_hex_rejects_non_hex() {
        assert!(decode_hex("zz").is_err());
    }

    #[test]
    fn hex_round_trip() {
        let original: Vec<u8> = (0u8..=255).collect();
        let encoded = encode_hex(&original);
        let decoded = decode_hex(&encoded).unwrap();
        assert_eq!(decoded, original);
    }
}
