# Phase 4 spike — Ghidra P-code as optional 2nd IR

**Date:** 2026-05-16
**Original verdict:** STOP gate triggered → "document the limit".

## RESOLUTION (2026-05-17) — OVERTURNED, P-code shipped

The spike's blockers were misdiagnosed and were fixable:

- The "ABI-broken / no fixture" finding was caused by a **stale
  `libcore_pdd.dylib` (built for r2 6.0.9, ABI 66)** shadowing the
  correctly-built `core_ghidra.dylib` (6.1.4). Removing the stale
  plugin made `pdgsd` emit clean P-code; a real fixture was captured
  from a controlled compiled binary.
- "No structured schema": `pdgsd` text is in fact a **stable,
  documented SLEIGH grammar** (`(space,off,size)` varnodes, ~30
  regular opcodes, explicit flag derivation) — sound to parse as a
  contract, not sample-specific scraping.

Delivered: `crates/r2smt-pcode` (pure parser + strict-subset
lifter), `Instruction.pcode` + provider `attach_pcode` (one `pdgsd`
call/function) + pure `split_pdgsd_by_instruction`, P-code-first
tier in `lift.rs` (ESIL fallback), global `--ir {esil|pcode|auto}`
(default `esil`, byte-identical). Sound: declines (→ ESIL fallback)
on any opcode/flag outside the proven subset; only the polarity-free
Z flag maps to the canonical model. All gates green.

---

(Historical spike record below.)

## Question

Can structured Ghidra P-code be extracted via r2pipe in a sound,
parseable form, with a fixture capturable for fixture-discipline
tests, so it can feed the existing slicer → SSA → SMT pipeline as an
optional `--ir pcode` source?

## Evidence (this host, radare2 6.1.4 / darwin-arm64)

- `r2pm -l` → `r2ghidra`, `r2ghidra-sleigh`, `r2dec`, `decai`
  installed.
- `pdg?` lists: `pdg` / `pdgj` (decompiled **C** as JSON) /
  `pdgx` / `pdgd` (XML of the decompiled function) /
  `pdgsd N` ("Disassemble N instructions with Sleigh and print
  pcode").
- **No JSON form of raw P-code.** `pdgsd` is the only raw-P-code
  command and emits ad-hoc TEXT. There is no `pdgsdj`. The
  structured JSON commands (`pdgj`) return decompiled C, not the
  P-code op stream.
- **r2ghidra non-functional here:** `WARN: ABI mismatch: Expect 83
  vs 66 from libcore_pdd.dylib`. Every `pdgsd`/`pdg` call on
  `/bin/ls` returned `ERROR: 0x..: invalid` / empty. No P-code
  fixture could be captured.

## Why this fails the gate (independent reasons)

1. **Schema:** raw P-code is text-only (`pdgsd`); no stable JSON
   contract. A parser would be coupled to Ghidra's textual P-code
   rendering, which changes across Ghidra versions — fails
   "sound, parseable form" and the Sample-Agnostic / parser-parity
   policy.
2. **Fixture:** the plan forbids code before a fixture is captured;
   the ABI-broken backend makes capture impossible here.
3. **Dependency fragility:** r2ghidra must be ABI-matched to the
   user's radare2. A routine r2 upgrade silently breaks the path —
   unacceptable foundation for a *core IR* of a general-purpose
   tool. (Rebuilding r2ghidra would not change verdict #1.)

## Decision

Do **not** add `crates/r2smt-pcode` or `--ir {auto,pcode,esil}`.
Building schema-less `pdgsd` text scraping would violate Fixture
Discipline, parser-parity, and the spike gate the user approved.

ESIL remains the IR; `r2smt-ssa::optimize_slice` is the
decompiler-grade-cleanup substitute (const-fold, copy/const-prop,
dead-flag elim). Phase 3 already delivers Ghidra/r2dec *pseudocode
context* (analyst-facing, never feeds verdicts) via the `Decompiler`
port — the safe, structured slice of the original ambition.

## Revisit criteria

Reopen only if r2ghidra (or another backend) exposes raw P-code as
a **versioned JSON schema** over r2pipe, decoupled from r2's C ABI.
Track upstream r2ghidra for a `pdgsdj`-style structured command.
