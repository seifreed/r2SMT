# Bug #3 — basic blocks decoded from non-executable sections

## Symptom (evidence)

Sample `1f04eb8aa288…` (ELF .so, x86_64, 16 MB, 9375 functions).

- `r2smt solve` reports 22 349 findings; **838** have `branch_address <
  function_start` (structurally impossible) and **437** sit outside
  `.text`.
- Verified case: finding `addr=0x56018 fn=0x140710`.
  - `0x140710` = `sym.cards::game::bridge::BridgeBid::toString() const`
    — real `.text`.
  - `0x56018` is inside **`.dynstr`** (dynamic string table — pure
    data), perm `-r--`.
  - `agfj 0x140710` from r2 itself returns **215 ops, 188 outside
    `.text`** decoded as garbage (`outsb dx, byte gs:[rsi]`,
    `xor al, 0x5f`, `insb byte [rdi], dx`).

## Root cause

radare2's `aaa` over-extends the function CFG for this binary into the
string table and `agfj` faithfully reports those data bytes as
instructions. r2smt parses the response correctly but **trusts r2's
CFG blindly** — it never checks that an instruction address lies in an
executable mapping.

This is not r2smt parser corruption and not sample-specific: any
binary where r2 mis-analyses data as code produces the same shape.

## Fix (generic, format-grounded invariant)

Per CLAUDE.md *Parser parity*: reject the malformed structure with a
generic reason rather than silently consuming it. The universal
invariant is **"a basic block's instructions must lie within an
executable section"** — true for ELF / PE / Mach-O alike, expressed
through r2's own section/permission model (`iSj`), no hardcoded
section names or sample values.

### Layering

- `r2smt-r2pipe::parse` (pure): add
  `parse_executable_ranges(isj_json) -> Result<Vec<(u64,u64)>>` and
  `address_in_ranges(&[(u64,u64)], u64) -> bool`. Pure, fixture-tested,
  no radare2 needed.
- `r2smt-r2pipe::provider::load_program` (adapter): query `iSj`, build
  the executable ranges, then for each parsed function retain only
  blocks whose start address is inside an executable range; if a
  function has no executable block left, skip it (reuse the existing
  no-CFG skip counter + `warn`).

Domain crates stay format-agnostic — section knowledge lives in the
adapter, consistent with Clean Architecture boundaries.

### Graceful degradation

If `iSj` yields **zero** executable ranges (fully stripped binary, no
section view), do **not** filter — that would discard every block.
Preserve prior behaviour in that case and log it.

### Granularity

Block-start filtering is sufficient: r2's mis-analysis creates whole
bogus blocks whose `addr` is in the data section; r2 never splits a
single block across a section boundary. Per-instruction filtering
would risk fragmenting real blocks for no gain.

## Tests

- `parse_executable_ranges_extracts_x_perm_sections`
- `parse_executable_ranges_ignores_non_exec`
- `parse_executable_ranges_empty_when_no_sections`
- `address_in_ranges_boundaries` (inclusive start, exclusive end)
- provider-level: a `Function` with one `.text` block and one
  data-section block keeps only the `.text` block; a function whose
  every block is in data is dropped and counted.

## Verification

Re-run `r2smt solve` on `1f04eb8aa288…` and assert:
`addr < function_start` count → 0, all finding addresses inside an
executable range, actionable count reflects only real `.text`
branches. Re-run the smaller verified sample `fb69475443bc…` to prove
the CPUID finding is unaffected (no regression).
