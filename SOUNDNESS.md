# Soundness contract

In r2SMT, "sound" means that an actionable `AlwaysTrue` or `AlwaysFalse`
verdict is valid for every assignment admitted by the symbolic model that was
actually sent to the authoritative solver. It does not mean that r2SMT models
every instruction or that radare2's recovered program is infallible.

## Assumptions

The guarantee is conditional on all of the following:

1. radare2 decoded the instruction bytes, architecture mode, branch target,
   and control-flow edges correctly.
2. The selected ESIL, P-code, or mnemonic lowering implements the relevant ISA
   semantics correctly.
3. The SSA and SMT encoders preserve those lifted semantics.
4. The authoritative solver implements its advertised theory correctly.
5. Self-modifying code, concurrent mutation, external device state, and
   behavior outside the extracted slice do not invalidate the model.

A finding is therefore a proof about the normalized model and these
assumptions, not a universal proof about every possible execution environment.

## Fail-closed behavior

- Unsupported or inconsistent semantics become free symbolic inputs or an
  `Unsound`/`Unknown` result; they are not replaced by convenient constants.
- Free inputs can turn a fixed condition into `BothPossible`, but cannot create
  a fixed condition that did not hold for all input values.
- Truncated slices are non-actionable by default. Explicit recovery modes keep
  unresolved roots symbolic and lower confidence.
- A lifter disagreement is reported as an engine-integrity finding, not as an
  actionable branch verdict.
- CVC5 and Bitwuzla may act as authoritative backends when selected. The
  optional witness oracle is corroboration-only and cannot change a sound
  verdict or raise confidence.

## Confidence and actions

| Confidence | Meaning | Default action policy |
|---|---|---|
| `high` | Definitive solver result over a complete, fully modelled slice | Eligible for annotate and patch |
| `medium` | Definitive result with free symbolic inputs | Reportable; annotation requires an explicit lower threshold |
| `low` | Evidence was downgraded by a semantic or oracle guard | Informational unless explicitly requested |
| `unknown` | Truncated, timed out, unsupported, or unsound | Never actionable |

Patching defaults to high confidence and explicit `--apply`. A successful SMT
proof does not by itself prove that a byte patch preserves file-format,
relocation, decoding, or whole-program behavior; the patch transaction and
post-patch verification are separate obligations.

## Non-goals

r2SMT does not claim complete branch discovery, zero false negatives, general
program equivalence, or safety against hostile resource consumption in the
current in-process analysis path. Measured claims must name the corpus,
radare2 version, solver settings, and relevant feature gates.
