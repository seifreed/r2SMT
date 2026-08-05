#!/usr/bin/env python3
"""Count how often mnemonics land in a basic block that ends in a
conditional branch — the only placement that can move a verdict.

Usage:

    python3 scripts/block-prevalence.py palignr,pshufb -- sample1 sample2 ...

Why this and not `objdump | grep -c`: a raw disassembly count is off by
more than an order of magnitude, in both directions, and picking work
from it wastes the work.

* It counts bytes radare2 never analyses as a function. Measured on this
  corpus, `palignr` was 284 by objdump and 9 by real CFG; `pblendvb` was
  502 by objdump and **0**. The 502 were in code no function reaches.
* It counts placements that cannot matter. A packed instruction whose
  block ends in an unconditional jump or a return never feeds a branch
  predicate, so lifting it buys slice coverage and no verdict.

Requires radare2 on PATH. `aaa` over a large sample is slow — budget
minutes per binary, and prefer a filtered candidate list over the whole
corpus.
"""

import collections
import json
import subprocess
import sys

FUNCTION_LIMIT = 400
ANALYSIS_TIMEOUT_S = 900
BLOCK_DUMP_TIMEOUT_S = 1200


def function_addresses(path):
    """Every analysed function entry, capped so a huge binary terminates."""
    try:
        out = subprocess.run(
            ["r2", "-q", "-c", "aaa;aflqj", "-N", path],
            capture_output=True,
            text=True,
            timeout=ANALYSIS_TIMEOUT_S,
        ).stdout
        functions = json.loads(out or "[]")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return []
    addresses = [f if isinstance(f, int) else f.get("offset") for f in functions]
    return [a for a in addresses if a][:FUNCTION_LIMIT]


def blocks(path, addresses):
    """Yield each basic block as its list of mnemonics, in order."""
    script = ";".join(f"agfj @ {a}" for a in addresses)
    try:
        out = subprocess.run(
            ["r2", "-q", "-c", f"aaa;{script}", "-N", path],
            capture_output=True,
            text=True,
            timeout=BLOCK_DUMP_TIMEOUT_S,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return
    # r2 emits one JSON document per `agfj`, concatenated without a
    # separator, so the stream is decoded incrementally.
    decoder, index = json.JSONDecoder(), 0
    while index < len(out):
        while index < len(out) and out[index] not in "[{":
            index += 1
        if index >= len(out):
            return
        try:
            document, index = decoder.raw_decode(out, index)
        except json.JSONDecodeError:
            index += 1
            continue
        for function in document if isinstance(document, list) else [document]:
            for block in function.get("blocks", []):
                mnemonics = [
                    (op.get("disasm") or op.get("opcode") or "").split()[0]
                    for op in block.get("ops", [])
                    if (op.get("disasm") or op.get("opcode"))
                ]
                if mnemonics:
                    yield mnemonics


def main(argv):
    if "--" not in argv:
        print(f"usage: {argv[0]} mnem1,mnem2 -- sample ...", file=sys.stderr)
        return 2
    split = argv.index("--")
    families = argv[1].split(",")
    samples = argv[split + 1 :]

    total, conditional = collections.Counter(), collections.Counter()
    analysed = 0
    for path in samples:
        addresses = function_addresses(path)
        if not addresses:
            continue
        for mnemonics in blocks(path, addresses):
            ends_conditional = mnemonics[-1].startswith("j") and mnemonics[-1] != "jmp"
            for family in families:
                count = mnemonics.count(family)
                if not count:
                    continue
                total[family] += count
                if ends_conditional:
                    conditional[family] += count
        analysed += 1

    print(f"samples analysed: {analysed}")
    print(f"{'mnemonic':14} {'total':>7} {'in a block ending in jcc':>26}")
    for family in families:
        print(f"{family:14} {total[family]:7} {conditional[family]:26}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
