#!/usr/bin/env python3
"""Diff two `r2smt batch --json` runs into a finding-kind histogram.

Usage:

    r2smt batch <dir> --threads 8 --rlimit 2000000 --json before.json
    # ... apply the change, rebuild ...
    r2smt batch <dir> --threads 8 --rlimit 2000000 --json after.json
    python3 scripts/verdict-diff.py before.json after.json

Two rules make the measurement mean anything, both learned the hard way:

* **Diff with `--rlimit`, never with `--timeout-ms`.** A wall-clock budget
  makes the verdict depend on host load, so the same binary over the same
  sample classifies a branch `real_branch` on an idle machine and
  `suspicious_but_unknown` on a busy one. `--rlimit` counts solver work
  instead and is reproducible. It reduces the noise rather than
  eliminating it: a branch sitting exactly on the resource boundary has
  been observed moving between two runs of an unchanged binary.

  Read that rule precisely, because the obvious reading is wrong:
  `--rlimit` does **not** replace the wall clock. `batch` applies both
  budgets, and omitting `--timeout-ms` leaves it at its 500 ms default
  rather than disabling it, so whichever fires first decides. What makes
  the rule work is that at `--rlimit 2000000` the solver finishes far
  inside 500 ms — measured byte-identical histograms between a 500 ms and
  a 60 s wall clock over 7 151 ARM32 branches and 27 419 x86-64 branches,
  the second at ~1.5 ms of CPU per branch. So the wall clock is a safety
  net there, not the binding constraint.

  Two things follow. Raising `--rlimit` much past 2 000 000 needs a
  matching `--timeout-ms` or the clock silently becomes the budget again.
  And a batch run that takes minutes per binary is not evidence that the
  solver is the cost — radare2's own analysis dominates on samples with
  tens of thousands of functions, and no solver budget shortens it.

* **Filter by ISA.** `data/` is 75 x86-64, 16 AArch64 and 2 CIL samples
  and is near-blind to 32-bit ARM, so running all 102 for an ARM change
  costs an hour and proves nothing. Point this at the corpus whose ISA
  the change touches.

State the direction the diff is expected to move before running it. A
widening should show findings *arriving* in the resolved kinds and
leaving `suspicious_but_unknown`; a fix that removes fabricated verdicts
should show the reverse. The actionable kinds (`opaque_predicate`,
`dead_branch`, `constant_condition`) should be unchanged unless the
change is specifically about them — and any movement there gets checked
by hand rather than accepted from the histogram.

Stdlib only, by design: this has to run against any checkout without a
virtualenv.
"""

import collections
import json
import sys

NAME_WIDTH = 12


def load(path):
    """Map each sample's short name to (arch, branches analysed, histogram)."""
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    out = {}
    for sample in report.get("samples", []):
        name = sample["path"].split("/")[-1][:NAME_WIDTH]
        summary = (sample.get("outcome", {}) or {}).get("summary")
        if not summary:
            out[name] = ("?", 0, {})
            continue
        out[name] = (
            summary.get("arch", "?"),
            summary.get("branches_analyzed", 0),
            dict(summary.get("summary", {})),
        )
    return out


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} before.json after.json", file=sys.stderr)
        return 2

    before, after = load(argv[1]), load(argv[2])
    totals_before, totals_after = collections.Counter(), collections.Counter()
    moved = []

    for name in sorted(set(before) | set(after)):
        arch_b, count_b, hist_b = before.get(name, ("?", 0, {}))
        arch_a, count_a, hist_a = after.get(name, ("?", 0, {}))
        totals_before.update(hist_b)
        totals_after.update(hist_a)
        if hist_b != hist_a or count_b != count_a:
            moved.append((name, arch_a or arch_b, count_b, count_a, hist_b, hist_a))

    print("=== per-sample movement ===")
    if not moved:
        print("(none)")
    for name, arch, count_b, count_a, hist_b, hist_a in moved:
        print(f"{name} [{arch}] branches {count_b} -> {count_a}")
        for kind in sorted(set(hist_b) | set(hist_a)):
            if hist_b.get(kind, 0) != hist_a.get(kind, 0):
                print(f"    {kind}: {hist_b.get(kind, 0)} -> {hist_a.get(kind, 0)}")

    print()
    print("=== corpus totals ===")
    for kind in sorted(set(totals_before) | set(totals_after)):
        old, new = totals_before.get(kind, 0), totals_after.get(kind, 0)
        delta = "" if new == old else f"   ({new - old:+d})"
        print(f"{kind:26} {old:6} -> {new:6}{delta}")
    print(f"{'TOTAL':26} {sum(totals_before.values()):6} -> {sum(totals_after.values()):6}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
