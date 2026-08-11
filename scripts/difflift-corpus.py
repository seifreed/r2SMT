#!/usr/bin/env python3
"""Run the differential-lift harness over a corpus and aggregate it.

Usage:

    cargo build --release --bin r2smt
    python3 scripts/difflift-corpus.py ../aarch64-neon-samples
    python3 scripts/difflift-corpus.py data --per-sample-timeout 900

The harness cross-checks each instruction's independent lowerings
(P-code, ESIL, per-mnemonic) against each other and reports a proven
disagreement as a `lifter_disagreement` finding. It lives only on
`solve`, so a corpus pass is a loop over samples -- which is what this
script is.

Four things about reading the output, each learned the hard way:

* **`disagree` is the only actionable number.** A lowering that *fails*
  adds to none of the four tallies, so the agreement percentage moves
  when coverage changes and says nothing about correctness. Read the
  disagreement histogram, not the rate.

* **A histogram by mnemonic groups symptoms, not causes.** Of 114
  disagreements in one earlier cycle, 112 were a single bug of ours.
  Expect one root cause per cluster, not per row.

* **Not every disagreement is ours, and the attributions expire.**
  radare2's own ESIL carries defects -- it seeds the flag context from
  the *destination* write, so `subs`/`rsbs` with `dst != src` compare
  against the wrong snapshot. Those are attributed in CLAUDE.md and are
  why ARM cannot reach a literal zero; the bar that is reachable is zero
  of ours.

  But record `radare2 -v` alongside any histogram you keep. The jump
  from 6.1.8 to 6.2.0 **fixed** ARM32 `movt`, which this project had
  written down as a standing radare2 defect, and one sample's count fell
  from 2 to 1 with no change on our side. A count that drops is not a
  fix until you have looked at *which* entry left.

* **Match the timeout to the work, which is the harness and not r2.**
  radare2's analysis of the largest sample in these corpora takes ~10 s;
  the harness costs ~3.9 ms per instruction, so a 100 000-instruction
  binary needs ~6 minutes. Sizing the timeout off the file size or the
  function count gets this wrong in both directions.

* **Pin the budget, and prefer `--rlimit`.** Pass solver flags through
  after `--`; they reach every sample identically. Note that the budget
  is not neutral even at a fixed `--rlimit`: Z3's per-check resource
  limit is not independent of what the shared context has already
  accumulated, so a longer run decides fewer checks. `disagree` has been
  measured stable across budgets where `agree`/`inconclusive` were not.

A sample that exceeds `--per-sample-timeout` is reported as **not
measured** rather than as zero -- a distinction that matters, because
`data/` contains a sample whose radare2 analysis never terminates (it
spawns an `xz -d` that hangs), and counting it as a clean zero would be
a fabricated result. That is also why this script needs no exclusion
list.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

# Mirrors `print_lifter_agreement` in crates/r2smt-cli/src/main.rs. The
# two are a contract: this regex failing to match reports the sample as
# having produced no tally, which is deliberately distinct from a zero
# and from a timeout, but is still a silent loss of the whole corpus if
# the line changes and this does not.
AGREEMENT = re.compile(
    r"lifter-agreement:\s+(?P<rate>[\d.]+%|n/a)\s+"
    r"\(agree=(?P<agree>\d+)\s+disagree=(?P<disagree>\d+)\s+"
    r"inconclusive=(?P<inconclusive>\d+)\)\s+over\s+(?P<compared>\d+)\s+comparisons,"
    r"\s+(?P<queries>\d+)\s+solver\s+queries"
    r"(?P<truncated>.*)"
)


def run_sample(binary, sample, timeout, extra):
    """Harness one sample.

    Returns a dict on success, or the string `"timeout"` / `"failed"`.
    Those two are kept apart deliberately: collapsing them is what let a
    passthrough-argument bug — every r2smt invocation dying on an
    unknown flag — read as a corpus too slow to measure.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = tmp.name
    try:
        proc = subprocess.run(
            [binary, "solve", sample, "--differential-lift", "--json", report_path]
            + extra,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        os.unlink(report_path)
        return "timeout"

    match = AGREEMENT.search(proc.stdout)
    if match is None:
        os.unlink(report_path)
        return "failed"

    findings = []
    try:
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
        findings = [
            f
            for f in report.get("findings", [])
            if f.get("kind") == "lifter_disagreement"
        ]
    except (OSError, json.JSONDecodeError):
        # The tally still stands; only the per-mnemonic breakdown is lost.
        pass
    finally:
        if os.path.exists(report_path):
            os.unlink(report_path)

    return {
        "agree": int(match.group("agree")),
        "disagree": int(match.group("disagree")),
        "inconclusive": int(match.group("inconclusive")),
        "compared": int(match.group("compared")),
        "queries": int(match.group("queries")),
        "truncated": "TRUNCATED" in match.group("truncated"),
        "mnemonics": Counter(f.get("mnemonic", "?") for f in findings),
        "addresses": [(f.get("address"), f.get("mnemonic")) for f in findings],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the differential-lift harness over a corpus."
    )
    parser.add_argument("corpus", help="directory of samples")
    parser.add_argument(
        "--binary",
        default="./target/release/r2smt",
        help="path to the r2smt binary (default: ./target/release/r2smt)",
    )
    parser.add_argument(
        "--per-sample-timeout",
        type=int,
        default=600,
        help="seconds before a sample is reported as not measured (default: 600)",
    )
    # `parse_known_args`, not an `argparse.REMAINDER` positional: the
    # latter also swallows the options declared above, so this script's
    # own `--per-sample-timeout` reached r2smt, which rejected it — and
    # every sample then read as unmeasurable.
    args, passthrough = parser.parse_known_args()

    extra = [a for a in passthrough if a != "--"]
    samples = sorted(
        os.path.join(args.corpus, name)
        for name in os.listdir(args.corpus)
        if os.path.isfile(os.path.join(args.corpus, name))
    )
    if not samples:
        sys.exit(f"no samples in {args.corpus}")

    totals = Counter()
    histogram = Counter()
    timed_out = []
    failed = []
    truncated = []

    print(
        f"{'sample':<24} {'compared':>9} {'queries':>8} {'agree':>7} "
        f"{'disagree':>9} {'inconc':>8}"
    )
    for sample in samples:
        label = os.path.basename(sample)[:20]
        result = run_sample(args.binary, sample, args.per_sample_timeout, extra)
        if isinstance(result, str):
            (timed_out if result == "timeout" else failed).append(
                os.path.basename(sample)
            )
            print(f"{label:<24} {result.upper():>44}")
            continue
        if result["truncated"]:
            truncated.append(os.path.basename(sample))
        for key in ("agree", "disagree", "inconclusive", "compared", "queries"):
            totals[key] += result[key]
        histogram.update(result["mnemonics"])
        print(
            f"{label:<24} {result['compared']:>9} {result['queries']:>8} "
            f"{result['agree']:>7} {result['disagree']:>9} "
            f"{result['inconclusive']:>8}"
        )
        for address, mnemonic in result["addresses"]:
            print(f"{'':<24}   disagree @ {address} {mnemonic}")

    measured = len(samples) - len(timed_out) - len(failed)
    print()
    print(
        f"corpus: {measured}/{len(samples)} measured, "
        f"{totals['compared']} comparisons ({totals['queries']} solver queries), "
        f"agree={totals['agree']} disagree={totals['disagree']} "
        f"inconclusive={totals['inconclusive']}"
    )
    if timed_out:
        print(
            f"timed out past {args.per_sample_timeout}s ({len(timed_out)}): "
            f"{', '.join(timed_out)}"
        )
    if failed:
        # Never a slow corpus: r2smt ran and printed no tally. Re-run one
        # by hand before reading anything into the totals.
        print(f"r2smt reported no tally ({len(failed)}): {', '.join(failed)}")
    if measured == 0:
        sys.exit("no sample measured — treat this as a tooling failure, not a zero")
    if truncated:
        print(
            f"comparison budget exhausted on {len(truncated)} sample(s) — "
            f"coverage below what the totals suggest: {', '.join(truncated)}"
        )
    if histogram:
        print("\ndisagreements by mnemonic (symptoms, not causes):")
        for mnemonic, count in histogram.most_common():
            print(f"  {count:>5}  {mnemonic}")


if __name__ == "__main__":
    main()
