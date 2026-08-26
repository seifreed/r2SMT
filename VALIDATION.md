# Independent Validation

r2SMT is not a `1.0` claim until analysts who did not build the project have
used it on real binaries and reported the results. This page is the smallest
repeatable protocol for doing that without sharing samples.

## Reproduce the public corpus

Requirements: Rust `1.95+`, radare2 `6.2.0`, a C compiler, `jq`, and Python 3.

```bash
git clone https://github.com/seifreed/r2SMT.git
cd r2SMT
scripts/fetch-bench-corpus.sh target/r2smt-corpus
cargo run --quiet -p r2smt-bench -- validate target/r2smt-corpus
scripts/quality-gates.sh real-binaries
```

The pinned corpus publishes its current aggregate at
[`r2smt-corpus/metrics-baseline.json`](https://github.com/seifreed/r2smt-corpus/blob/main/metrics-baseline.json).
The gate writes the same schema to `target/r2smt-bench-metrics.json` and keeps
per-fixture reports under `target/r2smt-bench-metrics/`.

The published x86_64 reference (r2SMT commit
`fda0aee6e1bc4a41656ba4ba0cd933a41e37933c`, corpus source commit
`5fafa2f4c4db3a5147d789b8fbbc9be12204b341`, radare2 `6.2.0`) reports:

| Metric | Value |
|---|---:|
| Covered branches / expected | 15 / 15 (100%) |
| Actionable precision | 100% |
| False actionable findings | 0 |
| Expected actionables not observed | 2 |
| Unknown findings | 16 |
| Runtime p50 / p95 | 107 / 108 ms |
| Verified patch rate | 100% |
| Rollback success rate | 100% |

This is a benchmark reference, not a claim about arbitrary malware or
unsupported instructions. The exact CI provenance is recorded in the JSON
(`ci_run: 32913427778`).

## Report a real-binary run

Do not upload malware or proprietary binaries. Attach a redacted JSON summary
to an issue or discussion with:

```json
{
  "tool_commit": "<git SHA>",
  "radare2_version": "<r2 -v>",
  "platform": "<OS/arch>",
  "binary_count": 0,
  "sha256_only": true,
  "definitive_findings": 0,
  "unknown_by_reason": {},
  "actionable_findings": 0,
  "false_actionable_findings": 0,
  "elapsed_ms_p50": 0,
  "elapsed_ms_p95": 0,
  "notes": "<limits, unsupported instructions, or reproducibility details>"
}
```

The maintainer will record independent runs here only when the report includes
the tool commit, radare2 version, platform, and enough aggregate data to audit
the claim. A passing corpus gate is necessary evidence, but it is not a
substitute for sustained use on unrelated binaries.
