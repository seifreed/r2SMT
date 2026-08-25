#!/usr/bin/env bash

set -euo pipefail

out="${1:?usage: aggregate-bench-metrics.sh <output.json> <metrics.json>...}"
shift
if [[ $# -eq 0 ]]; then
  echo "no per-fixture metrics supplied" >&2
  exit 1
fi

jq -s '
  def sum_by($key): map(.[$key] // 0) | add;
  def percent($part; $total):
    if $total == 0 then null else ($part * 100.0 / $total) end;
  def percentile($p):
    sort as $values
    | if ($values | length) == 0 then null
      else $values[((((($values | length) - 1) * $p) | floor))]
      end;
  def merge_unknowns:
    reduce .[] as $metric ({};
      reduce (($metric.unknown_by_reason // {}) | to_entries[]) as $entry
        (.;
          .[$entry.key] = ((.[$entry.key] // 0) + $entry.value)));

  . as $metrics
  | (sum_by("true_positive_actionable_findings")) as $true_positive
  | (sum_by("false_actionable_findings")) as $false_positive
  | (sum_by("expected_branches")) as $expected
  | (sum_by("discovered_branches")) as $discovered
  | (sum_by("findings")) as $findings
  | (sum_by("complete_findings")) as $complete
  | (sum_by("definitive_findings")) as $definitive
  | (sum_by("false_negative_actionable_findings")) as $false_negative
  | (sum_by("verified_patches")) as $verified
  | (sum_by("patch_attempts")) as $patch_attempts
  | (sum_by("rollback_successes")) as $rollback_successes
  | (sum_by("rollback_attempts")) as $rollback_attempts
  | {
      schema_version: 1,
      fixtures: ($metrics | map(.fixture // "unknown")),
      expected_branches: $expected,
      discovered_branches: $discovered,
      branch_discovery_recall: percent($discovered; $expected),
      findings: $findings,
      complete_findings: $complete,
      complete_slice_percent: percent($complete; $findings),
      definitive_findings: $definitive,
      definitive_percent: percent($definitive; $findings),
      true_positive_actionable_findings: $true_positive,
      actionable_precision: percent($true_positive; ($true_positive + $false_positive)),
      false_actionable_findings: $false_positive,
      false_negative_actionable_findings: $false_negative,
      unknown_by_reason: ($metrics | merge_unknowns),
      lifter_disagreements: (sum_by("lifter_disagreements")),
      solver_disagreements: (sum_by("solver_disagreements")),
      elapsed_ms_p50: ($metrics | map(.elapsed_ms) | map(select(. != null)) | percentile(0.50)),
      elapsed_ms_p95: ($metrics | map(.elapsed_ms) | map(select(. != null)) | percentile(0.95)),
      verified_patches: $verified,
      patch_attempts: $patch_attempts,
      verified_patch_rate: percent($verified; $patch_attempts),
      rollback_successes: $rollback_successes,
      rollback_attempts: $rollback_attempts,
      rollback_success_rate: percent($rollback_successes; $rollback_attempts)
    }
' "$@" > "$out"

cat "$out"
