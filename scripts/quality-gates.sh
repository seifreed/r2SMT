#!/usr/bin/env bash
# r2SMT quality gates — single entrypoint for repo gates.
#
# Each phase adds its own subcommand. `check`, `solver-contracts` and
# `supply-chain` are implemented; placeholders below mark the seams
# where future gates will land.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: scripts/quality-gates.sh <gate>

Available gates:
  check                fmt --check + clippy + cargo check + cargo doc.
  solver-contracts     Verdict-ladder soundness invariants: the combine
                       table byte-equal across the Z3/CVC5 backends plus
                       the sound-decline guards. Fails closed if the
                       contract is weakened.
  determinism          Z3 (FFI) verdict determinism: the same query
                       under a pinned seed yields a stable, seed-
                       invariant verdict. Run single-threaded
                       (`--test-threads=1`) so a contended host cannot
                       mask a determinism regression.
  supply-chain         `cargo audit` plus the committed cargo-deny
                       advisory/license/ban/source policy.
  r2pipe-contracts     Validate live radare2 command response shapes.
  real-binaries        Build and score the labeled host fixture; fail
                       on any false actionable finding.
  all                  Run check + solver-contracts + determinism +
                       supply-chain.

EOF
}

gate_check() {
  # `--all-features` is applied to the whole workspace EXCEPT r2smt-explore.
  # Its off-by-default `oracle-radius2` feature pulls in radius2 -> boolector-sys
  # 0.7.2, whose vendored `cmake_minimum_required(VERSION 3.3)` is rejected by
  # cmake >= 4.0 (a transitive C-toolchain incompatibility, not a Rust issue).
  # r2smt-cli requests no features of r2smt-explore, so excluding it here keeps
  # oracle-radius2 out of the graph; its default-feature code is linted
  # separately below.
  echo "==> cargo fmt --all -- --check"
  cargo fmt --all -- --check

  echo "==> cargo clippy --workspace --exclude r2smt-explore --all-targets --all-features -- -D warnings"
  cargo clippy --workspace --exclude r2smt-explore --all-targets --all-features -- -D warnings

  echo "==> cargo clippy -p r2smt-explore --all-targets -- -D warnings"
  cargo clippy -p r2smt-explore --all-targets -- -D warnings

  echo "==> cargo check --workspace --exclude r2smt-explore --all-targets --all-features"
  cargo check --workspace --exclude r2smt-explore --all-targets --all-features

  echo "==> cargo check -p r2smt-explore --all-targets"
  cargo check -p r2smt-explore --all-targets

  echo "==> cargo doc --workspace --exclude r2smt-explore --no-deps --all-features"
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude r2smt-explore --no-deps --all-features

  echo "==> cargo doc -p r2smt-explore --no-deps"
  RUSTDOCFLAGS="-D warnings" cargo doc -p r2smt-explore --no-deps
}

gate_solver_contracts() {
  # Focused, fast soundness invariants — not the full Z3 suite.
  # Multiple positional filters are OR-matched by libtest
  # (Rust >= 1.70; workspace MSRV is 1.85).
  echo "==> solver-contracts: combine table + sound-decline guards"
  # `combine_table_contract_is_exhaustive_and_sound` is defined
  # identically in solver.rs (Z3), cvc5.rs and bitwuzla.rs, so this
  # single filter asserts the verdict ladder byte-equal across all
  # three backends.
  # The float filters guard the QF_BVFP renderer: the two decline
  # contracts (a shape with no portable encoding must never reach a
  # solver) and the inversion's anti-fabrication contract (a NaN
  # reinterpret must leave both polarities satisfiable — the constant
  # placeholder it replaced fabricated an always-false verdict there).
  # The x87 filters pin the double-extended sort on all three backends:
  # the text pair must decline it (x87 is Z3-only), Z3 must resolve its
  # value and its 80-bit stored image, and a free extended load must
  # stay undecided rather than resolve a non-canonical encoding to its
  # canonical counterpart.
  # `--test-threads=1` for the same reason the determinism gate uses it:
  # two of the x87 contracts are seconds of Z3 work rather than
  # milliseconds, and running them beside the rest of the file on a
  # contended host is what turned them into a flaky gate. Their budget
  # is now a resource limit rather than a wall clock, so this is belt
  # and braces — but it also keeps the gate honest if a future contract
  # arrives with a wall-clock budget.
  cargo test -p r2smt-smt --lib -- --test-threads=1 \
    combine_table_contract_is_exhaustive_and_sound \
    truncated_slice_is_reported_unsound \
    slice_with_unknown_is_declined_without_spawning_cvc5 \
    slice_with_unknown_is_declined_without_spawning_bitwuzla \
    slice_with_unrenderable_float_is_declined_without_spawning_cvc5 \
    slice_with_unrenderable_float_is_declined_without_spawning_bitwuzla \
    x87_extended_sort_is_declined_without_spawning_cvc5 \
    x87_extended_sort_is_declined_without_spawning_bitwuzla \
    x87_extended_sort_stores_the_binary64_image_of_one \
    x87_one_operand_pop_writes_the_named_slot_not_the_top \
    x87_reverse_suffix_swaps_the_operands_of_a_one_operand_pop \
    x87_extended_sort_stores_the_explicit_integer_bit \
    x87_extended_sort_keeps_a_free_extended_load_undecided \
    x87_equality_after_a_free_compare_stays_undecided \
    smtlib_fp_inversion_on_nan_keeps_both_polarities_satisfiable \
    smtlib_ssa_renamed_slice_solves_to_the_same_verdict_as_the_z3_backend
}

gate_determinism() {
  echo "==> determinism: pinned-seed verdict stability (Z3 FFI, serial)"
  # `--test-threads=1` serialises the Z3 FFI tests so a contended
  # host cannot turn a real determinism regression into a flaky
  # pass. The filter matches both `determinism_*` contracts in
  # solver/tests.rs.
  cargo test -p r2smt-smt --lib -- --test-threads=1 determinism
}

gate_supply_chain() {
  echo "==> cargo audit"
  if ! command -v cargo-audit >/dev/null 2>&1; then
    echo "cargo-audit not found. Install: cargo install --locked cargo-audit" >&2
    exit 1
  fi
  cargo audit

  echo "==> cargo deny check"
  if ! command -v cargo-deny >/dev/null 2>&1; then
    echo "cargo-deny not found. Install: cargo install --locked cargo-deny" >&2
    exit 1
  fi
  cargo deny check
}

gate_r2pipe_contracts() {
  echo "==> r2pipe contracts"
  cargo test -p r2smt-r2pipe --test contract_fixtures
  "${SCRIPT_DIR}/r2pipe-contracts.sh" "${R2SMT_SMOKE_BIN:-$(type -P true)}"
}

gate_real_binaries() {
  echo "==> labeled real-binary corpus"
  corpus_root="${R2SMT_CORPUS_ROOT:-${REPO_ROOT}/target/r2smt-corpus}"
  if [[ -z "${R2SMT_CORPUS_ROOT:-}" ]]; then
    corpus_root="$("${SCRIPT_DIR}/fetch-bench-corpus.sh" "$corpus_root")"
  fi
  cargo build --quiet -p r2smt-cli -p r2smt-bench
  cli="${R2SMT_CLI:-${REPO_ROOT}/target/debug/r2smt}"
  bench="${R2SMT_BENCH:-${REPO_ROOT}/target/debug/r2smt-bench}"
  "$bench" validate "$corpus_root"
  binary="$("${SCRIPT_DIR}/build-bench-corpus.sh" "target/r2smt-bench-corpus" "$corpus_root")"
  metrics_dir="target/r2smt-bench-metrics"
  rm -rf "$metrics_dir"
  mkdir -p "$metrics_dir"

  now_ms() {
    python3 -c 'import time; print(time.time_ns() // 1_000_000)'
  }

  score_binary() {
    local fixture="$1"
    local sample="$2"
    local report="$metrics_dir/$fixture.report.json"
    local metrics="$metrics_dir/$fixture.metrics.json"
    local started finished elapsed raw
    echo "==> scoring fixture ${fixture}"
    started="$(now_ms)"
    "$cli" solve "$sample" \
      --include-real --include-suspicious --json "$report"
    finished="$(now_ms)"
    elapsed=$((finished - started))
    raw="$metrics.raw"
    "$bench" score \
      "$report" "$corpus_root/${fixture%%-O2}" --elapsed-ms "$elapsed" \
      --verified-patches 0 --patch-attempts 0 \
      --rollback-successes 0 --rollback-attempts 0 > "$raw"
    jq --arg fixture "$fixture" '. + {fixture: $fixture}' "$raw" > "$metrics"
    rm -f "$raw"
  }

  score_binary control-flow "$binary"
  score_binary edge-cases "${binary%/*}/edge-cases"
  score_binary dataflow "${binary%/*}/dataflow"
  score_binary dataflow-O2 "${binary%/*}/dataflow-O2"
  score_binary loop-memory "${binary%/*}/loop-memory"
  score_binary signed-unsigned "${binary%/*}/signed-unsigned"

  for dataflow in "${binary%/*}/dataflow" "${binary%/*}/dataflow-O2"; do
    "$cli" analyze "$dataflow" \
      --json "${dataflow}.analysis.json"
  done

  matrix_dir="${binary%/*}/portable-matrix"
  "${SCRIPT_DIR}/validate-bench-matrix.sh" "$matrix_dir"
  matrix_cli="$cli"
  while IFS= read -r artifact; do
    "$matrix_cli" analyze "$matrix_dir/$artifact" \
      --json "$matrix_dir/$artifact.analysis.json"
  done < <(jq -r '.variants[].artifact' "$matrix_dir/manifest.json")

  patchable="$(dirname "$binary")/patch-control-flow"
  patched="${patchable}.patched"
  manifest="${patched}.r2smt.manifest.json"
  rm -f "$patched" "$manifest"
  if command -v sha256sum >/dev/null 2>&1; then
    precondition="$(sha256sum "$patchable" | awk '{print $1}')"
  else
    precondition="$(shasum -a 256 "$patchable" | awk '{print $1}')"
  fi
  "$cli" patch "$patchable" \
    --apply --expect-sha256 "$precondition" --output "$patched"
  "$cli" patch "$patched" --verify-only

  in_place="${patchable}.in-place"
  backup="${in_place}.r2smt.bak"
  in_place_manifest="${in_place}.r2smt.manifest.json"
  rm -f "$in_place" "$backup" "$in_place_manifest"
  cp "$patchable" "$in_place"
  "$cli" patch "$in_place" \
    --apply --expect-sha256 "$precondition" --in-place --backup "$backup"
  "$cli" patch "$in_place" --verify-only
  "$cli" patch "$in_place" --rollback
  if command -v sha256sum >/dev/null 2>&1; then
    postcondition="$(sha256sum "$in_place" | awk '{print $1}')"
  else
    postcondition="$(shasum -a 256 "$in_place" | awk '{print $1}')"
  fi
  [[ "$postcondition" == "$precondition" ]]

  jq '.verified_patches = 1
      | .patch_attempts = 1
      | .verified_patch_rate = 100
      | .rollback_successes = 1
      | .rollback_attempts = 1
      | .rollback_success_rate = 100' \
    "$metrics_dir/control-flow.metrics.json" > "$metrics_dir/control-flow.metrics.json.tmp"
  mv "$metrics_dir/control-flow.metrics.json.tmp" "$metrics_dir/control-flow.metrics.json"
  aggregate_output="target/r2smt-bench-metrics.json"
  rm -f "$aggregate_output"
  "${SCRIPT_DIR}/aggregate-bench-metrics.sh" \
    "$aggregate_output" "$metrics_dir"/*.metrics.json
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
    check)
      gate_check
      ;;
    solver-contracts)
      gate_solver_contracts
      ;;
    determinism)
      gate_determinism
      ;;
    supply-chain)
      gate_supply_chain
      ;;
    r2pipe-contracts)
      gate_r2pipe_contracts
      ;;
    real-binaries)
      gate_real_binaries
      ;;
    all)
      gate_check
      gate_solver_contracts
      gate_determinism
      gate_supply_chain
      gate_r2pipe_contracts
      gate_real_binaries
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "Unknown gate: $1" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
