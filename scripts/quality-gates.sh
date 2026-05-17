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
  supply-chain         `cargo audit` against the RustSec advisory DB.
                       NOTE: `cargo deny` is intentionally NOT run here —
                       deny.toml is kept local-only by maintainer
                       decision, so the license/bans/sources policy is a
                       manual local check, not a CI gate.
  all                  Run check + solver-contracts + supply-chain.

Planned gates (not yet implemented):
  r2pipe-contracts     Verify r2pipe extraction JSON shape.
  real-binaries        End-to-end on synthetic labeled fixtures.
EOF
}

gate_check() {
  echo "==> cargo fmt --all -- --check"
  cargo fmt --all -- --check

  echo "==> cargo clippy --workspace --all-targets --all-features -- -D warnings"
  cargo clippy --workspace --all-targets --all-features -- -D warnings

  echo "==> cargo check --workspace --all-targets --all-features"
  cargo check --workspace --all-targets --all-features

  echo "==> cargo doc --workspace --no-deps --all-features"
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
}

gate_solver_contracts() {
  # Focused, fast soundness invariants — not the full Z3 suite.
  # Multiple positional filters are OR-matched by libtest
  # (Rust >= 1.70; workspace MSRV is 1.85).
  echo "==> solver-contracts: combine table + sound-decline guards"
  cargo test -p r2smt-smt --lib -- \
    combine_table_contract_is_exhaustive_and_sound \
    truncated_slice_is_reported_unsound \
    slice_with_unknown_is_declined_without_spawning_cvc5
}

gate_supply_chain() {
  echo "==> cargo audit"
  if ! command -v cargo-audit >/dev/null 2>&1; then
    echo "cargo-audit not found. Install: cargo install --locked cargo-audit" >&2
    exit 1
  fi
  cargo audit
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
    supply-chain)
      gate_supply_chain
      ;;
    all)
      gate_check
      gate_solver_contracts
      gate_supply_chain
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
