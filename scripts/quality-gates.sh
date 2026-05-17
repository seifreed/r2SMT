#!/usr/bin/env bash
# r2SMT quality gates — single entrypoint for repo gates.
#
# Each phase adds its own subcommand. Today only `check` is implemented;
# placeholders below mark the seams where future gates will land.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: scripts/quality-gates.sh <gate>

Available gates:
  check                Run fmt --check + clippy + cargo check + cargo doc.
  all                  Alias for `check` today; expands as phases land.

Planned gates (not yet implemented):
  r2pipe-contracts     Phase 1+ — verify r2pipe extraction JSON shape.
  solver-contracts     Phase 6+ — verify SMT classification semantics.
  real-binaries        Phase 6+ — end-to-end on synthetic fixtures.
  sbom                 Future — supply-chain review.
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

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
    check|all)
      gate_check
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
