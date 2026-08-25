#!/usr/bin/env bash

set -euo pipefail

if [[ ${R2SMT_PLUGIN_FIXTURE_CLI:-0} == 1 ]]; then
  if [[ ${1:-} == annotate ]]; then
    printf '# r2SMT annotations\n\n'
    printf 'CCu base64:cjJzbXQtcGx1Z2luLXNtb2tl @ %s\n' "$4"
    printf '# patch hint only\n'
  else
    printf '%s' "${1:-}"
    shift || true
    printf ' %s' "$@"
    printf '\n'
  fi
  exit 0
fi

plugin=${1:?usage: r2-plugin-smoke.sh CORE_PLUGIN [BINARY]}
binary=${2:-/bin/ls}
test -f "$plugin"
test -f "$binary"

work=$(mktemp -d "${TMPDIR:-/tmp}/r2smt-plugin.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
plugdir="$work/radare2/plugins"
mkdir -p "$plugdir"
cp "$plugin" "$plugdir/"
test_binary="$work/sample binary"
cp "$binary" "$test_binary"
fixture_cli="$work/r2smt fixture"
cp "$0" "$fixture_cli"
chmod +x "$fixture_cli"

run_r2() {
  XDG_DATA_HOME="$work" R2SMT_CLI="$fixture_cli" \
    R2SMT_PLUGIN_FIXTURE_CLI=1 r2 -2Nq -e scr.color=false "$@"
}

help=$(run_r2 -c 'r2smt?;q' "$test_binary")
grep -q 'r2smt annotate' <<<"$help"

bridge=$(run_r2 -c 'r2smt explain --solver z3;q' "$test_binary")
grep -Eq "^at .*sample binary 0x[[:xdigit:]]+ --explain --solver z3$" <<<"$bridge"

solve=$(run_r2 -c 'r2smt solve;q' "$test_binary")
grep -Eq '^solve .*sample binary --at 0x[[:xdigit:]]+ --include-suspicious$' <<<"$solve"

patch=$(run_r2 -c 'r2smt patch;q' "$test_binary")
grep -Eq '^at .*sample binary 0x[[:xdigit:]]+ --patch$' <<<"$patch"

patch_dry=$(run_r2 -c 'r2smt patch-dry;q' "$test_binary")
grep -Eq '^patch .*sample binary --at 0x[[:xdigit:]]+$' <<<"$patch_dry"

annotate=$(run_r2 -c 'r2smt annotate;CC.;q' "$test_binary")
grep -q 'r2smt-plugin-smoke' <<<"$annotate"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
legacy=$(run_r2 -i "$repo_root/r2pm/r2smt.r2" -c '$r2smt-at-v;q' "$test_binary")
grep -Eq '^at .*sample binary 0x[[:xdigit:]]+ --explain$' <<<"$legacy"

echo "r2smt core plugin smoke test passed"
