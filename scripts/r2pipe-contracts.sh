#!/usr/bin/env bash

set -euo pipefail

binary="${1:-$(type -P true)}"
if [[ ! -f "$binary" ]]; then
  echo "contract binary not found: $binary" >&2
  exit 1
fi
command -v radare2 >/dev/null || { echo "radare2 not found" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }

r2_json() {
  local command="$1"
  radare2 -2q -e scr.color=false -c "aaa;${command};q" "$binary"
}

check_json() {
  local command="$1"
  local filter="$2"
  local response
  response="$(r2_json "$command")"
  if ! jq -e "$filter" >/dev/null <<<"$response"; then
    echo "radare2 contract changed for '$command'" >&2
    jq . <<<"$response" >&2 || printf '%s\n' "$response" >&2
    exit 1
  fi
}

check_json "ij" '(.bin.arch | type) == "string" and (.bin.bits | type) == "number"'
aflj="$(r2_json "aflj")"
jq -e 'type == "array" and length > 0 and all(.[]; (.addr | type == "number") and (.name | type == "string"))' >/dev/null <<<"$aflj" || {
  echo "radare2 contract changed for 'aflj'" >&2
  exit 1
}
address="$(jq -r '.[0].addr' <<<"$aflj")"
check_json "agfj @ $address" 'type == "array" and length > 0 and (.[0].blocks | type == "array") and all(.[0].blocks[]; (.addr | type == "number") and (.ops | type == "array"))'
check_json "aoj 1 @ $address" 'type == "array" and length == 1 and (.[0].addr | type == "number") and (.[0].size | type == "number") and ((.[0].opcode // .[0].disasm) | type == "string")'
check_json "pdj 1 @ $address" 'type == "array" and length == 1 and (.[0].addr | type == "number") and (.[0].size | type == "number") and ((.[0].opcode // .[0].disasm) | type == "string")'
check_json "afvj @ $address" 'type == "object" or type == "array"'
check_json "iSj" 'type == "array" and all(.[]; (.vaddr | type == "number") and (.vsize | type == "number") and (.perm | type == "string"))'

pdgsd="$(radare2 -2q -e scr.color=false -c "aaa;pdgsd 1 @ $address;q" "$binary")"
if [[ "$pdgsd" != *"Unknown command"* && "$pdgsd" != *"Cannot find"* ]]; then
  grep -Eq '^0x[0-9a-fA-F]+:' <<<"$pdgsd" || {
    echo "radare2 contract changed for 'pdgsd'" >&2
    printf '%s\n' "$pdgsd" >&2
    exit 1
  }
fi

echo "r2pipe contracts ok: $(radare2 -qv)"
