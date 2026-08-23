#!/usr/bin/env bash

set -euo pipefail

out="${1:-target/r2smt-bench-corpus}"
cc="${CC:-cc}"
mkdir -p "$out"

pie_flags=(-fno-pie -no-pie)
if [[ "$(uname -s)" == Darwin ]]; then
  pie_flags=(-fno-pie "-Wl,-no_pie")
fi

"$cc" -O0 -g0 "${pie_flags[@]}" \
  bench/corpus/control-flow/source/main.c \
  bench/corpus/control-flow/source/opaque.c \
  -o "$out/control-flow"

case "$(uname -s):$(uname -m)" in
  Darwin:arm64) patch_asm=bench/corpus/control-flow/assembly/patch_aarch64.s ;;
  *) patch_asm=bench/corpus/control-flow/assembly/patch_x86_64.s ;;
esac
"$cc" -O0 -g0 "${pie_flags[@]}" \
  bench/corpus/control-flow/source/patch_main.c "$patch_asm" \
  -o "$out/patch-control-flow"

echo "$out/control-flow"
