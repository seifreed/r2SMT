#!/usr/bin/env bash

set -euo pipefail

out="${1:-target/r2smt-bench-corpus}"
corpus_root="${2:-bench/corpus}"
cc="${CC:-cc}"
clang="${CLANG:-$(command -v clang || true)}"
gcc="${GCC:-$(command -v gcc || true)}"
mkdir -p "$out"

pie_flags=(-fno-pie -no-pie)
if [[ "$(uname -s)" == Darwin ]]; then
  pie_flags=(-fno-pie "-Wl,-no_pie")
fi

"$cc" -O0 -g0 "${pie_flags[@]}" \
  "$corpus_root/control-flow/source/main.c" \
  "$corpus_root/control-flow/source/opaque.c" \
  -o "$out/control-flow"

"$cc" -O0 -g0 "${pie_flags[@]}" \
  "$corpus_root/dataflow/source/main.c" \
  -o "$out/dataflow"

"$cc" -O2 -g0 "${pie_flags[@]}" \
  "$corpus_root/dataflow/source/main.c" \
  -o "$out/dataflow-O2"

case "$(uname -s):$(uname -m)" in
  Darwin:arm64) patch_asm="$corpus_root/control-flow/assembly/patch_aarch64.s" ;;
  *) patch_asm="$corpus_root/control-flow/assembly/patch_x86_64.s" ;;
esac
"$cc" -O0 -g0 "${pie_flags[@]}" \
  "$corpus_root/control-flow/source/patch_main.c" "$patch_asm" \
  -o "$out/patch-control-flow"

"$cc" -O0 -g0 "${pie_flags[@]}" \
  "$corpus_root/edge-cases/source/main.c" \
  -o "$out/edge-cases"

"$cc" -O0 -g0 "${pie_flags[@]}" \
  "$corpus_root/loop-memory/source/main.c" \
  -o "$out/loop-memory"

"$cc" -O0 -g0 "${pie_flags[@]}" \
  "$corpus_root/signed-unsigned/source/main.c" \
  -o "$out/signed-unsigned"

matrix_out="$out/portable-matrix"
mkdir -p "$matrix_out"
if [[ -z "$clang" || ! -x "$clang" ]]; then
  echo "clang is required for portable-matrix targets" >&2
  exit 1
fi
if [[ -z "$gcc" || ! -x "$gcc" ]]; then
  echo "gcc is required for portable-matrix targets" >&2
  exit 1
fi

portable_source="$corpus_root/portable-matrix/source/main.c"
gcc_arch_flags=()
if [[ "$(uname -s):$(uname -m)" == Darwin:arm64 ]]; then
  gcc_arch_flags=(-arch x86_64)
fi
"$gcc" "${gcc_arch_flags[@]}" -O0 -g0 -ffreestanding -nostdlib -c "$portable_source" \
  -o "$matrix_out/x86_64-gcc-O0.o"
"$clang" --target=x86_64-unknown-linux-gnu -O2 -g0 -ffreestanding -nostdlib -c \
  "$portable_source" -o "$matrix_out/x86_64-clang-O2.o"
"$clang" --target=i686-unknown-linux-gnu -O2 -g0 -ffreestanding -nostdlib -c \
  "$portable_source" -o "$matrix_out/i686-clang-O2.o"
"$clang" --target=aarch64-none-elf -O2 -g0 -ffreestanding -nostdlib -c \
  "$portable_source" -o "$matrix_out/aarch64-clang-O2.o"
"$clang" --target=armv7-none-eabi -marm -O3 -g0 -ffreestanding -nostdlib -c \
  "$portable_source" -o "$matrix_out/armv7-clang-O3.o"
"$clang" --target=thumbv7-none-eabi -mthumb -Os -g0 -ffreestanding -nostdlib -c \
  "$portable_source" -o "$matrix_out/thumb-clang-Os.o"

cat >"$matrix_out/manifest.json" <<'EOF'
{
  "schema_version": 1,
  "fixture": "portable-matrix",
  "variants": [
    {"artifact": "x86_64-gcc-O0.o", "architecture": "x86_64", "compiler": "gcc", "optimization": "O0", "format": "relocatable"},
    {"artifact": "x86_64-clang-O2.o", "architecture": "x86_64", "compiler": "clang", "optimization": "O2", "format": "relocatable"},
    {"artifact": "i686-clang-O2.o", "architecture": "x86", "compiler": "clang", "optimization": "O2", "format": "relocatable"},
    {"artifact": "aarch64-clang-O2.o", "architecture": "aarch64", "compiler": "clang", "optimization": "O2", "format": "relocatable"},
    {"artifact": "armv7-clang-O3.o", "architecture": "aarch32", "compiler": "clang", "optimization": "O3", "format": "relocatable"},
    {"artifact": "thumb-clang-Os.o", "architecture": "thumb", "compiler": "clang", "optimization": "Os", "format": "relocatable"}
  ]
}
EOF

echo "$out/control-flow"
