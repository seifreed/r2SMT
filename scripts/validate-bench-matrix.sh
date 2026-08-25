#!/usr/bin/env bash

set -euo pipefail

matrix_dir=${1:?usage: validate-bench-matrix.sh <portable-matrix-dir>}
manifest="$matrix_dir/manifest.json"
test -s "$manifest"

jq -e '
  .schema_version == 1
  and (.fixture == "portable-matrix")
  and (.variants | length >= 6)
  and ([.variants[].architecture] | unique | length >= 5)
  and ([.variants[].compiler] | unique | length >= 2)
  and ([.variants[].optimization] | unique | length >= 4)
' "$manifest" >/dev/null

while IFS= read -r artifact; do
  test -s "$matrix_dir/$artifact"
done < <(jq -r '.variants[].artifact' "$manifest")

echo "portable matrix ok: $(jq '.variants | length' "$manifest") variants"
