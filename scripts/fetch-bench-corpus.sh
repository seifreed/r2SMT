#!/usr/bin/env bash

set -euo pipefail

readonly REPO_URL="${R2SMT_CORPUS_REPO:-https://github.com/seifreed/r2smt-corpus.git}"
readonly REVISION="${R2SMT_CORPUS_REV:-81a6c43014e169cacdc0711234319956fea90a94}"
readonly OUT="${1:-target/r2smt-corpus}"

if [[ -f "$OUT/.r2smt-corpus-revision" ]] \
  && [[ "$(<"$OUT/.r2smt-corpus-revision")" == "$REVISION" ]]; then
  echo "$OUT"
  exit 0
fi

parent="$(dirname "$OUT")"
mkdir -p "$parent"
tmp="$(mktemp -d "$parent/.r2smt-corpus.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

git clone --filter=blob:none --no-checkout "$REPO_URL" "$tmp"
git -C "$tmp" fetch --depth=1 origin "$REVISION"
git -C "$tmp" checkout --detach FETCH_HEAD
rm -rf "$tmp/.git"
printf '%s\n' "$REVISION" > "$tmp/.r2smt-corpus-revision"

rm -rf "$OUT"
mv "$tmp" "$OUT"
trap - EXIT
echo "$OUT"
