#!/usr/bin/env bash

set -euo pipefail

cli=${1:?usage: r2pm-annotate-smoke.sh R2SMT_BINARY PATCHABLE_BINARY WORK_DIR}
binary=${2:?usage: r2pm-annotate-smoke.sh R2SMT_BINARY PATCHABLE_BINARY WORK_DIR}
work=${3:?usage: r2pm-annotate-smoke.sh R2SMT_BINARY PATCHABLE_BINARY WORK_DIR}
mkdir -p "$work"

script="$work/annotations.r2"
"$cli" annotate "$binary" --r2-script >"$script"
grep -q '^CCu ' "$script"

home="$work/home"
project_name=triage
mkdir -p "$home"
HOME="$home" "$cli" annotate "$binary" --save-project "$project_name" >"$work/annotate.log"
project="$home/.local/share/radare2/projects/$project_name/rc.r2"
test -s "$project"
grep -q 'CCu' "$project"
