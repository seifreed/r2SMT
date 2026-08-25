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
git_config="$home/.gitconfig"
HOME="$home" GIT_CONFIG_GLOBAL="$git_config" git config --global user.name "r2SMT CI"
HOME="$home" GIT_CONFIG_GLOBAL="$git_config" git config --global user.email "r2smt-ci@example.invalid"
HOME="$home" GIT_CONFIG_GLOBAL="$git_config" "$cli" annotate "$binary" --save-project "$project_name" >"$work/annotate.log"
project="$home/.local/share/radare2/projects/$project_name/rc.r2"
test -s "$project"
grep -q 'CCu' "$project"
