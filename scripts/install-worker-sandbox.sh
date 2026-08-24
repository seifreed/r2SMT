#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y apparmor apparmor-profiles apparmor-utils bubblewrap

profile=/usr/share/apparmor/extra-profiles/bwrap-userns-restrict
if [[ ! -f "$profile" ]]; then
  echo "missing AppArmor profile: $profile" >&2
  exit 1
fi

sudo install -Dm644 "$profile" /etc/apparmor.d/bwrap-userns-restrict
sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict

# Fail during setup instead of turning a sandbox regression into an analysis failure.
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
bwrap --die-with-parent --unshare-net --ro-bind / / --bind "$tmpdir" "$tmpdir" true
