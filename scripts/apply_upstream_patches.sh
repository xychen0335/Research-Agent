#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIRD_ROOT="${ROOT}/upstream/BIRD-RL"
PATCH="${ROOT}/patches/bird-rl-readonly.patch"

[[ -d "${BIRD_ROOT}/.git" ]] || { echo "Run scripts/setup_autodl.sh first." >&2; exit 1; }
if git -C "${BIRD_ROOT}" apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
  echo "BIRD-RL read-only patch is already applied."
elif git -C "${BIRD_ROOT}" apply --check "${PATCH}"; then
  git -C "${BIRD_ROOT}" apply "${PATCH}"
  echo "Applied BIRD-RL read-only SQLite patch."
else
  echo "BIRD-RL read-only patch does not match the pinned commit." >&2
  exit 1
fi
