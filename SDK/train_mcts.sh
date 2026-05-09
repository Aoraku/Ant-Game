#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=""
declare -a CANDIDATES=()
[[ -n "${PYTHON:-}" ]] && CANDIDATES+=("$PYTHON")
[[ -n "${ANTWAR_PYTHON:-}" ]] && CANDIDATES+=("$ANTWAR_PYTHON")
CANDIDATES+=(
  "${HOME}/miniconda3/envs/env/bin/python"
  "python3"
  "python"
)
for candidate in "${CANDIDATES[@]}"; do
  if [[ "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi
  [[ -x "$candidate" ]] || continue
  if "$candidate" - <<'PY' >/dev/null 2>&1; then
import sys
if sys.version_info < (3, 10):
    raise SystemExit(1)
import gymnasium  # noqa: F401
import numpy  # noqa: F401
import pettingzoo  # noqa: F401
PY
    PYTHON_BIN="$candidate"
    break
  fi
done
[[ -n "$PYTHON_BIN" ]] || PYTHON_BIN="python3"
exec "$PYTHON_BIN" "${SCRIPT_DIR}/train_mcts.py" "$@"
