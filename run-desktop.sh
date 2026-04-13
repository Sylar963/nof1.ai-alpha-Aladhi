#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/venv-gtk/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Missing GTK virtualenv at %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

export PYWEBVIEW_GUI="${PYWEBVIEW_GUI:-gtk}"
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  export GDK_BACKEND="${GDK_BACKEND:-wayland}"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/main.py"
