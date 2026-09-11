#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec "${NBA_PYTHON:-$SCRIPT_DIR/.venv/bin/python}" -m nba_research "$@"
