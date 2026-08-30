#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=aci_major_revision_integration
cd "$REPO_ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  SESSION_RUNNING=1
  echo "tmux: running ($SESSION)"
  tmux capture-pane -pt "$SESSION" -S -40
else
  SESSION_RUNNING=0
  echo "tmux: not running ($SESSION)"
fi

if [[ -s "$REPO_ROOT/state/revision_pipeline_status.json" ]]; then
  "$REPO_ROOT/.venv/bin/python" -m json.tool "$REPO_ROOT/state/revision_pipeline_status.json"
  "$REPO_ROOT/.venv/bin/python" - "$SESSION_RUNNING" <<'PY'
import json
from pathlib import Path
import sys

state = json.loads(Path("state/revision_pipeline_status.json").read_text(encoding="utf-8"))
session_running = bool(int(sys.argv[1]))
if state.get("pipeline_status") == "failed" or state.get("status") == "failed":
    sys.exit(1)
if not session_running and (
    state.get("pipeline_status") == "running" or state.get("status") == "running"
):
    print("state: interrupted (running state without live tmux session)", file=sys.stderr)
    sys.exit(1)
PY
else
  echo "state: no status file"
fi
