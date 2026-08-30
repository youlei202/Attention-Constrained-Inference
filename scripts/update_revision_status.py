#!/usr/bin/env python3
"""Atomically annotate the experiment state with end-to-end pipeline progress."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.suite import STATE_PATH, atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--pipeline-status",
        choices=("running", "completed", "failed"),
        required=True,
    )
    parser.add_argument("--exit-code", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if STATE_PATH.is_file():
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["pipeline_phase"] = args.phase
    payload["pipeline_status"] = args.pipeline_status
    payload["pipeline_updated_utc"] = datetime.now(timezone.utc).isoformat()
    if args.exit_code is not None:
        payload["pipeline_exit_code"] = args.exit_code
    else:
        payload.pop("pipeline_exit_code", None)
    atomic_write_json(STATE_PATH, payload)


if __name__ == "__main__":
    main()
