#!/usr/bin/env python3
"""Run the checkpointed major-revision experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.suite import run_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke", help="smoke, paper, or a YAML path")
    parser.add_argument("--workers", type=int, default=None, help="maximum ordinary task workers")
    parser.add_argument("--resume", action="store_true", help="reuse checksum-validated checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = run_suite(args.profile, workers=args.workers, resume=args.resume)
    print(f"Revision experiments completed: {status['profile']}")


if __name__ == "__main__":
    main()
