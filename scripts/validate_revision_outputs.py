#!/usr/bin/env python3
"""Command-line wrapper for revision output validation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acli.revision.validation import validate_all_outputs


if __name__ == "__main__":
    report = validate_all_outputs(ROOT, require_paper_profile=True)
    if report["status"] != "passed":
        raise SystemExit("Revision output validation failed; see result/artifact/output_validation.json")
    print("Revision output validation passed")
