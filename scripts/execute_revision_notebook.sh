#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source .venv/bin/activate
export MPLBACKEND=Agg

mkdir -p result/artifact result/figure/paper
python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
manifest = yaml.safe_load((root / "paper_figures/manifest.yaml").read_text(encoding="utf-8"))
for item in manifest["figures"]:
    for extension in ("pdf", "png"):
        (root / "result/figure/paper" / f"{item['output']}.{extension}").unlink(missing_ok=True)
for name in (
    "06_major_revision_paper_figures.executed.ipynb",
    "06_major_revision_paper_figures.executed.html",
):
    (root / "result/artifact" / name).unlink(missing_ok=True)
PY
jupyter nbconvert \
  --to notebook \
  --execute notebook/06_major_revision_paper_figures.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --output "$REPO_ROOT/result/artifact/06_major_revision_paper_figures.executed.ipynb"
jupyter nbconvert \
  --to html \
  "$REPO_ROOT/result/artifact/06_major_revision_paper_figures.executed.ipynb" \
  --output-dir "$REPO_ROOT/result/artifact"

test -s result/artifact/06_major_revision_paper_figures.executed.ipynb
test -s result/artifact/06_major_revision_paper_figures.executed.html
