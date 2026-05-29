#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/receipt-vn-ie}"
DATASET_DIR="${DATASET_DIR:-/kaggle/input/receipt-vn-ie-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/kaggle/input/receipt-vn-ie-checkpoints}"

cd "$PROJECT_DIR"

python -m pip install -e .
python -m pip install -r requirements/app.txt
python -m pip install -r requirements/ocr.txt
python -m pip install paddlepaddle-gpu

mkdir -p data checkpoints outputs

if [ -f "$DATASET_DIR/receipt_dataset.zip" ]; then
  PROJECT_DIR="$PROJECT_DIR" DATASET_DIR="$DATASET_DIR" python - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile

zip_path = Path(os.environ["DATASET_DIR"]) / "receipt_dataset.zip"
out_dir = Path(os.environ["PROJECT_DIR"])
with ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
print(f"Extracted {zip_path} -> {out_dir}")
if not (out_dir / "data").exists():
    raise SystemExit(f"Expected extracted dataset to contain {out_dir / 'data'}")
PY
else
  echo "Warning: $DATASET_DIR/receipt_dataset.zip not found. Data bootstrap skipped."
fi

if [ -d "$CHECKPOINT_DIR" ]; then
  PROJECT_DIR="$PROJECT_DIR" CHECKPOINT_DIR="$CHECKPOINT_DIR" python - <<'PY'
import os
import shutil
from pathlib import Path

project_dir = Path(os.environ["PROJECT_DIR"])
checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
target_root = project_dir / "checkpoints"
target_root.mkdir(parents=True, exist_ok=True)

expected = {
    "donut": target_root / "donut" / "receipt_ie" / "final",
    "layoutxlm": target_root / "layoutxlm" / "receipt_ie" / "final",
}

def copy_tree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied {src} -> {dst}")
    return True

copied = False
for name, dst in expected.items():
    candidates = [
        checkpoint_dir / name / "receipt_ie" / "final",
        checkpoint_dir / "checkpoints" / name / "receipt_ie" / "final",
        checkpoint_dir / f"{name}_final",
    ]
    for src in candidates:
        if copy_tree(src, dst):
            copied = True
            break

if not copied:
    print(f"Warning: no known checkpoint layout found under {checkpoint_dir}.")
PY
else
  echo "Warning: checkpoint dataset not found. Checkpoint bootstrap skipped."
fi

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
