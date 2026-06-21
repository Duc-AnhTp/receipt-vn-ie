"""Sidecar metadata for future inference runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_inference_sidecar(
    output_jsonl: str | Path,
    *,
    method: str,
    checkpoint: str | None,
    device: str,
    prediction_count: int,
    inference_arguments: Dict[str, Any],
    validity_status: str = "unvalidated_new_run",
) -> Path:
    output_path = Path(output_jsonl)
    sidecar_path = output_path.with_suffix(".meta.json")
    payload = {
        "method": method,
        "prediction_path": output_path.as_posix(),
        "prediction_sha256": sha256_file(output_path),
        "prediction_count": prediction_count,
        "checkpoint": checkpoint,
        "device": device,
        "inference_arguments": inference_arguments,
        "git_commit": current_git_commit(),
        "validity_status": validity_status,
    }
    sidecar_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sidecar_path
