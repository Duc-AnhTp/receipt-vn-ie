"""Build a reproducibility manifest without modifying frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

import yaml
from transformers import AutoTokenizer

from receipt_ie.data.build_donut_dataset import target_to_donut_sequence


DEFAULT_GENERATION_MAX_LENGTH = 20


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_generation_max_length(checkpoint: Path) -> Dict[str, Any]:
    model_config_path = checkpoint / "config.json"
    generation_config_path = checkpoint / "generation_config.json"
    model_config = (
        json.loads(model_config_path.read_text(encoding="utf-8"))
        if model_config_path.exists()
        else {}
    )
    generation_config = (
        json.loads(generation_config_path.read_text(encoding="utf-8"))
        if generation_config_path.exists()
        else {}
    )
    if "max_length" in generation_config:
        return {
            "value": int(generation_config["max_length"]),
            "source": "generation_config.json",
        }
    if "max_length" in model_config:
        return {
            "value": int(model_config["max_length"]),
            "source": "config.json",
        }
    return {
        "value": DEFAULT_GENERATION_MAX_LENGTH,
        "source": "transformers_default",
    }


def donut_truncation_evidence(
    checkpoint: Path,
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    effective_max_length: int,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    target_lengths = [
        len(
            tokenizer(
                target_to_donut_sequence(
                    item.get("target") or item,
                    "<s_receipt_ie>",
                ),
                add_special_tokens=False,
            ).input_ids
        )
        for item in ground_truths
    ]
    output_lengths = [
        len(
            tokenizer(
                str(item.get("raw_output") or ""),
                add_special_tokens=False,
            ).input_ids
        )
        for item in predictions
    ]
    return {
        "n_targets_over_effective_max_length": sum(
            length > effective_max_length for length in target_lengths
        ),
        "target_token_length_median": median(target_lengths),
        "target_token_length_max": max(target_lengths),
        "n_outputs_at_effective_max_length": sum(
            length == effective_max_length for length in output_lengths
        ),
        "output_token_length_min": min(output_lengths),
        "output_token_length_max": max(output_lengths),
    }


def ocr_cache_coverage(ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = 0
    nonempty = 0
    word_counts = []
    for item in ground_truths:
        cache_path = Path(str(item.get("ocr_cache_path") or ""))
        if not cache_path.exists():
            continue
        existing += 1
        try:
            words = json.loads(
                cache_path.read_text(encoding="utf-8")
            ).get("words", [])
        except (OSError, json.JSONDecodeError):
            words = []
        word_counts.append(len(words))
        if words:
            nonempty += 1
    return {
        "n_ground_truth_samples": len(ground_truths),
        "n_cache_files_existing": existing,
        "n_nonempty_cache_files": nonempty,
        "min_words": min(word_counts) if word_counts else 0,
        "median_words": median(word_counts) if word_counts else 0,
    }


def build_manifest(config_path: Path) -> Dict[str, Any]:
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gold_path = Path(policy["gold_path"])
    ground_truths = read_jsonl(gold_path)
    manifest: Dict[str, Any] = {
        "version": policy.get("version", 1),
        "generated_from": config_path.as_posix(),
        "git_commit": git_commit(),
        "gold": {
            "path": gold_path.as_posix(),
            "sha256": sha256_file(gold_path),
            "n_samples": len(ground_truths),
        },
        "ocr_cache_coverage": ocr_cache_coverage(ground_truths),
        "methods": {},
    }
    for method, method_policy in policy["methods"].items():
        prediction_path = Path(method_policy["prediction_path"])
        predictions = read_jsonl(prediction_path)
        entry = dict(method_policy)
        entry.update({
            "prediction_path": prediction_path.as_posix(),
            "prediction_sha256": sha256_file(prediction_path),
            "n_predictions": len(predictions),
            "n_inference_errors": sum(
                item.get("status") == "error" for item in predictions
            ),
        })
        checkpoint_value = entry.get("checkpoint")
        if checkpoint_value:
            checkpoint = Path(checkpoint_value)
            entry["checkpoint"] = checkpoint.as_posix()
            config_file = checkpoint / "config.json"
            if config_file.exists():
                entry["checkpoint_config_sha256"] = sha256_file(config_file)
        if method == "donut" and checkpoint_value:
            generation_limit = checkpoint_generation_max_length(
                Path(checkpoint_value)
            )
            entry["effective_generation_max_length"] = generation_limit["value"]
            entry["effective_generation_max_length_source"] = generation_limit[
                "source"
            ]
            entry["truncation_evidence"] = donut_truncation_evidence(
                Path(checkpoint_value),
                ground_truths,
                predictions,
                generation_limit["value"],
            )
        manifest["methods"][method] = entry
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/artifact_validity.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics/artifact_manifest.json",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_manifest(Path(args.config)),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Artifact manifest saved to {output}")


if __name__ == "__main__":
    main()
