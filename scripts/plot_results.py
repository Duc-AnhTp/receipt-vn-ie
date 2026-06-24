"""Create figures from metrics and error-analysis artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIELDS = ["store_name", "date", "total", "address", "macro"]
FIELD_LABELS = ["STORE", "DATE", "TOTAL", "ADDRESS", "MACRO"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "layoutxlm": "LayoutXLM",
    "donut": "Donut",
}
ERROR_TYPES = [
    "EMPTY_PRED",
    "FORMAT_ERROR",
    "OCR_MISS",
    "OCR_WRONG",
    "POSTPROCESS_BAD",
    "MODEL_BAD",
    "GT_EMPTY_PRED_NONEMPTY",
]


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_view(method_metrics, view: str):
    if view == "all_samples":
        return method_metrics.get("all_samples", method_metrics)
    return method_metrics[view]


def save_figure(fig, filename: str, output_dir: Path, report_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    shutil.copy2(output_path, report_dir / filename)
    plt.close(fig)


def grouped_metric_plot(
    data,
    methods,
    *,
    view: str,
    metric: str,
    title: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(FIELDS))
    width = 0.36
    offsets = np.linspace(
        -width * (len(methods) - 1) / 2,
        width * (len(methods) - 1) / 2,
        len(methods),
    )
    for offset, method in zip(offsets, methods):
        current = metric_view(data[method], view)
        values = [current[field][metric] * 100 for field in FIELDS]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS[method],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(FIELD_LABELS)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_metrics(output_dir: Path, report_dir: Path) -> None:
    manifest = load_json("outputs/metrics/artifact_manifest.json")
    metrics = load_json("outputs/metrics/combined_metrics.json")
    methods = [
        method
        for method, item in manifest["methods"].items()
        if item.get("valid_for_main_comparison")
    ]
    figures = [
        (
            grouped_metric_plot(
                metrics,
                methods,
                view="all_samples",
                metric="EM",
                title="Exact Match on all test samples",
                ylabel="Exact Match (%)",
            ),
            "em_comparison.png",
        ),
        (
            grouped_metric_plot(
                metrics,
                methods,
                view="all_samples",
                metric="NES",
                title="Character-sequence similarity on all test samples",
                ylabel="Normalized Edit Similarity (%)",
            ),
            "nes_comparison.png",
        ),
        (
            grouped_metric_plot(
                metrics,
                methods,
                view="present_only",
                metric="EM",
                title="Present-only Exact Match",
                ylabel="Exact Match (%)",
            ),
            "em_present_only_comparison.png",
        ),
    ]
    for figure, filename in figures:
        save_figure(figure, filename, output_dir, report_dir)


def plot_error_analysis(output_dir: Path, report_dir: Path) -> None:
    for method in ("baseline", "layoutxlm", "donut"):
        path = Path(f"outputs/error_analysis/{method}_error_by_field.csv")
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        counts = (
            frame.groupby(["field", "error_type"])
            .size()
            .unstack(fill_value=0)
        )
        for error_type in ERROR_TYPES:
            if error_type not in counts.columns:
                counts[error_type] = 0
        counts = counts[ERROR_TYPES]
        fig, ax = plt.subplots(figsize=(11, 6))
        counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_ylabel("Error count")
        ax.set_xlabel("Field")
        suffix = " - diagnostic only" if method == "donut" else ""
        ax.set_title(
            f"Error taxonomy by field - {METHOD_LABELS[method]}{suffix}"
        )
        ax.legend(
            title="Error type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        plt.xticks(rotation=0)
        fig.tight_layout()
        save_figure(
            fig,
            f"{method}_error_distribution.png",
            output_dir,
            report_dir,
        )


def main() -> None:
    output_dir = Path("outputs/plots")
    report_dir = Path("report/figures")
    plot_metrics(output_dir, report_dir)
    plot_error_analysis(output_dir, report_dir)
    print("Generated plots from frozen prediction-derived artifacts.")


if __name__ == "__main__":
    main()
