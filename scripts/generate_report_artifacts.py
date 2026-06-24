"""Generate LaTeX tables and appendix snippets from measured artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from receipt_ie.metrics.evaluate_fields import evaluate_predictions


FIELDS = ["store_name", "date", "total", "address", "macro"]
FIELD_LABELS = {
    "store_name": "Tên CH",
    "date": "Ngày",
    "total": "Tổng",
    "address": "Địa chỉ",
    "macro": "Macro",
}
METHOD_LATEX = {
    "baseline": "Baseline",
    "layoutxlm": r"VietOCR + \model{LayoutXLM}",
    "donut": r"\model{Donut}",
}


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percent(value: float) -> str:
    return f"{value * 100:.2f}".replace(".", ",")


def decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def valid_methods(manifest: Dict[str, Any]) -> list[str]:
    return [
        method
        for method, item in manifest["methods"].items()
        if item.get("valid_for_main_comparison")
    ]


def metric_view(method_metrics: Dict[str, Any], view: str) -> Dict[str, Any]:
    """Support both legacy metrics and the explicit all_samples schema."""
    if view == "all_samples":
        return method_metrics.get("all_samples", method_metrics)
    return method_metrics[view]


def metric_rows(
    metrics: Dict[str, Any],
    methods: Iterable[str],
    view: str,
    metric: str,
) -> str:
    rows = []
    for method in methods:
        data = metric_view(metrics[method], view)
        values = " & ".join(percent(data[field][metric]) for field in FIELDS)
        rows.append(f"{METHOD_LATEX[method]} & {values} \\\\")
    return "\n".join(rows)


def metric_table(
    metrics: Dict[str, Any],
    methods: Iterable[str],
    *,
    view: str,
    caption: str,
    label: str,
) -> str:
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{{caption}}}
  \label{{{label}}}
  \small
  \begin{{tabularx}}{{\textwidth}}{{l C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.7cm}}}}
    \toprule
    \textbf{{Phương pháp}} & \textbf{{Tên CH}} & \textbf{{Ngày}} & \textbf{{Tổng}} & \textbf{{Địa chỉ}} & \textbf{{Macro}} \\
    \midrule
    \multicolumn{{6}}{{l}}{{\textbf{{Exact Match (EM \%) $\uparrow$}}}} \\
{metric_rows(metrics, methods, view, "EM")}
    \midrule
    \multicolumn{{6}}{{l}}{{\textbf{{Normalized Edit Similarity (NES \%) $\uparrow$}}}} \\
{metric_rows(metrics, methods, view, "NES")}
    \midrule
    \multicolumn{{6}}{{l}}{{\textbf{{Character Error Rate (CER \%) $\downarrow$}}}} \\
{metric_rows(metrics, methods, view, "CER")}
    \bottomrule
  \end{{tabularx}}
\end{{table}}"""


def coverage_table(metrics: Dict[str, Any], methods: Iterable[str]) -> str:
    rows = []
    for method in methods:
        coverage = metrics[method]["prediction_coverage"]
        sample_count = (
            metrics[method]["n_samples"]
            if "n_samples" in metrics[method]
            else metrics[method]["n_evaluated"]
        )
        rates = " & ".join(
            percent(coverage[field]["rate"]) for field in FIELDS[:-1]
        )
        rows.append(
            f"{METHOD_LATEX[method]} & {rates} & "
            f"{sample_count} & "
            f"{metrics[method]['n_inference_errors']} \\\\"
        )
    first_metrics = metrics[next(iter(methods))]
    present = (
        first_metrics["n_present"]
        if "n_present" in first_metrics
        else first_metrics["n_ground_truth_present"]
    )
    return rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Độ phủ prediction và số mẫu đánh giá. Số mẫu present-only lần lượt là {present["store_name"]}, {present["date"]}, {present["total"]} và {present["address"]} cho bốn trường.}}
  \label{{tab:coverage-results}}
  \small
  \begin{{tabular}}{{lcccccc}}
    \toprule
    \textbf{{Phương pháp}} & \textbf{{Tên CH}} & \textbf{{Ngày}} & \textbf{{Tổng}} & \textbf{{Địa chỉ}} & \textbf{{$n$}} & \textbf{{Lỗi suy luận}} \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def generate_results_tables(
    metrics: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    methods = valid_methods(manifest)
    all_table = metric_table(
        metrics,
        methods,
        view="all_samples",
        caption=(
            "Kết quả trên toàn bộ 234 mẫu kiểm thử của các artifact hợp lệ. "
            "Macro là trung bình không trọng số của bốn trường."
        ),
        label="tab:all-results",
    )
    present_table = metric_table(
        metrics,
        methods,
        view="present_only",
        caption=(
            "Kết quả present-only của các artifact hợp lệ: mỗi trường chỉ "
            "được chấm trên mẫu có ground-truth khác rỗng."
        ),
        label="tab:present-results",
    )
    return (
        "% Generated by scripts/generate_report_artifacts.py.\n"
        + all_table
        + "\n\n"
        + present_table
        + "\n\n"
        + coverage_table(metrics, methods)
        + "\n"
    )


def generate_analysis_tables(
    bootstrap: Dict[str, Any],
    sensitivity: Dict[str, Any],
) -> str:
    bootstrap_rows = []
    for metric in ("EM", "NES"):
        data = bootstrap["metrics"][metric]
        bootstrap_rows.append(
            f"{metric} & {percent(data['observed_difference'])} & "
            f"[{percent(data['ci95_percentile'][0])}; "
            f"{percent(data['ci95_percentile'][1])}] & "
            f"{percent(data['probability_positive'])} \\\\"
        )

    sensitivity_rows = []
    for method in ("baseline", "layoutxlm"):
        macro = sensitivity["metrics"][method]["present_only"]["macro"]
        sensitivity_rows.append(
            f"{METHOD_LATEX[method]} & {percent(macro['EM'])} & "
            f"{percent(macro['NES'])} & {percent(macro['CER'])} \\\\"
        )

    return rf"""% Generated by scripts/generate_report_artifacts.py.
\begin{{table}}[htbp]
  \centering
  \caption{{Paired bootstrap present-only cho chênh lệch LayoutXLM trừ Baseline, seed {bootstrap["seed"]}, {bootstrap["n_resamples"]:,} lần lấy mẫu.}}
  \label{{tab:paired-bootstrap}}
  \begin{{tabular}}{{lccc}}
    \toprule
    \textbf{{Metric}} & \textbf{{Chênh lệch (điểm \%)}} & \textbf{{Khoảng 95\%}} & \textbf{{$P(\Delta>0)$ (\%)}} \\
    \midrule
{chr(10).join(bootstrap_rows)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[htbp]
  \centering
  \caption{{Phân tích độ nhạy present-only sau khi loại bốn mẫu có sai lệch hệ tọa độ ảnh/OCR ($n={sensitivity["n_samples"]}$).}}
  \label{{tab:coordinate-sensitivity}}
  \begin{{tabular}}{{lccc}}
    \toprule
    \textbf{{Phương pháp}} & \textbf{{Macro EM (\%)}} & \textbf{{Macro NES (\%)}} & \textbf{{Macro CER (\%)}} \\
    \midrule
{chr(10).join(sensitivity_rows)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def generate_diagnostic_table(
    metrics: Dict[str, Any],
    latency: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    donut_metrics = metrics["donut"]
    donut_all = metric_view(donut_metrics, "all_samples")
    donut_manifest = manifest["methods"]["donut"]
    evidence = donut_manifest["truncation_evidence"]
    return rf"""% Generated by scripts/generate_report_artifacts.py.
\begin{{table}}[htbp]
  \centering
  \caption{{Thông tin bổ sung về artifact Donut (generation length analysis).}}
  \label{{tab:donut-diagnostic}}
  \small
  \begin{{tabularx}}{{\textwidth}}{{lX}}
    \toprule
    \textbf{{Quan sát}} & \textbf{{Giá trị}} \\
    \midrule
    Trạng thái & Hợp lệ cho so sánh định lượng, diễn giải như ca model collapse \\
    Checkpoint \code{{max\_length}} nếu không override & {donut_manifest["effective_generation_max_length"]} token ({latex_escape(donut_manifest["effective_generation_max_length_source"])}) \\
    Lượt suy luận thực tế & Truyền explicit \code{{max\_length}}={donut_manifest["configured_generation_max_length"]} token vào \code{{generate()}} \\
    Nhãn dài hơn fallback metadata & {evidence["n_targets_over_effective_max_length"]}/{manifest["gold"]["n_samples"]} mẫu; trung vị {evidence["target_token_length_median"]}, tối đa {evidence["target_token_length_max"]} token \\
    Output đúng bằng fallback metadata & {evidence["n_outputs_at_effective_max_length"]}/{donut_manifest["n_predictions"]} mẫu \\
    Macro EM all-samples & {percent(donut_all["macro"]["EM"])}\% \\
    Macro EM present-only & {percent(donut_metrics["present_only"]["macro"]["EM"])}\% \\
    Latency artifact & {decimal(latency["donut"]["e2e"]["mean_ms"])} ms/ảnh; phép đo bị ảnh hưởng bởi sinh tự hồi quy dài/lặp \\
    \bottomrule
  \end{{tabularx}}
\end{{table}}
"""


def generate_oracle_subset_table(manifest: Dict[str, Any]) -> str:
    gold = load_jsonl(manifest["gold"]["path"])
    subset_gold = [
        item
        for item in gold
        if item.get("source") == "mc_ocr_2021" and item.get("oracle_ocr")
    ]
    subset_ids = {item["id"] for item in subset_gold}
    method_paths = {
        "layoutxlm": manifest["methods"]["layoutxlm"]["prediction_path"],
        "layoutxlm_oracle": manifest["methods"]["layoutxlm_oracle"]["prediction_path"],
    }
    method_labels = {
        "layoutxlm": r"VietOCR + \model{LayoutXLM}",
        "layoutxlm_oracle": r"\model{LayoutXLM} (Oracle OCR)",
    }
    subset_predictions = {
        method: [
            item
            for item in load_jsonl(prediction_path)
            if item.get("id") in subset_ids
        ]
        for method, prediction_path in method_paths.items()
    }
    subset_metrics = {
        method: evaluate_predictions(predictions, subset_gold)
        for method, predictions in subset_predictions.items()
    }
    present = subset_metrics["layoutxlm"]["n_present"]
    metric_labels = {
        "EM": r"Exact Match (EM \%) $\uparrow$",
        "NES": r"Normalized Edit Similarity (NES \%) $\uparrow$",
        "CER": r"Character Error Rate (CER \%) $\downarrow$",
    }
    rows = []
    for metric in ("EM", "NES", "CER"):
        rows.append(
            rf"    \multicolumn{{6}}{{l}}{{\textbf{{{metric_labels[metric]}}}}} " + r"\\"
        )
        for method in ("layoutxlm", "layoutxlm_oracle"):
            values = " & ".join(
                percent(subset_metrics[method]["present_only"][field][metric])
                for field in FIELDS
            )
            rows.append(f"    {method_labels[method]} & {values} " + r"\\")
        if metric != "CER":
            rows.append(r"    \midrule")
    return rf"""% Generated by scripts/generate_report_artifacts.py.
\begin{{table}}[htbp]
  \centering
  \caption{{So sánh \model{{LayoutXLM}} với OCR thực tế và Oracle OCR trên {len(subset_gold)} mẫu MC-OCR có ground-truth OCR. Các số liệu là present-only; số mẫu theo bốn trường lần lượt là {present["store_name"]}, {present["date"]}, {present["total"]} và {present["address"]}.}}
  \label{{tab:oracle-subset}}
  \small
  \begin{{tabularx}}{{\textwidth}}{{l C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.7cm}}}}
    \toprule
    \textbf{{Phương pháp}} & \textbf{{Tên CH}} & \textbf{{Ngày}} & \textbf{{Tổng}} & \textbf{{Địa chỉ}} & \textbf{{Macro}} \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabularx}}
\end{{table}}
"""


def generate_latency_table(
    latency: Dict[str, Any],
    manifest: Dict[str, Any],
) -> str:
    methods = valid_methods(manifest)
    scope_labels = {
        "baseline": "Hậu xử lý từ OCR cache",
        "layoutxlm": "Suy luận và hậu xử lý từ OCR cache",
        "donut": "Suy luận trực tiếp từ ảnh đầu vào",
    }
    # Split into post-OCR methods and end-to-end methods
    post_ocr_methods = [m for m in methods if m in ("baseline", "layoutxlm")]
    e2e_methods = [m for m in methods if m in ("donut",)]

    post_ocr_rows = [
        f"{METHOD_LATEX[method]} & {scope_labels[method]} & "
        f"{decimal(latency[method]['e2e']['mean_ms'])} \\\\"
        for method in post_ocr_methods
    ]
    table_a = rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Latency hậu xử lý từ OCR cache. Phép đo bắt đầu sau khi OCR cache đã sẵn sàng; \textbf{{chưa bao gồm thời gian chạy PaddleOCR + VietOCR từ ảnh thô}}.}}
  \label{{tab:latency-post-ocr}}
  \begin{{tabular}}{{llr}}
    \toprule
    \textbf{{Phương pháp}} & \textbf{{Phạm vi phép đo}} & \textbf{{Trung bình (ms/ảnh)}} \\
    \midrule
{chr(10).join(post_ocr_rows)}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""

    e2e_rows = [
        f"{METHOD_LATEX[method]} & {scope_labels[method]} & "
        f"{decimal(latency[method]['e2e']['mean_ms'])} \\\\"
        for method in e2e_methods
    ]
    table_b = rf"""\begin{{table}}[htbp]
  \centering
  \caption{{Latency suy luận trực tiếp từ ảnh thô. Phép đo này \textbf{{không so sánh ngang hàng}} với bảng post-OCR cache vì phạm vi đo hoàn toàn khác nhau.}}
  \label{{tab:latency-e2e}}
  \begin{{tabular}}{{llr}}
    \toprule
    \textbf{{Phương pháp}} & \textbf{{Phạm vi phép đo}} & \textbf{{Trung bình (ms/ảnh)}} \\
    \midrule
{chr(10).join(e2e_rows)}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""

    return (
        "% Generated by scripts/generate_report_artifacts.py.\n"
        + table_a
        + "\n\n"
        + table_b
        + "\n"
    )


def generate_validity_table(manifest: Dict[str, Any]) -> str:
    rows = []
    statuses = {
        "baseline": "Hợp lệ",
        "layoutxlm": "Hợp lệ, kèm sensitivity check",
        "donut": "Hợp lệ, diễn giải như model collapse",
    }
    uses = {
        "baseline": "Bảng chính, RQ1, latency thành phần",
        "layoutxlm": "Bảng chính, RQ1, latency thành phần",
        "donut": "Bảng chính, RQ1, phụ lục chẩn đoán",
    }
    for method in ("baseline", "layoutxlm", "donut"):
        item = manifest["methods"][method]
        rows.append(
            f"{METHOD_LATEX[method]} & {statuses[method]} & "
            f"{item['n_predictions']} & "
            rf"\code{{{item['prediction_sha256'][:10]}\ldots}} & "
            f"{uses[method]} \\\\"
        )
    return rf"""% Generated by scripts/generate_report_artifacts.py.
\begin{{table}}[htbp]
  \centering
  \caption{{Trạng thái bằng chứng của các prediction artifact đã khóa bằng SHA-256.}}
  \label{{tab:artifact-validity}}
  \small
  \begin{{tabularx}}{{\textwidth}}{{lL{{3cm}}cL{{2.7cm}}X}}
    \toprule
    \textbf{{Artifact}} & \textbf{{Trạng thái}} & \textbf{{$n$}} & \textbf{{SHA-256}} & \textbf{{Phạm vi sử dụng}} \\
    \midrule
{chr(10).join(rows)}
    \bottomrule
  \end{{tabularx}}
\end{{table}}
"""


def generate_jsonl_sample(gold_path: str | Path) -> str:
    first = next(
        json.loads(line)
        for line in Path(gold_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    keys = [
        "id",
        "group_id",
        "store_group",
        "source",
        "image_path",
        "width",
        "height",
        "annotation_level",
        "raw_target",
        "target",
    ]
    sample = {key: first.get(key) for key in keys if key in first}
    body = json.dumps(sample, ensure_ascii=False, indent=2)
    return (
        "% Generated from the first row of the current processed test JSONL.\n"
        "\\begin{lstlisting}[style=json, caption={Một mẫu thực tế trong "
        "processed test JSONL}, label={lst:jsonl-sample}]\n"
        f"{body}\n"
        "\\end{lstlisting}\n"
    )


def main() -> None:
    metrics = load_json("outputs/metrics/combined_metrics.json")
    latency = load_json("outputs/metrics/latency_by_method.json")
    manifest = load_json("outputs/metrics/artifact_manifest.json")
    bootstrap = load_json("outputs/metrics/paired_bootstrap.json")
    sensitivity = load_json("outputs/metrics/sensitivity_230.json")
    output_dir = Path("report/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {
        "results_tables.tex": generate_results_tables(metrics, manifest),
        "analysis_tables.tex": generate_analysis_tables(
            bootstrap,
            sensitivity,
        ),
        "diagnostic_tables.tex": generate_diagnostic_table(
            metrics,
            latency,
            manifest,
        ),
        "oracle_subset_table.tex": generate_oracle_subset_table(manifest),
        "latency_table.tex": generate_latency_table(latency, manifest),
        "artifact_validity_table.tex": generate_validity_table(manifest),
        "jsonl_sample.tex": generate_jsonl_sample(manifest["gold"]["path"]),
    }
    for filename, content in generated.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    print(f"Generated {len(generated)} report artifacts in {output_dir}")


if __name__ == "__main__":
    main()
