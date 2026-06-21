import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_report_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("report_artifacts", SCRIPT_PATH)
REPORT_ARTIFACTS = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(REPORT_ARTIFACTS)


class TestReportArtifacts(unittest.TestCase):
    def test_invalid_donut_is_not_in_main_results(self):
        metrics = {}
        for method in ("baseline", "layoutxlm", "donut"):
            metrics[method] = {}
            for view in ("all_samples", "present_only"):
                metrics[method][view] = {
                    field: {"EM": 0.0, "NES": 0.0, "CER": 0.0}
                    for field in REPORT_ARTIFACTS.FIELDS
                }
            metrics[method]["prediction_coverage"] = {
                field: {"rate": 0.0}
                for field in REPORT_ARTIFACTS.FIELDS[:-1]
            }
            metrics[method]["n_samples"] = 1
            metrics[method]["n_inference_errors"] = 0
            metrics[method]["n_present"] = {
                field: 1 for field in REPORT_ARTIFACTS.FIELDS[:-1]
            }
        manifest = {
            "methods": {
                "baseline": {"valid_for_main_comparison": True},
                "layoutxlm": {"valid_for_main_comparison": True},
                "donut": {"valid_for_main_comparison": False},
            }
        }
        output = REPORT_ARTIFACTS.generate_results_tables(metrics, manifest)
        self.assertNotIn(r"\model{Donut}", output)
        self.assertIn(r"\model{LayoutXLM}", output)


if __name__ == "__main__":
    unittest.main()
