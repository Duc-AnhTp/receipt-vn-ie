import unittest

from receipt_ie.metrics.comparison_analysis import (
    paired_bootstrap_present_only,
    sensitivity_analysis,
)


class TestComparisonAnalysis(unittest.TestCase):
    def setUp(self):
        self.gold = [
            {
                "id": "a",
                "target": {
                    "store_name": "A",
                    "date": "2026-01-01",
                    "total": "100",
                    "address": "HN",
                },
            },
            {
                "id": "b",
                "target": {
                    "store_name": "B",
                    "date": "",
                    "total": "200",
                    "address": "HCM",
                },
            },
        ]
        self.baseline = [
            {
                "id": item["id"],
                "normalized_prediction": {
                    "store_name": "",
                    "date": "",
                    "total": "",
                    "address": "",
                },
            }
            for item in self.gold
        ]
        self.layoutxlm = [
            {
                "id": item["id"],
                "normalized_prediction": item["target"],
            }
            for item in self.gold
        ]

    def test_bootstrap_is_deterministic(self):
        first = paired_bootstrap_present_only(
            self.gold,
            self.baseline,
            self.layoutxlm,
            seed=123,
            n_resamples=500,
            batch_size=100,
        )
        second = paired_bootstrap_present_only(
            self.gold,
            self.baseline,
            self.layoutxlm,
            seed=123,
            n_resamples=500,
            batch_size=100,
        )
        self.assertEqual(first, second)
        self.assertGreater(
            first["metrics"]["EM"]["observed_difference"],
            0,
        )

    def test_sensitivity_excludes_requested_id(self):
        result = sensitivity_analysis(
            self.gold,
            {
                "baseline": self.baseline,
                "layoutxlm": self.layoutxlm,
            },
            ["b"],
        )
        self.assertEqual(result["n_samples"], 1)
        self.assertEqual(result["excluded_ids"], ["b"])


if __name__ == "__main__":
    unittest.main()
