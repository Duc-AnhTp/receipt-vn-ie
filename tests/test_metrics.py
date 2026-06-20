import unittest
from receipt_ie.metrics.evaluate_fields import compute_em, compute_nes, compute_cer, evaluate_predictions
from receipt_ie.metrics.structure_validity import check_donut_structure, compute_structure_validity_rates
from receipt_ie.metrics.latency import compute_latency_stats

class TestMetrics(unittest.TestCase):
    def test_evaluate_fields_math(self):
        # 1. Test EM
        self.assertEqual(compute_em("MINIMART ANAN", "MINIMART ANAN"), 1.0)
        self.assertEqual(compute_em("minimart anan", "MINIMART ANAN"), 0.0) # Khác case
        self.assertEqual(compute_em("ANAN", "ANAN "), 1.0) # Tự động strip

        # 2. Test NES
        # Levenshtein distance của "ANAN" và "ANAM" là 1
        # max length = 4 -> NES = 1 - 1/4 = 0.75
        self.assertAlmostEqual(compute_nes("ANAN", "ANAM"), 0.75)
        self.assertEqual(compute_nes("", ""), 1.0)
        self.assertEqual(compute_nes("A", ""), 0.0)

        # 3. Test CER
        # Levenshtein distance của "ABC" và "AB" là 1
        # len(gold) = 2 -> CER = 1/2 = 0.5
        self.assertAlmostEqual(compute_cer("ABC", "AB"), 0.5)
        # len(gold) = 3 -> CER = 1/3 = 0.3333
        self.assertAlmostEqual(compute_cer("AB", "ABC"), 1/3)
        self.assertEqual(compute_cer("", ""), 0.0)

    def test_evaluate_predictions_batch(self):
        predictions = [
            {"store_name": "ANAN", "date": "2020-08-09", "total": "100000", "address": "Hà Nội"},
            {"store_name": "MINI", "date": "2020-08-10", "total": "200000", "address": "HCM"}
        ]
        ground_truths = [
            {"store_name": "ANAM", "date": "2020-08-09", "total": "100000", "address": "Hà Nội"},
            {"store_name": "MINI", "date": "2020-08-10", "total": "250000", "address": "Đà Nẵng"}
        ]
        
        results = evaluate_predictions(predictions, ground_truths)
        
        # Kiểm tra sự tồn tại của các trường trong kết quả
        for key in ["store_name", "date", "total", "address", "macro"]:
            self.assertIn(key, results)
            self.assertIn("EM", results[key])
            self.assertIn("NES", results[key])
            self.assertIn("CER", results[key])
            
        # Kiểm tra giá trị cụ thể
        # date: 2/2 mẫu khớp hoàn toàn -> EM = 1.0
        self.assertEqual(results["date"]["EM"], 1.0)
        # store_name: 1 mẫu khớp hoàn toàn ("MINI"), 1 mẫu lệch ("ANAN" vs "ANAM") -> EM = 0.5
        self.assertEqual(results["store_name"]["EM"], 0.5)

    def test_evaluate_predictions_by_id_counts_missing_as_wrong(self):
        ground_truths = [
            {
                "id": "a",
                "target": {
                    "store_name": "Store A",
                    "date": "2026-05-21",
                    "total": "100000",
                    "address": "Ha Noi",
                },
            },
            {
                "id": "b",
                "target": {
                    "store_name": "Store B",
                    "date": "2026-05-22",
                    "total": "200000",
                    "address": "HCM",
                },
            },
        ]
        predictions = [
            {
                "id": "b",
                "status": "ok",
                "normalized_prediction": {
                    "store_name": "Store B",
                    "date": "2026-05-22",
                    "total": "200000",
                    "address": "HCM",
                },
            },
            {
                "id": "a",
                "status": "error",
                "normalized_prediction": {
                    "store_name": "Store A",
                    "date": "2026-05-21",
                    "total": "100000",
                    "address": "Ha Noi",
                },
            },
        ]

        results = evaluate_predictions(predictions, ground_truths)

        self.assertEqual(results["n_evaluated"], 2)
        self.assertEqual(results["n_skipped_error"], 0)
        self.assertEqual(results["n_inference_errors"], 1)
        self.assertEqual(results["missing_prediction_ids"], [])
        self.assertEqual(results["store_name"]["EM"], 0.5)
        self.assertEqual(results["date"]["EM"], 0.5)
        self.assertEqual(results["total"]["EM"], 0.5)
        self.assertEqual(results["address"]["EM"], 0.5)

    def test_present_only_and_prediction_coverage(self):
        ground_truths = [
            {"id": "a", "target": {"store_name": "", "date": "", "total": "", "address": ""}},
            {"id": "b", "target": {"store_name": "B", "date": "2026-05-22", "total": "100", "address": "HN"}},
        ]
        predictions = [
            {"id": "a", "status": "ok", "normalized_prediction": {"store_name": "", "date": "", "total": "", "address": ""}},
            {"id": "b", "status": "ok", "normalized_prediction": {"store_name": "", "date": "", "total": "", "address": ""}},
        ]

        results = evaluate_predictions(predictions, ground_truths)

        self.assertEqual(results["date"]["EM"], 0.5)
        self.assertEqual(results["present_only"]["date"]["EM"], 0.0)
        self.assertEqual(results["present_only"]["date"]["n_samples"], 1)
        self.assertEqual(results["prediction_coverage"]["date"]["rate"], 0.0)
        self.assertEqual(results["macro"]["averaging"], "unweighted_mean_over_fields")

    def test_missing_prediction_id_is_scored_as_empty(self):
        ground_truths = [
            {"id": "a", "target": {"store_name": "A", "date": "2026-05-22", "total": "100", "address": "HN"}},
        ]
        results = evaluate_predictions([], ground_truths)
        self.assertEqual(results["store_name"]["EM"], 0.0)
        self.assertEqual(results["n_evaluated"], 1)
        self.assertEqual(results["missing_prediction_ids"], ["a"])

    def test_structure_validity(self):
        task_token = "<s_receipt_ie>"
        
        # 1. Chuỗi chuẩn
        seq_valid = (
            "<s_receipt_ie>"
            "<s_store_name>ANAN</s_store_name>"
            "<s_date>2020-08-09</s_date>"
            "<s_total>1000</s_total>"
            "<s_address>HN</s_address>"
            "</s_receipt_ie>"
        )
        res_valid = check_donut_structure(seq_valid, task_token)
        self.assertTrue(res_valid["overall_valid"])
        self.assertTrue(res_valid["task_tokens_valid"])
        self.assertTrue(res_valid["field_tags_valid"])
        
        # 2. Chuỗi thiếu tag đóng task
        seq_no_task_end = (
            "<s_receipt_ie>"
            "<s_store_name>ANAN</s_store_name>"
        )
        res_no_task_end = check_donut_structure(seq_no_task_end, task_token)
        self.assertFalse(res_no_task_end["overall_valid"])
        self.assertFalse(res_no_task_end["task_tokens_valid"])
        
        # 3. Chuỗi thiếu tag đóng trường con
        seq_no_field_end = (
            "<s_receipt_ie>"
            "<s_store_name>ANAN"
            "<s_date>2020-08-09</s_date>"
            "</s_receipt_ie>"
        )
        res_no_field_end = check_donut_structure(seq_no_field_end, task_token)
        self.assertFalse(res_no_field_end["overall_valid"])
        self.assertFalse(res_no_field_end["field_tags_valid"])

        # 4. Tỷ lệ hợp lệ
        seqs = [seq_valid, seq_no_task_end, seq_no_field_end]
        rates = compute_structure_validity_rates(seqs, task_token)
        self.assertEqual(rates["overall_valid_rate"], 0.3333)

    def test_latency_stats(self):
        latencies = [100.0, 150.0, 200.0, 250.0, 300.0]
        stats = compute_latency_stats(latencies)
        
        self.assertEqual(stats["mean_ms"], 200.0)
        self.assertEqual(stats["median_ms"], 200.0)
        self.assertTrue(stats["p90_ms"] > 200.0)
        self.assertTrue(stats["p99_ms"] >= stats["p95_ms"])

if __name__ == "__main__":
    unittest.main()
