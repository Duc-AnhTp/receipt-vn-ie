import json
import tempfile
import unittest
from pathlib import Path
from receipt_ie.data.build_donut_dataset import target_to_donut_sequence, donut_sequence_to_target
from receipt_ie.inference.infer_donut import checkpoint_generation_max_length

class TestDonutDataConversion(unittest.TestCase):
    def setUp(self):
        self.task_token = "<s_receipt_ie>"
        self.cord_task_token = "<s_cord_receipt_parse>"
        self.sample_target = {
            "store_name": "MINIMART ANAN",
            "date": "2020-08-09",
            "total": "115000",
            "address": "Chợ Sủi Phú Thị Gia Lâm"
        }

    def test_target_to_donut_sequence_viet(self):
        seq = target_to_donut_sequence(self.sample_target, self.task_token)
        expected = (
            "<s_receipt_ie>"
            "<s_store_name>MINIMART ANAN</s_store_name>"
            "<s_date>2020-08-09</s_date>"
            "<s_total>115000</s_total>"
            "<s_address>Chợ Sủi Phú Thị Gia Lâm</s_address>"
            "</s_receipt_ie>"
        )
        self.assertEqual(seq, expected)

    def test_target_to_donut_sequence_cord(self):
        # CORD warm-up chỉ trích xuất total
        seq = target_to_donut_sequence(self.sample_target, self.cord_task_token)
        expected = "<s_cord_receipt_parse><s_total>115000</s_total></s_cord_receipt_parse>"
        self.assertEqual(seq, expected)

    def test_donut_sequence_to_target_perfect(self):
        seq = (
            "<s_receipt_ie>"
            "<s_store_name>MINIMART ANAN</s_store_name>"
            "<s_date>2020-08-09</s_date>"
            "<s_total>115000</s_total>"
            "<s_address>Chợ Sủi Phú Thị Gia Lâm</s_address>"
            "</s_receipt_ie>"
        )
        parsed = donut_sequence_to_target(seq, self.task_token)
        self.assertEqual(parsed["store_name"], "MINIMART ANAN")
        self.assertEqual(parsed["date"], "2020-08-09")
        self.assertEqual(parsed["total"], "115000")
        self.assertEqual(parsed["address"], "Chợ Sủi Phú Thị Gia Lâm")

    def test_donut_sequence_to_target_broken_tags(self):
        # Trường hợp mô hình sinh thiếu tag đóng (ví dụ bị cắt cụt do max_length)
        seq_truncated = (
            "<s_receipt_ie>"
            "<s_store_name>MINIMART ANAN</s_store_name>"
            "<s_date>2020-08-09</s_date>"
            "<s_total>115000</s_total>"
            "<s_address>Chợ Sủi Phú Thị"
        )
        parsed = donut_sequence_to_target(seq_truncated, self.task_token)
        self.assertEqual(parsed["store_name"], "MINIMART ANAN")
        self.assertEqual(parsed["date"], "2020-08-09")
        self.assertEqual(parsed["total"], "115000")
        # Hệ thống fallback regex phải trích xuất được phần text của thẻ chưa đóng cho đến hết chuỗi
        self.assertEqual(parsed["address"], "Chợ Sủi Phú Thị")

        # Trường hợp mô hình sinh liền kề tag mở khác mà không đóng tag cũ
        seq_no_close = (
            "<s_receipt_ie>"
            "<s_store_name>MINIMART ANAN"
            "<s_date>2020-08-09</s_date>"
            "<s_total>115000"
            "<s_address>Gia Lâm"
        )
        parsed_no_close = donut_sequence_to_target(seq_no_close, self.task_token)
        self.assertEqual(parsed_no_close["store_name"], "MINIMART ANAN")
        self.assertEqual(parsed_no_close["date"], "2020-08-09")
        self.assertEqual(parsed_no_close["total"], "115000")
        self.assertEqual(parsed_no_close["address"], "Gia Lâm")

    def test_checkpoint_generation_limit_detects_implicit_default(self):
        with tempfile.TemporaryDirectory(dir=r"C:\tmp") as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                json.dumps({"model_type": "vision-encoder-decoder"}),
                encoding="utf-8",
            )
            value, source = checkpoint_generation_max_length(str(checkpoint))
            self.assertEqual(value, 20)
            self.assertEqual(source, "transformers_default")

    def test_checkpoint_generation_limit_prefers_generation_config(self):
        with tempfile.TemporaryDirectory(dir=r"C:\tmp") as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(
                json.dumps({"max_length": 30}),
                encoding="utf-8",
            )
            (checkpoint / "generation_config.json").write_text(
                json.dumps({"max_length": 768}),
                encoding="utf-8",
            )
            value, source = checkpoint_generation_max_length(str(checkpoint))
            self.assertEqual(value, 768)
            self.assertEqual(source, "generation_config.json")


if __name__ == "__main__":
    unittest.main()
