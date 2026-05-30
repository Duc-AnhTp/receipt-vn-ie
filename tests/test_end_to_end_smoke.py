import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw

from receipt_ie.inference.infer_baseline import BaselineExtractor
from receipt_ie.inference.mock_extractor import MockExtractor


class TestEndToEndSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tạo ảnh mock chứa thông tin đơn giản
        cls.mock_image_path = Path("tests/mock_smoke_receipt.png")
        img = Image.new("RGB", (300, 150), color="white")
        draw = ImageDraw.Draw(img)
        # Giả lập vẽ vài dòng chữ màu đen
        draw.rectangle([5, 5, 80, 20], fill="black")
        draw.rectangle([100, 8, 200, 22], fill="black")
        img.save(cls.mock_image_path)
        img.close()

    @classmethod
    def tearDownClass(cls):
        cls.mock_image_path.unlink(missing_ok=True)

    def setUp(self):
        self.temp_project = tempfile.TemporaryDirectory()

        # Patch các hàm OCR trong cache_manager để chạy hoàn toàn offline
        self.patch_load_paddle = patch("receipt_ie.inference.infer_baseline.load_paddle_detector")
        self.patch_load_vietocr = patch("receipt_ie.inference.infer_baseline.load_vietocr_model")
        self.patch_detect = patch("receipt_ie.ocr.cache_manager.detect_text_regions")
        self.patch_recognize = patch("receipt_ie.ocr.cache_manager.recognize_regions")

        self.mock_load_paddle = self.patch_load_paddle.start()
        self.mock_load_vietocr = self.patch_load_vietocr.start()
        self.mock_detect = self.patch_detect.start()
        self.mock_recognize = self.patch_recognize.start()

        # Giả lập hoạt động của các engine
        self.mock_load_paddle.return_value = MagicMock()
        self.mock_load_vietocr.return_value = MagicMock()

        # Mock OCR trả về đủ 4 trường mục tiêu để kiểm tra logic heuristics
        self.mock_detect.return_value = [
            {"bbox": [10, 10, 100, 30], "polygon": [[10, 10], [100, 10], [100, 30], [10, 30]]},
            {"bbox": [10, 40, 100, 60], "polygon": [[10, 40], [100, 40], [100, 60], [10, 60]]},
            {"bbox": [10, 70, 100, 90], "polygon": [[10, 70], [100, 70], [100, 90], [10, 90]]},
            {"bbox": [10, 100, 100, 120], "polygon": [[10, 100], [100, 100], [100, 120], [10, 120]]},
        ]
        self.mock_recognize.return_value = [
            "Cửa hàng Circle K",
            "Địa chỉ: 123 Đường Nguyễn Trãi, Quận Thanh Xuân, Hà Nội",
            "Ngày: 22/05/2026",
            "Tổng cộng: 150.000đ",
        ]

    def tearDown(self):
        self.patch_load_paddle.stop()
        self.patch_load_vietocr.stop()
        self.patch_detect.stop()
        self.patch_recognize.stop()
        self.temp_project.cleanup()

    def test_baseline_e2e(self):
        # Test BaselineExtractor chạy e2e thật sự trên ảnh mock (đã được mock OCR)
        extractor = BaselineExtractor(ocr_config_path="configs/ocr.yaml", project_root=self.temp_project.name)
        extractor.load()
        
        with Image.open(self.mock_image_path) as img:
            res = extractor.predict(img)
        
        # Xác nhận cấu trúc trả về
        self.assertIn("prediction", res)
        self.assertIn("normalized_prediction", res)
        self.assertIn("latency_ocr_ms", res)
        self.assertIn("latency_model_ms", res)
        self.assertIn("latency_postprocess_ms", res)
        self.assertIn("latency_e2e_ms", res)
        self.assertIn("status", res)
        self.assertEqual(res["status"], "ok")


        # Kiểm tra nội dung trích xuất
        norm_pred = res["normalized_prediction"]
        self.assertEqual(norm_pred["store_name"], "Circle K")
        self.assertEqual(norm_pred["address"], "123 Đường Nguyễn Trãi, Quận Thanh Xuân, Hà Nội")
        self.assertEqual(norm_pred["date"], "2026-05-22")
        self.assertEqual(norm_pred["total"], "150000")

    def test_donut_mock_e2e(self):
        # Test Mock Donut Extractor dùng trong Gradio app
        baseline = BaselineExtractor(ocr_config_path="configs/ocr.yaml", project_root=self.temp_project.name)
        baseline.load()
        
        mock_donut = MockExtractor("donut", baseline)
        with Image.open(self.mock_image_path) as img:
            res = mock_donut.predict(img)
        
        self.assertIn("prediction", res)
        self.assertIn("normalized_prediction", res)
        self.assertIn("raw_output", res)
        self.assertTrue(res.get("is_mock"))
        self.assertEqual(res["status"], "ok")

        norm_pred = res["normalized_prediction"]
        self.assertEqual(norm_pred["store_name"], "Circle K")
        self.assertEqual(norm_pred["address"], "123 Đường Nguyễn Trãi, Quận Thanh Xuân, Hà Nội")
        self.assertEqual(norm_pred["date"], "2026-05-22")
        self.assertEqual(norm_pred["total"], "150000")

    def test_layoutxlm_mock_e2e(self):
        # Test Mock LayoutXLM Extractor dùng trong Gradio app
        baseline = BaselineExtractor(ocr_config_path="configs/ocr.yaml", project_root=self.temp_project.name)
        baseline.load()
        
        mock_layout = MockExtractor("layoutxlm", baseline)
        with Image.open(self.mock_image_path) as img:
            res = mock_layout.predict(img)
        
        self.assertIn("prediction", res)
        self.assertIn("normalized_prediction", res)
        self.assertIn("words", res)
        self.assertIn("word_labels", res)
        self.assertTrue(res.get("is_mock"))
        self.assertEqual(res["status"], "ok")

        norm_pred = res["normalized_prediction"]
        self.assertEqual(norm_pred["store_name"], "Circle K")
        self.assertEqual(norm_pred["address"], "123 Đường Nguyễn Trãi, Quận Thanh Xuân, Hà Nội")
        self.assertEqual(norm_pred["date"], "2026-05-22")
        self.assertEqual(norm_pred["total"], "150000")


if __name__ == "__main__":
    unittest.main()
