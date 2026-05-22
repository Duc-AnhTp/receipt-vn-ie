import os
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from receipt_ie.inference.pipeline import get_extractor
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

    @classmethod
    def tearDownClass(cls):
        if cls.mock_image_path.exists():
            os.remove(cls.mock_image_path)

    def test_baseline_e2e(self):
        # Test BaselineExtractor chạy e2e thật sự trên ảnh mock
        extractor = BaselineExtractor(ocr_config_path="configs/ocr.yaml", project_root=".")
        extractor.load()
        
        img = Image.open(self.mock_image_path)
        res = extractor.predict(img)
        
        # Xác nhận cấu trúc trả về
        self.assertIn("prediction", res)
        self.assertIn("normalized_prediction", res)
        self.assertIn("latency_cached_ms", res)
        self.assertIn("latency_e2e_ms", res)
        self.assertIn("status", res)
        self.assertIn("words", res)
        self.assertEqual(res["status"], "ok")

    def test_donut_mock_e2e(self):
        # Test Mock Donut Extractor dùng trong Gradio app
        baseline = BaselineExtractor(ocr_config_path="configs/ocr.yaml", project_root=".")
        baseline.load()
        
        mock_donut = MockExtractor("donut", baseline)
        img = Image.open(self.mock_image_path)
        res = mock_donut.predict(img)
        
        self.assertIn("prediction", res)
        self.assertIn("normalized_prediction", res)
        self.assertIn("raw_output", res)
        self.assertTrue(res.get("is_mock"))
        self.assertEqual(res["status"], "ok")

    def test_layoutxlm_mock_e2e(self):
        # Test Mock LayoutXLM Extractor dùng trong Gradio app
        baseline = BaselineExtractor(ocr_config_path="configs/ocr.yaml", project_root=".")
        baseline.load()
        
        mock_layout = MockExtractor("layoutxlm", baseline)
        img = Image.open(self.mock_image_path)
        res = mock_layout.predict(img)
        
        self.assertIn("prediction", res)
        self.assertIn("normalized_prediction", res)
        self.assertIn("words", res)
        self.assertIn("word_labels", res)
        self.assertTrue(res.get("is_mock"))
        self.assertEqual(res["status"], "ok")


if __name__ == "__main__":
    unittest.main()
