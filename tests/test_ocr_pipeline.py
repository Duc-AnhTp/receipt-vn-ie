import torch
import os
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from receipt_ie.ocr.detect_paddle import load_paddle_detector, detect_text_regions, crop_region
from receipt_ie.ocr.recognize_vietocr import load_vietocr_model, recognize_regions
from receipt_ie.ocr.reading_order import sort_reading_order


class TestOcrPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tạo ảnh mock chứa chữ vẽ đơn giản
        cls.mock_image_path = Path("tests/mock_receipt.png")
        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)
        # Vẽ một vài hình chữ nhật màu đen giả lập vùng chữ
        draw.rectangle([10, 10, 100, 30], fill="black")
        draw.rectangle([120, 12, 220, 32], fill="black")
        draw.rectangle([15, 50, 115, 70], fill="black")
        img.save(cls.mock_image_path)
        img.close()

    @classmethod
    def tearDownClass(cls):
        cls.mock_image_path.unlink(missing_ok=True)

    def test_crop_region(self):
        with Image.open(self.mock_image_path) as img:
            bbox = [10, 10, 100, 30]
            cropped = crop_region(img, bbox, padding=2)
            
            self.assertIsInstance(cropped, Image.Image)
            self.assertEqual(cropped.size[0], 90 + 4) # 94 do padding=2 cả 2 bên

    def test_reading_order_integration(self):
        regions = [
            {"bbox": [120, 12, 220, 32], "text": "Right"},
            {"bbox": [10, 10, 100, 30], "text": "Left"},
        ]
        flat, grouped = sort_reading_order(regions, y_threshold=10)
        self.assertEqual(len(flat), 2)
        self.assertEqual(flat[0]["text"], "Left")
        self.assertEqual(flat[1]["text"], "Right")


@unittest.skipUnless(os.environ.get("RUN_INTEGRATION_TESTS") == "1", "Requires RUN_INTEGRATION_TESTS=1")
class TestOcrPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tạo ảnh mock cho integration tests
        cls.mock_image_path = Path("tests/mock_receipt.png")
        if not cls.mock_image_path.exists():
            img = Image.new("RGB", (400, 200), color="white")
            draw = ImageDraw.Draw(img)
            draw.rectangle([10, 10, 100, 30], fill="black")
            draw.rectangle([120, 12, 220, 32], fill="black")
            draw.rectangle([15, 50, 115, 70], fill="black")
            img.save(cls.mock_image_path)
            img.close()

        # Khởi tạo các engine thật
        cls.detector = load_paddle_detector(use_gpu=False, use_angle_cls=False, lang="vi")
        cls.recognizer = load_vietocr_model(config_name="vgg_transformer", use_gpu=False)

    @classmethod
    def tearDownClass(cls):
        cls.mock_image_path.unlink(missing_ok=True)

    def test_detection(self):
        regions = detect_text_regions(self.detector, str(self.mock_image_path))
        self.assertIsInstance(regions, list)
        for r in regions:
            self.assertIn("bbox", r)
            self.assertIn("polygon", r)
            self.assertEqual(len(r["bbox"]), 4)
            self.assertEqual(len(r["polygon"]), 4)

    def test_recognition(self):
        with Image.open(self.mock_image_path) as img:
            bbox = [10, 10, 100, 30]
            cropped = crop_region(img, bbox, padding=2)
            
            # Test recognition
            texts = recognize_regions(self.recognizer, [cropped])
            self.assertEqual(len(texts), 1)
            self.assertIsInstance(texts[0], str)


if __name__ == "__main__":
    unittest.main()
