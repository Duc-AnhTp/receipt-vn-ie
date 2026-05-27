import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image
import tempfile
import json
from pathlib import Path

from receipt_ie.data.build_layoutxlm_dataset import LayoutXLMDataset, layoutxlm_collate_fn

class TestLayoutXlmDatasetImage(unittest.TestCase):
    def setUp(self):
        # Tạo thư mục tạm và cấu trúc dự án
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        
        # Tạo một file ảnh mock
        self.image_path = self.project_root / "test_image.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(self.image_path)
        img.close()
        
        # Tạo dữ liệu JSONL mẫu
        self.jsonl_path = self.project_root / "test.jsonl"
        self.sample_data = {
            "id": "test_sample_01",
            "image_path": "test_image.png",
            "width": 100,
            "height": 100,
            "annotation_level": "json_and_boxes",
            "oracle_ocr": [
                {"text": "Cửa", "box": [10, 10, 30, 20]},
                {"text": "Hàng", "box": [35, 10, 60, 20]}
            ],
            "field_boxes": {
                "store_name": [[10, 10, 60, 20]]
            }
        }
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.sample_data) + "\n")
            
        # Mock Tokenizer và Image Processor để chạy hoàn toàn offline
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.bos_token_id = 0
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.pad_token_id = 1
        self.mock_tokenizer.tokenize.side_effect = lambda x: [x]
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: [100]
        
        self.mock_image_processor = MagicMock()
        mock_pixel_values = torch.zeros(3, 224, 224)
        self.mock_image_processor.return_value = MagicMock(pixel_values=[mock_pixel_values])

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("transformers.LayoutLMv2ImageProcessor.from_pretrained")
    def test_dataset_outputs_image(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self.mock_image_processor
        
        # Khởi tạo dataset ở chế độ oracle_ocr
        dataset = LayoutXLMDataset(
            jsonl_path=str(self.jsonl_path),
            tokenizer=self.mock_tokenizer,
            mode="oracle_ocr",
            max_length=10,
            project_root=str(self.project_root),
            annotation_level_filter="json_and_boxes"
        )
        
        self.assertEqual(len(dataset), 1)
        
        # Lấy sample đầu tiên
        item = dataset[0]
        
        # Xác nhận các trường đầu ra
        self.assertIn("input_ids", item)
        self.assertIn("bbox", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertIn("image", item) # Phải có trường này
        self.assertIn("id", item)
        
        # Kiểm tra kích thước tensor ảnh
        self.assertEqual(item["image"].shape, (3, 224, 224))
        self.assertEqual(item["id"], "test_sample_01")
        
        # Kiểm tra chức năng ghép batch (collate)
        batch = layoutxlm_collate_fn([item, item])
        self.assertIn("image", batch)
        self.assertEqual(batch["image"].shape, (2, 3, 224, 224))
        self.assertEqual(batch["input_ids"].shape, (2, 10))

if __name__ == "__main__":
    unittest.main()
