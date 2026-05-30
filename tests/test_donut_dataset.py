import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from receipt_ie.models.donut_dataset import DonutDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDonutDatasetStrictImage(unittest.TestCase):
    def test_missing_image_raises_when_strict(self):
        tmp_path = PROJECT_ROOT / "tmp" / "test_donut_dataset"
        tmp_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = tmp_path / "data.jsonl"
        sample = {
            "id": "missing",
            "image_path": "missing.png",
            "target": {"store_name": "", "date": "", "total": "", "address": ""},
        }
        jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        processor = MagicMock()
        processor.image_processor.size = {"width": 64, "height": 64}
        dataset = DonutDataset(str(jsonl_path), processor=processor, project_root=str(tmp_path), strict_image=True)

        with self.assertRaises(FileNotFoundError):
            _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
