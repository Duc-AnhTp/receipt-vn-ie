import unittest
from pathlib import Path
from PIL import Image

from receipt_ie.ocr.build_ocr_cache import build_cache_base, file_md5, save_preprocessed_image
from receipt_ie.preprocessing.image_preprocess import preprocess_receipt_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestOcrCacheSchema(unittest.TestCase):
    def test_cache_base_schema_for_resize_profile(self):
        tmp_path = PROJECT_ROOT / "tmp" / "test_ocr_cache_schema"
        tmp_path.mkdir(parents=True, exist_ok=True)
        image_path = tmp_path / "receipt.png"
        Image.new("RGB", (100, 50), color="white").save(image_path)

        pre = preprocess_receipt_image(Image.open(image_path).convert("RGB"), profile="resize", max_long_side=50)
        pre_path = save_preprocessed_image(pre, "sample_01", tmp_path / "preprocessed")
        cache = build_cache_base(
            sample_id="sample_01",
            image_path=image_path,
            image_hash=file_md5(image_path),
            ocr_engine={"detector": "PaddleOCR", "recognizer": "VietOCR"},
            pre=pre,
            cache_version="v2",
            preprocessed_image_path=pre_path,
        )

        self.assertEqual(cache["preprocess"]["profile"], "resize")
        self.assertEqual(cache["preprocess"]["cache_version"], "v2")
        self.assertEqual(cache["preprocessed_size"], [50, 25])
        self.assertIn("image_hash", cache)
        self.assertIn("preprocessed_image_path", cache)


if __name__ == "__main__":
    unittest.main()
