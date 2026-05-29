import ast
import unittest
from pathlib import Path

from PIL import Image

from receipt_ie.ocr.build_ocr_cache import build_cache_base, preprocess_image_for_ocr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _calls_to(source_path: Path, function_name: str) -> list[ast.Call]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == function_name:
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == function_name:
            calls.append(node)
    return calls


def _has_keyword(call: ast.Call, keyword_name: str, value: bool) -> bool:
    for keyword in call.keywords:
        if keyword.arg != keyword_name:
            continue
        return isinstance(keyword.value, ast.Constant) and keyword.value.value is value
    return False


class TestTrainingWiring(unittest.TestCase):
    def test_donut_training_enables_augmentation_only_for_train_dataset(self):
        calls = _calls_to(PROJECT_ROOT / "src/receipt_ie/training/train_donut.py", "DonutDataset")

        self.assertTrue(any(_has_keyword(call, "is_train", True) for call in calls))
        self.assertTrue(any(_has_keyword(call, "is_train", False) for call in calls))

    def test_layoutxlm_training_uses_train_flag_and_overlap_threshold(self):
        calls = _calls_to(PROJECT_ROOT / "src/receipt_ie/training/train_layoutxlm.py", "LayoutXLMDataset")

        self.assertTrue(any(_has_keyword(call, "is_train", True) for call in calls))
        self.assertTrue(any(_has_keyword(call, "is_train", False) for call in calls))
        self.assertTrue(any(any(keyword.arg == "overlap_threshold" for keyword in call.keywords) for call in calls))


class TestOcrPreprocessWiring(unittest.TestCase):
    def test_preprocess_metadata_and_cache_schema_for_scaled_image(self):
        tmp_path = PROJECT_ROOT / "tmp" / "test_training_wiring"
        tmp_path.mkdir(parents=True, exist_ok=True)
        image_path = tmp_path / "receipt.png"
        Image.new("RGB", (100, 50), color="white").save(image_path)

        image = Image.open(image_path).convert("RGB")
        processed, metadata, detector_path = preprocess_image_for_ocr(
            image=image,
            sample_id="sample/01",
            image_path=image_path,
            preprocess_config={"image": {"ocr": {"max_side": 50, "rectify": False, "binarize": False}}},
            output_dir=tmp_path / "preprocessed",
        )

        self.assertEqual(processed.size, (50, 25))
        self.assertTrue(metadata["applied"])
        self.assertEqual(metadata["steps"], ["resize_max_side"])
        self.assertEqual(metadata["coordinate_transform"], "scale")
        self.assertEqual(metadata["original_size"], [100, 50])
        self.assertEqual(metadata["processed_size"], [50, 25])
        self.assertTrue(detector_path.exists())

        cache_base = build_cache_base("sample/01", "ocr_v1", metadata)
        self.assertEqual(cache_base["preprocessed_size"], [50, 25])
        self.assertEqual(cache_base["coordinate_transform"], "scale")
        self.assertIn("preprocessed_image_path", cache_base)


if __name__ == "__main__":
    unittest.main()
