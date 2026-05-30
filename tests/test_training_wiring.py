import ast
import unittest
from pathlib import Path

from PIL import Image


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

if __name__ == "__main__":
    unittest.main()
