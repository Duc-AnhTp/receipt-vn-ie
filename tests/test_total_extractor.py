import unittest

from receipt_ie.data.normalize_text import normalize_money
from receipt_ie.postprocess.total_extractor import extract_total_from_lines


class TestTotalExtractor(unittest.TestCase):
    def test_total_with_vnd_dot(self):
        self.assertEqual(normalize_money("Tổng cộng: 115.000đ"), "115000")

    def test_total_with_comma(self):
        self.assertEqual(normalize_money("Total: 115,000"), "115000")

    def test_total_ignore_phone(self):
        lines = ["ĐT: 0981234567", "Tổng cộng: 115.000đ"]
        self.assertEqual(extract_total_from_lines(lines), "115000")


if __name__ == "__main__":
    unittest.main()
