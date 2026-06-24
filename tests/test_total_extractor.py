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

    def test_extract_total_does_not_use_phone_as_total(self):
        self.assertEqual(extract_total_from_lines(["Thanh toán: 0912345678"]), "")

    def test_extract_total_still_extracts_real_money(self):
        self.assertEqual(extract_total_from_lines(["Thanh toán: 115.000đ"]), "115000")

    def test_extract_total_ignores_phone_but_keeps_money_same_line(self):
        self.assertEqual(extract_total_from_lines(["Thanh toán: 115.000đ - Hotline: 0912345678"]), "115000")


if __name__ == "__main__":
    unittest.main()
