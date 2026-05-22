import os
import json
import unittest
from pathlib import Path

from receipt_ie.baseline.rule_extractor import extract_fields_from_ocr


class TestBaselineExtractor(unittest.TestCase):
    def test_extract_fields(self):
        # Thiết lập dữ liệu OCR cache mock
        ocr_data = {
            "lines": [
                [
                    {"bbox": [10, 10, 150, 30], "text": "Cửa Hàng"},
                    {"bbox": [160, 10, 300, 30], "text": "MINIMART ANAN"}
                ],
                [
                    {"bbox": [10, 35, 100, 50], "text": "Địa Chỉ:"},
                    {"bbox": [110, 35, 400, 50], "text": "Chợ Sủi, Phú Thị, Gia Lâm"}
                ],
                [
                    {"bbox": [10, 60, 120, 75], "text": "Số ĐT: 0123456789"},
                    {"bbox": [150, 60, 250, 75], "text": "MST: 00112233"}
                ],
                [
                    {"bbox": [10, 90, 180, 105], "text": "Ngày GD: 09/08/2020"},
                    {"bbox": [200, 90, 280, 105], "text": "14:30:15"}
                ],
                [
                    {"bbox": [10, 130, 100, 145], "text": "Sản phẩm A"},
                    {"bbox": [300, 130, 350, 145], "text": "50.000"}
                ],
                [
                    {"bbox": [10, 150, 100, 165], "text": "Sản phẩm B"},
                    {"bbox": [300, 150, 350, 165], "text": "65.000"}
                ],
                [
                    {"bbox": [10, 180, 150, 195], "text": "TỔNG CỘNG:"},
                    {"bbox": [300, 180, 390, 195], "text": "115.000đ"}
                ]
            ]
        }

        extracted = extract_fields_from_ocr(ocr_data)

        # Kiểm tra độ chính xác của các trường trích xuất được
        self.assertEqual(extracted["store_name"], "MINIMART ANAN")
        self.assertEqual(extracted["date"], "2020-08-09")
        self.assertEqual(extracted["total"], "115000")
        self.assertEqual(extracted["address"], "Chợ Sủi, Phú Thị, Gia Lâm")

    def test_fallback_total(self):
        # Dữ liệu OCR không chứa từ khóa tổng cộng, chỉ có cụm số ở cuối
        ocr_data = {
            "lines": [
                [{"bbox": [10, 10, 200, 30], "text": "MIMI COFFEE"}],
                [{"bbox": [10, 50, 100, 65], "text": "50.000"}],
                [{"bbox": [10, 80, 100, 95], "text": "20.000"}]  # Fallback sẽ chọn cụm số lớn nhất ở nửa cuối
            ]
        }
        extracted = extract_fields_from_ocr(ocr_data)
        self.assertEqual(extracted["total"], "50000")


if __name__ == "__main__":
    unittest.main()
