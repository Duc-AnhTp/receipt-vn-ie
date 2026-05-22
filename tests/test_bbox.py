import unittest
from receipt_ie.ocr.reading_order import sort_reading_order


class TestReadingOrder(unittest.TestCase):
    def test_empty_regions(self):
        flat, grouped = sort_reading_order([])
        self.assertEqual(flat, [])
        self.assertEqual(grouped, [])

    def test_simple_reading_order(self):
        # Thiết lập các bboxes nằm trên các dòng khác nhau và thứ tự x khác nhau
        # Dòng 1: y quanh khoảng 10-20
        # Dòng 2: y quanh khoảng 50-60
        regions = [
            {"bbox": [100, 10, 200, 25], "text": "Dòng 1 Phải"},  # x0 = 100
            {"bbox": [10, 12, 80, 26], "text": "Dòng 1 Trái"},   # x0 = 10
            {"bbox": [50, 52, 120, 68], "text": "Dòng 2 Giữa"},  # x0 = 50
            {"bbox": [5, 50, 45, 65], "text": "Dòng 2 Trái"},    # x0 = 5
        ]

        flat, grouped = sort_reading_order(regions, y_threshold=15)

        # Kiểm tra gom nhóm dòng
        self.assertEqual(len(grouped), 2)  # Có 2 dòng
        self.assertEqual(len(grouped[0]), 2)  # Dòng 1 có 2 vùng
        self.assertEqual(len(grouped[1]), 2)  # Dòng 2 có 2 vùng

        # Kiểm tra thứ tự sắp xếp trong dòng 1 (Trái trước Phải)
        self.assertEqual(grouped[0][0]["text"], "Dòng 1 Trái")
        self.assertEqual(grouped[0][1]["text"], "Dòng 1 Phải")

        # Kiểm tra thứ tự sắp xếp trong dòng 2 (Trái trước Giữa)
        self.assertEqual(grouped[1][0]["text"], "Dòng 2 Trái")
        self.assertEqual(grouped[1][1]["text"], "Dòng 2 Giữa")

        # Kiểm tra danh sách phẳng
        self.assertEqual(flat[0]["text"], "Dòng 1 Trái")
        self.assertEqual(flat[1]["text"], "Dòng 1 Phải")
        self.assertEqual(flat[2]["text"], "Dòng 2 Trái")
        self.assertEqual(flat[3]["text"], "Dòng 2 Giữa")


if __name__ == "__main__":
    unittest.main()
