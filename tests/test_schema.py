import unittest
from receipt_ie.data.schemas import FIELDS, LABELS, LABEL2ID, ID2LABEL, BaseExtractor

class TestSchemaDefinitions(unittest.TestCase):
    def test_fields(self):
        # Đảm bảo 4 trường thông tin bắt buộc đều có mặt
        self.assertIn("store_name", FIELDS)
        self.assertIn("date", FIELDS)
        self.assertIn("total", FIELDS)
        self.assertIn("address", FIELDS)
        self.assertEqual(len(FIELDS), 4)

    def test_labels_bio(self):
        # Đảm bảo nhãn "O" tồn tại
        self.assertIn("O", LABELS)
        
        # Đảm bảo có đủ cặp B- và I- cho từng trường trong FIELDS
        for field in FIELDS:
            bio_b = f"B-{field.upper()}"
            bio_i = f"I-{field.upper()}"
            self.assertIn(bio_b, LABELS)
            self.assertIn(bio_i, LABELS)
            
        # Tổng số nhãn: 1 nhãn O + 4 trường * 2 = 9 nhãn
        self.assertEqual(len(LABELS), 9)

    def test_mappings(self):
        # Kiểm tra hai chiều ánh xạ nhãn và ID
        self.assertEqual(len(LABEL2ID), len(LABELS))
        self.assertEqual(len(ID2LABEL), len(LABELS))
        
        for i, label in enumerate(LABELS):
            self.assertEqual(LABEL2ID[label], i)
            self.assertEqual(ID2LABEL[i], label)

if __name__ == "__main__":
    unittest.main()
