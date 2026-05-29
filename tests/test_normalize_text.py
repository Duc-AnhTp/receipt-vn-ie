import unittest

from receipt_ie.data.normalize_text import (
    normalize_vietnamese_text,
    normalize_store_name,
    normalize_address,
    normalize_date,
    normalize_money
)

class TestNormalizeText(unittest.TestCase):
    
    def test_normalize_vietnamese_text(self):
        self.assertEqual(normalize_vietnamese_text("  Xin   chào   "), "Xin chào")
        self.assertEqual(normalize_vietnamese_text("Cửa hàng : Bách Hóa Xanh - "), "Cửa hàng : Bách Hóa Xanh")
        
    def test_normalize_store_name(self):
        self.assertEqual(normalize_store_name("Cửa hàng Bách Hóa Xanh"), "Bách Hóa Xanh")
        self.assertEqual(normalize_store_name("Siêu thị Co.opmart"), "Co.opmart")
        self.assertEqual(normalize_store_name("Highlands Coffee"), "Highlands Coffee")
        
    def test_normalize_address(self):
        self.assertEqual(normalize_address("Địa chỉ: 123 Nguyễn Trãi, Q.5"), "123 Nguyễn Trãi, Q.5")
        self.assertEqual(normalize_address("Address: 456 Lê Lợi"), "456 Lê Lợi")
        
    def test_normalize_date(self):
        self.assertEqual(normalize_date("Ngày 22/05/2024 lúc 17:00"), "2024-05-22")
        self.assertEqual(normalize_date("Ngày: 09-08-20"), "2020-08-09")
        self.assertEqual(normalize_date("Ngày bán: 09.08.2020 18:32"), "2020-08-09")
        self.assertEqual(normalize_date("2024/05/22"), "2024-05-22")
        self.assertEqual(normalize_date("32/13/2024"), "") # Ngày tháng không thực tế
        self.assertEqual(normalize_date("Không có ngày"), "")
        
    def test_normalize_money(self):
        self.assertEqual(normalize_money("Tổng cộng: 115.000đ"), "115000")
        self.assertEqual(normalize_money("115,000 VND"), "115000")
        self.assertEqual(normalize_money("0115000"), "115000")
        self.assertEqual(normalize_money("Chỉ có chữ"), "")
        self.assertEqual(normalize_money("0"), "0")

if __name__ == "__main__":
    unittest.main()
