import unittest

from receipt_ie.postprocess.address_extractor import extract_address_from_lines
from receipt_ie.postprocess.store_extractor import extract_store_name_from_lines


class TestAddressAndStoreExtractor(unittest.TestCase):
    def test_address_keyword(self):
        lines = ["MINIMART ANAN", "Địa chỉ: 123 Nguyễn Trãi, Q.5", "ĐT: 0981234567"]
        self.assertEqual(extract_address_from_lines(lines), "123 Nguyễn Trãi, Q.5")

    def test_store_from_top_lines(self):
        lines = ["HÓA ĐƠN BÁN HÀNG", "MINIMART ANAN", "Địa chỉ: 123 Nguyễn Trãi"]
        self.assertEqual(extract_store_name_from_lines(lines), "MINIMART ANAN")


if __name__ == "__main__":
    unittest.main()
