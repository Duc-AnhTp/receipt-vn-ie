import unittest
import torch
from receipt_ie.data.build_layoutxlm_labels import (
    compute_intersection_area,
    compute_box_area,
    assign_word_labels,
    normalize_bbox,
    align_tokens_layoutxlm,
    LABEL_TO_ID
)

class MockTokenizer:
    def __init__(self):
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1

    def tokenize(self, text):
        # Nếu từ dài, tách thành các subwords tượng trưng (ví dụ: chia đôi nếu độ dài > 5)
        if len(text) > 5:
            return [text[:4], text[4:]]
        return [text]

    def convert_tokens_to_ids(self, tokens):
        # Gán ID tượng trưng cho các tokens
        return [100 + i for i in range(len(tokens))]

class TestLayoutXlmLabels(unittest.TestCase):
    def test_box_math(self):
        box1 = [10, 10, 50, 50]
        box2 = [20, 20, 60, 60]
        
        # Giao nhau là [20, 20, 50, 50] -> dt = 30 * 30 = 900
        intersect = compute_intersection_area(box1, box2)
        self.assertEqual(intersect, 900)
        
        area = compute_box_area(box1)
        self.assertEqual(area, 1600) # 40 * 40

    def test_assign_word_labels(self):
        words = ["Cửa", "Hàng", "An", "An", "Ngày", "20/12/2020", "Tổng", "100.000"]
        # Thiết lập các hộp của từ
        word_boxes = [
            [10, 10, 30, 25],   # Cửa
            [35, 10, 60, 25],   # Hàng
            [65, 10, 90, 25],   # An
            [95, 10, 120, 25],  # An
            [10, 40, 40, 55],   # Ngày
            [45, 40, 100, 55],  # 20/12/2020
            [10, 70, 40, 85],   # Tổng
            [45, 70, 90, 85]    # 100.000
        ]
        
        # Thiết lập ground truth field boxes
        field_boxes = {
            "store_name": [[8, 8, 125, 28]], # Chứa "Cửa Hàng An An"
            "date": [[42, 38, 105, 58]],      # Chứa "20/12/2020"
            "total": [[42, 68, 95, 88]]        # Chứa "100.000"
        }
        
        labels = assign_word_labels(words, word_boxes, field_boxes, overlap_threshold=0.5)
        
        # Kỳ vọng:
        # "Cửa" -> B-STORE_NAME
        # "Hàng" -> I-STORE_NAME
        # "An" -> I-STORE_NAME
        # "An" -> I-STORE_NAME
        # "Ngày" -> O (không nằm trong box date)
        # "20/12/2020" -> B-DATE
        # "Tổng" -> O
        # "100.000" -> B-TOTAL
        
        expected = [
            "B-STORE_NAME",
            "I-STORE_NAME",
            "I-STORE_NAME",
            "I-STORE_NAME",
            "O",
            "B-DATE",
            "O",
            "B-TOTAL"
        ]
        self.assertEqual(labels, expected)

    def test_normalize_bbox(self):
        box = [100, 200, 300, 400]
        # Kích thước ảnh 1000x2000
        norm = normalize_bbox(box, 1000, 2000)
        # x: 100 -> 100, 300 -> 300
        # y: 200 -> 100, 400 -> 200 (scale 2000 về 1000)
        self.assertEqual(norm, [100, 100, 300, 200])

    def test_align_tokens_layoutxlm(self):
        words = ["Cửa", "Hàng", "AnAn"] # AnAn dài 6 chữ sẽ bị tách đôi thành ["AnAn"[:4], "AnAn"[4:]] -> ["AnAn", ""]
        word_boxes = [
            [10, 10, 30, 20],
            [35, 10, 60, 20],
            [65, 10, 95, 20]
        ]
        word_labels = ["B-STORE_NAME", "I-STORE_NAME", "I-STORE_NAME"]
        
        tokenizer = MockTokenizer()
        aligned = align_tokens_layoutxlm(
            words=words,
            word_boxes=word_boxes,
            word_labels=word_labels,
            tokenizer=tokenizer,
            max_length=10,
            image_size=(100, 100)
        )
        
        # Chiều dài chuỗi token phải đúng bằng max_length (10)
        self.assertEqual(len(aligned["input_ids"]), 10)
        self.assertEqual(len(aligned["bbox"]), 10)
        self.assertEqual(len(aligned["labels"]), 10)
        self.assertEqual(len(aligned["attention_mask"]), 10)
        
        # Token đầu tiên phải là <s> (bos_token_id = 0)
        self.assertEqual(aligned["input_ids"][0], 0)
        self.assertEqual(aligned["bbox"][0], [0, 0, 0, 0])
        self.assertEqual(aligned["labels"][0], -100)
        
        # Word "Cửa" không bị tách -> 1 token
        # Word "Hàng" không bị tách -> 1 token
        # Word "AnAn" có độ dài 4 -> không bị tách theo định nghĩa mock (<= 5). Khoan, mock tokenizer tách nếu > 5. 
        # Độ dài "AnAn" là 4 <= 5 -> 1 token.
        # Tổng số tokens từ words = 3.
        # Tổng số tokens thực tế = BOS + 3 + EOS + 5 PADS = 10.
        
        # Vị trí 1: "Cửa" -> nhãn B-STORE_NAME (ID 1)
        self.assertEqual(aligned["labels"][1], LABEL_TO_ID["B-STORE_NAME"])
        
        # Vị trí 2: "Hàng" -> nhãn I-STORE_NAME (ID 2)
        self.assertEqual(aligned["labels"][2], LABEL_TO_ID["I-STORE_NAME"])
        
        # Vị trí 4: EOS (eos_token_id = 2)
        self.assertEqual(aligned["input_ids"][4], 2)
        self.assertEqual(aligned["labels"][4], -100)
        
        # Vị trí 5: PAD -> nhãn -100, attention_mask = 0
        self.assertEqual(aligned["input_ids"][5], 1)
        self.assertEqual(aligned["labels"][5], -100)
        self.assertEqual(aligned["attention_mask"][5], 0)

if __name__ == "__main__":
    unittest.main()
