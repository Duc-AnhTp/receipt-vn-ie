import unittest

from receipt_ie.data.build_layoutxlm_labels import assign_word_labels


class TestLayoutXlmLabels(unittest.TestCase):
    def test_overlap_threshold_controls_label_assignment(self):
        words = ["Store"]
        boxes = [[0, 0, 100, 100]]
        field_boxes = {"store_name": [[0, 0, 40, 100]]}

        self.assertEqual(assign_word_labels(words, boxes, field_boxes, overlap_threshold=0.5), ["O"])
        self.assertEqual(assign_word_labels(words, boxes, field_boxes, overlap_threshold=0.3), ["B-STORE_NAME"])


if __name__ == "__main__":
    unittest.main()
