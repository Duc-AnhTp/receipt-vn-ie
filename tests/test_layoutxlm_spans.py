import unittest

from receipt_ie.inference.bio_aggregation import aggregate_bio_entities, aggregate_bio_spans


class TestLayoutXlmSpanAggregation(unittest.TestCase):
    def test_aggregate_bio_spans_keeps_best_span(self):
        words = ["Store", "A", "noise", "Long", "Address", "Line"]
        labels = ["B-STORE_NAME", "I-STORE_NAME", "O", "B-ADDRESS", "I-ADDRESS", "I-ADDRESS"]

        result = aggregate_bio_spans(words, labels)

        self.assertEqual(result["store_name"], "Store A")
        self.assertEqual(result["address"], "Long Address Line")

    def test_field_switch_starts_new_span(self):
        words = ["A", "B", "2026"]
        labels = ["B-STORE_NAME", "I-ADDRESS", "B-DATE"]

        result = aggregate_bio_spans(words, labels)

        self.assertEqual(result["store_name"], "A")
        self.assertEqual(result["address"], "B")
        self.assertEqual(result["date"], "2026")

    def test_entities_include_confidence_bbox_and_i_warning(self):
        words = ["115.000"]
        labels = ["I-TOTAL"]
        entities = aggregate_bio_entities(words, labels, confidences=[0.91], boxes=[[10, 20, 100, 40]])

        self.assertEqual(entities[0]["field"], "total")
        self.assertEqual(entities[0]["text"], "115.000")
        self.assertEqual(entities[0]["confidence"], 0.91)
        self.assertEqual(entities[0]["bbox"], [10, 20, 100, 40])
        self.assertEqual(entities[0]["warning"], "I-without-active-B")


if __name__ == "__main__":
    unittest.main()
