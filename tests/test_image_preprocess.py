import unittest
from PIL import Image

from receipt_ie.preprocessing.image_preprocess import preprocess_receipt_image, resize_long_side


class TestImagePreprocess(unittest.TestCase):
    def test_resize_long_side(self):
        image = Image.new("RGB", (100, 50), color="white")
        resized, sx, sy = resize_long_side(image, max_long_side=50)
        self.assertEqual(resized.size, (50, 25))
        self.assertEqual(sx, 0.5)
        self.assertEqual(sy, 0.5)

    def test_profiles(self):
        image = Image.new("RGB", (100, 50), color="white")
        none = preprocess_receipt_image(image, profile="none", max_long_side=50)
        self.assertEqual(none.image.size, (100, 50))
        self.assertEqual(none.scale_x, 1.0)

        resized = preprocess_receipt_image(image, profile="resize", max_long_side=50)
        self.assertEqual(resized.image.size, (50, 25))
        self.assertEqual(resized.profile, "resize")
        self.assertEqual(resized.metadata["coordinate_transform"], "scale")


if __name__ == "__main__":
    unittest.main()
