import unittest

import cv2
from PIL import Image

from src.utils.image_process import parse_games


class MyTestCase(unittest.TestCase):
    def test_image_process(self):
        test_image = cv2.imread("../data/games/0059_r18.jpg")
        model = "../data/model/model_tile_classifier.pt"
        self.assertEqual(parse_games(test_image, model),
                         ["1m", "1m", "6m", "6m", "2z", "2z", "6z"])


if __name__ == '__main__':
    unittest.main()
