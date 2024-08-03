import unittest

import cv2

from src.utils.image_process import parse_games


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.model = "../data/model/model_tile_classifier.pt"

    def test_parse_games_0(self):
        test_image = cv2.imread("../data/games/0048_r12.jpg")
        test_result = (['6p', '7p', '8p', '8p', '9p', '3s', '4s', '6s', '7s', '8s'],
                       ['4p', '2p', '3p'],
                       ['5p', '1s', '2s', '5s', '5s', '5z', '5z'],
                       ['3p', '2p', '1p', '4z', '4z', '4z'],
                       ['6m', '7m', '3p', '4p', '6p', '8p', '8p', '2s', '3s', '4s'],
                       ['7s', '0s', '6s'],
                       ['3m', '3m', '4m', '0m', '6m', '0p', '5p', '5p', '5z', '5z'],
                       ['8m', '8m', '8m'],
                       ['1m', '6z', '9s', '1s', '2p', '5m', '3m', '1s'],
                       ['9p', '1m', '3z', '9s', '2p', '6m', '3m', '3z', '4m'],
                       ['3z', '7z', '1m', '9m', '1p', '2z', '9p', '9p', '7z', '1s', '1p'],
                       ['2s', '8s', '6z', '7z', '3z', '2z', '9m', '4s', '4p', '5m', '1z'])
        self.assertEqual(parse_games(test_image, self.model), test_result)

    def test_parse_games_1(self):
        test_image = cv2.imread("../data/games/0059_r18.jpg")
        test_result = (['1p', '2p', '3p', '5p'],
                       ['7z', '7z', '7z', '7m', '0m', '6m', '4p', '4p', '4p'],
                       ['2m', '2m', '2m', '6m', '7m', '8m', '5p', '4s', '7s', '7s'],
                       ['4p', '3p', '2p'],
                       ['4m', '4m', '5m', '5m', '6m', '9m', '4s', '7s', '8s', '9s'],
                       ['4s', '3s', '5s'],
                       ['5p', '7p', '7p', '8p', '8p', '8p', '1s', '2s', '3s', '3s'],
                       ['6s', '6s', '6s'],
                       ['9p', '6z', '9m', '5z', '4z', '2s', '8s', '4s', '9p', '1p', '7m', '1z', '5m', '6p', '9m'],
                       ['1z', '3z', '1s', '5z', '2z', '9s', '2s', '3m', '4z', '5z', '6p', '3z', '1s'],
                       ['1z', '1s', '5z', '2z', '8s', '9s', '3p', '1m', '1z', '7m', '6p', '6z', '4z', '9m'],
                       ['6z', '3z', '3z', '1m', '1m', '1m', '3m', '8s', '3m', '3m', '9p', '5s', '2p', '3p'])
        self.assertEqual(parse_games(test_image, self.model), test_result)

    def test_parse_games_2(self):
        test_image = cv2.imread("../data/games/0008_r18.jpg")
        test_result = (['1m', '1m', '2m', '3m', '2p', '2p', '2p', '7p', '8p', '9p', '3s', '4s', '0s'],
                       [],
                       ['6m', '6m', '7m', '7m', '1p', '1p', '3p', '3p', '4p', '4p', '9p', '4s', '4s'],
                       [],
                       ['3m', '4m', '0m', '5m', '9m', '9m', '3p', '4p', '0p', '6p', '7p', '6s', '7s'],
                       [],
                       ['2m', '3m', '4m', '4m', '9p', '2s', '5s', '5s', '6s', '7s'],
                       ['3s', '1s', '2s'],
                       ['4z', '9s', '5z', '2z', '1z', '7m',
                        '1s', '7z', '7p', '9s', '6p', '6p',
                        '5p', '8s', '6m', '1z', '2s'],
                       ['5z', '6z', '3z', '3z', '7s', '5s',
                        '8p', '2z', '3z', '8s', '5p', '2m',
                        '5z', '9s', '3m', '3s', '5z'],
                       ['1s', '2z', '1z', '2m', '1p', '7p',
                        '1m', '4z', '3p', '8m', '7z', '9s',
                        '4z', '8m', '4z', '8s'],
                       ['9m', '9m', '6m', '1m', '6z', '2p',
                        '8p', '7m', '1p', '6z', '3z', '5p',
                        '6p', '8m', '8s', '1z', '1s'])
        self.assertEqual(parse_games(test_image, self.model), test_result)


if __name__ == '__main__':
    unittest.main()
