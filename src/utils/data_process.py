from logging import DEBUG, INFO

import cv2

from src.utils.dbgf import DebugPrintf, DATA_PROCESS_DBG_LVL
from src.utils.image_process import parse_games

dbgf = DebugPrintf("data_process", DATA_PROCESS_DBG_LVL)


def encode_data():
    model = "../../data/model/model_tile_classifier.pt"
    image = cv2.imread("../../data/games/0029_r12.jpg")
    reference_list = ["0m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                      "0p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                      "0s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                      "0z", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]

    data_list = parse_games(image, model)

    code_output = []
    for _list in data_list:
        code = []
        for index, target in enumerate(reference_list):
            code_temp = [0, 0, 0, 0, 0]
            t_num = _list.count(target)
            code_temp[t_num] = 1

            # debug print
            tile_type_num = int(index / 10)
            tile_digit = index % 10
            tile_type = ["m", "p", "s", "z"]
            dbgf(DEBUG, "%s%s %s" % (tile_digit, tile_type[tile_type_num], code_temp))

            code += code_temp
        code_output.append(code)

    return code_output


if __name__ == "__main__":
    encode_data()
