from logging import DEBUG, INFO

import cv2
import torch

from src.utils.dbgf import DebugPrintf, DATA_PROCESS_DBG_LVL
from src.utils.image_process import parse_games

dbgf = DebugPrintf("data_process", DATA_PROCESS_DBG_LVL)

reference_list = ["0m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                  "0p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                  "0s", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                  "0z", "1z", "2z", "3z", "4z", "5z", "6z", "7z"]


def translate_data(data):
    """
    translate code-like data into numeric data
    :param data: coded data list, e.g. ["1m", "2p"]
    :return: numeric data list, e.g. [1, 12]
    """
    data_output = []
    for element in data:
        data_trans = reference_list.index(element)
        data_output.append(data_trans)
    return data_output


def data_to_code(data):
    """
    encode data into a 13*38 matrix, each row represents a one-hot encoded tile symbol
    :param data: numeric data tensor
    :return: encoded one-hot code tensor
    """
    data_len = len(data)
    code = torch.zeros(data_len, 38)
    for i in range(data_len):
        code[i][data[i]] = 1
    return code


def code_to_data(code):
    """
    decode the one-hot encoding matrix into numeric data list
    :param code: 13*38 one-hot code tensor
    :return: numeric data tensor
    """
    data = torch.empty(len(code))
    for i, code_row in enumerate(code):
        tile = code_row.argmax(0)
        data[i] = tile
    return data


def encode_image_to_data():
    """
    test parse data
    :return:
    """
    model = "../../data/model/model_tile_classifier.pt"
    image = cv2.imread("../../data/games/0029_r12.jpg")
    code_lists = parse_games(image, model)

    data_lists = []
    for _list in code_lists:
        # _list = translate_data(_list)
        data = code_to_data(_list)
        data_lists.append(data)

    return data_lists


if __name__ == "__main__":
    encode_image_to_data()
