import os
import re
from logging import INFO, DEBUG

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utils.dbgf import DebugPrintf

dbgf = DebugPrintf("image_process", DEBUG)


def imshow_normalized(tensor, mean, std):
    # 如果张量在 GPU 上，先将它转移到 CPU 上
    if tensor.device.type == "cuda":
        tensor = tensor.cpu()

    # 将张量转换为 NumPy 数组
    image = tensor.numpy()

    # 将图像的通道维度放在最后
    image = np.transpose(image, (1, 2, 0))

    # 反归一化操作，将像素值按标准化的均值和标准差还原
    image = image * std + mean

    # 将图像缩放到 [0, 1] 范围内
    image = np.clip(image, 0, 1)

    # 绘制图像
    plt.imshow(image)
    plt.show()


def test_canny_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Canny Edge Detection", cv2.WINDOW_AUTOSIZE)

    # 创建滑动条来调整阈值
    cv2.createTrackbar("Threshold1", "Canny Edge Detection", 0, 500, lambda x: None)
    cv2.createTrackbar("Threshold2", "Canny Edge Detection", 0, 500, lambda x: None)
    while True:
        # 获取当前滑动条位置
        threshold1 = cv2.getTrackbarPos("Threshold1", "Canny Edge Detection")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Canny Edge Detection")
        edges = cv2.Canny(gray, threshold1, threshold2)
        cv2.imshow("Canny Edge Detection", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def sample_image_coordinates(image):
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

    # 定义鼠标回调函数
    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Coordinates: (x={x}, y={y})")
            # 在图像上绘制一个小圆圈以标记点击位置
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("image", image)

    if image is None:
        print("Error: Could not load image")
    else:
        # 显示图像并设置鼠标回调
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", get_coordinates, image)

        # 按任意键退出
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def crop_image(image, coordinate_list):
    cropped_image_list = []
    for coordinate in coordinate_list:
        start_y, end_y, start_x, end_x = coordinate
        cropped_image = image[start_y:end_y, start_x:end_x]
        cropped_image_list.append(cropped_image)
    return cropped_image_list


def translate_feature(feature):
    if (feature == 30):
        dbgf(DEBUG, "None")
    number = feature % 10
    tile_type = None
    if (feature < 10):
        tile_type = 'm'
    elif (feature < 20):
        tile_type = 'p'
    elif (feature < 30):
        tile_type = 's'
    else:
        tile_type = 'z'
    dbgf(DEBUG, "%d%s" % (number, tile_type))


def crop_games(image_path, save_path: str, discard_rows):
    image = cv2.imread(image_path)
    root_image_name, _ = os.path.splitext(os.path.basename(image_path))

    # hand_self
    hand_self_coordinate_list = []
    for i in range(13):
        coordinate = 860, 983, 284 + round(1129 / 13 * i), 284 + round(1129 / 13 * (i + 1))
        hand_self_coordinate_list.append(coordinate)
    hand_self_image_list = crop_image(image, hand_self_coordinate_list)

    for index, img in enumerate(hand_self_image_list):
        img_temp = cv2.resize(img, (40, 60))
        image_name = os.path.join(save_path, root_image_name + "_0%02d.jpg" % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, "save image: %s" % image_name)
        # cv2.imshow("Cropped Image", img_temp)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # hand_next
    hand_next_coordinate_list = [(683, 729, 1605, 1693),
                                 (637, 683, 1593, 1678),
                                 (592, 635, 1584, 1664),
                                 (549, 591, 1568, 1651),
                                 (507, 549, 1556, 1638),
                                 (468, 508, 1546, 1625),
                                 (430, 468, 1535, 1613),
                                 (393, 429, 1524, 1602),
                                 (357, 392, 1514, 1591),
                                 (322, 357, 1504, 1579),
                                 (290, 323, 1496, 1568),
                                 (257, 290, 1486, 1558),
                                 (227, 257, 1478, 1548)]
    hand_next_image_list = crop_image(image, hand_next_coordinate_list)

    for index, img in enumerate(hand_next_image_list):
        img_temp = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_temp = cv2.resize(img_temp, (40, 60))
        image_name = os.path.join(save_path, root_image_name + "_1%02d.jpg" % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, "save image: %s" % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # hand_opposite
    hand_opposite_coordinate_list = []
    for i in range(13):
        coordinate = 45, 79, 746 + round(520 / 12 * i), 789 + round(526 / 12 * i)
        hand_opposite_coordinate_list.append(coordinate)
    hand_opposite_image_list = crop_image(image, hand_opposite_coordinate_list)

    for index, img in enumerate(reversed(hand_opposite_image_list)):
        img_temp = cv2.rotate(img, cv2.ROTATE_180)
        img_temp = cv2.resize(img_temp, (40, 60))
        image_name = os.path.join(save_path, root_image_name + "_2%02d.jpg" % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, "save image: %s" % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # hand_preceding
    hand_preceding_coordinate_list = [(105, 132, 412, 475),
                                      (132, 160, 403, 468),
                                      (161, 189, 395, 460),
                                      (189, 219, 386, 452),
                                      (219, 250, 376, 443),
                                      (250, 282, 365, 435),
                                      (282, 315, 356, 426),
                                      (315, 349, 345, 417),
                                      (349, 384, 334, 408),
                                      (384, 421, 323, 398),
                                      (421, 458, 311, 388),
                                      (459, 496, 299, 377),
                                      (498, 537, 288, 367)]
    hand_preceding_image_list = crop_image(image, hand_preceding_coordinate_list)

    for index, img in enumerate(hand_preceding_image_list):
        img_temp = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_temp = cv2.resize(img_temp, (40, 60))
        image_name = os.path.join(save_path, root_image_name + "_3%02d.jpg" % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, "save image: %s" % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # discard_self
    discard_self_coordinate_list = []
    for i in range(6):
        coordinate = 499, 551, 780 + round(343 / 6 * i), 856 + round(343 / 6 * i)
        discard_self_coordinate_list.append(coordinate)
    if (discard_rows > 1):
        for i in range(6):
            coordinate = 551, 609, 780 + round(343 / 6 * i), 856 + round(343 / 6 * i)
            discard_self_coordinate_list.append(coordinate)
    if (discard_rows > 2):
        for i in range(6):
            coordinate = 609, 674, 780 + round(343 / 6 * i), 856 + round(343 / 6 * i)
            discard_self_coordinate_list.append(coordinate)
    discard_self_image_list = crop_image(image, discard_self_coordinate_list)

    for index, img in enumerate(discard_self_image_list):
        img_temp = cv2.resize(img, (40, 60))
        image_name = os.path.join(save_path, root_image_name + "_4%02d.jpg" % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, "save image: %s" % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # discard_next
    discard_next_coordinate_list = []
    for i in range(6):
        coordinate = (437 - round(183 / 5 * i), 483 - round(183 / 5 * i),
                      1136 - round(14 / 5 * i), 1212 - round(22 / 5 * i))
        discard_next_coordinate_list.append(coordinate)
    if (discard_rows > 1):
        for i in range(6):
            coordinate = (437 - round(183 / 5 * i), 483 - round(183 / 5 * i),
                          1212 - round(22 / 5 * i), 1284 - round(26 / 5 * i))
            discard_next_coordinate_list.append(coordinate)
    if (discard_rows > 2):
        for i in range(6):
            coordinate = (437 - round(183 / 5 * i), 483 - round(183 / 5 * i),
                          1284 - round(26 / 5 * i), 1362 - round(37 / 5 * i))
            discard_next_coordinate_list.append(coordinate)
    discard_next_image_list = crop_image(image, discard_next_coordinate_list)

    for index, img in enumerate(discard_next_image_list):
        img_temp = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_temp = cv2.resize(img_temp, (40, 60))
        image_name = os.path.join(save_path, root_image_name + '_5%02d.jpg' % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, 'save image: %s' % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # discard_opposite
    discard_opposite_coordinate_list = []
    if (discard_rows > 2):
        for i in range(6):
            coordinate = 142, 180, 802 + round(236 / 5 * i), 866 + round(236 / 5 * i)
            discard_opposite_coordinate_list.append(coordinate)
    if (discard_rows > 1):
        for i in range(6):
            coordinate = 180, 220, 800 + round(239 / 5 * i), 864 + round(239 / 5 * i)
            discard_opposite_coordinate_list.append(coordinate)
    for i in range(6):
        coordinate = 220, 261, 797 + round(243 / 5 * i), 861 + round(243 / 5 * i)
        discard_opposite_coordinate_list.append(coordinate)
    discard_opposite_image_list = crop_image(image, discard_opposite_coordinate_list)

    for index, img in enumerate(reversed(discard_opposite_image_list)):
        img_temp = cv2.rotate(img, cv2.ROTATE_180)
        img_temp = cv2.resize(img_temp, (40, 60))
        image_name = os.path.join(save_path, root_image_name + '_6%02d.jpg' % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, 'save image: %s' % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # discard_preceding
    discard_preceding_coordinate_list = []
    for i in range(6):
        coordinate = (265 + round(216 / 6 * i), 311 + round(216 / 6 * i),
                      726 - round(26 / 6 * i), 799 - round(26 / 6 * i))
        discard_preceding_coordinate_list.append(coordinate)
    if (discard_rows > 1):
        for i in range(6):
            coordinate = (264 + round(216 / 6 * i), 310 + round(216 / 6 * i),
                          657 - round(32 / 6 * i), 730 - round(32 / 6 * i))
            discard_preceding_coordinate_list.append(coordinate)
    if (discard_rows > 2):
        for i in range(6):
            coordinate = (263 + round(216 / 6 * i), 309 + round(216 / 6 * i),
                          588 - round(35 / 6 * i), 661 - round(35 / 6 * i))
            discard_preceding_coordinate_list.append(coordinate)
    discard_preceding_image_list = crop_image(image, discard_preceding_coordinate_list)

    for index, img in enumerate(discard_preceding_image_list):
        img_temp = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_temp = cv2.resize(img_temp, (40, 60))
        image_name = os.path.join(save_path, root_image_name + '_7%02d.jpg' % index)
        cv2.imwrite(image_name, img_temp)
        dbgf(DEBUG, 'save image: %s' % image_name)
    #     cv2.imshow("Cropped Image", img_temp)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return 0


# def digitalize_games(image):
#     # Load the trained model
#     trained_model = TileClassifier()
#     assert os.path.exists("../../data/model/model_tile_classifier.pt"), "model_tile_classifier does not exist"
#     trained_model.load_state_dict(torch.load("../../data/model/model_tile_classifier.pt",
#                                              map_location=torch.device("cpu")))
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.6710, 0.6679, 0.6726), (0.2327, 0.2291, 0.2132))])
#
#     img_tensor = transform(image).unsqueeze(0)
#     outputs = trained_model(img_tensor)
#     translate_feature(outputs.argmax(1))


def batch_crop_games(directory, save_path):
    for img in os.listdir(directory):
        img_path = os.path.join(directory, img)
        assert os.path.isfile(img_path)
        if (not img.find(".jpg")):
            continue
        match = re.search(r'_r(\d+)\.', img)
        if (match):
            rounds = match.group(1)
            rows = int(rounds) / 6
            crop_games(img_path, save_path, rows)
        else:
            dbgf(INFO, "No match found")


if __name__ == "__main__":
    # sample_image_coordinates(test_img)
    batch_crop_games("../../data/games", "../../data/tiles/raw")
