import numpy as np
import torch
from matplotlib import pyplot as plt


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


inputs = torch.Tensor()
imshow_normalized(inputs[0], mean=[0.6710, 0.6679, 0.6726], std=[0.2327, 0.2291, 0.2132])
