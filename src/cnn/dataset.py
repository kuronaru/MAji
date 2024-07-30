import os

import cv2
import torch
from torch.utils.data import Dataset


class MajDataset(Dataset):
    def __init__(self, path, n_img, start_index=0, transform=None) -> None:
        super(MajDataset, self).__init__()
        self.directory = path

        img_paths = []
        labels = []
        for i in range(n_img):
            img_name = "%04d_r6.jpg" % (i + start_index)
            img_path = os.path.join(self.directory, img_name)
            if (os.path.exists(img_path)):
                print("add image ", img_path)
                img_paths.append(img_path)
            # if (self.directory.find("cat") != -1):
            #     labels.append(int(1))
            # elif (self.directory.find("airplane") != -1):
            #     labels.append(int(0))

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image = cv2.imread(img_path)
        if (self.transform):
            image = self.transform(image)
        label = torch.Tensor([label])
        return image, label


class TileDataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        super(TileDataset, self).__init__()
        self.directory = path  # ../../data/tiles/train_data

        img_paths = []
        labels = []
        for dir_name in os.listdir(self.directory):
            directory = os.path.join(self.directory, dir_name)
            if (os.path.isdir(directory)):
                # 0-9: 0m-9m
                # 10-19: 0p-9p
                # 20-29: 0s-9s
                # 30: none
                # 31-37: 1z-7z
                label = 0
                if (dir_name.endswith("m")):
                    label = int(dir_name[0])
                elif (dir_name.endswith("p")):
                    label = int(dir_name[0]) + 10
                elif (dir_name.endswith("s")):
                    label = int(dir_name[0]) + 20
                elif (dir_name.endswith("z")):
                    label = int(dir_name[0]) + 30

                for img_name in os.listdir(directory):
                    img_path = os.path.join(directory, img_name)
                    if (os.path.exists(img_path)):
                        img_paths.append(img_path)
                        labels.append(label)

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image = cv2.imread(img_path)
        if (self.transform):
            image = self.transform(image)
        label = torch.Tensor([label])
        return image, label