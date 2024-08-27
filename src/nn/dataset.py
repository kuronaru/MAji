import os

import cv2
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset

from src.nn.hand_infer import HandInfer
from src.utils.data_process import data_to_code
from src.utils.image_process import parse_games


class MajDataset(Dataset):
    def __init__(self, path, n_img, start_index=0, trained_model="../../data/model/model_tile_classifier.pt") -> None:
        super(MajDataset, self).__init__()

        self.directory = path  # ../../data/games
        self.model = trained_model

        img_paths = []
        for i in range(n_img):
            index = i + start_index
            img_name = "%04d_r6.jpg" % index
            img_path = os.path.join(self.directory, img_name)
            if (os.path.exists(img_path)):
                img_paths.append(img_path)
            else:
                continue

            img_name = "%04d_r12.jpg" % index
            img_path = os.path.join(self.directory, img_name)
            if (os.path.exists(img_path)):
                img_paths.append(img_path)
            else:
                continue

            img_name = "%04d_r18.jpg" % index
            img_path = os.path.join(self.directory, img_name)
            if (os.path.exists(img_path)):
                img_paths.append(img_path)

        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        data_list = parse_games(image, self.model)
        data_pairs = [(data_list[8], data_list[0] + data_list[1]),
                      (data_list[9], data_list[2] + data_list[3]),
                      (data_list[10], data_list[4] + data_list[5]),
                      (data_list[11], data_list[6] + data_list[7])]

        input_list = []
        target_list = []
        for pair in data_pairs:
            input_seq = pair[0]
            target_seq = pair[1]
            input_list.append(torch.tensor(input_seq))
            target_code = data_to_code(target_seq)
            target_list.append(torch.tensor(target_code, dtype=torch.float))
        input_tensor = pad_sequence(input_list, batch_first=True, padding_value=30)
        target_tensor = torch.stack(target_list)  # (batch_size, sequence_length)
        return input_tensor, target_tensor


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
