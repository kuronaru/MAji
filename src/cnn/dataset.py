import os
import torch
from PIL import Image
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
            if (self.directory.find("cat") != -1):
                labels.append(int(1))
            elif (self.directory.find("airplane") != -1):
                labels.append(int(0))

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image = Image.open(img_path)
        image = self.transform(image)
        label = torch.Tensor([label])
        # print(image, label)
        return image, label
