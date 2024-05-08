import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, n_img, index=0, transform=None) -> None:
        super(MyDataset, self).__init__()
        self.path = path

        img_paths = []
        labels = []
        for i in range(n_img):
            img_name = "%03d.jpg" % (i + index)
            # print("add ", img_name)
            img_paths.append(os.path.join(self.path, img_name))
            if (self.path.find("cat") != -1):
                labels.append(int(1))
            elif (self.path.find("airplane") != -1):
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
