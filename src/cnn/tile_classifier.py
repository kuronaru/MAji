from torch import nn
from torch.nn import functional

from src.cnn.resnet import ResidualBlock


class TileClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TileClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = ResidualBlock(64, 128, 2)
        self.conv2_2 = ResidualBlock(128, 128, 1)
        self.conv3_1 = ResidualBlock(128, 256, 2)
        self.conv3_2 = ResidualBlock(256, 256, 1)
        self.fc = nn.Linear(256, 38)

    def forward(self, x):
        y = self.max_pool(self.conv1(x))
        y = self.conv2_2(self.conv2_1(y))
        y = self.conv3_2(self.conv3_1(y))
        y = functional.adaptive_avg_pool2d(y, 1)
        y = y.view(y.shape[0], -1)
        return functional.softmax(self.fc(y), 1)
