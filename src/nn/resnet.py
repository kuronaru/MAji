from torch import nn
from torch.nn import functional


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1):
        """
        :param input_channels: the number of channels of input x.
        :param num_channels: the number of channels of the output of the residual block.
        :param strides: the strides for the first convolutional layer in the residual block,
                        note that this is not applied to the second convolutional layer in the residual block.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        if input_channels == num_channels:
            self.conv3 = nn.Identity()
        else:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, padding=0)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = functional.relu(self.bn1(self.conv1(x)))
        y = functional.relu(self.bn2(self.conv2(y)) + self.conv3(x))
        return y


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = ResidualBlock(64, 64, 1)
        self.conv2_2 = ResidualBlock(64, 64, 1)
        self.conv3_1 = ResidualBlock(64, 128, 2)
        self.conv3_2 = ResidualBlock(128, 128, 1)
        self.conv4_1 = ResidualBlock(128, 256, 2)
        self.conv4_2 = ResidualBlock(256, 256, 1)
        self.conv5_1 = ResidualBlock(256, 512, 2)
        self.conv5_2 = ResidualBlock(512, 512, 1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        y = self.max_pool(self.conv1(x))
        y = self.conv2_2(self.conv2_1(y))
        y = self.conv3_2(self.conv3_1(y))
        y = self.conv4_2(self.conv4_1(y))
        y = self.conv5_2(self.conv5_1(y))
        y = functional.adaptive_avg_pool2d(y, 1)
        y = y.view(y.shape[0], -1)
        return functional.softmax(self.fc(y), 1)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        y = self.conv1(x)
        return y
