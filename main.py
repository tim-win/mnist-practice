#!/usr/bin/env python
import torch
from torch.nn import (
    Linear,
    Conv2d,
    ReLU,
    MaxPool2d,
    Dropout,
    AdaptiveAvgPool2d,
    Sequential,
    BatchNorm2d)
from torch.nn import Module
import torch.functional as F
from data_loaders import generate_dataset
from torch.utils.data import DataLoader

import os

image_files = [
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte',
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte'
]


class MNISTNet(Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.features = Sequential(
        Linear(
            in_features=28*28,
            out_features=14*14),
        ReLU(),
        Linear(
            in_features=14*14,
            out_features=10))

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.features(x)
        return x


class CNNNet(Module):

    def __init__(self, num_classes=10):
        super(CNNNet, self).__init__()
        self.features = Sequential(
            Conv2d(1, 256, kernel_size=5, stride=1, padding=2),
            ReLU(),
            BatchNorm2d(256),
            Conv2d(256, 64, kernel_size=3, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            BatchNorm2d(64),
            Conv2d(64, 32, kernel_size=3, padding=2),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(32, 16, kernel_size=3, padding=2),
            ReLU(),
            BatchNorm2d(16),
            # BatchNorm2d(32),
            # MaxPool2d(kernel_size=3, stride=2),
            # Conv2d(32, 32, kernel_size=3, padding=2),
            # ReLU(),
            # BatchNorm2d(32),
            # MaxPool2d(kernel_size=3, stride=2),
            # Conv2d(32, 32, kernel_size=3, padding=2),
            # ReLU(),
            # Conv2d(192, 384, kernel_size=3, padding=1),
            # ReLU(),
            # Conv2d(384, 256, kernel_size=3, padding=1),
            # ReLU(),
            # Conv2d(256, 256, kernel_size=3, padding=1),e
            # ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2d((6, 6))
        self.classifier = Sequential(
            Dropout(),
            Linear(16 * 6 * 6, 64),
            ReLU(),
            Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNNNet9910(Module):

    def __init__(self, num_classes=10):
        super(CNNNet, self).__init__()
        self.features = Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            BatchNorm2d(64),
            Conv2d(64, 32, kernel_size=3, padding=2),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(32, 16, kernel_size=3, padding=2),
            ReLU(),
            BatchNorm2d(16),
            # BatchNorm2d(32),
            # MaxPool2d(kernel_size=3, stride=2),
            # Conv2d(32, 32, kernel_size=3, padding=2),
            # ReLU(),
            # BatchNorm2d(32),
            # MaxPool2d(kernel_size=3, stride=2),
            # Conv2d(32, 32, kernel_size=3, padding=2),
            # ReLU(),
            # Conv2d(192, 384, kernel_size=3, padding=1),
            # ReLU(),
            # Conv2d(384, 256, kernel_size=3, padding=1),
            # ReLU(),
            # Conv2d(256, 256, kernel_size=3, padding=1),e
            # ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2d((6, 6))
        self.classifier = Sequential(
            Dropout(),
            Linear(16 * 6 * 6, 64),
            ReLU(),
            Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    return 0


if __name__ == '__main__':
    sys.exit(main())
