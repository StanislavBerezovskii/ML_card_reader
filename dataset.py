import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd


current_dir = Path.cwd()
train_dir = f"{current_dir}/data_cards/train"
target_to_class = {v: k for k, v in ImageFolder(train_dir).class_to_idx.items()}


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


transformation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = PlayingCardDataset(data_dir=train_dir, transform=transformation)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    print(train_dir)
    print(len(dataset))
    print(dataset[0])
    print(target_to_class)
    print(dataset.classes)
    image, label = dataset[0]
    print(image.shape)
