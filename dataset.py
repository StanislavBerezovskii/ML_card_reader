from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from pathlib import Path


current_dir = Path.cwd()
train_dir = f"{current_dir}/data_cards/train"
valid_dir = f"{current_dir}/data_cards/valid"
test_dir = f"{current_dir}/data_cards/test"
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


# Image alterations
transformation = transforms.Compose([
    transforms.Resize((128, 128)),  # resizing images
    transforms.ToTensor()  # transforming images to tensor
])

# Creating the datasets
train_dataset = PlayingCardDataset(data_dir=train_dir, transform=transformation)
valid_dataset = PlayingCardDataset(data_dir=valid_dir, transform=transformation)
test_dataset = PlayingCardDataset(data_dir=test_dir, transform=transformation)

# Creating the dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
