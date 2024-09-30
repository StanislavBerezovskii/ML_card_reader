import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import (current_dir, train_loader, test_loader, valid_loader)
from model import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# example_criterion = criterion(example_output, labels)

num_epochs = 5
train_losses, val_losses = [], []

def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # Setting the model to train
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training loop"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Setting the model to validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validation loop"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)
        # Printing Epoch stats
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}")


if __name__ == "__main__":
    train_model(model=model, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)
    file_name = os.path.join(current_dir, "global_2000_sticker_classifier.pth")
    torch.save(model, file_name)
