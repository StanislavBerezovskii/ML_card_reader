from glob import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from dataset import test_dir, transformation, train_dataset
from model import model
from training_loop import device


# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


# Visualize predictions
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim([0, 1])

    plt.tight_layout()
    plt.show()


# Example usage
"""test_image = f"{test_dir}/five of diamonds/2.jpg"
original_image, image_tensor = preprocess_image(test_image, transformation)
probabilities = predict(model, image_tensor, device)

class_names = train_dataset.classes
visualize_predictions(original_image, probabilities, class_names)"""

test_images = glob(f"{test_dir}/*/*")
test_examples = np.random.choice(test_images, 10)


if __name__ == "__main__":
    # Draw 10 cards at random from the test set and visualize the predictions
    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, transformation)
        probabilities = predict(model, image_tensor, device)
        class_names = train_dataset.classes
        visualize_predictions(original_image, probabilities, class_names)
