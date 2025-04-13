import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve
import torch.nn.functional as F  # Importing torch.nn.functional for softmax


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=16):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjust based on image size
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model from .pth file
def load_model(model, model_path, device='cpu'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


# Preprocess the image (resize, normalize, etc.)
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    return image


# Function to test the model with a new image
def test_with_new_image(model, image_path, transform, class_name_to_label, device='cpu'):
    # Preprocess the image
    image = preprocess_image(image_path, transform)

    # Move the image to the correct device
    image = image.to(device)

    # Forward pass to get predictions
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get class with max probability

    predicted_class = list(class_name_to_label.keys())[predicted.item()]
    return predicted_class, outputs.cpu().numpy()


# Main function to run the model and make predictions on new image
def main(csv_file, image_dir, model_path, test_image_path):
    # Define class names and map them to numeric labels
    class_names = ['Chimney', 'Concrete', 'Construction Worker', 'Earth Mover',
                   'Electric Generator', 'Exacavated Pit', 'Land', 'Power Lines',
                   'Residential (Bathroom)', 'Residential (Bedroom)', 'Residential (Kitchen)',
                   'Solar Panel', 'Staircase', 'Tower Crane', 'Tree', 'Water Tank']
    class_name_to_label = {name: index for index, name in enumerate(class_names)}

    print(f"Unique class labels in the dataset: {class_name_to_label.keys()}")

    # Define transformations (resize to 256x256 for consistency)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = CNN(num_classes=len(class_names))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model, model_path, device)

    # Test with a new image
    print(f"Testing new image: {test_image_path}")
    predicted_class, _ = test_with_new_image(model, test_image_path, transform, class_name_to_label, device)
    print(f"Predicted class: {predicted_class}")

    # Optionally, visualize the image and its prediction
    img = Image.open(test_image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')
    plt.show()


# Example usage
csv_file = 'E:/annotations.csv'
image_dir = 'E:/Project_Yelloskye'
model_path = 'model_saves/model_epoch_30.pth'  # Path to the trained model .pth file
test_image_path = 'E:/Test_Dataset/38.jpg'  # Path to the new image you want to test

main(csv_file, image_dir, model_path, test_image_path)
