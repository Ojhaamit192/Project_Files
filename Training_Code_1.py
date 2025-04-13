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
from tqdm import tqdm  # Import tqdm for progress bar
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize


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


# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, class_name_to_label, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.class_name_to_label = class_name_to_label
        self.transform = transform

        # Filter out rows with invalid class names
        self.valid_data = self.filter_invalid_data()

    def filter_invalid_data(self):
        valid_data = []
        for _, row in self.data.iterrows():
            class_name = row['class']  # Assuming class name is in 'class' column
            if class_name in self.class_name_to_label:
                valid_data.append(row)
        return valid_data

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        row = self.valid_data[idx]
        img_name = os.path.join(self.image_dir, row['filename'])  # Assuming filename is in 'filename' column
        image = Image.open(img_name).convert('RGB')

        # Get label from class_name_to_label dictionary
        class_name = row['class']  # Assuming class name is in 'class' column
        label = self.class_name_to_label[class_name]  # Convert class name to label

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Training function with additional metrics and debugging output
def train_model(train_loader, model, criterion, optimizer, num_epochs=1, device='cpu', save_dir='model_saves'):
    model.train()

    # Lists to store data for plotting
    train_losses = []
    train_accuracies = []
    all_labels = []
    all_predictions = []
    all_pred_probs = []  # Store prediction probabilities for Precision-Recall curves

    os.makedirs(save_dir, exist_ok=True)  # Ensure model save directory exists

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Use tqdm to create a progress bar
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            # Send inputs and labels to the correct device (CPU or GPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Ensure inputs and labels have matching batch sizes
            assert inputs.size(0) == labels.size(0), f"Batch size mismatch: {inputs.size(0)} != {labels.size(0)}"

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            # Store labels and predictions for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Store prediction probabilities for Precision-Recall curves
            all_pred_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy())

        # Metrics calculation
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

        # Print metrics for current epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Store loss and accuracy for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Save the model checkpoint after every epoch in .pth format
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)  # Save model state_dict in .pth format
        print(f"Model saved to {model_save_path}")

        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_loader.dataset.class_name_to_label.keys(),
                    yticklabels=train_loader.dataset.class_name_to_label.keys())
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        # Calculate Precision-Recall for each class
        all_labels_bin = label_binarize(all_labels, classes=np.unique(all_labels))  # One-hot encode true labels

        # For each class, calculate the Precision-Recall curve
        for i in range(len(np.unique(all_labels))):  # For each class
            precision_vals, recall_vals, _ = precision_recall_curve(all_labels_bin[:, i], np.array(all_pred_probs)[:, i])

            # Plot precision-recall curve
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, color='b', label=f'Class {i} Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for Class {i} - Epoch {epoch + 1}')
            plt.legend()
            plt.show()

    # Plot Loss and Accuracy over Epochs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()


# Main function to start training
def main(csv_file, image_dir):
    # Define the class names and map to numeric labels
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

    # Load the dataset
    dataset = CustomDataset(csv_file, image_dir, class_name_to_label, transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = CNN(num_classes=len(class_names)).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs=30, device='cuda' if torch.cuda.is_available() else 'cpu')


# Run the training
csv_file = 'E:/annotations.csv'
image_dir = 'E:/Project_Yelloskye'
main(csv_file, image_dir)




