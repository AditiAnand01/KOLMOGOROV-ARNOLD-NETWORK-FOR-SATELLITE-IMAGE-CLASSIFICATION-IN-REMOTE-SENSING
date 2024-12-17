import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from kcn import ConvNeXtKAN


def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    """Plots and saves the confusion matrix as an image."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()


# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    

# Testing function
def test(model, test_loader, criterion, device, class_names, log_file='metrics_log.txt'):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass to get predictions
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Store predictions and true labels for metrics
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall loss
    epoch_loss = running_loss / len(test_loader.dataset)

    # Generate metrics (Precision, Recall, F1-score)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    accuracy = report['accuracy'] * 100  # Accuracy in percentage

    # Save the metrics to a log file
    with open(log_file, 'a') as f:
        f.write(f'Test Loss: {epoch_loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.2f}%\n')
        f.write('Classification Report:\n')
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write('\n\n')

    print(f'Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Create and save the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(conf_matrix, class_names)




