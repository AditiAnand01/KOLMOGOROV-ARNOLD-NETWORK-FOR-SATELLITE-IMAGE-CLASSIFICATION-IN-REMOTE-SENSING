import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from kcn import ConvNeXtKAN
from train_and_test import train,test



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for training and testing data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Standard normalization for pre-trained models
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Datasets
train_dataset = datasets.ImageFolder(root=r'C:\VSCode\KAN_satellite\Dataset\train', transform=train_transform)
test_dataset = datasets.ImageFolder(root=r'C:\VSCode\KAN_satellite\Dataset\test', transform=test_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

class_names = train_dataset.classes

# Model, Loss, and Optimizer
model = ConvNeXtKAN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train(model, train_loader, criterion, optimizer, device)
    test(model, test_loader, criterion, device, class_names)

# Save the trained model
torch.save(model.state_dict(), 'convnext_kan.pth')
print("Model saved as 'convnext_kan.pth'")