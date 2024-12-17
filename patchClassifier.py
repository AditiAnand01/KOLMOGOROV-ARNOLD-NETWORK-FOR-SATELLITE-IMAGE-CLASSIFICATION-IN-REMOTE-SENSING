import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm

from kan import KANLinear

class ConvNeXtKAN(nn.Module):
    def __init__(self, patch_size=56):
        super(ConvNeXtKAN, self).__init__()
        self.convnext = models.convnext_tiny(pretrained=True)

        # Adjust ConvNeXt for smaller patches
        self.convnext.features[0][0].stride = (1, 1) # Disable aggressive downsampling

        # Modify the classifier part of ConvNeXt
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()

        # Custom layers
        self.kan1 = KANLinear(num_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        # Reshape patches for ConvNeXt
        batch_size, num_patches, channels, height, width = x.shape
        x = x.view(batch_size * num_patches, channels, height, width)

        # Pass through ConvNeXt
        x = self.convnext(x)

        # Reshape back to separate batch and patch dimensions
        x = x.view(batch_size, num_patches, -1)

        # Aggregate patch features
        x = x.mean(dim=1)

        # Classification
        x = self.kan1(x)
        x = self.kan2(x)
        return x

# Define patch generation function
def split_into_patches(img, patch_size):
    """Split an image tensor into non-overlapping patches."""
    _, h, w = img.shape # C, H, W
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size) # Flatten patches
    return patches


# Define PatchDataset
class PatchDataset(Dataset):
    def __init__(self, dataset, patch_size):
        self.dataset = dataset
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        patches = split_into_patches(image, self.patch_size)
        return patches, label



# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for patches, labels in tqdm(dataloader, desc="Training"):
        patches, labels = patches.to(device), torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for patches, labels in tqdm(dataloader, desc="Evaluating"):
            patches, labels = patches.to(device), torch.tensor(labels).to(device)
            outputs = model(patches)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Main training script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    patch_size = 56
    num_classes = 10
    batch_size = 8 # Lower batch size due to large patch tensors
    num_epochs = 10
    learning_rate = 0.001

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = PatchDataset(
        datasets.ImageFolder(root=r'C:\VSCode\KAN_satellite\Classification\EuroSAT_Dataset\train', transform=train_transform),
        patch_size=patch_size,
    )
    test_dataset = PatchDataset(
        datasets.ImageFolder(root=r'C:\VSCode\KAN_satellite\Classification\EuroSAT_Dataset\test', transform=test_transform),
        patch_size=patch_size,
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, loss, and optimizer
    model = ConvNeXtKAN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        evaluate(model, test_loader, criterion, device)

    # Save the trained model
    torch.save(model.state_dict(), "convnext_kan_patch_classifier.pth")
    print("Model saved as 'convnext_kan_patch_classifier.pth'")
