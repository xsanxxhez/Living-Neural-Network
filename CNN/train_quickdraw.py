#!/usr/bin/env python3
"""
Phase 2: Train a CNN on Quick Draw dataset
Downloads data, trains model, saves to quickdraw_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import urllib.request
import os
import gzip
import shutil
from tqdm import tqdm

# Configuration
CLASSES = ["cat", "sun", "tree", "house", "car", "fish", "flower", "star", "bird", "apple"]
SAMPLES_PER_CLASS = 5000  # Training samples
TEST_SAMPLES = 500
BATCH_SIZE = 128
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Training on {len(CLASSES)} classes: {CLASSES}")

class QuickDrawDataset(Dataset):
    """Dataset for Google Quick Draw sketches"""
    def __init__(self, class_name, samples, train=True):
        self.data = []
        self.labels = []

        # Download if not exists
        cache_dir = "quickdraw_cache"
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, f"{class_name}.npy")

        if not os.path.exists(filepath):
            print(f"Downloading {class_name}...")
            url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
            urllib.request.urlretrieve(url, filepath)

        # Load data
        print(f"Loading {class_name}...")
        data = np.load(filepath, mmap_mode='r')

        # Use subset
        if train:
            data = data[:SAMPLES_PER_CLASS]
        else:
            data = data[SAMPLES_PER_CLASS:SAMPLES_PER_CLASS + TEST_SAMPLES]

        # Preprocess: reshape to 28x28 and normalize
        data = data.reshape(-1, 28, 28).astype(np.float32) / 255.0

        self.data = data
        self.label = CLASSES.index(class_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Add channel dimension and apply simple augmentation for training
        img = self.data[idx]
        img = torch.from_numpy(img).unsqueeze(0)  # Shape: (1, 28, 28)
        return img, self.label

class QuickDrawCNN(nn.Module):
    """CNN for sketch recognition with hooks for visualization"""
    def __init__(self, num_classes=len(CLASSES)):
        super(QuickDrawCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        # Hook storage for visualization
        self.conv_outputs = {}
        self.fc_outputs = {}

    def register_hooks(self):
        """Register forward hooks to capture activations"""
        def get_hook(name, storage):
            def hook(module, input, output):
                storage[name] = output.detach()
            return hook

        self.conv1.register_forward_hook(get_hook('conv1', self.conv_outputs))
        self.conv2.register_forward_hook(get_hook('conv2', self.conv_outputs))
        self.conv3.register_forward_hook(get_hook('conv3', self.conv_outputs))
        self.fc1.register_forward_hook(get_hook('fc1', self.fc_outputs))

    def forward(self, x, return_activations=False):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if return_activations:
            return x, self.conv_outputs.copy(), self.fc_outputs.copy()
        return x

def train_model():
    """Train the CNN"""
    print("\nLoading datasets...")

    # Create datasets
    train_datasets = []
    test_datasets = []

    for cls in CLASSES:
        train_datasets.append(QuickDrawDataset(cls, SAMPLES_PER_CLASS, train=True))
        test_datasets.append(QuickDrawDataset(cls, TEST_SAMPLES, train=False))

    # Concatenate all classes
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize model
    model = QuickDrawCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f'{train_loss/total*BATCH_SIZE:.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        scheduler.step()

        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}: Train Acc = {100*correct/total:.2f}%, Test Acc = {100*test_correct/test_total:.2f}%")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': CLASSES,
        'model_class': 'QuickDrawCNN'
    }, 'quickdraw_model.pth')
    print("\n✅ Model saved to 'quickdraw_model.pth'")

    return model

if __name__ == "__main__":
    print("="*60)
    print("PHASE 2: Training Quick Draw CNN")
    print("="*60)

    # Check if model already exists
    if os.path.exists('quickdraw_model.pth'):
        print("\n⚠️  Model 'quickdraw_model.pth' already exists!")
        response = input("Retrain? (y/n): ")
        if response.lower() != 'y':
            print("Skipping training. Using existing model.")
            exit(0)

    model = train_model()
    print("\n🎉 Training complete! You can now run: python sketch_recognizer_phase2.py")
