"""
train_universal.py — Trains a 62-class CNN on EMNIST ByClass dataset.
Classes: 0-9 (digits) + A-Z (uppercase, 10-35) + a-z (lowercase, 36-61)

Run: python train_universal.py
Output: models/universal_cnn.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.makedirs('models', exist_ok=True)

# ── EMNIST ByClass has 62 classes ──────────────────────────────────────────────
# Class indices in EMNIST ByClass:
#   0-9   → digits '0'-'9'
#   10-35 → uppercase 'A'-'Z'
#   36-61 → lowercase 'a'-'z'
NUM_CLASSES = 62


class UniversalCNN(nn.Module):
    """Deeper CNN for 62-class EMNIST ByClass recognition."""
    def __init__(self):
        super(UniversalCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28→14
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14→7
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 7→3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Training {NUM_CLASSES}-class Universal CNN (EMNIST ByClass)")
    print("=" * 60)

    # ── Transforms ────────────────────────────────────────────────────────────
    # EMNIST images need to be transposed (they're rotated 90° in raw form)
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,)),   # EMNIST ByClass mean/std
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,)),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nDownloading EMNIST ByClass (this may take a few minutes ~500MB)...")
    train_set = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=True, download=True,
        transform=transform_train
    )
    test_set = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=False, download=True,
        transform=transform_test
    )

    print(f"Train samples: {len(train_set):,}  |  Test samples: {len(test_set):,}")

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model, Loss, Optimizer ────────────────────────────────────────────────
    model     = UniversalCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    best_acc  = 0.0
    epochs    = 15

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}]  Step [{i+1}/{len(train_loader)}]  "
                      f"Loss: {running_loss/200:.4f}")
                running_loss = 0.0

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        print(f"\n>>> Epoch {epoch+1}/{epochs}  Test Accuracy: {acc:.2f}%\n")
        scheduler.step(acc)

        # ── Save best checkpoint ───────────────────────────────────────────────
        torch.save(model.state_dict(), 'models/universal_cnn.pth')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/universal_cnn_best.pth')
            print(f"  ★ New best model saved ({acc:.2f}%)")

    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    print("Saved: models/universal_cnn.pth  &  models/universal_cnn_best.pth")


if __name__ == "__main__":
    train()
