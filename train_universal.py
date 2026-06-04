"""
train_universal.py — Trains a 62-class CNN on EMNIST ByClass dataset.

KEY FIXES (v2) over the original:
1. IMAGE ORIENTATION: Transposes EMNIST images to match how users draw on screen.
   The original model was trained on transposed (column-major) EMNIST images,
   causing ~8% accuracy on user-drawn input.
2. IMAGE INVERSION: Inverts EMNIST images so stroke=bright, bg=dark, matching
   the app.py preprocess_image pipeline which inverts canvas drawings.
3. DEEPER ARCHITECTURE: Residual-style CNN with skip connections and GAP.
   ~1.4M parameters for strong 62-class discrimination.
4. LABEL SMOOTHING: Prevents overconfident predictions.
5. COSINE ANNEALING: Better learning rate scheduling.

The trained model works with the EXISTING app.py preprocessing:
  Canvas (black stroke on white bg) → invert → resize 28x28 → normalize

Run: python train_universal.py
Output: models/universal_cnn_best.pth
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

os.makedirs('models', exist_ok=True)

NUM_CLASSES = 62
EMNIST_MEAN = 0.1736
EMNIST_STD  = 0.3317


class EMNISTForApp(Dataset):
    """Prepares EMNIST images to match exactly what app.py preprocess_image produces.

    The app.py pipeline:
      1. Canvas: user draws black stroke on white background
      2. Invert: white stroke on black background
      3. Resize to 28x28
      4. Normalize: (pixel - 0.1736) / 0.3317

    This dataset applies:
      1. Transpose: correct EMNIST column-major orientation
      2. Invert: 1.0 - pixel (dark stroke on white bg → bright stroke on dark bg)
      3. Normalize: same as app.py
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img_np = img.numpy()
        # Transpose spatial dims (EMNIST stores images in column-major order)
        img_corrected = np.ascontiguousarray(img_np.transpose(0, 2, 1))
        # Invert: EMNIST has dark stroke on white bg, app sends bright stroke on dark bg
        img_inverted = 1.0 - img_corrected
        # Normalize with EMNIST stats (same as app.py)
        img_normalized = (img_inverted - EMNIST_MEAN) / EMNIST_STD
        return torch.from_numpy(img_normalized.astype('float32')), label


class UniversalCNN(nn.Module):
    """Deeper CNN with residual connections for 62-class EMNIST ByClass.

    Architecture:
      - 3 convolutional stages with residual (skip) connections
      - Global average pooling
      - Larger classifier with batch normalization
      - ~1.4M parameters

    Must match the class in app.py exactly.
    """
    def __init__(self):
        super(UniversalCNN, self).__init__()
        # Block 1: 1 → 64 channels, 28→14
        self.b1_conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.b1_bn1   = nn.BatchNorm2d(64)
        self.b1_conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.b1_bn2   = nn.BatchNorm2d(64)
        # Block 2: 64 → 128 channels, 14→7
        self.b2_conv1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.b2_bn1   = nn.BatchNorm2d(128)
        self.b2_conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.b2_bn2   = nn.BatchNorm2d(128)
        # Block 3: 128 → 256 channels, 7→3
        self.b3_conv1 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.b3_bn1   = nn.BatchNorm2d(256)
        self.b3_conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.b3_bn2   = nn.BatchNorm2d(256)
        # GAP + Classifier
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Linear(256, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2    = nn.Linear(512, 256)
        self.fc3    = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.b1_bn1(self.b1_conv1(x)), inplace=True)
        identity = x
        x = F.relu(self.b1_bn2(self.b1_conv2(x)) + identity, inplace=True)
        x = F.max_pool2d(F.dropout2d(x, 0.1, self.training), 2)

        x = F.relu(self.b2_bn1(self.b2_conv1(x)), inplace=True)
        identity = x
        x = F.relu(self.b2_bn2(self.b2_conv2(x)) + identity, inplace=True)
        x = F.max_pool2d(F.dropout2d(x, 0.2, self.training), 2)

        x = F.relu(self.b3_bn1(self.b3_conv1(x)), inplace=True)
        identity = x
        x = F.relu(self.b3_bn2(self.b3_conv2(x)) + identity, inplace=True)
        x = F.max_pool2d(F.dropout2d(x, 0.25, self.training), 2)

        x = self.gap(x).flatten(1)
        x = F.dropout(F.relu(self.fc1_bn(self.fc1(x)), inplace=True), 0.5, self.training)
        x = F.dropout(F.relu(self.fc2(x), inplace=True), 0.3, self.training)
        return self.fc3(x)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Training {NUM_CLASSES}-class Universal CNN (EMNIST ByClass)")
    print("With corrected orientation AND inversion to match app.py preprocessing")
    print("=" * 60)

    # ── Transforms (augmentation only; normalization done in EMNISTForApp) ───
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5
        ),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # ── Dataset ─────────────────────────────────────────────────────────────
    print("\nLoading EMNIST ByClass dataset...")
    raw_train = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=True, download=True,
        transform=transform_train
    )
    train_set = EMNISTForApp(raw_train)

    raw_test = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=False, download=True,
        transform=transform_test
    )
    test_set = EMNISTForApp(raw_test)

    print(f"Train samples: {len(train_set):,}  |  Test samples: {len(test_set):,}")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # ── Model ───────────────────────────────────────────────────────────────
    model     = UniversalCNN().to(device)
    params    = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    best_acc  = 0.0
    epochs    = 25

    for epoch in range(epochs):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 500 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}]  Step [{i+1}/{len(train_loader)}]  "
                      f"Loss: {running_loss/(i+1):.4f}  Acc: {100*correct/total:.1f}%")

        train_acc = 100.0 * correct / total
        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        class_correct = [0] * NUM_CLASSES
        class_total   = [0] * NUM_CLASSES

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

                for j in range(labels.size(0)):
                    lbl = labels[j].item()
                    class_total[lbl] += 1
                    if predicted[j] == lbl:
                        class_correct[lbl] += 1

        acc = 100.0 * correct / total
        digit_acc = 100.0 * sum(class_correct[:10]) / max(sum(class_total[:10]), 1)
        upper_acc = 100.0 * sum(class_correct[10:36]) / max(sum(class_total[10:36]), 1)
        lower_acc = 100.0 * sum(class_correct[36:]) / max(sum(class_total[36:]), 1)

        print(f"\n>>> Epoch {epoch+1}/{epochs}  Train: {train_acc:.1f}%  Test: {acc:.2f}%  "
              f"Digits: {digit_acc:.1f}%  Upper: {upper_acc:.1f}%  Lower: {lower_acc:.1f}%")

        # ── Save checkpoints ────────────────────────────────────────────────
        torch.save(model.state_dict(), 'models/universal_cnn.pth')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/universal_cnn_best.pth')
            print(f"  ★ New best model saved ({acc:.2f}%)")

    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    print("Saved: models/universal_cnn.pth  &  models/universal_cnn_best.pth")
    print("Model is compatible with the existing app.py preprocessing pipeline.")


if __name__ == "__main__":
    train()
