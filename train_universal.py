"""
train_universal.py — Train 62-class Universal CNN on preprocessed EMNIST data.

Usage: First run data extraction, then: python train_universal.py
The data files models/train_subset.pt and models/test_data.pt must exist.

Supports checkpoint resumption: if models/training_ckpt.pt exists, resumes from it.
"""

import os, time, random, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

os.makedirs('models', exist_ok=True)
NUM_CLASSES = 62
EMNIST_MEAN = 0.1736
EMNIST_STD  = 0.3317


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se_reduction=16, drop_rate=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=se_reduction)
        self.drop = nn.Dropout2d(drop_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class UniversalCNN(nn.Module):
    """Must match the class in app.py exactly."""
    def __init__(self):
        super(UniversalCNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(32, 64,  num_blocks=2, stride=2, drop_rate=0.05)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2, drop_rate=0.10)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2, drop_rate=0.15)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride, drop_rate):
        layers = [ResBlock(in_ch, out_ch, stride=stride, se_reduction=16, drop_rate=drop_rate)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1, se_reduction=16, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = F.dropout(F.relu(self.fc1_bn(self.fc1(x)), inplace=True), 0.4, self.training)
        return self.fc2(x)


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


def pflush(msg):
    """Print and flush immediately."""
    print(msg, flush=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    pflush(f"Device: {device} | Threads: {torch.get_num_threads()}")
    pflush(f"Training {NUM_CLASSES}-class Universal CNN (ResNet + SE Attention)")
    pflush("=" * 60)

    # Load preprocessed data
    pflush("\nLoading preprocessed training data...")
    d = torch.load('models/train_subset.pt', map_location='cpu')
    X_train, y_train = d['images'], d['labels']
    pflush(f"  Train: {X_train.shape}")

    pflush("Loading preprocessed test data...")
    d = torch.load('models/test_small.pt', map_location='cpu')
    X_test, y_test = d['images'], d['labels']
    pflush(f"  Test: {X_test.shape}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False, num_workers=0)

    # Free raw tensors (DataLoader holds copies)
    del X_train, y_train, X_test, y_test, d
    import gc; gc.collect()

    model = UniversalCNN().to(device)
    pflush(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs = 40
    start_epoch = 0

    # Check for checkpoint to resume from
    ckpt_path = 'models/training_ckpt.pt'
    if os.path.exists(ckpt_path):
        pflush(f"\nResuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        wait = ckpt.get('wait', 0)
        pflush(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%, wait={wait}")
    else:
        best_acc = 0.0
        wait = 0

    scheduler = OneCycleLR(optimizer, max_lr=3e-3, steps_per_epoch=len(train_loader),
                           epochs=epochs, pct_start=0.1, anneal_strategy='cos',
                           div_factor=25.0, final_div_factor=1000.0)

    # If resuming, step scheduler to the right position
    if start_epoch > 0:
        total_steps_so_far = start_epoch * len(train_loader)
        for _ in range(total_steps_so_far):
            scheduler.step()
        pflush(f"  Scheduler stepped {total_steps_so_far} times to resume position")

    patience = 12
    label_map = ([str(i) for i in range(10)] +
                 [chr(c) for c in range(ord('A'), ord('Z')+1)] +
                 [chr(c) for c in range(ord('a'), ord('z')+1)])

    pflush(f"\nTraining {epochs} epochs (patience={patience}), {len(train_loader)} steps/epoch")
    pflush(f"Starting from epoch {start_epoch + 1}")
    pflush("=" * 60)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        running_loss = correct = total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if epoch < 25:
                mx, ya, yb, lam = mixup_data(images, labels, alpha=0.2)
                outputs = model(mx)
                loss = mixup_criterion(criterion, outputs, ya, yb, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % 100 == 0:
                pflush(f"  Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] "
                      f"Loss: {running_loss/(i+1):.4f} Acc: {100*correct/total:.1f}%")

        train_acc = 100.0 * correct / total
        elapsed = time.time() - t0

        # Validation
        model.eval()
        correct = total = 0
        class_correct = [0]*NUM_CLASSES; class_total = [0]*NUM_CLASSES

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for j in range(labels.size(0)):
                    lbl = labels[j].item()
                    if lbl < NUM_CLASSES:
                        class_total[lbl] += 1
                        if predicted[j] == lbl: class_correct[lbl] += 1

        acc = 100.0 * correct / total
        da = 100.0*sum(class_correct[:10])/max(sum(class_total[:10]),1)
        ua = 100.0*sum(class_correct[10:36])/max(sum(class_total[10:36]),1)
        la = 100.0*sum(class_correct[36:])/max(sum(class_total[36:]),1)

        pflush(f"\n>>> Epoch {epoch+1}/{epochs} ({elapsed:.0f}s) "
              f"Train: {train_acc:.1f}% Test: {acc:.2f}% "
              f"Digits: {da:.1f}% Upper: {ua:.1f}% Lower: {la:.1f}%")

        torch.save(model.state_dict(), 'models/universal_cnn.pth')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/universal_cnn_best.pth')
            pflush(f"  ** New best: {acc:.2f}% **")
            wait = 0
        else:
            wait += 1
            pflush(f"  No improvement {wait}/{patience} (best: {best_acc:.2f}%)")

        # Save checkpoint after every epoch for resumption
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'wait': wait,
        }, ckpt_path)
        pflush(f"  Checkpoint saved: epoch {epoch+1}")

        if wait >= patience:
            pflush(f"\nEarly stopping epoch {epoch+1}. Best: {best_acc:.2f}%")
            break

    # Final evaluation
    pflush(f"\n{'='*60}\nTraining complete! Best: {best_acc:.2f}%")
    model.load_state_dict(torch.load('models/universal_cnn_best.pth', map_location=device))
    model.eval()

    correct = total = 0
    class_correct = [0]*NUM_CLASSES; class_total = [0]*NUM_CLASSES
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for j in range(labels.size(0)):
                lbl = labels[j].item()
                if lbl < NUM_CLASSES:
                    class_total[lbl] += 1
                    if predicted[j] == lbl: class_correct[lbl] += 1

    pflush(f"\nFinal Test Accuracy: {100.0*correct/total:.2f}%")
    for group_name, start, end in [("Digits",0,10),("Uppercase",10,36),("Lowercase",36,62)]:
        pflush(f"\n--- {group_name} ---")
        for i in range(start, end):
            pct = 100.0*class_correct[i]/max(class_total[i],1)
            pflush(f"  {label_map[i]}: {pct:.1f}%")

    da = 100.0*sum(class_correct[:10])/max(sum(class_total[:10]),1)
    ua = 100.0*sum(class_correct[10:36])/max(sum(class_total[10:36]),1)
    la = 100.0*sum(class_correct[36:])/max(sum(class_total[36:]),1)
    pflush(f"\nCategory Averages:\n  Digits: {da:.1f}%\n  Upper: {ua:.1f}%\n  Lower: {la:.1f}%\n  Overall: {100.0*correct/total:.2f}%")

    # Clean up data files only after training is fully complete
    for f in ['models/train_subset.pt', 'models/test_small.pt', ckpt_path]:
        if os.path.exists(f): os.remove(f); pflush(f"Cleaned: {f}")


if __name__ == "__main__":
    train()
