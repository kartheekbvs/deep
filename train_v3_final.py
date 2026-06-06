#!/usr/bin/env python3
"""
train_v3_final.py — CharSenseNet-V3 Training Script

Downloads EMNIST ByClass data, creates a balanced training subset,
and trains the CharSenseNet-V3 model with CBAM attention.

Usage:
  python3 train_v3_final.py              # Full training
  SKIP_TRAINING=1 python3 train_v3_final.py  # Skip if model exists

Training techniques:
  - Mixup data augmentation (first 25 epochs)
  - Random shift augmentation
  - Label smoothing (0.1)
  - OneCycleLR scheduler
  - Gradient clipping (max_norm=1.0)
  - AdamW optimizer with weight decay

Target accuracy: 90%+ on EMNIST ByClass 62 classes
"""
import os, random, time, gc, sys

random.seed(42)

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
np.random.seed(42)

NUM_CLASSES = 62
EMNIST_MEAN = 0.1736
EMNIST_STD  = 0.3317


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (must match app.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.fc(F.adaptive_avg_pool2d(x, 1).view(b, c))
        return x * y.view(b, c, 1, 1)


class CBAM(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.channel_fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.channel_fc(F.adaptive_avg_pool2d(x, 1).view(b, c))
        max_out = self.channel_fc(F.adaptive_max_pool2d(x, 1).view(b, c))
        ch_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * ch_att
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True)[0]
        sp_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        x = x * sp_att
        return x


class ResBlockV3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, attention='se', drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.attn = CBAM(out_ch) if attention == 'cbam' else SEBlock(out_ch)
        self.drop = nn.Dropout2d(drop_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.attn(out)
        out += identity
        return F.relu(out, inplace=True)


class CharSenseNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(32, 64, 3, 2, 'se', 0.02)
        self.stage2 = self._make_stage(64, 128, 3, 2, 'se', 0.05)
        self.stage3 = self._make_stage(128, 256, 3, 2, 'cbam', 0.10)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride, attention, drop_rate):
        layers = [ResBlockV3(in_ch, out_ch, stride=stride, attention=attention, drop_rate=drop_rate)]
        for _ in range(1, num_blocks):
            layers.append(ResBlockV3(out_ch, out_ch, stride=1, attention=attention, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = F.dropout(F.relu(self.fc1_bn(self.fc1(x)), inplace=True), 0.3, self.training)
        return self.fc2(x)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_emnist_data():
    """Download EMNIST ByClass and create balanced training subset."""
    from torchvision.datasets import EMNIST

    data_dir = 'emnist_data'
    os.makedirs(data_dir, exist_ok=True)

    print('Downloading EMNIST ByClass...', flush=True)
    ds_train = EMNIST(root=data_dir, split='byclass', train=True, download=True)
    ds_test = EMNIST(root=data_dir, split='byclass', train=False, download=True)
    print(f'  Train: {len(ds_train)}, Test: {len(ds_test)}', flush=True)

    # Process training data - balanced subset of 150K
    print('Processing training data (balanced 150K)...', flush=True)
    target_per_class = 150000 // NUM_CLASSES
    X_list, y_list = [], []
    class_buffers = {c: [] for c in range(NUM_CLASSES)}

    for i in range(len(ds_train)):
        img, label = ds_train[i]
        if label >= NUM_CLASSES:
            continue
        arr = np.array(img).astype('float32') / 255.0
        arr = np.transpose(arr, (1, 0))[:, ::-1]  # Fix EMNIST orientation
        class_buffers[label].append(arr)

    for cls in range(NUM_CLASSES):
        bufs = class_buffers[cls]
        if len(bufs) > target_per_class:
            indices = random.sample(range(len(bufs)), target_per_class)
            bufs = [bufs[i] for i in indices]
        X_list.extend(bufs)
        y_list.extend([cls] * len(bufs))

    random.shuffle(list(zip(X_list, y_list)))
    X_train = np.stack(X_list)
    y_train = np.array(y_list, dtype=np.int64)
    del X_list, y_list, class_buffers; gc.collect()

    # Process test data - balanced subset of 50K
    print('Processing test data (balanced 50K)...', flush=True)
    target_test = 50000 // NUM_CLASSES
    X_list, y_list = [], []
    class_buffers = {c: [] for c in range(NUM_CLASSES)}

    for i in range(len(ds_test)):
        img, label = ds_test[i]
        if label >= NUM_CLASSES:
            continue
        arr = np.array(img).astype('float32') / 255.0
        arr = np.transpose(arr, (1, 0))[:, ::-1]
        class_buffers[label].append(arr)

    for cls in range(NUM_CLASSES):
        bufs = class_buffers[cls]
        if len(bufs) > target_test:
            indices = random.sample(range(len(bufs)), target_test)
            bufs = [bufs[i] for i in indices]
        X_list.extend(bufs)
        y_list.extend([cls] * len(bufs))

    X_test = np.stack(X_list)
    y_test = np.array(y_list, dtype=np.int64)
    del X_list, y_list, class_buffers; gc.collect()

    print(f'  Train: {X_train.shape}, Test: {X_test.shape}', flush=True)

    # Normalize and convert to tensors
    X_train = (X_train - EMNIST_MEAN) / EMNIST_STD
    X_test = (X_test - EMNIST_MEAN) / EMNIST_STD
    X_train_t = torch.from_numpy(X_train).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test)

    # Clean up raw data
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)

    return X_train_t, y_train_t, X_test_t, y_test_t


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 4)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'CharSenseNet-V3 Training - {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'Device: {device} | Threads: {torch.get_num_threads()}', flush=True)
    print('=' * 60, flush=True)

    # Skip if model already exists
    model_path = 'models/universal_cnn_best.pth'
    if os.path.exists(model_path):
        print(f'Model already exists: {model_path}', flush=True)
        print('Skipping training. Delete the model to retrain.', flush=True)
        return

    # Load data
    X_train, y_train, X_test, y_test = load_emnist_data()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256,
                             shuffle=False, num_workers=0)
    del X_train, y_train, X_test, y_test; gc.collect()

    # Model
    model = CharSenseNetV3().to(device)
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params', flush=True)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_path = 'models/v3_ckpt.pt'
    start_epoch = 0
    best_acc = 0.0
    wait = 0
    patience = 10
    epochs = 40

    # Resume from checkpoint
    if os.path.exists(ckpt_path):
        print(f'Resuming from checkpoint...', flush=True)
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c['model'])
        opt.load_state_dict(c['opt'])
        start_epoch = c['epoch']
        best_acc = c['best_acc']
        wait = c.get('wait', 0)
        print(f'  Epoch {start_epoch}, best={best_acc:.2f}%', flush=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1000.0
    )

    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    label_map = ([str(i) for i in range(10)] +
                 [chr(c) for c in range(65, 91)] +
                 [chr(c) for c in range(97, 123)])

    print(f'Training {epochs} epochs (patience={patience}), {len(train_loader)} steps/epoch', flush=True)
    print('=' * 60, flush=True)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        rl = cn = tn = 0

        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Random shift augmentation
            if random.random() < 0.5:
                imgs = torch.roll(imgs, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))

            opt.zero_grad()

            # Mixup for first 25 epochs
            if epoch < 25:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(imgs.size(0), device=device)
                mx = lam * imgs + (1 - lam) * imgs[idx]
                out = model(mx)
                loss = lam * crit(out, lbls) + (1 - lam) * crit(out, lbls[idx])
            else:
                out = model(imgs)
                loss = crit(out, lbls)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            rl += loss.item()
            _, pred = torch.max(out, 1)
            tn += lbls.size(0)
            cn += (pred == lbls).sum().item()

            if (i + 1) % 200 == 0:
                print(f'  E{epoch+1} S{i+1} L:{rl/(i+1):.4f} A:{100*cn/tn:.1f}%', flush=True)

        train_acc = 100 * cn / tn
        elapsed = time.time() - t0

        # Validation
        model.eval()
        vc = vt = 0
        cc = [0] * NUM_CLASSES
        ct = [0] * NUM_CLASSES
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, pred = torch.max(model(imgs), 1)
                vt += lbls.size(0)
                vc += (pred == lbls).sum().item()
                for j in range(lbls.size(0)):
                    l = lbls[j].item()
                    if l < NUM_CLASSES:
                        ct[l] += 1
                        if pred[j] == l:
                            cc[l] += 1

        acc = 100 * vc / vt
        da = 100 * sum(cc[:10]) / max(sum(ct[:10]), 1)
        ua = 100 * sum(cc[10:36]) / max(sum(ct[10:36]), 1)
        la = 100 * sum(cc[36:]) / max(sum(ct[36:]), 1)

        print(f'>>> E{epoch+1}/{epochs} ({elapsed:.0f}s) T:{train_acc:.1f}% V:{acc:.2f}% '
              f'D:{da:.1f}% U:{ua:.1f}% L:{la:.1f}%', flush=True)

        # Save model
        torch.save(model.state_dict(), 'models/universal_cnn_v3.pth')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f'  ** New best: {acc:.2f}% **', flush=True)
            wait = 0
        else:
            wait += 1
            print(f'  No improvement {wait}/{patience} (best: {best_acc:.2f}%)', flush=True)

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'best_acc': best_acc,
            'wait': wait,
        }, ckpt_path)

        if wait >= patience:
            print(f'Early stopping at epoch {epoch+1}. Best: {best_acc:.2f}%', flush=True)
            break

    # Final evaluation
    print(f'\n{"="*60}', flush=True)
    print(f'Training complete! Best: {best_acc:.2f}%', flush=True)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    vc = vt = 0
    cc = [0] * NUM_CLASSES
    ct = [0] * NUM_CLASSES
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            _, pred = torch.max(model(imgs), 1)
            vt += lbls.size(0)
            vc += (pred == lbls).sum().item()
            for j in range(lbls.size(0)):
                l = lbls[j].item()
                if l < NUM_CLASSES:
                    ct[l] += 1
                    if pred[j] == l:
                        cc[l] += 1

    print(f'\nFinal Test Accuracy: {100*vc/vt:.2f}%', flush=True)
    for gn, s, e in [('Digits', 0, 10), ('Uppercase', 10, 36), ('Lowercase', 36, 62)]:
        a = 100 * sum(cc[s:e]) / max(sum(ct[s:e]), 1)
        print(f'  {gn}: {a:.1f}%', flush=True)

    # Cleanup
    for f in [ckpt_path, 'models/universal_cnn_v3.pth']:
        if os.path.exists(f):
            os.remove(f)
    print('Cleanup done.', flush=True)


if __name__ == '__main__':
    train()
