#!/usr/bin/env python3
"""
CharSenseNet-V4 Training — Optimized for CPU with RL-Enhanced Training

Uses the proven V3 ResNet+SE+CBAM architecture scaled up for more power,
with RL hard sample mining and confidence regularization.
Trains on balanced EMNIST ByClass subset (3000/class = 186K samples).
"""

import os, random, time, gc, sys
sys.stdout.reconfigure(line_buffering=True)
random.seed(42)

import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
np.random.seed(42)
from torchvision.datasets import EMNIST
from collections import defaultdict

NUM_CLASSES = 62
EMNIST_MEAN = 0.1736
EMNIST_STD  = 0.3317

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — CharSenseNetV4 (matches app.py)
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
        return x * sp_att

class BottleneckResBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, attention='se', drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
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
        out = self.drop(out)
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        out = self.attn(out)
        out += identity
        return F.relu(out, inplace=True)

class CharSenseNetV4(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(64, 64, 256, 4, 2, 'se', 0.02)
        self.stage2 = self._make_stage(256, 128, 512, 4, 2, 'se', 0.04)
        self.stage3 = self._make_stage(512, 256, 1024, 4, 2, 'cbam', 0.06)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
    def _make_stage(self, in_ch, mid_ch, out_ch, num_blocks, stride, attention, drop_rate):
        layers = [BottleneckResBlock(in_ch, mid_ch, out_ch, stride=stride, attention=attention, drop_rate=drop_rate)]
        for _ in range(1, num_blocks):
            layers.append(BottleneckResBlock(out_ch, mid_ch, out_ch, stride=1, attention=attention, drop_rate=drop_rate))
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
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 4)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'CharSenseNet-V4 RL Training - {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'Device: {device} | Threads: {torch.get_num_threads()}', flush=True)
    print('=' * 70, flush=True)

    model_path = 'models/universal_cnn_best.pth'
    if os.path.exists(model_path):
        print(f'Model already exists: {model_path}', flush=True)
        return

    os.makedirs('models', exist_ok=True)

    # ── Load EMNIST data ──
    data_dir = 'emnist_data'
    TARGET_TRAIN = 3000
    TARGET_VAL = 500

    print('Loading EMNIST ByClass...', flush=True)
    ds_train = EMNIST(root=data_dir, split='byclass', train=True, download=True)
    ds_test = EMNIST(root=data_dir, split='byclass', train=False, download=True)
    print(f'  Train: {len(ds_train)}, Test: {len(ds_test)}', flush=True)

    print(f'Creating balanced subset ({TARGET_TRAIN}/class)...', flush=True)
    class_buffers = defaultdict(list)
    for i in range(len(ds_train)):
        img, label = ds_train[i]
        if label >= 62 or len(class_buffers[label]) >= TARGET_TRAIN + TARGET_VAL:
            continue
        arr = np.array(img).astype('float32') / 255.0
        arr = np.transpose(arr, (1, 0))[:, ::-1]
        class_buffers[label].append(arr)
    del ds_train
    gc.collect()

    X_tr, y_tr, X_va, y_va = [], [], [], []
    for cls in range(62):
        bufs = class_buffers[cls]
        random.shuffle(bufs)
        X_tr.extend(bufs[:TARGET_TRAIN])
        y_tr.extend([cls] * min(len(bufs), TARGET_TRAIN))
        val_bufs = bufs[TARGET_TRAIN:TARGET_TRAIN + TARGET_VAL]
        X_va.extend(val_bufs)
        y_va.extend([cls] * len(val_bufs))
    del class_buffers
    gc.collect()

    random.shuffle(list(zip(X_tr, y_tr)))
    X_train = (np.stack(X_tr).astype('float32') - EMNIST_MEAN) / EMNIST_STD
    y_train = np.array(y_tr, dtype=np.int64)
    del X_tr, y_tr
    gc.collect()

    X_val = (np.stack(X_va).astype('float32') - EMNIST_MEAN) / EMNIST_STD
    y_val = np.array(y_va, dtype=np.int64)
    del X_va, y_va
    gc.collect()

    X_te, y_te = [], []
    for i in range(len(ds_test)):
        img, label = ds_test[i]
        if label >= 62:
            continue
        arr = np.array(img).astype('float32') / 255.0
        arr = np.transpose(arr, (1, 0))[:, ::-1]
        X_te.append(arr)
        y_te.append(label)
    X_test = (np.stack(X_te).astype('float32') - EMNIST_MEAN) / EMNIST_STD
    y_test = np.array(y_te, dtype=np.int64)
    del X_te, y_te, ds_test
    gc.collect()

    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)

    print(f'  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}', flush=True)

    # Create dataloaders
    train_X = torch.from_numpy(X_train).unsqueeze(1)
    train_y = torch.from_numpy(y_train)
    val_X = torch.from_numpy(X_val).unsqueeze(1)
    val_y = torch.from_numpy(y_val)
    test_X = torch.from_numpy(X_test).unsqueeze(1)
    test_y = torch.from_numpy(y_test)
    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=256, shuffle=False, num_workers=0)
    del train_X, train_y, val_X, val_y, test_X, test_y
    gc.collect()

    # ── Model ──
    model = CharSenseNetV4().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'CharSenseNet-V4: {num_params:,} parameters', flush=True)

    # ── RL Components ──
    class RLSampleMiner:
        def __init__(self, n, dev='cpu'):
            self.difficulty = torch.zeros(n, device=dev)
            self.weight_logits = torch.zeros(n, device=dev)
            self.update_count = torch.zeros(n, device=dev)
        def update(self, idx, losses):
            with torch.no_grad():
                for i, l in zip(idx, losses):
                    ii = i.item()
                    if self.update_count[ii] == 0:
                        self.difficulty[ii] = l.item()
                    else:
                        self.difficulty[ii] = 0.99 * self.difficulty[ii] + 0.01 * l.item()
                    self.update_count[ii] += 1
                    self.weight_logits[ii] += 0.1 * (self.difficulty[ii] - losses.mean().item())
        def get_weights(self, idx):
            with torch.no_grad():
                w = F.softmax(self.weight_logits[idx], dim=0)
                w = 0.5 * w + 0.5 / len(idx)
                return w / w.sum()

    rl_miner = RLSampleMiner(len(train_loader.dataset), device)

    # ── Training config ──
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
    crit_mean = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    wait = 0
    patience = 12
    total_epochs = 40
    ckpt_path = 'models/v4_ckpt.pt'
    start_epoch = 0

    if os.path.exists(ckpt_path):
        print('Resuming...', flush=True)
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c['model'])
        opt.load_state_dict(c['opt'])
        start_epoch = c['epoch']
        best_acc = c['best_acc']
        wait = c.get('wait', 0)
        print(f'  E{start_epoch}, best={best_acc:.2f}%', flush=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-3, steps_per_epoch=len(train_loader),
        epochs=total_epochs, pct_start=0.1, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1000.0
    )
    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    print(f'\nPhase 1 (E1-20): Mixup augmentation', flush=True)
    print(f'Phase 2 (E21-35): RL hard sample mining', flush=True)
    print(f'Phase 3 (E36-40): Fine-tuning', flush=True)
    print(f'{len(train_loader)} steps/epoch, batch_size=128', flush=True)
    print('=' * 70, flush=True)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        rl = cn = tn = 0
        phase = 'mixup' if epoch < 20 else ('rl' if epoch < 35 else 'finetune')

        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Augmentation
            if random.random() < 0.5:
                dy, dx = random.randint(-2, 2), random.randint(-2, 2)
                imgs = torch.roll(imgs, shifts=(dy, dx), dims=(2, 3))

            opt.zero_grad()

            if phase == 'mixup' and random.random() < 0.5:
                lam = np.random.beta(0.4, 0.4)
                idx = torch.randperm(imgs.size(0), device=device)
                out = model(lam * imgs + (1 - lam) * imgs[idx])
                loss = (lam * crit(out, lbls) + (1 - lam) * crit(out, lbls[idx])).mean()
            elif phase == 'rl':
                out = model(imgs)
                per_sample_ce = crit(out, lbls)
                # RL: update miner
                batch_idx = torch.arange(i * 128, min((i+1) * 128, len(train_loader.dataset)),
                                          device=device)[:len(lbls)]
                rl_miner.update(batch_idx, per_sample_ce.detach())
                weights = rl_miner.get_weights(batch_idx)
                weighted_ce = (per_sample_ce * weights).sum()
                # Confidence regularization
                probs = F.softmax(out, dim=1)
                entropy = -(probs * F.log_softmax(out, dim=1)).sum(dim=1)
                correct = (out.argmax(1) == lbls).float()
                conf_reg = (-0.05 * entropy * correct - 0.3 * probs.max(dim=1)[0] * (1 - correct)).mean()
                loss = weighted_ce + conf_reg
            else:
                out = model(imgs)
                loss = crit_mean(out, lbls)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            rl += loss.item()
            _, pred = torch.max(out, 1)
            tn += lbls.size(0)
            cn += (pred == lbls).sum().item()

        train_acc = 100 * cn / tn
        elapsed = time.time() - t0

        # Validation
        model.eval()
        vc = vt = 0
        cc = [0] * NUM_CLASSES
        ct = [0] * NUM_CLASSES
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, pred = torch.max(model(imgs), 1)
                vt += lbls.size(0)
                vc += (pred == lbls).sum().item()
                for j in range(lbls.size(0)):
                    l = lbls[j].item()
                    if l < NUM_CLASSES:
                        ct[l] += 1
                        if pred[j] == l: cc[l] += 1

        acc = 100 * vc / vt
        da = 100 * sum(cc[:10]) / max(sum(ct[:10]), 1)
        ua = 100 * sum(cc[10:36]) / max(sum(ct[10:36]), 1)
        la = 100 * sum(cc[36:]) / max(sum(ct[36:]), 1)

        print(f'>>> [{phase.upper()}] E{epoch+1}/{total_epochs} ({elapsed:.0f}s) '
              f'T:{train_acc:.1f}% V:{acc:.2f}% D:{da:.1f}% U:{ua:.1f}% L:{la:.1f}%', flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            torch.save({'mean': EMNIST_MEAN, 'std': EMNIST_STD}, 'models/norm_stats.pt')
            print(f'  ** New best: {acc:.2f}% **', flush=True)
            wait = 0
        else:
            wait += 1
            print(f'  No improvement {wait}/{patience} (best: {best_acc:.2f}%)', flush=True)

        torch.save({'epoch': epoch+1, 'model': model.state_dict(), 'opt': opt.state_dict(),
                     'best_acc': best_acc, 'wait': wait}, ckpt_path)

        if wait >= patience:
            print(f'Early stopping. Best: {best_acc:.2f}%', flush=True)
            break

    # Final test
    print(f'\n{"="*70}', flush=True)
    print(f'Training complete! Best validation: {best_acc:.2f}%', flush=True)
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
                    if pred[j] == l: cc[l] += 1

    print(f'Final Test: {100*vc/vt:.2f}%', flush=True)
    for gn, s, e in [('Digits', 0, 10), ('Uppercase', 10, 36), ('Lowercase', 36, 62)]:
        print(f'  {gn}: {100*sum(cc[s:e])/max(sum(ct[s:e]),1):.1f}%', flush=True)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print('Done!', flush=True)


if __name__ == '__main__':
    train()
