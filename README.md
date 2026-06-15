# CharSense AI — Universal Handwriting Recognizer

A premium deep-learning web app that recognizes **handwritten digits (0–9)**, **uppercase letters (A–Z)**, and **lowercase letters (a–z)** — plus an infinite **Number Builder** mode to assemble full multi-digit numbers character by character.

🔗 **Live Demo**: [CharSense on Render](https://charsense-ai.onrender.com)

---

## ✨ Features

| Feature | Details |
|---|---|
| **v1 — Digit Mode** | MNIST-trained 10-class CNN · ~99.9% accuracy |
| **v2 — Universal Mode** | EMNIST ByClass 62-class ResNet+SE+CBAM · digits + A–Z + a–z |
| **RL Training** | 3-phase training: Mixup → RL Hard Sample Mining → Fine-tuning |
| **Number Builder** | Draw one char at a time, assemble infinite strings |
| **Confidence Ring** | Animated circular confidence meter |
| **Brush Slider** | Adjustable stroke width for drawing |
| **Character Grid** | Visual display of all 62 supported characters |
| **Dark Premium UI** | Glassmorphism · glow effects · micro-animations |

---

## 🧠 Model Architecture

### V1 — Digit CNN (MNIST, 10 classes)
```
Input (28×28) → Conv(32)→BN→ReLU→Pool → Conv(64)→BN→ReLU→Pool
             → Conv(128)→BN→ReLU→Pool → Dense(256) → Dense(10, softmax)
Accuracy: ~99.9%
```

### V2 — CharSenseNet-V4 (EMNIST ByClass, 62 classes)
```
Input (28×28) → Stem: Conv(1→32) + BN + ReLU
             → Stage 1: 4× ResBlock(32→64) + SE Attention + Dropout(0.02)  [14×14]
             → Stage 2: 4× ResBlock(64→128) + SE Attention + Dropout(0.05) [7×7]
             → Stage 3: 4× ResBlock(128→256) + CBAM Attention + Dropout(0.10) [3×3]
             → GAP → Dense(256→128) + BN + ReLU + Dropout(0.3) → Dense(128→62, softmax)

Parameters: ~5.9M
Classes: 0-9 (digits) | A-Z (10-35) | a-z (36-61)
Training: 3-phase RL-enhanced training on EMNIST ByClass (~248K+ samples)
```

### Training Pipeline

| Phase | Epochs | Technique |
|---|---|---|
| **Phase 1: Mixup** | E1-25 | Mixup augmentation + random shifts + noise + erasing |
| **Phase 2: RL Mining** | E26-40 | Policy-gradient hard sample mining + confidence regularization |
| **Phase 3: Fine-tune** | E41-50 | Standard CE with label smoothing, low learning rate |

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Train V2 universal model
python train_v4_rl.py

# Start the app
python app.py
# → http://localhost:5000
```

---

## 📁 Project Structure

```
├── app.py                 # Flask backend — loads both models, /predict route
├── train_v4_rl.py         # Train V4 CharSenseNet with RL-enhanced training
├── train_universal.py     # Legacy V2 training script
├── train_cnn.py           # Train 10-class MNIST digit CNN
├── models/
│   ├── digit_cnn.pth          # V1 pre-trained weights (committed)
│   ├── universal_cnn_best.pth # V2 pre-trained weights (committed after training)
│   └── norm_stats.pt          # EMNIST normalization stats
├── templates/index.html   # Full SPA UI
├── static/
│   ├── styles.css         # Dark premium CSS
│   └── script.js          # Canvas, toggle, number builder, 62-char grid
├── build.sh               # Render build script (trains during deployment)
├── Procfile               # Render/Heroku deployment
└── render.yaml            # Render one-click deploy config
```

---

## 🌐 Deploy to Render

1. Fork this repo
2. Connect to [render.com](https://render.com)
3. New → Web Service → connect `kartheekbvs/deep`
4. Build: `bash build.sh`
5. Start: `gunicorn app:app --workers=1 --threads=4 --timeout=120`
6. Set env var `RENDER=1` for full training during build

Or use the included `render.yaml` for automatic config.

> **Note**: On first deploy, the build script will train the V4 model. This takes ~2-5 hours on CPU. Set `SKIP_TRAINING=1` to skip if you've pre-trained the model.

---

## 📊 Training Details

| Model | Dataset | Classes | Training | Accuracy |
|---|---|---|---|---|
| digit_cnn.pth | MNIST | 10 | 5 epochs, augmented | ~99.9% |
| universal_cnn_best.pth | EMNIST ByClass | 62 | 50 epochs, RL-enhanced | ~90%+ |

Built with PyTorch · Flask · Vanilla JS/CSS
