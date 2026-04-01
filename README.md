# CharSense AI — Universal Handwriting Recognizer

A premium deep-learning web app that recognizes **handwritten digits (0–9)**, **uppercase letters (A–Z)**, and **lowercase letters (a–z)** — plus an infinite **Number Builder** mode to assemble full multi-digit numbers character by character.

🔗 **Live Demo**: [CharSense on Render](https://charsense-ai.onrender.com)

---

## ✨ Features

| Feature | Details |
|---|---|
| **v1 — Digit Mode** | MNIST-trained 10-class CNN · ~99.9% accuracy |
| **v2 — Universal Mode** | EMNIST ByClass 62-class CNN · digits + A–Z + a–z |
| **Number Builder** | Draw one char at a time, assemble infinite strings |
| **Confidence Ring** | Animated circular confidence meter |
| **Brush Slider** | Adjustable stroke width for drawing |
| **Dark Premium UI** | Glassmorphism · glow effects · micro-animations |

---

## 🧠 Model Architecture

### V1 — Digit CNN (MNIST, 10 classes)
```
Input (28×28) → Conv(32)→BN→ReLU→Pool → Conv(64)→BN→ReLU→Pool
             → Conv(128)→BN→ReLU→Pool → Dense(256) → Dense(10, softmax)
Accuracy: ~99.9%
```

### V2 — Universal CNN (EMNIST ByClass, 62 classes)
```
Input (28×28) → [Conv(32)×2→BN→ReLU→Pool→Drop] × 2
             → Conv(128)→BN→ReLU→Pool
             → Dense(512) → Dense(62, softmax)
Classes: 0-9 (digits) | A-Z (10-35) | a-z (36-61)
Dataset: EMNIST ByClass — 814,255 training images
```

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Train V2 universal model
python train_universal.py

# Start the app
python app.py
# → http://localhost:5000
```

---

## 📁 Project Structure

```
├── app.py                 # Flask backend — loads both models, /predict route
├── train_universal.py     # Train 62-class EMNIST CNN (saves models/universal_cnn_best.pth)
├── train_cnn.py           # Train 10-class MNIST digit CNN
├── models/
│   ├── digit_cnn.pth          # V1 pre-trained weights (committed)
│   └── universal_cnn_best.pth # V2 pre-trained weights (committed after training)
├── templates/index.html   # Full SPA UI
├── static/
│   ├── styles.css         # Dark premium CSS
│   └── script.js          # Canvas, toggle, number builder logic
├── Procfile               # Render/Heroku deployment
└── render.yaml            # Render one-click deploy config
```

---

## 🌐 Deploy to Render

1. Fork this repo
2. Connect to [render.com](https://render.com)
3. New → Web Service → connect `kartheekbvs/deep`
4. Build: `pip install -r requirements.txt`
5. Start: `gunicorn app:app --workers=1 --threads=4 --timeout=120`

Or use the included `render.yaml` for automatic config.

---

## 📊 Training Details

| Model | Dataset | Classes | Training | Accuracy |
|---|---|---|---|---|
| digit_cnn.pth | MNIST | 10 | 5 epochs, augmented | ~99.7% |
| universal_cnn_best.pth | EMNIST ByClass | 62 | 15 epochs, augmented | ~92–95% |

Built with PyTorch · Flask · Vanilla JS/CSS
