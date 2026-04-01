import os
import io
import base64
import numpy as np
from PIL import Image
import PIL.ImageOps
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    import torch
    import torch.nn as nn
    import joblib
    import threading
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    print("WARNING: ML dependencies (Torch/Joblib) not installed.")

app = Flask(__name__)
CORS(app)

# Thread safety for lazy loading
model_lock   = threading.Lock()
models_loaded = False

MODEL_DIR = 'models'

# ── Loaded model holders ──────────────────────────────────────────────────────
pytorch_model   = None   # V1 – 10-class digit CNN
universal_model = None   # V2 – 62-class universal CNN
pca_transformer = None
lr_model        = None

# ── Class label map for V2 (62 classes) ──────────────────────────────────────
# EMNIST ByClass: 0-9 → digits, 10-35 → A-Z, 36-61 → a-z
_LABEL_MAP = (
    [str(i) for i in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z') + 1)] +
    [chr(c) for c in range(ord('a'), ord('z') + 1)]
)   # length = 62

# ── V1: 10-class CNN ──────────────────────────────────────────────────────────
class DigitCNN(nn.Module if HAS_ML_DEPS else object):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ── V2: 62-class CNN (must match train_universal.py exactly) ──────────────────
class UniversalCNN(nn.Module if HAS_ML_DEPS else object):
    def __init__(self):
        super(UniversalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ── Model Loader ──────────────────────────────────────────────────────────────
def load_all_models():
    global pca_transformer, lr_model, pytorch_model, universal_model, models_loaded
    if not HAS_ML_DEPS or models_loaded:
        return

    with model_lock:
        if models_loaded:
            return
        try:
            print(f"Loading models from {os.path.abspath(MODEL_DIR)}...")

            # V1 — Digit CNN (10-class)
            pth_v1 = os.path.join(MODEL_DIR, 'digit_cnn.pth')
            if os.path.exists(pth_v1):
                try:
                    pytorch_model = DigitCNN()
                    pytorch_model.load_state_dict(
                        torch.load(pth_v1, map_location='cpu'))
                    pytorch_model.eval()
                    print("✓ V1 Digit CNN loaded (10 classes)")
                except Exception as e:
                    print(f"✗ V1 load failed: {e}")

            # V2 — Universal CNN (62-class)
            pth_v2 = os.path.join(MODEL_DIR, 'universal_cnn_best.pth')
            if not os.path.exists(pth_v2):
                pth_v2 = os.path.join(MODEL_DIR, 'universal_cnn.pth')
            if os.path.exists(pth_v2):
                try:
                    universal_model = UniversalCNN()
                    universal_model.load_state_dict(
                        torch.load(pth_v2, map_location='cpu'))
                    universal_model.eval()
                    print("✓ V2 Universal CNN loaded (62 classes)")
                except Exception as e:
                    print(f"✗ V2 load failed: {e}")

            # Classical fallback (PCA + LR)
            pca_path = os.path.join(MODEL_DIR, 'pca_transformer.pkl')
            lr_path  = os.path.join(MODEL_DIR, 'lr_model.pkl')
            if os.path.exists(pca_path):
                pca_transformer = joblib.load(pca_path)
                print("✓ PCA transformer loaded")
            if os.path.exists(lr_path):
                lr_model = joblib.load(lr_path)
                print("✓ LR model loaded")

        except Exception as e:
            print(f"Exception in load_all_models: {e}")
            import traceback; traceback.print_exc()
        finally:
            models_loaded = True


# ── Image Preprocessor ────────────────────────────────────────────────────────
def preprocess_image(base64_string, mode='v1'):
    """Returns (1, 1, 28, 28) float32 tensor for PyTorch, normalised."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data)).convert('RGBA')

    # Composite on white then greyscale
    bg = Image.new('RGBA', img.size, (255, 255, 255))
    gray = Image.alpha_composite(bg, img).convert('L')

    # Invert: white pen on black bg → black digit on white bg for the model
    gray = PIL.ImageOps.invert(gray)

    img28 = gray.resize((28, 28), resample=Image.Resampling.LANCZOS)
    arr   = np.array(img28).astype('float32') / 255.0  # (28, 28)
    return arr.reshape(1, 1, 28, 28)   # (B, C, H, W)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':       'healthy',
        'models_loaded': models_loaded,
        'v1_ready':     pytorch_model   is not None,
        'v2_ready':     universal_model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        mode = data.get('mode', 'v1')   # 'v1' or 'v2'

        if not models_loaded:
            load_all_models()

        arr = preprocess_image(data['image'], mode)

        # ── V2 Universal (62-class) ────────────────────────────────────────
        if mode == 'v2':
            if universal_model is None:
                return jsonify({'error': 'Universal model (V2) not loaded yet. '
                                         'Run python train_universal.py first.'}), 503
            # EMNIST ByClass normalisation
            x = (arr - 0.1736) / 0.3317
            t = torch.from_numpy(x).float()
            with torch.no_grad():
                probs = torch.softmax(universal_model(t), dim=1).numpy()[0]

            top_idx = int(np.argmax(probs))
            top_label = _LABEL_MAP[top_idx]

            confidences = [
                {'label': _LABEL_MAP[i], 'class_idx': i,
                 'conf': round(float(probs[i]) * 100, 2)}
                for i in range(62)
            ]
            confidences.sort(key=lambda x: x['conf'], reverse=True)

            return jsonify({
                'success':    True,
                'mode':       'v2',
                'prediction': top_label,
                'class_idx':  top_idx,
                'confidences': confidences[:10],
            })

        # ── V1 Digit (10-class) ───────────────────────────────────────────
        else:
            if pytorch_model:
                x = (arr - 0.1307) / 0.3081
                t = torch.from_numpy(x).float()
                with torch.no_grad():
                    probs = torch.softmax(pytorch_model(t), dim=1).numpy()[0]
                top_idx = int(np.argmax(probs))
                method  = 'CNN'

            elif pca_transformer and lr_model:
                flat = arr.reshape(1, 784)
                pca_f = pca_transformer.transform(flat)
                probs = lr_model.predict_proba(pca_f)[0]
                top_idx = int(np.argmax(probs))
                method  = 'Classical'
            else:
                return jsonify({'error': 'No V1 model loaded. '
                                         'Run python train_cnn.py first.'}), 503

            confidences = [
                {'label': str(i), 'class_idx': i,
                 'conf': round(float(probs[i]) * 100, 2)}
                for i in range(10)
            ]
            confidences.sort(key=lambda x: x['conf'], reverse=True)

            return jsonify({
                'success':    True,
                'mode':       'v1',
                'prediction': str(top_idx),
                'class_idx':  top_idx,
                'confidences': confidences,
                'method':     method,
            })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
