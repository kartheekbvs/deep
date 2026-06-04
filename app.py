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
    import torch.nn.functional as F
    import threading
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    print("WARNING: ML dependencies (Torch) not installed.")

app = Flask(__name__)
CORS(app)

# Thread safety for lazy loading
model_lock   = threading.Lock()
models_loaded = False

MODEL_DIR = 'models'

# ── Loaded model holders ──────────────────────────────────────────────────────
pytorch_model   = None   # V1 – 10-class digit CNN
universal_model = None   # V2 – 62-class universal CNN

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


# ── V2: Squeeze-and-Excitation Block ─────────────────────────────────────────
class SEBlock(nn.Module if HAS_ML_DEPS else object):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ── V2: Residual Block with SE Attention ─────────────────────────────────────
class ResBlock(nn.Module if HAS_ML_DEPS else object):
    """Residual block: two 3x3 convs + skip connection + SE attention."""
    def __init__(self, in_ch, out_ch, stride=1, se_reduction=16, drop_rate=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch, reduction=se_reduction)
        self.drop  = nn.Dropout2d(drop_rate)

        # Shortcut (projection if channels or spatial size change)
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


# ── V2: 62-class UniversalCNN with ResNet + SE Attention (~3.2M params) ─────
class UniversalCNN(nn.Module if HAS_ML_DEPS else object):
    """Powerful ResNet-style CNN with Squeeze-and-Excitation attention.

    Architecture:
      - Stem: 1 -> 32 channels
      - Stage 1: 32  -> 64   (2 ResBlocks, 14x14, stride-2 downsample)
      - Stage 2: 64  -> 128  (2 ResBlocks, 7x7, stride-2 downsample)
      - Stage 3: 128 -> 256  (2 ResBlocks, 3x3, stride-2 downsample)
      - GAP -> FC(256, 128) -> FC(128, 62)
    """
    def __init__(self):
        super(UniversalCNN, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 3 stages of residual blocks with SE attention
        self.stage1 = self._make_stage(32, 64,  num_blocks=2, stride=2, drop_rate=0.05)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2, drop_rate=0.10)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2, drop_rate=0.15)

        # Classifier
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc1    = nn.Linear(256, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2    = nn.Linear(128, 62)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride, drop_rate):
        layers = []
        layers.append(ResBlock(in_ch, out_ch, stride=stride, se_reduction=16, drop_rate=drop_rate))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1, se_reduction=16, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)          # (B, 32, 28, 28)
        x = self.stage1(x)        # (B, 64, 14, 14)
        x = self.stage2(x)        # (B, 128, 7, 7)
        x = self.stage3(x)        # (B, 256, 3x3)
        x = self.gap(x).flatten(1)  # (B, 256)
        x = F.dropout(F.relu(self.fc1_bn(self.fc1(x)), inplace=True), 0.4, self.training)
        return self.fc2(x)


# ── Model Loader ──────────────────────────────────────────────────────────────
def load_all_models():
    global pytorch_model, universal_model, models_loaded
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
                    print("  V1 Digit CNN loaded (10 classes)")
                except Exception as e:
                    print(f"  V1 load failed: {e}")

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
                    print("  V2 Universal CNN loaded (62 classes, ResNet+SE)")
                except Exception as e:
                    print(f"  V2 load failed: {e}")

        except Exception as e:
            print(f"Exception in load_all_models: {e}")
            import traceback; traceback.print_exc()
        finally:
            models_loaded = True


# ── Image Preprocessor ────────────────────────────────────────────────────────
def preprocess_image(base64_string, mode='v1'):
    """Returns (1, 1, 28, 28) float32 tensor for PyTorch, normalised.

    V2 model was trained with inverted+corrected EMNIST images that match
    the output of this preprocessing pipeline exactly.
    """
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data)).convert('RGBA')

    # Composite on white then greyscale
    bg = Image.new('RGBA', img.size, (255, 255, 255))
    gray = Image.alpha_composite(bg, img).convert('L')

    # Invert: black pen on white bg -> white pen on black bg for the model
    # This matches the EMNIST-inverted training data format
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
            # EMNIST ByClass normalisation (model trained with corrected orientation)
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
