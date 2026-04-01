/* ============================================================
   CharSense AI — Frontend Logic
   V1: 10-class digit CNN  |  V2: 62-class universal CNN
   Features: model toggle, number builder, auto-predict on mouseup
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ──────────────────────────────────────────────────────────
    const canvas        = document.getElementById('drawingCanvas');
    const ctx           = canvas.getContext('2d');
    const clearBtn      = document.getElementById('clearBtn');
    const predictBtn    = document.getElementById('predictBtn');
    const brushSlider   = document.getElementById('brush-slider');

    const btnV1         = document.getElementById('btn-v1');
    const btnV2         = document.getElementById('btn-v2');
    const versionBadge  = document.getElementById('version-badge');
    const modelBadge    = document.getElementById('model-label-badge');
    const canvasBadge   = document.getElementById('canvas-badge');
    const charGuide     = document.getElementById('char-guide');

    const step2Title    = document.getElementById('step2-title');
    const step2Desc     = document.getElementById('step2-desc');
    const step3Title    = document.getElementById('step3-title');
    const step3Desc     = document.getElementById('step3-desc');

    const steps = {
        cnn: document.getElementById('step-cnn'),
        pca: document.getElementById('step-pca'),
        lr:  document.getElementById('step-lr')
    };

    const resultContainer   = document.getElementById('result-container');
    const predictedCharEl   = document.getElementById('predicted-digit');
    const confidenceList    = document.getElementById('confidence-list');
    const resultMetaEl      = document.getElementById('result-meta');
    const ringFill          = document.getElementById('ring-fill');
    const ringText          = document.getElementById('ring-text');

    // Number Builder
    const nbSequence    = document.getElementById('nb-sequence');
    const nbAddBtn      = document.getElementById('nb-add-btn');
    const nbClearBtn    = document.getElementById('nb-clear-btn');
    const nbCopyBtn     = document.getElementById('nb-copy-btn');

    // ── App State ─────────────────────────────────────────────────────────
    let currentMode     = 'v1';         // 'v1' | 'v2'
    let isDrawing       = false;
    let hasDrawing      = false;
    let lastPrediction  = null;         // holds last successful prediction label
    let nbChars         = [];           // number builder sequence

    // ── Canvas Init ───────────────────────────────────────────────────────
    function initCtx() {
        ctx.lineWidth   = parseInt(brushSlider.value);
        ctx.lineCap     = 'round';
        ctx.lineJoin    = 'round';
        ctx.strokeStyle = '#000000';   // always black on white canvas
    }
    initCtx();

    brushSlider.addEventListener('input', () => {
        ctx.lineWidth = parseInt(brushSlider.value);
    });

    // ── Drawing events ────────────────────────────────────────────────────
    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const src  = e.touches ? e.touches[0] : e;
        return {
            x: src.clientX - rect.left,
            y: src.clientY - rect.top
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true; hasDrawing = true;
        resetPipelineUI();
        lastPrediction = null;
        const { x, y } = getPos(e);
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function moveDraw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const { x, y } = getPos(e);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function endDraw(e) {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.beginPath();
    }

    canvas.addEventListener('mousedown',  startDraw);
    canvas.addEventListener('mousemove',  moveDraw);
    canvas.addEventListener('mouseup',    endDraw);
    canvas.addEventListener('mouseleave', endDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove',  moveDraw,  { passive: false });
    canvas.addEventListener('touchend',   endDraw);

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasDrawing     = false;
        lastPrediction = null;
        resetPipelineUI();
    }
    clearBtn.addEventListener('click', clearCanvas);

    // ── Model Toggle ──────────────────────────────────────────────────────
    function switchMode(mode) {
        currentMode = mode;
        document.body.classList.toggle('mode-v2', mode === 'v2');

        btnV1.classList.toggle('active', mode === 'v1');
        btnV2.classList.toggle('active', mode === 'v2');

        if (mode === 'v1') {
            versionBadge.textContent  = 'v1 Digit';
            modelBadge.textContent    = 'CNN · v1 · 10 classes';
            canvasBadge.textContent   = '28 × 28 MNIST Scale';
            charGuide.style.display   = 'none';
            step2Title.textContent    = 'MaxPool + Dropout';
            step2Desc.textContent     = 'Spatial downsampling and dropout prevent overfitting.';
            step3Title.textContent    = 'Softmax · 10 Classes';
            step3Desc.textContent     = 'Dense head maps features to digit probabilities.';
        } else {
            versionBadge.textContent  = 'v2 Universal';
            modelBadge.textContent    = 'CNN · v2 · 62 classes';
            canvasBadge.textContent   = 'EMNIST 28 × 28 Scale';
            charGuide.style.display   = 'block';
            step2Title.textContent    = 'MaxPool × 3 + Dropout2D';
            step2Desc.textContent     = 'Deeper feature pyramid handles curved letters vs. sharp digits.';
            step3Title.textContent    = 'Softmax · 62 Classes';
            step3Desc.textContent     = 'Dense(512) → Dense(62): digits, uppercase A–Z, lowercase a–z.';
        }

        // Update CSS accent vars via body class (CSS handles the rest)
        resetPipelineUI();
    }

    btnV1.addEventListener('click', () => switchMode('v1'));
    btnV2.addEventListener('click', () => switchMode('v2'));

    // ── Pipeline Animation ─────────────────────────────────────────────────
    function animateStep(stepEl, ms) {
        stepEl.classList.remove('completed');
        stepEl.classList.add('active');
        const bar = stepEl.querySelector('.progress-bar');
        bar.style.width = '0%';

        return new Promise(resolve => {
            let start = null;
            function tick(ts) {
                if (!start) start = ts;
                const pct = Math.min(((ts - start) / ms) * 100, 100);
                bar.style.width = pct + '%';
                if (pct < 100) {
                    requestAnimationFrame(tick);
                } else {
                    stepEl.classList.remove('active');
                    stepEl.classList.add('completed');
                    resolve();
                }
            }
            requestAnimationFrame(tick);
        });
    }

    function resetPipelineUI() {
        Object.values(steps).forEach(s => {
            s.classList.remove('active', 'completed');
            const pb = s.querySelector('.progress-bar');
            if (pb) pb.style.width = '0%';
        });
        resultContainer.classList.remove('show');
        predictedCharEl.textContent = '?';
        confidenceList.innerHTML    = '';
        updateRing(0);
    }

    // ── Confidence Ring ───────────────────────────────────────────────────
    const CIRCUMFERENCE = 2 * Math.PI * 24;   // r=24 → ≈150.8

    function updateRing(pct) {
        const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
        ringFill.setAttribute('stroke-dasharray',  CIRCUMFERENCE);
        ringFill.setAttribute('stroke-dashoffset', offset);
        ringText.textContent = Math.round(pct) + '%';
    }

    // ── Render Results ────────────────────────────────────────────────────
    function renderResults(data) {
        lastPrediction = data.prediction;

        predictedCharEl.textContent = data.prediction;
        resultMetaEl.textContent    = currentMode === 'v1' ? 'Digit Prediction' : 'Character Prediction';

        // Fill confidence ring
        const topConf = data.confidences[0].conf;
        setTimeout(() => updateRing(topConf), 100);

        confidenceList.innerHTML = '';
        const top5 = data.confidences.slice(0, 5);

        top5.forEach((item, i) => {
            const isTop = i === 0;
            const div   = document.createElement('div');
            div.className = `confidence-item${isTop ? ' top-prediction' : ''}`;
            div.innerHTML = `
                <span class="conf-label">${item.label}</span>
                <div class="conf-bar-wrapper"><div class="conf-bar" style="width:0%"></div></div>
                <span class="conf-value">${item.conf.toFixed(1)}%</span>
            `;
            confidenceList.appendChild(div);
            setTimeout(() => {
                div.querySelector('.conf-bar').style.width = item.conf + '%';
            }, 60 + i * 90);
        });

        resultContainer.classList.add('show');
    }

    // ── Predict ───────────────────────────────────────────────────────────
    async function runPrediction() {
        if (!hasDrawing) {
            canvas.style.outline = '2px solid #f87171';
            setTimeout(() => canvas.style.outline = '', 1000);
            return;
        }

        predictBtn.disabled = true;
        clearBtn.disabled   = true;
        resetPipelineUI();

        const imageData = canvas.toDataURL('image/png');
        const apiCall   = fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, mode: currentMode })
        }).then(r => r.json());

        try {
            await animateStep(steps.cnn, currentMode === 'v2' ? 900 : 700);
            await animateStep(steps.pca, 550);
            await animateStep(steps.lr,  currentMode === 'v2' ? 450 : 350);

            const result = await apiCall;
            if (result.error) {
                showError(result.error);
            } else {
                renderResults(result);
            }
        } catch (err) {
            console.error(err);
            showError('Connection error — is Flask running on port 5000?');
        } finally {
            predictBtn.disabled = false;
            clearBtn.disabled   = false;
        }
    }

    predictBtn.addEventListener('click', runPrediction);

    function showError(msg) {
        resultMetaEl.textContent = '⚠ Error';
        predictedCharEl.textContent = '!';
        confidenceList.innerHTML = `<p style="color:#f87171;font-size:13px">${msg}</p>`;
        resultContainer.classList.add('show');
    }

    // ── Number Builder ────────────────────────────────────────────────────
    function renderNbSequence() {
        nbSequence.innerHTML = '';
        if (nbChars.length === 0) {
            nbSequence.innerHTML = '<span class="nb-placeholder">Draw a char, then press ⊕ Add</span>';
            return;
        }
        nbChars.forEach((ch, idx) => {
            const chip = document.createElement('span');
            chip.className   = 'nb-chip';
            chip.textContent = ch;
            chip.title       = `Click to remove "${ch}"`;
            chip.addEventListener('click', () => {
                nbChars.splice(idx, 1);
                renderNbSequence();
            });
            nbSequence.appendChild(chip);
        });
    }

    nbAddBtn.addEventListener('click', () => {
        if (lastPrediction === null) {
            // Try to auto-predict first
            if (hasDrawing) {
                runPrediction().then(() => {
                    if (lastPrediction !== null) {
                        nbChars.push(lastPrediction);
                        renderNbSequence();
                        clearCanvas();
                    }
                });
            }
            return;
        }
        nbChars.push(lastPrediction);
        renderNbSequence();
        lastPrediction = null;
        clearCanvas();
    });

    nbClearBtn.addEventListener('click', () => {
        nbChars = [];
        renderNbSequence();
    });

    nbCopyBtn.addEventListener('click', async () => {
        if (nbChars.length === 0) return;
        const text = nbChars.join('');
        try {
            await navigator.clipboard.writeText(text);
            nbCopyBtn.style.color = '#34d399';
            setTimeout(() => nbCopyBtn.style.color = '', 1500);
        } catch {
            prompt('Copy this:', text);
        }
    });

    // ── Init ──────────────────────────────────────────────────────────────
    switchMode('v1');
    renderNbSequence();
});
