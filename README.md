# Digit Classification from Audio (FSDD) — Lightweight Prototype

Small, fast baseline that listens to short WAV clips of spoken digits (0–9) and predicts the number.
The goal here is **speed, clarity, and extensibility** — not a giant model.

---

## Key ideas

- **Tiny features, big mileage**: MFCCs + Δ + ΔΔ with simple stats → a compact vector per clip.
- **Simple models**: Logistic Regression or Linear SVM in a scikit-learn `Pipeline` with `StandardScaler`.
- **Snappy I/O**: Train in seconds; infer in milliseconds. Optional one-shot microphone demo.

---

## Project layout

```
.
├── README.md
├── requirements.txt
├── data/
│   └── fsdd/                     # dataset is auto-cloned here
├── models/
│   └── fsdd_logreg.joblib        # saved pipeline (created after training)
├── reports/                      # (optional) where you can save figures
└── src/
    ├── __init__.py
    ├── features.py               # MFCC + deltas + stats featurizer
    ├── dataset_fsdd.py           # dataset clone + load + split
    ├── train.py                  # train + quick test report + save
    ├── evaluate.py               # accuracy + confusion matrix
    ├── predict_wav.py            # predict a single WAV (with Top-K)
    └── serve_mic_once.py         # one-shot microphone prediction
```

---

## Setup

**Prereqs**
- Python 3.9–3.12
- Git (for dataset auto-clone)
- (Mic demo) microphone permission for your terminal app

**Create a virtual environment and install**

```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Linux extras for mic/audio (if needed):  
> `sudo apt-get install -y portaudio19-dev libsndfile1`

---

## Dataset (FSDD)

The **Free Spoken Digit Dataset** (FSDD) will be cloned automatically on first use to `data/fsdd`:
- Source: https://github.com/Jakobovski/free-spoken-digit-dataset  
- Files end up in `data/fsdd/recordings/*.wav`

You can also clone manually:

```bash
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git data/fsdd --depth 1
```

---

## Train

```bash
python -m src.train --model logreg --seed 1337 --test-size 0.2
```

What it does:
- Loads and featurizes FSDD
- Trains a small model (`StandardScaler` + `LogisticRegression`)
- Prints accuracy report on the test split
- Saves the pipeline to `models/fsdd_logreg.joblib`

Try the Linear SVM variant:

```bash
python -m src.train --model linsvm
```

---

## Evaluate

```bash
python -m src.evaluate --model-path models/fsdd_logreg.joblib --seed 1337 --test-size 0.2
```
This prints test accuracy and shows a confusion matrix window.

> If you’re on a headless system where windows can’t open, run the command from a local machine or modify `evaluate.py` to save the figure.

---

## Predict a WAV

**Using a file from the dataset:**

```bash
# macOS / Linux
python -m src.predict_wav data/fsdd/recordings/0_jackson_0.wav --model-path models/fsdd_logreg.joblib

# Windows (PowerShell) — using forward slashes to avoid escape issues
python -m src.predict_wav "data/fsdd/recordings/0_jackson_0.wav" --model-path "models/fsdd_logreg.joblib"
```

You should see output like:

```
Prediction: 0 (p=0.99)
Top-3: 0:0.99, 8:0.01, 3:0.00
```

**Using your own recording:** provide any short WAV where you say a single digit.
(If needed) convert to WAV:

```bash
ffmpeg -i input.m4a -ar 8000 -ac 1 my_digit.wav
python -m src.predict_wav my_digit.wav --model-path models/fsdd_logreg.joblib
```

---

## One-shot microphone prediction (bonus)

Records once, predicts once, exits:

```bash
# Default 0.9s at 8 kHz
python -m src.serve_mic_once --model-path models/fsdd_logreg.joblib --duration 0.9 --sr 8000
```

If it says “No speech detected,” either:
- Move closer to the mic,
- Increase duration: `--duration 1.2`,
- Or lower the silence threshold: `--silence-thresh 0.002`.

List audio devices on Windows:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## Technical Explanation — What the Code Does and Why

### Problem framing
We want a **lightweight digit recognizer** that maps a short audio clip saying a single digit (0–9) to the correct class. Priorities are speed, simplicity, and clarity over heavy modeling.

### Data & labeling (FSDD)
- **Dataset**: Free Spoken Digit Dataset (FSDD), cloned into `data/fsdd` on first use.
- **Files**: WAV clips at ~8 kHz with filenames like `7_jackson_0.wav`.
- **Labeling**: We parse the **first token** of each filename (e.g., `7`) as the class label.

**Where in code**: `src/dataset_fsdd.py` — `ensure_fsdd_downloaded()` clones; `load_fsdd(...)` builds features and a stratified train/test split.

### Preprocessing & feature extraction
1. **Resample** to **8 kHz** (`TARGET_SR = 8000`).
2. **Normalize** by max amplitude to reduce loudness variance.
3. **Trim silence** (`librosa.effects.trim(..., top_db=25)`).
4. **Frame features**: 40 **MFCCs** + **Δ** + **ΔΔ** → `(120, T)`.
5. **Pool** over time with **mean, std, median, p10, p90** → **600-D** vector.

**Why**: Robust, compact, fast. Works well with linear models and tiny datasets.

**Where in code**: `src/features.py` — `extract_features(y, sr)` returns the fixed-length vector.

### Models & pipelines
- Pipeline: `StandardScaler()` → linear classifier.
- Options:
  - **Logistic Regression** (OvR): fast and exposes `predict_proba`.
  - **Linear SVM**: strong margin classifier, no probabilities.

**Where in code**: `src/train.py` — `MODELS` registry builds the pipelines, trains, evaluates, and saves to `models/`.

### Training & evaluation
- **Train**: `python -m src.train ...` → prints test accuracy + classification report.
- **Evaluate**: `python -m src.evaluate ...` → prints accuracy + confusion matrix.

### Prediction paths
- **File**: `src/predict_wav.py` reads a WAV, extracts features, predicts, and prints Top‑K probabilities.
- **Mic (one-shot)**: `src/serve_mic_once.py` records once, gates on silence, predicts, and exits.

### Design choices & trade-offs
- 8 kHz audio, MFCC + deltas + stats → small & fast.
- Linear models keep training and inference trivial; LR gives calibrated‑ish confidences (can improve with calibration).
- One-shot mic avoids complexity of streaming buffers and event loops for an assessment-ready demo.

### Error handling & UX
- Auto-clone dataset if missing; clear messages for mic permissions and path issues.
- Top‑K display helps sanity-check predictions.

### Performance notes
- Train: seconds. Inference: tens of ms for features, sub‑ms for predict on CPU.
- Memory: tiny (600-D features + linear weights).

### Limitations & future work
- Add noise/shift augmentation for robustness.
- Consider calibration if probabilities matter.
- Streaming version with a ring buffer for lower-latency UX.
- Try a tiny 1D‑CNN on log‑mels; export to ONNX if deploying.

### File-by-file tour
- `src/features.py` — featurizer (600-D vector).
- `src/dataset_fsdd.py` — clone + load + split.
- `src/train.py` — build pipeline, train, report, save.
- `src/evaluate.py` — accuracy + confusion matrix.
- `src/predict_wav.py` — single-file prediction with Top‑K.
- `src/serve_mic_once.py` — one-shot mic → prediction.

---

## Evaluation Criteria — Our Answers

### 1) Modeling choices
- **Features:** MFCCs + Δ + ΔΔ at 8 kHz, summarized with mean, std, median, p10, and p90 → **600-D** vector.  
  *Why appropriate:* classic, robust speech features; pooling removes timing sensitivity and keeps the model tiny.
- **Models:** `StandardScaler` → **Logistic Regression (OvR)** or **Linear SVM** in a scikit‑learn Pipeline.  
  *Why appropriate:* fast to train, millisecond inference; LR provides probabilities for UX/debug.

### 2) Model performance
- **Metric:** overall **accuracy** on an 80/20 stratified split (seed `1337`), plus a **confusion matrix** for digits 0–9.
- **Fill after your run:**  
  - Test accuracy: `XX.XX%`  
  - Confusion matrix: `reports/confusion_matrix.png`

### 3) Responsiveness
- **Featurization:** tens of ms on CPU for <1s clips. **Prediction:** sub‑ms.  
- End-to-end latency is dominated by **record duration** (e.g., 0.9s).

### 4) Code architecture
- Clear separation: `features.py`, `dataset_fsdd.py`, `train.py`, `evaluate.py`, `predict_wav.py`, `serve_mic_once.py`.  
- Pipelines ensure the same preprocessing at train/test and avoid probability mismatches.

### 5) LLM collaboration
- Used LLMs to scaffold modules/README, debug a proba mismatch (pipeline vs last estimator), and add one‑shot mic with a silence gate and clear errors.  
- `quick_test.py` validates the full path (load → read → featurize → predict).

### 6) Creative energy
- Top‑K display, silence gate, reproducible CLIs, and a roadmap for extensions (augmentation, streaming, tiny CNN, calibration, ONNX).

### How to reproduce quickly

```bash
# 1) Train (auto-clones FSDD if needed)
python -m src.train --model logreg --seed 1337 --test-size 0.2

# 2) Evaluate and save confusion matrix
python -m src.evaluate --model-path models/fsdd_logreg.joblib --seed 1337 --test-size 0.2

# 3) Predict a known WAV
python -m src.predict_wav data/fsdd/recordings/0_jackson_0.wav --model-path models/fsdd_logreg.joblib

# 4) One-shot mic prediction (speak one digit)
python -m src.serve_mic_once --model-path models/fsdd_logreg.joblib --duration 0.9 --sr 8000
```

---

## Troubleshooting

- **“File not found”**
  - Verify after clone/training:
    ```powershell
    Get-ChildItem "data/fsdd/recordings/0_jackson_0.wav"
    Get-ChildItem "models/fsdd_logreg.joblib"
    ```
- **Mic permission issues**
  - Allow microphone access for your terminal/VS Code (macOS & Windows privacy settings).
- **Matplotlib window doesn’t appear**
  - Use `--save-path` on `evaluate.py` to write a PNG in headless setups.
- **Package not found**
  - Ensure you’re using your venv’s Python: `where python` (Windows) / `which python` (macOS/Linux).

---

## Credits

- **Dataset**: Free Spoken Digit Dataset (FSDD) by Zohar Jackson and contributors  
  https://github.com/Jakobovski/free-spoken-digit-dataset
