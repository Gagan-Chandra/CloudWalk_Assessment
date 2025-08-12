# quick_test.py
# --------------
# Minimal sanity-check script:
# - Loads the trained sklearn pipeline (scaler + classifier)
# - Reads one FSDD WAV file
# - Extracts MFCC-based features using our project featurizer
# - Runs a single prediction and (if supported) prints a probability

import joblib, soundfile as sf, numpy as np
from src.features import extract_features

# Paths to a known WAV and the saved model.
# Using raw strings (r"...") so backslashes work on Windows without escaping.
WAV = r"data\fsdd\recordings\0_jackson_0.wav"
MODEL = r"models\fsdd_logreg.joblib"

print("Loading model:", MODEL)
clf = joblib.load(MODEL)  # load the full sklearn pipeline (e.g., StandardScaler + LogisticRegression)

print("Reading wav:", WAV)
audio, sr = sf.read(WAV)  # audio: np.ndarray, sr: sample rate (int)
if audio.ndim > 1:
    # If the file is stereo/multi-channel, average the channels to mono.
    import numpy as np
    audio = np.mean(audio, axis=1)

print("Featurizing…")
# Convert waveform → fixed-length feature vector (includes resample, trim, MFCCs, deltas, stats)
x = extract_features(audio, sr).reshape(1, -1)
print("Feature shape:", x.shape)

# Predict the most likely digit (0–9)
pred = int(clf.predict(x)[0])
print("Prediction:", pred)

# If the underlying estimator exposes probabilities (e.g., LogisticRegression),
# print the probability for the predicted class. Some models (e.g., LinearSVC) won't have this.
if hasattr(clf[-1], "predict_proba"):
    p = float(clf.predict_proba(x)[0][pred])
    print("Probability:", p)
