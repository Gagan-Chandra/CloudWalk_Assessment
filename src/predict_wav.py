# predict_wav.py
# ---------------
# CLI utility to run a single-file prediction on a spoken-digit WAV.
# - Loads the saved scikit-learn pipeline (scaler + classifier)
# - Extracts MFCC-based features
# - Prints the predicted digit and (if supported) class probabilities

from __future__ import annotations
import argparse, sys
import joblib
import soundfile as sf
import numpy as np
from .features import extract_features

def main():
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    ap = argparse.ArgumentParser(description="Predict digit for a single WAV")
    ap.add_argument("wav_path", help="Path to a WAV file (single spoken digit)")
    ap.add_argument("--model-path", required=True, help="Path to saved sklearn pipeline")
    ap.add_argument("--topk", type=int, default=3, help="Show top-K classes if probabilities available")
    args = ap.parse_args()

    # -----------------------------
    # Load the trained pipeline
    # -----------------------------
    # Using a try/except so we fail with a readable error if the path is wrong
    try:
        clf = joblib.load(args.model_path)
    except Exception as e:
        print(f"[Error] Could not load model: {e}", file=sys.stderr)
        sys.exit(2)

    # -----------------------------
    # Read the audio file (WAV)
    # -----------------------------
    # soundfile returns (audio_array, sample_rate)
    try:
        audio, sr = sf.read(args.wav_path)
    except Exception as e:
        print(f"[Error] Could not read WAV: {e}", file=sys.stderr)
        sys.exit(3)

    # If the audio has multiple channels, average to mono
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # -----------------------------
    # Extract features
    # -----------------------------
    # extract_features handles resampling, trimming, MFCCs, deltas, and statistics
    feats = extract_features(audio, sr).reshape(1, -1)

    # -----------------------------
    # Predict the digit
    # -----------------------------
    # The pipeline includes scaling, so we can call predict directly on the full pipeline
    pred = int(clf.predict(feats)[0])

    # -----------------------------
    # (Optional) Probabilities
    # -----------------------------
    # If the final estimator supports predict_proba (e.g., LogisticRegression),
    # use the pipeline's predict_proba so the same preprocessing is applied.
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(feats)[0]          # class probabilities for this sample
        classes = clf.classes_                        # class order aligned with proba array

        # Confidence for the predicted class: find index of `pred` in classes
        ix = int(np.where(classes == pred)[0][0])
        conf = float(proba[ix])

        # Show top-K classes by probability for quick sanity checks
        topk = min(args.topk, len(classes))
        order = np.argsort(proba)[::-1][:topk]
        tops = ", ".join(f"{int(classes[i])}:{proba[i]:.2f}" for i in order)

        print(f"Prediction: {pred} (p={conf:.3f})")
        print(f"Top-{topk}: {tops}")
    else:
        # Some classifiers (e.g., LinearSVC) don't expose calibrated probabilities
        print(f"Prediction: {pred}")

if __name__ == "__main__":
    main()
