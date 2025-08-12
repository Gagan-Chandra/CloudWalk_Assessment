# serve_mic_once.py
# ------------------
# Record a short chunk from the system microphone *once* and predict the spoken digit.
# This script is intentionally one-shot: it records, runs inference, prints the result, and exits.

from __future__ import annotations
import argparse
import sys
import joblib
import numpy as np
import sounddevice as sd
from .features import extract_features, TARGET_SR  # project feature extractor + target sample rate


def record_once(duration: float, sr: int) -> np.ndarray:
    """
    Capture a single mono audio buffer from the default input device.

    Parameters
    ----------
    duration : float
        How many seconds to record.
    sr : int
        Sampling rate for the recording.

    Returns
    -------
    np.ndarray
        1-D float32 waveform (mono). Shape: (duration * sr,)
    """
    print(f"Recording {duration:.2f}s at {sr} Hz…")
    # sd.rec allocates a (num_samples, channels) float32 array and starts recording
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # block until recording finishes
    return audio[:, 0]  # flatten (N, 1) → (N,)


def main():
    # -----------------------------
    # CLI arguments
    # -----------------------------
    ap = argparse.ArgumentParser(description="One-shot mic capture → digit prediction")
    ap.add_argument("--model-path", required=True, help="Path to saved sklearn pipeline")
    ap.add_argument("--duration", type=float, default=0.9, help="Seconds to record (default: 0.9)")
    ap.add_argument("--sr", type=int, default=TARGET_SR, help=f"Sample rate (default: {TARGET_SR})")
    ap.add_argument("--silence-thresh", type=float, default=0.005, help="Mean abs amplitude threshold to detect voice")
    args = ap.parse_args()

    # -----------------------------
    # Load trained pipeline
    # -----------------------------
    try:
        clf = joblib.load(args.model_path)
    except Exception as e:
        # If joblib can't load the model (bad path/corrupt file), print and exit gracefully
        print(f"Error loading model: {e}")
        return

    # -----------------------------
    # Record a single utterance
    # -----------------------------
    try:
        y = record_once(args.duration, args.sr)
    except Exception as e:
        # Common issues: no input device, permissions blocked, invalid sample rate
        print(f"Audio capture error: {e}")
        print("Tip: ensure microphone permissions are granted to your terminal/VS Code.")
        return

    # -----------------------------
    # Cheap silence gate
    # -----------------------------
    # If the mean absolute amplitude is below a small threshold, assume we captured silence.
    if float(np.mean(np.abs(y))) < args.silence_thresh:
        print("No speech detected. Try speaking closer to the mic, increasing duration, or reducing --silence-thresh.")
        return

    # -----------------------------
    # Feature extraction + prediction
    # -----------------------------
    feats = extract_features(y, args.sr).reshape(1, -1)  # 2D shape for sklearn
    pred = int(clf.predict(feats)[0])                    # top-1 predicted class (digit 0–9)

    # If the underlying estimator exposes predict_proba, print a confidence value too
    if hasattr(clf[-1], "predict_proba"):
        # Note: calling predict_proba on the pipeline's last step
        p = float(clf.predict_proba(feats)[0][pred])
        print(f"Predicted digit: {pred} (p={p:.3f})")
    else:
        print(f"Predicted digit: {pred}")


if __name__ == "__main__":
    main()
