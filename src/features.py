# features.py
# -----------
# Feature extraction utilities for spoken-digit classification.
# This file turns a raw audio waveform into a compact, fixed-length vector
# using MFCCs (plus first/second-order deltas) and simple statistics.
# The output is designed to be lightweight and fast for classical ML models.

from __future__ import annotations
import numpy as np
import librosa

# Keep everything at 8 kHz per FSDD spec
TARGET_SR = 8000

# Core featurization: MFCCs + deltas + simple statistics
# Returns a 1D feature vector

def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Convert a waveform into a single feature vector:
    - Resample to 8 kHz if needed
    - Normalize loudness
    - Trim leading/trailing silence
    - Compute 40 MFCCs + Δ + ΔΔ
    - Pool with simple stats (mean, std, median, p10, p90)
    """
    # If the incoming sample rate differs, resample to our target
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Energy normalize to reduce loudness variance
    # (avoid division by zero if the signal is silent)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Trim leading/trailing silence (conservative threshold)
    # Returns the trimmed audio and the intervals; we only need the audio
    y, _ = librosa.effects.trim(y, top_db=25)

    # 40 MFCCs at ~25 ms windows / 10 ms hop (given n_fft and hop_length below)
    # n_mels controls the mel filterbank size used internally by librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=80, n_mels=64)

    # First- and second-order temporal derivatives (Δ and ΔΔ)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    # Stack static + deltas → a (120, T) time–feature matrix
    feats = np.vstack([mfcc, d1, d2])  # shape: (120, T)

    # Frame-level stats → utterance-level vector (order matters but is consistent)
    stats = [
        np.mean(feats, axis=1),           # average over time
        np.std(feats, axis=1),            # variability over time
        np.median(feats, axis=1),         # robust central tendency
        np.percentile(feats, 10, axis=1), # lower-tail behavior
        np.percentile(feats, 90, axis=1), # upper-tail behavior
    ]

    # Concatenate all statistics into a single 1D vector and keep float32
    x = np.concatenate(stats, axis=0).astype(np.float32)
    return x
