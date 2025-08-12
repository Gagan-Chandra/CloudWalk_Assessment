# dataset_fsdd.py
# ----------------
# Minimal loader for the Free Spoken Digit Dataset (FSDD).
# - Ensures the dataset is present by shallow-cloning the official repo.
# - Reads WAV files, converts to mono if needed, and extracts features.
# - Returns train/test splits ready for modeling.

from __future__ import annotations

# Standard library
import subprocess
from pathlib import Path

# Third-party libs
import soundfile as sf
import numpy as np

# Project code: MFCC-based featurizer (TARGET_SR import is informational here)
from .features import extract_features, TARGET_SR

# --------- Paths & constants ---------
# Root directory where FSDD will live
DATA_DIR = Path("data/fsdd")
# Folder inside the repo that actually contains the WAV files
RECORDINGS = DATA_DIR / "recordings"
# Upstream FSDD repository (public)
GIT_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"

# Valid digit labels as strings, e.g., "0", "1", ..., "9"
LABELS = [str(i) for i in range(10)]


def ensure_fsdd_downloaded() -> None:
    """
    Ensure the FSDD dataset is available locally at data/fsdd/recordings.
    If it's missing, perform a shallow clone to save time and space.
    """
    # If recordings exist, nothing to do
    if not RECORDINGS.exists():
        # Make sure the parent "data" directory exists
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        print("Cloning FSDD dataset…")
        # Shallow clone the repo into data/fsdd
        subprocess.run(
            ["git", "clone", "--depth", "1", GIT_URL, str(DATA_DIR)],
            check=True
        )
        # Basic sanity check: recordings folder should now exist
        assert RECORDINGS.exists(), "FSDD recordings folder not found after clone"


def load_fsdd(split_seed: int = 1337, test_size: float = 0.2):
    """
    Load features and labels from FSDD and return a stratified train/test split.

    Parameters
    ----------
    split_seed : random seed for reproducible splits
    test_size  : fraction of the data to place in the test set
    """
    # Imported lazily so sklearn isn't required until this function is used
    from sklearn.model_selection import train_test_split

    # Make sure the dataset is present
    ensure_fsdd_downloaded()

    # Collect all .wav files (sorted for stability)
    wavs = sorted(RECORDINGS.glob("*.wav"))
    assert wavs, f"No WAV files found in {RECORDINGS}"

    # Containers for features (X) and labels (y)
    X, y = [], []

    # Iterate over each WAV file to extract its label and features
    for p in wavs:
        # Filenames look like: 7_jackson_0.wav → label is the first token before "_"
        label = p.name.split("_")[0]
        # Skip files with unexpected naming
        if label not in LABELS:
            continue

        # Read audio and sample rate
        audio, sr = sf.read(p)
        # If stereo/multi-channel, average to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Extract a fixed-length feature vector (handles resampling internally)
        feats = extract_features(audio, sr)

        # Accumulate sample
        X.append(feats)
        y.append(int(label))

    # Stack into arrays for sklearn
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    # Stratified split preserves class balance across train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=y
    )
    # Return as ((X_train, y_train), (X_test, y_test))
    return (X_train, y_train), (X_test, y_test)
