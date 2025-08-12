# evaluate.py
# -----------
# Evaluate a trained classifier on the FSDD test split.
# - Loads the saved scikit-learn pipeline (scaler + model).
# - Computes accuracy on the held-out test set.
# - Plots a confusion matrix for digits 0–9.
# Usage (from project root):
#   python -m src.evaluate --model-path models/fsdd_logreg.joblib --seed 1337 --test-size 0.2

from __future__ import annotations

# Standard library
import argparse

# Third-party
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Project code
from .dataset_fsdd import load_fsdd


def main():
    # -----------------------------
    # Parse CLI arguments
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)      # path to saved sklearn pipeline (.joblib)
    ap.add_argument("--seed", type=int, default=1337)   # random seed for deterministic split
    ap.add_argument("--test-size", type=float, default=0.2)  # fraction for test set (e.g., 0.2 = 80/20 split)
    args = ap.parse_args()

    # -----------------------------
    # Load data and model
    # -----------------------------
    # load_fsdd() returns (X_train, y_train), (X_test, y_test); we only need the test part here
    (Xtr, ytr), (Xte, yte) = load_fsdd(split_seed=args.seed, test_size=args.test_size)
    # Load the full pipeline (e.g., StandardScaler + classifier)
    clf = joblib.load(args.model_path)

    # -----------------------------
    # Run inference on the test set
    # -----------------------------
    yhat = clf.predict(Xte)              # predicted digit for each sample
    acc = accuracy_score(yte, yhat)      # overall accuracy
    print(f"Accuracy: {acc*100:.2f}%")   # human-friendly percentage

    # -----------------------------
    # Confusion matrix visualization
    # -----------------------------
    # Rows: true labels, Columns: predicted labels
    cm = confusion_matrix(yte, yhat, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(values_format='d')         # show integer counts per cell
    plt.title("FSDD Confusion Matrix")
    plt.tight_layout()
    plt.show()                           # display the figure (use --save-path in headless contexts if needed)


if __name__ == "__main__":
    main()
