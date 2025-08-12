# train.py
# --------
# Train a lightweight spoken-digit classifier on the FSDD features.
# - Uses simple scikit-learn pipelines (StandardScaler + linear model)
# - Supports Logistic Regression ("logreg") and Linear SVM ("linsvm")
# - Prints a quick test-set report and saves the trained pipeline to disk

from __future__ import annotations
import argparse
from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from .dataset_fsdd import load_fsdd

# Registry of available models. Each entry returns a ready-to-fit Pipeline.
# We keep models linear and small for speed and low latency.
MODELS = {
    # Logistic Regression with one-vs-rest (OvR). Works well with standardized MFCC stats.
    "logreg": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, n_jobs=1, multi_class="ovr", C=2.0)
    ),
    # Linear SVM (hinge loss). Fast and strong baseline; doesn't provide probabilities.
    "linsvm": lambda: make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0)
    ),
}


def main():
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), default="logreg")  # which model to train
    ap.add_argument("--seed", type=int, default=1337)                    # random seed for split
    ap.add_argument("--test-size", type=float, default=0.2)              # test fraction (e.g., 0.2 = 80/20)
    ap.add_argument("--out", type=str, default="models/fsdd_{}.joblib")  # where to save the trained pipeline
    args = ap.parse_args()

    # -----------------------------
    # Load features and labels
    # -----------------------------
    # Returns ((X_train, y_train), (X_test, y_test))
    (Xtr, ytr), (Xte, yte) = load_fsdd(split_seed=args.seed, test_size=args.test_size)

    # -----------------------------
    # Build and fit the classifier
    # -----------------------------
    clf = MODELS[args.model]()  # instantiate the pipeline
    clf.fit(Xtr, ytr)           # train on the training split

    # -----------------------------
    # Quick evaluation on the test set
    # -----------------------------
    yhat = clf.predict(Xte)
    acc = accuracy_score(yte, yhat)
    print(f"Test accuracy: {acc*100:.2f}%")
    print(classification_report(yte, yhat, digits=3))

    # -----------------------------
    # Save the trained pipeline
    # -----------------------------
    out_path = Path(args.out.format(args.model))
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure models/ exists
    joblib.dump(clf, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
