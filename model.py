"""
Future Scalability:
  - Replace generate_realtime_sample() with OpenCV frame reading
  - Upgrade model: sklearn -> TensorFlow/ONNX without changing interface
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
import joblib
import os

from data_generator import generate_offside_sample, generate_realtime_sample

FEATURES = ["passer_x", "passer_y", "teammate_x", "teammate_y", "defender_x", "defender_y", "x_diff"]
LABEL = "offside"
MODEL_PATH = "saot_model.pkl"
RANDOM_STATE = 42

class OffsideDetector:
    """
    Wrapper over a sklearn Pipeline.
    Stable interface - you can later replace the internal pipeline
    with a TensorFlow/PyTorch model without changing main.py code.
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.pipeline = self._build_pipeline(model_type)
        self.is_trained = False

    def _build_pipeline(self, model_type: str) -> Pipeline:
        if model_type == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        elif model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.is_trained, "Model not trained!"
        return self.pipeline.predict(X)

    def predict_probe(self, X: pd.DataFrame) -> np.ndarray:
        assert self.is_trained, "Model not trained!"
        return self.pipeline.predict_proba(X)

    def save(self, path: str = MODEL_PATH):
        joblib.dump(self.pipeline, path)
        print(f"  [OK] Model saved: {path}")

    def load(self, path: str = MODEL_PATH):
        self.pipeline = joblib.load(path)
        self.is_trained = True
        print(f"  [OK] Model loaded: {path}")


def print_section(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metrics(model_name: str, y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else None

    print(f"\n  > Model: {model_name}")
    print(f"    Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    if auc:
        print(f"    ROC-AUC  : {auc:.4f}")
    print()
    print("  Confusion Matrix (Onside / Offside):")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print()
    print("  Classification Report:")
    report = classification_report(y_true, y_pred, target_names=["Onside", "Offside"])
    for line in report.strip().split("\n"):
        print(f"    {line}")


def run_cross_validation(detector: OffsideDetector, X: pd.DataFrame, y: pd.Series):
    scores = cross_val_score(detector.pipeline, X, y, cv=5, scoring="accuracy")
    print(f"\n  Cross-Validation (5-fold):")
    print(f"    Scores:  {[f'{s:.4f}' for s in scores]}")
    print(f"    Mean:    {scores.mean():.4f} +/- {scores.std():.4f}")


REALTIME_TESTS = [
    ((70.0, 50.0), (72.0, 50.0), (65.0, 48.0), "Clear OFFSIDE - teammate 7m ahead of defender"),
    ((65.0, 30.0), (60.0, 30.0), (65.0, 35.0), "Onside - teammate 5m behind defender"),
    ((65.0, 50.0), (65.1, 50.0), (65.0, 50.0), "Borderline - 0.1m difference (OFFSIDE)"),
    ((65.0, 50.0), (64.9, 50.0), (65.0, 50.0), "Borderline - -0.1m difference (Onside)"),
    ((75.0, 20.0), (80.0, 20.0), (55.0, 40.0), "Flagrant OFFSIDE - 25m ahead of defender"),
    ((50.0, 60.0), (40.0, 60.0), (70.0, 55.0), "Comfortable Onside - own half"),
    ((65.0, 50.0), (65.0, 50.0), (65.0, 50.0), "Perfectly aligned - Onside (tie goes to attacker)"),
]


def run_realtime_tests(detector: OffsideDetector):
    print_section("REAL-TIME TESTS (simulated live readings)")
    print(f"  {'No':>3}  {'Description':<48}  {'Prediction':>10}  {'Conf%':>6}")
    print(f"  {'---'}  {'-'*48}  {'-'*10}  {'-'*6}")

    for i, (ps, tm, df, desc) in enumerate(REALTIME_TESTS, 1):
        sample = generate_realtime_sample(ps, tm, df)
        X_test = pd.DataFrame(sample)[FEATURES]
        pred = int(detector.predict(X_test)[0])
        proba = detector.predict_probe(X_test)[0]
        conf = proba[pred] * 100
        label = "[OFFSIDE]" if pred == 1 else "[Onside] "
        print(f"  {i:>3}.  {desc:<48}  {label}  {conf:>5.1f}%")


def main():
    print("\n" + "+" + "=" * 58 + "+")
    print("  SAOT - Semi-Automated Offside Technology v1.0")
    print("  Phase 1: Detection based on player coordinates")
    print("+" + "=" * 58 + "+")

    print_section("1. TRAINING DATA GENERATION")
    df = generate_offside_sample(n_samples=3000, seed=RANDOM_STATE)
    X = df[FEATURES]
    y = df[LABEL]

    print(f"  Total samples    : {len(df)}")
    print(f"  Offside (1)      : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Onside  (0)      : {(y == 0).sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"  Features used    : {FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Split: {len(X_train)} train / {len(X_test)} test (80/20, stratified)")

    print_section("2. MODEL TRAINING")

    models = {
        "Logistic Regression": OffsideDetector("logistic_regression"),
        "Random Forest      ": OffsideDetector("random_forest"),
    }

    trained = {}
    for name, detector in models.items():
        print(f"\n  Training: {name.strip()}...")
        detector.train(X_train, y_train)
        run_cross_validation(detector, X_train, y_train)
        trained[name] = detector

    print_section("3. TEST SET EVALUATION")

    best_model = None
    best_auc = 0

    for name, detector in trained.items():
        y_pred = detector.predict(X_test)
        y_proba = detector.predict_probe(X_test)
        print_metrics(name.strip(), y_test, y_pred, y_proba)

        auc = roc_auc_score(y_test, y_proba[:, 1])
        if auc > best_auc:
            best_auc = auc
            best_model = (name.strip(), detector)

    print_section("4. MODEL SAVING")
    print(f"\n  Best model: {best_model[0]} (AUC={best_auc:.4f})")
    best_model[1].save(MODEL_PATH)

    run_realtime_tests(best_model[1])

    rf_detector = trained["Random Forest      "]
    rf_clf = rf_detector.pipeline.named_steps["classifier"]
    print_section("6. FEATURE IMPORTANCES (Random Forest)")
    importances = rf_clf.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {feat:<15} {imp:.4f}  {bar}")

    print("\n" + "=" * 60)
    print("  [OK] Training and evaluation complete!")
    print(f"  [->] Model saved in: {os.path.abspath(MODEL_PATH)}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()