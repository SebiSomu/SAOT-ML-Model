import sys
import os
import argparse

MODEL_PATH = "saot_model.pkl"


def ensure_model(force_retrain: bool = False):
    if force_retrain or not os.path.exists(MODEL_PATH):
        print("[SAOT] Model not found or retrain requested. Training now...")
        print("=" * 50)

        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, roc_auc_score
        import joblib

        from data_generator import generate_offside_sample

        FEATURES = ["passer_x", "passer_y", "teammate_x", "teammate_y",
                    "defender_x", "defender_y", "x_diff"]

        df = generate_offside_sample(n_samples=3000, seed=42)
        X = df[FEATURES]
        y = df["offside"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"  Accuracy : {acc*100:.2f}%  |  ROC-AUC: {auc:.4f}")

        joblib.dump(pipeline, MODEL_PATH)
        print(f"  Model saved: {MODEL_PATH}")
        print("=" * 50 + "\n")
    else:
        print(f"[SAOT] Using existing model: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="SAOT Phase 2 - OpenCV Offside Detector")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining before launching")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    ensure_model(force_retrain=args.retrain)

    from detector_bridge import MLOffsideJudge
    from opencv_field import SAOTApp

    judge = MLOffsideJudge(MODEL_PATH)
    app = SAOTApp(judge)
    app.run()


if __name__ == "__main__":
    main()