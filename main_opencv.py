import sys
import os
import argparse

MODEL_PATH = "saot_model.pkl"


def ensure_model(force_retrain=False):
    if force_retrain or not os.path.exists(MODEL_PATH):
        print("[SAOT] Training model...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, roc_auc_score
        import joblib
        from data_generator import generate_offside_sample

        FEATURES = ["teammate_x", "teammate_y", "defender_x", "defender_y", "x_diff"]
        df = generate_offside_sample(n_samples=3000, seed=42)
        X, y = df[FEATURES], df["offside"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"  Accuracy: {acc*100:.2f}%  AUC: {auc:.4f}")
        joblib.dump(pipe, MODEL_PATH)
        print(f"  Model saved: {MODEL_PATH}\n")
    else:
        print(f"[SAOT] Model found: {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    ensure_model(args.retrain)

    from detector_bridge import MLOffsideJudge
    from opencv_field import SAOTApp3

    judge = MLOffsideJudge(MODEL_PATH)
    app   = SAOTApp3(judge)
    app.run()


if __name__ == "__main__":
    main()