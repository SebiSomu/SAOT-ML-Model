import sys
import os
import argparse
import time
import threading

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
        import joblib
        joblib.dump(pipe, MODEL_PATH)
        print(f"  Model saved: {MODEL_PATH}\n")
    else:
        print(f"[SAOT] Model found: {MODEL_PATH}")


class LivePositionInjector:

    def __init__(self, app, smoothing: float = 0.35):
        self.app       = app
        self.alpha     = 1.0 - smoothing  
        self._lock     = threading.Lock()
        self.active    = False
        self.last_update = 0.0
        self.frames_received = 0
        self.offside_count   = 0

    def on_verdict(self, det, verdict: dict):
        from esp32_stream_bridge import DetectedPlayers

        with self._lock:
            self.frames_received += 1
            self.last_update = time.time()

            if det.teammate_pos:
                old = self.app.positions["teammate"]
                new = det.teammate_pos
                self.app.positions["teammate"] = (
                    round(old[0] * (1 - self.alpha) + new[0] * self.alpha, 2),
                    round(old[1] * (1 - self.alpha) + new[1] * self.alpha, 2),
                )

            if det.defender_pos:
                old = self.app.positions["defender"]
                new = det.defender_pos
                self.app.positions["defender"] = (
                    round(old[0] * (1 - self.alpha) + new[0] * self.alpha, 2),
                    round(old[1] * (1 - self.alpha) + new[1] * self.alpha, 2),
                )

            if det.passer_pos:
                old = self.app.positions["passer"]
                new = det.passer_pos
                self.app.positions["passer"] = (
                    round(old[0] * (1 - self.alpha) + new[0] * self.alpha, 2),
                    round(old[1] * (1 - self.alpha) + new[1] * self.alpha, 2),
                )

            if verdict.get("label") not in ("No detection", "No defender"):
                self.app.verdict = verdict
                self.active = True
                if verdict["is_offside"]:
                    self.offside_count += 1
            else:
                self.app.verdict = self.app.judge.judge(
                    self.app.positions["teammate"],
                    self.app.positions["defender"],
                )

    def is_stale(self, timeout_sec: float = 2.0) -> bool:
        return (time.time() - self.last_update) > timeout_sec

    def stats(self) -> str:
        return (f"Frames: {self.frames_received} | "
                f"Offside detected: {self.offside_count}x | "
                f"Active: {'Yes' if self.active else 'No'}")


def main():
    parser = argparse.ArgumentParser(description="SAOT Main Live")
    parser.add_argument("--ip",        default="192.168.1.100",
                        help="IP ESP32-CAM (ex: 192.168.1.55)")
    parser.add_argument("--port",      default=80, type=int)
    parser.add_argument("--retrain",   action="store_true",
                        help="Forteaza reantrenarea modelului")
    parser.add_argument("--offline",   action="store_true",
                        help="Ruleaza fara camera (mouse drag, fallback)")
    parser.add_argument("--debug",     action="store_true",
                        help="Afiseaza fereastra de debug camera")
    parser.add_argument("--calibrate", action="store_true",
                        help="Starts HSV calibration before starting")
    parser.add_argument("--smoothing", default=0.35, type=float,
                        help="Smoothing pozitii [0.0-0.9] (default: 0.35)")
    parser.add_argument("--rect",      nargs=4, type=int, metavar=("X0","Y0","X1","Y1"),
                        help="Dreptunghi manual teren in pixeli")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    ensure_model(args.retrain)

    from detector_bridge import MLOffsideJudge
    from opencv_field import SAOTApp3

    judge = MLOffsideJudge(MODEL_PATH)
    app   = SAOTApp3(judge)

    if args.calibrate:
        from esp32_stream_bridge import run_calibration
        cam_url = f"http://{args.ip}:{args.port}/stream"
        print("[SAOT] HSV calibration mode started. Close window to continue.")
        run_calibration(cam_url)
        print("[SAOT] Calibration complete. Starting application...\n")

    if args.offline:
        print("\n[SAOT] OFFLINE mode (no camera). Drag & drop active.")
        app.run()
        return
    cam_url = f"http://{args.ip}:{args.port}/stream"
    print(f"\n[SAOT] LIVE mode -> {cam_url}")

    from esp32_stream_bridge import ESP32StreamBridge

    injector = LivePositionInjector(app, smoothing=args.smoothing)
    field_rect = tuple(args.rect) if args.rect else None

    bridge = ESP32StreamBridge(
        cam_url=cam_url,
        judge=judge,
        on_verdict=injector.on_verdict,
        field_rect=field_rect,
        show_debug=args.debug,
    )

    bridge.start()

    print("[SAOT] Waiting for live detections (max 8s)...")
    for _ in range(80):
        if injector.active:
            print("[SAOT] Live detections received. Starting visualization.\n")
            break
        time.sleep(0.1)
    else:
        print("[SAOT] WARN: No live detections in 8s. "
              "Check IP or calibrate HSV colors.")
        print("  Starting visualization with default positions (mouse drag active).\n")

    def status_loop():
        while bridge._running:
            time.sleep(10)
            stale_warn = " [STALE!]" if injector.is_stale(3) else ""
            print(f"\n  [Status] {injector.stats()}{stale_warn} | "
                  f"Camera FPS: {bridge.reader.fps:.1f}")

    t_status = threading.Thread(target=status_loop, daemon=True)
    t_status.start()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        print(f"\n[SAOT] Session finished. {injector.stats()}")


if __name__ == "__main__":
    main()