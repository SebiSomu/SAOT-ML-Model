"""
SAOT Phase 2 - Detector Bridge
Converts pixel coordinates <-> field coordinates (0-100 scale)
and interfaces with the trained ML model.

Future upgrade path:
    Phase 3 - replace pixel positions (currently set by mouse drag)
    with YOLOv8 detections from a real camera frame:

    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    results = yolo(frame)
    for box in results[0].boxes:
        pixel_pos = box.xywh[0][:2].tolist()
        field_pos = bridge.pixel_to_field(pixel_pos)
"""

import numpy as np
import pandas as pd
import joblib
import os

FEATURES = ["passer_x", "passer_y", "teammate_x", "teammate_y",
            "defender_x", "defender_y", "x_diff"]
MODEL_PATH = "saot_model.pkl"


class CoordinateBridge:

    def __init__(self, field_rect: tuple):
        self.x0, self.y0, self.x1, self.y1 = field_rect
        self.fw = self.x1 - self.x0  # field width in pixels
        self.fh = self.y1 - self.y0  # field height in pixels

    def pixel_to_field(self, px: int, py: int) -> tuple[float, float]:
        fx = np.clip((px - self.x0) / self.fw * 100, 0, 100)
        fy = np.clip((py - self.y0) / self.fh * 100, 0, 100)
        return round(float(fx), 2), round(float(fy), 2)

    def field_to_pixel(self, fx: float, fy: float) -> tuple[int, int]:
        px = int(self.x0 + fx / 100 * self.fw)
        py = int(self.y0 + fy / 100 * self.fh)
        return px, py

    def offside_line_pixels(self, defender_fx: float) -> tuple[int, int, int, int]:
        px, _ = self.field_to_pixel(defender_fx, 0)
        return px, self.y0, px, self.y1


class MLOffsideJudge:

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run model.py first to train and save the model."
            )
        self.pipeline = joblib.load(model_path)
        print(f"[OK] Model loaded from: {os.path.abspath(model_path)}")

    def judge(self, passer_pos: tuple, teammate_pos: tuple,
              defender_pos: tuple) -> dict:

        px, py = passer_pos
        tx, ty = teammate_pos
        dx, dy = defender_pos
        x_diff = tx - dx

        sample = {
            "passer_x": [px], "passer_y": [py],
            "teammate_x": [tx], "teammate_y": [ty],
            "defender_x": [dx], "defender_y": [dy],
            "x_diff": [x_diff],
        }
        X = pd.DataFrame(sample)[FEATURES]
        pred = int(self.pipeline.predict(X)[0])
        probe = self.pipeline.predict_proba(X)[0]
        confidence = float(probe[pred])

        return {
            "is_offside": bool(pred),
            "confidence": confidence,
            "x_diff": x_diff,
            "label": "OFFSIDE" if pred else "Onside",
        }