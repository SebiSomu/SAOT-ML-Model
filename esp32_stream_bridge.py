import cv2
import numpy as np
import requests
import threading
import time
import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional

HSV_TEAMMATE = {
    "name": "Teammate (orange)",
    "lower": np.array([5,  120, 120]),
    "upper": np.array([20, 255, 255]),
}

HSV_DEFENDER = {
    "name": "Defender (red)",
    "lower1": np.array([0,   120, 120]), 
    "upper1": np.array([8,   255, 255]),
    "lower2": np.array([170, 120, 120]),
    "upper2": np.array([179, 255, 255]),
}

STREAM_TIMEOUT = 10 
FRAME_QUEUE_SIZE = 2    
MIN_BLOB_AREA = 200  
FIELD_RECT_OVERRIDE: Optional[tuple] = None


@dataclass
class DetectedPlayers:
    teammate_px:  Optional[tuple] = None  
    defender_px:  Optional[tuple] = None
    teammate_pos: Optional[tuple] = None  
    defender_pos: Optional[tuple] = None
    frame_id:     int = 0
    timestamp:    float = field(default_factory=time.time)

    @property
    def has_offside_pair(self) -> bool:
        return self.teammate_pos is not None and self.defender_pos is not None


class ESP32StreamReader:

    def __init__(self, url: str):
        self.url = url
        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.fps = 0.0
        self._last_fps_time = time.time()
        self._frames_since_fps = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[StreamReader] Connected to: {self.url}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _read_loop(self):
        while self._running:
            try:
                response = requests.get(
                    self.url,
                    stream=True,
                    timeout=STREAM_TIMEOUT,
                    headers={"Accept": "multipart/x-mixed-replace"}
                )
                print(f"[StreamReader] Connection established (status {response.status_code})")

                buffer = b""
                for chunk in response.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    buffer += chunk

                    while True:
                        start = buffer.find(b'\xff\xd8') 
                        end   = buffer.find(b'\xff\xd9') 
                        if start == -1 or end == -1 or end < start:
                            break
                        jpeg_data = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        frame = cv2.imdecode(
                            np.frombuffer(jpeg_data, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            with self._lock:
                                self._frame = frame
                            self.frame_count += 1
                            self._frames_since_fps += 1

                            now = time.time()
                            if now - self._last_fps_time >= 1.0:
                                self.fps = self._frames_since_fps / (now - self._last_fps_time)
                                self._frames_since_fps = 0
                                self._last_fps_time = now

            except requests.exceptions.ConnectionError:
                print(f"[StreamReader] Connection lost, retry in 2s...")
                time.sleep(2)
            except Exception as e:
                print(f"[StreamReader] Error: {e}, retry in 2s...")
                time.sleep(2)


class PlayerColorTracker:

    def __init__(self, min_blob_area: int = MIN_BLOB_AREA):
        self.min_area = min_blob_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def _find_blob(self, mask: np.ndarray) -> Optional[tuple]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        if area < self.min_area:
            return None

        M = cv2.moments(biggest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    def detect(self, frame: np.ndarray) -> dict:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_tm = cv2.inRange(hsv, HSV_TEAMMATE["lower"], HSV_TEAMMATE["upper"])
        mask_def = cv2.bitwise_or(
            cv2.inRange(hsv, HSV_DEFENDER["lower1"], HSV_DEFENDER["upper1"]),
            cv2.inRange(hsv, HSV_DEFENDER["lower2"], HSV_DEFENDER["upper2"]),
        )

        return {
            "teammate": self._find_blob(mask_tm),
            "defender": self._find_blob(mask_def),
            "masks":    {"tm": mask_tm, "def": mask_def},
        }

    def draw_detections(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        vis = frame.copy()
        colors = {
            "teammate": (0, 170, 255),  
            "defender": (50,  50, 220),  
        }
        labels = {"teammate": "TM", "defender": "DEF"}

        for key in ["teammate", "defender"]:
            pos = detections.get(key)
            if pos:
                cx, cy = pos
                color = colors[key]
                cv2.circle(vis, (cx, cy), 12, color, -1)
                cv2.circle(vis, (cx, cy), 14, (255, 255, 255), 2)
                cv2.putText(vis, labels[key], (cx - 12, cy - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return vis


class FieldAutoDetector:

    @staticmethod
    def detect(frame: np.ndarray) -> tuple:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(biggest)
            margin = 10
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(w, x + bw + margin)
            y1 = min(h, y + bh + margin)
            print(f"[FieldAutoDetect] Field detected: ({x0},{y0}) -> ({x1},{y1})")
            return x0, y0, x1, y1

        print(f"[FieldAutoDetect] Green not found, using full frame: (0,0) -> ({w},{h})")
        return 0, 0, w, h


class ESP32StreamBridge:

    def __init__(
        self,
        cam_url: str,
        judge,
        on_verdict: Optional[Callable] = None,
        field_rect: Optional[tuple] = None,
        show_debug: bool = False,
    ):
        self.cam_url    = cam_url
        self.judge      = judge
        self.on_verdict = on_verdict
        self.show_debug = show_debug

        self.reader  = ESP32StreamReader(cam_url)
        self.tracker = PlayerColorTracker()

        self._field_rect   = field_rect or FIELD_RECT_OVERRIDE
        self._coord_bridge = None  
        self._running      = False
        self._thread: Optional[threading.Thread] = None

        self._lock          = threading.Lock()
        self.last_detection: Optional[DetectedPlayers] = None
        self.last_verdict:   Optional[dict] = None
        self.frame_id = 0

    def start(self):
        self.reader.start()
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("[Bridge] Started. Waiting for first frame...")

    def stop(self):
        self._running = False
        self.reader.stop()
        if self._thread:
            self._thread.join(timeout=3)
        cv2.destroyAllWindows()
        print("[Bridge] Stopped.")

    def get_latest(self) -> tuple[Optional[DetectedPlayers], Optional[dict]]:
        """Returns the latest detection and verdict (thread-safe)."""
        with self._lock:
            return self.last_detection, self.last_verdict

    def _ensure_coord_bridge(self, frame: np.ndarray):
        """Initializes CoordinateBridge after first frame."""
        if self._coord_bridge is not None:
            return
        from detector_bridge import CoordinateBridge

        if self._field_rect:
            rect = self._field_rect
        else:
            rect = FieldAutoDetector.detect(frame)

        self._coord_bridge = CoordinateBridge(rect)
        print(f"[Bridge] CoordinateBridge initialized with rect={rect}")

    def _process_loop(self):
        from detector_bridge import CoordinateBridge

        last_frame_id = -1
        print("[Bridge] Processing loop started.")

        while self._running:
            frame = self.reader.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            current_id = self.reader.frame_count
            if current_id == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = current_id

            self._ensure_coord_bridge(frame)

            detections = self.tracker.detect(frame)
            self.frame_id += 1

            det = DetectedPlayers(frame_id=self.frame_id)

            for role in ["teammate", "defender"]:
                px_pos = detections.get(role)
                if px_pos:
                     setattr(det, f"{role}_px", px_pos)
                     field_pos = self._coord_bridge.pixel_to_field(*px_pos)
                     setattr(det, f"{role}_pos", field_pos)

            verdict = None
            if det.has_offside_pair:
                verdict = self.judge.judge(det.teammate_pos, det.defender_pos)
            elif det.teammate_pos and not det.defender_pos:
                verdict = {"label": "No defender", "is_offside": False,
                            "confidence": 0.0, "x_diff": 0.0}
            else:
                verdict = {"label": "No detection", "is_offside": False,
                            "confidence": 0.0, "x_diff": 0.0}

            with self._lock:
                self.last_detection = det
                self.last_verdict   = verdict

            if self.on_verdict and verdict:
                try:
                     self.on_verdict(det, verdict)
                except Exception as e:
                     print(f"[Bridge] Callback error: {e}")

            if self.show_debug:
                vis = self.tracker.draw_detections(frame, detections)
                self._draw_debug_overlay(vis, det, verdict)
                cv2.imshow("SAOT - ESP32 Debug", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                     self._running = False
                     break

    def _draw_debug_overlay(self, frame: np.ndarray,
                              det: DetectedPlayers, verdict: Optional[dict]):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 25), -1)

        if verdict:
            is_off = verdict.get("is_offside", False)
            lbl    = verdict.get("label", "N/A")
            conf   = verdict.get("confidence", 0.0)
            color  = (0, 40, 220) if is_off else (50, 200, 50)
            cv2.putText(frame, f"{lbl}  {conf*100:.0f}%", (10, 38),
                        font, 0.9, color, 2, cv2.LINE_AA)

        fps_str = f"FPS: {self.reader.fps:.1f}  Frame: {det.frame_id}"
        cv2.putText(frame, fps_str, (w - 230, 22),
                    font, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
        y_off = h - 70
        for role, label in [("teammate", "TM"), ("defender", "DEF")]:
            pos = getattr(det, f"{role}_pos")
            if pos:
                txt = f"{label}: ({pos[0]:.1f}, {pos[1]:.1f})"
                cv2.putText(frame, txt, (10, y_off),
                            font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
                y_off += 20

        if verdict and verdict.get("is_offside") and det.defender_px:
            dx, dy = det.defender_px
            cv2.line(frame, (dx, 60), (dx, h), (0, 40, 220), 2)
            cv2.putText(frame, "OFFSIDE LINE", (dx + 5, 80),
                        font, 0.4, (0, 40, 220), 1, cv2.LINE_AA)


def run_calibration(cam_url: str):
    print("\n[Calibration] Connecting to camera...")
    reader = ESP32StreamReader(cam_url)
    reader.start()
    for _ in range(50):
        if reader.get_frame() is not None:
            break
        time.sleep(0.1)

    cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H_min", "HSV Calibration",  0,  179, lambda x: None)
    cv2.createTrackbar("H_max", "HSV Calibration", 20,  179, lambda x: None)
    cv2.createTrackbar("S_min", "HSV Calibration", 100, 255, lambda x: None)
    cv2.createTrackbar("S_max", "HSV Calibration", 255, 255, lambda x: None)
    cv2.createTrackbar("V_min", "HSV Calibration", 100, 255, lambda x: None)
    cv2.createTrackbar("V_max", "HSV Calibration", 255, 255, lambda x: None)

    print("[Calibration] Adjust sliders. S=Save, Q=Quit")

    while True:
        frame = reader.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H_min", "HSV Calibration")
        h_max = cv2.getTrackbarPos("H_max", "HSV Calibration")
        s_min = cv2.getTrackbarPos("S_min", "HSV Calibration")
        s_max = cv2.getTrackbarPos("S_max", "HSV Calibration")
        v_min = cv2.getTrackbarPos("V_min", "HSV Calibration")
        v_max = cv2.getTrackbarPos("V_max", "HSV Calibration")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask  = cv2.inRange(hsv, lower, upper)

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay    = cv2.bitwise_and(frame, frame, mask=mask)
        combined   = np.hstack([mask_color, overlay])
        
        cv2.imshow("HSV Calibration", frame)
        cv2.imshow("Mask",         combined)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            print(f"\n[Calibration] Values saved:")
            print(f'  "lower": np.array([{h_min}, {s_min}, {v_min}]),')
            print(f'  "upper": np.array([{h_max}, {s_max}, {v_max}]),')
        elif key in [ord('q'), 27]:
            break

    reader.stop()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SAOT ESP32-CAM Bridge")
    parser.add_argument("--ip",        default="192.168.1.100",
                        help="ESP32-CAM IP address (ex: 192.168.1.55)")
    parser.add_argument("--port",      default=80, type=int,
                        help="HTTP port (default: 80)")
    parser.add_argument("--show",      action="store_true",
                        help="Show debug window with detections")
    parser.add_argument("--calibrate", action="store_true",
                        help="Start HSV calibration mode")
    parser.add_argument("--model",     default="saot_model.pkl",
                        help="Path to ML model")
    parser.add_argument("--rect",      nargs=4, type=int, metavar=("X0","Y0","X1","Y1"),
                        help="Manual field rectangle in pixels")
    args = parser.parse_args()

    cam_url = f"http://{args.ip}:{args.port}/stream"

    if args.calibrate:
        run_calibration(cam_url)
        return

    from detector_bridge import MLOffsideJudge
    judge = MLOffsideJudge(args.model)

    prev_verdict = [None]
    def on_verdict(det: DetectedPlayers, verdict: dict):
        if verdict["label"] in ("No detection", "No defender"):
            return
        if verdict["is_offside"] != prev_verdict[0]:
            icon = "🚩" if verdict["is_offside"] else "✅"
            tm   = det.teammate_pos or (0, 0)
            df   = det.defender_pos or (0, 0)
            print(f"  {icon} {verdict['label']:<9}  "
                  f"conf={verdict['confidence']*100:.1f}%  "
                  f"TM({tm[0]:.1f},{tm[1]:.1f})  "
                  f"DEF({df[0]:.1f},{df[1]:.1f})  "
                  f"x_diff={verdict['x_diff']:+.2f}")
            prev_verdict[0] = verdict["is_offside"]

    field_rect = tuple(args.rect) if args.rect else None

    bridge = ESP32StreamBridge(
        cam_url=cam_url,
        judge=judge,
        on_verdict=on_verdict,
        field_rect=field_rect,
        show_debug=args.show,
    )

    print(f"\n[SAOT] Bridge started -> {cam_url}")
    print("  Ctrl+C or Q to stop\n")

    bridge.start()

    try:
        while True:
            time.sleep(1)
            det, verdict = bridge.get_latest()
            if det:
                print(f"  [Live] Frame {det.frame_id} | FPS: {bridge.reader.fps:.1f} | "
                      f"TM: {det.teammate_pos} | DEF: {det.defender_pos}", end="\r")
    except KeyboardInterrupt:
        print("\n[SAOT] Interrupt received.")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()