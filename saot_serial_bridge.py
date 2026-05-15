import serial
import threading
import time
from typing import Callable, Optional
from dataclasses import dataclass

@dataclass
class SerialDetection:
    tm_x: float
    tm_y: float
    def_x: float
    def_y: float

class SAOTSerialBridge:
    def __init__(self, port: str, baudrate: int = 115200, 
                 on_data: Optional[Callable] = None,
                 field_rect: tuple = (0, 0, 320, 240)):
        self.port = port
        self.baudrate = baudrate
        self.on_data = on_data
        self.field_rect = field_rect
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ser: Optional[serial.Serial] = None

    def start(self):
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            print(f"[SerialBridge] Connected to {self.port}")
        except Exception as e:
            print(f"[SerialBridge] Error connecting to {self.port}: {e}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._ser:
            self._ser.close()
        print("[SerialBridge] Stopped.")

    def _run_loop(self):
        while self._running:
            if self._ser and self._ser.in_waiting > 0:
                try:
                    line = self._ser.readline().decode('utf-8').strip()
                    if line.startswith("DATA:"):
                        data_str = line.split(":")[1]
                        parts = [float(p) for p in data_str.split(",")]
                        if len(parts) == 4:
                            det = SerialDetection(*parts)
                            if self.on_data:
                                self._process_and_callback(det)
                except Exception as e:
                    print(f"[SerialBridge] Parse error: {e}")
            time.sleep(0.01)

    def _process_and_callback(self, det: SerialDetection):
        x0, y0, x1, y1 = self.field_rect
        fw = x1 - x0
        fh = y1 - y0

        def to_field(px, py):
            if px < 0 or py < 0: return None
            fx = (px - x0) / fw * 100
            fy = (py - y0) / fh * 100
            return (round(fx, 2), round(fy, 2))

        tm_pos = to_field(det.tm_x, det.tm_y)
        def_pos = to_field(det.def_x, det.def_y)

        from esp32_stream_bridge import DetectedPlayers
        mock_det = DetectedPlayers()
        mock_det.teammate_pos = tm_pos
        mock_det.defender_pos = def_pos
        mock_det.teammate_px = (int(det.tm_x), int(det.tm_y)) if tm_pos else None
        mock_det.defender_px = (int(det.def_x), int(det.def_y)) if def_pos else None

        from detector_bridge import MLOffsideJudge
        
        if self.on_data:
            self.on_data(mock_det)

if __name__ == "__main__":
    def test_cb(det):
        print(f"Detected: TM={det.teammate_pos}, DEF={det.defender_pos}")
    
    bridge = SAOTSerialBridge("COM3", on_data=test_cb)
    bridge.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()
