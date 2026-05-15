import time
import random

def simulate_esp32_serial(port: str = "COM4"):
    import serial
    try:
        ser = serial.Serial(port, 115200)
        print(f"[Sim] Started simulator on {port}")
    except Exception as e:
        print(f"[Sim] Error: {e}")
        print("Tip: Use a virtual serial port pair (e.g., COM3 <-> COM4) to test.")
        return

    while True:
        tm_x = 160 + random.uniform(-20, 100)
        tm_y = 120 + random.uniform(-50, 50)
        
        def_x = 180 + random.uniform(-5, 5)
        def_y = 120 + random.uniform(-60, 60)
        
        data_line = f"DATA:{tm_x:.2f},{tm_y:.2f},{def_x:.2f},{def_y:.2f}\n"
        ser.write(data_line.encode('utf-8'))
        print(f"[Sim] Sent: {data_line.strip()}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    simulate_esp32_serial()
