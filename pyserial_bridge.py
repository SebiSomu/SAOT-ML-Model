import serial

def listen_to_esp32():
    # Change to the right port of the real breadboard (ex: 'COM3')
    try:
        ser = serial.Serial('COM3', 115200)
        while True:
            data = ser.readline()
            print(f"I received data: {data}")
    except:
        print("The breadboard is not connected yet.")