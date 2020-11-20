import serial

class Sensors_driver():
    def __init__(self, Port_name):
        self.ser=serial.Serial(
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1)
        self.ser.port = Port_name

        try:
            self.ser.open()
            print("SensorsBoard connected on port " + Port_name)
        except serial.serialutil.SerialException:
            print("SensorsBoard NOT connected on port " + Port_name)

    def __del__(self):
        self.ser.close()
        print("Sensors driver connection closed")
