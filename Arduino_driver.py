import serial

class LED_driver():
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
            print("LED driver connected on port " + Port_name)
            self.conn_state = True
        except serial.serialutil.SerialException:
            print("LED driver NOT connected on port " + Port_name)
            self.conn_state = False

    def __del__(self):
        self.ser.close()
        print("LEDs driver connection closed")

    def send_state(self, val):
        if (self.conn_state == True):
            self.ser.write(val)
