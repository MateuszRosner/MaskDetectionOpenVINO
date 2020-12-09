import serial

ser=serial.Serial(
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1)
ser.port = "/dev/ttyACM0"

ser.open()

while True:
    x = input()
    if x == 'a':
        ser.write(b'1')
    elif x == 's':
        ser.write(b'2')