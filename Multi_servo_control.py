import pyfirmata2
import time
import serial

# list usb devices
# import serial.tools.list_ports
# ports = list(serial.tools.list_ports.comports())
# for p in ports:
#     print(p)
# exit()

try:
    board = pyfirmata2.Arduino('COM5')
except AttributeError as e:
    raise RuntimeError("Tens Arduino appears to be unplugged.")
except serial.serialutil.SerialException:
    raise RuntimeError("tens Arduino appears to be unplugged.")

# digital:pin5:seeervo...?
servo_range:list[int] = [130, 193]  # actual mechanical limits of servo range when attached to tens



def set_servo(v_:int, servo_pin_num:str) -> None:
    servo_pin = 'd:' + str(servo_pin_num) + ':s'
    servo_pin.write(v_)


def set_intensity(v_:int, servo_pin_num:str) -> None:
    servo_pin_i = 'd:' + str(servo_pin_num) + ':s'
    global servo_range
    if v_ > 100: v_ = 100
    if v_ < 0: v_ = 0
    norm = v_/100
    abs_pos = norm*(servo_range[1]-servo_range[0]) + servo_range[0]
    set_servo(abs_pos,servo_pin_i)


def main():
    """
    set servo position to arbitrary position
    """
    set_servo(servo_range[0], '5')
    while True:
        try:
            vstr = input("val: ")
            pin_num = input('pin num:')
            if vstr == '': v = 0
            else: v = int(vstr)

            set_intensity(v, pin_num)
        except KeyboardInterrupt:
            exit()





if __name__ == "__main__":
    main()
