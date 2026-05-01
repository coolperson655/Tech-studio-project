import pyfirmata2
import time
import serial
from pyfirmata2 import Pin, Arduino
import random
import os
from dotenv import load_dotenv

# list usb devices
# import serial.tools.list_ports
# ports = list(serial.tools.list_ports.comports())
# for p in ports:
#     print(p)
# exit()



def servo_d_pin_str(i_:int) -> str:
    return 'd:' + str(i_) + ':s'

def set_servo(v_:int, servo_pin_:int) -> None:
    global PINS
    PINS[servo_pin_].write(v_)


def set_intensity(v_:int, servo_pin_num:str) -> None:
    global SERVO_RANGE
    if v_ > 100: v_ = 100
    if v_ < 0: v_ = 0
    norm = v_/100
    abs_pos = norm*(SERVO_RANGE[1]-SERVO_RANGE[0]) + SERVO_RANGE[0]
    set_servo(abs_pos, int(servo_pin_num))


# board initialization
try:
    load_dotenv()
    COM_PORT:str = os.getenv("COM_PORT")
    BOARD = Arduino('COM5')
    PINS:dict[int, Pin] = {i : BOARD.get_pin(servo_d_pin_str(i)) for i in range(2, 14, 2)}  # claim pins [2,4,6,8,10,12]

    # not sure if you're using this still
    SERVO_RANGE:list[int] = [130, 193]  # actual mechanical limits of servo range when attached to tens

except AttributeError as e:
    raise RuntimeError("Tens Arduino appears to be unplugged.")
except serial.serialutil.SerialException:
    raise RuntimeError("tens Arduino appears to be unplugged.")




def main():
    """
    set servo position to arbitrary position
    """
    while True:
        try:
            # # user setting
            # vstr = input("val: ")
            # pin_num = input('pin num:')
            # if vstr == '': v = 0
            # else: v = int(vstr)
            # set_intensity(v, pin_num)

            # loop to test
            sleep_amt:float = 1
            # while 1:
            #     for i in range(2,14,2):
            #         v = random.randint(0,100)
            #         set_intensity(v, i)               
            #         time.sleep(sleep_amt)
            v = 0
            while 2:
                v += 20
                if v > 100:
                    v = 0
                set_intensity(v, 2)  
                set_intensity(v, 4)  
                set_intensity(v, 6)  
                set_intensity(v, 8)  
                set_intensity(v, 10)  
                set_intensity(v, 12)
                time.sleep(sleep_amt)   


        except KeyboardInterrupt:
            exit()





if __name__ == "__main__":
    main()
