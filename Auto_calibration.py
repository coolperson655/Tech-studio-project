from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput, parse_line
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import Hand_tracking as ht
import Multi_servo_control as msc
from queue import Queue
import threading

'''
STEP FOR CALIBRATION

safety features:
* determine max value first
* have max speed that finger can change, assume anything faster is hand tracking error and stop stim
* physical stop built into motor mount

Main loop:
* Determine max from user input(or have preset)
* have user start with standard curled hand position, increase until no curl or close to it is reached
* Start with known likely movments and moniter hand movement
    * Record when hand starts to move and position/speed associated with a stim level
* After trying know correlations if desired movements are not achieved start to change stim inputs
* use values to create the feedback function
* do test movements and have user confrom comfort
'''

class linearized_mapper:
    def __init__(self, raw_data):
        raw_data = np.asarray(raw_data)

        # Ensure monotonically increasing data
        raw_data = np.sort(raw_data)

        self.input_scale = np.linspace(0, 100, len(raw_data))
        self.output_data = raw_data

    def __call__(self, x):
        x = np.clip(x, 0, 100)
        return np.interp(x, self.input_scale, self.output_data)
    
    # data = np.exp(np.linspace(0, 5, 1000))  # exponential curve

    # mapper = LinearizedMapper(data)

    # mapper(0)     # smallest value
    # mapper(50)    # middle of stretched range
    # mapper(100)   # largest value

client = ht.client
frame_queue = Queue(maxsize=1)

def frame_producer():
    for frame in client.iter_events():
        if frame_queue.full():
            frame_queue.get_nowait()
        frame_queue.put(frame)

def shock_mapper(val:int, low_shock:int= 20, max_shock:int= 45,val_offset:int=0) -> int:
    val += val_offset
    intensity = (val / 100) * (max_shock - low_shock) + low_shock
    '''returns servo insensity for a disered shock intensity val (0-100)'''
    return int(intensity)

def calibration() -> tuple[int, int]:
    '''Returns low_shock and max_shock values for shock_mapper function'''
    # set_intensity(0) # Make sure tens is off before calibration
    input("TENS Calibration: Press Enter to start...")
    low_shock = None # First time the user can feel the shock (small tingle)
    move_start = None # First time the user can feel the shock start to move their arm(not currently used)
    max_shock = None # Maximum shock intensity the user can tolerate
    cancelled = False
    print('at any promt enter anything besides y or n to turn off tens and stop the program')
    i = 0
    while low_shock is None:
            i += 5
            # set_intensity(i)
            kep_press = input("Press y and enter if you feel the shock(small tingle), press n if not: ")
            if kep_press.lower() == 'y':
                low_shock = i
                print(f"Low shock set to {low_shock}")
            elif kep_press.lower() == 'n':
                print("Increasing shock intensity...")
            else:
                print("cancelling")
                # set_intensity(0)
                low_shock = 0
                move_start = 0
                max_shock = 0
                cancelled = True

    while max_shock is None:
            i += 1
            # set_intensity(i)
            kep_press = input("Press y and enter if you feel the shock reach your maximum intensity, press n if not: ")
            if kep_press.lower() == 'y':
                max_shock = i
                print(f"Max shock set to {max_shock}")
            elif kep_press.lower() == 'n':
                print("Increasing shock intensity...")
            else:
                print("cancelling")
                # set_intensity(0)
                low_shock = 0
                move_start = 0
                max_shock = 0
                cancelled = True    
    if cancelled:
        print("TENS calibration cancelled.")
        exit()


    print(f"Calibration complete. Low shock: {ls}, Max shock: {ms}")
    # set_intensity(0)
    return (low_shock, max_shock)

threading.Thread(target=frame_producer, daemon=True).start()

# ---------------- MAIN LOOP ----------------
TARGET_CURL = 0.85
CURL_TOLERANCE = 0.05
MAX_CURL_RATE = 0.02 # normalized curl/second,  highest allowed response speed for auto calibration response
state = "WAITING"
calibrating = True
previous_curl = None
max_stim = 50
starting_stim = 20
servo_pin_list = [1,2,3,4,5,6]
calibration_data = {} # each point in dict is pinnum:[stim intensity, [IM deltacurl, RP deltacurl, T deltacurl], [IM curl, RP curl, T curl]]

while calibrating == True:

    print('starting automatic calibration curl fingers to ~90 degrees')
    # ---------------- CHECK HAND POSITION ----------------
    while state == 'WAITING':
        curls = ht.normalize_curl(ht.finger_curls(frame_queue))
        # curl = curls[0]  # finger of interest

        if previous_curl is None:
            previous_curls = curls

        curl_rates = abs(np.array(curls) - np.array(previous_curls))
        previous_curls = curls

        curl_in_range = abs(all(curls) - TARGET_CURL) <= CURL_TOLERANCE
        curl_stable = curl_rates <= MAX_CURL_RATE
        if curl_in_range and curl_stable:
            state = "CALIBRATING"

    # ---------------- APPLY STIM AND MONITER ----------------
    breaker = False
    while state == "CALIBRATING":
        for servo_pin in servo_pin_list:
            intensities = np.empty(1)
            if breaker:
                break
            for intensity in range(starting_stim,max_stim + 1):
                msc.set_intensity(intensity,str(servo_pin))
                time.sleep(0.2)
                curls = ht.normalize_curl(ht.finger_curls(frame_queue))
                curl_rates = abs(np.array(curls) - np.array(previous_curls))
                previous_curls = curls
                curl_stable = curl_rates <= MAX_CURL_RATE
                # NEED TO MAKE FULL ARRAYS OF BEFORE ADDING TO DICT
                if curl_stable:
                    np.append(intensities,intensity)
                    calibration_data[servo_pin] = [intensity, np.array(curl_rates), np.array(curls)]
                    
                else:
                    print('loss of tracking detected: restarting')
                    state = 'WAITING'
                    breaker = True
                    break

# use calibration values to determine which servo is best for each finger
pin_2_curl = {} # pin:[string of finger, avg difference]
for pin in calibration_data:
    stim_i, delta_curls, pin_curls = calibration_data[pin]
    clearest_curl = None
    clearest_name = None
    for curl,name in zip(pin_curls, ['IM','RP','T']):
        subtractors = pin_curls.remove(curl)
        current_curl = curl
        for thing in subtractors:
           current_curl = np.subtract(current_curl,thing)
        if current_curl > clearest_curl: # might need to change this bc its arrays
            clearest_curl = current_curl
            clearest_name = name
        pin_2_curl[pin] = [clearest_name, np.mean(clearest_curl)]
