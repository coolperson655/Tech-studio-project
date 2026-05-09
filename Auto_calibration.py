from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput, parse_line
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import Hand_tracking as ht
import Multi_servo_control as msc
from queue import Queue
import threading
import pandas as pd

from collections import defaultdict

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

#FUNCTIONS FOR MAPPING
def score_pin(entry, target:str):
    '''target: IM, RP, or T '''
    stim, (im, rp, t), (imcurl,rpcurl,tcurl) = entry
    if target == 'IM':
        return im - (abs(rp) + abs(t))  # reward IM, penalize others
    elif target == 'RP':
        return rp - (abs(im) + abs(t))
    elif target == 'T':
        return t - (abs(rp) + abs(im))
    elif target == 'IMCURL':
        return im - (abs(rp) + abs(t))  # reward IM, penalize others
    elif target == 'RPCURL':
        return rp - (abs(im) + abs(t))
    elif target == 'TCURL':
        return t - (abs(rp) + abs(im))


def group_by_pin(dataset):
    grouped = defaultdict(list)
    for pin, entry in dataset.items():
        grouped[pin].append(entry)
    return grouped

def select_best_pin(data, target):
    '''target: IM, RP, or T '''
    grouped = group_by_pin(data)
    best_pin = None
    best_score = float('-inf')

    for pin, entries in grouped.items():
        avg_score = sum(score_pin(e, target) for e in entries) / len(entries)
        if avg_score > best_score:
            best_score = avg_score
            best_pin = pin

    return best_pin

def extract_data(entries,target, threshold=0.05):
    pairs = []
    sd ={} # storage_dict
    for stim, (sd['IM'], sd['RP'], sd['T']), (sd['IMCURL'],sd['IMCURL'],sd['TCURL']) in entries:
        if abs(imcurl) > threshold:  # filter non-responsive
            pairs.append((stim, im))
    return sorted(pairs)


def build_linear_map(pairs):
    if len(pairs) < 2:
        raise ValueError("Not enough data")

    stim_vals, im_vals = zip(*pairs)

    # normalize stim → 0–100
    min_s, max_s = min(stim_vals), max(stim_vals)
    norm_stim = [(s - min_s) / (max_s - min_s) * 100 for s in stim_vals]

    def mapper(x):
        x = np.clip(x, 0, 100)
        return np.interp(x, norm_stim, im_vals)

    return mapper

def build_controller(data, target):
    '''target: IM, RP, or T '''
    target = str(target.upper())
    if target not in ["IM", "RP", "T","IMCURL",'RPCURL','TCURL']:
        ValueError('Invalid target; use IM, RP, T, IMCURL, RPCURL, or TCURL')
    grouped = group_by_pin(data)
    pin = select_best_pin(data, target)

    entries = grouped[pin]
    pairs = extract_data(entries,target)

    func = build_linear_map(pairs)

    return pin, func


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
# ---------------- THREADING FUNCTIONS ----------------
def frame_producer():
    for frame in client.iter_events():
        # if stop_event.is_set():   # 👈 check if we should stop
        #     break

        try:
            frame_queue.put(frame, block=False)
        except:
            frame_queue.get_nowait()
            frame_queue.put(frame)

# ---------------- INITIALIZE CLIENT AND FRAME LOOP ----------------
try:
    client = ht.create_client(port=8000)
except:
    print('client not established')
    quit()
frame_queue = Queue(maxsize=1)
thread = threading.Thread(target=frame_producer, daemon=True)
thread.start()
try:
    print(f'thread initiated, test frame {ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))}')
except:
    time.sleep(1)
    try:
        print(f'thread initiated, test frame {ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))}')
    except:
        print(f'failed to retrieve from que available data:{frame_queue.qsize()}')
        quit()
# ---------------- MAIN LOOP ----------------
TARGET_CURL = 0.55
CURL_TOLERANCE = 0.10
MAX_CURL_RATE = 0.02 # normalized curl/cycle,  highest allowed response speed for auto calibration response
state = "WAITING"
calibrating = True
previous_curl = None
max_stim = 50
starting_stim = 20
stim_levels = range(starting_stim,max_stim+1)
servo_pin_list = [2,4,6,8,10,12] # pins to calibrate back curl with
multindex = pd.MultiIndex.from_product([servo_pin_list, stim_levels ], names=["servo pin", "stim_level"])
col_names = ['IM','RP','T','IMCURL','RPCURL','TCURL']
df = pd.DataFrame(index=multindex, columns=col_names)
# df['IM'][2][35] indexing example
while calibrating == True:

    print('starting automatic calibration curl fingers to ~90 degrees and relax')
    # ---------------- CHECK HAND POSITION ----------------
    while state == 'WAITING':
        curls = ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))
        # curl = curls[0]  # finger of interest
        if previous_curl is None:
            previous_curl = curls
            print('first loop')
            continue
        curl_rates = abs(np.array(curls) - np.array(previous_curl))
        previous_curl = curls
        if np.all(curl_rates <= MAX_CURL_RATE): # checking if curl is stable
            previous_curls = curls
            curl_in_range = np.all(np.abs(np.array(curls) - TARGET_CURL) <= CURL_TOLERANCE)
            # curl_in_range = abs(all(curls) - TARGET_CURL) <= CURL_TOLERANCE
            curl_stable = np.all(curl_rates <= MAX_CURL_RATE)
            if curl_in_range:
                print('\ngood hand position try and relax and keep your hand there for 1 second')
                time.sleep(3)
                curls_check = ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))
                curl_in_range_check = np.all(np.abs(np.array(curls) - TARGET_CURL) <= CURL_TOLERANCE)
                if curl_in_range_check:
                    print('\npostition confirmed continuing to motor calibration')
                    state = "CALIBRATING"
                    continue

            print(end='\r')
            for i in [0,1,2]:
                curl_diff = curls[i] - TARGET_CURL
                if curl_diff >= CURL_TOLERANCE:
                    print(f'curl {col_names[i]} LESS ', end='')#curl is {curls[i]}', end='')
                    break
                elif curl_diff <= -1 * CURL_TOLERANCE:
                    print(f'curl {col_names[i]} MORE  ', end='')#curl is {curls[i]}', end='')
                else:
                    print(f'curl {col_names[i]} HOLD ', end='')#curl is {curls[i]}', end='')
        else:
            print(end='\r')
            print('tracking unstable please hold still and get in view of the cameras',end='')


        # except:
        #     state = "WAITING"
        #     print('\r')``
        #     print('tracking lost')
    print('\nloop ended')
    calibrating = False
    # ---------------- APPLY STIM AND MONITER ----------------
    print('waiting for user input to continue')
    input_val = input('')
#     breaker = False
#     while state == "CALIBRATING":
#         for servo_pin in servo_pin_list:
#             if breaker:
#                 break
#             for intensity in range(starting_stim,max_stim + 1):
#                 msc.set_intensity(intensity,str(servo_pin))
#                 time.sleep(0.2)
#                 curls = ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))
#                 curl_rates = abs(np.array(curls) - np.array(previous_curls))
#                 if np.any(curls) < 0.1: # This checks if any finger have uncurled
#                     print(f'full extenstion detected: continuing, intesity={intensity}')
#                     break
#                 previous_curls = curls
#                 curl_stable = curl_rates <= MAX_CURL_RATE
#                 # NEED TO MAKE FULL ARRAYS OF BEFORE ADDING TO DICT
#                 if curl_stable:
#                     df["IM"][servo_pin][intensity] = curls[0]
#                     df["RP"][servo_pin][intensity] = curls[1]
#                     df["T"][servo_pin][intensity] = curls[2]
#                     df["IMCURLS"][servo_pin][intensity] = curl_rates[0]
#                     df["RPURLS"][servo_pin][intensity] = curl_rates[1]
#                     df["TCURLS"][servo_pin][intensity] = curl_rates[2]
                    
#                 else:
#                     print('loss of tracking detected: restarting')
#                     state = 'WAITING'
#                     breaker = True
#                     break
# print(df)

# #FOR CALCULATION CREATE MULTIINDEX DF WITH PIN,INTENSITY AND COLUMNS OF IM PR T IMCURL RPCURL TCURL
# # FOR EACH DESIRED TARGET FIND THE PIN THAT BEST ALLIGNS WITH IT
# # CREATE THE LINEAR MAP BETWEEN PIN AND TARGET AS BEFORE

# # # use calibration values to determine which servo is best for each finger
# # pin_2_curl = {} # pin:[string of finger, avg difference]
# # for pin in calibration_data:
# #     stim_i, delta_curls, pin_curls = calibration_data[pin]
# #     clearest_curl = None
# #     clearest_name = None
# #     for curl,name in zip(pin_curls, ['IM','RP','T']):
# #         subtractors = pin_curls.remove(curl)
# #         current_curl = curl
# #         for thing in subtractors:
# #            current_curl = np.subtract(current_curl,thing)
# #         if current_curl > clearest_curl: # might need to change this bc its arrays
# #             clearest_curl = current_curl
# #             clearest_name = name
# #         pin_2_curl[pin] = [clearest_name, np.mean(clearest_curl)]
