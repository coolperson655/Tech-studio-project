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
import sys 

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
def get_client_and_thread():
    global frame_queue
    global client
    try:
        client = ht.create_client(port=8000)
    except Exception as exc:
        raise RuntimeError("Client not retrieved, check port and IP are matching and devices are on the samw wifi") from exc
    try:
        frame_queue = Queue(maxsize=1)
        threading.Thread(target=frame_producer, daemon=True).start()
    except Exception as exc:
        raise RuntimeError("Failed to initilize thread") from exc
    try:
        print(f'thread initiated, test frame {ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))}')
    except:
        time.sleep(1)
        try:
            print(f'thread initiated, test frame {ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))}')
        except Exception as exc:
            raise RuntimeError(f'failed to retrieve from queue, available data:{frame_queue.qsize()}') from exc

def check_hand_curl(target_curl:float = 0.5,
                    curl_tolerance:float=0.1,
                    max_curl_rate:float=0.02):
    """Checks if hand is within TOLERANCE of TARGET CURL if not guides via text prompts
    Uses MAX_CURL_RATE as a globals"""
    global frame_queue
    previous_curl = None
    internal_state = 'checking'
    while internal_state == 'checking':
        curls = ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))
        if previous_curl is None:
            previous_curl = curls
            # print('first loop')
            continue
        curl_rates = abs(np.array(curls) - np.array(previous_curl))
        previous_curl = curls
        if np.all(curl_rates <= max_curl_rate): # checking if curl is stable
            previous_curl = curls
            curl_in_range = np.all(np.abs(np.array(curls) - target_curl) <= curl_tolerance)
            curl_stable = np.all(curl_rates <= max_curl_rate)
            if curl_in_range:
                sys.stdout.write('\x1b[2K') # should delete last line in terminal
                print('\ngood hand position try and relax and keep your hand there for 1 second')
                time.sleep(3)
                curls_check = ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))
                curl_in_range_check = np.all(np.abs(np.array(curls_check) - target_curl) <= curl_tolerance)
                if curl_in_range_check:
                    print('\npostition confirmed continuing to motor calibration')
                    internal_state = "CALIBRATING"
                    continue

            print(end='\r')
            for i in [0,1,2]:
                curl_diff = curls[i] - target_curl
                if curl_diff >= curl_tolerance:
                    print(f'curl {col_names[i]} LESS ', end='')#curl is {curls[i]}', end='')
                    break
                elif curl_diff <= -1 * curl_tolerance:
                    print(f'curl {col_names[i]} MORE  ', end='')#curl is {curls[i]}', end='')
                else:
                    print(f'curl {col_names[i]} HOLD ', end='')#curl is {curls[i]}', end='')
        else:
            print(end='\r')
            print('tracking unstable please hold still and get in view of the cameras',end='')
    return None

def create_mapping_df(servo_pins:list,
                      starting_stim:int,
                      max_stim:int,
                      return_df:pd.DataFrame,
                      max_curl_rate:float=0.04):
 """Adds the mapping values into the given dataframe, calls hand curl between servo to ensure hand is positioned correctly"""
 global frame_queue
 breaker = False
 internal_state = 'mapping'
 while internal_state == "mapping":
        for servo_pin in servo_pins:
            print(f'starting calibration of pin {servo_pin}')
            previous_curl = None
            if breaker:
                break
            for intensity in range(starting_stim,max_stim + 1):
                msc.set_intensity(intensity,str(servo_pin))
                time.sleep(0.04)
                curls = np.array(ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5))))
                if previous_curl is None:
                    previous_curl = curls
                curl_rates = abs(curls - np.array(previous_curl))
                if np.any(curls < 0.1): # This checks if any finger have uncurled
                    print(f'full extenstion detected: continuing, intesity={intensity}, reset hand to neutral position')
                    time.sleep(2)
                    msc.set_intensity(0,str(servo_pin))
                    break
                previous_curl = curls
                curl_stable = curl_rates <= max_curl_rate # MAYBE CHANGE THIS IN THE FUTURE
                if curl_stable.all():
                    return_df["IM"][servo_pin][intensity] = curls[0]
                    return_df["RP"][servo_pin][intensity] = curls[1]
                    return_df["T"][servo_pin][intensity] = curls[2]
                    return_df["IMCURL"][servo_pin][intensity] = curl_rates[0]
                    return_df["RPCURL"][servo_pin][intensity] = curl_rates[1]
                    return_df["TCURL"][servo_pin][intensity] = curl_rates[2]
                    
                else:
                    print('loss of tracking detected: restarting')
                    print(curl_rates)
                    time.sleep(1)
                    # state = 'WAITING'
                    # breaker = True
                    msc.set_intensity(0,str(servo_pin))
                    break
            check_hand_curl()
        print('calibration complete')
        state = 'finished'
        return return_df

def shock_mapper(val:int, low_shock:int= 20, max_shock:int= 45,val_offset:int=0) -> int:
    val += val_offset
    intensity = (val / 100) * (max_shock - low_shock) + low_shock
    '''returns servo insensity for a disered shock intensity val (0-100)'''
    return int(intensity)

def shock_calibration() -> tuple[int, int]:
    '''Returns low_shock and max_shock values for shock_mapper function'''
    # set_intensity(0) # Make sure tens is off before calibration
    input("TENS Calibration: Press Enter to start...")
    low_shock = None # First time the user can feel the shock (small tingle)
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
if __name__ == '__main__':
    # ---------------- INITIALIZE CLIENT AND FRAME LOOP ----------------
    get_client_and_thread()
    # frame_queue = Queue(maxsize=1)
    # thread = threading.Thread(target=frame_producer, daemon=True).start()
    # thread.start()
    # try:
    #     print(f'thread initiated, test frame {ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))}')
    # except:
    #     time.sleep(1)
    #     try:
    #         print(f'thread initiated, test frame {ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5)))}')
    #     except:
    #         print(f'failed to retrieve from que available data:{frame_queue.qsize()}')
    #         quit()
    # ---------------- MAIN LOOP ----------------
    TARGET_CURL = 0.55
    CURL_TOLERANCE = 0.10
    MAX_CURL_RATE = 0.02 # normalized curl/cycle,  highest allowed response speed for auto calibration response

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
    
    starting_stim,max_stim = shock_calibration()
    while calibrating:
        print('starting automatic calibration curl fingers to ~90 degrees and relax')
        # ---------------- CHECK HAND POSITION ----------------
        check_hand_curl(TARGET_CURL,CURL_TOLERANCE,MAX_CURL_RATE) # check for hand to be in the correct position
        # ---------------- APPLY STIM AND MONITER ----------------
        filled_df = create_mapping_df(servo_pins=servo_pin_list,
                          starting_stim=starting_stim,
                          max_stim=max_stim,
                          return_df=df,
                          max_curl_rate=0.04)


    # #FOR CALCULATION CREATE MULTIINDEX DF WITH PIN,INTENSITY AND COLUMNS OF IM PR T IMCURL RPCURL TCURL
    # # FOR EACH DESIRED TARGET FIND THE PIN THAT BEST ALLIGNS WITH IT
    # # CREATE THE LINEAR MAP BETWEEN PIN AND TARGET AS BEFORE