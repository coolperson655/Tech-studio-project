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
import seaborn as sns 
import cv2
import mediapipe as mp
import numpy as np
import time
import Hand_tracking as ht
import mix_tracking as mt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame

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

def get_client_and_thread():
    global frame_queue
    try:
        frame_queue = Queue(maxsize=1)
        threading.Thread(target=frame_producer, daemon=True).start()
    except Exception as exc:
        raise RuntimeError("Failed to initilize thread") from exc
    try:
        print(f'thread initiated, test frame {ht.normalize_curl(mt.compute_finger_curl(frame_queue.get(timeout=0.5)[0], frame_queue.get(timeout=0.5)[1], frame_queue.get(timeout=0.5)[2]),10,160)}')
    except:
        time.sleep(5)
        try:
            print(f'thread initiated, test frame {ht.normalize_curl(mt.compute_finger_curl(frame_queue.get(timeout=0.5)[0], frame_queue.get(timeout=0.5)[1], frame_queue.get(timeout=0.5)[2]),10,160)}')
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

class CalibrationUI:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 28)
        self.message = ""
    
    def update(self, msg):
        self.message = msg
        self.draw()

    def draw(self):
        self.screen.fill((30, 30, 30))
        text_surface = self.font.render(self.message, True, (200, 200, 200))
        self.screen.blit(text_surface, (50, 100))
        pygame.display.flip()

def notify(msg):
    global ui_update
    if ui_update:
        ui_update(msg)
    else:
        print(msg)

def run_calibration(screen):
    ui = CalibrationUI(screen)

    def ui_callback(msg):
        ui.update(msg)

        # Keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    df = create_mapping_df(
        servo_pins=[1,2,3],
        starting_stim=0,
        max_stim=100,
        return_df=your_df,
        ui_update=ui_callback
    )

    return df

def create_mapping_df(servo_pins:list,
                      starting_stim:int,
                      max_stim:int,
                      return_df:pd.DataFrame,
                      max_curl_rate:float=0.04,
                      ui_update=None):
 """Adds the mapping values into the given dataframe, calls hand curl between servos to ensure hand is positioned correctly"""
 global frame_queue
 breaker = False
 internal_state = 'mapping'
 while internal_state == "mapping":
        for servo_pin in servo_pins:
            notify(f'starting calibration of pin {servo_pin}')
            if breaker:
                break
            complete = False
            while complete == False:
                previous_curl = None
                failed = False
                for intensity in range(starting_stim,max_stim + 1):
                    msc.set_intensity(intensity,str(servo_pin))
                    time.sleep(0.04)
                    curls = ht.normalize_curl(mt.compute_finger_curl(frame_queue.get(timeout=0.5)[0], frame_queue.get(timeout=0.5)[1], frame_queue.get(timeout=0.5)[2]),10,160)
                    if previous_curl is None:
                        previous_curl = curls
                    curl_rates = abs(curls - np.array(previous_curl))
                    if np.any(curls < 0.1): # This checks if any finger have uncurled
                        notify(f'full extenstion detected: continuing, intesity={intensity}, reset hand to neutral position')
                        time.sleep(2)
                        msc.set_intensity(0,str(servo_pin))
                        complete = True
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
                        notify('loss of tracking detected: restarting')
                        notify(curl_rates)
                        time.sleep(0.5)
                        # state = 'WAITING'
                        # breaker = True
                        msc.set_intensity(0,str(servo_pin))
                        failed = True
                        break
                if not failed:
                    notify('pin calibration complete')
                    msc.set_intensity(0,str(servo_pin))
                    time.sleep(1)
                    check_hand_curl()
                    complete = True
        notify('calibration complete')
        state = 'finished'
        return return_df


def build_curl_controller(df, target="IMCURL", others=("RPCURL", "TCURL"), thresh=0.05):

    reactive = df.dropna(subset=[target])

    score = (
        reactive
        .assign(
            selectivity=lambda x:
                x[target].abs()
                - x[list(others)].abs().sum(axis=1)
        )
        .groupby(level="servo pin")["selectivity"]
        .mean()
    )
    # import pdb;pdb.set_trace()
    score = pd.to_numeric(score, errors='coerce').fillna(0, downcast='infer')
    best_pin = score.idxmax()

    pin_df = df.xs(best_pin, level="servo pin")
    pin_df = pin_df[pin_df[target].abs() > thresh]

    stim = pin_df.index.to_numpy(dtype=float)
    curl = pin_df[target].to_numpy()

    stim_norm = (stim - stim.min()) / (stim.max() - stim.min()) * 100

    mapper = lambda u: np.interp(np.clip(u, 0, 100), curl.tolist(), stim_norm.tolist())

    return best_pin, mapper


def compute_pin_heatmap(
    df,
    columns=None,
    threshold=0.00,
    agg="mean",
):
    """
    Compute pin × signal heatmap values.

    df: MultiIndex DataFrame (servo pin, stim_level)
    columns: list of columns to score (default = all curl columns)
    threshold: dead-zone cutoff
    agg: 'mean' | 'max' | 'sum'
    """

    if columns is None:
        columns = [c for c in df.columns]# if "CURL" in c]

    # Remove dead zones
    active = df[columns].where(df[columns].abs() > threshold)

    # Aggregate over stim_level
    grouped = active.groupby(level="servo pin")

    if agg == "mean":
        heatmap = grouped.mean()
    elif agg == "max":
        heatmap = grouped.max()
    elif agg == "sum":
        heatmap = grouped.sum()
    else:
        raise ValueError("agg must be 'mean', 'max', or 'sum'")

    return heatmap


def plot_pin_heatmap(
    heatmap_df,
    title="Pin vs Output Heatmap",
    cmap="viridis",
    annotate=True,
):
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        annot=annotate,
        fmt=".3f",
        linewidths=0.5,
    )
    sns.heatmap(heatmap_df, cmap ='RdYlGn', linewidths = 0.30, annot = True)
    plt.title(title)
    plt.ylabel("Servo Pin")
    plt.xlabel("Output")
    plt.tight_layout()
    plt.show()

def plot_mappings(df_raw,controller_df,curls):
    x = np.arange(0,1,0.01)
    fig, axes = plt.subplots(len(curls),2,figsize=(12,2*len(curls)))
    fig_curls = []
    for curl in curls:
        fig_curls.append(curl)
        fig_curls.append(curl)
    for curl,counter in zip(fig_curls,range(1,(len(fig_curls))+1)):
        if counter%2==1:
            y = np.empty(0)
            for i in x:
                y = np.append(y,controller_df['mapper'][curl](i))
            ax = axes[(counter//2)-1,0]
            ax.plot(x,y)
            ax.set_title(f'{str(curl)} linearized')
            ax.set_xlabel('input hand curl')
            ax.set_ylabel('output motor intensity')
        else:
            plot_stims = df_raw[curl][controller_df['best_pin'][curl]].index.to_numpy(dtype=float)
            plot_curls = df_raw[curl][controller_df['best_pin'][curl]].to_numpy(dtype=float)
            # plot not linearized response
            ax = axes[(counter//2)-1,1]
            ax.plot(plot_stims,plot_curls)
            ax.set_title(f'{str(curl)} NON-linearized')
            ax.set_xlabel('input hand curl')
            ax.set_ylabel('output motor intensity')
    fig.tight_layout()
    plt.show()

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
HAND_MODEL = "models/hand_landmarker.task"
POSE_MODEL = "models/pose_landmarker_full.task"

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    #running_mode=RunningMode.LIVE_STR,
    #result_callback=hand_callback,
    num_hands=1
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    #running_mode=RunningMode.LIVE_STREAM,
    #result_callback=pose_callback
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
def frame_producer():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ---- Run detectors ----
        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        h, w, _ = frame.shape

        # ---- Draw Hand Landmarks ----
        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                h_landmarks = hand_landmarks
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    # cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            payload = [h_landmarks,w,h]
            try:
                frame_queue.put(payload, block=False)
            except:
                frame_queue.get_nowait()
                frame_queue.put(payload)
            curl_vals = ht.normalize_curl(mt.compute_finger_curl(h_landmarks, w, h),10,160)
            curl_vals = list(map(mt.rounder,curl_vals))
            # cv2.putText(
            #     frame,
            #     f"Curls: {curl_vals}",
            #     (10, 80),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1.0,
            #     (0, 255, 0),
            #     2
            # )



        # ---- Draw Pose (shoulder + elbow + wrist) ----
        if pose_result.pose_landmarks:
            for person in pose_result.pose_landmarks:

            # ---- Extract joints ----
                left_shoulder = mt.to_vec(person[11], w, h)
                left_elbow    = mt.to_vec(person[13], w, h)
                left_wrist    = mt.to_vec(person[15], w, h)

                right_shoulder = mt.to_vec(person[12], w, h)
                right_elbow    = mt.to_vec(person[14], w, h)
                right_wrist    = mt.to_vec(person[16], w, h)

                left_elbow_angle = mt.joint_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = mt.joint_angle(right_shoulder, right_elbow, right_wrist)


                # simpler version: horizontal reference
                def horizontal_ref(p):
                    return p + np.array([1, 0, 0])

                # left_shoulder_angle = joint_angle(horizontal_ref(left_shoulder), left_shoulder, left_elbow)
                # right_shoulder_angle = joint_angle(horizontal_ref(right_shoulder), right_shoulder, right_elbow)

                # ---- Draw joints ----
                important_points = [11, 12, 13, 14, 15, 16]

                for idx in important_points:
                    lm = person[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

                # ---- Draw connections ----
                def draw_line(a, b):
                    x1, y1 = int(person[a].x * w), int(person[a].y * h)
                    x2, y2 = int(person[b].x * w), int(person[b].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                draw_line(11, 13)
                draw_line(13, 15)
                draw_line(12, 14)
                draw_line(14, 16)

                # ---- Display angles ----
                cv2.putText(
                    frame,
                    f"L-Elbow: {left_elbow_angle:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"R-Elbow: {right_elbow_angle:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 0),
                    2
                )
        cv2.imshow("Hands + Pose (Tasks API)",frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

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
    CURL_TOLERANCE = 0.15
    MAX_CURL_RATE = 0.02 # normalized curl/cycle,  highest allowed response speed for auto calibration response

    calibrating = True
    previous_curl = None

    max_stim = 50
    starting_stim = 20
    stim_levels = range(starting_stim,max_stim+1)

    servo_pin_list = [2,4,6,8,10,12] # pins to calibrate back curl with
    curls = ["IMCURL", "RPCURL", "TCURL","IM", "RP", "T"]
    mapping_curls = ["IM", "RP", "T"]
    multindex = pd.MultiIndex.from_product([servo_pin_list, stim_levels ], names=["servo pin", "stim_level"])
    col_names = ['IM','RP','T','IMCURL','RPCURL','TCURL']
    df = pd.DataFrame(index=multindex, columns=col_names)
    # df['IM'][2][35] indexing example

    starting_stim,max_stim = [10,80]#shock_calibration()
    while calibrating:
        print('starting automatic calibration curl fingers to ~90 degrees and relax')
        # ---------------- CHECK HAND POSITION ----------------
        check_hand_curl(TARGET_CURL,CURL_TOLERANCE,MAX_CURL_RATE) # check for hand to be in the correct position
        # ---------------- APPLY STIM AND MONITER ----------------
        filled_df = create_mapping_df(servo_pins=servo_pin_list,
                          starting_stim=starting_stim,
                          max_stim=max_stim,
                          return_df=dsf,
                          max_curl_rate=0.1)
        
        controller_df = pd.DataFrame(index=curls,columns=['best_pin','mapper'])
        hmap = compute_pin_heatmap(filled_df, agg='sum')
        hmap = hmap.to_numpy(dtype=float)
        plt.imshow(hmap)
        plt.show()
        # import pdb;pdb.set_trace()
        # print(filled_df.head())
        # for current_curl in mapping_curls:
        #     controller_df['best_pin'][current_curl], controller_df['mapper'][current_curl] = build_curl_controller(filled_df, others=[other_curl for other_curl in mapping_curls if other_curl != current_curl], target=current_curl)
        # new_raw = filled_df.drop(columns=['IMCURL','RPCURL','TCURL'])
        # plot_mappings(df_raw=new_raw,controller_df=controller_df,curls=mapping_curls)
        # calibrating = False

    # #FOR CALCULATION CREATE MULTIINDEX DF WITH PIN,INTENSITY AND COLUMNS OF IM PR T IMCURL RPCURL TCURL
    # # FOR EACH DESIRED TARGET FIND THE PIN THAT BEST ALLIGNS WITH IT
    # # CREATE THE LINEAR MAP BETWEEN PIN AND TARGET AS BEFORE