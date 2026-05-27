import time
import numpy as np
import math
import matplotlib.pyplot as plt
import Hand_tracking as ht
import Functions.Multi_servo_control as msc
import pandas as pd
import sys
import seaborn as sns 
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
    Compute pin x signal heatmap values.

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