from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput, parse_line
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import Hand_tracking as ht
import Functions.Multi_servo_control as msc
from queue import Queue
import threading
import pandas as pd
import sys
import seaborn as sns 




def get_client_and_thread():
    def frame_producer():
        for frame in client.iter_events():
            # if stop_event.is_set():   # 👈 check if we should stop
            #     break

            try:
                frame_queue.put(frame, block=False)
            except:
                frame_queue.get_nowait()
                frame_queue.put(frame)
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
        
def get_curls():
    try:
        return np.array(ht.normalize_curl(ht.finger_curls(frame_queue.get(timeout=0.5))))
    except Exception as exc:
        raise RuntimeError(f'failed to retrieve from queue, available data:{frame_queue.qsize()}') from exc
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