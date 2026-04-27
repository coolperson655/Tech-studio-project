from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import Hand_tracking as ht

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

client = ht.client