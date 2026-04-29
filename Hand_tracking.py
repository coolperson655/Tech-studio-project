from hand_tracking_sdk import HTSClient, HTSClientConfig, JointName, StreamOutput
import time
import numpy as np
import math
import matplotlib.pyplot as plt

finger_map = {
    "index": [
        JointName.INDEX_PROXIMAL,
        JointName.INDEX_INTERMEDIATE,
        JointName.INDEX_DISTAL,
        JointName.INDEX_TIP,
    ],
    "middle": [
        JointName.MIDDLE_PROXIMAL,
        JointName.MIDDLE_INTERMEDIATE,
        JointName.MIDDLE_DISTAL,
        JointName.MIDDLE_TIP,
    ],
    "ring": [
        JointName.RING_PROXIMAL,
        JointName.RING_INTERMEDIATE,
        JointName.RING_DISTAL,
        JointName.RING_TIP,
    ],
    "pinky": [
        JointName.LITTLE_PROXIMAL,
        JointName.LITTLE_INTERMEDIATE,
        JointName.LITTLE_DISTAL,
        JointName.LITTLE_TIP,
    ],
        "thumb": [
        JointName.THUMB_METACARPAL,
        JointName.THUMB_PROXIMAL,
        JointName.THUMB_DISTAL,
        JointName.THUMB_TIP,
    ],
}

def joint_vec(frame, joint):
    x, y, z = frame.get_joint(joint)
    return np.array([x, y, z])

def finger_curls(frame, all=False):
    '''
    Returns avg(index + middle), avg(ring + pinky), thumb by default.
    set all to TRUE to get individual fingers 
    '''
    curl_list = []
    for finger in finger_map.keys():
        p0 = joint_vec(frame, finger_map[finger][0]) # INDEX_PROXIMAL)
        p1 = joint_vec(frame, finger_map[finger][1])# INDEX_INTERMEDIATE)
        p2 = joint_vec(frame, finger_map[finger][2])# INDEX_DISTAL)
        p3 = joint_vec(frame, finger_map[finger][3])# INDEX_TIP)

        v1 = p1 - p0
        v2 = p2 - p1
        v3 = p3 - p2

        a1 = angle_between(v1, v2)
        a2 = angle_between(v2, v3)

        curl_deg = a1 + a2
        curl_list.append(curl_deg)
    if all:
        return curl_list[0], curl_list[1], curl_list[2], curl_list[3], curl_list[4]
    else:
        return (curl_list[0] + curl_list[1])/2, (curl_list[2]+ curl_list[3])/2, curl_list[4]

def angle_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))

def get_feedback_values(curl_vals,feedback_func,normalized=True):
    """
       curl_vals: tuple of values representing the normalized curl values
       feedback_func:calucation to apply to curl values
       normalized: boolean indicating whether to normalize between 0 and 1
       returns tuple the same length as input of curl_vals
    """
    output = curl_vals
    for i in range(len(curl_vals)):
        output[i] = feedback_func(curl_vals[i])
        if normalized:
            output[i] = (output[i] - 0) / (1 - 0)
    return output


def feedback_fuction(val:float) ->float:
    return float(val * 0.3)
# def finger_curl(joint_positions):
#     """
#     joint_positions = [prox, inter, dist, tip]
#     returns curl in degrees
#     """

#     v1 = joint_positions[1] - joint_positions[0]
#     v2 = joint_positions[2] - joint_positions[1]
#     v3 = joint_positions[3] - joint_positions[2]

#     a1 = angle_between(v1, v2)
#     a2 = angle_between(v2, v3)

#     return a1 + a2

def normalize_curl(degrees, min_deg=0, max_deg=160):
    return_list = []
    for degree in degrees:
        output = np.clip((degree - min_deg) / (max_deg - min_deg), 0.0, 1.0)
        return_list.append(output)
    return return_list

def draw_bar(value, width=30):
    filled = int(value * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"



client = HTSClient(
    HTSClientConfig(
        output=StreamOutput.FRAMES,
        host="0.0.0.0",
        port=9000,
    )
)

# plt.ion()
# fig, ax = plt.subplots()

# bars = ax.bar(["Index"], [0.0])
# ax.set_ylim(0, 1)
# ax.set_ylabel("Curl")

# print(get_feedback_values([0.3,0.4,0.5],feedback_fuction))

print('starting')
for frame in client.iter_events():
    curls = normalize_curl(finger_curls(frame))
    for curl in curls:
        print('index', draw_bar(curl))

    # print('index', draw_bar(index_curl), 'middle', draw_bar(middle_curl), 'ring', draw_bar(ring_curl),'pinky', draw_bar(pinky_curl))
    
    
    
    
    
    # time.sleep(0.25)
    # x, y, z = frame.get_joint(JointName.INDEX_TIP)
    # print(f"index tip xyz=({x:.5f}, {y:.5f}, {z:.5f})")
    # index_joints = frame.get_finger("index")
    # print(index_joints[JointName.INDEX_PROXIMAL])
    # time.sleep(3)
    # print('==================')