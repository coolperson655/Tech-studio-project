import cv2
import mediapipe as mp
import numpy as np
import time
import Hand_tracking as ht
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from queue import Queue

cv2.setNumThreads(1)
# ---------------------------
# Utility math functions
# ---------------------------

class EMAFilter:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value

        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value

        return self.value

def to_vec(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z])

def angle_between(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return 0.0  # avoid NaNs

    cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def joint_angle(p0, p1, p2):
    v1 = p0 - p1
    v2 = p2 - p1  
    return angle_between(v1, v2)


# ---------------------------
# Finger mapping (MediaPipe)
# ---------------------------

finger_map = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
    "thumb": [1, 2, 3, 4]
}

def compute_finger_curl(hand_landmarks, w, h, min_deg=10, max_deg=160):
    """
    Computes curl for each finger and normalizes it to [0, 1] range based on min and max curl angles.
    Returns a tuple: (index_middle_avg, ring_pinky_avg, thumb)
    """
    curls = []

    for finger in finger_map.keys():
        i0, i1, i2, i3 = finger_map[finger]

        p0 = to_vec(hand_landmarks[i0], w, h)
        p1 = to_vec(hand_landmarks[i1], w, h)
        p2 = to_vec(hand_landmarks[i2], w, h)
        p3 = to_vec(hand_landmarks[i3], w, h)

        v1 = p1 - p0
        v2 = p2 - p1
        v3 = p3 - p2

        a1 = angle_between(v1, v2)
        a2 = angle_between(v2, v3)

        curl_deg = a1 + a2
        curl_deg = np.clip((curl_deg - min_deg) / (max_deg - min_deg), 0.0, 1.0)
        curls.append(curl_deg)

    # Same grouping logic you used
    return (
        (curls[0] + curls[1]) / 2,  # index + middle
        (curls[2] + curls[3]) / 2,  # ring  + pinky
        curls[4],                   # thumb
    )

# ---------------------------
# Initialize models
# ---------------------------

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
def rounder(val):
    return round(val,2)

left_elbow_filter = EMAFilter(alpha=0.3)
right_elbow_filter = EMAFilter(alpha=0.3)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
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
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            curl_vals = ht.normalize_curl(compute_finger_curl(h_landmarks, w, h),10,160)
            curl_vals = list(map(rounder,curl_vals))
            cv2.putText(
                frame,
                f"Curls: {curl_vals}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )



        # ---- Draw Pose (shoulder + elbow + wrist) ----
        if pose_result.pose_landmarks:
            for person in pose_result.pose_landmarks:

            # ---- Extract joints ----
                left_shoulder = to_vec(person[11], w, h)
                left_elbow    = to_vec(person[13], w, h)
                left_wrist    = to_vec(person[15], w, h)

                right_shoulder = to_vec(person[12], w, h)
                right_elbow    = to_vec(person[14], w, h)
                right_wrist    = to_vec(person[16], w, h)

                left_elbow_angle = joint_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = joint_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = left_elbow_filter.update(left_elbow_angle)
                right_elbow_angle = right_elbow_filter.update(right_elbow_angle)
                # (optional) shoulder angles (torso → shoulder → elbow)
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
