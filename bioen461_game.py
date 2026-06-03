import time
from queue import Empty

import pygame
import sys
import numpy as np
import math
from Functions.Calculation_functions import check_hand_curl
# from Functions.Streaming_functions import get_client_and_thread, get_curls
import Functions.Multi_servo_control as msc
import pandas as pd
import Calibration_with_cam as calib
import matplotlib.pyplot as plt

from mix_tracking import EMAFilter

# =====================================================================
# SYSTEM CONFIGURATION & WINDOW SETUP
# =====================================================================

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 950, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Integrated VR Arm-Sleeve Engineering Suite")
clock = pygame.time.Clock()

# Color Palette Definitions
SPACE_GRAY = (24, 26, 30)
PANEL_DARK = (36, 40, 48)
BG_INNER = (16, 18, 22)
TEXT_WHITE = (245, 245, 245)
TEXT_BLACK = (15, 15, 15)
TEXT_MUTED = (140, 145, 155)
GLOVE_CYAN = (0, 220, 255)
VOLTAGE_RED = (255, 50, 85)
INDICATOR_GREEN = (40, 220, 110)
CALIBRATION_RED = (252, 65, 3)
CALIBRATION_YELLOW = (252, 219, 3)

# Fonts
font_title = pygame.font.SysFont("Arial", 22, bold=True)
font_subtitle = pygame.font.SysFont("Arial", 18, bold=True)
font_body = pygame.font.SysFont("Arial", 15)
font_small = pygame.font.SysFont("Arial", 12)

# =====================================================================
# DATA MATRICES & PROFILES
# =====================================================================
BALL_PROFILES = {
    "STEEL":  {"color": (160, 165, 170), "stiffness": 750.0, "radius": 0.040, "desc": "Solid rigid steel sphere."},
    "TENNIS": {"color": (210, 255, 30),  "stiffness": 140.0, "radius": 0.033, "desc": "Compliant, elastic core felt."},
    "SPONGE": {"color": (255, 195, 45),  "stiffness": 20.0,  "radius": 0.050, "desc": "Porous, low resistance foam."}
}
ball_list = list(BALL_PROFILES.keys())

# =====================================================================
# Calibration variables
# =====================================================================
TARGET_CURL = 0.55
CURL_TOLERANCE = 0.15
MAX_CURL_RATE = 0.6 # normalized curl/cycle,  highest allowed response speed for auto calibration response

calibrating = True
previous_curl = [0.0, 0.0, 0.0]

max_stim = 180
starting_stim = 50
stim_levels = range(starting_stim,max_stim+1)

servo_pin_list = [2,4,6]#,8,10,12] # pins to calibrate back curl with
curls = ["IMCURL", "RPCURL", "TCURL","IM", "RP", "T"]
mapping_curls = ["IM", "RP", "T"]
multindex = pd.MultiIndex.from_product([servo_pin_list, stim_levels ], names=["servo pin", "stim_level"])
col_names = ['IM','RP','T','IMCURL','RPCURL','TCURL']
df = pd.DataFrame(index=multindex, columns=col_names)
controller_df = pd.DataFrame(index=curls,columns=['best_pin','mapper'])
check_color_dict = {'IM_COLOR':['Waiting...',CALIBRATION_RED], 'RP_COLOR': ['Waiting...',CALIBRATION_RED], 'T_COLOR': ['Waiting...',CALIBRATION_RED]}

calibration_state = "idle"
calibration_pin_index = 0
calibration_intensity = starting_stim
calibration_failed = False
calibration_message = "Ready to begin calibration sequence."
calibration_prev_curl = None
servo_pins = servo_pin_list.copy()

# =====================================================================
# SYSTEM VARIABLES & LOGIC STATE STATES
# =====================================================================
APP_STATE = "MENU" # MENU, BALL_GAME, ARM_GAME

# --- Ball Game State ---
current_ball_idx = 0
# Tuple configuration representing 3 fingers: (Thumb, Index, Middle)
hand_curl = [0.0, 0.0, 0.0] 
active_finger_idx = 1 # Default index finger selection for manual calibration / debugging

# --- Calibration Controls ---
# Maps uncalibrated tracker raw output scale metrics to clean product thresholds
calibrated_min = 0.0
calibrated_max = 1.0
calibration_mode = False

# --- Arm Game State ---
chosen_weight_lbs = 15.0 # Choosable slider limits: 1 to 30 lbs
arm_velocity = 0.0
arm_acceleration = 0.0
arm_moving_up = True
time_accumulator = 0.0 # Operates kinematic sine motion loops
previous_arm_angle = None
previous_arm_velocity = 0.0
frame_time = 0.0  # Tracks timing for velocity/acceleration calculations
smoothed_bicep_intensity = 0
smoothed_tricep_intensity = 0
smoothed_shock = 0.0
tracking_status = "waiting"
calibrated = False
# left_elbow_filter = EMAFilter(alpha=0.3)
# right_elbow_filter = EMAFilter(alpha=0.3)

# =====================================================================
# CORE ALGEBRAIC MATH KERNELS (Combined Architecture)
# =====================================================================
class ArmKinematicsEstimator:
    def __init__(self, gender_neutral_height_m=1.75):
        self.height = gender_neutral_height_m
        self.head_to_shoulder_drop = self.height * 0.12
        self.shoulder_width_half = self.height * 0.13
        self.upper_arm_length = self.height * 0.186
        self.forearm_length = self.height * 0.146
        
    def estimate_arm_chain(self, head_pos, hand_pos):
        head = np.array(head_pos, dtype=float)
        hand = np.array(hand_pos, dtype=float)
        shoulder_pos = np.array([head[0] + self.shoulder_width_half, head[1] - self.head_to_shoulder_drop, head[2] - 0.05])
        
        shoulder_to_hand = hand - shoulder_pos
        total_arm_reach = np.linalg.norm(shoulder_to_hand)
        max_reach = self.upper_arm_length + self.forearm_length
        
        if total_arm_reach >= max_reach:
            direction = shoulder_to_hand / total_arm_reach
            elbow_pos = shoulder_pos + (direction * self.upper_arm_length)
            return shoulder_pos, elbow_pos

        a, b, c = self.forearm_length, self.upper_arm_length, total_arm_reach
        cos_angle = np.clip((b**2 + c**2 - a**2) / (2.0 * b * c), -1.0, 1.0)
        shoulder_angle = np.arccos(cos_angle)
        
        down_vector = np.array([0, -1, 0])
        forward_axis = np.cross(shoulder_to_hand, down_vector)
        axis_len = np.linalg.norm(forward_axis)
        
        if axis_len < 0.001:
            forward_axis = np.array([1, 0, 0])
        else:
            forward_axis = forward_axis / axis_len
            
        elbow_direction_base = np.cross(forward_axis, shoulder_to_hand)
        elbow_direction_base = elbow_direction_base / np.linalg.norm(elbow_direction_base)
        arm_direction = shoulder_to_hand / total_arm_reach
        elbow_vector = (elbow_direction_base * np.sin(shoulder_angle)) + (arm_direction * np.cos(shoulder_angle))
        
        elbow_pos = shoulder_pos + (elbow_vector * self.upper_arm_length)
        return shoulder_pos, elbow_pos

estimator = ArmKinematicsEstimator()

def calculate_haptic_feedback(mode, **kwargs):
    """
    Computes system feedback array. All output stimulation values are normalized 0 to 1.
    """
    MAX_MA_SCALE = 100.0 # Reference internal hardware saturation scaling boundary
    
    if mode == "BALL":
        curl_tuple = kwargs.get("curl_tuple", (0.0, 0.0, 0.0))
        profile = kwargs.get("profile")
        stiffness = profile["stiffness"]
        radius = profile["radius"]
        
        # Calculate individual outputs per micro-node (Finger tracking channel matching)
        channel_stimulations = []
        total_deformation = 0.0
        
        for finger_curl in curl_tuple:
            current_hand_radius = radius * (1.0 - (finger_curl * 0.5))
            penetration_depth = max(0.0, radius - current_hand_radius)
            total_deformation += penetration_depth
            
            raw_force = stiffness * penetration_depth
            stim_level = min(1.0, (raw_force * 15.0) / MAX_MA_SCALE)
            channel_stimulations.append(stim_level if stim_level > 0.05 else 0.0)
            
        return channel_stimulations, total_deformation / 3.0
        
    elif mode == "ARM":
        weight_lbs = kwargs.get("weight_lbs", 15.0)
        velocity = kwargs.get("velocity", 0.0)
        acceleration = kwargs.get("acceleration", 0.0)
        
        # Convert weight tracking metric to kilograms: 1 lb = 0.453592 kg
        mass_kg = weight_lbs * 0.453592
        
        gravitational_force = mass_kg * 9.81
        inertial_force = mass_kg * acceleration
        total_force = max(0.0, gravitational_force + inertial_force)
        
        base_stim = (total_force * 8.0) / MAX_MA_SCALE
        bicep = 0.0
        tricep = 0.0
        
        if total_force > 0:
            if velocity >= 0:
                bicep = min(1.0, base_stim)
                tricep = min(1.0, base_stim * 0.15)
            else:
                tricep = min(1.0, base_stim)
                bicep = min(1.0, base_stim * 0.10)
                
        return bicep if bicep > 0.05 else 0.0, tricep if tricep > 0.05 else 0.0

def arm_angle2motor(degrees, min_deg=30, max_deg=150):
    return_list = []
    for degree in degrees:
        output = 2 * ((degree - min_deg))/ (max_deg - min_deg) -1
        return_list.append(output)
    return return_list
# =====================================================================
# GRAPHICS INTERFACE RENDER LOOPS
# =====================================================================
def draw_ui_button(surf, rect, text, color, text_color=TEXT_WHITE):
    pygame.draw.rect(surf, color, rect, border_radius=6)
    txt = font_body.render(text, True, text_color)
    txt_rect = txt.get_rect(center=(rect[0] + rect[2]/2, rect[1] + rect[3]/2))
    surf.blit(txt, txt_rect)
    return pygame.Rect(rect)
# streamf.get_client_and_thread()
# hand_curl = {0:None,1:None,2:None}
# Main Application Lifecycle Execution Loop
frame_queue = calib.get_client_and_thread()

while True:
    screen.fill(SPACE_GRAY)
    mx, my = pygame.mouse.get_pos()
    click = False
    
    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                click = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                APP_STATE = "MENU"

    # -----------------------------------------------------------------
    # SCENE ARCHITECTURE: MAIN MENU CONTROLS INTERFACE
    # -----------------------------------------------------------------
    if APP_STATE == "MENU":
        # Interface Header Branding
        title = font_title.render("ARM-SLEEVE WEARABLE CONTROLLER SYSTEM DEVELOPMENT CORE", True, GLOVE_CYAN)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
        
        subtitle = font_body.render("Neuromuscular Electrical Pulse Targeting Simulation Platform", True, TEXT_MUTED)
        screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 115))
        
        # Grid Navigation Panel Arrays
        ball_btn = draw_ui_button(screen, (50, 220, 260, 160), "1. DEFORMABLE BALL LAB", PANEL_DARK)
        arm_btn = draw_ui_button(screen, (350, 220, 260, 160), "2. KINETIC ARM WORKSPACE", PANEL_DARK)
        cali_btn = draw_ui_button(screen, (650, 220, 260, 160), "3. CALIBRATION INTERFACE", PANEL_DARK)
        calc_btn = draw_ui_button(screen, (50, 320, 160, 160), "4. CALCULATION INTERFACE", PANEL_DARK)
        
        # Interactive Selection Evaluation

        if calc_btn.collidepoint((mx, my)):
            draw_ui_button(screen, (50, 320, 160, 160), "4. CALCULATION INTERFACE", GLOVE_CYAN, SPACE_GRAY)
            if click: APP_STATE = "CALCULATING"

        if ball_btn.collidepoint((mx, my)):
            draw_ui_button(screen, (50, 220, 260, 160), "1. DEFORMABLE BALL LAB", GLOVE_CYAN, SPACE_GRAY)
            if click: APP_STATE = "BALL_GAME"
            
        if arm_btn.collidepoint((mx, my)):
            draw_ui_button(screen, (350, 220, 260, 160), "2. KINETIC ARM WORKSPACE", GLOVE_CYAN, SPACE_GRAY)
            if click: APP_STATE = "ARM_GAME"

        if cali_btn.collidepoint((mx, my)):
            draw_ui_button(screen, (650, 220, 260, 160), "3. CALIBRATION INTERFACE", GLOVE_CYAN, SPACE_GRAY)
            if click: 
                APP_STATE = "CHECKING"
                calibration_pin_index = 0 # set this once so it doesnt change on every loop during calibration
            
        # Global Operational Controls HUD Panel
        pygame.draw.rect(screen, PANEL_DARK, (175, 430, 600, 130), border_radius=8)
        lbl_control = font_subtitle.render("CORE DEVELOPMENT UTILITIES & INTERFACE KEYBOARD MAP", True, TEXT_WHITE)
        screen.blit(lbl_control, (195, 450))
        
        ctrl_1 = font_small.render("• [ESC]: Escape back to central Main Choice Menu from within active workspaces.", True, TEXT_MUTED)
        ctrl_2 = font_small.render("• [SPACE]: Cycle through physical material compound matrix profiles (Steel, Tennis, Sponge).", True, TEXT_MUTED)
        ctrl_3 = font_small.render("• [UP/DOWN ARROWS]: Drive tracking data inputs (Flex finger cluster / lift kinetic weight load arm array).", True, TEXT_MUTED)
        screen.blit(ctrl_1, (195, 485))
        screen.blit(ctrl_2, (195, 505))
        screen.blit(ctrl_3, (195, 525))

    # -----------------------------------------------------------------
    # SCENE ARCHITECTURE: BALL SQUEEZE ENGINE & CALIBRATION INTERFACE
    # -----------------------------------------------------------------
    elif APP_STATE == "BALL_GAME":
        if calibrated is False:
            screen.fill(SPACE_GRAY)
            calibration_text = font_title.render("CALIBRATION REQUIRED: Please perform the calibration sequence before accessing the ball squeeze lab.", True, CALIBRATION_RED)
            screen.blit(calibration_text, (WIDTH//2 - calibration_text.get_width()//2, HEIGHT//2 - calibration_text.get_height()//2))
            pygame.display.flip()
            time.sleep(2)
            APP_STATE = "CHECKING"
            continue
        # Extract live hand tracking curl data directly from the streaming data queue
        try:
            # Non-blocking fetch to keep Pygame's render loop fluid
            hand_curl = list(frame_queue.get_nowait().curl)
            hand_curl = [0 if x is None else x for x in hand_curl]
            # Guarantee data structure consistency for the 3 tracking nodes
            if len(hand_curl) < 3:
                hand_curl += [0.0] * (3 - len(hand_curl))
        except Exception:
            # Maintain last known data coordinate position if stream is temporarily dry
            pass
            
        # Core Math Evaluation Run
        active_ball = ball_list[current_ball_idx]
        profile = BALL_PROFILES[active_ball]
        stims, avg_deform = calculate_haptic_feedback("BALL", curl_tuple=tuple(hand_curl), profile=profile)
        
        # Interface Shell Generation
        pygame.draw.rect(screen, PANEL_DARK, (40, 30, 440, 580), border_radius=10)
        pygame.draw.rect(screen, BG_INNER, (510, 30, 400, 580), border_radius=10)
        
        # Workspace Text Labels Headers
        screen.blit(font_title.render("FINGER MATRIX LAB ENVIRONMENT", True, GLOVE_CYAN), (60, 55))
        screen.blit(font_body.render(f"Active Mesh Structure: {active_ball}", True, TEXT_WHITE), (60, 95))
        screen.blit(font_small.render(profile["desc"], True, TEXT_MUTED), (60, 120))
        
        # Active Interactive Buttons
        mat_btn = draw_ui_button(screen, (60, 150, 160, 30), "Swap Compound", GLOVE_CYAN, SPACE_GRAY)
        if mat_btn.collidepoint((mx, my)) and click:
            current_ball_idx = (current_ball_idx + 1) % len(ball_list)
            hand_curl = [0.0, 0.0, 0.0]
            
        # Render Multi-Finger Calibration Suite System Window Block
        pygame.draw.rect(screen, BG_INNER, (60, 200, 400, 195), border_radius=6)
        screen.blit(font_subtitle.render("INTEGRATED SLEEVE SENSOR CALIBRATION MATRIX", True, TEXT_WHITE), (75, 215))
        
        # Pull down fingers list
        fingers_list = ["1. THUMB", "2. INDEX", "3. MIDDLE"]
        for i, f_name in enumerate(fingers_list):
            y_offset = 255 + (i * 45)
            text_color = GLOVE_CYAN if i == active_finger_idx else TEXT_WHITE
            screen.blit(font_body.render(f_name, True, text_color), (75, y_offset))
            
            # Draw tracking bar background track matching real telemetry
            pygame.draw.rect(screen, PANEL_DARK, (165, y_offset + 2, 210, 14), border_radius=4)
            pygame.draw.rect(screen, GLOVE_CYAN, (165, y_offset + 2, int(210 * np.clip(hand_curl[i], 0.0, 1.0)), 14), border_radius=4)
            
            f_rect = pygame.Rect(75, y_offset, 300, 20)
            if f_rect.collidepoint((mx, my)) and click:
                active_finger_idx = i

        if calibration_mode:
            screen.blit(font_small.render("CALIBRATION MODE LOCK ACTIVE: Setting tracking clip points.", True, INDICATOR_GREEN), (75, 365))
            
        # Multi-Channel Haptic Pulse Current Output Bar Arrays
        screen.blit(font_subtitle.render("MULTI-CHANNEL PULSE FEEDBACK OUTPUT", True, TEXT_WHITE), (60, 420))
        channels = ["CH1 (Forearm Flexor Thumb):", "CH2 (Forearm Flexor Index/Middle):", "CH3 (Forearm Flexor Ring/Pinky):"]
        for i, ch_lbl in enumerate(channels):
            servo_pin = (i+1)*2
            msc.set_intensity(int(stims[i]*100), servo_pin)
            y_offset = 450 + (i * 50)
            screen.blit(font_small.render(f"{ch_lbl} {stims[i]:.2f} / 1.00 Intensity", True, TEXT_WHITE), (60, y_offset))
            pygame.draw.rect(screen, (50, 55, 65), (60, y_offset + 18, 400, 12), border_radius=3)
            pygame.draw.rect(screen, VOLTAGE_RED, (60, y_offset + 18, int(400 * stims[i]), 12), border_radius=3)

        # Right Side Visualizer Rendering Canvas Panel Window
        bx, by = 710, 320
        br = 90  # Base radius of the un-deformed ball
        
        # --- DYNAMIC SQUEEZE MULTIPLIER CORE ---
        # As avg_deform increases, the ball flattens vertically and expands horizontally (conservation of volume)
        # We scale this effect slightly based on material stiffness so softer objects deform more dramatically
        deformation_factor = avg_deform * (2.0 - (profile["stiffness"] / 100.0))
        
        # Calculate squashed dimensions (clamp to ensure the ball never scales below zero or flips)
        rx = int(br * (1.0 + max(0.0, deformation_factor * 0.45))) # Widens out to the sides
        ry = int(br * max(0.1, 1.0 - (deformation_factor * 0.65))) # Flattens down from the top
        
        # Render the dynamically squeezing material compound shape
        pygame.draw.ellipse(screen, profile["color"], (bx - rx, by - ry, rx * 2, ry * 2))
        
        # Virtual Mesh Skeleton Hand Boundary Track Overlays (Clings to the squashed shape)
        for i, curl in enumerate(hand_curl):
            if curl > 0:
                # Dynamically match the track rings to the current compressed boundary radius
                arc_rx = rx + 10 + (i * 8)
                arc_ry = ry + 10 + (i * 8)
                pygame.draw.arc(screen, GLOVE_CYAN, (bx - arc_rx, by - arc_ry, arc_rx * 2, arc_ry * 2), 0.2, 2.9, 2)
                
        screen.blit(font_small.render("SURFACE GRASP MESH DEFORMATION CORE", True, TEXT_MUTED), (bx - 110, by + 180))

    # -----------------------------------------------------------------
    # SCENE ARCHITECTURE: KINETIC KINEMATICS & LOAD WEIGHT INTELLIGENCE
    # -----------------------------------------------------------------
    elif APP_STATE == "ARM_GAME":
        # Interactive Object Weight Controller Slider Array Updates
        pygame.draw.rect(screen, PANEL_DARK, (40, 30, 440, 580), border_radius=10)
        pygame.draw.rect(screen, BG_INNER, (510, 30, 400, 580), border_radius=10)
        
        screen.blit(font_title.render("KINETIC WEIGHT SELECTION CORE", True, GLOVE_CYAN), (60, 55))
        screen.blit(font_body.render("Simulate mass, inertia loads, and target bicep/tricep arrays.", True, TEXT_MUTED), (60, 90))
        
        screen.blit(font_subtitle.render(f"Simulated Weight Load Target: {chosen_weight_lbs:.1f} lbs", True, TEXT_WHITE), (60, 140))
        pygame.draw.rect(screen, BG_INNER, (60, 175, 360, 16), border_radius=4)
        
        slider_knob_x = int(60 + ((chosen_weight_lbs - 1.0) / 29.0) * 360)
        pygame.draw.circle(screen, GLOVE_CYAN, (slider_knob_x, 183), 10)
        
        if click and pygame.Rect(60, 165, 360, 30).collidepoint((mx, my)):
            pct = np.clip((mx - 60) / 360.0, 0.0, 1.0)
            chosen_weight_lbs = 1.0 + (pct * 29.0)
            
        # Derive velocity and acceleration from actual arm tracking data
        arm_velocity = 0.0
        arm_acceleration = 0.0
        
        try:
            current_arm_angle = frame_queue.get_nowait().left_elbow
            frame_delta = 0.016  # Approximate 60 FPS frame time
            
            if previous_arm_angle is not None:
                # Calculate velocity as change in angle per frame (degrees/frame)
                arm_velocity = (current_arm_angle - previous_arm_angle) / frame_delta
                
                # Calculate acceleration as change in velocity per frame
                current_velocity = arm_velocity
                arm_acceleration = (current_velocity - previous_arm_velocity) / frame_delta
            
            previous_arm_angle = current_arm_angle
            previous_arm_velocity = arm_velocity
        except:
            # No tracking available, set kinematics to zero
            arm_velocity = 0.0
            arm_acceleration = 0.0
            
        # Core Feedback Matrix Execution Run (Outputs scaled normalized 0-1)
        # bicep_val, tricep_val = calculate_haptic_feedback("ARM", weight_lbs=chosen_weight_lbs, velocity=arm_velocity, acceleration=arm_acceleration)
        
        # Single queue read for current arm tracking frame, avoid double get_nowait() calls
        # current_arm_angle = None
        # tracking_status = "no frame"
        try:
            # frame_data = frame_queue.get_nowait()
            # current_arm_angle = frame_data.left_elbow
            tracking_status = "tracking" if current_arm_angle is not None else "no angle"
        except Empty:
            tracking_status = "queue empty"
        except Exception as exc:
            tracking_status = f"error {type(exc).__name__}"

        if current_arm_angle is not None:
            shock = arm_angle2motor([current_arm_angle])[0]
        else:
            shock = 0.0

        # Smooth the shock value to prevent large jumps
        smooth_alpha = 0.8
        smoothed_shock = (smoothed_shock * (1.0 - smooth_alpha)) + (shock * smooth_alpha)
        shock_value = np.clip(smoothed_shock, -1.0, 1.0)

        # Combine arm angle with weight-based haptic feedback
        # combined_bicep = bicep_val if shock_value > 0 else 0.0
        # combined_tricep = tricep_val if shock_value < 0 else 0.0
        combined_bicep = shock_value * chosen_weight_lbs / 30.0 if shock_value > 0 else 0.0
        combined_tricep = (-shock_value) * chosen_weight_lbs / 30.0 if shock_value < 0 else 0.0

        # Scale by absolute arm angle magnitude for smooth control
        arm_magnitude = math.log(abs(shock_value),1000) + 1
        target_bicep = int(combined_bicep * arm_magnitude * 100)
        target_tricep = int(combined_tricep * arm_magnitude * 100)
        # target_bicep = math.log(target_bicep,80) + 1
        # target_tricep = math.log(target_tricep,80) + 1

        # Smooth motor outputs to avoid sudden jumps
        smoothing_alpha = 0.22
        smoothed_bicep_intensity = int((smoothed_bicep_intensity * (1.0 - smoothing_alpha)) + (target_bicep * smoothing_alpha))
        smoothed_tricep_intensity = int((smoothed_tricep_intensity * (1.0 - smoothing_alpha)) + (target_tricep * smoothing_alpha))

        smoothed_bicep_intensity = np.clip(smoothed_bicep_intensity, 0, 180)
        smoothed_tricep_intensity = np.clip(smoothed_tricep_intensity, 0, 180)
        
        msc.set_servo(smoothed_bicep_intensity, 4)
        msc.set_servo(smoothed_tricep_intensity, 2)

        # Debug overlay for arm tracking state and motor outputs
        debug_lines = [
            f"Track: {tracking_status}",
            f"Arm angle: {current_arm_angle if current_arm_angle is not None else 'N/A'}",
            f"Raw shock: {shock:.2f}",
            f"Smoothed shock: {shock_value:.2f}",
            f"Bicep: {smoothed_bicep_intensity}",
            f"Tricep: {smoothed_tricep_intensity}"
        ]
        debug_y = 520
        for line in debug_lines:
            screen.blit(font_small.render(line, True, VOLTAGE_RED if "error" in tracking_status or tracking_status == "no angle" else INDICATOR_GREEN), (60, debug_y))
            debug_y += 18

        # Kinematic Telemetry Analytics Data Logging
        screen.blit(font_subtitle.render("REAL-TIME ARM KINEMATICS MATRIX", True, TEXT_WHITE), (60, 230))
        screen.blit(font_body.render(f"Arm Center Speed Vector: {arm_velocity:.2f} m/s", True, TEXT_WHITE), (75, 265))
        screen.blit(font_body.render(f"Acceleration Inertia Force: {arm_acceleration:.2f} m/s²", True, TEXT_WHITE), (75, 295))
        
        # Dynamic Multi-Channel Wearable Upper Node Stimulation Bars
        screen.blit(font_subtitle.render("SLEEVE UPPER NODE ESTIMATED OUTPUTS", True, TEXT_WHITE), (60, 360))
        
        screen.blit(font_small.render(f"CH4 (Upper Bicep Target Array): {combined_bicep:.2f} / 1.00 Intensity", True, TEXT_WHITE), (60, 400))
        pygame.draw.rect(screen, BG_INNER, (60, 420, 360, 12), border_radius=3)
        pygame.draw.rect(screen, VOLTAGE_RED, (60, 420, int(360 * combined_bicep), 12), border_radius=3)
        
        screen.blit(font_small.render(f"CH5 (Upper Tricep Stabilizer Array): {combined_tricep:.2f} / 1.00 Intensity", True, TEXT_WHITE), (60, 460))
        pygame.draw.rect(screen, BG_INNER, (60, 480, 360, 12), border_radius=3)
        pygame.draw.rect(screen, VOLTAGE_RED, (60, 480, int(360 * combined_tricep), 12), border_radius=3)

        # Right Side Coordinate Visualization Render Pipeline Window
        # sh, el = estimator.estimate_arm_chain(h_mock, w_mock)
        
        # # Projection of 3D skeletal vectors down onto 2D Pygame surface layout coordinates
        # ox, oy = 660, 200
        # scale_f = 220
        
        # s_2d = (int(ox + sh[0]*scale_f), int(oy + (1.65 - sh[1])*scale_f))
        # e_2d = (int(ox + el[0]*scale_f), int(oy + (1.65 - el[1])*scale_f))
        # w_2d = (int(ox + w_mock[0]*scale_f), int(oy + (1.65 - w_mock[1])*scale_f))
        
        # # Draw skeleton bone linkages paths
        # pygame.draw.line(screen, TEXT_MUTED, s_2d, e_2d, 5)
        # pygame.draw.line(screen, TEXT_MUTED, e_2d, w_2d, 4)
        
        # # Joint node markers overlays
        # pygame.draw.circle(screen, INDICATOR_GREEN, s_2d, 8)
        # pygame.draw.circle(screen, VOLTAGE_RED, e_2d, 7)
        # pygame.draw.circle(screen, GLOVE_CYAN, w_2d, 6)
        
        # # Graphical labels indicators mapping layout coordinates
        # screen.blit(font_small.render(f"Shoulder Joint: {sh[0]:.2f}, {sh[1]:.2f}", True, TEXT_WHITE), (s_2d[0]+12, s_2d[1]-5))
        # screen.blit(font_small.render(f"Elbow Joint: {el[0]:.2f}, {el[1]:.2f}", True, TEXT_WHITE), (e_2d[0]+12, e_2d[1]-5))
        # screen.blit(font_small.render(f"End Effector Hand: {w_mock[0]:.2f}, {w_mock[1]:.2f}", True, TEXT_WHITE), (w_2d[0]+12, w_2d[1]-5))
    elif APP_STATE == "CHECKING":
        message = font_body.render("Checking sensor data stream for calibration readiness...", True, TEXT_WHITE)
        try:
            curls = frame_queue.get_nowait().curl
            curl_rates = abs(np.array(curls) - np.array(previous_curl))
            previous_curl = curls
            if np.all(curl_rates <= MAX_CURL_RATE):
                curl_in_range = np.all(np.abs(np.array(curls) - TARGET_CURL) <=CURL_TOLERANCE)
                for i, color in zip(range(3), check_color_dict.keys()):
                    curl_diff = curls[i] - TARGET_CURL
                    if curl_diff <= CURL_TOLERANCE and curl_diff >= -1* CURL_TOLERANCE:
                        check_color_dict[color][1] = INDICATOR_GREEN
                        check_color_dict[color][0] = 'HOLD'
                    elif curl_diff > CURL_TOLERANCE * 2:
                        check_color_dict[color][1] = CALIBRATION_RED
                        check_color_dict[color][0] = 'CURL MUCH LESS'
                    elif curl_diff > CURL_TOLERANCE:
                        check_color_dict[color][1] = CALIBRATION_YELLOW
                        check_color_dict[color][0] = 'CURL LESS'
                    elif curl_diff > -1* CURL_TOLERANCE * 2:
                        check_color_dict[color][1] = CALIBRATION_RED
                        check_color_dict[color][0] = 'CURL MUCH MORE'
                    elif curl_diff > -1* CURL_TOLERANCE:
                        check_color_dict[color][1] = CALIBRATION_YELLOW
                        check_color_dict[color][0] = 'CURL MORE'
                if curl_in_range:
                    message = font_body.render("Calibration successful! Tracking stable and within target range. Proceeding to calibration interface...", True, TEXT_WHITE)
                    APP_STATE = "CALIBRATION"
            else:
                message = font_body.render("'tracking unstable please hold still and get in view of the camera", True, TEXT_WHITE)
                check_color_dict = {'IM_COLOR':['Waiting...',CALIBRATION_RED], 'RP_COLOR': ['Waiting...',CALIBRATION_RED], 'T_COLOR': ['Waiting...',CALIBRATION_RED]}
        except:
            message = font_body.render("No data stream detected. Please ensure sensors are active and hand is in view.", True, TEXT_WHITE)
        # Dict keys | [text, color]
        # IM_COLOR 
        # RP_COLOR  
        # T_COLOR
        current_curls = list(curls) if 'curls' in locals() else [0.0, 0.0, 0.0]
        if current_curls[0] is None:
            current_curls = [0.0, 0.0, 0.0]
        screen.blit(message, (60, 100))
        screen.blit(font_small.render("Target curl target shown by the red marker. Move each finger to the green zone and hold steady.", True, TEXT_MUTED), (60, 130))
        screen.blit(font_small.render("Target Curl: {:.2f} | Tolerance: ±{:.2f} | Max Rate: {:.2f}".format(TARGET_CURL, CURL_TOLERANCE, MAX_CURL_RATE), True, TEXT_MUTED), (60, 150))

        panel_positions = [(50, 220), (350, 220), (650, 220)]
        finger_labels = ['IM', 'RP', 'T']
        for idx, (label, pos) in enumerate(zip(finger_labels, panel_positions)):
            px, py = pos
            panel_color = check_color_dict[f"{label}_COLOR"][1]
            panel_text = check_color_dict[f"{label}_COLOR"][0]
            pygame.draw.rect(screen, panel_color, (px, py, 260, 160), border_radius=12)
            pygame.draw.rect(screen, BG_INNER, (px+10, py+10, 240, 140), border_radius=10)

            screen.blit(font_subtitle.render(f"{label} Finger", True, TEXT_WHITE), (px+16, py+16))
            curl_value = current_curls[idx]
            text_color = INDICATOR_GREEN if panel_color == INDICATOR_GREEN else TEXT_WHITE
            screen.blit(font_small.render(f"Current: {curl_value}", True, text_color), (px+16, py+44))
            screen.blit(font_small.render(f"Target: {TARGET_CURL:.2f}", True, TEXT_MUTED), (px+16, py+64))

            bar_x = px + 16
            bar_y = py + 94
            bar_w = 228
            bar_h = 18
            pygame.draw.rect(screen, SPACE_GRAY, (bar_x, bar_y, bar_w, bar_h), border_radius=8)
            fill_width = int(bar_w * np.clip(curl_value, 0.0, 1.0))
            pygame.draw.rect(screen, GLOVE_CYAN, (bar_x, bar_y, fill_width, bar_h), border_radius=8)
            target_x = bar_x + int(bar_w * TARGET_CURL)
            pygame.draw.line(screen, VOLTAGE_RED, (target_x, bar_y-3), (target_x, bar_y+bar_h+3), 4)

            guide = panel_text
            guide_color = INDICATOR_GREEN if panel_color == INDICATOR_GREEN else (CALIBRATION_YELLOW if panel_color == CALIBRATION_YELLOW else VOLTAGE_RED)
            screen.blit(font_small.render(guide, True, guide_color), (px+16, py+120))

        # time.sleep(0.1) # Small delay to prevent excessive CPU usage during checking loop
    elif APP_STATE == "CALIBRATION":
        calibration_state = "running"
        calibration_failed = False
        calibration_message = f"Starting calibration for pin {servo_pins[0]}"
        calibration_prev_curl = None
        pygame.draw.rect(screen, PANEL_DARK, (40, 30, 440, 580), border_radius=10)
        pygame.draw.rect(screen, BG_INNER, (510, 30, 400, 580), border_radius=10)
        screen.blit(font_title.render("SENSOR CALIBRATION INTERFACE", True, GLOVE_CYAN), (60, 55))
        screen.blit(font_body.render("Automated servo calibration workflow: keep your hand steady and in view.", True, TEXT_MUTED), (60, 90))
        screen.blit(font_small.render(calibration_message, True, TEXT_WHITE if not calibration_failed else VOLTAGE_RED), (60, 120))

        start_btn_color = INDICATOR_GREEN if calibration_state == "running" else PANEL_DARK
        start_btn_label = "Calibration Running" if calibration_state == "running" else "Start Calibration"
        start_btn = draw_ui_button(screen, (60, 160, 200, 40), start_btn_label, start_btn_color, TEXT_BLACK)
        # if start_btn.collidepoint((mx, my)) and click and calibration_state != "running":
        #     calibration_state = "running"
        #     calibration_pin_index = 0
        #     calibration_intensity = starting_stim
        #     calibration_failed = False
        #     calibration_message = f"Starting calibration for pin {servo_pins[0]}"
        #     calibration_prev_curl = None

        if calibration_state == "running":
            if calibration_pin_index >= len(servo_pins):
                calibration_state = "finished"
                calibration_message = "Calibration complete. All servos mapped."
            else:
                current_pin = servo_pins[calibration_pin_index]
                try:
                    msc.set_servo(calibration_intensity, current_pin)
                    curls = frame_queue.get(timeout=0.5).curl
                    if calibration_prev_curl is None:
                        calibration_prev_curl = curls

                    curl_rates = abs(np.array(curls) - np.array(calibration_prev_curl))
                    calibration_prev_curl = curls
                    if np.any(np.array(curls) < 0.05):
                        msc.set_servo(0, current_pin)
                        calibration_message = f"Full extension detected. Advancing from pin {current_pin}."
                        calibration_intensity = starting_stim
                        calibration_pin_index += 1
                        calibration_prev_curl = None
                        time.sleep(1)
                        APP_STATE = "CHECKING"
                    elif np.all(curl_rates <= MAX_CURL_RATE):
                        df.at[(current_pin, calibration_intensity), "IM"] = curls[0]
                        df.at[(current_pin, calibration_intensity), "RP"] = curls[1]
                        df.at[(current_pin, calibration_intensity), "T"] = curls[2]
                        df.at[(current_pin, calibration_intensity), "IMCURL"] = curl_rates[0]
                        df.at[(current_pin, calibration_intensity), "RPCURL"] = curl_rates[1]
                        df.at[(current_pin, calibration_intensity), "TCURL"] = curl_rates[2]
                        calibration_message = f"Pin {current_pin} mapped at {calibration_intensity} intensity."
                        calibration_intensity += 1
                        time.sleep(0.04) #delay for servo movment and tracking response

                        if calibration_intensity > max_stim:
                            msc.set_servo(0, current_pin)
                            calibration_message = f"Pin {current_pin} calibration complete."
                            calibration_intensity = starting_stim
                            calibration_pin_index += 1
                            calibration_prev_curl = None
                            APP_STATE = "CHECKING"
                    else:
                        msc.set_servo(0, current_pin)
                        calibration_failed = True
                        calibration_state = "idle"
                        calibration_message = f"Tracking unstable on pin {current_pin}. Reset and retry."
                except Exception as exc:
                    calibration_failed = True
                    calibration_state = "idle"
                    calibration_message = f"Calibration paused: {exc}"

        screen.blit(font_small.render(f"Pin {calibration_pin_index + 1} / {len(servo_pins)}", True, TEXT_MUTED), (60, 220))
        screen.blit(font_small.render(f"Intensity: {calibration_intensity if calibration_state == 'running' else starting_stim}", True, TEXT_MUTED), (60, 240))
        progress = int(100 * (calibration_pin_index / len(servo_pins))) if len(servo_pins) else 0
        pygame.draw.rect(screen, TEXT_MUTED, (60, 270, 360, 16), border_radius=4)
        pygame.draw.rect(screen, GLOVE_CYAN, (60, 270, int(360 * (progress / 100.0)), 16), border_radius=4)
        screen.blit(font_small.render(f"Progress: {progress}%", True, TEXT_WHITE), (60, 295))
        if progress >= 100:
            screen.blit(font_small.render("Calibration complete!", True, INDICATOR_GREEN), (60, 330))
            APP_STATE = "CALCULATING"
            # Optionally display the calibration DataFrame or save it to a file here
    elif APP_STATE == 'CALCULATING':
        pygame.draw.rect(screen, PANEL_DARK, (40, 30, 440, 580), border_radius=10)
        pygame.draw.rect(screen, BG_INNER, (510, 30, 400, 580), border_radius=10)
        screen.blit(font_body.render("Calculating optimal servo mappings based on calibration data...", True, TEXT_WHITE), (60, 120))
        # hmap = calib.compute_pin_heatmap(df, agg='sum')
        # hmap = hmap.to_numpy(dtype=float)
        print(df[['IM','RP','T']].dropna(how='all').head())
        # plt.imshow(hmap)
        # plt.show()
        for current_curl in mapping_curls:
            best_pin, mapper = calib.build_curl_controller(df, 
                                                           others=[other_curl for other_curl in mapping_curls if other_curl != current_curl],
                                                           target=current_curl)
            controller_df.at[current_curl, 'best_pin'] = best_pin
            controller_df.at[current_curl, 'mapper'] = mapper
        new_raw = df.drop(columns=['IMCURL','RPCURL','TCURL'])
        print(controller_df)
        calib.plot_mappings(df_raw=new_raw,controller_df=controller_df,curls=mapping_curls)
        # Placeholder for potential future implementation of servo mapping optimization logic
        time.sleep(2) # Simulate processing delay
        APP_STATE = "MENU"
        calibrated = True
    pygame.display.flip()
    clock.tick(60)