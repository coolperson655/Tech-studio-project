import pygame
import sys
import numpy as np
import math
from Functions.Calculation_functions import check_hand_curl
from Functions.Streaming_functions import get_client_and_thread, get_curls
# import Functions.Multi_servo_control as msc
import pandas as pd

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
TEXT_MUTED = (140, 145, 155)
GLOVE_CYAN = (0, 220, 255)
VOLTAGE_RED = (255, 50, 85)
INDICATOR_GREEN = (40, 220, 110)

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
        ball_btn = draw_ui_button(screen, (175, 220, 260, 160), "1. DEFORMABLE BALL LAB", PANEL_DARK)
        arm_btn = draw_ui_button(screen, (515, 220, 260, 160), "2. KINETIC ARM WORKSPACE", PANEL_DARK)
        
        # Interactive Selection Evaluation
        if ball_btn.collidepoint((mx, my)):
            draw_ui_button(screen, (175, 220, 260, 160), "1. DEFORMABLE BALL LAB", GLOVE_CYAN, SPACE_GRAY)
            if click: APP_STATE = "BALL_GAME"
            
        if arm_btn.collidepoint((mx, my)):
            draw_ui_button(screen, (515, 220, 260, 160), "2. KINETIC ARM WORKSPACE", GLOVE_CYAN, SPACE_GRAY)
            if click: APP_STATE = "ARM_GAME"
            
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
        # Input Controller Map Tracking Handling

        # hand_curl = streamf.get_curls() 

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            hand_curl[active_finger_idx] = min(calibrated_max, hand_curl[active_finger_idx] + 0.02)
        if keys[pygame.K_DOWN]:
            hand_curl[active_finger_idx] = max(calibrated_min, hand_curl[active_finger_idx] - 0.02)
            
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
            
        cal_btn_txt = "Calibration Active" if calibration_mode else "Calibrate Sensors"
        cal_btn_col = INDICATOR_GREEN if calibration_mode else PANEL_DARK
        cal_btn = draw_ui_button(screen, (240, 150, 160, 30), cal_btn_txt, cal_btn_col, TEXT_WHITE)
        if cal_btn.collidepoint((mx, my)) and click:
            calibration_mode = not calibration_mode
            
        # Render Multi-Finger Calibration Suite System Window Block
        pygame.draw.rect(screen, BG_INNER, (60, 200, 400, 195), border_radius=6)
        screen.blit(font_subtitle.render("INTEGRATED SLEEVE SENSOR CALIBRATION MATRIX", True, TEXT_WHITE), (75, 215))
        
        fingers_list = ["1. THUMB", "2. INDEX", "3. MIDDLE"]
        for i, f_name in enumerate(fingers_list):
            y_offset = 255 + (i * 45)
            # Accent color highlighting current calibrated selector focus channel tracking array
            text_color = GLOVE_CYAN if i == active_finger_idx else TEXT_WHITE
            screen.blit(font_body.render(f_name, True, text_color), (75, y_offset))
            
            # Draw tracking bar background track
            pygame.draw.rect(screen, PANEL_DARK, (165, y_offset + 2, 210, 14), border_radius=4)
            pygame.draw.rect(screen, GLOVE_CYAN, (165, y_offset + 2, int(210 * hand_curl[i]), 14), border_radius=4)
            
            # Handle selection clicks to map tracking system calibration arrays
            f_rect = pygame.Rect(75, y_offset, 300, 20)
            if f_rect.collidepoint((mx, my)) and click:
                active_finger_idx = i

        if calibration_mode:
            screen.blit(font_small.render("CALIBRATION MODE LOCK ACTIVE: Setting tracking clip points.", True, INDICATOR_GREEN), (75, 365))
            
        # Multi-Channel Haptic Pulse Current Output Bar Arrays (Normalized values 0-1)
        screen.blit(font_subtitle.render("MULTI-CHANNEL PULSE FEEDBACK OUTPUT", True, TEXT_WHITE), (60, 420))
        channels = ["CH1 (Forearm Flexor Thumb):", "CH2 (Forearm Flexor Index/Middle):", "CH3 (Forearm Flexor Ring/Pinky):"]
        for i, ch_lbl in enumerate(channels):
            y_offset = 450 + (i * 50)
            screen.blit(font_small.render(f"{ch_lbl} {stims[i]:.2f} / 1.00 Intensity", True, TEXT_WHITE), (60, y_offset))
            pygame.draw.rect(screen, (50, 55, 65), (60, y_offset + 18, 400, 12), border_radius=3)
            pygame.draw.rect(screen, VOLTAGE_RED, (60, y_offset + 18, int(400 * stims[i]), 12), border_radius=3)

        # Right Side Visualizer Rendering Canvas Panel Window
        bx, by = 710, 320
        br = 90
        # Average deform compression multiplier scalar implementation mapping layout tracking shapes
        sq_fac = 1.0 - (avg_deform * (1.0 - profile["stiffness"]/800.0) * 0.4)
        rx = int(br * (1.0 + avg_deform * 0.15))
        ry = int(br * sq_fac)
        pygame.draw.ellipse(screen, profile["color"], (bx - rx, by - ry, rx * 2, ry * 2))
        
        # Virtual Mesh Skeleton Hand Boundary Track Overlays
        for i, curl in enumerate(hand_curl):
            if curl > 0:
                arc_r = br + 10 + (i * 8)
                pygame.draw.arc(screen, GLOVE_CYAN, (bx - arc_r, by - arc_r, arc_r * 2, arc_r * 2), 0.2, 2.9, 2)
        screen.blit(font_small.render("SURFACE GRASP MESH DEFORMATION CORE", True, TEXT_MUTED), (bx - 110, by + 180))
        
        # Set the servo intensities
        for stim in stims:

        # msc.set_intensity()

    # -----------------------------------------------------------------
    # SCENE ARCHITECTURE: KINETIC KINEMATICS & LOAD WEIGHT INTELLIGENCE
    # -----------------------------------------------------------------
    elif APP_STATE == "ARM_GAME":
        # Interactive Object Weight Controller Slider Array Updates (1 to 30 Pounds Interface Tracking)
        pygame.draw.rect(screen, PANEL_DARK, (40, 30, 440, 580), border_radius=10)
        pygame.draw.rect(screen, BG_INNER, (510, 30, 400, 580), border_radius=10)
        
        screen.blit(font_title.render("KINETIC WEIGHT SELECTION CORE", True, GLOVE_CYAN), (60, 55))
        screen.blit(font_body.render("Simulate mass, inertia loads, and target bicep/tricep arrays.", True, TEXT_MUTED), (60, 90))
        
        # Weight Load Configuration Slider Widget Render Architecture
        screen.blit(font_subtitle.render(f"Simulated Weight Load Target: {chosen_weight_lbs:.1f} lbs", True, TEXT_WHITE), (60, 140))
        pygame.draw.rect(screen, BG_INNER, (60, 175, 360, 16), border_radius=4)
        
        # Capture slider movement coordinates logic map loops
        slider_knob_x = int(60 + ((chosen_weight_lbs - 1.0) / 29.0) * 360)
        pygame.draw.circle(screen, GLOVE_CYAN, (slider_knob_x, 183), 10)
        
        if click and pygame.Rect(60, 165, 360, 30).collidepoint((mx, my)):
            pct = np.clip((mx - 60) / 360.0, 0.0, 1.0)
            chosen_weight_lbs = 1.0 + (pct * 29.0)
            
        # Kinematic Motion Acceleration Simulation Data Loop Generator
        time_accumulator += 0.04
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            # Simulate real arm swinging: derived displacement maps
            arm_velocity = math.sin(time_accumulator) * 1.5
            arm_acceleration = abs(math.cos(time_accumulator) * 3.5)
        else:
            arm_velocity = 0.0
            arm_acceleration = 0.0
            
        # Core Feedback Matrix Execution Run (Outputs scaled normalized 0-1)
        bicep_val, tricep_val = calculate_haptic_feedback("ARM", weight_lbs=chosen_weight_lbs, velocity=arm_velocity, acceleration=arm_acceleration)
        
        # Kinematic Telemetry Analytics Data Logging
        screen.blit(font_subtitle.render("REAL-TIME ARM KINEMATICS MATRIX", True, TEXT_WHITE), (60, 230))
        screen.blit(font_body.render(f"Arm Center Speed Vector: {arm_velocity:.2f} m/s", True, TEXT_WHITE), (75, 265))
        screen.blit(font_body.render(f"Acceleration Inertia Force: {arm_acceleration:.2f} m/s²", True, TEXT_WHITE), (75, 295))
        
        # Dynamic Multi-Channel Wearable Upper Node Stimulation Bars (Normalized 0-1 outputs)
        screen.blit(font_subtitle.render("SLEEVE UPPER NODE ESTIMATED OUTPUTS", True, TEXT_WHITE), (60, 360))
        
        screen.blit(font_small.render(f"CH4 (Upper Bicep Target Array): {bicep_val:.2f} / 1.00 Intensity", True, TEXT_WHITE), (60, 400))
        pygame.draw.rect(screen, BG_INNER, (60, 420, 360, 12), border_radius=3)
        pygame.draw.rect(screen, VOLTAGE_RED, (60, 420, int(360 * bicep_val), 12), border_radius=3)
        
        screen.blit(font_small.render(f"CH5 (Upper Tricep Stabilizer Array): {tricep_val:.2f} / 1.00 Intensity", True, TEXT_WHITE), (60, 460))
        pygame.draw.rect(screen, BG_INNER, (60, 480, 360, 12), border_radius=3)
        pygame.draw.rect(screen, VOLTAGE_RED, (60, 480, int(360 * tricep_val), 12), border_radius=3)

        # Right Side Coordinate Visualization Render Pipeline Window
        # Simulate tracking system reference origins mapping
        h_mock = [0.0, 1.65, 0.0]
        w_mock = [0.25, 1.10 + (math.sin(time_accumulator) * 0.25 if arm_velocity != 0 else 0), 0.35]
        
        sh, el = estimator.estimate_arm_chain(h_mock, w_mock)
        
        # Projection of 3D skeletal vectors down onto 2D Pygame surface layout coordinates
        ox, oy = 660, 200
        scale_f = 220
        
        s_2d = (int(ox + sh[0]*scale_f), int(oy + (1.65 - sh[1])*scale_f))
        e_2d = (int(ox + el[0]*scale_f), int(oy + (1.65 - el[1])*scale_f))
        w_2d = (int(ox + w_mock[0]*scale_f), int(oy + (1.65 - w_mock[1])*scale_f))
        
        # Draw skeleton bone linkages paths
        pygame.draw.line(screen, TEXT_MUTED, s_2d, e_2d, 5) # Upper Humeral arm tracking link
        pygame.draw.line(screen, TEXT_MUTED, e_2d, w_2d, 4) # Lower Radial forearm tracking link
        
        # Joint node markers overlays
        pygame.draw.circle(screen, INDICATOR_GREEN, s_2d, 8) # Shoulder Joint Node
        pygame.draw.circle(screen, VOLTAGE_RED, e_2d, 7)      # Estimated Target Elbow Node
        pygame.draw.circle(screen, GLOVE_CYAN, w_2d, 6)       # Hand End-Effector Tracking Target
        
        # Graphical labels indicators mapping layout coordinates
        screen.blit(font_small.render(f"Shoulder Joint: {sh[0]:.2f}, {sh[1]:.2f}", True, TEXT_WHITE), (s_2d[0]+12, s_2d[1]-5))
        screen.blit(font_small.render(f"Elbow Joint: {el[0]:.2f}, {el[1]:.2f}", True, TEXT_WHITE), (e_2d[0]+12, e_2d[1]-5))
        screen.blit(font_small.render(f"End Effector Hand: {w_mock[0]:.2f}, {w_mock[1]:.2f}", True, TEXT_WHITE), (w_2d[0]+12, w_2d[1]-5))

    pygame.display.flip()
    clock.tick(60)