import pygame
import sys
"""
Changes to be made:
* Curl will be three values in a tuple, the ball squeeze should represent the three 'fingers'
* Adding a menu to choose between the ball and arm game with controlls bulit in to the interface. 
* Potentially add calibration script as a visual part of the ball game.
* Arm game will be bicep/tricep with cooseable 'weight'
* Output stimulation in 0-1 values.  
"""
# Initialize Pygame
pygame.init()
pygame.font.init()

# Window Setup
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("VR Hand Sleeve Haptic Simulation")
clock = pygame.time.Clock()

# Colors (Hex)
SPACE_GRAY = (30, 32, 36)
TEXT_WHITE = (240, 240, 240)
GLOVE_CYAN = (0, 230, 255)
VOLTAGE_RED = (255, 45, 85)

# Ball Types & Materials (Haptic Profiles)
# Softness dictates how easily the ball deforms and how much electrical feedback is sent
BALL_PROFILES = {
    "STEEL":  {"color": (160, 165, 170), "softness": 0.05, "desc": "Solid metal. High electrical feedback instantly."},
    "TENNIS": {"color": (212, 255, 0),   "softness": 0.60, "desc": "Springy resistance. Moderate, gradual feedback."},
    "SPONGE": {"color": (255, 200, 50),  "softness": 0.95, "desc": "Squishy foam. Minimal feedback until fully compressed."}
}
ball_list = list(BALL_PROFILES.keys())
current_ball_idx = 0

# Simulation Variables
hand_curl = 0.0       # 0.0 = Open Hand, 1.0 = Closed Fist (Simulating tracking input)
ball_deformation = 0.0 # How much the ball is physically squished
electrical_feedback = 0.0 # EMS/TENS current level sent to the sleeve (0V to 100V scaled)

# Fonts
font_title = pygame.font.SysFont("Arial", 24, bold=True)
font_body = pygame.font.SysFont("Arial", 18)

while True:
    screen.fill(SPACE_GRAY)
    
    # 1. Handle Inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Toggle between materials
                current_ball_idx = (current_ball_idx + 1) % len(ball_list)
                hand_curl = 0.0 # Reset squeeze on switch

    # Simulate Hand Tracking (Hold UP arrow to close hand, DOWN arrow to open)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        hand_curl = min(1.0, hand_curl + 0.02)
    if keys[pygame.K_DOWN]:
        hand_curl = max(0.0, hand_curl - 0.02)

    # 2. Haptic & Physics Math (The Core Product Tech)
    active_ball = ball_list[current_ball_idx]
    softness = BALL_PROFILES[active_ball]["softness"]
    
    # Hooke's Law / Resistance Simulation
    # Harder objects limit hand movement (deformation) but spike electrical feedback early
    stiffness = 1.0 / softness
    
    # Calculate how much the ball actually yields to the hand
    ball_deformation = hand_curl * softness 
    
    # Calculate Electrical Feedback (Force = Depth * Stiffness)
    if hand_curl > 0:
        # Electrical signal matches the physical pushback force
        raw_force = hand_curl * stiffness
        electrical_feedback = min(100.0, raw_force * 10) 
    else:
        electrical_feedback = 0.0

    # 3. Render Visuals
    
    # Draw Background Panels
    pygame.draw.rect(screen, (40, 44, 52), (50, 40, 400, 520), border_radius=10) # Control Panel
    pygame.draw.rect(screen, (22, 24, 28), (480, 40, 370, 520), border_radius=10) # Visualizer

    # Text Layout: Left Panel (Product Status)
    title_text = font_title.render("VR SLEEVE TECH DEMO", True, GLOVE_CYAN)
    screen.blit(title_text, (70, 60))
    
    desc_title = font_body.render(f"Active Material: {active_ball}", True, TEXT_WHITE)
    desc_detail = font_body.render(BALL_PROFILES[active_ball]["desc"], True, (150, 150, 150))
    screen.blit(desc_title, (70, 110))
    screen.blit(desc_detail, (70, 140))

    # Control Instructions
    instruct_1 = font_body.render("• Hold [UP ARROW] to Curl Fingers / Squeeze", True, TEXT_WHITE)
    instruct_2 = font_body.render("• Hold [DOWN ARROW] to Open Hand", True, TEXT_WHITE)
    instruct_3 = font_body.render("• Press [SPACEBAR] to Change Object Material", True, GLOVE_CYAN)
    screen.blit(instruct_1, (70, 200))
    screen.blit(instruct_2, (70, 230))
    screen.blit(instruct_3, (70, 260))

    # Haptic Telemetry Bars
    # Tracker Input Bar
    pygame.draw.rect(screen, (60, 65, 75), (70, 360, 350, 20), border_radius=5)
    pygame.draw.rect(screen, GLOVE_CYAN, (70, 360, int(350 * hand_curl), 20), border_radius=5)
    track_lbl = font_body.render(f"Sleeve Tracking (Finger Curl): {int(hand_curl * 100)}%", True, TEXT_WHITE)
    screen.blit(track_lbl, (70, 330))

    # Electrical Impulse Output Bar
    pygame.draw.rect(screen, (60, 65, 75), (70, 440, 350, 20), border_radius=5)
    pygame.draw.rect(screen, VOLTAGE_RED, (70, 440, int(350 * (electrical_feedback / 100.0)), 20), border_radius=5)
    voltage_lbl = font_body.render(f"Electrical Feedback Level: {int(electrical_feedback)} mA (Sensation)", True, TEXT_WHITE)
    screen.blit(voltage_lbl, (70, 410))

    # Draw Right Panel Visualizer (The Ball getting squeezed)
    center_x, center_y = 665, 300
    base_radius = 100
    
    # Calculate squeezed dimensions (As hand curl increases, height squishes based on softness)
    squish_factor = 1.0 - (hand_curl * (1.0 - softness) * 0.5)
    radius_y = int(base_radius * squish_factor)
    radius_x = int(base_radius * (1.0 + (hand_curl * (1.0 - softness) * 0.2))) # Slight outward bulge

    # Draw the reactive ball
    ball_color = BALL_PROFILES[active_ball]["color"]
    pygame.draw.ellipse(screen, ball_color, (center_x - radius_x, center_y - radius_y, radius_x * 2, radius_y * 2))
    
    # Draw "Virtual Hand" boundary rings compressing it
    if hand_curl > 0:
        hand_color = (GLOVE_CYAN[0], GLOVE_CYAN[1], GLOVE_CYAN[2], 100)
        pygame.draw.arc(screen, GLOVE_CYAN, (center_x - radius_x - 5, center_y - radius_y - 5, radius_x * 2 + 10, radius_y * 2 + 10), 0, 3.14, 3)

    pygame.display.flip()
    clock.tick(60)