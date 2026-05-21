import time
import math

class VRSleeveProcessor:
    def __init__(self):
        # Hardware limits for safe surface electrical stimulation (scaled 0 to 100% duty cycle / mA intensity)
        self.MAX_STIM_LEVEL = 100.0
        self.MIN_STIM_THRESHOLD = 5.0 # Noise gate to prevent constant buzzing
        
        # Simulated Physical Environment Constraints
        self.GRAVITY = 9.81 # m/s^2
        
    def calculate_sleeve_feedback(self, arm_velocity, arm_acceleration, finger_curl, object_profile):
        """
        Calculates the real-time electrical output arrays for the sleeve's multi-channel nodes.
        
        Inputs:
          - arm_velocity (float): Current speed of the arm tracking center (m/s)
          - arm_acceleration (float): Current acceleration vector magnitude (m/s^2)
          - finger_curl (float): User's actual finger closure amount (0.0 = Flat Open, 1.0 = Tight Fist)
          - object_profile (dict): Haptic attributes of the virtual target
        """
        
        # Extract object physical properties
        obj_mass = object_profile.get("mass", 0.0)          # Kilograms
        obj_stiffness = object_profile.get("stiffness", 0.0) # Hooke's constant k
        obj_radius = object_profile.get("radius", 0.035)     # Meters (e.g., 3.5cm for tennis ball)
        
        # ---------------------------------------------------------
        # 1. CHANNEL A: FOREARM STIMULATION (Compressive Force / Squeeze)
        # ---------------------------------------------------------
        # Determine if fingers have crossed the virtual boundary of the object
        # Virtual Hand Radius decreases as finger curl increases
        current_hand_radius = obj_radius * (1.0 - (finger_curl * 0.5))
        
        # Squeeze Penetration (x)
        penetration_depth = max(0.0, obj_radius - current_hand_radius)
        
        # Hooke's Law: F = k * x
        raw_squeeze_force = obj_stiffness * penetration_depth
        
        # Map physical force to Forearm Flexor Stim Current
        forearm_stim = (raw_squeeze_force * 15.0) 
        forearm_stim = min(self.MAX_STIM_LEVEL, max(0.0, forearm_stim))
        
        # ---------------------------------------------------------
        # 2. CHANNEL B & C: BICEP / TRICEP STIMULATION (Kinetic Weight & Inertia)
        # ---------------------------------------------------------
        # Total gravitational force pulling down on the held mass: F = m * g
        gravitational_force = obj_mass * self.GRAVITY
        
        # Inertial resistance during rapid movement: F = m * a
        inertial_force = obj_mass * arm_acceleration
        
        total_kinetic_force = gravitational_force + inertial_force
        
        # Directional mapping: 
        # If moving upward rapidly (positive velocity), stimulate Bicep to oppose motion
        # If moving downward or slowing down, split load to Tricep to simulate drag/drop
        bicep_stim = 0.0
        tricep_stim = 0.0
        
        if total_kinetic_force > 0:
            base_kinetic_stim = total_kinetic_force * 8.0 # Scaling multiplier
            if arm_velocity >= 0:
                bicep_stim = min(self.MAX_STIM_LEVEL, base_kinetic_stim)
                tricep_stim = min(self.MAX_STIM_LEVEL, base_kinetic_stim * 0.15) # Secondary balancing stabilization
            else:
                tricep_stim = min(self.MAX_STIM_LEVEL, base_kinetic_stim)
                bicep_stim = min(self.MAX_STIM_LEVEL, base_kinetic_stim * 0.10)

        # ---------------------------------------------------------
        # 3. SAFETY AND NOISE COMPRESSION GATE
        # ---------------------------------------------------------
        final_forearm = forearm_stim if forearm_stim >= self.MIN_STIM_THRESHOLD else 0.0
        final_bicep = bicep_stim if bicep_stim >= self.MIN_STIM_THRESHOLD else 0.0
        final_tricep = tricep_stim if tricep_stim >= self.MIN_STIM_THRESHOLD else 0.0
        
        return {
            "node_forearm_ma": round(final_forearm, 2),
            "node_bicep_ma": round(final_bicep, 2),
            "node_tricep_ma": round(final_tricep, 2),
            "telemetry": {
                "calculated_load_newtons": round(total_kinetic_force, 2),
                "deformation_mm": round(penetration_depth * 1000, 2)
            }
        }

# =====================================================================
# RUNNABLE EVALUATION LOOP
# =====================================================================
if __name__ == "__main__":
    processor = VRSleeveProcessor()
    
    # Define our testing targets (matching your 5-ball matrix)
    VIRTUAL_OBJECTS = {
        "STEEL_BALL":  {"mass": 4.5,  "stiffness": 800.0, "radius": 0.040, "desc": "Heavy, completely rigid"},
        "TENNIS_BALL": {"mass": 0.057, "stiffness": 120.0, "radius": 0.033, "desc": "Lightweight, elastic spring response"},
        "SPONGE_BALL": {"mass": 0.015, "stiffness": 15.0,  "radius": 0.050, "desc": "Negligible weight, highly compliant soft foam"}
    }
    
    print("=====================================================================")
    print(" SYSTEM INITIALIZED: RUNNING VR SLEEVE ELECTRO-HAPTIC RESPONSES     ")
    print("=====================================================================\n")
    
    # SIMULATION TEST CASE 1: User interacts with a TENNIS BALL
    print("--- SCENARIO 1: Squeezing and Lifting an ITF Standard Tennis Ball ---")
    tennis_profile = VIRTUAL_OBJECTS["TENNIS_BALL"]
    
    # Timeline matrix simulating tracking inputs over a 3-second capture window
    # Sequence format: (Timestamp, Arm Velocity m/s, Arm Accel m/s^2, Finger Curl 0-1)
    interactive_sequence_1 = [
        (0.0, 0.0, 0.0, 0.0),  # Idle resting on table
        (0.5, 0.0, 0.0, 0.4),  # Initial contact, light hand closure
        (1.0, 0.2, 1.5, 0.85), # Squeezing hard while lifting the arm upward
        (1.5, 1.1, 4.2, 0.85), # Accelerating arm upward sharply
        (2.0, 0.0, 0.0, 0.85), # Arm held perfectly still at peak position
        (2.5, 0.0, 0.0, 0.0)   # Opened hand, dropped ball
    ]
    
    for t, vel, accel, curl in interactive_sequence_1:
        output = processor.calculate_sleeve_feedback(vel, accel, curl, tennis_profile)
        print(f"[Time {t}s] Inputs -> Curl: {int(curl*100)}% | Accel: {accel} m/s²")
        print(f"         Sleeve Outputs -> Forearm Node: {output['node_forearm_ma']} mA | Bicep: {output['node_bicep_ma']} mA | Tricep: {output['node_tricep_ma']} mA")
        print(f"         Telemetry      -> Mass Deflection: {output['telemetry']['deformation_mm']} mm | Kinetic Load: {output['telemetry']['calculated_load_newtons']} N\n")
        time.sleep(0.1)

    print("---------------------------------------------------------------------")
    print("--- SCENARIO 2: Lifting a Solid Steel Ball (Comparison Test)       ---")
    steel_profile = VIRTUAL_OBJECTS["STEEL_BALL"]
    
    # Test identical acceleration and mid-range squeeze on steel to show mathematical variation
    output_steel = processor.calculate_sleeve_feedback(
        arm_velocity=1.1, 
        arm_acceleration=4.2, 
        finger_curl=0.4, 
        object_profile=steel_profile
    )
    print(f"[Steel Test] Inputs -> Curl: 40% | Accel: 4.2 m/s²")
    print(f"             Sleeve Outputs -> Forearm Node: {output_steel['node_forearm_ma']} mA | Bicep: {output_steel['node_bicep_ma']} mA | Tricep: {output_steel['node_tricep_ma']} mA")
    print(f"             Telemetry      -> Mass Deflection: {output_steel['telemetry']['deformation_mm']} mm | Kinetic Load: {output_steel['telemetry']['calculated_load_newtons']} N\n")