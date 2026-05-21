import numpy as np

class ArmKinematicsEstimator:
    def __init__(self, gender_neutral_height_m=1.75):
        """
        Initializes body segment lengths based on standard anthropometric ratios.
        Default height is 1.75 meters (approx 5'9").
        """
        self.height = gender_neutral_height_m
        
        # Anthropometric proportions (Percentage of total height)
        # Drill down from head tracking to estimate the shoulder joint position
        self.head_to_shoulder_drop = self.height * 0.12  # Distance from eyes/head center down to collarbone
        self.shoulder_width_half = self.height * 0.13    # Distance from spine center out to shoulder joint
        
        # Bone lengths for the geometric arm calculation
        self.upper_arm_length = self.height * 0.186       # Shoulder to Elbow (Humeral length)
        self.forearm_length = self.height * 0.146         # Elbow to Wrist (Radial length)
        
    def estimate_arm_chain(self, head_pos, hand_pos, is_right_hand=True):
        """
        Estimates Shoulder and Elbow 3D coordinates.
        
        Inputs:
          - head_pos: np.array([x, y, z]) -> Main Camera / Headset location
          - hand_pos: np.array([x, y, z]) -> Controller / Glove location
          - is_right_hand: bool          -> Changes shoulder offset direction
        """
        # Convert inputs to numpy arrays just in case
        head = np.array(head_pos, dtype=float)
        hand = np.array(hand_pos, dtype=float)
        
        # -----------------------------------------------------------------
        # STEP 1: ESTIMATE THE SHOULDER POSITION
        # -----------------------------------------------------------------
        # Real-world assumption: The shoulder stays relatively stable relative to the head
        shoulder_offset_x = self.shoulder_width_half if is_right_hand else -self.shoulder_width_half
        
        shoulder_pos = np.array([
            head[0] + shoulder_offset_x,
            head[1] - self.head_to_shoulder_drop,
            head[2] - 0.05 # Slightly behind head plane by 5cm
        ])
        
        # -----------------------------------------------------------------
        # STEP 2: CALCULATE THE ELBOW VIA TRIANGULATION (Law of Cosines)
        # -----------------------------------------------------------------
        # Vector from shoulder straight to hand
        shoulder_to_hand = hand - shoulder_pos
        total_arm_reach = np.linalg.norm(shoulder_to_hand)
        
        # Upper arm and forearm maximum combined reach
        max_reach = self.upper_arm_length + self.forearm_length
        
        # Safety check: If hand is pulled further than real bones reach, "stretch" the arm
        if total_arm_reach >= max_reach:
            # Arm is fully straight. Elbow lies perfectly along the line to the hand.
            direction = shoulder_to_hand / total_arm_reach
            elbow_pos = shoulder_pos + (direction * self.upper_arm_length)
            return shoulder_pos, elbow_pos

        # If the arm is bent, we treat the Shoulder-Elbow-Hand as a triangle.
        # Find the angle at the shoulder joint using the Law of Cosines
        a = self.forearm_length
        b = self.upper_arm_length
        c = total_arm_reach
        
        # cos(Angle_Shoulder) = (b^2 + c^2 - a^2) / (2 * b * c)
        cos_angle = (b**2 + c**2 - a**2) / (2.0 * b * c)
        cos_angle = np.clip(cos_angle, -1.0, 1.0) # Prevent float math errors out of bounds
        shoulder_angle = np.arccos(cos_angle)
        
        # -----------------------------------------------------------------
        # STEP 3: SOLVE ELBOW PLANE (Swivel Angle / Hinge direction)
        # -----------------------------------------------------------------
        # To determine if the elbow points down, outward, or inward, we establish a target plane.
        # Default human behavior: The elbow naturally drops downwards due to gravity when relaxed.
        down_vector = np.array([0, -1, 0])
        
        # Find an axis perpendicular to the arm-reach line to rotate our elbow outwards
        forward_axis = np.cross(shoulder_to_hand, down_vector)
        axis_len = np.linalg.norm(forward_axis)
        
        if axis_len < 0.001:
            # If tracking directly below the shoulder, fallback axis points right/left
            forward_axis = np.array([1, 0, 0]) if is_right_hand else np.array([-1, 0, 0])
        else:
            forward_axis = forward_axis / axis_len
            
        # Create a reference direction pointing outward/downward from the shoulder line
        elbow_direction_base = np.cross(forward_axis, shoulder_to_hand)
        elbow_direction_base = elbow_direction_base / np.linalg.norm(elbow_direction_base)
        
        # Rotate the vector up by the calculated shoulder angle toward the hand
        arm_direction = shoulder_to_hand / total_arm_reach
        elbow_vector = (elbow_direction_base * np.sin(shoulder_angle)) + (arm_direction * np.cos(shoulder_angle))
        
        elbow_pos = shoulder_pos + (elbow_vector * self.upper_arm_length)
        
        return shoulder_pos, elbow_pos

# =====================================================================
# SIMULATION TESTING
# =====================================================================
if __name__ == "__main__":
    estimator = ArmKinematicsEstimator(gender_neutral_height_m=1.75)
    
    print("=====================================================================")
    print(" KINEMATIC RESOLVER: ESTIMATING SENSORY NODE PLACEMENT               ")
    print("=====================================================================\n")
    
    # Mock positions (Simulating a user standing upright at the center of their VR room)
    head_tracking_data = [0.0, 1.65, 0.0]     # Head at 1.65 meters off the floor
    
    # Scenario A: Right hand out straight forward reaching for a ball
    hand_reaching_forward = [0.22, 1.30, 0.45] 
    
    sh_pos, el_pos = estimator.estimate_arm_chain(head_tracking_data, hand_reaching_forward, is_right_hand=True)
    
    print("--- SCENARIO A: Reaching Forward for a Virtual Ball ---")
    print(f"Tracked Head Input:  {head_tracking_data}")
    print(f"Tracked Hand Input:  {hand_reaching_forward}")
    print(f"Estimated Shoulder:  [{sh_pos[0]:.2f}, {sh_pos[1]:.2f}, {sh_pos[2]:.2f}]")
    print(f"Calculated Elbow:    [{el_pos[0]:.2f}, {el_pos[1]:.2f}, {el_pos[2]:.2f}]")
    print(f"-> Apply Bicep/Tricep feedback payload around elevation plane Y={el_pos[1]:.2f}\n")

    # Scenario B: Right hand pulled in close to the chest (Deep Squeeze)
    hand_at_chest = [0.15, 1.35, 0.12] 
    
    sh_pos_b, el_pos_b = estimator.estimate_arm_chain(head_tracking_data, hand_at_chest, is_right_hand=True)
    
    print("--- SCENARIO B: Hand Pulled in Close to Chest (Squeezing) ---")
    print(f"Tracked Hand Input:  {hand_at_chest}")
    print(f"Estimated Shoulder:  [{sh_pos_b[0]:.2f}, {sh_pos_b[1]:.2f}, {sh_pos_b[2]:.2f}]")
    print(f"Calculated Elbow:    [{el_pos_b[0]:.2f}, {el_pos_b[1]:.2f}, {el_pos_b[2]:.2f}]")
    print(f"-> Notice how the elbow coordinates shifted outward and down to accommodate the bend.")
    print("=====================================================================")