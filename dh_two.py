# DH Parameter Visualizer
#
# This script visualizes a robot arm in two configurations:
# 1. The "home" pose (all q=0), shown in gray.
# 2. A "target" pose (at your specified q angles), shown in green.
#
# Instructions:
# 1. Place this file in the same directory as your `kinematics.py` and `visualization.py` files.
# 2. Modify the `dh_parameters_deg`, `joint_types`, and `q_target_deg` variables below.
# 3. Run the script.

import kinematics as kin
from visualization import VizScene
import numpy as np

# --- DEFINE YOUR ROBOT HERE ---

# Enter the DH parameters for your robot.
# Each inner list represents one joint and should be in the order:
# [theta_offset, d, a, alpha]
# Note: For revolute joints, the 'theta_offset' value is added to the joint variable q.
dh_parameters_deg = [
    [0, 4, 0, -90],
    [60, 0, 2, 0],
    [0, 0, 2, 0],
    [-60, 0, 2, 90]
]

# Specify the joint types for each row in the dh_parameters table.
# 'r' for revolute, 'p' for prismatic.
joint_types = ['r', 'r', 'r', 'r']

# --- DEFINE YOUR TARGET POSE HERE ---
# Enter the desired joint angles (q values) in degrees.
# This will be visualized in GREEN.
q_target_deg = [0, 0, 0, 0]


# --- END OF ROBOT DEFINITION ---

# Convert all degrees to radians for calculations
dh_parameters = [[np.deg2rad(theta), d, a, np.deg2rad(alpha)] for theta, d, a, alpha in dh_parameters_deg]
q_target_rad = [np.deg2rad(q) for q in q_target_deg]


def visualize_home_and_target(dh, jt, q_target):
    """
    Initializes a robot from DH parameters and displays two poses:
    1. The "home" configuration (all q=0), shown in gray.
    2. The "target" configuration (at q_target), shown in green.
    """
    print("--- Dual Pose Visualizer ---")
    
    if not (len(dh) == len(jt) == len(q_target)):
        print("Error: 'dh_parameters_deg', 'joint_types', and 'q_target_deg' must all have the same length.")
        print(f"  DH params length: {len(dh)}")
        print(f"  Joint types length: {len(jt)}")
        print(f"  q_target length: {len(q_target)}")
        return

    try:
        # Create the robot arm for the "home" pose (gray)
        arm_home = kin.SerialArm(dh, jt=jt)
        num_joints = arm_home.n
        
        # Create a second, identical robot arm for the "target" pose (green)
        arm_target = kin.SerialArm(dh, jt=jt)

        # The "home" configuration is a list of zeros.
        # These are the joint *variables* (q_i), not the theta offsets.
        q_home = [0.0] * num_joints
        
        print(f"Visualizing a {num_joints}-DOF robot.")
        print("DH Parameters (with offsets):")
        print(arm_home)
        print(f"\nDisplaying 'Home' pose (Gray): q = {q_home}")
        print(f"Displaying 'Target' pose (Green): q = {q_target_deg} (deg)")

        # Set up the visualization
        viz = VizScene()
        
        # Add the "home" arm (gray, semi-transparent)
        # We add this one first.
        viz.add_arm(arm_home, joint_colors=[np.array([0.5, 0.5, 0.5, 0.5])]*num_joints) 
        
        # Add the "target" arm (green, solid)
        # We add this one second.
        viz.add_arm(arm_target, joint_colors=[np.array([0.1, 0.8, 0.1, 1.0])]*num_joints)
        
        # Update the visualization with both q vectors
        # The order in the list [q_home, q_target] must match
        # the order the arms were added (viz.add_arm).
        viz.update(qs=[q_home, q_target])
        
        # Hold the plot window open
        print("\nClose the visualization window to exit the script.")
        viz.hold()

    except Exception as e:
        print("\nAn error occurred:")
        print(e)
        print("\nPlease ensure 'kinematics.py' and 'visualization.py' are in the same directory.")


if __name__ == "__main__":
    visualize_home_and_target(dh_parameters, joint_types, q_target_rad)