# DH Parameter Visualizer
#
# This script is designed to help you quickly visualize a robot arm
# based on a set of DH parameters to verify its zero-angle ("home") configuration.
#
# Instructions:
# 1. Place this file in the same directory as your `kinematics.py` and `visualization.py` files.
# 2. Modify the `dh_parameters` and `joint_types` variables below to match the robot you want to test.
# 3. Run the script. A 3D window will appear showing the robot with all joint angles set to zero.

import kinematics as kin
from visualization import VizScene
import numpy as np

# --- DEFINE YOUR ROBOT HERE ---

# Enter the DH parameters for your robot.
# Each inner list represents one joint and should be in the order:
# [theta, d, a, alpha]
# Note: For revolute joints, the 'theta' value here is the offset. The joint variable q will be added to it.
# For prismatic joints, the 'd' value is the offset.

dh_parameters_deg = [
    [90, 4, 0, 90],
    [-30, 0, 2, -90],
    [0, 0, 2, -90],
    [-30, 0, 2, 90]
]

dh_parameters_deg = [
    [0, 4, 0, -90],
    [60, 0, 2, 0],
    [0, 0, 2, 0],
    [-60, 0, 2, 90]
]

dh_parameters = [[np.deg2rad(theta), d, a, np.deg2rad(alpha)] for theta, d, a, alpha in dh_parameters_deg]

# Specify the joint types for each row in the dh_parameters table.
# 'r' for revolute, 'p' for prismatic.
joint_types = ['r', 'r', 'r', 'r']

# --- END OF ROBOT DEFINITION ---


def visualize_zero_config(dh, jt):
    """
    Initializes a robot from DH parameters and displays it in its zero configuration.
    """
    print("--- DH Parameter Visualizer ---")
    
    if len(dh) != len(jt):
        print("Error: The number of rows in dh_parameters must match the number of joint_types.")
        return

    try:
        # Create the robot arm object
        arm = kin.SerialArm(dh, jt=jt)
        num_joints = arm.n
        
        # The zero configuration is a list of zeros, one for each joint.
        q_zero = [0.0] * num_joints
        
        print(f"Visualizing a {num_joints}-DOF robot with DH parameters:")
        print(arm)
        print(f"Displaying robot at zero configuration: q = {q_zero}")

        # Set up the visualization
        viz = VizScene()
        viz.add_arm(arm, joint_colors=[np.array([0.1, 0.8, 0.1, 1.0])]*num_joints) # Green joints
        
        # Update the visualization to show the arm at the zero configuration
        viz.update(qs=[q_zero])
        
        # Hold the plot window open until it is manually closed
        print("Close the visualization window to exit the script.")
        viz.hold()

    except Exception as e:
        print("\nAn error occurred:")
        print(e)
        print("\nPlease ensure that your `kinematics.py` and `visualization.py` files are in the same directory.")


if __name__ == "__main__":
    visualize_zero_config(dh_parameters, joint_types)
