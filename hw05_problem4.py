# hw05_problem4.py
# Analysis of the Stanford Manipulator based on homework problem 4.

import numpy as np
import kinematics as kin
import transforms as tr
from visualization import VizScene

# Set numpy print options for clean output
np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# (a) Verification of DH Parameters at Zero Configuration
# =============================================================================
print("="*60)
print("Part (a): DH Parameter Verification")
print("="*60)

# =============================================================================
# (b) Create the SerialArm Object
# =============================================================================
print("\n" + "="*60)
print("Part (b): Creating the SerialArm Object")
print("="*60)

# DH parameters from the homework table [theta, d, a, alpha]
# Note the offsets for q4, q5, q6 are part of the 'theta' parameter.
dh_stanford = [
    [0,         0,       0,     -np.pi/2],  # q1
    [0,         0.154,   0,      np.pi/2],  # q2
    [0,         0.25,    0,      0       ],  # q3 (prismatic, d varies)
    [-np.pi/2,  0,       0,     -np.pi/2],  # q4
    [-np.pi/2,  0,       0,      np.pi/2],  # q5
    [np.pi/2,   0.263,   0,      0       ]   # q6
]

# Define joint types. Joint 3 (index 2) is prismatic.
jt_stanford = ['r', 'r', 'p', 'r', 'r', 'r']

# Define the tip transformation matrix T_tip_in_6
# This rotates the final tool frame by -90 degrees around its y-axis.
T_tip_in_6 = np.array([
    [0, 0, -1, 0],
    [0, 1,  0, 0],
    [1, 0,  0, 0],
    [0, 0,  0, 1]
])

# Create the SerialArm object for the Stanford arm
stanford_arm = kin.SerialArm(dh=dh_stanford, jt=jt_stanford, tip=T_tip_in_6)

print("Stanford Arm Object created successfully.")
print(stanford_arm)

# =============================================================================
# (c) Confirm Frame Matching with Visualization
# =============================================================================
print("\n" + "="*60)
print("Part (c): Visualizing the Arm at Zero Configuration")
print("="*60)

# Zero configuration vector
q_zero = [0, 0, 0, 0, 0, 0]

print("Generating visualization for q = [0, 0, 0, 0, 0, 0]...")
print("Compare the generated plot to the homework image.")

# Create a visualization scene
viz = VizScene()

# Add the Stanford arm to the scene, instructing it to draw the frames.
viz.add_arm(stanford_arm, draw_frames=True)

# Update the scene with the zero joint angles.
# The 'qs' argument expects a list of configurations for multiple arms,
# so we provide a list containing just our single configuration.
viz.update(qs=[q_zero])

# Display the plot
viz.hold()

# =============================================================================
# (d) Jacobian Calculation and Discussion
# =============================================================================
print("\n" + "="*60)
print("Part (d): Jacobian Calculation and Discussion")
print("="*60)

# --- Jacobian at q = [0, 0, 0, 0, 0, 0] ---
J_zero = stanford_arm.jacob(q_zero, tip=True) 
print("Jacobian at q = [0, 0, 0, 0, 0, 0]:")
print(J_zero)

# --- Jacobian at q = [0, 0, 0.1, 0, 0, 0] ---
q_prismatic_moved = [0, 0, 0.1, 0, 0, 0]
J_moved = stanford_arm.jacob(q_prismatic_moved, tip=True)
print("\nJacobian when prismatic joint moves by +0.1m:")
print(J_moved)

print("\n--- Discussion ---")
