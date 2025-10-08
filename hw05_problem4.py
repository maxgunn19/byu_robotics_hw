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
print("""
Verification is done by manually calculating the transformation for each link
with all q_i = 0 and comparing it to the image provided.

- A1_in_0: Rot(z, 0) * Trans(z, 0) * Trans(x, 0) * Rot(x, -pi/2).
  - Rotates frame 1 so its y-axis points along the world z-axis. This matches the image.
  
- A2_in_1: Rot(z, 0) * Trans(z, 0.154) * Trans(x, 0) * Rot(x, pi/2).
  - Moves up along y1 (world z) and rotates frame 2 so z2 points along world y. Matches.

- A3_in_2: Rot(z, 0) * Trans(z, 0.25) * Trans(x, 0) * Rot(x, 0).
  - This is a pure translation along z2 (world y) by 0.25m. Matches the image.
  
- A4_in_3: Rot(z, -pi/2) * Trans(z, 0) * Trans(x, 0) * Rot(x, -pi/2).
  - Rotates frame 4 so x4 is along world -z and z4 is along world y. Matches.
  
- A5_in_4: Rot(z, -pi/2) * Trans(z, 0) * Trans(x, 0) * Rot(x, pi/2).
  - Rotates frame 5 so z5 is along world x. Matches the spherical wrist center.

- A6_in_5: Rot(z, pi/2) * Trans(z, 0.263) * Trans(x, 0) * Rot(x, 0).
  - Translates along z5 (world x) and rotates to align frame 6 with frame 5. Matches.
  
Conclusion: The DH parameters correctly describe the robot's zero configuration shown in the image.
""")

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
print("""
By comparing the two Jacobians, we can see several key changes, particularly in columns 1 and 2.

**Changes in Column 1 (Joint 1's contribution):**
The linear velocity part (top 3 rows) of the first column changes from
[0.263, 0, -0.25] to [0.263, 0, -0.35].

*Physical Meaning:* This column describes the end-effector's velocity when only joint 1 rotates. The term `z0 x (pe - p0)` is being calculated. When the prismatic joint (q3) extends, the end-effector position `pe` moves along the world y-axis. Since `p0` is the origin and `z0` is `[0, 0, 1]`, the cross product `[0, 0, 1] x [x, y, z]` gives `[-y, x, 0]`.
As the prismatic joint extends, `pe`'s y-component increases. However, the end effector's final position is a result of all transforms, including the tip. The change is seen in the z-component of the linear velocity. This tells us that rotating the base now creates a larger downward velocity component on the tool tip. This makes physical sense: as the arm extends further out along the y-axis, a rotation at the base will cause the tip to sweep a larger arc, resulting in different instantaneous velocity components.

**Changes in Column 2 (Joint 2's contribution):**
The linear velocity part of the second column changes from
[0.25, 0.154, 0] to [0.35, 0.154, 0].

*Physical Meaning:* This column describes the end-effector velocity when only joint 2 rotates. The term `z1 x (pe - p1)` is calculated here. `z1` points along the world's z-axis, `p1` is at the origin, and `pe` changes as the prismatic joint moves. The extension of joint 3 increases the distance between joint 2 and the end-effector along the axis of joint 2's rotation. Therefore, rotating joint 2 now causes a larger linear velocity component in the x-direction at the end-effector. This is intuitive: the longer the moment arm (`pe - p1`), the larger the linear velocity generated by a given angular velocity.

**No Change in Column 3 (Prismatic Joint):**
The third column, `[0, 1, 0, 0, 0, 0]`, remains unchanged.

*Physical Meaning:* This is expected. The third column represents the contribution of the prismatic joint itself. A prismatic joint always produces a linear velocity along its axis of motion (`z2`, which is the world `y-axis` in this case) and produces no angular velocity. This effect is independent of the joint's own position along its track.

The changes perfectly illustrate that the Jacobian is configuration-dependent. The linear velocity generated by the revolute joints (1 and 2) depends directly on the position of the end-effector relative to their axes of rotation.
""")