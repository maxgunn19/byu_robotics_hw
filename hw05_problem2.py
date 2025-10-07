# hw05_problem2.py
# Solves problem 2 from homework 5.

import numpy as np
import kinematics as kin # Assumes kinematics.py is in the same folder

# Set numpy print options
np.set_printoptions(precision=4, suppress=True)

# ======================================================
# (a) Symbolic Jacobian Derivation
# =====================================================


print("="*60)
print("Part (a): Symbolic Jacobian (by hand)")
print("="*60)
print("""
      The symbolic Jacobian J(q) for the R-P robot, with link length L,
is derived as follows (using the textbook's [linear; angular] convention):

      [ -L*sin(q1)   0 ]
      [  L*cos(q1)   0 ]
J(q) = [     0         1 ]
      [     0         0 ]
      [     0         0 ]
      [     1         0 ]
"""
)

# Function to calculate the symbolic Jacobian for comparison
def symbolic_jacobian(q, L):
    """Calculates the symbolic jacobian for the R-P robot."""
    q1, _ = q
    J = np.zeros((6, 2))
    
    # Column 1 (Revolute joint)
    # Linear velocity part
    J[0, 0] = -L * np.sin(q1)
    J[1, 0] = L * np.cos(q1)
    # Angular velocity part
    J[5, 0] = 1
    
    # Column 2 (Prismatic joint)
    # Linear velocity part
    J[2, 1] = 1
    # Angular velocity part is already zero

    return J

# =============================================================================
# (b) Code Implementation and Comparison
# =============================================================================
print("\n" + "="*60)
print("Part (b): Code Implementation and Comparison")
print("="*60)

# Robot parameters
L = 0.30  # 30 cm link length

# DH parameters in the format [theta, d, a, alpha]
# Link 1 (Revolute): Variable is theta (q1)
# Link 2 (Prismatic): Variable is d (q2), with a radial offset 'a' of L
dh_params = [[0, 0, 0, 0],
             [0, 0, L, 0]]

# Joint types ('r' for revolute, 'p' for prismatic)
joint_types = ['r', 'p']

# Create the SerialArm object
rp_robot = kin.SerialArm(dh=dh_params, jt=joint_types)

print("Robot DH Parameters:")
print(rp_robot)

# Define a few joint configurations to test
q_configs = {
    "Config 1: q = [0 rad, 0.1 m]": [0, 0.1],
    "Config 2: q = [pi/4 rad, 0.2 m]": [np.pi/4, 0.2],
    "Config 3: q = [pi/2 rad, 0.15 m]": [np.pi/2, 0.15]
}

# Loop through configurations and compare Jacobians
for name, q_vec in q_configs.items():
    print(f"\n--- {name} ---")

    # Calculate Jacobian using your updated kinematics.py code.
    # No row-swapping is needed now.
    J_from_code = rp_robot.jacob(q_vec)

    # Calculate Jacobian using the symbolic formula
    J_from_symbolic = symbolic_jacobian(q_vec, L)

    print("Jacobian from kinematics.py code ([linear; angular]):")
    print(J_from_code)

    print("\nJacobian from symbolic formula:")
    print(J_from_symbolic)

    # Check if the results are numerically close
    if np.allclose(J_from_code, J_from_symbolic):
        print("\nResults Match")
    else:
        print("\nResults DO NOT Match")