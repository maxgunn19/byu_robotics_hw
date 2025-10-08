# hw05_problem2.py
# Solves problem 3 from homework 5.

import numpy as np
import sympy as sp
import kinematics as kin 

# Set numpy print options
np.set_printoptions(precision=4, suppress=True)

def symbolic_fk_oc(a1, ac):
    """
    Computes the symbolic forward kinematics for point Oc on link 2.
    
    Args:
        a1: SymPy symbol for the length of link 1.
        ac: SymPy symbol for the distance from joint 2 to point Oc.
        
    Returns:
        A tuple containing the symbolic position vector for Oc and the symbols used.
    """
    q1, q2 = sp.symbols('q1 q2')
    c1, s1 = sp.cos(q1), sp.sin(q1)
    c12, s12 = sp.cos(q1 + q2), sp.sin(q1 + q2)

    # Position vector of Oc
    oc = sp.Matrix([
        [a1 * c1 + ac * c12],
        [a1 * s1 + ac * s12],
        [0]
    ])
    
    return oc, (q1, q2)

def symbolic_jacobian_oc(a1, ac):
    """
    Computes the symbolic Jacobian for the velocity of a frame at point Oc on link 2.
    
    Args:
        a1: SymPy symbol for the length of link 1.
        ac: SymPy symbol for the distance from joint 2 to point Oc.
        
    Returns:
        The 6x3 symbolic Jacobian matrix.
    """
    q1, q2 = sp.symbols('q1 q2')
    c1, s1 = sp.cos(q1), sp.sin(q1)
    c12, s12 = sp.cos(q1 + q2), sp.sin(q1 + q2)
    
    # Initialize a 6x3 zero matrix for the Jacobian
    J = sp.zeros(6, 3)

    # --- Column 1 (Contribution from joint 1) ---
    # Linear velocity part
    J[0, 0] = -(a1 * s1 + ac * s12)
    J[1, 0] = a1 * c1 + ac * c12
    # Angular velocity part
    J[5, 0] = 1

    # --- Column 2 (Contribution from joint 2) ---
    # Linear velocity part
    J[0, 1] = -ac * s12
    J[1, 1] = ac * c12
    # Angular velocity part
    J[5, 1] = 1

    # --- Column 3 (Contribution from joint 3) ---
    # Since Oc is on link 2, joint 3's motion does not affect it.
    # The column remains zero.
    
    return J

if __name__ == "__main__":
    # Define symbolic variables for link lengths
    a1_sym, ac_sym = sp.symbols('a1 ac')
    
    # --- Part 1: Compute Symbolic Forward Kinematics ---
    print("="*60)
    print("Symbolic Forward Kinematics for o_c")
    print("="*60)
    oc_vector, (q1_sym, q2_sym) = symbolic_fk_oc(a1_sym, ac_sym)
    sp.pprint(oc_vector)

    # --- Part 2: Compute Symbolic Jacobian ---
    print("\n" + "="*60)
    print("Symbolic Jacobian for o_c")
    print("="*60)
    J_oc_symbolic = symbolic_jacobian_oc(a1_sym, ac_sym)
    sp.pprint(J_oc_symbolic)
    
    # --- Part 3: Numerical Verification (Optional) ---
    # We can verify our symbolic Jacobian by plugging in numbers and comparing
    # it to the result from the kinematics.py module.
    
    print("\n" + "="*60)
    print("Numerical Verification")
    print("="*60)

    try:
        import kinematics as kin
        
        # Define numerical values for the robot
        a1_val = 0.5
        a2_val = 0.5  # Let's assume link 2 has a length
        a3_val = 0.3  # Let's assume link 3 has a length
        ac_val = 0.2  # Point Oc is 0.2m along link 2
        
        # DH parameters for a 3R planar arm
        dh_params = [[0, 0, a1_val, 0],
                     [0, 0, a2_val, 0],
                     [0, 0, a3_val, 0]]
        
        # Create the arm object
        arm = kin.SerialArm(dh_params, jt=['r', 'r', 'r'])

        # Create a "tip" transformation to place the frame of interest at Oc on link 2
        # This requires moving back from frame 2 (joint 3) along the x-axis
        # by a distance of (a2 - ac)
        tip_transform = np.array([
            [1, 0, 0, -(a2_val - ac_val)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Create a new arm representing just the first two links with the tip at Oc
        # We'll use this to calculate the jacobian using kinematics.py
        arm_for_oc = kin.SerialArm(dh_params[0:2], jt=['r', 'r'], tip=tip_transform)
        
        # Choose a joint configuration for testing
        q_vec_test = [np.pi/6, np.pi/3, np.pi/4] # q1=30, q2=60, q3=45 deg

        print(f"Verifying at configuration q = {np.rad2deg(q_vec_test)} deg")

        # --- Calculate Jacobian using kinematics.py ---
        # Note: We calculate the Jacobian for the 2-link arm with the special tip
        # This will correctly give a 6x2 Jacobian for the point Oc
        # We then add a zero column for the 3rd joint.
        J_from_code_2link = arm_for_oc.jacob(q_vec_test[0:2], index=2, tip=True)
        J_from_code = np.hstack([J_from_code_2link, np.zeros((6, 1))])
        
        print("\nJacobian from kinematics.py:")
        print(J_from_code)
        
        # --- Calculate Jacobian from our symbolic formula ---
        # Substitute numerical values into the symbolic expression
        J_from_symbolic_func = sp.lambdify((a1_sym, ac_sym, q1_sym, q2_sym), J_oc_symbolic)
        J_from_symbolic_num = J_from_symbolic_func(a1_val, ac_val, q_vec_test[0], q_vec_test[1])
        
        print("\nJacobian from symbolic formula (numerical):")
        print(J_from_symbolic_num.astype(float))
        
        # --- Compare results ---
        if np.allclose(J_from_code, J_from_symbolic_num):
            print("\nNumerical results match the symbolic derivation")
        else:
            print("\nResults DO NOT Match")

    except ImportError:
        print("\nCould not import kinematics.py. Skipping numerical verification.")
    except Exception as e:
        print(f"\nAn error occurred during numerical verification: {e}")






