"""
Kinematics Module - Contains code for:
- Forward Kinematics, from a set of DH parameters to a serial linkage arm with callable forward kinematics
- Inverse Kinematics
- Jacobian

John Morrell, Jan 26 2022
Tarnarmour@gmail.com

modified by:
Marc Killpack, Sept 21, 2022 and Sept 21, 2023
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Iterable


# this is a convenience function that makes it easy to define a function that calculates
# "A_i(q_i)", given the DH parameters for link and joint "i" only.
def dh2A(dh: list[float], jt: str) -> Callable[[float], NDArray]:
    """
    Creates a function, A(q), that will generate a homogeneous transform T for a single
    joint/link given a set of DH parameters. A_i(q_i) represents the transform from link
    i-1 to link i, e.g. A1(q1) gives T_1_in_0. This follows the "standard" DH convention.

    :param list[float] dh: list of 4 dh parameters (single row from DH table) for the
        transform from link i-1 to link i, in the order [theta d a alpha] - THIS IS NOT
        THE CONVENTION IN THE BOOK!!! But it is the order of operations.
    :param str jt: joint type: 'r' for revolute joint, 'p' for prismatic joint
    :return A: a function of the corresponding joint angle, A(q), that generates a 4x4
        numpy array representing the homogeneous transform from one link to the next
    """
    # if joint is revolute implement correct equations here:
    if jt == 'r':
        # although A(q) is only a function of "q", the dh parameters are available to these next functions
        # because they are passed into the function above.

        def A(q: float) -> NDArray:
            # See eq. (2.52), pg. 64
            # TODO - complete code that defines the "A" or "T" homogenous matrix for a given set of DH parameters.
            # Do this in terms of the variables "dh" and "q" (so that one of the entries in your dh list or array
            # will need to be added to q).
            theta, d, a, alpha = dh
            theta += q  # since this is a revolute joint, theta is variable

            T = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0,              np.sin(alpha),                np.cos(alpha),               d],
                [0,              0,                            0,                           1]
            ])

            return T

    # if joint is prismatic implement correct equations here:
    else:
        def A(q: float) -> NDArray:
            # See eq. (2.52), pg. 64
            # TODO - complete code that defines the "A" or "T" homogenous matrix for a given set of DH parameters.
            # Do this in terms of the variables "dh" and "q" (so that one of the entries in your dh list or array
            # will need to be added to q).
            theta, d, a, alpha = dh
            d += q  # since this is a prismatic joint, d is variable

            T = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0,              np.sin(alpha),                np.cos(alpha),               d],
                [0,              0,                            0,                           1]
            ])

            return T

    return A


class SerialArm:
    """
    SerialArm - A class designed to represent a serial link robot arm

    SerialArms have frames 0 to n defined, with frame 0 located at the first joint and
    aligned with the robot body frame, and frame n located at the end of link n.
    """

    def __init__(self, dh: list[list[float]], jt: list[str]|None=None,
                 base: NDArray=np.eye(4), tip: NDArray=np.eye(4),
                 joint_limits: NDArray|None=None):
        """
        arm = SerialArm(dh, jt, base=I, tip=I, joint_limits=None)

        :param list[list[float]] dh: n length list where each entry is another list of
            4 dh parameters: [theta d a alpha]
        :param list[str] | None jt: n length list of strings for joint types,
            'r' for revolute joint and 'p' for prismatic joint.
            If None, all joints are set to revolute.
        :param NDArray base: 4x4 numpy array representing SE3 transform from world or
            inertial frame to frame 0 (T_0_in_base)
        :param NDArray tip: 4x4 numpy array representing SE3 transform from frame n to
            tool frame or tip of robot (T_tip_in_n)
        :param NDArray | None joint_limits: 2*n array, min joint limit in 1st row then
            max joint limit in 2nd row (values in radians/meters).
            None for not implemented (these are only used in visualization).
        """
        self.dh = dh
        self.n = len(dh)

        # we will use this list to store the A matrices for each set/row of DH parameters.
        self.transforms: list[Callable[[float], NDArray]] = []

        # assigning a joint type
        if jt is None:
            self.jt = ['r'] * self.n
        else:
            assert len(jt) == self.n, "Joint type list does not have the same size as dh param list!"
            self.jt = jt

        # using the code we wrote above to generate the function A(q) for each set of DH parameters
        for i in range(self.n):
            # TODO use the function definition above (dh2A), and the dh parameters and
            # joint type to make a function and then append that function to the
            # "transforms" list (use the versions from self because they have error checks).
            A = dh2A(self.dh[i], self.jt[i])
            self.transforms.append(A)

        # assigning the base, and tip transforms that will be added to the default DH transformations.
        self.base = base.copy()
        self.tip = tip.copy()
        
        # Store joint limits
        if joint_limits is None:
            # Default to -pi to pi for revolute, -1 to 1 for prismatic
            self.qlim = np.zeros((2, self.n))
            for i, j_type in enumerate(self.jt):
                if j_type == 'r':
                    self.qlim[:, i] = [-np.pi, np.pi]
                else:
                    self.qlim[:, i] = [-1.0, 1.0]
        else:
             self.qlim = joint_limits.T # Transpose to get 2xN
        
        # calculating rough numbers to understand the workspace for drawing the robot
        self.reach = 0
        for dh_row in self.dh:
            # Using norm of (d, a) parameters
            self.reach += np.linalg.norm([dh_row[1], dh_row[2]])
        self.reach += np.linalg.norm(self.tip[:3, 3]) # Add tip offset


    # You don't need to touch this function, but it is helpful to be able to "print"
    # a description about the robot that you make.
    def __str__(self):
        """
        This function just provides a nice interface for printing information about the arm.
        If we call "print(arm)" on an SerialArm object "arm", then this function gets called.
        See example in "main" below.
        """
        chars_per_col = 9
        dh_string = 'Serial Arm: DH Parameters\n'
        labels = ['θ', 'd', 'a', 'α', 'jt']
        cols = len(labels)
        dh_string += f'┌{"┬".join(["—"*chars_per_col for i in range(cols)])}┐\n'
        dh_string += f"|{''.join([f'{l.center(chars_per_col)}|' for l in labels])}\n"
        line = f"{'—'*chars_per_col}|"
        dh_string += f'|{line*cols}\n'
        for dh, jt in zip(self.dh, self.jt):
            row = [f'{val:.3f}'.rstrip('0') if isinstance(val,float) else f'{val}' for val in [*dh,jt]]
            dh_string += f"|{''.join([f'{str(s).center(chars_per_col)}|' for s in row])}\n"
        dh_string += f'└{"┴".join(["—"*chars_per_col for i in range(cols)])}┘\n'
        return dh_string


    def fk(self, q: Iterable[float], index: int|Iterable[int]|None=None,
           base: bool=False, tip: bool=False) -> NDArray:
        """
        T_n_in_0 = arm.fk(q, index=None, base=False, tip=False)

        Returns the transform from a specified frame to another given a set of
        joint angles q, the index of the starting and ending frames, and whether
        or not to include the base and tip transforms created in the constructor.

        :param Iterable[float] q: list or iterable of floats which represent the joint angles.
        :param int | Iterable[int] | None index: integer, list of two integers, or None.
            If an integer, it represents end_frame and start_frame is 0.
            If an iterable of two integers, they represent (start_frame, end_frame).
            If None, then start_frame is 0 and end_frame is n.
        :param bool base: specify whether to use the base transform (T_0_in_base) in the calculation.
            If start_frame is not 0, the frames do not line up and the base transform will not be used.
        :param bool tip: specify whether to use the tip transform (T_tip_in_n) in the calculation.
            If end_frame is not n, the frames do not line up and the tip transform will not be used.
        :return T: the 4 x 4 homogeneous transform between the specified frames.
        """
        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of it, but it is helpful to keep.

        if not hasattr(q, '__getitem__'):
            q = [q]

        if len(q) != self.n:
             raise ValueError(f"q must be the same size as the number of links! Expected {self.n}, got {len(q)}")

        if isinstance(index, int):
            start_frame = 0
            end_frame = index
        elif hasattr(index, '__getitem__'):
            start_frame = index[0]
            end_frame = index[1]
        elif index == None:
            start_frame = 0
            end_frame = self.n
        else:
            raise TypeError("Invalid index type!")

        if not (0 <= start_frame <= end_frame <= self.n):
            raise ValueError(f"Invalid index values! Must be 0 <= {start_frame} <= {end_frame} <= {self.n}")
        ###############################################################################################
        ###############################################################################################

        # TODO - Write code to calculate the total homogeneous transform "T" based on variables stored
        # in "base", "tip", "start_frame", and "end_frame". Look at the function definition if you are
        # unsure about the role of each of these variables. This is mostly easily done with some if/else
        # statements and a "for" loop to add the effect of each subsequent A_i(q_i). But you can
        # organize the code any way you like.

        T = np.eye(4)
        
        # multiply thgough the A_i matrices
        for i in range(start_frame, end_frame):
            T = T @ self.transforms[i](q[i])
            
        # include base and tip transforms if specified
        if base and start_frame == 0:
            T = self.base @ T
            
        if tip and end_frame == self.n:
            T = T @ self.tip

        return T
    
    def jacob(self, q: list[float]|NDArray, index: int|None=None, base: bool=False,
              tip: bool=False) -> NDArray:
        """
        J = arm.jacob(q)

        Calculates the geometric jacobian for a specified frame of the arm in a given configuration
        in the [linear; angular] format consistent with the Siciliano textbook.

        :param list[float] | NDArray q: joint positions
        :param int | None index: joint frame at which to calculate the Jacobian
        :param bool base: specify whether to include the base transform in the Jacobian calculation
        :param bool tip: specify whether to include the tip transform in the Jacobian calculation
        :return J: 6xN numpy array, geometric jacobian of the robot arm
        """

        if index is None:
            index = self.n
        assert 0 <= index <= self.n, 'Invalid index value!'
        
        if len(q) != self.n:
            raise ValueError(f"q must be the same size as the number of links! Expected {self.n}, got {len(q)}")

        # Initialize a zero matrix for the Jacobian of the correct size.
        J = np.zeros((6, self.n))

        # Calculate the transform to the end-effector to get its position.
        # This T is T_index_in_0 (or T_index_in_base if base=True)
        T_end_effector = self.fk(q, index=index, base=base, tip=tip)
        p_e = T_end_effector[:3, 3]

        # Loop through each joint to calculate its column in the Jacobian.
        # We only calculate for joints that affect the 'index' frame.
        for i in range(min(index, self.n)):
            
            # Get the transform from the base to the current joint's frame (frame i).
            # T_i is the transform T_i_in_0 (or T_i_in_base)
            T_i = self.fk(q, index=i, base=base) # T_i_in_0

            # Extract the z-axis and position of the origin of frame i.
            z_i = T_i[:3, 2]
            p_i = T_i[:3, 3]

            # Check if the joint is revolute ('r').
            if self.jt[i] == 'r':
                # Top 3 rows (linear velocity)
                J[:3, i] = np.cross(z_i, p_e - p_i)
                # Bottom 3 rows (angular velocity)
                J[3:, i] = z_i
            # Otherwise, the joint is prismatic ('p').
            else:
                # Top 3 rows (linear velocity)
                J[:3, i] = z_i
                # Bottom 3 rows (angular velocity)
                J[3:, i] = np.zeros(3)
        
        # If index < self.n, the trailing columns of J corresponding to
        # joints j >= index should be zero, which they are by initialization.
        return J

    # insert this function into your SerialArm class and complete it.
    # Please keep the function definition, and what it returns the same.
    def ik_position(self, target: NDArray, q0: list[float]|NDArray|None=None,
                    method: str='J_T', force: bool=True, tol: float=1e-4,
                    K: NDArray=None, kd: float=0.001, max_iter: int=100,
                    debug: bool=False, debug_step: bool=False
                    ) -> tuple[NDArray, NDArray, int, bool]:
        """
        qf, error_f, iters, converged = arm.ik_position(target, q0, 'J_T', K=np.eye(3))

        Computes the inverse kinematics solution (position only) for a given target
        position using a specified method by finding a set of joint angles that
        place the end effector at the target position without regard to orientation.

        :param NDArray target: 3x1 numpy array that defines the target location.
        :param list[float] | NDArray | None q0: list or array of initial joint positions,
            defaults to q0=0 (which is often a singularity - other starting positions
            are recommended).
        :param str method: select which IK algorithm to use. Options include:
            - 'pinv': damped pseudo-inverse solution, qdot = J_dag * e * dt, where
            J_dag = J.T * (J * J.T + kd**2)^-1
            - 'J_T': jacobian transpose method, qdot = J.T * K * e
        :param bool force: specify whether to attempt to solve even if a naive reach
            check shows the target is outside the reach of the arm.
        :param float tol: tolerance in the norm of the error in pose used as
            termination criteria for while loop.
        :param NDArray K: 3x3 numpy array. For both pinv and J_T, K is the positive
            definite gain matrix.
        :param float kd: used in the pinv method to make sure the matrix is invertible.
        :param int max_iter: maximum attempts before giving up.
        :param bool debug: specify whether to plot the intermediate steps of the algorithm.
        :param bool debug_step: specify whether to pause between each iteration when debugging.

        :return qf: 6x1 numpy array of final joint values. If IK fails to converge
            within the max iterations, the last set of joint angles is still returned.
        :return error_f: 3x1 numpy array of the final positional error.
        :return iters: int, number of iterations taken.
        :return converged: bool, specifies whether the IK solution converged within
            the max iterations.
        """
        ###############################################################################################
        # the following lines of code are data type and error checking. You don't need to understand
        # all of
        if isinstance(q0, np.ndarray):
            q = q0.copy() # Use a copy to avoid modifying the original
        elif q0 is None:
            q = np.array([0.0]*self.n)
        elif isinstance(q0, list):
            q = np.array(q0)
        else:
            raise TypeError("Invlid type for initial joint positions 'q0'")
        
        target = target.flatten() # Ensure target is 1D array

        # Try basic check for if the target is in the workspace.
        # This check is from base frame, assuming base is at origin
        base_pos = self.base[:3, 3]
        target_distance = np.linalg.norm(target - base_pos)
        
        target_in_reach = target_distance <= self.reach
        if not force:
            assert target_in_reach, "Target outside of reachable workspace!"
        if not target_in_reach:
            print(f"Warning: Target distance ({target_distance:.3f}) may be outside of arm reach ({self.reach:.3f}).")

        if K is None:
             K = np.eye(3)
        assert isinstance(K, np.ndarray), "Gain matrix 'K' must be provided as a numpy array"
        ###############################################################################################
        ###############################################################################################

        # you may want to define some functions here to help with operations that you will
        # perform repeatedly in the while loop below. Alternatively, you can also just define
        # them as class functions and use them as self.<function_name>.

        # for example:
        def get_error(q_current):
            # Get FK from base to tip
            T_tip_in_base = self.fk(q_current, base=True, tip=True)
            cur_position = T_tip_in_base[:3, 3]
            e = target - cur_position
            return e

        iters = 0
        error = get_error(q)

        while np.linalg.norm(error) > tol and iters < max_iter:

        # In this while loop you will update q for each iteration, and update, then
        # your error to see if the problem has converged. You may want to print the error
        # or the "count" at each iteration to help you see the progress as you debug.
        # You may even want to plot an arm initially for each iteration to make sure
        # it's moving in the right direction towards the target.
        
            # We need the Jacobian from the base frame, including the tip
            J = self.jacob(q, base=True, tip=True)
            Jv = J[:3, :]  # Linear (position) Jacobian

            # --- Damped Pseudo-Inverse Method ---
            if method == 'pinv':
                # J_dag = Jv.T * inv(Jv * Jv.T + kd**2 * I)
                # K gain matrix K is applied to the error
                term_to_invert = Jv @ Jv.T + (kd**2) * np.eye(3)
                J_dag = Jv.T @ np.linalg.inv(term_to_invert)
                delta_q = J_dag @ K @ error

            # --- Jacobian Transpose Method ---
            elif method == 'J_T':
                # Need a step size, alpha, for this to converge
                # A common way to set alpha:
                JvT_K_e = Jv.T @ K @ error
                num = error @ K @ Jv @ JvT_K_e
                den = JvT_K_e @ JvT_K_e
                
                # Avoid division by zero if JvT_K_e is zero
                if den < 1e-9:
                    alpha = 0.01 # small step
                else:
                    alpha = num / den
                
                # We cap alpha to prevent huge steps
                alpha = min(alpha, 0.5) 

                delta_q = alpha * JvT_K_e

            else:
                raise ValueError("Invalid method selected. Choose 'pinv' or 'J_T'.")
            
            # Update joint angles
            q += delta_q
            
            # (Optional but recommended) Enforce joint limits
            q = np.clip(q, self.qlim[0, :], self.qlim[1, :])

            # Update error & iter count for next loop
            error = get_error(q)
            iters += 1
            
            if debug:
                print(f"Iter: {iters}, Error: {np.linalg.norm(error):.6f}")
                # You could add a viz.update(q) call here if debugging
                if debug_step:
                    input("Press Enter to continue...")

            
        # when "while" loop is done, return the relevant info.
        return q, error, iters, np.linalg.norm(error) <= tol


if __name__ == "__main__":
    # from visualization import VizScene # Keep this import, but we'll use it in the new script
    np.set_printoptions(precision=4, suppress=True)

    # Defining a table of DH parameters where each row corresponds to another joint.
    # The order of the DH parameters is [theta, d, a, alpha] - which is the order of operations.
    # The symbolic joint variables "q" do not have to be explicitly defined here.
    # This is a two link, planar robot arm with two revolute joints.
    dh = [[0, 0, 0.3, 0],
          [0, 0, 0.3, 0]]
    
    # Example joint limits (min row, max row)
    qlim = np.array([[-np.pi, -np.pi],
                     [ np.pi,  np.pi]])

    # make robot arm (assuming all joints are revolute)
    arm = SerialArm(dh, joint_limits=qlim)

    # defining joint configuration
    q = [np.pi/4.0, np.pi/4.0]  # 45 degrees and 45 degrees

    # show an example of calculating the entire forward kinematics
    Tn_in_0 = arm.fk(q, base=True, tip=True)
    print("Tn_in_0:\n", Tn_in_0, "\n")

    # show an example of calculating the kinematics between frames 0 and 1
    T1_in_0 = arm.fk(q, index=1, base=True)
    print("T1_in 0:\n", T1_in_0, "\n")

    # showing how to use "print" with the arm object
    print(arm)

    # --- Example IK ---
    print("--- IK Example ---")
    target_pos = np.array([0.4, 0.2, 0.0])
    q_initial = np.array([0.0, 0.0])
    K_gain = np.eye(3) * 0.5

    print(f"Targeting: {target_pos}")
    
    q_final, final_err, iters, converged = arm.ik_position(
        target_pos, 
        q0=q_initial, 
        method='pinv', 
        K=K_gain,
        tol=1e-5,
        max_iter=200
    )
    
    print(f"IK Converged: {converged} in {iters} iterations.")
    print(f"Final q: {q_final}")
    print(f"Final error: {np.linalg.norm(final_err):.6f}")
    
    T_final = arm.fk(q_final, base=True, tip=True)
    print(f"Final Position: {T_final[:3, 3]}")


    # We will move the visualization to the main run_presentation.py script
    # so it's not needed here anymore.

    # # now visualizing the coordinate frames that we've calculated
    # viz = VizScene()

    # viz.add_frame(arm.base, label='base')
    # viz.add_frame(Tn_in_0, label="Tn_in_0")
    # viz.add_frame(T1_in_0, label="T1_in_0")

    # viz.hold()