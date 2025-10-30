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
from typing import Callable, Iterable, Union


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
    :param str jt: joint type: 'r' for revolute, 'p' for prismatic
    """

    # error checking
    if len(dh) != 4:
        raise ValueError("dh parameter list must be length 4")
    if jt not in ('r', 'p'):
        raise ValueError("joint type must be 'r' or 'p'")

    # "T_z(theta) * T_z(d) * T_x(a) * T_x(alpha)"
    # order of operations for transforms:
    # 1. rotate about z_i-1 by theta_i
    # 2. translate along z_i-1 by d_i
    # 3. translate along x_i by a_i
    # 4. rotate about x_i by alpha_i

    # here is a function that creates the A matrix, given a joint variable "q"
    def A(q: float) -> NDArray:
        # T_z(theta)
        if jt == 'r':  # theta = q
            th = dh[0] + q
            d = dh[1]
        else:  # d = q
            th = dh[0]
            d = dh[1] + q
        
        a = dh[2]
        al = dh[3]
        
        # C = cos, S = sin
        Cth = np.cos(th)
        Sth = np.sin(th)
        Cal = np.cos(al)
        Sal = np.sin(al)

        # return T_i_in_i-1
        return np.array([
            [Cth, -Sth * Cal,  Sth * Sal, a * Cth],
            [Sth,  Cth * Cal, -Cth * Sal, a * Sth],
            [0,         Sal,         Cal,        d],
            [0,           0,           0,        1]
        ])

    return A


class SerialArm:
    """
    This class creates a serial robot arm that can calculate forward kinematics (fk),
    inverse kinematics (ik), and the Jacobian (jacob).
    """

    def __init__(self, dh: list[list[float]], jt: Union[str, Iterable[str]] = 'r'):
        """
        :param list[list[float]] dh: table of DH parameters, where each row is a list of
            4 parameters for a single joint/link: [theta, d, a, alpha]
        :param str | Iterable[str] jt: joint type for each joint. If a single string
            (e.g. 'r') is given, it is assumed all joints are of that type. Otherwise,
            it must be an iterable (e.g. a list or tuple) of the same length as dh,
            specifying the type of each joint.
        """
        self.dh = dh
        self.n = len(dh)

        if isinstance(jt, str):
            self.jt = [jt] * self.n
        elif len(jt) == self.n:
            self.jt = jt
        else:
            raise ValueError("jt must be a string or an iterable of length n")

        # create a list of functions, where each function A_i(q_i) returns T_i_in_i-1
        self.A = [dh2A(self.dh[i], self.jt[i]) for i in range(self.n)]

    def fk(self, q: list[float], index: Union[list[int], int] = -1) -> NDArray:
        """
        Calculate the forward kinematics for a given joint configuration.

        :param list[float] q: list of n joint variables
        :param list[int] | int index: which transform to return.
            If -1 (default), return the transform from base to end-effector (T_n_in_0)
            If 0, return T_0_in_0 (identity)
            If n, return T_n_in_0 (same as -1)
            If [i, j], return T_j_in_i (transform from frame i to frame j)
        :return: 4x4 homogeneous transform matrix
        """

        if len(q) != self.n:
            raise ValueError(f"joint variable list 'q' must be length {self.n}")

        # T_i_in_k = A_k+1(q_k+1) * A_k+2(q_k+2) * ... * A_i(q_i)

        if isinstance(index, int):
            if index == -1:
                i = 0
                j = self.n
            elif index == 0:
                return np.eye(4)
            elif index > 0 and index <= self.n:
                i = 0
                j = index
            else:
                raise ValueError("index must be -1, 0, or a positive integer <= n")
        elif isinstance(index, (list, tuple)):
            if len(index) == 2:
                i, j = index
            else:
                raise ValueError("if index is a list, it must be of length 2: [i, j]")
        else:
            raise ValueError("index must be an int or a list of length 2")

        if j < i:
            raise ValueError("j must be greater than or equal to i")
        
        T = np.eye(4)
        for k in range(i, j):
            T = T @ self.A[k](q[k])

        return T

    def jacob(self, q: list[float], index: int = -1) -> NDArray:
        """
        Calculates the Jacobian matrix for the end-effector.
        [linear velocity; angular velocity]

        :param list[float] q: list of n joint variables
        :param int index: which frame to calculate the Jacobian for.
            If -1 (default), return the Jacobian for the end-effector (frame n)
            Otherwise, return the Jacobian for frame `index`.
        :return: 6xn Jacobian matrix
        """
        if len(q) != self.n:
            raise ValueError(f"joint variable list 'q' must be length {self.n}")
        
        if index == -1:
            index = self.n
        elif index < 0 or index > self.n:
            raise ValueError(f"index must be -1 or between 0 and {self.n}")

        J = np.zeros((6, self.n))
        
        # T_i_in_0
        T = np.eye(4)
        
        # list of T_i_in_0, for i=0 to n
        T_list = [T]
        for i in range(self.n):
            T = T @ self.A[i](q[i])
            T_list.append(T)

        # T_n_in_0
        Tn_in_0 = T_list[index]
        pn = Tn_in_0[0:3, 3]

        for i in range(self.n):
            # if i > index, this joint doesn't affect frame `index`
            if i >= index:
                J[:, i] = np.zeros(6)
                continue
                
            T_i_in_0 = T_list[i]
            z_i = T_i_in_0[0:3, 2]
            p_i = T_i_in_0[0:3, 3]

            if self.jt[i] == 'r':
                J[0:3, i] = np.cross(z_i, pn - p_i)  # linear
                J[3:6, i] = z_i                     # angular
            else:
                J[0:3, i] = z_i                     # linear
                J[3:6, i] = np.zeros(3)             # angular

        return J

    def ik(self, T_des: NDArray, q0: list[float] = None, method: str = 'newton',
           tol: float = 1e-6, max_iter: int = 100, alpha: float = 0.1) -> tuple[list, NDArray, int, bool]:
        """
        Calculates the inverse kinematics for a given desired end-effector pose.

        :param NDArray T_des: 4x4 desired homogeneous transform matrix (T_n_in_0)
        :param list[float] q0: initial guess for joint variables. If None, uses zeros.
        :param str method: 'newton' (default) or 'transpose'
        :param float tol: tolerance for error (norm of error vector)
        :param int max_iter: maximum number of iterations
        :param float alpha: step size (learning rate)
        :return: tuple of (q, error, iterations, success)
            q: list of n joint variables
            error: 6x1 error vector (final)
            iterations: number of iterations performed
            success: bool, True if error < tol
        """

        if q0 is None:
            q = np.zeros(self.n)
        elif len(q0) == self.n:
            q = np.array(q0, dtype=float)
        else:
            raise ValueError(f"initial guess 'q0' must be length {self.n}")

        if method not in ('newton', 'transpose'):
            raise ValueError("method must be 'newton' or 'transpose'")

        # T_current
        T_curr = self.fk(q)

        # error transform
        T_err = np.linalg.inv(T_curr) @ T_des

        # translation error
        err_p = T_curr[0:3, 3] - T_des[0:3, 3]

        # rotation error
        err_o = 0.5 * (np.cross(T_curr[0:3, 0], T_des[0:3, 0]) +
                       np.cross(T_curr[0:3, 1], T_des[0:3, 1]) +
                       np.cross(T_curr[0:3, 2], T_des[0:3, 2]))
        
        error = np.concatenate((err_p, err_o))
        
        iters = 0
        while np.linalg.norm(error) > tol and iters < max_iter:
            
            # calculate Jacobian
            J = self.jacob(q)

            # update q
            if method == 'newton':
                # q_new = q - alpha * J_pinv * error
                J_pinv = np.linalg.pinv(J)
                q = q - alpha * J_pinv @ error
            else: # transpose
                # q_new = q - alpha * J_T * error
                q = q - alpha * J.T @ error

            # update T_current
            T_curr = self.fk(q)

            # update error
            err_p = T_curr[0:3, 3] - T_des[0:3, 3]
            err_o = 0.5 * (np.cross(T_curr[0:3, 0], T_des[0:3, 0]) +
                           np.cross(T_curr[0:3, 1], T_des[0:3, 1]) +
                           np.cross(T_curr[0:3, 2], T_des[0:3, 2]))
            error = np.concatenate((err_p, err_o))
            
            iters += 1
            
        # when "while" loop is done, return the relevant info.
        return q, error, iters, iters < max_iter


if __name__ == "__main__":
    from visualization import VizScene
    np.set_printoptions(precision=4, suppress=True)

    # Defining a table of DH parameters where each row corresponds to another joint.
    # The order of the DH parameters is [theta, d, a, alpha] - which is the order of operations.
    # The symbolic joint variables "q" do not have to be explicitly defined here.
    # This is a two link, planar robot arm with two revolute joints.
    dh = [[0, 0, 0.3, 0],
          [0, 0, 0.3, 0]]

    # make robot arm (assuming all joints are revolute)
    arm = SerialArm(dh)

    # defining joint configuration
    q = [np.pi/4.0, np.pi/4.0]  # 45 degrees and 45 degrees

    # show an example of calculating the entire forward kinematics
    Tn_in_0 = arm.fk(q)
    print("Tn_in_0:\n", Tn_in_0, "\n")

    # show an example of calculating the kinematics between frames 0 and 1
    T1_in_0 = arm.fk(q, index=[0,1])
    print("T1_in_0:\n", T1_in_0, "\n")

    # show an example of calculating the jacobian
    J = arm.jacob(q)
    print("J:\n", J, "\n")

