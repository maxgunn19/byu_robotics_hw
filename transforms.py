"""
Transforms Module - Contains code for to learn about rotations
and eventually homogenous transforms.

Empty outline derived from code written by John Morrell, former TA.
"""

import numpy as np
from numpy import sin, cos, sqrt
from numpy.typing import NDArray
from utility import clean_rotation_matrix


## 2D Rotations
def rot2(theta: float) -> NDArray:
    """
    R = rot2(th)

    :param float theta: angle of rotation (rad)
    :return R: 2x2 numpy array representing rotation in 2D by theta
    """

    ## TODO - Fill this out
    R = np.array([[cos(theta), -sin(theta)],
                  [sin(theta), cos(theta)]])
    return clean_rotation_matrix(R)


## 3D Transformations
def rotx(theta: float) -> NDArray:
    """
    R = rotx(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about x-axis by amount theta
    """
    ## TODO - Fill this out
    R = np.array([[1, 0, 0],
                  [0, cos(theta), -sin(theta)],
                  [0, sin(theta), cos(theta)]])

    return clean_rotation_matrix(R)


def roty(theta: float) -> NDArray:
    """
    R = roty(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about y-axis by amount theta
    """
    ## TODO - Fill this out
    R = np.array([[cos(theta), 0, sin(theta)],
                  [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])

    return clean_rotation_matrix(R)


def rotz(theta: float) -> NDArray:
    """
    R = rotz(theta)

    :param float theta: angle of rotation (rad)
    :return R: 3x3 numpy array representing rotation about z-axis by amount theta
    """
    ## TODO - Fill this out
    R = np.array([[cos(theta), -sin(theta), 0],
                    [sin(theta), cos(theta), 0],
                    [0, 0, 1]])

    return clean_rotation_matrix(R)


# inverse of rotation matrix
def rot_inv(R: NDArray) -> NDArray:
    '''
    R_inv = rot_inv(R)

    :param NDArray R: 2x2 or 3x3 numpy array representing a proper rotation matrix
    :return R_inv: 2x2 or 3x3 inverse of the input rotation matrix
    '''
    ## TODO - Fill this out
    R_inv = np.transpose(R)
    return R_inv

def se3(R: NDArray=np.eye(3), p: NDArray=np.zeros(3)) -> NDArray:
    """
    T = se3(R, p)

    Creates a 4x4 homogeneous transformation matrix "T" from a 3x3 rotation matrix
    and a position vector.

    :param NDArray R: 3x3 numpy array representing orientation, defaults to identity.
    :param NDArray p: numpy array representing position, defaults to [0, 0, 0].
    :return T: 4x4 numpy array representing the homogeneous transform.
    """
    # TODO - fill out "T"
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    T[3, 3] = 1

    return T

def inv(T: NDArray) -> NDArray:
    """
    T_inv = inv(T)

    Returns the inverse transform to T.

    :param NDArray T: 4x4 homogeneous transformation matrix
    :return T_inv: 4x4 numpy array that is the inverse to T so that T @ T_inv = I
    """

    #TODO - fill this out
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    R_inv = rot_inv(R)
    p_inv = -R_inv @ p
    T_inv = np.eye(4)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = p_inv
    T_inv[3, 3] = 1

    return T_inv
