'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv_open/blob/master/python/uzh_fpv/pose.py


import numpy as np
import pyquaternion
import scipy.linalg

import rpg_vision_based_slam.transformations as tf


def cross2Matrix(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def matrix2Cross(M):
    skew = (M - M.T)/2
    return np.array([-skew[1, 2], skew[0, 2], -skew[0, 1]])


class Pose(object):
    def __init__(self, R, t):
        assert type(R) is np.ndarray
        assert type(t) is np.ndarray
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        self.R = R
        self.t = t

    def inverse(self):
        return Pose(self.R.T, -np.dot(self.R.T, self.t))

    def __mul__(self, other):
        if isinstance(other, Pose):
            return Pose(np.dot(self.R, other.R), np.dot(self.R, other.t) + self.t)
        if type(other) is np.ndarray:
            assert len(other.shape) == 2
            assert other.shape[0] == 3
            return np.dot(self.R, other) + self.t
        raise Exception('Multiplication with unknown type!')

    def asArray(self):
        return np.vstack((np.hstack((self.R, self.t)), np.array([0, 0, 0, 1])))
        
    def asTwist(self):
        so_matrix = scipy.linalg.logm(self.R)
        if np.sum(np.imag(so_matrix)) > 1e-10:
            raise Exception('logm called for a matrix with angle Pi. ' +
                'Not defined! Consider using another representation!')
        so_matrix = np.real(so_matrix)
        return np.hstack((np.ravel(self.t), matrix2Cross(so_matrix)))

    def q_wxyz(self):
        return pyquaternion.Quaternion(matrix=self.R).unit.q

    def fix(self):
        self.R = fixRotationMatrix(self.R)

    def fixed(self):
        return Pose(fixRotationMatrix(self.R), self.t)

    def __repr__(self):
        return self.asArray().__repr__()


def fromTwist(twist):
    # Using Rodrigues' formula
    w = twist[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-6:
        return Pose(np.eye(3), twist[:3].reshape(3, 1))
    M = cross2Matrix(w/theta)
    R = np.eye(3) + M * np.sin(theta) + np.dot(M, M) * (1 - np.cos(theta))
    return Pose(R, twist[:3].reshape((3, 1)))


def fromPositionAndQuaternion(xyz, q_wxyz):
    R = pyquaternion.Quaternion(
        q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]).rotation_matrix
    t = xyz.reshape(3, 1)
    return Pose(R, t)


# ROS geometry_msgs/Pose
def fromPoseMessage(pose_msg):
    pos = pose_msg.position
    ori = pose_msg.orientation
    R = pyquaternion.Quaternion(ori.w, ori.x, ori.y, ori.z).rotation_matrix
    t = np.array([pos.x, pos.y, pos.z]).reshape(3, 1)
    return Pose(R, t)


def identity():
    return Pose(np.eye(3), np.zeros((3, 1)))


def geodesicDistanceSO3(R1, R2):
    return getRotationAngle(np.dot(R1, R2.T))


def getRotationAngle(R):
    return np.arccos((np.trace(R) - 1 - 1e-6) / 2)


def fixRotationMatrix(R):
    u, _, vt = np.linalg.svd(R)
    R_new = np.dot(u, vt)
    if np.linalg.det(R_new) < 0:
        R_new = -R_new
    return R_new


def similarityTransformation(T_A_B, p_B_C, scale):
    return scale * np.dot(T_A_B.R, p_B_C) + T_A_B.t # 3x1 np array


def similarityTransformationPose(T_A_B, T_B_C, scale):
    R_A_C = np.dot(T_A_B.R, T_B_C.R)
    t_A_C = scale * np.dot(T_A_B.R, T_B_C.t) + T_A_B.t
    return Pose(R_A_C, t_A_C)


def loadFromTxt(fn):
    T = np.loadtxt(fn)
    return Pose(T[0:3, 0:3], np.array(T[0:3, 3]).reshape(3,1))


# [in] = np.array([[qx, qy, qz, qw], [], ...])
# [out] = np.array([[rotz, roty, rotx], [], ...])
def fromQuatToEulerAng(quats):
    rot_zyx = []
    for q in quats:
        R = pyquaternion.Quaternion(np.array([q[3], q[0], q[1], q[2]])).rotation_matrix
        rz, ry, rx = tf.euler_from_matrix(R, 'rzyx')
        rot_zyx.append(np.array([rz, ry, rx]))
    rot_zyx = np.asarray(rot_zyx)
    rot_zyx = np.rad2deg(rot_zyx)
    return rot_zyx

