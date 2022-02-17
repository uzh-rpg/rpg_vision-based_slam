'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''


# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv/blob/full-batch-optimization/python/uzh_fpv_open/calibration.py

import numpy as np


def get_best_yaw(C):
    '''
    maximize trace(Rz(theta) * C)
    '''
    assert C.shape == (3, 3)

    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)

    return theta


# Source: https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py
def associateTimestamps(first_stamps, second_stamps, offset=0.0, max_difference=1.0):
    """
    associate timestamps
    first_stamps, second_stamps: list of timestamps to associate
    Output:
    sorted list of matches (match_first_idx, match_second_idx)
    """
    potential_matches = [(abs(a - (b + offset)), idx_a, idx_b)
                         for idx_a, a in enumerate(first_stamps)
                         for idx_b, b in enumerate(second_stamps)
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()  # prefer the closest

    matches = []
    first_idxes = list(range(len(first_stamps)))
    second_idxes = list(range(len(second_stamps)))
    for diff, idx_a, idx_b in potential_matches:
        if idx_a in first_idxes and idx_b in second_idxes:
            first_idxes.remove(idx_a)
            second_idxes.remove(idx_b)
            matches.append((int(idx_a), int(idx_b)))

    matches.sort()
    return matches


# Source: https://github.com/uzh-rpg/rpg_trajectory_evaluation/blob/master/src/rpg_trajectory_evaluation/align_trajectory.py
def alignUmeyama(model, data, known_scale=False, yaw_only=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.
    model = s * R * data + t
    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type
    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)
    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)
    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if(np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    if yaw_only:
        rot_C = np.dot(data_zerocentered.transpose(), model_zerocentered)
        theta = get_best_yaw(rot_C)
        R = rot_z(theta)
    else:
        R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t

