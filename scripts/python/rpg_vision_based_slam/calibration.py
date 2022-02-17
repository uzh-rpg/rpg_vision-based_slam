'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import numpy as np

import rpg_vision_based_slam.pose


class NoDistortion(object):
    def __init__(self):
        self.k = 1.0

    def distort(self, p):
        return p * self.k

    def undistort(self, p):
        return p * self.k 


class RadialTangentialDistortion(object):
    def __init__(self, k):
        self.k = k


# Source:  https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cameras/include/aslam/cameras/implementation/EquidistantDistortion.hpp
class EquidistantDistortion(object):
    def __init__(self, k):
        self.k = k

    def distort(self, p):
        r = np.sqrt(p[0] * p[0] + p[1] * p[1])
        theta = np.arctan(r)
        theta2468 = np.zeros(4)
        theta2468[0] = theta * theta
        theta2468[1] = theta2468[0] * theta2468[0]
        theta2468[2] = theta2468[1] * theta2468[0]
        theta2468[3] = theta2468[1] * theta2468[1]
        thetad = theta * (1 + np.dot(self.k, theta2468))
        if r > 1e-8:
            return p * thetad / r
        else:
            return p

    def undistort(self, p):
        thetad = np.sqrt(p[0] * p[0] + p[1] * p[1])
        theta = thetad
        theta2468 = np.zeros(4)
        for _ in range(20, 0, -1):
            theta2468[0] = theta * theta
            theta2468[1] = theta2468[0] * theta2468[0]
            theta2468[2] = theta2468[1] * theta2468[0]
            theta2468[3] = theta2468[1] * theta2468[1]
            theta = thetad / (1 + np.dot(self.k, theta2468))
        scaling = np.tan(theta) / thetad
        return p * scaling


class CamCalibration(object):
    def __init__(self, f, c, distortion, shape, T_C_B, timeshift_cam_imu):
        self.f = f
        self.c = c
        self.distortion = distortion
        self.shape = shape
        self.T_C_B = T_C_B
        self.T_B_C = T_C_B.inverse()
        # t_imu = t_cam + t_shift
        self.timeshift_cam_imu = timeshift_cam_imu

    def project(self, p_C):
        if len(p_C.shape) > 1:
            assert len(p_C.shape) == 2
            prj = np.array([self.project(i) for i in p_C if i[2] > 0])
            if prj.size == 0:
                return np.array([])
            prj_rc = np.fliplr(prj)
            return np.array([i for i in prj_rc if 0 <= i[0] < self.shape[0] and
                             0 <= i[1] < self.shape[1]])
        assert len(p_C) == 3
        in_plane = p_C[:2] / p_C[2]
        dist = self.distortion.distort(in_plane)
        return self.f * dist + self.c


class ImuCalibration(object):
    def __init__(self, rate, acc_noise_density, acc_random_walk, \
        gyro_noise_density, gyro_random_walk):
        self.rate = rate # [Hz]
        
        self.acc_noise_density = acc_noise_density
        self.acc_random_walk = acc_random_walk
        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk

        self.g = 9.80665
        self.g_W = np.array([0., 0., self.g])

