'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv_open/blob/master/python/uzh_fpv/flags.py

import numpy as np
import os
import yaml

import calibration
import flags
import uzhfpv_flags
import pose


def importCamI(y, i):
    y_cam = y['cam%d' % i]
    T_C_I4 = np.array(y_cam['T_cam_imu'])
    T_C_B = pose.Pose(T_C_I4[:3, :3], T_C_I4[:3, 3:])
    timeshift_cam_imu = float(y_cam['timeshift_cam_imu'])
    dist = calibration.EquidistantDistortion(y_cam['distortion_coeffs'])
    intr = y_cam['intrinsics']
    shape = list(reversed(y_cam['resolution']))
    return calibration.CamCalibration(intr[:2], intr[2:], dist, shape, T_C_B, timeshift_cam_imu)


def readCamCalibration(cam_idx):
    calib_path = os.path.join(flags.datasetsPath(), uzhfpv_flags.calibRelativePath())
    f_calib = open(os.path.join(calib_path, 'camchain-imucam-..%s_calib_%s_imu.yaml' \
        % (uzhfpv_flags.envCamString(), uzhfpv_flags.sensorString())))
    y = yaml.load(f_calib)
    calibs = importCamI(y, cam_idx)
    f_calib.close()
    return calibs

