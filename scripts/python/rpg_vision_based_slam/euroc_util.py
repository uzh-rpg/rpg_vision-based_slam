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
import euroc_flags
import pose


def importCamI(y, i):
    y_cam = y['cameras'][i]
    T_B_C_arr = y_cam['T_B_C']['data']
    T_B_C = pose.Pose( 
        np.array([[T_B_C_arr[0], T_B_C_arr[1], T_B_C_arr[2]], 
            [T_B_C_arr[4], T_B_C_arr[5], T_B_C_arr[6]], 
            [T_B_C_arr[8], T_B_C_arr[9], T_B_C_arr[10]]]), 
        np.array([T_B_C_arr[3], T_B_C_arr[7], T_B_C_arr[11]]).reshape(3,1) 
        )
    T_C_B = T_B_C.inverse()
    y_camcal = y_cam['camera']
    # timeshift not used here
    timeshift_cam_imu = 0.0 
    dist = calibration.RadialTangentialDistortion(y_camcal['distortion']['parameters']['data'])
    intr = y_camcal['intrinsics']['data']
    shape = [y_camcal['image_height'], y_camcal['image_width']]
    return calibration.CamCalibration(intr[:2], intr[2:], dist, shape, T_C_B, timeshift_cam_imu)


def readCamCalibration(cam_idx):
    calib_path = os.path.join(flags.datasetsPath(), euroc_flags.calibRelativePath())

    f_calib = open(os.path.join(calib_path, 'calib.yaml'))
    y = yaml.load(f_calib)
    calibs = importCamI(y, cam_idx)
    f_calib.close()
    return calibs

