'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import os
import sys
import numpy as np
import yaml

import rpg_vision_based_slam.calibration as calibration
import rpg_vision_based_slam.pose as pose


def writeTrajEstToTxt(out_file, timestamps, pose_list):
    assert len(timestamps) == len(pose_list)
    
    f = open(out_file, 'w')
    f.write('# timestamp tx ty tz qx qy qz qw\n')
    for i in range(len(timestamps)):
        ts = timestamps[i]
        t = pose_list[i].t
        q = pose_list[i].q_wxyz()
        f.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
            (ts, t[0][0], t[1][0], t[2][0], q[1], q[2], q[3], q[0]))
    f.close()

    print("Written %d estimates to %s." % (len(timestamps), out_file))


def readImageIdAndTimestamps(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ids = []
    timestamps = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ids.append(int(line.split(' ')[0]))
            timestamps.append(float(line.split(' ')[1]))
    f_img_infos.close()
    return ids, timestamps


def readImageIdTimestampAndName(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ids = []
    timestamps = []
    names = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ids.append(int(line.split(' ')[0]))
            timestamps.append(float(line.split(' ')[1]))
            names.append(line.split(' ')[2][4:])
    f_img_infos.close()
    return ids, timestamps, names


def importCamUZHFPV(y, i):
    assert False, "Not implemented!"


def importCamEuRoC(y, i):
    y_cam = y['cameras'][i]['camera']
    distortion_coeffs = y_cam['distortion']['parameters']['data']
    dist = calibration.EquidistantDistortion(distortion_coeffs)
    intr = y_cam['intrinsics']['data']
    shape = [y_cam['image_height'], y_cam['image_width']]

    y_trans = y['cameras'][i]['T_B_C'] 
    T_B_C_arr = np.array(y_trans['data']).reshape((4,4))
    T_B_C = pose.Pose(T_B_C_arr[:3, :3], T_B_C_arr[:3, 3:])
    T_C_B = T_B_C.inverse()

    y_imu = y['imu_params']
    timeshift_cam_imu = -1.0 * float(y_imu['delay_imu_cam'])

    return calibration.CamCalibration(intr[:2], intr[2:], dist, shape, T_C_B, timeshift_cam_imu)


def readCamCalibration(dataset, calib_file, cam_idx):
    f_calib = open(calib_file)
    y = yaml.load(f_calib)
    if dataset == 'UZH_FPV':
        calibs = importCamUZHFPV(y, cam_idx)
    elif dataset == 'EuRoC' or dataset == 'Custom':
        calibs = importCamEuRoC(y, cam_idx)
    f_calib.close()
    return calibs

