'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import math
import os
import sys

import IPython
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
import yaml

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.plots as plots
import rpg_vision_based_slam.pose as pose

FLAGS = flags.FLAGS

# W: fixed world frame. Global measurements are expressed in this frame.
# G: fixed colmap frame. Spline is expressed in this frame.


def run():
    config_file = open(FLAGS.config)
    configs = yaml.load(config_file, Loader=yaml.FullLoader)
    g_meas_fn = configs['gp_fn']
    order = str(configs['spline_order'])
    spline_knot_dt = str(int(1000 * configs['spline_control_nodes_dt_s']))
    res_dir = configs['spline_global_ref_alignment_dir']
    res_dir += '/order_' + order + '/dt_' + spline_knot_dt + '_ms'
    spline_fn = os.path.join(res_dir, 'spline.txt')

    # Load measurements
    g_meas = np.loadtxt(g_meas_fn)
    g_ts = g_meas[:, 0]
    g_pos = g_meas[:, 1:]

    spline = np.loadtxt(spline_fn)
    spline_ts = spline[:, 0]
    spline_poses = [pose.Pose(Quaternion(array=np.array([t[7], t[4], t[5], t[6]])).rotation_matrix, 
        np.array([t[1], t[2], t[3]]).reshape(3,1)) for t in spline]

    scale_init = np.loadtxt(os.path.join(res_dir, 'scale_init.txt'))
    T_wg_init = pose.loadFromTxt(os.path.join(res_dir, 'T_wg_init.txt'))
    p_bp_int = np.array([0., 0., 0.])

    scale = np.loadtxt(os.path.join(res_dir, 'scale.txt'))
    T_wg = pose.loadFromTxt(os.path.join(res_dir, 'T_wg.txt'))
    p_bp = np.loadtxt(os.path.join(res_dir, 'p_bp.txt'))

    # Plot
    spline_poses_init_align = [pose.similarityTransformationPose(T_wg_init, T, scale_init) \
        for T in spline_poses]
    g_pred_init_align = np.asarray([(pose.similarityTransformation(T, p_bp_int.reshape(3,1), scale_init)).ravel() \
        for T in spline_poses_init_align])

    spline_poses_align = [pose.similarityTransformationPose(T_wg, T, scale) \
        for T in spline_poses]
    g_pred_align = np.asarray([(pose.similarityTransformation(T, p_bp.reshape(3,1), scale)).ravel() \
        for T in spline_poses_align])

    plt.figure(0)
    plots.xyPlot('Global Frame - Colmap alignment', g_pos[:, 0:2], 'global measurements', \
        g_pred_init_align[:, 0:2], 'initial alignment', g_pred_align[:, 0:2], 'final alignment')
    plt.show()


if __name__ == '__main__':
	sys.argv = flags.FLAGS(sys.argv)
	run()

