'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

'''
Reference frames (spatial and temporal):
W: (fixed) inertial frame. Where spline is expressed in.
B: (moving) body (= imu) frame.
C: (moving) camera frame.
P: rigid point attached to B 
(e.g., position of the GPS antenna or nodal point of the prism tracked by a total station).
t_gp: time frame of the global position sensor.
t_s: (= t_c) time frame of the spline (= time frame of frames).
t_c: time frame of the camera.
'''


import os
import sys

import IPython
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pyquaternion import Quaternion
import yaml

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.plots as plots
import rpg_vision_based_slam.pose as pose

FLAGS = flags.FLAGS


def run():
	config_file = open(FLAGS.config)
	configs = yaml.load(config_file, Loader=yaml.FullLoader)
	res_dir = configs['colmap_spline_dir']
	spline_knot_dt = str(int(1000 * configs['spline_control_nodes_dt_s']))
	spline_order = str(int(configs['spline_order']))
	res_dir += '/colmap_fitted_spline/order_' + spline_order + '/dt_' + spline_knot_dt + '_ms'
	colmap_fn = configs['colmap_fn']

	# Load data
	spline_fn = os.path.join(res_dir, 'spline.txt')
	spline_knots_fn = os.path.join(res_dir, 'spline_knots.txt')
	spline = np.loadtxt(spline_fn)
	spline_knots = np.loadtxt(spline_knots_fn)
	colmap_traj = np.loadtxt(colmap_fn)

	# Plots
	fig = plt.figure('2D views')
	gs = gridspec.GridSpec(2, 2)
	fig.add_subplot(gs[:, 0])
	plots.xyPlot('XY plot', spline_knots[:, 1:3], 'knots', spline[:, 1:3], 'spline', colmap_traj[:, 1:3], 'colmap')
	fig.add_subplot(gs[0, 1])
	plots.xzPlot('XZ plot', spline_knots[:, [1,3]], 'knots', spline[:, [1,3]], 'spline', colmap_traj[:, [1,3]], 'colmap')
	fig.add_subplot(gs[1, 1])
	plots.yzPlot('YZ plot', spline_knots[:, [2,3]], 'knots', spline[:, [2,3]], 'spline', colmap_traj[:, [2,3]], 'colmap')

	# compute euler angles
	spline_euler_angles_zyx = pose.fromQuatToEulerAng(spline[:,4:])
	colmap_euler_angles_zyx = pose.fromQuatToEulerAng(colmap_traj[:,4:])

	plt.figure('Euler angles')
	plots.plotEulerAngles(spline[:,0], spline_euler_angles_zyx, 'spline', \
		colmap_traj[:,0], colmap_euler_angles_zyx, 'colmap')

	plt.show()

	# IPython.embed()


if __name__ == '__main__':
	sys.argv = flags.FLAGS(sys.argv)
	run()

