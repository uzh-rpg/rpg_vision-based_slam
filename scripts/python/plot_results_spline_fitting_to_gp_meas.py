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
import numpy as np
from pyquaternion import Quaternion
import yaml

import rpg_vision_based_slam.plots as plots
import rpg_vision_based_slam.flags as flags

FLAGS = flags.FLAGS


def run():
	config_file = open(FLAGS.config)
	configs = yaml.load(config_file, Loader=yaml.FullLoader)
	res_dir = configs['spline_dir']
	spline_knot_dt = str(int(1000 * configs['spline_control_nodes_dt_s']))
	res_dir += '/fit_spline_on_gp_meas/dt_' + spline_knot_dt + '_ms'
	gp_fn = configs['gp_fn']

	# Load data
	spline_fn = os.path.join(res_dir, 'spline.txt')
	spline_knots_fn = os.path.join(res_dir, 'spline_knots.txt')
	spline = np.loadtxt(spline_fn)
	spline_knots = np.loadtxt(spline_knots_fn)
	gp_meas = np.loadtxt(gp_fn)

	# Plots
	plt.figure(0)
	plots.xyPlot('XY plot', gp_meas[:, 1:3], 'gp meas.', spline[:, 1:3], 'fitted spline')

	plt.show()

	# IPython.embed()


if __name__ == '__main__':
	sys.argv = flags.FLAGS(sys.argv)
	run()

