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
	results_dir = os.path.join(configs['full_batch_optimization_dir'], 'discrete_time')
	imu_meas_fn = configs['imu_fn']

	# Load data
	imu_meas = np.loadtxt(imu_meas_fn)

	init_traj = np.loadtxt(os.path.join(results_dir,'initial_trajectory.txt'))
	init_reproj_errs = np.loadtxt(os.path.join(results_dir,'initial_reprojection_errors.txt'))
	init_scale = np.loadtxt(os.path.join(results_dir,'initial_scale.txt'))

	opti_traj = np.loadtxt(os.path.join(results_dir,'trajectory.txt'))
	opti_reproj_errs = np.loadtxt(os.path.join(results_dir,'reprojection_errors.txt'))
	acc_bias = np.loadtxt(os.path.join(results_dir,'acc_bias.txt'))
	gyro_bias = np.loadtxt(os.path.join(results_dir,'gyro_bias.txt'))
	scale = np.loadtxt(os.path.join(results_dir,'scale.txt'))

	# Remove 0 errors
	init_reproj_errs = np.asarray([e for e in init_reproj_errs if e > 1e-6])
	opti_reproj_errs = np.asarray([e for e in opti_reproj_errs if e > 1e-6])

	# Remove outliers
	init_reproj_errs = np.asarray([e for e in init_reproj_errs if e < 10])
	opti_reproj_errs = np.asarray([e for e in opti_reproj_errs if e < 10])

	# Plots
	plt.figure(0)
	plots.xyPlotViFusion('XY plot', init_traj[:, 1:3], 'initial traj', opti_traj[:, 1:3], 'optimized traj')

	# Reprojection errors
	plt.figure(1)
	plots.compareHistogramsErrorPlot(init_reproj_errs, opti_reproj_errs, 'Reprojection', \
		'Comparison Initial - Optimized Reprojection Errors')
	print('-- Reprojection errors --')
	print('Before optimization')
	print('n. errs: %d' % len(init_reproj_errs))
	print('mean: %.6f' % np.mean(init_reproj_errs))
	print('std: %.6f' % np.std(init_reproj_errs))
	print('After optimization')
	print('n. errs: %d' % len(opti_reproj_errs))
	print('mean: %.6f' % np.mean(opti_reproj_errs))
	print('std: %.6f' % np.std(opti_reproj_errs))
	print('\n')

	# Imu biases
	plt.figure(4)
	plots.plotBias(acc_bias[:, 0], acc_bias[:, 1:], 'Acc. bias')
	plt.figure(5)
	plots.plotBias(gyro_bias[:, 0], gyro_bias[:, 1:], 'Gyro bias')

	plt.show()

	#IPython.embed()


if __name__ == '__main__':
	sys.argv = flags.FLAGS(sys.argv)
	run()

