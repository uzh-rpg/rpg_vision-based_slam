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
	results_dir = os.path.join(configs['full_batch_optimization_dir'], 'continuous_time')
	spline_knot_dt = str(int(1000 * configs['spline_control_nodes_dt_s']))
	order = str(configs['spline_order'])
	results_dir += '/order_' + order + '/dt_' + spline_knot_dt + '_ms'
	imu_meas_fn = configs['imu_fn']

	# Load data
	imu_meas = np.loadtxt(imu_meas_fn)

	init_spline = np.loadtxt(os.path.join(results_dir,'initial_spline.txt'))
	init_predicted_imu = np.loadtxt(os.path.join(results_dir,'initial_predicted_imu.txt'))
	init_acc_errs = np.loadtxt(os.path.join(results_dir,'initial_acc_errors.txt'))
	init_gyro_errs = np.loadtxt(os.path.join(results_dir,'initial_gyro_errors.txt'))
	init_reproj_errs = np.loadtxt(os.path.join(results_dir,'initial_reprojection_errors.txt'))
	init_scale = np.loadtxt(os.path.join(results_dir,'initial_scale.txt'))

	opti_spline = np.loadtxt(os.path.join(results_dir,'spline.txt'))
	opti_predicted_imu = np.loadtxt(os.path.join(results_dir,'predicted_imu.txt'))
	opti_acc_errs = np.loadtxt(os.path.join(results_dir,'acc_errors.txt'))
	opti_gyro_errs = np.loadtxt(os.path.join(results_dir,'gyro_errors.txt'))
	opti_reproj_errs = np.loadtxt(os.path.join(results_dir,'reprojection_errors.txt'))
	acc_bias_spline = np.loadtxt(os.path.join(results_dir,'acc_bias_spline.txt'))
	gyro_bias_spline = np.loadtxt(os.path.join(results_dir,'gyro_bias_spline.txt'))

	scale = np.loadtxt(os.path.join(results_dir,'scale.txt'))

	# Remove 0 errors
	init_reproj_errs = np.asarray([e for e in init_reproj_errs if e > 1e-6])
	opti_reproj_errs = np.asarray([e for e in opti_reproj_errs if e > 1e-6])

	# Remove outliers
	init_reproj_errs = np.asarray([e for e in init_reproj_errs if e < 10])
	opti_reproj_errs = np.asarray([e for e in opti_reproj_errs if e < 10])

	# Plots
	init_traj = [] # (time, x, y, z)
	for sample in init_spline:
		time_and_t_wb = np.array([sample[0], sample[1], sample[2], sample[3]])
		#q_wb = Quaternion(array=np.array([sample[7], sample[4], sample[5], sample[6]]))
		#R_wb = q_wb.rotation_matrix
		init_traj.append(time_and_t_wb)
	init_traj = np.asarray(init_traj)

	opti_traj = [] # (time, x, y, z)
	for sample in opti_spline:
		time_and_t_wb = np.array([sample[0], sample[1], sample[2], sample[3]])
		#q_wb = Quaternion(array=np.array([sample[7], sample[4], sample[5], sample[6]]))
		#R_wb = q_wb.rotation_matrix
		opti_traj.append(time_and_t_wb)
	opti_traj = np.asarray(opti_traj)

	plt.figure(0)
	plots.xyPlotViFusion('XY plot', init_traj[:, 1:3], 'initial trajectory', opti_traj[:, 1:3], 'optimized trajectory')

	# Imu measurements and predictions
	plt.figure(1)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 2:5], 
		init_predicted_imu[:, 0] - imu_meas[0, 1], init_predicted_imu[:, 1:4], 'Initial Gyro')
	plt.figure(2)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 5:], 
		init_predicted_imu[:, 0] - imu_meas[0, 1], init_predicted_imu[:, 4:], 'Initial Accelerations')

	plt.figure(3)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 2:5], 
		opti_predicted_imu[:, 0] - imu_meas[0, 1], opti_predicted_imu[:, 1:4], 
		'Final Gyro')
	plt.figure(4)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 5:], 
		opti_predicted_imu[:, 0] - imu_meas[0, 1], opti_predicted_imu[:, 4:], 
		'Final Accelerations')

	# Reprojection errors
	plt.figure(5)
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
	plt.figure(8)
	plots.plotBias(acc_bias_spline[:, 0] - acc_bias_spline[0, 0], acc_bias_spline[:, 1:], 'Acc. bias')
	plt.figure(9)
	plots.plotBias(gyro_bias_spline[:, 0] - gyro_bias_spline[0, 0], gyro_bias_spline[:, 1:], 'Gyro bias')

	plt.show()

	#IPython.embed()


if __name__ == '__main__':
	sys.argv = flags.FLAGS(sys.argv)
	run()

