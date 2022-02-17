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
	if FLAGS.results_dir is not '':
		results_dir = FLAGS.results_dir
		print('Reading results from %s' % results_dir)
	gp_meas_fn = configs['gp_fn']
	imu_meas_fn = configs['imu_fn']

	# Load data
	gp_meas = np.loadtxt(gp_meas_fn)
	imu_meas = np.loadtxt(imu_meas_fn)

	init_spline = np.loadtxt(os.path.join(results_dir,'initial_spline.txt'))
	init_predicted_imu = np.loadtxt(os.path.join(results_dir,'initial_predicted_imu.txt'))
	init_acc_errs = np.loadtxt(os.path.join(results_dir,'initial_acc_errors.txt'))
	init_gyro_errs = np.loadtxt(os.path.join(results_dir,'initial_gyro_errors.txt'))
	init_gp_errs = np.loadtxt(os.path.join(results_dir,'initial_global_position_errors.txt'))
	init_reproj_errs = np.loadtxt(os.path.join(results_dir,'initial_reprojection_errors.txt'))
	init_p_bp = np.loadtxt(os.path.join(results_dir,'initial_p_bp.txt'))
	init_scale = np.loadtxt(os.path.join(results_dir,'initial_scale.txt'))

	opti_spline = np.loadtxt(os.path.join(results_dir,'spline.txt'))
	opti_predicted_imu = np.loadtxt(os.path.join(results_dir,'predicted_imu.txt'))
	opti_acc_errs = np.loadtxt(os.path.join(results_dir,'acc_errors.txt'))
	opti_gyro_errs = np.loadtxt(os.path.join(results_dir,'gyro_errors.txt'))
	opti_gp_errs = np.loadtxt(os.path.join(results_dir,'global_position_errors.txt'))
	opti_reproj_errs = np.loadtxt(os.path.join(results_dir,'reprojection_errors.txt'))
	acc_bias_spline = np.loadtxt(os.path.join(results_dir,'acc_bias_spline.txt'))
	gyro_bias_spline = np.loadtxt(os.path.join(results_dir,'gyro_bias_spline.txt'))
	opti_p_bp = np.loadtxt(os.path.join(results_dir,'p_bp.txt'))
	opti_t_offset_s_gp = np.loadtxt(os.path.join(results_dir, 'spline_globalsensor_time_offset.txt'))

	scale = np.loadtxt(os.path.join(results_dir,'scale.txt'))

	# Remove 0 errors
	init_reproj_errs = np.asarray([e for e in init_reproj_errs if e > 1e-6])
	opti_reproj_errs = np.asarray([e for e in opti_reproj_errs if e > 1e-6])

	# Remove outliers
	init_reproj_errs = np.asarray([e for e in init_reproj_errs if e < 10])
	opti_reproj_errs = np.asarray([e for e in opti_reproj_errs if e < 10])

	# Plots
	# Global positional (gp) measurements and predictions
	init_gp_predictions = [] # (time, x, y, z)
	for sample in init_spline:
		t_wb = np.array([sample[1], sample[2], sample[3]])
		q_wb = Quaternion(array=np.array([sample[7], sample[4], sample[5], sample[6]]))
		R_wb = q_wb.rotation_matrix
		t_wp = init_scale * np.dot(R_wb, init_p_bp) + t_wb
		prediction = np.array([sample[0], t_wp[0], t_wp[1], t_wp[2]])
		init_gp_predictions.append(prediction)
	init_gp_predictions = np.asarray(init_gp_predictions)

	opti_gp_predictions = [] # (time, x, y, z)
	for sample in opti_spline:
		t_wb = np.array([sample[1], sample[2], sample[3]])
		q_wb = Quaternion(array=np.array([sample[7], sample[4], sample[5], sample[6]]))
		R_wb = q_wb.rotation_matrix
		t_wp = scale * np.dot(R_wb, opti_p_bp) + t_wb
		prediction = np.array([sample[0], t_wp[0], t_wp[1], t_wp[2]])
		opti_gp_predictions.append(prediction)
	opti_gp_predictions = np.asarray(opti_gp_predictions)

	plt.figure(0)
	plots.xyPlot('XY plot', gp_meas[:, 1:3], 'global pos sens', init_gp_predictions[:, 1:3], 'initial predictions', 
		opti_gp_predictions[:, 1:3], 'optimized predictions')

	plt.figure(1)
	gp_init_pred_times = np.asarray([m[0] for m in init_gp_predictions])
	gp_pred_times = np.asarray([m[0] for m in opti_gp_predictions])
	spline_times = np.asarray([m[0] for m in init_spline])
	gp_times = np.asarray([(m[0] - opti_t_offset_s_gp) for m in gp_meas if (m[0] - opti_t_offset_s_gp) > spline_times[0]])
	gp_times_0 = gp_times - gp_times[0]
	plots.xyztPlot('XYZt', gp_times_0, gp_meas[(gp_meas.shape[0] - gp_times.shape[0]):, 1:4], 'global pos sens', 
		gp_init_pred_times - gp_times[0], init_gp_predictions[:, 1:], 'initial predictions', 
		gp_pred_times - gp_times[0], opti_gp_predictions[:, 1:], 'optimized predictions')

	# Imu measurements and predictions
	plt.figure(2)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 2:5], 
		init_predicted_imu[:, 0] - imu_meas[0, 1], init_predicted_imu[:, 1:4], 'Initial Gyro')
	plt.figure(3)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 5:], 
		init_predicted_imu[:, 0] - imu_meas[0, 1], init_predicted_imu[:, 4:], 'Initial Accelerations')

	plt.figure(4)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 2:5], 
		opti_predicted_imu[:, 0] - imu_meas[0, 1], opti_predicted_imu[:, 1:4], 
		'Final Gyro')
	plt.figure(5)
	plots.imuMeasPredPlot(imu_meas[:, 1] - imu_meas[0, 1], imu_meas[:, 5:], 
		opti_predicted_imu[:, 0] - imu_meas[0, 1], opti_predicted_imu[:, 4:], 
		'Final Accelerations')

	# Reprojection errors
	plt.figure(6)
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

	# Global positional errors
	plt.figure(7)
	plots.cumulativeErrorPlot(init_gp_errs, 'initial', opti_gp_errs, 'optimized', 'Global Position', 'Cumulative distribution')
	print('-- Global Position errors --')
	print('Before optimization')
	print('mean: %.6f' % np.mean(init_gp_errs))
	print('std: %.6f' % np.std(init_gp_errs))
	print('After optimization')
	print('mean: %.6f' % np.mean(opti_gp_errs))
	print('std: %.6f' % np.std(opti_gp_errs))
	print('percentile 50: %.3f' % np.percentile(opti_gp_errs, 50))
	print('percentile 60: %.3f' % np.percentile(opti_gp_errs, 60))
	print('percentile 70: %.3f' % np.percentile(opti_gp_errs, 70))
	print('percentile 80: %.3f' % np.percentile(opti_gp_errs, 80))
	print('percentile 90: %.3f' % np.percentile(opti_gp_errs, 90))
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

