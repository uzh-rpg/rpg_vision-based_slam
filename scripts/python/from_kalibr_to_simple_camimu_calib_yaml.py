'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

'''
Transform Kalibr camimu calib yaml in a simpler (namely cpp friendly) version that can be read by rpg_vision_based_slam
'''

import argparse
import os
import sys

import numpy as np
import yaml


def run(args):
	n_cam = args.n_cam
	print("Load / write %d cameras." % n_cam)

	# load kalibr camimu .yaml config file
	kalibr_yaml = yaml.load(open(args.kalibr_yaml_fn))

	T_cam_imu = []
	camera_model = []
	distortion_coeffs = []
	distortion_model = []
	intrinsics = []
	resolution = []
	timeshift_cam_imu = []

	for i in range(n_cam):
		cam_i_yaml = kalibr_yaml['cam%d' % i]
		T_cam_imu_nested_lists = cam_i_yaml['T_cam_imu']
		T_cam_imu.append([item for sublist in T_cam_imu_nested_lists for item in sublist])
		

		camera_model.append(cam_i_yaml['camera_model'])
		distortion_coeffs.append(cam_i_yaml['distortion_coeffs'])
		distortion_model.append(cam_i_yaml['distortion_model'])
		intrinsics.append(cam_i_yaml['intrinsics'])
		resolution.append(cam_i_yaml['resolution'])
		timeshift_cam_imu.append(float(cam_i_yaml['timeshift_cam_imu']))

	# save to a simpler .yaml config file
	cams = {}

	for i in range(n_cam):
		cam = {}
		cam['T_cam_imu'] = T_cam_imu[i]
		cam['camera_model'] = camera_model[i]
		cam['distortion_coeffs'] = distortion_coeffs[i]
		cam['distortion_model'] = distortion_model[i]
		cam['intrinsics'] = intrinsics[i]
		cam['resolution'] = resolution[i]
		cam['timeshift_cam_imu'] = timeshift_cam_imu[i]

		cams['cam%d' % i] = cam

	out_fn = args.kalibr_yaml_fn[:-5] + '_simple.yaml'
	f = open(out_fn, 'w')
	yaml.dump(cams, f, default_flow_style=None)
	f.close()

	print("simplfied yaml file saved to %s" % out_fn)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--kalibr_yaml_fn", type=str)
	parser.add_argument("--n_cam", type=int, default=2)
	args = parser.parse_args()

	run(args)

