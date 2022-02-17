'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv_open/blob/master/python/uzh_fpv/flags.py

from absl import flags
import os


FLAGS = flags.FLAGS


# General Flags
flags.DEFINE_string('config', '', 'path to yaml file')
flags.DEFINE_bool('gui', False, 'visualize stuff')
flags.DEFINE_string('results_dir', '', 'result dir if not in config.yaml')

# Flags for custom dataset 
flags.DEFINE_string('dataset_name', 'Custom', 'Dataset name')
flags.DEFINE_string('rosbag_fn', '', 'path to rosbag if a custom dataset')
flags.DEFINE_string('img0_topic', '', 'img topic name')
flags.DEFINE_string('img1_topic', None, 'img topic name')
flags.DEFINE_string('imu_topic', '', 'imu topic name')
flags.DEFINE_bool('extract_imgs', False, 'extract images')
flags.DEFINE_string('colmap_dir', '', 'path to colmap results')
flags.DEFINE_integer('colmap_model_id', 0, 'colmap model id')
flags.DEFINE_integer('cam_idx', 0, '0: left, 1: right')
flags.DEFINE_string('images_fn', '', 'path to .txt containing images infos')
flags.DEFINE_string('calib_fn', '', 'path to .yaml containing cam-imu calib')


def repoRoot():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def datasetsPath():
    return os.path.join(repoRoot(), 'datasets')


def resultsPath():
    return os.path.join(repoRoot(), 'results')

