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


REPO_NAME = 'EuRoC'


# Dataset specification
flags.DEFINE_string('room', '', 'MH / V1 / V2')
flags.DEFINE_string('nr', '', 'sequence number')
flags.DEFINE_string('cam', 'left', 'left or right cam')


def repoName():
    return REPO_NAME


def roomSeqString():
    if FLAGS.room == 'MH':
        result = 'MH_'
    elif FLAGS.room == 'V1':
        result = 'V1_'
    else:
        assert FLAGS.room == 'V2'
        result = 'V2_'
    return result


def sequenceString():
    result = roomSeqString() + '0' + FLAGS.nr + '_'

    if FLAGS.room == 'MH':
        if FLAGS.nr == '1' or FLAGS.nr == '2':
            result += 'easy'
        elif FLAGS.nr == '3':
            result += 'medium'
        else:
            assert FLAGS.nr == '4' or FLAGS.nr == '5'
            result += 'difficult'
    else:
        assert FLAGS.room == 'V1' or FLAGS.room == 'V2'
        if FLAGS.nr == '1':
            result += 'easy'
        elif FLAGS.nr == '2':
            result += 'medium'
        else:
            assert FLAGS.nr == '3'
            result += 'difficult'

    return result


def camIdxString():
    if FLAGS.cam == 'right':
        return 'right'
    else:
        assert FLAGS.cam == 'left'
        return 'left'


def camIdxInt():
    if FLAGS.cam == 'right':
        return 1
    else:
        assert FLAGS.cam == 'left'
        return 0


def calibString():
    if FLAGS.room == 'MH':
        return 'Machine_hall'
    else:
        assert FLAGS.room == 'V1' or FLAGS.room == 'V2'
        return 'Vicon_room'


def calibRelativePath():
    return os.path.join(REPO_NAME + '/calib', calibString())


def repoRelPath():
    return os.path.join(REPO_NAME, sequenceString())


def colmapBaseDirRelativePath():
    return os.path.join(REPO_NAME, 'colmap')


def colmapRelativePath():
    return os.path.join(REPO_NAME + '/colmap', sequenceString())


def rawGroundtruthRelativePath():
    return os.path.join(REPO_NAME, sequenceString() + '/state_groundtruth_estimate0/data.csv')


def groundtruthRelativePath():
    return os.path.join(REPO_NAME, sequenceString() + '/state_groundtruth_estimate0/groundtruth.txt')


def gpMeasRelativePath(freq, noise):
    fn = 'gp_measurements_freq_%.1f_hz_std_%.2f_m.txt' % (freq, noise) 
    return os.path.join(REPO_NAME, sequenceString() + '/' + fn)

