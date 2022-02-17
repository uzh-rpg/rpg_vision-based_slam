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


REPO_NAME = 'UZH_FPV'


# Flags for UZH-FPV dataset
# Dataset specification
flags.DEFINE_string('env', '', 'indoor / outdoor (i/o)')
flags.DEFINE_string('cam', '', 'camera (fw/45)')
flags.DEFINE_string('nr', '0', 'sequence number')

# Sensor specification
flags.DEFINE_string('sens', '', 'davis or snap')
flags.DEFINE_string('cam_i', 'left', 'if sensor is snap, left or right cam')


def initial_t_G_L():
    """ Good initial guesses for the time offset. """
    if FLAGS.env == 'o':
        if FLAGS.cam == 'fw' and FLAGS.nr in ['9', '10']:
            return -25265.
        if FLAGS.cam == '45':
            return 3600.
    return -1.


def envCamString():
    if FLAGS.env == 'i':
        result = 'indoor_'
    else:
        assert FLAGS.env == 'o'
        result = 'outdoor_'
    if FLAGS.cam == 'fw':
        result = result + 'forward'
    else:
        assert FLAGS.cam == '45'
        result = result + '45'
    return result


def repoName():
    return os.path.join(REPO_NAME)

    
def sequenceString():
    return envCamString() + '_' + FLAGS.nr


def sensorString():
    if FLAGS.sens == 'davis':
        return 'davis'
    else:
        assert FLAGS.sens == 'snap'
        return 'snapdragon'


def camIdxString():
    if FLAGS.cam_i == 'right':
        return 'right'
    else:
        assert FLAGS.cam_i == 'left'
        return 'left'


def sequenceSensorString():
    return sequenceString() + '_' + sensorString()


def calibString():
    result = envCamString() + '_calib_'
    if FLAGS.sens == 'davis':
        result = result + 'davis'
    else:
        assert FLAGS.sens == 'snap'
        result = result + 'snapdragon'
    return result


def calibRelativePath():
    return os.path.join(REPO_NAME + '/calib', calibString())


def colmapBaseDirRelativePath():
    return os.path.join(REPO_NAME, 'colmap')


def colmapRelativePath():
    return os.path.join(REPO_NAME + '/colmap', sequenceSensorString())


def rawDataRelativePath():
    return os.path.join(REPO_NAME + '/raw', sequenceString())


def outputDataRelativePath():
    return os.path.join(REPO_NAME + '/output', sequenceSensorString() + '_with_gt')


def newOutputDataRelativePath():
    return os.path.join(REPO_NAME + '/new_output', sequenceSensorString() + '_with_gt')


def imuTopic():
    if FLAGS.sens == 'davis':
        return '/dvs/imu'
    else:
        assert FLAGS.sens == 'snap'
        return '/snappy_imu'


def unzippedGtRelativePath():
    wd = os.path.join(REPO_NAME, 'output')
    wgt = sequenceSensorString() + '_with_gt'
    result = os.path.join(wd, wgt)
    return result


def newGtRelativePath():
    result = os.path.join(REPO_NAME, sequenceSensorString())
    if FLAGS.sens == 'snap':
        result = os.path.join(result, 'full_batch_optimization/continuous_time/order_6/dt_100_ms')
    return result


def renderGtRelativePath():
    tmp = os.path.join(REPO_NAME, sequenceSensorString())
    result = os.path.join(tmp, 'render_gt')
    return result

