'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import absl
import os
import sys

import numpy as np

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.euroc_flags as euroc_flags

absl.flags.DEFINE_float('freq', 10.0, 'Freq [hz] of gp measurements.')
absl.flags.DEFINE_float('noise', 0.50, 'Std of gaussian noise added to groundtruth.')

FLAGS = flags.FLAGS


def run(gt_fn, freq, noise):
    gt = np.loadtxt(gt_fn)
    gt_t = gt[:, 0]
    gt_p = gt[:, 1:4]

    # Generate gp measurements at the desired freq.
    dt = 1.0 / freq

    gp_meas = [] # [t, x, y, z]
    t_prev = gt_t[0]
    noisy_meas = gt_p[0] + np.random.normal(0.0, noise)
    gp_meas.append(np.array([t_prev, noisy_meas[0], noisy_meas[1], noisy_meas[2]]))

    for i, t in enumerate(gt_t):
        if t >= t_prev + dt:
            t_prev = t
            noisy_meas = gt_p[i] + np.random.normal(0.0, noise)
            gp_meas.append(np.array([t_prev, noisy_meas[0], noisy_meas[1], noisy_meas[2]]))

    # save
    outfn = os.path.join(flags.datasetsPath(), euroc_flags.gpMeasRelativePath(freq, noise))
    f = open(outfn, 'w')
    f.write('# timestamp x y z\n')
    np.savetxt(f, np.asarray(gp_meas), fmt='%.12f %.12f %.12f %.12f')
    f.close()
    print('Simulated global measurements saved to %s' % outfn)


if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    gt_fn = os.path.join(flags.datasetsPath(), euroc_flags.groundtruthRelativePath())

    run(gt_fn, FLAGS.freq, FLAGS.noise)

