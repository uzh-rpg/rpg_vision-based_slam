'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import os
import sys

import numpy as np

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.euroc_flags as euroc_flags


FLAGS = flags.FLAGS


def extract(gt, out_filename):
    n = 0
    fout = open(out_filename, 'w')
    fout.write('# timestamp tx ty tz qx qy qz qz\n')    

    with open(gt,'rb') as fin:
        data = np.genfromtxt(fin, delimiter=",")
        for l in data:
            fout.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                      (l[0]/1e9, l[1], l[2], l[3], l[5], l[6], l[7], l[4]))
            n += 1
    print('wrote ' + str(n) + ' measurements to the file: ' + out_filename)
    fout.close()


if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    raw_gt_fn = os.path.join(flags.datasetsPath(), euroc_flags.rawGroundtruthRelativePath())
    gt_fn = os.path.join(flags.datasetsPath(), euroc_flags.groundtruthRelativePath())

    print('Extract ground truth pose from file ' + raw_gt_fn)
    print('Saving to file ' + gt_fn)
    extract(raw_gt_fn, gt_fn)

