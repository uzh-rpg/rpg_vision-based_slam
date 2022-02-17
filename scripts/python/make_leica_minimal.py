'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
''' 

# Use this script to make a minimal .txt containing leica measurements
# [in] original leica .txt as in UZH-FPV raw/'sequence'/leica.txt
# [out] leica_minimal.txt generate in the folder where leica.txt is.

import numpy as np
import os
import sys

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.uzhfpv_flags as uzhfpv_flags
import rpg_vision_based_slam.uzhfpv_leica_parser as leica_parser

FLAGS = flags.FLAGS


def run():
	datasets_dir = flags.datasetsPath()
	raw_data_dir = os.path.join(datasets_dir, uzhfpv_flags.rawDataRelativePath())
	leica_fn = os.path.join(raw_data_dir, 'leica.txt')

	t, pos = leica_parser.parseTextFile(leica_fn)
	assert t.shape[0] == pos.shape[0]
	data = np.hstack((t.reshape(-1, 1), pos))

	data_fn = os.path.join(raw_data_dir, 'leica_minimal.txt')
	f = open(data_fn, 'w')
	f.write('# timestamp x y z\n')
	np.savetxt(f, data)
	f.close()
	

if __name__ == '__main__':
	sys.argv = flags.FLAGS(sys.argv)
	run()

