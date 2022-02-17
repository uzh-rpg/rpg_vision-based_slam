'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import absl
import numpy as np
import os
import sys

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.pose as pose
import rpg_vision_based_slam.utils as utils
import rpg_vision_based_slam.euroc_util as euroc_util
import rpg_vision_based_slam.util_colmap as util_colmap
import rpg_vision_based_slam.euroc_flags as euroc_flags


FLAGS = flags.FLAGS


def run():
    # This script extracts camera traj estimates (=T_W_C) from COLMAP output.
    colmap_dir = os.path.join(flags.datasetsPath(), euroc_flags.colmapRelativePath())
    if FLAGS.colmap_model_id == '0':
        model_dir = os.path.join(colmap_dir, 'output/0')
    if FLAGS.colmap_model_id == '1':
        model_dir = os.path.join(colmap_dir, 'output/1')
    else:        
        assert os.path.exists(model_dir), 'Model %d not found' % FLAGS.colmap_model_id

    # Try to read colmap output from .bin
    colmap_output = None
    fns = os.listdir(model_dir)
    for fn in fns:
        if fn == 'images.bin':
            output_dir = os.path.join(model_dir, 'images.bin')
            colmap_output = util_colmap.read_images_binary(output_dir)
    # If .bin not available, try from .txt
    if colmap_output == None:
        for fn in fns:
            if fn == 'images.txt':
                output_dir = os.path.join(model_dir, 'images.txt')
                colmap_output = util_colmap.read_images_text(output_dir)
    assert colmap_output != None, 'COLMAP output not found.'

    # Read timestamps
    if euroc_flags.camIdxString() == 'left':
        f_img_infos = os.path.join(flags.datasetsPath(), euroc_flags.repoRelPath() + '/left_images.txt')
    elif euroc_flags.camIdxString() == 'right':
        f_img_infos = os.path.join(flags.datasetsPath(), euroc_flags.repoRelPath() + '/right_images.txt')
    else:
        assert False, "Please, provide camera index used to run COLMAP."
    ids, all_timestamps = utils.readImageIdAndTimestamps(f_img_infos)

    # Image names according to the COLMAP convention, e.g. image00000000.png
    img_names = ['image%08d.png' % img_idx for img_idx in range(len(ids))]

    # Save
    out_filename = os.path.join(model_dir, "colmap_cam_estimates.txt")

    # It may happen that colmap does not reconstruct all the images
    all_timestamps = np.asarray(all_timestamps)
    timestamps = []
    T_W_C = []
    keys = np.asarray(colmap_output.keys())

    for i in keys:
        img_name = colmap_output[i].name
        idx = img_names.index(img_name)
        assert 0 <= idx < len(img_names)

        # timestamp
        t = all_timestamps[idx] 
        timestamps.append(t)

        # W: world, C: camera.
        # colmap_output[i].tvec is t_C_W, 
        # colmap_output[i].qvec is q_C_W (= [qw,qx,qy,qz])
        t_C_W_i = colmap_output[i].tvec
        q_C_W_i = colmap_output[i].qvec
        T_C_W_i = pose.fromPositionAndQuaternion(t_C_W_i, q_C_W_i)
        T_W_C_i = T_C_W_i.inverse()
        T_W_C.append(T_W_C_i)
    timestamps = np.asarray(timestamps)

    # It might be that keys are not sorted
    if not np.all(np.diff(timestamps) >= 0):
        print('WARNING: COLMAP images are not sorted! Fixing it now ...')
        n = 0 
        for diff in np.diff(timestamps):
            if diff < 0:
                n+=1
        print('Found %d unorder images' % n)
    sorted_idxs = np.argsort(timestamps)
    timestamps = np.sort(timestamps)
    T_W_C = [T_W_C[i] for i in sorted_idxs]

    utils.writeTrajEstToTxt(out_filename, timestamps, T_W_C)

    # Transform to body
    calib_file = os.path.join(flags.datasetsPath(), euroc_flags.calibRelativePath() + '/calib.yaml')
    calibs = utils.readCamCalibration(euroc_flags.repoName(), calib_file, euroc_flags.camIdxInt())

    T_W_B = [T_W_Ci * calibs.T_C_B for T_W_Ci in T_W_C]

    out_filename = os.path.join(model_dir, "colmap_body_estimates.txt")
    utils.writeTrajEstToTxt(out_filename, timestamps, T_W_B)


if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    run()

