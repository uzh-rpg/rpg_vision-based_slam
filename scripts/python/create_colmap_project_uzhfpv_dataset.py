'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import argparse
import os
from pathlib2 import Path as path_lib
import re
import requests
import shutil
import sys
import yaml

import rpg_vision_based_slam.colmap_params as colmap_params
import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.uzhfpv_util as uzhfpv_util
import rpg_vision_based_slam.util_colmap as util_colmap
import rpg_vision_based_slam.uzhfpv_flags as uzhfpv_flags


# Code from https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def run(img_folder):
    # This script creates a COLMAP project.
    colmap_base_dir = os.path.join(flags.datasetsPath(), uzhfpv_flags.colmapBaseDirRelativePath())
    colmap_dir = os.path.join(flags.datasetsPath(), uzhfpv_flags.colmapRelativePath())
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)

    # We run COLMAP on images from 1 camera only.
    if uzhfpv_flags.camIdxString() == 'left':
        cam_idx = 0
    else:
        cam_idx = 1
    calibs = uzhfpv_util.readCamCalibration(cam_idx)
    
    # Create config files for each COLMAP module
    # Module: Feature extractor
    feature_extractor_params = colmap_params.getFeatureExtractorParams()
    feature_extractor_params['database_path'] = os.path.join(colmap_dir, 'database.db')
    feature_extractor_params['image_path'] = os.path.join(colmap_dir, 'images')
    feature_extractor_params['ImageReader']['camera_model'] = 'OPENCV_FISHEYE'
    camera_params_str = str(calibs.f[0]) + ',' + str(calibs.f[1]) + ',' + \
    str(calibs.c[0]) + ',' + str(calibs.c[1]) + ',' + \
    "%.9f" % calibs.distortion.k[0] + ',' + "%.9f" % calibs.distortion.k[1] + ',' + \
    "%.9f" % calibs.distortion.k[2] + ',' + "%.9f" % calibs.distortion.k[3]
    feature_extractor_params['ImageReader']['camera_params'] = camera_params_str

    feature_extractor_config_fn = os.path.join(colmap_dir, 'feature_extractor_config.ini')
    util_colmap.write_feature_extractor_config(feature_extractor_params, feature_extractor_config_fn)

    # Module: Sequential matcher
    sequential_matcher_params = colmap_params.getSequentialMatcherParams()
    sequential_matcher_params['database_path'] = os.path.join(colmap_dir, 'database.db')
    sequential_matcher_params['SequentialMatching']['vocab_tree_path'] = \
    os.path.join(colmap_base_dir, 'vocab_tree_flickr100K_words256K.bin')

    sequential_matcher_config_fn = os.path.join(colmap_dir, 'sequential_matcher_config.ini')
    util_colmap.write_sequential_matcher_config(sequential_matcher_params, sequential_matcher_config_fn)

    # Module: Mapper
    mapper_params = colmap_params.getMapperParams()
    mapper_params['database_path'] = os.path.join(colmap_dir, 'database.db')
    mapper_params['image_path'] = os.path.join(colmap_dir, 'images')
    mapper_params['output_path'] = os.path.join(colmap_dir, 'output')

    mapper_config_fn = os.path.join(colmap_dir, 'mapper_config.ini')
    util_colmap.write_mapper_config(mapper_params, mapper_config_fn)

    # Download pre-trained vocabulary tree for loop closing
    if 'vocab_tree_flickr100K_words256K.bin' not in os.listdir(colmap_base_dir):
        print('\nDownloading pre-trained vocabulary tree for loop closing ...')
        url = 'https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin'
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(colmap_base_dir, 'vocab_tree_flickr100K_words256K.bin'), 'wb').write(r.content)

    # Create image folder
    img_fns = os.listdir(img_folder)
    img_fns = sorted_alphanumeric(img_fns)

    dir_colmap_imgs = os.path.join(colmap_dir, 'images')
    path_lib(dir_colmap_imgs).mkdir(parents=True, exist_ok=True)

    img_idx = 0
    for img_fn in img_fns:
        if img_fn[0] == 'i':
            source = os.path.join(img_folder, img_fn)

            if int(img_fn[6]) == cam_idx:
                destination = os.path.join(dir_colmap_imgs, 'image%08d.png' % img_idx)
                img_idx += 1

            _ = shutil.copy(source, destination)
    print("\nWritten %d images to %s" % (img_idx, dir_colmap_imgs))

    # Create output dir
    dir_colmap_output = os.path.join(colmap_dir, 'output')
    path_lib(dir_colmap_output).mkdir(parents=True, exist_ok=True)

    print("\nA new COLMAP project: %s was successfully created.\n" % colmap_dir)
    print("To run SfM, execute sequentially the following commands")
    print("\tcolmap database_creator --database_path %s" % os.path.join(colmap_dir, 'database.db'))
    print("\tcolmap feature_extractor --project_path %s" % feature_extractor_config_fn)
    print("\tcolmap sequential_matcher --project_path %s" % sequential_matcher_config_fn)
    print("\tcolmap mapper --project_path %s" % mapper_config_fn)

    print("To print statistics after SfM is completed, run")
    print("\tcolmap model_analyzer --path %s" % os.path.join(dir_colmap_output, '0'))


if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    img_folder = os.path.join(flags.datasetsPath(), uzhfpv_flags.unzippedGtRelativePath() + '/img')
    assert os.path.exists(img_folder), "Folder %s not found" % img_folder
    run(img_folder)

