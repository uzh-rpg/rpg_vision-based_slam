'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py

from collections import defaultdict

FEATURE_EXTRACTOR_PARAMS = defaultdict(dict)
FEATURE_EXTRACTOR_PARAMS['random_seed'] = '0'
FEATURE_EXTRACTOR_PARAMS['log_to_stderr'] = 'false'
FEATURE_EXTRACTOR_PARAMS['log_level'] = '2'
FEATURE_EXTRACTOR_PARAMS['database_path'] = ''
FEATURE_EXTRACTOR_PARAMS['image_path'] = '' 
FEATURE_EXTRACTOR_PARAMS['ImageReader']['mask_path'] = ''
FEATURE_EXTRACTOR_PARAMS['ImageReader']['camera_model'] = 'OPENCV_FISHEYE'
FEATURE_EXTRACTOR_PARAMS['ImageReader']['single_camera'] = 'true'
FEATURE_EXTRACTOR_PARAMS['ImageReader']['single_camera_per_folder'] = 'false'
FEATURE_EXTRACTOR_PARAMS['ImageReader']['existing_camera_id'] = '-1'
FEATURE_EXTRACTOR_PARAMS['ImageReader']['camera_params'] = ''
FEATURE_EXTRACTOR_PARAMS['ImageReader']['default_focal_length_factor'] = '1.2'
FEATURE_EXTRACTOR_PARAMS['ImageReader']['camera_mask_path'] = ''
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['use_gpu'] = 'true'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['estimate_affine_shape'] = 'false'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['upright'] = 'false'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['domain_size_pooling'] = 'false'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['num_threads'] = '-1'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['max_image_size'] = '3200'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['max_num_features'] = '8192'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['first_octave'] = '-1'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['num_octaves'] = '4'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['octave_resolution'] = '3'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['max_num_orientations'] = '2'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['dsp_num_scales'] = '10'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['peak_threshold'] = '0.00667'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['edge_threshold'] = '10'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['dsp_min_scale'] = '0.16667'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['dsp_max_scale'] = '3'
FEATURE_EXTRACTOR_PARAMS['SiftExtraction']['gpu_index'] = '-1'

SEQUENTIAL_MATCHER_PARAMS = defaultdict(dict)
SEQUENTIAL_MATCHER_PARAMS['random_seed'] = '0'
SEQUENTIAL_MATCHER_PARAMS['log_to_stderr'] = 'false'
SEQUENTIAL_MATCHER_PARAMS['log_level'] = '2'
SEQUENTIAL_MATCHER_PARAMS['database_path'] = ''
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['use_gpu'] = 'true'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['cross_check'] = 'true'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['multiple_models'] = 'false'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['guided_matching'] = 'false'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['num_threads'] = '-1'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['max_num_matches'] = '32768'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['max_num_trials'] = '10000'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['min_num_inliers'] = '15'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['max_ratio'] = '0.8'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['max_distance'] = '0.7'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['max_error'] = '4'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['confidence'] = '0.999'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['min_inlier_ratio'] = '0.25'
SEQUENTIAL_MATCHER_PARAMS['SiftMatching']['gpu_index'] = '-1'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['quadratic_overlap'] = 'true'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection'] = 'true'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['overlap'] = '30'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection_period'] = '10'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection_num_images'] = '50'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection_num_nearest_neighbors'] = '1'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection_num_checks'] = '256'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection_num_images_after_verification'] = '0'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['loop_detection_max_num_features'] = '-1'
SEQUENTIAL_MATCHER_PARAMS['SequentialMatching']['vocab_tree_path'] = ''


MAPPER_PARAMS = defaultdict(dict)
MAPPER_PARAMS['random_seed'] = '0'
MAPPER_PARAMS['log_to_stderr'] = 'false'
MAPPER_PARAMS['log_level'] = '2'
MAPPER_PARAMS['database_path'] = ''
MAPPER_PARAMS['image_path'] = ''
MAPPER_PARAMS['output_path'] = ''
MAPPER_PARAMS['Mapper']['ignore_watermarks'] = 'false'
MAPPER_PARAMS['Mapper']['multiple_models'] = 'true'
MAPPER_PARAMS['Mapper']['extract_colors'] = 'true'
MAPPER_PARAMS['Mapper']['ba_refine_focal_length'] = 'false'
MAPPER_PARAMS['Mapper']['ba_refine_principal_point'] = 'false'
MAPPER_PARAMS['Mapper']['ba_refine_extra_params'] = 'false'
MAPPER_PARAMS['Mapper']['ba_global_use_pba'] = 'false'
MAPPER_PARAMS['Mapper']['fix_existing_images'] = 'false'
MAPPER_PARAMS['Mapper']['tri_ignore_two_view_tracks'] = 'true'
MAPPER_PARAMS['Mapper']['min_num_matches'] = '15'
MAPPER_PARAMS['Mapper']['max_num_models'] = '50'
MAPPER_PARAMS['Mapper']['max_model_overlap'] = '20'
MAPPER_PARAMS['Mapper']['min_model_size'] = '10'
MAPPER_PARAMS['Mapper']['init_image_id1'] = '-1'
MAPPER_PARAMS['Mapper']['init_image_id2'] = '-1'
MAPPER_PARAMS['Mapper']['init_num_trials'] = '200'
MAPPER_PARAMS['Mapper']['num_threads'] = '-1'
MAPPER_PARAMS['Mapper']['ba_min_num_residuals_for_multi_threading'] = '50000'
MAPPER_PARAMS['Mapper']['ba_local_num_images'] = '6'
MAPPER_PARAMS['Mapper']['ba_local_max_num_iterations'] = '25'
MAPPER_PARAMS['Mapper']['ba_global_pba_gpu_index'] = '-1'
MAPPER_PARAMS['Mapper']['ba_global_images_freq'] = '500'
MAPPER_PARAMS['Mapper']['ba_global_points_freq'] = '250000'
MAPPER_PARAMS['Mapper']['ba_global_max_num_iterations'] = '50'
MAPPER_PARAMS['Mapper']['ba_global_max_refinements'] = '5'
MAPPER_PARAMS['Mapper']['ba_local_max_refinements'] = '2'
MAPPER_PARAMS['Mapper']['snapshot_images_freq'] = '0'
MAPPER_PARAMS['Mapper']['init_min_num_inliers'] = '100'
MAPPER_PARAMS['Mapper']['init_max_reg_trials'] = '2'
MAPPER_PARAMS['Mapper']['abs_pose_min_num_inliers'] = '30'
MAPPER_PARAMS['Mapper']['max_reg_trials'] = '3'
MAPPER_PARAMS['Mapper']['tri_max_transitivity'] = '1'
MAPPER_PARAMS['Mapper']['tri_complete_max_transitivity'] = '5'
MAPPER_PARAMS['Mapper']['tri_re_max_trials'] = '1'
MAPPER_PARAMS['Mapper']['min_focal_length_ratio'] = '0.1'
MAPPER_PARAMS['Mapper']['max_focal_length_ratio'] = '10'
MAPPER_PARAMS['Mapper']['max_extra_param'] = '1'
MAPPER_PARAMS['Mapper']['ba_global_images_ratio'] = '1.1'
MAPPER_PARAMS['Mapper']['ba_global_points_ratio'] = '1.1'
MAPPER_PARAMS['Mapper']['ba_global_max_refinement_change'] = '0.0005'
MAPPER_PARAMS['Mapper']['ba_local_max_refinement_change'] = '0.001'
MAPPER_PARAMS['Mapper']['init_max_error'] = '4'
MAPPER_PARAMS['Mapper']['init_max_forward_motion'] = '0.950'
MAPPER_PARAMS['Mapper']['init_min_tri_angle'] = '16'
MAPPER_PARAMS['Mapper']['abs_pose_max_error'] = '12'
MAPPER_PARAMS['Mapper']['abs_pose_min_inlier_ratio'] = '0.25'
MAPPER_PARAMS['Mapper']['filter_max_reproj_error'] = '4'
MAPPER_PARAMS['Mapper']['filter_min_tri_angle'] = '1.5'
MAPPER_PARAMS['Mapper']['local_ba_min_tri_angle'] = '6'
MAPPER_PARAMS['Mapper']['tri_create_max_angle_error'] = '2'
MAPPER_PARAMS['Mapper']['tri_continue_max_angle_error'] = '2'
MAPPER_PARAMS['Mapper']['tri_merge_max_reproj_error'] = '4'
MAPPER_PARAMS['Mapper']['tri_complete_max_reproj_error'] = '4'
MAPPER_PARAMS['Mapper']['tri_re_max_angle_error'] = '5'
MAPPER_PARAMS['Mapper']['tri_re_min_ratio'] = '0.2'
MAPPER_PARAMS['Mapper']['tri_min_angle'] = '1.5'
MAPPER_PARAMS['Mapper']['snapshot_path'] = ''


def getFeatureExtractorParams():
    return FEATURE_EXTRACTOR_PARAMS


def getSequentialMatcherParams():
    return SEQUENTIAL_MATCHER_PARAMS


def getMapperParams():
    return MAPPER_PARAMS

