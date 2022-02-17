'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

import os
import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rosbag

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.euroc_flags as euroc_flags


def extract():
    data_folder = os.path.join(flags.datasetsPath(), euroc_flags.repoName())
    seq_name = euroc_flags.sequenceString()
    seq_folder = os.path.join(data_folder, seq_name)

    if not os.path.exists(seq_folder):
        os.makedirs(seq_folder)
    img_folder = os.path.join(seq_folder, 'img')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # extract data
    f_img_left = open(os.path.join(seq_folder, 'left_images.txt'), 'w')
    f_img_left.write('# id timestamp image_name\n')
    f_img_right = open(os.path.join(seq_folder, 'right_images.txt'), 'w')
    f_img_right.write('# id timestamp image_name\n')
    f_imu = open(os.path.join(seq_folder, 'imu.txt'), 'w')
    f_imu.write('# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z\n') 

    n_img_left = 0
    n_img_right = 0
    n_imu = 0

    cv_bridge = CvBridge()
    
    bagfile = os.path.join(data_folder, seq_name + '.bag')
    with rosbag.Bag(bagfile, 'r') as bag:
        for (topic, msg, ts) in bag.read_messages():
            if topic == '/cam0/image_raw':
                try:
                    img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                except CvBridgeError, e:
                    print e
                    
                ts = msg.header.stamp.to_sec()
                img_name = 'image_0_'+str(n_img_left)+'.png'
                f_img_left.write('%d %.12f img/%s \n' % (n_img_left, ts, img_name))
                img_fn = os.path.join(img_folder, img_name)
                cv2.imwrite(img_fn, img)
                n_img_left += 1

            elif topic == '/cam1/image_raw':
                try:
                    img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                except CvBridgeError, e:
                    print e
                    
                ts = msg.header.stamp.to_sec()
                img_name = 'image_1_'+str(n_img_right)+'.png'
                f_img_right.write('%d %.12f img/%s \n' % (n_img_right, ts, img_name))
                img_fn = os.path.join(img_folder, img_name)
                cv2.imwrite(img_fn, img)
                n_img_right += 1

            elif topic == '/imu0':
                f_imu.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (n_imu,
                         msg.header.stamp.to_sec(),
                         msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                         msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
                n_imu += 1
    
    
    f_img_left.close()
    f_img_right.close()
    f_imu.close()

    print('loaded ' + str(n_img_left) + ' left camera images')
    print('loaded ' + str(n_img_right) + ' right camera images')
    print('loaded ' + str(n_imu) + ' imu measurements')

          
if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    extract()

