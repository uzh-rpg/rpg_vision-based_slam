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

FLAGS = flags.FLAGS


def extract():
    rosbag_fn = FLAGS.rosbag_fn
    img0_topic = FLAGS.img0_topic
    img1_topic = FLAGS.img1_topic
    imu_topic = FLAGS.imu_topic
    res_dir = FLAGS.results_dir
    extract_imgs = FLAGS.extract_imgs

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if extract_imgs:
        img_folder = os.path.join(res_dir, 'img')
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

    # extract data
    if img1_topic is not None:
        f_img_left = open(os.path.join(res_dir, 'left_images.txt'), 'w')
        f_img_left.write('# id timestamp image_name\n')
        f_img_right = open(os.path.join(res_dir, 'right_images.txt'), 'w')
        f_img_right.write('# id timestamp image_name\n')
    else:
        f_img_left = open(os.path.join(res_dir, 'images.txt'), 'w')
        f_img_left.write('# id timestamp image_name\n')

    f_imu = open(os.path.join(res_dir, 'imu.txt'), 'w')
    f_imu.write('# timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z\n') 

    n_img_left = 0
    n_img_right = 0
    n_imu = 0

    cv_bridge = CvBridge()
    
    bagfile = os.path.join(rosbag_fn)
    with rosbag.Bag(bagfile, 'r') as bag:
        for (topic, msg, ts) in bag.read_messages():
            if topic == img0_topic:
                try:
                    img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                except CvBridgeError, e:
                    print e
                    
                ts = msg.header.stamp.to_sec()
                img_name = 'image_0_'+str(n_img_left)+'.png'
                f_img_left.write('%d %.12f img/%s \n' % (n_img_left, ts, img_name))
                if extract_imgs:
                    img_fn = os.path.join(img_folder, img_name)
                    cv2.imwrite(img_fn, img)
                n_img_left += 1

            elif topic == img1_topic:
                try:
                    img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                except CvBridgeError, e:
                    print e
                    
                ts = msg.header.stamp.to_sec()
                img_name = 'image_1_'+str(n_img_right)+'.png'
                f_img_right.write('%d %.12f img/%s \n' % (n_img_right, ts, img_name))
                if extract_imgs:
                    img_fn = os.path.join(img_folder, img_name)
                    cv2.imwrite(img_fn, img)
                n_img_right += 1

            elif topic == imu_topic:
                f_imu.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (n_imu,
                         msg.header.stamp.to_sec(),
                         msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                         msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
                n_imu += 1
    
    f_img_left.close()
    if img1_topic is not None:
        f_img_right.close()
    f_imu.close()

    if img1_topic is not None:
        print('loaded ' + str(n_img_left) + ' left camera images')
        print('loaded ' + str(n_img_right) + ' right camera images')
        print('loaded ' + str(n_imu) + ' imu measurements')
    else:
        print('loaded ' + str(n_img_left) + ' camera images')
        print('loaded ' + str(n_imu) + ' imu measurements')

          
if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)
    extract()

