'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''
 
import absl
import IPython
import numpy as np
import os
import sys

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rosbag
import rospy

import rpg_vision_based_slam.flags as flags
import rpg_vision_based_slam.uzhfpv_flags as uzhfpv_flags


FLAGS = flags.FLAGS


def make_bag(gt, old_bag_fn, new_bag_fn):
    in_bag = rosbag.Bag(old_bag_fn)
    out_bag = rosbag.Bag(new_bag_fn, 'w')

    for bag_msg in in_bag:
        msg = bag_msg.message
        if bag_msg.topic != '/groundtruth/odometry' and bag_msg.topic != '/groundtruth/pose':
            out_bag.write(bag_msg.topic, msg, msg.header.stamp)

    for i, v in enumerate(gt):
        od_msg = Odometry()
        od_msg.header.stamp = rospy.Time.from_sec(v[0])
        od_msg.header.frame_id = 'world'
        od_msg.header.seq = i

        od_msg.child_frame_id = 'body'

        od_msg.pose.pose.position.x = v[1]
        od_msg.pose.pose.position.y = v[2]
        od_msg.pose.pose.position.z = v[3]

        od_msg.pose.pose.orientation.x = v[4]
        od_msg.pose.pose.orientation.y = v[5]
        od_msg.pose.pose.orientation.z = v[6]
        od_msg.pose.pose.orientation.w = v[7]

        out_bag.write('/groundtruth/odometry', od_msg, od_msg.header.stamp)  

        ps_msg = PoseStamped()
        ps_msg.header.stamp = rospy.Time.from_sec(v[0])
        ps_msg.header.frame_id = 'world'
        ps_msg.header.seq = i

        ps_msg.pose.position.x = v[1]
        ps_msg.pose.position.y = v[2]
        ps_msg.pose.position.z = v[3]

        ps_msg.pose.orientation.x = v[4]
        ps_msg.pose.orientation.y = v[5]
        ps_msg.pose.orientation.z = v[6]
        ps_msg.pose.orientation.w = v[7]

        out_bag.write('/groundtruth/pose', ps_msg, ps_msg.header.stamp)  

    out_bag.close()


if __name__ == '__main__':
    sys.argv = flags.FLAGS(sys.argv)

    # Load data
    gt_dir = os.path.join(flags.resultsPath(), uzhfpv_flags.newGtRelativePath())
    if FLAGS.sens == 'davis':
        gt = np.loadtxt(os.path.join(gt_dir,'groundtruth.txt'))
    else:
        assert FLAGS.sens == 'snap'
        gt = np.loadtxt(os.path.join(gt_dir,'spline.txt'))

    old_bag_dir = os.path.join(flags.datasetsPath(), uzhfpv_flags.outputDataRelativePath())
    old_bag_fn = old_bag_dir + '.bag'

    new_bag_dir = os.path.join(flags.datasetsPath(), uzhfpv_flags.newOutputDataRelativePath())
    new_bag_fn = new_bag_dir + '.bag'

    make_bag(gt, old_bag_fn, new_bag_fn)

