'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv_open/blob/master/python/uzh_fpv/flags.py

import datetime
import math
import numpy as np
import re
import time


class GeneralizedTime(object):
    """ ROS time's typing and positivity check is too restrictive. """
    def __init__(self, secs_or_time_obj, nsecs=None):
        """ Secs and nsecs may be negative. """
        if hasattr(secs_or_time_obj, 'secs'):
            assert hasattr(secs_or_time_obj, 'nsecs')
            self.__init__(secs_or_time_obj.secs, secs_or_time_obj.nsecs)
        elif type(secs_or_time_obj) == float:
            dec, i = math.modf(secs_or_time_obj)
            self.__init__(int(i), int(dec * 1e9))
        else:
            assert nsecs is not None
            assert abs(nsecs) < 1e9
            self.secs = secs_or_time_obj
            self.nsecs = nsecs

    def to_sec(self):
        return self.secs + float(self.nsecs) / 1e9

    def __add__(self, other):
        secs = self.secs + other.secs
        nsecs = self.nsecs + other.nsecs
        if abs(nsecs) >= 1e9:
            if nsecs > 0:
                nsecs = nsecs - 1e9
                secs = secs + 1
            else:
                nsecs = nsecs + 1e9
                secs = secs - 1
        return GeneralizedTime(secs, nsecs)

    def __sub__(self, other):
        return self.__add__(-GeneralizedTime(other))

    def __neg__(self):
        return GeneralizedTime(-self.secs, -self.nsecs)


def parseTextFile(path):
    leica_file = open(path, 'r')

    # Patterns:
    date_time = '\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d.\d\d\d'
    value = '-?\d+\.\d+'

    op_line = '^0,%s,Operator,.*$' % date_time
    empty_line = '^[\r\n]?$'
    start_line = '^0,%s,Start![\r\n]?$' % date_time
    weird_line = '^1,%s,%%R1Q,0,\d:,%s,%%R1P,0,\d:0[\r\n]?$' % (date_time, date_time)
    meas_line = re.compile('^3,(%s),%%R1Q,2082,\d:10000,1,%s,%%R1P,0,\d:(\d+),(%s),(%s),(%s),\d+,%s,%s,%s,\d+[\r\n]?$'
                           % (date_time, date_time, value, value, value, value, value, value))
    outlier_line = '^3,%s,%%R1Q,2082,\d:10000,1,%s,%%R1P,0,\d:\d+,0,0,0,0,0,0,0,0[\r\n]?$' % (
        date_time, date_time)
    final_line = '^0,%s,Canceled![\r\n]?$' % date_time

    t_gentime = []
    x = []
    y = []
    z = []
    for line in leica_file:
        meas_match = meas_line.match(line)
        if meas_match:
            if meas_match.group(2) in ['0', '1284']:
                dt = datetime.datetime.strptime(meas_match.group(1) + '000', '%Y-%m-%d %H:%M:%S.%f')
                # Could also use ROS Time here, or any other time representation.
                t_gentime.append(GeneralizedTime(int(time.mktime(dt.timetuple())), dt.microsecond * 1000))
                x.append(float(meas_match.group(3)))
                y.append(float(meas_match.group(4)))
                z.append(float(meas_match.group(5)))
        elif re.match(op_line, line) or re.match(empty_line, line) or re.match(start_line, line) or \
                re.match(weird_line, line) or re.match(final_line, line) or re.match(outlier_line, line):
            pass
        else:
            print('The following line does not match any pattern:')
            print(line)

    t = [i.to_sec() for i in t_gentime]

    return np.asarray(t), np.array([x, y, z]).T

