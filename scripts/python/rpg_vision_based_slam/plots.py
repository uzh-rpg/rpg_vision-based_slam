'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: https://github.com/uzh-rpg/uzh_fpv_open/blob/master/python/uzh_fpv/plots.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# [in] pos: (N,2) np.array
def xyPlot(title, pos1, label1, pos2, label2, pos3=None, label3=None):
    plt.plot(pos1[:, 0], pos1[:, 1], 'g.', label=label1)
    plt.plot(pos2[:, 0], pos2[:, 1], label=label2)
    if label3 is not None:
        plt.plot(pos3[:, 0], pos3[:, 1], label=label3)
    plt.grid()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)


# [in] pos: (N,2) np.array
def xzPlot(title, pos1, label1, pos2, label2, pos3=None, label3=None):
    plt.plot(pos1[:, 0], pos1[:, 1], 'g.', label=label1)
    plt.plot(pos2[:, 0], pos2[:, 1], label=label2)
    if label3 is not None:
        plt.plot(pos3[:, 0], pos3[:, 1], label=label3)
    plt.grid()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(title)


# [in] pos: (N,2) np.array
def yzPlot(title, pos1, label1, pos2, label2, pos3=None, label3=None):
    plt.plot(pos1[:, 0], pos1[:, 1], 'g.', label=label1)
    plt.plot(pos2[:, 0], pos2[:, 1], label=label2)
    if label3 is not None:
        plt.plot(pos3[:, 0], pos3[:, 1], label=label3)
    plt.grid()
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title(title)


# [in] pos: (N,2) np.array
def xyPlotViFusion(title, pos1, label1, pos2, label2):
    plt.plot(pos1[:, 0], pos1[:, 1], color='darkorange', label=label1)
    plt.plot(pos2[:, 0], pos2[:, 1], color='b', label=label2)
    plt.grid()
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)


# [in] times: (N,) np.array
# [in] pos_i: (N,3) np.array
def xyztPlot(title, t1, pos1, label1, t2, pos2, label2, t3=None, pos3=None, label3=None):
    plt.subplot(311)
    plt.plot(t1, pos1[:, 0], 'g.', label=label1)
    plt.plot(t2, pos2[:, 0], label=label2)
    if label3 is not None:
        plt.plot(t3, pos3[:, 0], label=label3)
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)

    plt.subplot(312)
    plt.plot(t1, pos1[:, 1], 'g.', label=label1)
    plt.plot(t2, pos2[:, 1], label=label2)
    if label3 is not None:
        plt.plot(t3, pos3[:, 1], label=label3)
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')

    plt.subplot(313)
    plt.plot(t1, pos1[:, 2], 'g.', label=label1)
    plt.plot(t2, pos2[:, 2], label=label2)
    if label3 is not None:
        plt.plot(t3, pos3[:, 2], label=label3)
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('z')


# [in] errs_i: (N,) np.array
# [in] errs_type: Global Position or Reprojection
# [in] titile: title
def cumulativeErrorPlot(errs_1, label1, errs_2, label2, err_type, title):
    plt.semilogx(sorted(errs_1), np.arange(len(errs_1), dtype=float) / len(errs_1), label=label1)
    if errs_2 is not None:
        plt.semilogx(sorted(errs_2), np.arange(len(errs_2), dtype=float) / len(errs_2), label=label2)
    plt.grid()
    xlabel = err_type + ' errors'
    if err_type == 'Global Position':
        xlabel += ' [m]'
    elif err_type == 'Reprojection':
        xlabel += ' [px]'
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Fraction of measurements with smaller error')
    plt.title(title)


# [in] errs: (N,) np.array
# [in] errs_type: string
# [in] title: string
def histogramErrorPlot(errs, err_type, title):
    _ = plt.hist(errs, bins='auto')
    xlabel = err_type + ' errors'
    if err_type == 'Global Position':
        xlabel += ' [m]'
    elif err_type == 'Reprojection':
        xlabel += ' [px]'
    plt.xlabel(xlabel)
    plt.ylabel('#')
    plt.title(title)


# [in] init_errs, fin_errs: (N,) np.array
# [in] errs_type: string
# [in] title: string
def compareHistogramsErrorPlot(init_errs, fin_errs, err_type, title):
    _ = plt.hist(init_errs, bins='auto', label='Initial errors')
    _ = plt.hist(fin_errs, bins='auto', label='Optimized errors', color='darkorange', alpha=0.75)
    xlabel = err_type + ' error'
    if err_type == 'Global Position':
        xlabel += ' [m]'
    elif err_type == 'Reprojection':
        xlabel += ' [px]'
    plt.xlabel(xlabel)
    plt.ylabel('#')
    plt.legend()
    plt.title(title)


# [in] times: (N,) np.array
# [in] meas: (N,3) np.array
# [in] pred_meas: (N,3) np.array
def imuMeasPredPlot(t_meas, meas, t_pred, pred_meas, title):
    plt.subplot(311)
    plt.plot(t_meas, meas[:, 0], label='meas.')
    plt.plot(t_pred, pred_meas[:, 0], label='pred.')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    
    plt.subplot(312)
    plt.plot(t_meas, meas[:, 1], label='meas.')
    plt.plot(t_pred, pred_meas[:, 1], label='pred.')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')

    plt.subplot(313)
    plt.plot(t_meas, meas[:, 2], label='meas.')
    plt.plot(t_pred, pred_meas[:, 2], label='pred.')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('z')


# [in] axs.
# [in] err_acc: (N,) np.array
# [in] err_rot: (N,) np.array
def plotImuErrors(axs, err_acc, err_rot):
    axs[0].plot(err_acc, '-')
    axs[0].set_ylabel('Error linear acc')
    title = 'mean: %.3f, std: %.3f' % (np.mean(err_acc), np.std(err_acc))
    axs[0].set_title(title)

    axs[1].plot(err_rot, '-')
    axs[1].set_ylabel('Error ang vel')
    title = 'mean: %.3f, std: %.3f' % (np.mean(err_rot), np.std(err_rot)) 
    axs[1].set_title(title)


# [in] fig: figure
# [in] pos: (N,3) np.array
# [in] color: color
# [in] marker: marker
def plotLandmarks(fig, pos, color, marker):
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, marker=marker)

    plt.legend()
    plt.gca().set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


# [in] fig: figure
# [in] p1: (N,3) np.array. Usually initial points.
# [in] label1: label1.
# [in] p2: (N,3) np.array. Usually optimized points.
# [in] label2: label2.
def comparePointClouds(fig, p1, label1, p2, label2):
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c='b', marker='^', label=label1)
    ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], c='r', marker='o', label=label2)

    plt.legend()
    plt.gca().set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


# [in] ts: timestamps (N,) np.array.
# [in] bias_arr: (N,3) np.array.
# [in] title: str
def plotBias(ts, bias_arr, title):
    plt.plot(ts, bias_arr[:, 0], label='x')
    plt.plot(ts, bias_arr[:, 1], label='y')
    plt.plot(ts, bias_arr[:, 2], label='z')

    plt.grid()
    plt.legend()
    plt.xlabel('t [s]')
    plt.title(title)


def plotEulerAngles(t0, ori0_zyx, label0, t1=None, ori1_zyx=None, label1=None):
    plt.subplot(311)
    plt.plot(t0, ori0_zyx[:, 0], label=label0)
    if label1 is not None:
        plt.plot(t1, ori1_zyx[:, 0], label=label1)
    plt.grid()
    plt.title('Euler Angles')
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('yaw [deg]')

    plt.subplot(312)
    plt.plot(t0, ori0_zyx[:, 1], label=label0)
    if label1 is not None:
        plt.plot(t1, ori1_zyx[:, 1], label=label1)
    plt.grid()
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('pitch [deg]')

    plt.subplot(313)
    plt.plot(t0, ori0_zyx[:, 2], label=label0)
    if label1 is not None:
        plt.plot(t1, ori1_zyx[:, 2], label=label1)
    plt.grid()
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('roll [deg]')

