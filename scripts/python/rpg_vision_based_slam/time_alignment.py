'''
This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
'''

# This script has been adapted from: 
# https://github.com/ethz-asl/hand_eye_calibration/blob/966cd92518f24aa7dfdacc8ba9c5fa4a441270cd/hand_eye_calibration/python/hand_eye_calibration/time_alignment.py


import matplotlib
from matplotlib import pylab as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import interpolate
from scipy import signal


def plot_results(times_A, times_B, signal_A, signal_B, convoluted_signals, time_offset, block=True):
    fig = plt.figure()

    title_position = 1.05

    matplotlib.rcParams.update({'font.size': 20})

    # fig.suptitle("Time Alignment", fontsize='24')
    a1 = plt.subplot(1, 3, 1)

    a1.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.ylabel('angular velocity norm [rad]')
    plt.xlabel('time [s]')
    a1.set_title(
      "Before Time Alignment", y=title_position)
    plt.hold("on")

    min_time = min(np.amin(times_A), np.amin(times_B))
    times_A_zeroed = times_A - min_time
    times_B_zeroed = times_B - min_time

    plt.plot(times_A_zeroed, signal_A, c='r')
    plt.plot(times_B_zeroed, signal_B, c='b')

    times_A_shifted = times_A + time_offset

    a3 = plt.subplot(1, 3, 2)
    a3.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('correlation')
    plt.xlabel('sample idx offset')
    a3.set_title(
      "Correlation Result \n[Ideally has a single dominant peak.]",
      y=title_position)
    plt.hold("on")
    plt.plot(np.arange(-len(signal_A) + 1, len(signal_B)), convoluted_signals)

    a2 = plt.subplot(1, 3, 3)
    a2.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('angular velocity norm [rad]')
    plt.xlabel('time [s]')
    a2.set_title(
      "After Time Alignment", y=title_position)
    plt.hold("on")
    min_time = min(np.amin(times_A_shifted), np.amin(times_B))
    times_A_shifted_zeroed = times_A_shifted - min_time
    times_B_zeroed = times_B - min_time
    plt.plot(times_A_shifted_zeroed, signal_A, c='r')
    plt.plot(times_B_zeroed, signal_B, c='b')

    plt.subplots_adjust(left=0.04, right=0.99, top=0.8, bottom=0.15)

    if plt.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        max_size = mng.window.maxsize()
        max_size = (max_size[0], max_size[1] * 0.45)
        mng.resize(*max_size)
        plt.show(block=block)


def plot_angular_velocities(title, angular_velocities, angular_velocities_filtered, block=True):
    fig = plt.figure()

    title_position = 1.05

    fig.suptitle(title, fontsize='24')

    a1 = plt.subplot(1, 2, 1)
    a1.set_title(
      "Angular Velocities Before Filtering \nvx [red], vy [green], vz [blue]",
      y=title_position)
    plt.plot(angular_velocities[:, 0], c='r')
    plt.plot(angular_velocities[:, 1], c='g')
    plt.plot(angular_velocities[:, 2], c='b')

    a2 = plt.subplot(1, 2, 2)
    a2.set_title(
      "Angular Velocities After Filtering \nvx [red], vy [green], vz [blue]", y=title_position)
    plt.plot(angular_velocities_filtered[:, 0], c='r')
    plt.plot(angular_velocities_filtered[:, 1], c='g')
    plt.plot(angular_velocities_filtered[:, 2], c='b')

    plt.subplots_adjust(left=0.025, right=0.975, top=0.8, bottom=0.05)

    if plt.get_backend() == 'TkAgg':
        mng = plt.get_current_fig_manager()
        max_size = mng.window.maxsize()
        max_size = (max_size[0], max_size[1] * 0.45)
        mng.resize(*max_size)
        plt.show(block=block)


def calculate_time_offset_from_angvels_norms(times_A, signal_A, times_B, signal_B, plot=False, block=True):
    """ Calculates the time offset between signal A and signal B. 
    times_A_aligned = times_A + time_offset"""
    convoluted_signals = signal.correlate(signal_B, signal_A)
    dt_A = np.mean(np.diff(times_A))
    offset_indices = np.arange(-len(signal_A) + 1, len(signal_B))
    max_index = np.argmax(convoluted_signals)
    offset_index = offset_indices[max_index]
    time_offset = dt_A * offset_index + times_B[0] - times_A[0]

    if plot:
        plot_results(times_A, times_B, signal_A, signal_B, convoluted_signals,
                     time_offset, block=block)
    return time_offset


def resample_angvels(times, angvels, dt):
    interval = times[-1] - times[0]
    t_samples = np.linspace(times[0], times[-1], interval / dt + 1)
    
    f_interp = interpolate.interp1d(times, angvels.T)
    interp_angvels = f_interp(t_samples)
    return (t_samples, interp_angvels.T)


def filter_angvels(angular_velocity, low_pass_kernel_size, clip_percentile, plot=False):
    """Reduce the noise in a velocity signal."""
    max_value = np.percentile(angular_velocity, clip_percentile)
    print("Clipping angular velocity norms to {} rad/s ...".format(max_value))
    angular_velocity_clipped = np.clip(angular_velocity, -max_value, max_value)

    low_pass_kernel = np.ones((low_pass_kernel_size, 1)) / low_pass_kernel_size
    print("Smoothing with kernel size {} samples...".format(low_pass_kernel_size))

    angular_velocity_smoothed = \
    signal.correlate(angular_velocity_clipped, low_pass_kernel, 'same')
    print("Done smoothing angular velocity norms...")

    if plot:
        plot_angular_velocities("Angular Velocities", \
            angular_velocity, angular_velocity_smoothed, True)

    return angular_velocity_smoothed.copy()

