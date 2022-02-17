/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This code is based on: 
https://gitlab.com/tum-vision/lie-spline-experiments/-/blob/master/include/ceres_calib_split_residuals.h
*/

/*
Reference frames:
W: (fixed) world frame. Where spline is expressed in.
B: (moving) body (= imu) frame.
*/

#pragma once

#include <optimization/continuous_time/ceres_spline_helper.h>
#include <optimization/continuous_time/ceres_spline_helper_old.h>

using namespace gvi_fusion;


template <int _N, template <class> class GroupT>
struct MakeDavisGTAccelerometerCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MakeDavisGTAccelerometerCostFunctor(const Eigen::Vector3d& acc_davis,
  double st_s, double k, double dt_s, double inv_dt, double inv_std) : 
  acc_davis(acc_davis), st_s(st_s), k(k), dt_s(dt_s), inv_dt(inv_dt), 
  inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Tangent = typename GroupT<T>::Tangent;

    Eigen::Map<Vector3> residuals(sResiduals);

    // compute u with time offset correction
    T t_corr_s = sKnots[2 * N + 2][0];
    T st_corr_s = st_s + t_corr_s; 
    T u = (st_corr_s - k * dt_s) / dt_s;

    Sophus::SO3<T> R_w_snap;
    Tangent omega_snap;
    Tangent omega_dot_snap;
    CeresSplineHelper<N>::template evaluate_lie_uvar<T, Sophus::SO3>(sKnots, u, 
    inv_dt, &R_w_snap, &omega_snap, &omega_dot_snap);

    Vector3 w_accel_snap;
    CeresSplineHelper<N>::template evaluate_uvar<T, 3, 2>(sKnots + N, u, inv_dt, 
    &w_accel_snap);

    Eigen::Map<Sophus::SO3<T> const> const R_davis_snap(sKnots[2 * N]);
    Eigen::Map<Vector3 const> const w_r_snap_davis(sKnots[2 * N + 1]);

    Tangent w_omega_snap;
    w_omega_snap = R_w_snap * omega_snap;
    Tangent w_omega_dot_snap;
    w_omega_dot_snap = R_w_snap * omega_dot_snap;

    Vector3 drot = w_omega_snap.cross(w_r_snap_davis);
    Vector3 w_accel_davis_est = w_accel_snap + w_omega_dot_snap.cross(w_r_snap_davis) + 
    w_omega_snap.cross(drot);

    w_accel_davis_est[2] += gravity;
    Vector3 accel_davis_est = R_davis_snap * (R_w_snap.inverse() * w_accel_davis_est);

    residuals = inv_std * (accel_davis_est - acc_davis);

    return true;
  }

  Eigen::Vector3d acc_davis;
  double st_s, k, dt_s, inv_dt, inv_std;
  //Eigen::Vector3d w_gravity = Eigen::Vector3d(0.0, 0.0, 9.81);
  double gravity = 9.81;
};


template <int _N, template <class> class GroupT>
struct MakeDavisGTGyroCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MakeDavisGTGyroCostFunctor(const Eigen::Vector3d& gyro_davis,
  double st_s, double k, double dt_s, double inv_dt, double inv_std) : 
  gyro_davis(gyro_davis), st_s(st_s), k(k), dt_s(dt_s), inv_dt(inv_dt), 
  inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename GroupT<T>::Tangent;

    Eigen::Map<Tangent> residuals(sResiduals);

    // compute u with time offset correction
    T t_corr_s = sKnots[N + 1][0];
    T st_corr_s = st_s - t_corr_s; 
    T u = (st_corr_s - k * dt_s) / dt_s;

    Tangent gyro_snap;
    CeresSplineHelper<N>::template evaluate_lie_uvar<T, GroupT>(sKnots, u, 
    inv_dt, nullptr, &gyro_snap);

    Eigen::Map<Sophus::SO3<T> const> const R_davis_snap(sKnots[N]);

    Tangent gyro_davis_est;
    gyro_davis_est = R_davis_snap * gyro_snap;

    residuals = inv_std * (gyro_davis_est - gyro_davis);

    return true;
  }

  Eigen::Vector3d gyro_davis;
  double st_s, k, dt_s, inv_dt, inv_std;
};

