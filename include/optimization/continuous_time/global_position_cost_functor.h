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
W: (fixed) world frame. Where global measurements and spline are expressed.
B: (moving) body (= imu) frame.
*/

#pragma once

#include <optimization/continuous_time/ceres_spline_helper.h>
#include <optimization/continuous_time/ceres_spline_helper_old.h>

using namespace gvi_fusion;


template <int _N>
struct GlobalPositionSensorCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GlobalPositionSensorCostFunctor(const Eigen::Vector3d& measurement,
  double scale, double st_s, double k, double dt_s, double inv_dt, double inv_std) : 
  measurement(measurement), scale(scale), st_s(st_s), k(k), dt_s(dt_s), 
  inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    // compute u with time offset correction
    T t_corr_s = sKnots[2 * N + 1][0];
    T st_corr_s = st_s - t_corr_s; 
    T u = (st_corr_s - k * dt_s) / dt_s;

    Sophus::SO3<T> R_w_b;
    CeresSplineHelper<N>::template evaluate_lie_uvar<T, Sophus::SO3>(sKnots, u,
    inv_dt, &R_w_b);
    
    Vector3 p_w_b;
    CeresSplineHelper<N>::template evaluate_uvar<T, 3, 0>(sKnots + N, u, 
    inv_dt, &p_w_b);

    Eigen::Map<Vector3 const> const p_b_p(sKnots[2 * N]);
    T scale_corr = sKnots[2 * N + 2][0];
    T s = scale + scale_corr;
    Vector3 p_w_p = s * R_w_b.matrix() * p_b_p + p_w_b;
    
    residuals = inv_std * (p_w_p - measurement);

    return true;
  }

  Eigen::Vector3d measurement;
  double scale;
  double st_s, k, dt_s;
  double inv_dt, inv_std;
};


template <int _N>
struct GlobalPositionSensorWithAlignmentCorrectionCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GlobalPositionSensorWithAlignmentCorrectionCostFunctor(
    const Eigen::Vector3d& measurement, const Sophus::SE3d& T_w_g, double scale, 
    double st_s, double k, double dt_s, double inv_dt, double inv_std) : 
  measurement(measurement), T_w_g(T_w_g), scale(scale), 
  st_s(st_s), k(k), dt_s(dt_s), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Matrix3 = Eigen::Matrix<T, 3, 3>;

    Eigen::Map<Vector3> residuals(sResiduals);

    // compute u with time offset correction
    T t_corr_s = sKnots[2 * N][0];
    T st_corr_s = st_s - t_corr_s; 
    T u = (st_corr_s - k * dt_s) / dt_s;

    Sophus::SO3<T> R_g_b;
    CeresSplineHelper<N>::template evaluate_lie_uvar<T, Sophus::SO3>(sKnots, u,
    inv_dt, &R_g_b);
    
    Vector3 p_g_b;
    CeresSplineHelper<N>::template evaluate_uvar<T, 3, 0>(sKnots + N, u, 
    inv_dt, &p_g_b);

    T scale_corr = sKnots[2 * N + 1][0];
    T s = scale + scale_corr;

    Eigen::Map<Sophus::SO3<T> const> const R_w_g_corr(sKnots[2 * N + 2]);
    Eigen::Map<Vector3 const> const p_w_g_corr(sKnots[2 * N + 3]);

    Eigen::Map<Vector3 const> const p_b_p(sKnots[2 * N + 4]);

    Matrix3 R_w_g = T_w_g.rotationMatrix() * R_w_g_corr.matrix();
    Vector3 p_w_g = T_w_g.translation() + p_w_g_corr;
    //Vector3 p_w_g = T_w_g.rotationMatrix() * p_w_g_corr + T_w_g.translation();

    Matrix3 R_w_b = R_w_g * R_g_b.matrix();
    Vector3 p_w_b = s * R_w_g.matrix() * p_g_b + p_w_g;

    Vector3 p_w_p = s * R_w_b.matrix() * p_b_p + p_w_b;

    residuals = inv_std * (p_w_p - measurement);

    return true;
  }

  Eigen::Vector3d measurement;
  Sophus::SE3d T_w_g;
  double scale;
  double st_s, k, dt_s;
  double inv_dt, inv_std;
  double u_init_unscaled; // u_init = u_init_unscaled / dt_s
};

