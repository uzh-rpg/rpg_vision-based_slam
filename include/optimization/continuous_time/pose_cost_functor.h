/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
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
struct PoseCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PoseCostFunctor(const Sophus::SE3d& measurement,
  double u, double inv_dt, double inv_std) : 
  measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Sophus::SO3<T> r;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &r);

    Vector3 p;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots + N, u, inv_dt,
                                                     &p);

    Sophus::SE3d measurement_inv = measurement.inverse();
    Sophus::SO3<T> dr = measurement_inv.so3() * r;
    Vector3 dtheta = 2.0 * dr.unit_quaternion().vec();

    Vector3 dp = measurement_inv.rotationMatrix() * p + measurement_inv.translation();

    sResiduals[0] = inv_std * dp[0];
    sResiduals[1] = inv_std * dp[1];
    sResiduals[2] = inv_std * dp[2];

    sResiduals[3] = inv_std * dtheta[0];
    sResiduals[4] = inv_std * dtheta[1];
    sResiduals[5] = inv_std * dtheta[2];

    return true;
  }

  Sophus::SE3d measurement;
  double u, inv_dt, inv_std;
};

