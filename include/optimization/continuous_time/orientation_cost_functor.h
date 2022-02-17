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
struct OrientationCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OrientationCostFunctor(const Sophus::SO3d& measurement,
  double u, double inv_dt, double inv_std) : 
  measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Sophus::SO3<T> r;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &r);

    Sophus::SO3<T> dr = r * measurement.inverse();
    Vector3 dtheta = 2.0 * dr.unit_quaternion().vec();

    residuals = inv_std * dtheta;

    return true;
  }

  Sophus::SO3d measurement;
  double u, inv_dt, inv_std;
};

