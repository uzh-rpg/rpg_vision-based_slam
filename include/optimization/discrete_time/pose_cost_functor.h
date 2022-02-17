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

using namespace gvi_fusion;


struct PoseCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PoseCostFunctor(const Sophus::SE3d& measurement, double inv_std) : 
  measurement(measurement), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* parameters, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Sophus::SE3<T> const> const T_WB0(parameters[0]);
    
    Sophus::SO3<T> r = T_WB0.so3();
    Vector3 p = T_WB0.translation();

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
  double inv_std;
};

