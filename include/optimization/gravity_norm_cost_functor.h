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


struct GravityNormCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GravityNormCostFunctor(){}

  template <class T>
  bool operator()(T const* const* parameters, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3 const> const g_est(parameters[0]);
    T g_est_norm_squared = g_est[0] * g_est[0] + g_est[1] * g_est[1] + g_est[2] * g_est[2];
    sResiduals[0] = weight * (g_est_norm_squared - (g_norm * g_norm));

    return true;
  }

double g_norm = 9.80665;
double weight = 1e6;
};

