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


struct SpeedCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SpeedCostFunctor(const Eigen::Vector3d& measurement, double inv_std) : 
  measurement(measurement), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* parameters, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3 const> const speed_and_bias(parameters[0]);
    
    Vector3 dv;
    dv[0] = measurement[0] - speed_and_bias[0];
    dv[1] = measurement[1] - speed_and_bias[1];
    dv[2] = measurement[2] - speed_and_bias[2];

    sResiduals[0] = inv_std * dv[0];
    sResiduals[1] = inv_std * dv[1];
    sResiduals[2] = inv_std * dv[2];

    return true;
  }

  Eigen::Vector3d measurement;
  double inv_std;
};

