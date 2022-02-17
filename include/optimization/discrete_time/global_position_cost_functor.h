/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#pragma once

#include "sensor/global_position_sensor.h"

using namespace gvi_fusion;


struct GlobalPositionSensorCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GlobalPositionSensorCostFunctor(const GlobalPosMeasurement& measurement,
  double scale, double t_k0, double t_k1, double t_offset_gp_imu_init, double inv_std) : 
  measurement(measurement), scale(scale), t_k0(t_k0), t_k1(t_k1), 
  t_offset_gp_imu_init(t_offset_gp_imu_init), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* parameters, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Eigen::Map<Sophus::SE3<T> const> const T_WB0(parameters[0]);
    Eigen::Map<Sophus::SE3<T> const> const T_WB1(parameters[1]);
    T scale_corr = parameters[2][0];
    T s = scale + scale_corr;
    T t_offset_corr = parameters[3][0];
    T t_offset = t_offset_gp_imu_init + t_offset_corr;
    Eigen::Map<Vector3 const> const p_BP(parameters[4]);

    // interpolation factor
    T t_interp = measurement.t + t_offset;
    T lambda = (t_interp - t_k0) / (t_k1 - t_k0);

    Vector3 p_WP0 = s * T_WB0.rotationMatrix() * p_BP + T_WB0.translation();
    Vector3 p_WP1 = s * T_WB1.rotationMatrix() * p_BP + T_WB1.translation();

    Vector3 p_WP_interp = (1.0 - lambda) * p_WP0 + lambda * p_WP1;
    
    residuals = inv_std * (p_WP_interp - measurement.pos);

    return true;
  }

  GlobalPosMeasurement measurement;
  double scale;
  double t_k0, t_k1, t_offset_gp_imu_init;
  double inv_std;
};

