/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This script is based on: 
https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo_ceres_backend/include/svo/ceres_backend/pose_local_parameterization.hpp
*/

#pragma once

#include "util/matrix_operations.h"


// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool liftJacobian(const double* x, double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);
  // Translational part.
  J_lift.setZero();
  J_lift.topLeftCorner<3, 3>().setIdentity();

  const Eigen::Quaterniond q_inv(x[6], -x[3], -x[4], -x[5]);
  Eigen::Matrix4d Qplus = quaternionOplusMatrix(q_inv);
  J_lift.bottomRightCorner<3, 4>() = 2.0 * Qplus.topLeftCorner<3, 4>();
  return true;
}

