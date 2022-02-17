/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
this code is from: 
https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo_vio_common/include/svo/vio_common/matrix_operations.hpp
*/

#include <iostream>


// Skew symmetric matrix.
inline Eigen::Matrix<double,3,3> skewSymmetric(const double w1,
                             const double w2,
                             const double w3)
{
  return (Eigen::Matrix<double,3,3>() <<
           0.0f, -w3,  w2,
           w3,  0.0f, -w1,
          -w2,  w1,  0.0f).finished();
}


inline Eigen::Matrix<double,3,3> skewSymmetric(
    const Eigen::Ref<const Eigen::Matrix<double,3,1> >& w)
{
  return skewSymmetric(w(0), w(1), w(2));
}


// Right Jacobian for Exponential map in SO(3)
inline Eigen::Matrix<double,3,3> expmapDerivativeSO3(const Eigen::Matrix<double,3,1>& omega)
{
  double theta2 = omega.dot(omega);
  if (theta2 <= std::numeric_limits<double>::epsilon())
  {
    return Eigen::Matrix<double,3,3>::Identity();
  }
  double theta = std::sqrt(theta2);  // rotation angle
  // element of Lie algebra so(3): X = omega^, normalized by normx
  const Eigen::Matrix<double,3,3> Y = skewSymmetric(omega) / theta;
  return Eigen::Matrix<double,3,3>::Identity()
      - ((double{1} - std::cos(theta)) / (theta)) * Y
      + (double{1} - std::sin(theta) / theta) * Y * Y;
}


// -----------------------------------------------------------------------------
// Quaternion utils

//! Plus matrix for a quaternion. q_AB x q_BC = plus(q_AB) * q_BC.coeffs().
inline Eigen::Matrix<double,4,4> quaternionPlusMatrix(
    const Eigen::Quaternion<double>& q_AB)
{
  const Eigen::Matrix<double,4,1>& q = q_AB.coeffs();
  Eigen::Matrix<double,4,4> Q;
  Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
  Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
  Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

//! Opposite-Plus matrix for a quaternion q_AB x q_BC = oplus(q_BC) * q_AB.coeffs().
inline Eigen::Matrix<double,4,4> quaternionOplusMatrix(
    const Eigen::Quaternion<double>& q_BC)
{
  const   Eigen::Matrix<double,4,1>& q = q_BC.coeffs();
  Eigen::Matrix<double,4,4> Q;
  Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
  Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
  Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

