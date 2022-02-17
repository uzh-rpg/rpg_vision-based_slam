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
C: (moving) camera frame.
Li : i-th landmark position in W.
*/

#pragma once

#include <sensor/camera.h>

using namespace gvi_fusion;


struct ReprojectionCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Frame* frame, const Camera* camera, 
  double scale, double inv_std) : 
  frame(frame), camera(camera), scale(scale), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* parameters, T* sResiduals) const 
  {
    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Matrix3 = Eigen::Matrix<T, 3, 3>;

    Eigen::Map<Sophus::SE3<T> const> const T_WB(parameters[0]);

    T scale_corr = parameters[1][0];
    T s = scale + scale_corr;

    Eigen::Map<Sophus::SO3<T> const> const R_BC_corr(parameters[2]);
    Eigen::Map<Vector3 const> const p_BC_corr(parameters[3]);

    // t_imu = t_cam + (t_offset_cam_imu + t_offset_cam_imu_corr)
    T t_offset_cam_imu_corr = parameters[4][0];

    Matrix3 R_BC = camera->T_b_c.rotationMatrix() * R_BC_corr.matrix();
    // @ToDo: Do this?
    //Vector3 p_BC = camera->T_b_c.translation() + p_BC_corr;
    Vector3 p_BC = camera->T_b_c.rotationMatrix() * p_BC_corr + camera->T_b_c.translation();

    Matrix3 R_WC = T_WB.rotationMatrix() * R_BC;
    Vector3 p_WC = s * T_WB.rotationMatrix() * p_BC + T_WB.translation();

    Matrix3 R_CW = R_WC.transpose();
    Vector3 p_CW = -1.0 * R_CW * p_WC;

    size_t n_reproj = 0;
    for(const auto& point2d: frame->points2D)
    {
      if (!point2d.is_reproj_valid) { continue; }

      Eigen::Map<Vector3 const> const p_WLi(parameters[5 + n_reproj]);
      Vector3 p_CLi = R_CW * p_WLi + p_CW;
      Vector2 reproj_uv;
      camera->project(p_CLi, reproj_uv);

      // update feature position to account for cam-imu time offset
      Eigen::Vector2d vel;
      vel = point2d.velocity;
      // for features that are not tracked on next frame.
      /*if (vel.norm() < 1e-6)
      {
        vel = frame->avg_pixel_vel;
      }*/
      T u = point2d.uv[0] + vel[0] * (camera->t_offset_cam_imu + t_offset_cam_imu_corr);
      T v = point2d.uv[1] + vel[1] * (camera->t_offset_cam_imu + t_offset_cam_imu_corr);

      // Check if feature went outside the image plane.
      if ((u < T(0.0)) || (u > T(camera->w)) || (v < T(0.0)) || (v > T(camera->h)))
      {
        sResiduals[2 * n_reproj + 0] = T(0);
        sResiduals[2 * n_reproj + 1] = T(0);

        n_reproj += 1;
        continue;
      }

      // Do not consider errors above a threshold (= 10 pixel)
      T squared_err_norm_threshold = T(100.0);
      T squared_err_norm = (reproj_uv[0] - u) * (reproj_uv[0] - u) + (reproj_uv[1] - v) * (reproj_uv[1] - v);
      if (squared_err_norm > squared_err_norm_threshold)
      {
        sResiduals[2 * n_reproj + 0] = T(0);
        sResiduals[2 * n_reproj + 1] = T(0);

        n_reproj += 1;
        continue;
      }

      // Valid reprojection errors
      sResiduals[2 * n_reproj + 0] = inv_std * (reproj_uv[0] - u);
      sResiduals[2 * n_reproj + 1] = inv_std * (reproj_uv[1] - v);

      n_reproj += 1;
    }

    return true;
  }

  const Frame* frame;
  const Camera* camera;

  double scale, inv_std;
};

