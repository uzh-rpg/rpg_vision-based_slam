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

#include <optimization/continuous_time/ceres_spline_helper.h>
#include <optimization/continuous_time/ceres_spline_helper_old.h>
#include <sensor/camera.h>

using namespace gvi_fusion;


template <int _N>
struct ReprojectionCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Frame* frame, const Camera* camera, 
  double scale, double u, double inv_dt, double inv_std) : 
  frame(frame), camera(camera), scale(scale), u(u), inv_dt(inv_dt), inv_std(inv_std){}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    using Matrix3 = Eigen::Matrix<T, 3, 3>;

    Sophus::SO3<T> R_w_b;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &R_w_b);

    Vector3 p_w_b;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots + N, u, inv_dt,
                                                     &p_w_b);

    T scale_corr = sKnots[2 * N][0];
    T s = scale + scale_corr;

    Eigen::Map<Sophus::SO3<T> const> const R_b_c_corr(sKnots[2 * N + 1]);
    Eigen::Map<Vector3 const> const p_b_c_corr(sKnots[2 * N + 2]);

    Matrix3 R_b_c = camera->T_b_c.rotationMatrix() * R_b_c_corr.matrix();
    // @ToDo: Do this?
    Vector3 p_b_c = camera->T_b_c.rotationMatrix() * p_b_c_corr + camera->T_b_c.translation();
    //Vector3 p_b_c = camera->T_b_c.translation() + p_b_c_corr;

    Matrix3 R_w_c = R_w_b.matrix() * R_b_c;
    Vector3 p_w_c = s * R_w_b.matrix() * p_b_c + p_w_b;

    Matrix3 R_c_w = R_w_c.transpose();
    Vector3 p_c_w = -1.0 * R_c_w * p_w_c;

    size_t n_reproj = 0; 
    for(const auto& point2d: frame->points2D)
    {
        if (!point2d.is_reproj_valid) { continue; }

        Eigen::Map<Vector3 const> const p_w_li(sKnots[2 * N + 3 + n_reproj]);
        Vector3 p_c_li = R_c_w * p_w_li + p_c_w;
        Vector2 reproj_uv;
        camera->project(p_c_li, reproj_uv);

        // Do not consider errors above a threshold (= 10 pixel)
        T squared_err_norm_threshold = T(100.0);
        T squared_err_norm = (reproj_uv[0] - point2d.uv[0]) * (reproj_uv[0] - point2d.uv[0])
        + (reproj_uv[1] - point2d.uv[1]) * (reproj_uv[1] - point2d.uv[1]);

        if (squared_err_norm > squared_err_norm_threshold)
        {
          sResiduals[2 * n_reproj + 0] = T(0);
          sResiduals[2 * n_reproj + 1] = T(0);
        }
        else
        {
          sResiduals[2 * n_reproj + 0] = inv_std * (reproj_uv[0] - point2d.uv[0]);
          sResiduals[2 * n_reproj + 1] = inv_std * (reproj_uv[1] - point2d.uv[1]);
        }

        n_reproj += 1;
    }

    return true;
  }

  const Frame* frame;
  const Camera* camera;

  double scale, u, inv_dt, inv_std;
};

