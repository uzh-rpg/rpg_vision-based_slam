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
*/

#pragma once

#include <optimization/continuous_time/ceres_spline_helper.h>
#include <optimization/continuous_time/ceres_spline_helper_old.h>

using namespace gvi_fusion;


template <int _N>
struct AccelerometerCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int Nb = 4;        // Order of the bias spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AccelerometerCostFunctor(const Eigen::Vector3d& measurement,
  double st_s, double k, double dt_s, double inv_dt, 
  double u_bias, double inv_dt_bias, double inv_std) : 
  measurement(measurement), 
  st_s(st_s), k(k), dt_s(dt_s), inv_dt(inv_dt), u_bias(u_bias), 
  inv_dt_bias(inv_dt_bias), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    // compute u with time offset correction
    T t_corr_s = sKnots[3 * N + 1][0];
    T st_corr_s = st_s - t_corr_s; 
    T u = (st_corr_s - k * dt_s) / dt_s;

    Sophus::SO3<T> R_w_b;
    CeresSplineHelper<N>::template evaluate_lie_uvar<T, Sophus::SO3>(sKnots, u, 
    inv_dt, &R_w_b);

    Vector3 accel_w;
    CeresSplineHelper<N>::template evaluate_uvar<T, 3, 2>(sKnots + N, u, inv_dt, 
    &accel_w);

    Vector3 bias;
    CeresSplineHelper<Nb>::template evaluate<T, 3, 0>(sKnots + 2 * N, u_bias, inv_dt_bias, 
    &bias);

    Eigen::Map<Vector3 const> const g(sKnots[3 * N]);

    residuals = inv_std * (R_w_b.inverse() * (accel_w + g) - measurement + bias);

    return true;
  }

  Eigen::Vector3d measurement;
  double st_s, k, dt_s, inv_dt, u_bias, inv_dt_bias, inv_std;
};


template <int _N, template <class> class GroupT, bool OLD_TIME_DERIV>
struct GyroCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int Nb = 4;        // Order of the bias spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  using Tangentd = typename GroupT<double>::Tangent;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GyroCostFunctor(const Tangentd& measurement,
  double st_s, double k, double dt_s, double inv_dt, 
  double u_bias, double inv_dt_bias, double inv_std) : 
  measurement(measurement), 
  st_s(st_s), k(k), dt_s(dt_s), inv_dt(inv_dt), 
  u_bias(u_bias), inv_dt_bias(inv_dt_bias), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename GroupT<T>::Tangent;

    Eigen::Map<Tangent> residuals(sResiduals);

    // compute u with time offset correction
    T t_corr_s = sKnots[2 * N][0];
    T st_corr_s = st_s - t_corr_s; 
    T u = (st_corr_s - k * dt_s) / dt_s;

    Tangent rot_vel;

    if constexpr (OLD_TIME_DERIV) {
      CeresSplineHelperOld<N>::template evaluate_lie_vel_old<T, GroupT>(
          sKnots, u, inv_dt, nullptr, &rot_vel);
    } else {
      CeresSplineHelper<N>::template evaluate_lie_uvar<T, GroupT>(sKnots, u, inv_dt, 
      nullptr, &rot_vel);
    }

    /*Vector3 bias_vec;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots + N, u_bias, inv_dt_bias,
                                                     &bias_vec);
    Eigen::Map<Tangent const> const bias(bias_vec);*/

    Tangent bias;
    CeresSplineHelper<Nb>::template evaluate<T, 3, 0>(sKnots + N, u_bias, inv_dt_bias, 
    &bias);

    residuals = inv_std * (rot_vel - measurement + bias);

    return true;
  }

  Tangentd measurement;
  double st_s, k, dt_s, inv_dt, u_bias, inv_dt_bias, inv_std;
  double u_init_unscaled; // u_init = u_init_unscaled / dt_s
};


template <int _N>
struct AccelerometerCostFunctorFixedSamplingTime : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int Nb = 4;        // Order of the bias spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AccelerometerCostFunctorFixedSamplingTime(const Eigen::Vector3d& measurement,
  double u, double inv_dt, double u_bias, double inv_dt_bias, double inv_std) : 
  measurement(measurement), u(u), inv_dt(inv_dt), u_bias(u_bias), 
  inv_dt_bias(inv_dt_bias), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Sophus::SO3<T> R_w_b;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &R_w_b);

    Vector3 accel_w;
    CeresSplineHelper<N>::template evaluate<T, 3, 2>(sKnots + N, u, inv_dt,
                                                     &accel_w);

    Vector3 bias;
    CeresSplineHelper<Nb>::template evaluate<T, 3, 0>(sKnots + 2 * N, u_bias, inv_dt_bias,
                                                     &bias);

    Eigen::Map<Vector3 const> const g(sKnots[3 * N]);

    residuals = inv_std * (R_w_b.inverse() * (accel_w + g) - measurement + bias);

    return true;
  }

  Eigen::Vector3d measurement;
  double u, inv_dt, u_bias, inv_dt_bias, inv_std;
};


template <int _N, template <class> class GroupT, bool OLD_TIME_DERIV>
struct GyroCostFunctorFixedSamplingTime : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int Nb = 4;        // Order of the bias spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  using Tangentd = typename GroupT<double>::Tangent;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GyroCostFunctorFixedSamplingTime(const Tangentd& measurement,
  double u, double inv_dt, double u_bias, double inv_dt_bias, double inv_std) : 
  measurement(measurement), u(u), inv_dt(inv_dt), u_bias(u_bias), 
  inv_dt_bias(inv_dt_bias), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename GroupT<T>::Tangent;

    Eigen::Map<Tangent> residuals(sResiduals);

    Tangent rot_vel;

    if constexpr (OLD_TIME_DERIV) {
      CeresSplineHelperOld<N>::template evaluate_lie_vel_old<T, GroupT>(
          sKnots, u, inv_dt, nullptr, &rot_vel);
    } else {
      CeresSplineHelper<N>::template evaluate_lie<T, GroupT>(sKnots, u, inv_dt,
                                                             nullptr, &rot_vel);
    }

    Tangent bias;
    CeresSplineHelper<Nb>::template evaluate<T, 3, 0>(sKnots + N, u_bias, inv_dt_bias,
                                                     &bias);

    residuals = inv_std * (rot_vel - measurement + bias);

    return true;
  }

  Tangentd measurement;
  double u, inv_dt, u_bias, inv_dt_bias, inv_std;
};


template <int _N>
struct AccelerometerCostFunctorConstBias : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  AccelerometerCostFunctorConstBias(const Eigen::Vector3d& measurement,
  double u, double inv_dt, double inv_std) : 
  measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Sophus::SO3<T> R_w_b;
    CeresSplineHelper<N>::template evaluate_lie<T, Sophus::SO3>(sKnots, u,
                                                                inv_dt, &R_w_b);

    Vector3 accel_w;
    CeresSplineHelper<N>::template evaluate<T, 3, 2>(sKnots + N, u, inv_dt,
                                                     &accel_w);

    // Gravity
    Eigen::Map<Vector3 const> const g(sKnots[2 * N]);
    Eigen::Map<Vector3 const> const bias(sKnots[2 * N + 1]);

    residuals = inv_std * (R_w_b.inverse() * (accel_w + g) - measurement + bias);

    return true;
  }

  Eigen::Vector3d measurement;
  double u, inv_dt, inv_std;
};


template <int _N, template <class> class GroupT, bool OLD_TIME_DERIV>
struct GyroCostFunctorConstBias : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  using Tangentd = typename GroupT<double>::Tangent;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GyroCostFunctorConstBias(const Tangentd& measurement, double u, double inv_dt, double inv_std): 
  measurement(measurement), u(u), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const {
    using Tangent = typename GroupT<T>::Tangent;

    Eigen::Map<Tangent> residuals(sResiduals);

    Tangent rot_vel;

    if constexpr (OLD_TIME_DERIV) {
      CeresSplineHelperOld<N>::template evaluate_lie_vel_old<T, GroupT>(
          sKnots, u, inv_dt, nullptr, &rot_vel);
    } else {
      CeresSplineHelper<N>::template evaluate_lie<T, GroupT>(sKnots, u, inv_dt,
                                                             nullptr, &rot_vel);
    }

    Eigen::Map<Tangent const> const bias(sKnots[N]);

    residuals = inv_std * (rot_vel - measurement + bias);

    return true;
  }

  Tangentd measurement;
  double u, inv_dt, inv_std;
};


template <int _N>
struct BiasCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  using MatN = Eigen::Matrix<double, _N, _N>;
  using VecN = Eigen::Matrix<double, _N, 1>;

  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BiasCostFunctor(double u_0, double u_1, double inv_dt, double inv_std) : 
  u_0(u_0), u_1(u_1), inv_dt(inv_dt), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Vector3 bias_0;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots, u_0, inv_dt,
                                                     &bias_0);

    Vector3 bias_1;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots + N, u_1, inv_dt,
                                                     &bias_1);

    residuals = inv_std * (bias_0 - bias_1);

    return true;
  }

  double u_0, u_1, inv_dt, inv_std;
};


template <int _N>
struct BiasTerminalCostFunctor : public CeresSplineHelper<_N> {
  static constexpr int N = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BiasTerminalCostFunctor(double u, double inv_dt, Eigen::Vector3d& init_bias, 
  double inv_std) : 
  u(u), inv_dt(inv_dt), init_bias(init_bias), inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sKnots, T* sResiduals) const 
  {
    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3> residuals(sResiduals);

    Vector3 bias;
    CeresSplineHelper<N>::template evaluate<T, 3, 0>(sKnots, u, inv_dt,
                                                     &bias);

    residuals = inv_std * (bias - init_bias);

    return true;
  }

  double u, inv_dt;
  Eigen::Vector3d init_bias;
  double inv_std;
};


struct ConstantBiasCostFunctor
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ConstantBiasCostFunctor(double inv_std): inv_std(inv_std) {}

  template <class T>
  bool operator()(T const* const* sBias, T* sResiduals) const {

    using Vector3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<Vector3 const> const bias(sBias[0]);

    sResiduals[0] = inv_std * bias[0] * bias[0];
    sResiduals[1] = inv_std * bias[1] * bias[1];
    sResiduals[2] = inv_std * bias[2] * bias[2];

    return true;
  }

  double inv_std;
};

