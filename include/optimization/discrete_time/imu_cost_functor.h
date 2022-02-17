/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This script is based on: 
https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo_ceres_backend/include/svo/ceres_backend/imu_error.hpp
*/

#pragma once

#include <mutex>

#include "error_interface.h"
#include "pose_local_parameterization.h"
#include "sensor/imu.h"

using namespace gvi_fusion;


__inline__ Eigen::Quaterniond deltaQ(const Eigen::Vector3d& dAlpha)
{
  Eigen::Vector4d dq;
  double halfnorm = 0.5 * dAlpha.template tail<3>().norm();
  dq.template head<3>() = sinc(halfnorm) * 0.5 * dAlpha.template tail<3>();
  dq[3] = cos(halfnorm);
  return Eigen::Quaterniond(dq);
}


//Implements a nonlinear IMU factor.
struct ImuCostFunctor :
    public ceres::SizedCostFunction<15 /* number of residuals */,
        7 /* size of 1st parameter (pose k) */,
        9 /* size of 3rd parameter (vel and biases k) */,
        7 /* size of 6th parameter (pose k+1) */,
        9 /* size of 7th parameter (vel and biases k+1) */>,
    public ErrorInterface
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // helpers typedef
  typedef Eigen::aligned_vector<ImuMeasurement> ImuMeasurements;
  typedef std::shared_ptr<ImuCalibration> ImuParameters;
  typedef Eigen::Matrix<double, 9, 1> SpeedAndBias; // [v, bg, ba]

  // The base in ceres we derive from
  typedef ceres::SizedCostFunction<15, 7, 9, 7, 9> base_t;

  // The number of residuals
  static const int kNumResiduals = 15;

  // The type of the covariance.
  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  // The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  // The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

  // The type of the Jacobian w.r.t. poses --
  // This is w.r.t. minimal tangential space coordinates...
  typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

  // The type of Jacobian w.r.t. Speed and biases
  typedef Eigen::Matrix<double, 15, 9> jacobian1_t;

  // Default constructor -- assumes information recomputation.
  ImuCostFunctor() = default;

  // Trivial destructor.
  virtual ~ImuCostFunctor() = default;

  // Construct with measurements and parameters.
  // imu_measurements The IMU measurements including timestamp
  // imu_parameters The parameters to be used.
  // t_0 Start time.
  // t_1 End time.
  ImuCostFunctor(const ImuMeasurements& imu_measurements,
           const ImuParameters& imu_parameters, 
           const double& t_0, const double& t_1)
  {
    // @ToDo: do I need this?
    //std::lock_guard<std::mutex> lock(preintegration_mutex_);
    setImuMeasurements(imu_measurements);
    setImuParameters(imu_parameters);
    setT0(t_0);
    setT1(t_1);

    GVI_FUSION_ASSERT_STREAM(t0_ >= imu_measurements.front().t, 
    "First IMU measurement included in ImuError is not old enough!\n" << 
    t0_ << " <= " << imu_measurements.front().t);
    GVI_FUSION_ASSERT_STREAM(t1_ <= imu_measurements.back().t, 
    "Last IMU measurement included in ImuError is not new enough!"<< 
    t1_ << " >= " << imu_measurements.back().t);
  }

  /**
   * Propagates pose, speeds and biases with given IMU measurements.
   *  This can be used externally to perform propagation
   * [in] imu_measurements The IMU measurements including timestamp
   * [in] imu_params The parameters to be used.
   * [inout] T_WB Start pose.
   * [inout] speed_and_biases Start speed and biases.
   * [in] t_start Start time.
   * [in] t_end End time.
   * [out] covariance Covariance for GIVEN start states.
   * [out] jacobian Jacobian w.r.t. start states.
   * Number of integration steps.
   */
  static int propagation(const ImuMeasurements& imu_measurements,
                         const ImuParameters& imu_params,
                         Sophus::SE3d& T_WB,
                         SpeedAndBias& speed_and_biases,
                         const double& t_start, const double& t_end,
                         covariance_t* covariance = nullptr,
                         jacobian_t* jacobian = nullptr)
  {
    const double t_start_adjusted = t_start;
    const double t_end_adjusted = t_end;

    // ToDo: Remove if not necessary
    //const double t_start_adjusted = t_start - imu_params->delay_imu_cam;
    //const double t_end_adjusted = t_end - imu_params->delay_imu_cam;

    // sanity check:
    GVI_FUSION_ASSERT_STREAM(imu_measurements.front().t <= t_start_adjusted, 
    "first imu measurement newer than frame.");
    if (!(imu_measurements.back().t >= t_end_adjusted))
    {
      GVI_FUSION_ASSERT_STREAM(false, 
      "last imu measurement older than frame.");
      return -1;  // nothing to do...
    }

    // initial condition
    Eigen::Vector3d r_0 = T_WB.translation();
    Eigen::Quaterniond q_WB_0 = T_WB.unit_quaternion();
    Eigen::Matrix3d C_WB_0 = T_WB.rotationMatrix();

    // increments (initialise with identity)
    Eigen::Quaterniond Delta_q(1,0,0,0);
    Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
    Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

    // cross matrix accumulatrion
    Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();

    // sub-Jacobians
    Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();

    // the Jacobian of the increment (w/o biases)
    Eigen::Matrix<double,15,15> P_delta = Eigen::Matrix<double,15,15>::Zero();

    double Delta_t = 0;
    bool has_started = false;
    int num_propagated = 0;

    double time = t_start_adjusted;
    for (size_t i = 0; i < imu_measurements.size()-1; ++i)
    {
      Eigen::Vector3d omega_S_0 = imu_measurements[i].ang_vel;
      Eigen::Vector3d acc_S_0 = imu_measurements[i].lin_acc;
      Eigen::Vector3d omega_S_1 = imu_measurements[i + 1].ang_vel;
      Eigen::Vector3d acc_S_1 = imu_measurements[i + 1].lin_acc;
      double nexttime = imu_measurements[i + 1].t;

      // time delta
      double dt = nexttime - time;

      if (t_end_adjusted < nexttime)
      {
        double interval = nexttime - imu_measurements[i].t;
        nexttime = t_end_adjusted;
        dt = nexttime - time;
        const double r = dt / interval;
        omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt <= 0.0)
      {
        continue;
      }
      Delta_t += dt;

      if (!has_started)
      {
        has_started = true;
        const double r = dt / (nexttime - imu_measurements[i].t);
        omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
      }

      // ensure integrity
      double sigma_g_c = imu_params->gyro_noise_density_;
      double sigma_a_c = imu_params->acc_noise_density_;

      if (std::abs(omega_S_0[0]) > imu_params->g_max_
          || std::abs(omega_S_0[1]) > imu_params->g_max_
          || std::abs(omega_S_0[2]) > imu_params->g_max_
          || std::abs(omega_S_1[0]) > imu_params->g_max_
          || std::abs(omega_S_1[1]) > imu_params->g_max_
          || std::abs(omega_S_1[2]) > imu_params->g_max_)
      {
        sigma_g_c *= 100;
        LOG(WARNING) << "gyr saturation";
      }

      if (std::abs(acc_S_0[0]) > imu_params->a_max_
          || std::abs(acc_S_0[1]) > imu_params->a_max_
          || std::abs(acc_S_0[2]) > imu_params->a_max_
          || std::abs(acc_S_1[0]) > imu_params->a_max_
          || std::abs(acc_S_1[1]) > imu_params->a_max_
          || std::abs(acc_S_1[2]) > imu_params->a_max_)
      {
        sigma_a_c *= 100;
        LOG(WARNING) << "acc saturation";
      }

      // actual propagation
      // orientation:
      Eigen::Quaterniond dq;
      const Eigen::Vector3d omega_S_true =
          (0.5 *(omega_S_0+omega_S_1) - speed_and_biases.segment<3>(3));
      const double theta_half = omega_S_true.norm() * 0.5 * dt;
      const double sinc_theta_half = sinc(theta_half);
      const double cos_theta_half = cos(theta_half);
      dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
      dq.w() = cos_theta_half;
      Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
      // rotation matrix integral:
      const Eigen::Matrix3d C = Delta_q.toRotationMatrix();
      const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
      const Eigen::Vector3d acc_S_true =
          (0.5 * (acc_S_0 + acc_S_1) - speed_and_biases.segment<3>(6));
      const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 *(C + C_1) * dt;
      const Eigen::Vector3d acc_integral_1 =
          acc_integral + 0.5 * (C + C_1) * acc_S_true * dt;
      // rotation matrix double integral:
      C_doubleintegral += C_integral * dt + 0.25 * (C + C_1) * dt * dt;
      acc_doubleintegral +=
          acc_integral * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

      // Jacobian parts
      dalpha_db_g += dt * C_1;
      const Eigen::Matrix3d cross_1 =
          dq.inverse().toRotationMatrix()* cross +
          expmapDerivativeSO3(omega_S_true * dt)* dt;
      const Eigen::Matrix3d acc_S_x = skewSymmetric(acc_S_true);
      Eigen::Matrix3d dv_db_g_1 =
          dv_db_g + 0.5 * dt * (C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
      dp_db_g +=
          dt * dv_db_g
          + 0.25 * dt * dt *(C * acc_S_x * cross + C_1 * acc_S_x * cross_1);

      // covariance propagation
      if (covariance)
      {
        Eigen::Matrix<double,15,15> F_delta =
            Eigen::Matrix<double,15,15>::Identity();
        // transform
        F_delta.block<3,3>(0,3) =
            -skewSymmetric(acc_integral * dt + 0.25 * (C + C_1)
                          * acc_S_true * dt * dt);
        F_delta.block<3,3>(0,6) = Eigen::Matrix3d::Identity()* dt;
        F_delta.block<3,3>(0,9) =
            dt *dv_db_g
            + 0.25 * dt * dt *(C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
        F_delta.block<3,3>(0,12) = -C_integral* dt + 0.25 *(C + C_1) * dt * dt;
        F_delta.block<3,3>(3,9) = -dt * C_1;
        F_delta.block<3,3>(6,3) =
            -skewSymmetric(0.5 *(C + C_1) * acc_S_true * dt);
        F_delta.block<3,3>(6,9) =
            0.5 * dt *(C * acc_S_x * cross + C_1 * acc_S_x * cross_1);
        F_delta.block<3,3>(6,12) = -0.5 *(C + C_1) * dt;
        P_delta = F_delta * P_delta * F_delta.transpose();
        // add noise. Note that transformations with rotation matrices can be
        // ignored, since the noise is isotropic.
        //F_tot = F_delta*F_tot;
        const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
        P_delta(3,3) += sigma2_dalpha;
        P_delta(4,4) += sigma2_dalpha;
        P_delta(5,5) += sigma2_dalpha;
        const double sigma2_v = dt * sigma_a_c * imu_params->acc_noise_density_;
        P_delta(6,6) += sigma2_v;
        P_delta(7,7) += sigma2_v;
        P_delta(8,8) += sigma2_v;
        const double sigma2_p = 0.5 * dt * dt *sigma2_v;
        P_delta(0,0) += sigma2_p;
        P_delta(1,1) += sigma2_p;
        P_delta(2,2) += sigma2_p;
        const double sigma2_b_g =
            dt * imu_params->gyro_random_walk_ * imu_params->gyro_random_walk_;
        P_delta(9,9)   += sigma2_b_g;
        P_delta(10,10) += sigma2_b_g;
        P_delta(11,11) += sigma2_b_g;
        const double sigma2_b_a =
            dt * imu_params->acc_random_walk_ * imu_params->acc_random_walk_;
        P_delta(12,12) += sigma2_b_a;
        P_delta(13,13) += sigma2_b_a;
        P_delta(14,14) += sigma2_b_a;
      }

      // memory shift
      Delta_q = Delta_q_1;
      C_integral = C_integral_1;
      acc_integral = acc_integral_1;
      cross = cross_1;
      dv_db_g = dv_db_g_1;
      time = nexttime;

      ++num_propagated;

      if (nexttime == t_end_adjusted)
        break;

    }

    // actual propagation output:
    const Eigen::Vector3d g_W = imu_params->g_W_;
    Eigen::Quaterniond q_WB = q_WB_0 * Delta_q;
    Eigen::Vector3d p_WB = r_0 + speed_and_biases.head<3>() * Delta_t 
    + C_WB_0 * (acc_doubleintegral/*-C_doubleintegral*speedAndBiases.segment<3>(6)*/) 
    - 0.5 * g_W * Delta_t * Delta_t;
    T_WB = Sophus::SE3d(q_WB, p_WB);
    speed_and_biases.head<3>() += C_WB_0 * (acc_integral/*-C_integral*speedAndBiases.segment<3>(6)*/) - g_W * Delta_t;

    // assign Jacobian, if requested
    if (jacobian)
    {
      Eigen::Matrix<double,15,15>& F = *jacobian;
      F.setIdentity(); // holds for all states, including d/dalpha, d/db_g, d/db_a
      F.block<3,3>(0,3) = -skewSymmetric(C_WB_0 * acc_doubleintegral);
      F.block<3,3>(0,6) = Eigen::Matrix3d::Identity() * Delta_t;
      F.block<3,3>(0,9) = C_WB_0 * dp_db_g;
      F.block<3,3>(0,12) = -C_WB_0 * C_doubleintegral;
      F.block<3,3>(3,9) = -C_WB_0 * dalpha_db_g;
      F.block<3,3>(6,3) = -skewSymmetric(C_WB_0 * acc_integral);
      F.block<3,3>(6,9) = C_WB_0 * dv_db_g;
      F.block<3,3>(6,12) = -C_WB_0 * C_integral;
    }

    // overall covariance, if requested
    if (covariance)
    {
      Eigen::Matrix<double,15,15>& P = *covariance;
      // transform from local increments to actual states
      Eigen::Matrix<double,15,15> T = Eigen::Matrix<double,15,15>::Identity();
      T.topLeftCorner<3,3>() = C_WB_0;
      T.block<3,3>(3,3) = C_WB_0;
      T.block<3,3>(6,6) = C_WB_0;
      P = T * P_delta * T.transpose();
    }
    return num_propagated;
  }

  /**
   * Propagates pose, speeds and biases with given IMU measurements.
   * This is not actually const, since the re-propagation must somehow
   *          be stored...
   * [in] T_WB Start pose.
   * [in] speed_and_biases Start speed and biases.
   * Number of integration steps.
   */
  int redoPreintegration(const Sophus::SE3d& T_WB, 
                         const SpeedAndBias& speed_and_biases) const
  {
    // ToDo: do I need this?
    // ensure unique access
    //std::lock_guard<std::mutex> lock(preintegration_mutex_);

    // now the propagation
    double time = t0_;

    // sanity check:
    GVI_FUSION_ASSERT_STREAM(imu_measurements_.front().t <= time, 
    "first imu measurement newer than frame.");
    if (!(imu_measurements_.back().t >= t1_))
    {
      return -1;  // nothing to do...
    }

    // increments (initialise with identity)
    Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
    C_integral_ = Eigen::Matrix3d::Zero();
    C_doubleintegral_ = Eigen::Matrix3d::Zero();
    acc_integral_ = Eigen::Vector3d::Zero();
    acc_doubleintegral_ = Eigen::Vector3d::Zero();

    // cross matrix accumulation
    cross_ = Eigen::Matrix3d::Zero();

    // sub-Jacobians
    dalpha_db_g_ = Eigen::Matrix3d::Zero();
    dv_db_g_ = Eigen::Matrix3d::Zero();
    dp_db_g_ = Eigen::Matrix3d::Zero();

    // the Jacobian of the increment (w/o biases)
    P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

    //Eigen::Matrix<double, 15, 15> F_tot;
    //F_tot.setIdentity();

    double delta_t = 0;
    bool has_started = false;
    bool last_iteration = false;
    int n_integrated = 0;
    for (size_t i = 0; i < imu_measurements_.size()-1; ++i)
    {
      Eigen::Vector3d omega_S_0 = imu_measurements_[i].ang_vel;
      Eigen::Vector3d acc_S_0 = imu_measurements_[i].lin_acc;
      Eigen::Vector3d omega_S_1 = imu_measurements_[i+1].ang_vel;
      Eigen::Vector3d acc_S_1 = imu_measurements_[i+1].lin_acc;
      double nexttime = imu_measurements_[i+1].t;

      // time delta
      double dt = nexttime - time;

      if (t1_ < nexttime)
      {
        double interval = nexttime - imu_measurements_[i].t;
        nexttime = t1_;
        last_iteration = true;
        dt = nexttime - time;
        const double r = dt / interval;
        omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
        acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
      }

      if (dt <= 0.0)
      {
        continue;
      }
      delta_t += dt;

      if (!has_started)
      {
        has_started = true;
        const double r = dt / (nexttime - imu_measurements_[i].t);
        omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
        acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
      }

      // ensure integrity
      double sigma_g_c = imu_parameters_->gyro_noise_density_;
      double sigma_a_c = imu_parameters_->acc_noise_density_;

      if (std::abs(omega_S_0[0]) > imu_parameters_->g_max_
          || std::abs(omega_S_0[1]) > imu_parameters_->g_max_
          || std::abs(omega_S_0[2]) > imu_parameters_->g_max_
          || std::abs(omega_S_1[0]) > imu_parameters_->g_max_
          || std::abs(omega_S_1[1]) > imu_parameters_->g_max_
          || std::abs(omega_S_1[2]) > imu_parameters_->g_max_)
      {
        sigma_g_c *= 100;
        LOG(WARNING)<< "gyr saturation";
      }

      if (std::abs(acc_S_0[0]) > imu_parameters_->a_max_
          || std::abs(acc_S_0[1]) > imu_parameters_->a_max_
          || std::abs(acc_S_0[2]) > imu_parameters_->a_max_
          || std::abs(acc_S_1[0]) > imu_parameters_->a_max_
          || std::abs(acc_S_1[1]) > imu_parameters_->a_max_
          || std::abs(acc_S_1[2]) > imu_parameters_->a_max_)
      {
        sigma_a_c *= 100;
        LOG(WARNING)<< "acc saturation";
      }

      // actual propagation
      // orientation:
      Eigen::Quaterniond dq;
      const Eigen::Vector3d omega_S_true =
          (0.5 * (omega_S_0 + omega_S_1) - speed_and_biases.segment<3>(3));
      const double theta_half = omega_S_true.norm() * 0.5 * dt;
      const double sinc_theta_half = sinc(theta_half);
      const double cos_theta_half = cos(theta_half);
      dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
      dq.w() = cos_theta_half;
      Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
      // rotation matrix integral:
      const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
      const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
      const Eigen::Vector3d acc_S_true =
          (0.5 * (acc_S_0 + acc_S_1) - speed_and_biases.segment<3>(6));
      const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
      const Eigen::Vector3d acc_integral_1 =
          acc_integral_ + 0.5 * (C + C_1) * acc_S_true * dt;
      // rotation matrix double integral:
      C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
      acc_doubleintegral_ +=
          acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

      // Jacobian parts
      dalpha_db_g_ += C_1 * expmapDerivativeSO3(omega_S_true * dt) * dt;
      const Eigen::Matrix3d cross_1 =
          dq.inverse().toRotationMatrix() * cross_
          + expmapDerivativeSO3(omega_S_true * dt) * dt;
      const Eigen::Matrix3d acc_S_x = skewSymmetric(acc_S_true);
      Eigen::Matrix3d dv_db_g_1 =
          dv_db_g_ + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
      dp_db_g_ +=
          dt * dv_db_g_
          + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

      // covariance propagation
      Eigen::Matrix<double, 15, 15> F_delta =
          Eigen::Matrix<double, 15, 15>::Identity();
      // transform
      F_delta.block<3, 3>(0, 3) =
          -skewSymmetric(acc_integral_ * dt
                        + 0.25 * (C + C_1) * acc_S_true * dt * dt);
      F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
      F_delta.block<3, 3>(0, 9) =
          dt * dv_db_g_
          + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
      F_delta.block<3, 3>(0, 12) = -C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
      F_delta.block<3, 3>(3, 9) = -dt * C_1;
      F_delta.block<3, 3>(6, 3) =
          -skewSymmetric(0.5 * (C + C_1) * acc_S_true * dt);
      F_delta.block<3, 3>(6, 9) =
          0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
      F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
      P_delta_ = F_delta * P_delta_ * F_delta.transpose();
      // add noise. Note that transformations with rotation matrices can be
      // ignored, since the noise is isotropic.
      //F_tot = F_delta*F_tot;
      const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
      P_delta_(3, 3) += sigma2_dalpha;
      P_delta_(4, 4) += sigma2_dalpha;
      P_delta_(5, 5) += sigma2_dalpha;
      const double sigma2_v = dt * sigma_a_c * sigma_a_c;
      P_delta_(6, 6) += sigma2_v;
      P_delta_(7, 7) += sigma2_v;
      P_delta_(8, 8) += sigma2_v;
      const double sigma2_p = 0.5 * dt * dt * sigma2_v;
      P_delta_(0, 0) += sigma2_p;
      P_delta_(1, 1) += sigma2_p;
      P_delta_(2, 2) += sigma2_p;
      const double sigma2_b_g =
          dt * imu_parameters_->gyro_random_walk_ * imu_parameters_->gyro_random_walk_;
      P_delta_(9, 9) += sigma2_b_g;
      P_delta_(10, 10) += sigma2_b_g;
      P_delta_(11, 11) += sigma2_b_g;
      const double sigma2_b_a =
          dt * imu_parameters_->acc_random_walk_* imu_parameters_->acc_random_walk_;
      P_delta_(12, 12) += sigma2_b_a;
      P_delta_(13, 13) += sigma2_b_a;
      P_delta_(14, 14) += sigma2_b_a;

      // memory shift
      Delta_q_ = Delta_q_1;
      C_integral_ = C_integral_1;
      acc_integral_ = acc_integral_1;
      cross_ = cross_1;
      dv_db_g_ = dv_db_g_1;
      time = nexttime;

      ++n_integrated;

      if (last_iteration)
        break;

    }

    // store the reference (linearisation) point
    speed_and_biases_ref_ = speed_and_biases;

    // get the weighting:
    // enforce symmetric
    P_delta_ = 0.5 * P_delta_ + 0.5 * P_delta_.transpose().eval();

    // calculate inverse
    information_ = P_delta_.inverse();
    information_ = 0.5 * information_ + 0.5 * information_.transpose().eval();

    // square root
    Eigen::LLT<information_t> lltOfInformation(information_);
    square_root_information_ = lltOfInformation.matrixL().transpose();

    return n_integrated;
  }

  // setters
  void setRedo(const bool redo = true) const
  {
    redo_ = redo;
  }

  // (Re)set the parameters.
  // [in] imuParameters The parameters to be used.
  void setImuParameters(const ImuParameters& imu_parameters)
  {
    imu_parameters_ = imu_parameters;
  }

  // (Re)set the measurements
  void setImuMeasurements(const ImuMeasurements& imu_measurements)
  {
    imu_measurements_ = imu_measurements;
  }

  // (Re)set the start time.
  // t_0 Start time.
  void setT0(const double& t_0) { t0_ = t_0; }

  // (Re)set the end time.
  // t_1 End time.
  void setT1(const double& t_1) { t1_ = t_1; }

  // getters

  // Get the IMU Parameters.
  // return the IMU parameters.
  const ImuParameters& imuParameters() const
  {
    return imu_parameters_;
  }

  // Get the IMU measurements.
  const ImuMeasurements imuMeasurements() const
  {
    return imu_measurements_;
  }

  // Get the start time.
  double t0() const { return t0_; }

  // Get the end time.
  double t1() const { return t1_; }

  // error term and Jacobian implementation
  /**
   * This evaluates the error term and additionally computes the Jacobians.
   * parameters Pointer to the parameters (see ceres)
   * residuals Pointer to the residual vector (see ceres)
   * jacobians Pointer to the Jacobians (see ceres)
   * success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals, double** jacobians) const
  {
    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
  }

  /**
   * This evaluates the error term and additionally computes
   * the Jacobians in the minimal internal representation.
   * parameters Pointer to the parameters (see ceres)
   * residuals Pointer to the residual vector (see ceres)
   * jacobians Pointer to the Jacobians (see ceres)
   * jacobians_minimal Pointer to the minimal Jacobians
   * (equivalent to jacobians).
   * Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobians_minimal) const
  {
    // get poses
    Eigen::Map<Sophus::SE3d const> const T_WB_0(parameters[0]);
    Eigen::Map<Sophus::SE3d const> const T_WB_1(parameters[2]);

    // get speed and bias
    SpeedAndBias speed_and_biases_0;
    SpeedAndBias speed_and_biases_1;
    for (size_t i = 0; i < 9; ++i)
    {
      speed_and_biases_0[i] = parameters[1][i];
      speed_and_biases_1[i] = parameters[3][i];
    }

    // this will NOT be changed:
    const Eigen::Matrix3d C_WB_0 = T_WB_0.rotationMatrix();
    const Eigen::Matrix3d C_B0_W = C_WB_0.transpose();

    // call the propagation
    const double delta_t = t1_ - t0_;
    Eigen::Matrix<double, 6, 1> Delta_b;
    Delta_b = speed_and_biases_0.tail<6>() - speed_and_biases_ref_.tail<6>();
    
    redo_ = redo_ || (Delta_b.head<3>().norm() * delta_t > 0.0001);

    if (redo_)
    {
      redoPreintegration(T_WB_0, speed_and_biases_0);
      redoCounter_++;
      Delta_b.setZero();
      redo_ = false;
      /*if (redoCounter_ > 1) {
        std::cout << "pre-integration no. " << redoCounter_ << std::endl;
      }*/
    }

    // actual propagation output:
    const Eigen::Vector3d g_W = imu_parameters_->g_W_;

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double,15,15> F0 =
        Eigen::Matrix<double,15,15>::Identity(); // holds for d/db_g, d/db_a
    const Eigen::Vector3d delta_p_est_W =
        T_WB_0.translation() - T_WB_1.translation()
        + speed_and_biases_0.head<3>()* delta_t - 0.5 * g_W* delta_t * delta_t;
    const Eigen::Vector3d delta_v_est_W = speed_and_biases_0.head<3>()
        - speed_and_biases_1.head<3>() - g_W * delta_t;
    const Eigen::Quaterniond Dq =
        deltaQ(-dalpha_db_g_*Delta_b.head<3>())*Delta_q_;
    F0.block<3,3>(0,0) = C_B0_W;
    F0.block<3,3>(0,3) = C_B0_W * skewSymmetric(delta_p_est_W);
    F0.block<3,3>(0,6) = C_B0_W * Eigen::Matrix3d::Identity()* delta_t;
    F0.block<3,3>(0,9) = dp_db_g_;
    F0.block<3,3>(0,12) = -C_doubleintegral_;
    F0.block<3,3>(3,3) =
        (quaternionPlusMatrix(Dq * T_WB_1.unit_quaternion().inverse()) *
        quaternionOplusMatrix(T_WB_0.unit_quaternion())).topLeftCorner<3,3>();
    F0.block<3,3>(3,9) =
        (quaternionOplusMatrix(T_WB_1.unit_quaternion().inverse() *
                              T_WB_0.unit_quaternion()) *
        quaternionOplusMatrix(Dq)).topLeftCorner<3,3>() * (-dalpha_db_g_);
    F0.block<3,3>(6,3) = C_B0_W * skewSymmetric(delta_v_est_W);
    F0.block<3,3>(6,6) = C_B0_W;
    F0.block<3,3>(6,9) = dv_db_g_;
    F0.block<3,3>(6,12) = -C_integral_;

    // assign Jacobian w.r.t. x1
    Eigen::Matrix<double,15,15> F1 =
        -Eigen::Matrix<double,15,15>::Identity(); // holds for the biases
    F1.block<3,3>(0,0) = -C_B0_W;
    F1.block<3,3>(3,3) =
        -(quaternionPlusMatrix(Dq) *
          quaternionOplusMatrix(T_WB_0.unit_quaternion()) *
          quaternionPlusMatrix(T_WB_1.unit_quaternion().inverse()))
        .topLeftCorner<3,3>();
    F1.block<3,3>(6,6) = -C_B0_W;

    // the overall error vector
    Eigen::Matrix<double, 15, 1> error;
    error.segment<3>(0) =
        C_B0_W * delta_p_est_W + acc_doubleintegral_ + F0.block<3,6>(0,9)*Delta_b;
    error.segment<3>(3) =
        2.0 * (Dq * (T_WB_1.unit_quaternion().inverse() *
                    T_WB_0.unit_quaternion())).vec();
    //2*T_WB_0.unit_quaternion()*Dq*T_WB_1.unit_quaternion().inverse();//
    error.segment<3>(6) =
        C_B0_W * delta_v_est_W + acc_integral_ + F0.block<3,6>(6,9)*Delta_b;
    error.tail<6>() = speed_and_biases_0.tail<6>() - speed_and_biases_1.tail<6>();

    // error weighting
    Eigen::Map<Eigen::Matrix<double, 15, 1> > weighted_error(residuals);
    weighted_error = square_root_information_ * error;

    // get the Jacobians
    if (jacobians != nullptr)
    {
      if (jacobians[0] != nullptr)
      {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J0_minimal =
            square_root_information_ * F0.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        liftJacobian(parameters[0], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J0(
              jacobians[0]);
        J0 = J0_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobians_minimal != nullptr)
        {
          if (jacobians_minimal[0] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> >
                J0_minimal_mapped(jacobians_minimal[0]);
            J0_minimal_mapped = J0_minimal;
          }
        }
      }
      if (jacobians[1] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> >
            J1(jacobians[1]);
        J1 = square_root_information_ * F0.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobians_minimal != nullptr)
        {
          if (jacobians_minimal[1] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> >
                J1_minimal_mapped(jacobians_minimal[1]);
            J1_minimal_mapped = J1;
          }
        }
      }
      if (jacobians[2] != nullptr)
      {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J2_minimal = square_root_information_
            * F1.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        liftJacobian(parameters[2], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J2(
              jacobians[2]);
        J2 = J2_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobians_minimal != nullptr)
        {
          if (jacobians_minimal[2] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> >
                J2_minimal_mapped(jacobians_minimal[2]);
            J2_minimal_mapped = J2_minimal;
          }
        }
      }
      if (jacobians[3] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> >
            J3(jacobians[3]);
        J3 = square_root_information_ * F1.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobians_minimal != nullptr)
        {
          if (jacobians_minimal[3] != nullptr)
          {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> >
                J3_minimal_mapped(jacobians_minimal[3]);
            J3_minimal_mapped = J3;
          }
        }
      }
    }
    //}

    return true;
  }

  // sizes
  // Residual dimension.
  size_t residualDim() const
  {
    return kNumResiduals;
  }

  // Number of parameter blocks.
  virtual size_t parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  // Dimension of an individual parameter block.
  // [in] parameter_block_idx Index of the parameter block of interest.
  // The dimension.
  size_t parameterBlockDim(size_t parameter_block_idx) const
  {
    return base_t::parameter_block_sizes().at(parameter_block_idx);
  }

  // Return parameter block type as string
  virtual ErrorType typeInfo() const
  {
    return ErrorType::IMUError;
  }

 protected:
  // parameters
  ImuParameters imu_parameters_; ///< The IMU parameters.

  // measurements
  ImuMeasurements imu_measurements_;

  // times
  double t0_; ///< The start time (i.e. time of the first set of states).
  double t1_; ///< The end time (i.e. time of the sedond set of states).

  // preintegration stuff. the mutable is a TERRIBLE HACK, but what can I do.
  mutable std::mutex preintegration_mutex_;
  ///< Protect access of intermediate results.

  // increments (initialise with identity)
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0);
  ///< Intermediate result
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero();
  ///< Intermediate result
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero();
  ///< Intermediate result
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero();
  ///< Intermediate result
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero();
  ///< Intermediate result

  // cross matrix accumulatrion
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero();
  ///< Intermediate result

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();
  ///< Intermediate result
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();
  ///< Intermediate result
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero();
  ///< Intermediate result

  /// \brief The Jacobian of the increment (w/o biases).
  mutable Eigen::Matrix<double,15,15> P_delta_ =
      Eigen::Matrix<double,15,15>::Zero();

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBias speed_and_biases_ref_ = SpeedAndBias::Zero();

  mutable bool redo_ = true;
  ///< Keeps track of whether or not this redoPreintegration() needs to be done.
  mutable int redoCounter_ = 0;
  ///< Counts the number of preintegrations for statistics.

  // information matrix and its square root
  mutable information_t information_;
  ///< The information matrix for this error term.
  mutable information_t square_root_information_;
  ///< The square root information matrix for this error term.

};

