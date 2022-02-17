/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This code has been adapted from: 
https://gitlab.com/tum-vision/lie-spline-experiments/-/blob/master/include/ceres_calib_spline_split.h
*/

/* References:
[1] C. Sommer et al, Efficient Derivative Computation for Cumulative B-Splines on Lie Groups, CVPR2020.
[2] S. Lovegrov et al, Spline Fusion: A continuous-time representation for visual-inertial fusion 
with application to rolling shutter cameras, BMVC2013. 
[3] E. Mueller, Continuous-time visual-inertial odometry for event cameras, TRO2018.
*/

/*
Reference frames (spatial and temporal):
W: (fixed) world frame. Where the global measurements are expressed in.
G: (fixed) colmap frame. Colmap trajectory is expressed in this coordinate frame.
B: (moving) body (= imu) frame.
C: (moving) camera frame.
P: rigid point attached to B 
(e.g., position of the GPS antenna or nodal point of the prism tracked by a total station).
t_gp: time frame of the global position sensor.
t_imu: imu times.
t_cam: frame times.

Time offset formula to transform t from time frame a to time frame b:
t_b = t_a + t_a_b_offset
t_a_b_offset = -t_b_a_offset
E.g.,
t_imu = t_cam + t_cam_imu_offset
*/

#pragma once

#include <iostream>

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

#include "optimization/continuous_time/global_position_cost_functor.h"
#include "optimization/continuous_time/imu_cost_functor.h"
#include "optimization/continuous_time/reprojection_cost_functor.h"
#include "optimization/continuous_time/orientation_cost_functor.h"
#include "optimization/continuous_time/position_cost_functor.h"
#include "optimization/gravity_norm_cost_functor.h"
#include "optimization/lie_local_parametrization.h"
#include "sensor/global_position_sensor.h"
#include "sensor/imu.h"
#include "util/assert.h"
#include "util/eigen_utils.h"

using namespace gvi_fusion;

// Default OLD_TIME_DERIV=false, it uses [1] to compute spline derivatives.
// If OLD_TIME_DERIV=true, it uses [2], [3] to compute spline derivatives.
template <int _N, bool OLD_TIME_DERIV = false>
class BSplineSO3R3 {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int N_ = _N;        // Order of the spline.
  static constexpr int DEG = _N - 1;  // Degree of the spline.
  static constexpr int Nb_ = 4;        // Order of the bias spline.

  static constexpr double ns_to_s_ = 1e-9;  // Nanosecond to second conversion
  static constexpr double s_to_ns_ = 1e9;   // Second to nanosecond conversion

  BSplineSO3R3(int64_t time_interval_ns, int64_t start_time_ns = 0)
  : dt_ns_(time_interval_ns), start_t_ns_(start_time_ns) 
  {
    dt_s_ = dt_ns_ * ns_to_s_;
    inv_dt_ = s_to_ns_ / dt_ns_;

    t_s_gp_corr_s_ = 0.0;
    scale_corr_ = 0.0;

    R_b_c_corr_ = Sophus::SO3d(Eigen::Matrix3d::Identity());
    p_b_c_corr_ << 0.0, 0.0, 0.0;
    t_offset_cam_imu_corr_s_ = 0.0;

    R_w_g_corr_ = Sophus::SO3d(Eigen::Matrix3d::Identity());
    p_w_g_corr_ << 0.0, 0.0, 0.0;

    // Overwritten by set functions
    g_.setZero();
    p_bp_.setZero();
    optimize_p_bp_ = true;
    optimize_gravity_ = false;
    optimize_scale_ = false;
    optimize_camimu_extrinsics_ = true;
    optimize_t_offset_s_gs_ = true;
    t_s_gp_init_s_ = 0.0;
    t_s_gp_s_ = 0.0;
    scale_ = 0.0;
    t_offset_cam_imu_s_ = 0.0;

    // Overwritten by initializeGrid()
    cell_size_px_ = 0;
    n_grid_columns_ = 0;
    n_grid_rows_ = 0;

    // Overwritten by initializeBiases()
    dt_ns_bias_ = 1e9;
    inv_dt_bias_ = s_to_ns_ / dt_ns_bias_;
    
    // Overwritten by loadOptimizationParams()
    n_max_outer_iters_ = 10;
    n_max_outer_iters_ = 5;
    num_threads_ = 4;
  };

  // Destructor.
  ~BSplineSO3R3() {};

  Eigen::Vector3d getAccel(int64_t time_ns, bool in_world_frame = false) const 
  {
    int64_t st_ns = (time_ns - start_t_ns_);

    GVI_FUSION_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns_);

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    GVI_FUSION_ASSERT_STREAM(
        size_t(s + N_) <= so3_knots_.size(),
        "s " << s << " N " << N_ << " knots.size() " << so3_knots_.size());

    Eigen::Vector3d accel;

    Sophus::SO3d rot;
    Eigen::Vector3d lin_accel_world;

    {
      std::vector<const double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }

      CeresSplineHelper<N_>::template evaluate_lie<double, Sophus::SO3>(
          &vec[0], u, inv_dt_, &rot);
    }

    {
      std::vector<const double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }

      CeresSplineHelper<N_>::template evaluate<double, 3, 2>(&vec[0], u, inv_dt_,
                                                            &lin_accel_world);
    }

    if (in_world_frame)
    {
      accel = lin_accel_world + g_;  
    }
    else
    {
      accel = rot.inverse() * (lin_accel_world + g_);
    }

    return accel;
  }

  Eigen::Vector3d getGyro(int64_t time_ns, bool in_world_frame = false) const {
    int64_t st_ns = (time_ns - start_t_ns_);

    GVI_FUSION_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns_);

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    GVI_FUSION_ASSERT_STREAM(
        size_t(s + N_) <= so3_knots_.size(),
        "s " << s << " N " << N_ << " knots.size() " << so3_knots_.size());

    Eigen::Vector3d gyro;

    std::vector<const double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }

    Sophus::SO3d rot_body_world;

    CeresSplineHelper<N_>::template evaluate_lie<double, Sophus::SO3>(
        &vec[0], u, inv_dt_, &rot_body_world, &gyro);

    if (in_world_frame)
    {
      Eigen::Vector3d gyro_world;
      gyro_world = rot_body_world.inverse() * gyro;
      return gyro_world;
    }
    else
    {
      return gyro;
    }
  }

  Sophus::SE3d getKnot(int i) const {
    GVI_FUSION_ASSERT_STREAM(i >= 0, "i " << i);
    GVI_FUSION_ASSERT_STREAM(i < static_cast<int>(numKnots()), "i " << i);
    return Sophus::SE3d(so3_knots_[i], r3_knots_[i]);
  }

  Sophus::SE3d getPose(int64_t time_ns) const {
    // Sampling time
    int64_t st_ns = (time_ns - start_t_ns_);

    GVI_FUSION_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns_);

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    GVI_FUSION_ASSERT_STREAM(
        size_t(s + N_) <= so3_knots_.size(),
        "s " << s << " N " << N_ << " knots.size() " << so3_knots_.size());

    Sophus::SE3d res;
    Sophus::SO3d rot;
    Eigen::Vector3d trans;

    {
      std::vector<const double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }

      CeresSplineHelper<N_>::template evaluate_lie<double, Sophus::SO3>(
          &vec[0], u, inv_dt_, &rot);
    }

    {
      std::vector<const double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }

      CeresSplineHelper<N_>::template evaluate<double, 3, 0>(&vec[0], u, inv_dt_,
                                                            &trans);
    }

    res = Sophus::SE3d(rot, trans);
    return res;
  }

  Eigen::Vector3d getAccelBias(int64_t time_ns) const 
  {
    // Sampling time
    int64_t st_ns = (time_ns - start_t_ns_);

    GVI_FUSION_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
    << " start_t_ns " << start_t_ns_);

    int64_t s = st_ns / dt_ns_bias_;
    double u = double(st_ns % dt_ns_bias_) / double(dt_ns_bias_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    GVI_FUSION_ASSERT_STREAM(
        size_t(s + N_) <= accel_bias_.size(),
        "s " << s << " N " << N_ << " accel_bias_.size() " << accel_bias_.size());

    Eigen::Vector3d bias;
    {
      std::vector<const double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(accel_bias_[s + i].data());
      }

      CeresSplineHelper<N_>::template evaluate<double, 3, 0>(&vec[0], u, inv_dt_bias_,
                                                            &bias);
    }

    return bias;
  }

  Eigen::Vector3d getGyroBias (int64_t time_ns) const
  { 
    // Sampling time
    int64_t st_ns = (time_ns - start_t_ns_);

    GVI_FUSION_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
    << " start_t_ns " << start_t_ns_);

    int64_t s = st_ns / dt_ns_bias_;
    double u = double(st_ns % dt_ns_bias_) / double(dt_ns_bias_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    GVI_FUSION_ASSERT_STREAM(
        size_t(s + N_) <= gyro_bias_.size(),
        "s " << s << " N " << N_ << " gyro_bias_.size() " << gyro_bias_.size());

    Eigen::Vector3d bias;

    {
      std::vector<const double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(gyro_bias_[s + i].data());
      }

      CeresSplineHelper<N_>::template evaluate<double, 3, 0>(&vec[0], u, inv_dt_bias_,
                                                            &bias);
    }

    return bias;
  }

  void initialize(const Sophus::SE3d& init, int num_knots) 
  {
    so3_knots_ = Eigen::aligned_vector<Sophus::SO3d>(num_knots, init.so3());
    r3_knots_ = Eigen::aligned_vector<Eigen::Vector3d>(num_knots, init.translation());
  }

  void initializeFromNumTraj(const Eigen::aligned_vector<Sophus::SO3d> init_so3_knots,
  const Eigen::aligned_vector<Eigen::Vector3d> init_r3_knots) 
  {

    size_t n_init_so3_knots = init_so3_knots.size();
    size_t n_init_r3_knots = init_r3_knots.size();
    GVI_FUSION_ASSERT_STREAM(n_init_so3_knots == n_init_r3_knots, 
    "n_init_so3_knots: " << n_init_so3_knots << ", n_init_r3_knots: " << n_init_r3_knots);

    for (size_t i = 0; i < n_init_so3_knots; i++)
    {
      so3_knots_.emplace_back(init_so3_knots[i]);
    }

    for (size_t i = 0; i < n_init_r3_knots; i++)
    {
      r3_knots_.emplace_back(init_r3_knots[i]);
    }
  }

  void initializeBiases(const double dt_s_bias, 
  Eigen::Vector3d& init_acc_bias, Eigen::Vector3d& init_gyro_bias) 
  {
    init_acc_bias_ = init_acc_bias;
    init_gyro_bias_ = init_gyro_bias;

    dt_ns_bias_ = static_cast<uint64_t>(dt_s_bias / ns_to_s_);
    inv_dt_bias_ = s_to_ns_ / dt_ns_bias_;

    double d = dt_s_ * r3_knots_.size();
    int num_knots = d / dt_s_bias;

    accel_bias_ = Eigen::aligned_vector<Eigen::Vector3d>(num_knots, init_acc_bias);
    gyro_bias_ = Eigen::aligned_vector<Eigen::Vector3d>(num_knots, init_gyro_bias);
  }

  void initializeGrid(Frame* frame)
  {
    cell_size_px_ = frame->cell_size_px;
    n_grid_columns_ = frame->n_grid_columns;
    n_grid_rows_ = frame->n_grid_rows;
  }

  void addRotationLocalParametrization()
  {
    int n_knots = static_cast<int>(numKnots()); 
    for (int i = 0; i < n_knots; i++) 
    {
      ceres::LocalParameterization* local_parameterization = 
      new LieLocalParameterization<Sophus::SO3d>();

      problem_.AddParameterBlock(so3_knots_[i].data(), 
      Sophus::SO3d::num_parameters, local_parameterization);
    }

    // cam-imu rotation
    ceres::LocalParameterization* local_parameterization = 
    new LieLocalParameterization<Sophus::SO3d>();

    problem_.AddParameterBlock(R_b_c_corr_.data(), 
    Sophus::SO3d::num_parameters, local_parameterization);

    problem_.AddParameterBlock(R_w_g_corr_.data(), 
    Sophus::SO3d::num_parameters, local_parameterization);
  }

  // Each frame is split in a grid.
  // Points included in the previous frame grid are tracked.
  // A new 2d point is added to free cell grid.
  // All the 2d points assigned to the grid will be used compute
  // the reprojection error for the corresponding frame
  void computeValidReprojections(Eigen::aligned_vector<Frame>& frames)
  {
    size_t n_cells = n_grid_columns_ * n_grid_rows_;

    std::vector<std::vector<Point2D*>> grid_k_min_1;
    grid_k_min_1.resize(n_cells);

    int k = 0;

    for (auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
      int64_t time_ns = static_cast<int64_t>(frame->timestamp * s_to_ns_);
      int64_t st_ns = (time_ns - start_t_ns_);

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > so3_knots_.size()) {continue;}

      std::vector<std::vector<Point2D*>> grid_k;
      grid_k.resize(n_cells);

      // track 2d points from previous grid
      for (size_t cell_id = 0; cell_id < n_cells; cell_id++)
      {
        for (size_t point_id = 0; point_id < grid_k_min_1[cell_id].size(); point_id++)
        {
          // use id of the corresponding 3d point to find
          // the 2d point on the current frame
          uint64_t point3D_id = grid_k_min_1[cell_id][point_id]->point3d_id;
          auto point3D_idx_it = 
          std::find(frame->points3D_ids.begin(), frame->points3D_ids.end(), point3D_id);
          if (point3D_idx_it != frame->points3D_ids.end())
          {
            size_t point3D_idx = point3D_idx_it - frame->points3D_ids.begin();
            
            // compute location of the 2d point in the new grid
            Point2D* p;
            p = &frame->points2D[point3D_idx];
            p->is_reproj_valid = true;

            int idx_clmn = p->uv.x() / cell_size_px_;
            int idx_row = p->uv.y() / cell_size_px_;
            int idx_grid = idx_clmn + idx_row * n_grid_columns_;
            grid_k[idx_grid].push_back(p);
          } 
        }
      }

      // add new 2d points to free cells
      for (size_t cell_id = 0; cell_id < n_cells; cell_id++)
      {
        if (grid_k[cell_id].size() > 0) continue;

        if (frame->grid[cell_id].size() == 0) continue;

        size_t point2D_idx = frame->grid[cell_id][0];
        Point2D* p;
        p = &frame->points2D[point2D_idx];
        uint64_t point3D_idx = p->point3d_id;
        GVI_FUSION_ASSERT_STREAM(points3D_.find(point3D_idx) != points3D_.end(), 
        "Point 3D " <<  point3D_idx << " not found!");
        p->is_reproj_valid = true;
        grid_k[cell_id].push_back(p);
      }

      // Compute num. of reprojection errors.
      // Check that the corresponding 3d point 
      // is not too close (< 30 cm) to the camera
      // and does not reproject behind the camera
      Sophus::SE3d T_WB = getPose(time_ns);
      Eigen::Matrix3d R_WC = T_WB.rotationMatrix() * cameras_[0].T_b_c.rotationMatrix();
      double scale = getScale();
      Eigen::Vector3d p_WC = scale * T_WB.rotationMatrix() * cameras_[0].T_b_c.translation() + 
      T_WB.translation();
      Eigen::Matrix3d R_CW = R_WC.inverse();
      Eigen::Vector3d p_CW = -1.0 * R_CW * p_WC;

      size_t n_reprojs = 0;
      for (size_t cell_id = 0; cell_id < n_cells; cell_id++)
      {
        for (size_t point_id = 0; point_id < grid_k[cell_id].size(); point_id++)
        {
          uint64_t point3D_idx = grid_k[cell_id][point_id]->point3d_id;
          Eigen::Vector3d p_WLi = points3D_.find(point3D_idx)->second.p_wl;
          Eigen::Vector3d p_CLi = R_CW * p_WLi + p_CW;
          if ((p_CLi.norm() < 0.3) || (p_CLi.z() < 0.0))
          { 
            grid_k[cell_id][point_id]->is_reproj_valid = false;
          }
          else
          {
            n_reprojs += 1;
          }
        }
      }
      frame->n_valid_reprojections = n_reprojs;

      // Update grid
      for (size_t cell_id = 0; cell_id < n_cells; cell_id++)
      {
        grid_k_min_1[cell_id].clear();
      }

      for (size_t cell_id = 0; cell_id < n_cells; cell_id++)
      {
        for (size_t point_id = 0; point_id < grid_k[cell_id].size(); point_id++)
        {
          grid_k_min_1[cell_id].push_back(grid_k[cell_id][point_id]);
        }
      }

      k++;
    }
  }

  void addAccelMeasurement(const Eigen::Vector3d& meas, int64_t time_ns)
  {
    int64_t t_offset_imu_cam_ns = 
    static_cast<int64_t>(-t_offset_cam_imu_s_ * s_to_ns_);
    int64_t st_ns = (time_ns + t_offset_imu_cam_ns - start_t_ns_);
    double st_s = st_ns * ns_to_s_;

    if (st_ns < 0) {return;}

    int64_t s = st_ns / dt_ns_;
    
    // Helper variables to compute u corrected
    // with time offset correction in the cost functor.
    int k_int = st_ns / dt_ns_;
    // This is just to avoid integers in the cost functor
    // (Maybe, it's not needed.)
    double k = static_cast<double>(k_int);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > so3_knots_.size()) {return;}

    int64_t st_bias_ns = (time_ns - start_t_ns_);
    int64_t s_bias = st_bias_ns / dt_ns_bias_;
    double u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);

    GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);
    /* In case the bias spline knot freq. is much lower than the main spline,
    we won't be able to sample the bias spline at the end of the trajectory.
    This means that we'd not able to compute imu errors.
    To avoid this, when we are at the end of the trajectory, 
    when the imu time exceeds the end time of the bias spline, 
    we just sample the latest available bias with a time older than imu time.
    In this case, the bias is just used to define the imu error and it's kept constant.
    */
    if (size_t(s_bias + N_) > accel_bias_.size()) 
    {
      st_bias_ns = getEndTnsBiasSpline() - start_t_ns_;
      s_bias = st_bias_ns / dt_ns_bias_;
      u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);
      GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);
    }

    using FunctorT = AccelerometerCostFunctor<N_>;

    FunctorT* functor = 
    new FunctorT(meas, st_s, k, dt_s_, inv_dt_, u_bias, inv_dt_bias_, 
    1.0 / imu_ptr_->calib_ptr_->acc_noise_density_);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for rotation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(4);
    }
    // parameter blocks for translation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    // parameter blocks for bias
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    // parameter block for gravity
    cost_function->AddParameterBlock(3);
    // parameter block for time offset
    cost_function->AddParameterBlock(1);

    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(r3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(accel_bias_[s_bias + i].data());
    }
    vec.emplace_back(g_.data());
    vec.emplace_back(&t_offset_cam_imu_corr_s_);

    problem_.AddResidualBlock(cost_function, NULL, vec);

    if(!optimize_gravity_)
    {
      problem_.SetParameterBlockConstant(g_.data());
    }

    if(!optimize_camimu_extrinsics_)
    {
      problem_.SetParameterBlockConstant(&t_offset_cam_imu_corr_s_);
    }

  }

  void addGyroMeasurement(const Eigen::Vector3d& meas, int64_t time_ns) 
  {
    int64_t t_offset_imu_cam_ns = 
    static_cast<int64_t>(-t_offset_cam_imu_s_ * s_to_ns_);
    int64_t st_ns = (time_ns + t_offset_imu_cam_ns - start_t_ns_);
    double st_s = st_ns * ns_to_s_;

    if (st_ns < 0) {return;}

    int64_t s = st_ns / dt_ns_;
    
    // Helper variables to compute u corrected
    // with time offset correction in the cost functor.
    int k_int = st_ns / dt_ns_;
    // This is just to avoid integers in the cost functor
    // (Maybe, it's not needed.)
    double k = static_cast<double>(k_int);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > so3_knots_.size()) {return;}

    int64_t st_bias_ns = (time_ns - start_t_ns_);
    int64_t s_bias = st_bias_ns / dt_ns_bias_;
    double u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);

    GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);
    // See: addAccelMeasurement() for the reason of this.
    if (size_t(s_bias + N_) > gyro_bias_.size()) 
    {
      st_bias_ns = getEndTnsBiasSpline() - start_t_ns_;
      s_bias = st_bias_ns / dt_ns_bias_;
      u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);
      GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);
    }

    using FunctorT = GyroCostFunctor<N_, Sophus::SO3, OLD_TIME_DERIV>;

    FunctorT* functor = 
    new FunctorT(meas, st_s, k, dt_s_, inv_dt_, u_bias, inv_dt_bias_, 
    1.0 / imu_ptr_->calib_ptr_->gyro_noise_density_);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
        new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for rotation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(4);
    }
    // parameter blocks for bias
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    // parameter block for time offset
    cost_function->AddParameterBlock(1);
    
    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(gyro_bias_[s_bias + i].data());
    }
    vec.emplace_back(&t_offset_cam_imu_corr_s_);

    problem_.AddResidualBlock(cost_function, NULL, vec);

    if(!optimize_camimu_extrinsics_)
    {
      problem_.SetParameterBlockConstant(&t_offset_cam_imu_corr_s_);
    }
  }

  // [in] bias_err_freq_hz: Sampling freq of bias spline
  // [in] bias_err_dt_s: dt s.t.: b(t+dt) - b(t) = 0
  void addImuBiasCost(double bias_err_freq_hz, double bias_err_dt_s) 
  {
    double err_freq_s = 1.0 / bias_err_freq_hz;
    double err_freq_ns = static_cast<uint64_t>(err_freq_s / ns_to_s_);

    int64_t t_start_ns = getStartTns();
    int64_t t_end_ns = getEndTnsBiasSpline();

    int64_t bias_err_dt_ns = static_cast<int64_t>(bias_err_dt_s*s_to_ns_);

    for (int64_t t_ns = t_start_ns; t_ns < t_end_ns; t_ns+=err_freq_ns)
    {
      int64_t st_ns_0 = (t_ns - start_t_ns_);
      int64_t st_ns_1 = (t_ns + bias_err_dt_ns - start_t_ns_);

      int64_t s_0 = st_ns_0 / dt_ns_bias_;
      int64_t s_1 = st_ns_1 / dt_ns_bias_;
      
      double u_0 = double(st_ns_0 % dt_ns_bias_) / double(dt_ns_bias_);
      double u_1 = double(st_ns_1 % dt_ns_bias_) / double(dt_ns_bias_);

      GVI_FUSION_ASSERT_STREAM(s_0 >= 0, "s " << s_0);
      if (size_t(s_0 + Nb_) > accel_bias_.size()) {continue;}

      GVI_FUSION_ASSERT_STREAM(s_1 >= 0, "s " << s_1);
      if (size_t(s_1 + Nb_) > accel_bias_.size()) {continue;}

      using FunctorT = BiasCostFunctor<Nb_>;

      FunctorT* acc_bias_functor = 
      new FunctorT(u_0, u_1, inv_dt_bias_, 1.0 / (0.001*imu_ptr_->calib_ptr_->acc_random_walk_));

      ceres::DynamicAutoDiffCostFunction<FunctorT>* acc_bias_cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(acc_bias_functor);

      for (int i = 0; i < Nb_; i++) 
      {
        acc_bias_cost_function->AddParameterBlock(3);
      }
      for (int i = 0; i < Nb_; i++) 
      {
        acc_bias_cost_function->AddParameterBlock(3);
      }
      acc_bias_cost_function->SetNumResiduals(3);

      std::vector<double*> acc_vec;
      for (int i = 0; i < Nb_; i++) 
      {
        acc_vec.emplace_back(accel_bias_[s_0 + i].data());
      }
      for (int i = 0; i < Nb_; i++) 
      {
        acc_vec.emplace_back(accel_bias_[s_1 + i].data());
      }

      problem_.AddResidualBlock(acc_bias_cost_function, NULL, acc_vec);

      // Gyro bias
      FunctorT* gyro_bias_functor = 
      new FunctorT(u_0, u_1, inv_dt_bias_, 1.0 / (0.001*imu_ptr_->calib_ptr_->gyro_random_walk_));

      ceres::DynamicAutoDiffCostFunction<FunctorT>* gyro_bias_cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(gyro_bias_functor);

      for (int i = 0; i < Nb_; i++) 
      {
        gyro_bias_cost_function->AddParameterBlock(3);
      }
      for (int i = 0; i < Nb_; i++) 
      {
        gyro_bias_cost_function->AddParameterBlock(3);
      }
      gyro_bias_cost_function->SetNumResiduals(3);

      std::vector<double*> gyro_vec;
      for (int i = 0; i < Nb_; i++) 
      {
        gyro_vec.emplace_back(gyro_bias_[s_0 + i].data());
      }
      for (int i = 0; i < Nb_; i++) 
      {
        gyro_vec.emplace_back(gyro_bias_[s_1 + i].data());
      }

      problem_.AddResidualBlock(gyro_bias_cost_function, NULL, gyro_vec);
    }

    // Final cost
    /*int64_t st_ns = getEndTnsBiasSpline() - start_t_ns_;
    int64_t s_bias = st_ns / dt_ns_bias_;
    double u_bias = double(st_ns % dt_ns_bias_) / double(dt_ns_bias_);
    
    using FunctorT = BiasTerminalCostFunctor<N_>;
    
    // Acc. terminal cost
    FunctorT* acc_bias_functor = 
    new FunctorT(u_bias, inv_dt_bias_, init_acc_bias_, 
    1.0 / imu_ptr_->calib_ptr_->acc_random_walk_);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* acc_bias_cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(acc_bias_functor);

    for (int i = 0; i < N_; i++) 
    {
      acc_bias_cost_function->AddParameterBlock(3);
    }
    acc_bias_cost_function->SetNumResiduals(3);

    std::vector<double*> acc_vec;
    for (int i = 0; i < N_; i++) 
    {
      acc_vec.emplace_back(accel_bias_[s_bias + i].data());
    }

    problem_.AddResidualBlock(acc_bias_cost_function, NULL, acc_vec);

    // Gyro. terminal cost
    FunctorT* gyro_bias_functor = 
    new FunctorT(u_bias, inv_dt_bias_, init_gyro_bias_, 
    1.0 / imu_ptr_->calib_ptr_->gyro_random_walk_);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* gyro_bias_cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(gyro_bias_functor);

    for (int i = 0; i < N_; i++) 
    {
      gyro_bias_cost_function->AddParameterBlock(3);
    }
    gyro_bias_cost_function->SetNumResiduals(3);

    std::vector<double*> gyro_vec;
    for (int i = 0; i < N_; i++) 
    {
      gyro_vec.emplace_back(gyro_bias_[s_bias + i].data());
    }

    problem_.AddResidualBlock(gyro_bias_cost_function, NULL, acc_vec);*/
  }

  void addConstantImuBiasCost() 
  {
    using FunctorT = ConstantBiasCostFunctor;

    // Acc bias
    FunctorT* acc_bias_functor = 
    new FunctorT(1.0 / imu_ptr_->calib_ptr_->acc_random_walk_);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* acc_bias_cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(acc_bias_functor);

    acc_bias_cost_function->AddParameterBlock(3);
    acc_bias_cost_function->SetNumResiduals(3);

    std::vector<double*> acc_vec;
    acc_vec.emplace_back(accel_bias_.data());

    problem_.AddResidualBlock(acc_bias_cost_function, NULL, acc_vec);

    // Gyro bias
    FunctorT* gyro_bias_functor = 
    new FunctorT(1.0 / imu_ptr_->calib_ptr_->gyro_random_walk_);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* gyro_bias_cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(gyro_bias_functor);

    gyro_bias_cost_function->AddParameterBlock(3);
    gyro_bias_cost_function->SetNumResiduals(3);

    std::vector<double*> gyro_vec;
    gyro_vec.emplace_back(gyro_bias_.data());

    problem_.AddResidualBlock(gyro_bias_cost_function, NULL, gyro_vec);
  }

  void addGlobalPosMeasurement(const Eigen::Vector3d& meas, int64_t time_ns, double noise)
  {
    int64_t t_s_gp_init_ns = static_cast<int64_t>(t_s_gp_s_ * s_to_ns_);
    int64_t st_ns = (time_ns - start_t_ns_ - t_s_gp_init_ns);
    double st_s = st_ns * ns_to_s_;

    if (st_ns < 0) {return;}

    int64_t s = st_ns / dt_ns_;

    // Helper variables to compute u corrected
    // with time offset correction in the cost functor.
    int k_int = st_ns / dt_ns_;
    // This is just to avoid integers in the cost functor
    // (Maybe, it's not needed.)
    double k = static_cast<double>(k_int);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > so3_knots_.size()) {return;}

    using FunctorT = GlobalPositionSensorCostFunctor<N_>;

    // @ToDo: include noise
    FunctorT* functor = 
    new FunctorT(meas, scale_, st_s, k, dt_s_, inv_dt_, 1.0 / noise);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for rotation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(4);
    }
    // parameter blocks for translation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    // parameter block for p_bp
    cost_function->AddParameterBlock(3);
    // parameter block for time offset correction
    cost_function->AddParameterBlock(1);
    // parameter block for scale correction
    cost_function->AddParameterBlock(1);

    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(r3_knots_[s + i].data());
    }
    vec.emplace_back(p_bp_.data());
    vec.emplace_back(&t_s_gp_corr_s_);
    vec.emplace_back(&scale_corr_);

    problem_.AddResidualBlock(cost_function, NULL, vec);

    // ToDo: choose from .yaml if set scale constant
    if (!optimize_scale_)
    {
      problem_.SetParameterBlockConstant(&scale_corr_);
    }

    if(!optimize_p_bp_)
    {
      problem_.SetParameterBlockConstant(p_bp_.data());
    }

    if(!optimize_t_offset_s_gs_)
    {
      problem_.SetParameterBlockConstant(&t_s_gp_corr_s_);
    }
  }

  void addImagePointsMeasurement(Frame* frame, int64_t time_ns, double noise)
  {
    int64_t st_ns = (time_ns - start_t_ns_);

    if (st_ns < 0) {return;}

    size_t n_reproj = frame->n_valid_reprojections;
    if (n_reproj == 0) {return;}

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > so3_knots_.size()) {return;}

    using FunctorT = ReprojectionCostFunctor<N_>;

    FunctorT* functor = 
    new FunctorT(frame, cameras_.data(), scale_, u, inv_dt_, 1.0 / noise);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for rotation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(4);
    }
    // parameter blocks for translation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    // parameter blocks for scale correction
    cost_function->AddParameterBlock(1);
    // parameter blocks for imu-cam rotatation correction
    cost_function->AddParameterBlock(4);
    // parameter blocks for imu-cam translation correction
    cost_function->AddParameterBlock(3);
    
    // parameter block for 3d points
    for (size_t i = 0; i < n_reproj; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    cost_function->SetNumResiduals(2*n_reproj);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(r3_knots_[s + i].data());
    }
    vec.emplace_back(&scale_corr_);
    vec.emplace_back(R_b_c_corr_.data());
    vec.emplace_back(p_b_c_corr_.data());
    for(const auto& point2d: frame->points2D) 
    {
      if(point2d.is_reproj_valid)
      {
        uint64_t point3D_idx = point2d.point3d_id;
        GVI_FUSION_ASSERT_STREAM(points3D_.find(point3D_idx) != points3D_.end(), 
        "Point 3D " <<  point3D_idx << " not found!");
        vec.emplace_back(points3D_.find(point3D_idx)->second.p_wl.data());
      }
    }
    
    // @ToDo: Load huber loss param.
    //problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(3.0), vec);
    problem_.AddResidualBlock(cost_function, new ceres::CauchyLoss(3.0), vec);

    // Set scale constant
    if (!optimize_scale_)
    {
      problem_.SetParameterBlockConstant(&scale_corr_);
    }

    // Set cam-imu extrinsics constant
    if(!optimize_camimu_extrinsics_)
    {
      problem_.SetParameterBlockConstant(R_b_c_corr_.data());
      problem_.SetParameterBlockConstant(p_b_c_corr_.data());
    }
  }

  void addPositionMeasurement(const Eigen::Vector3d& meas, int64_t time_ns, double noise)
  {
    int64_t st_ns = (time_ns - start_t_ns_);

    if (st_ns < 0) {return;}

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > r3_knots_.size()) {return;}

    using FunctorT = PositionCostFunctor<N_>;

    // @ToDo: include noise
    FunctorT* functor = 
    new FunctorT(meas, u, inv_dt_, 1.0 / noise);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for translation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(r3_knots_[s + i].data());
    }

    problem_.AddResidualBlock(cost_function, NULL, vec);
  }

  void addOrientationMeasurement(const Sophus::SO3d& meas, int64_t time_ns, double noise)
  {
    int64_t st_ns = (time_ns - start_t_ns_);

    if (st_ns < 0) {return;}

    int64_t s = st_ns / dt_ns_;
    double u = double(st_ns % dt_ns_) / double(dt_ns_);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > so3_knots_.size()) {return;}

    using FunctorT = OrientationCostFunctor<N_>;

    FunctorT* functor = 
    new FunctorT(meas, u, inv_dt_, 1.0 / noise);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for orientation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(4);
    }
    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }

    problem_.AddResidualBlock(cost_function, NULL, vec);
  }

  void addWorldColmapAlignmentMeasurement(const Eigen::Vector3d& meas, int64_t time_ns, double noise)
  {
    int64_t t_s_gp_init_ns = static_cast<int64_t>(t_s_gp_s_ * s_to_ns_);
    int64_t st_ns = (time_ns - start_t_ns_ - t_s_gp_init_ns);
    double st_s = st_ns * ns_to_s_;

    if (st_ns < 0) {return;}

    int64_t s = st_ns / dt_ns_;
    
    // Helper variables to compute u corrected
    // with time offset correction in the cost functor.
    int k_int = st_ns / dt_ns_;
    // This is just to avoid integers in the cost functor
    // (Maybe, it's not needed.)
    double k = static_cast<double>(k_int);

    GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
    if (size_t(s + N_) > so3_knots_.size()) {return;}

    using FunctorT = 
    GlobalPositionSensorWithAlignmentCorrectionCostFunctor<N_>;

    // @ToDo: include noise
    FunctorT* functor = 
    new FunctorT(meas, T_w_g_, scale_, st_s, k, dt_s_, inv_dt_, 1.0 / noise);

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for rotation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(4);
    }
    // parameter blocks for translation
    for (int i = 0; i < N_; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    // parameter block for time offset correction
    cost_function->AddParameterBlock(1);
    // parameter block for scale correction
    cost_function->AddParameterBlock(1);
    // parameter block for rotation correction
    cost_function->AddParameterBlock(4);
    // parameter block for translation correction
    cost_function->AddParameterBlock(3);
    // parameter block for p_bp
    cost_function->AddParameterBlock(3);

    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(so3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      vec.emplace_back(r3_knots_[s + i].data());
    }
    vec.emplace_back(&t_s_gp_corr_s_);
    vec.emplace_back(&scale_corr_);
    vec.emplace_back(R_w_g_corr_.data());
    vec.emplace_back(p_w_g_corr_.data());
    vec.emplace_back(p_bp_.data());
    
    problem_.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.10), vec);

    // Do not optimize spline
    for (int i = 0; i < N_; i++) 
    {
      problem_.SetParameterBlockConstant(so3_knots_[s + i].data());
    }
    for (int i = 0; i < N_; i++) 
    {
      problem_.SetParameterBlockConstant(r3_knots_[s + i].data());
    }

    if(!optimize_p_bp_)
    {
      problem_.SetParameterBlockConstant(p_bp_.data());
    }

    if(!optimize_t_offset_s_gs_)
    {
      problem_.SetParameterBlockConstant(&t_s_gp_corr_s_);
    }

  }

  void addGravityNormError()
  {
    using FunctorT = GravityNormCostFunctor;
    FunctorT* functor = new FunctorT();

    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    cost_function->AddParameterBlock(3);
    cost_function->SetNumResiduals(1);
    std::vector<double*> vec;
    vec.emplace_back(g_.data());

    problem_.AddResidualBlock(cost_function, NULL, vec);
  }

  Eigen::aligned_vector<double> evaluateAccelErrors(const Eigen::aligned_vector<ImuMeasurement>& measurements)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      ImuMeasurement meas = measurements[i];
      Eigen::Vector3d acc_meas = meas.lin_acc;
      
      int64_t time_ns = static_cast<int64_t>(meas.t * s_to_ns_);
      int64_t t_offset_imu_cam_ns = 
      static_cast<int64_t>(-t_offset_cam_imu_s_ * s_to_ns_);
      int64_t st_ns = (time_ns + t_offset_imu_cam_ns - start_t_ns_);
      double st_s = st_ns * ns_to_s_;

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      
      // Helper variables to compute u corrected
      // with time offset correction in the cost functor.
      int k_int = st_ns / dt_ns_;
      // This is just to avoid integers in the cost functor
      double k = static_cast<double>(k_int);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > so3_knots_.size()) {continue;}

      int64_t st_bias_ns = (time_ns - start_t_ns_);
      int64_t s_bias = st_bias_ns / dt_ns_bias_;
      double u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);

      GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);
      if (size_t(s_bias + N_) > accel_bias_.size()) 
      {
        st_bias_ns = getEndTnsBiasSpline() - start_t_ns_;
        s_bias = st_bias_ns / dt_ns_bias_;
        u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);
        GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);
      }

      using FunctorT = AccelerometerCostFunctor<N_>;

      FunctorT* functor = 
      new FunctorT(acc_meas, st_s, k, dt_s_, inv_dt_, u_bias, inv_dt_bias_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for rotation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(4);
      }
      // parameter blocks for translation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      // parameter blocks for bias
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      // parameter block for gravity
      cost_function->AddParameterBlock(3);
      // parameter block for time offset
      cost_function->AddParameterBlock(1);

      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(accel_bias_[s_bias + i].data());
      }
      vec.emplace_back(g_.data());
      vec.emplace_back(&t_offset_cam_imu_corr_s_);

      Eigen::Vector3d residual;
      cost_function->Evaluate(&vec[0], residual.data(), NULL);
      double res_norm = residual.norm();
      
      errors.emplace_back(res_norm);
    }

    return errors;
  }

  Eigen::aligned_vector<double> evaluateGyroErrors(const Eigen::aligned_vector<ImuMeasurement>& measurements)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      ImuMeasurement meas = measurements[i];
      Eigen::Vector3d gyro_meas = meas.ang_vel;

      int64_t time_ns = static_cast<int64_t>(meas.t * s_to_ns_);
      int64_t t_offset_imu_cam_ns = 
      static_cast<int64_t>(-t_offset_cam_imu_s_ * s_to_ns_);
      int64_t st_ns = (time_ns + t_offset_imu_cam_ns - start_t_ns_);
      double st_s = st_ns * ns_to_s_;

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;

      // Helper variables to compute u corrected
      // with time offset correction in the cost functor.
      int k_int = st_ns / dt_ns_;
      // This is just to avoid integers in the cost functor
      // (Maybe, it's not needed.)
      double k = static_cast<double>(k_int);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > so3_knots_.size()) {continue;}

      int64_t st_bias_ns = (time_ns - start_t_ns_);
      int64_t s_bias = st_bias_ns / dt_ns_bias_;
      double u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);
      GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias);

      if (size_t(s_bias + N_) > gyro_bias_.size()) 
      {
        st_bias_ns = getEndTnsBiasSpline() - start_t_ns_;
        s_bias = st_bias_ns / dt_ns_bias_;
        u_bias = double(st_bias_ns % dt_ns_bias_) / double(dt_ns_bias_);
        GVI_FUSION_ASSERT_STREAM(s_bias >= 0, "s " << s_bias); 
      }

      using FunctorT = GyroCostFunctor<N_, Sophus::SO3, OLD_TIME_DERIV>;

      // @ToDo: include gyro noise
      FunctorT* functor = 
      new FunctorT(gyro_meas, st_s, k, dt_s_, inv_dt_, u_bias, inv_dt_bias_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
          new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for rotation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(4);
      }
      // parameter blocks for bias
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      // parameter block for time offset
      cost_function->AddParameterBlock(1);
      
      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(gyro_bias_[s_bias + i].data());
      }
      vec.emplace_back(&t_offset_cam_imu_corr_s_);

      Eigen::Vector3d residual;
      cost_function->Evaluate(&vec[0], residual.data(), NULL);
      double res_norm = residual.norm();
      
      errors.emplace_back(res_norm);
    }

    return errors;
  }

  Eigen::aligned_vector<double> evaluateGlobalPosErrors(const Eigen::aligned_vector<GlobalPosMeasurement>& measurements)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      GlobalPosMeasurement meas = measurements[i];
      Eigen::Vector3d gp_meas = meas.pos;
      
      int64_t time_ns = static_cast<int64_t>(meas.t * s_to_ns_);
      int64_t t_s_gp_init_ns = static_cast<int64_t>(t_s_gp_s_ * s_to_ns_);
      int64_t st_ns = (time_ns - start_t_ns_ - t_s_gp_init_ns);
      double st_s = st_ns * ns_to_s_;

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      
      // Helper variables to compute u corrected
      // with time offset correction in the cost functor.
      int k_int = st_ns / dt_ns_;
      // This is just to avoid integers in the cost functor
      // (Maybe, it's not needed.)
      double k = static_cast<double>(k_int);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > r3_knots_.size()) {continue;}

      using FunctorT = GlobalPositionSensorCostFunctor<N_>;

      FunctorT* functor = 
      new FunctorT(gp_meas, scale_, st_s, k, dt_s_, inv_dt_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for rotation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(4);
      }
      // parameter blocks for translation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      // parameter block for p_bp
      cost_function->AddParameterBlock(3);
      // parameter block for time offset correction
      cost_function->AddParameterBlock(1);
      // parameter block for scale correction
      cost_function->AddParameterBlock(1);

      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }
      vec.emplace_back(p_bp_.data());
      vec.emplace_back(&t_s_gp_corr_s_);
      vec.emplace_back(&scale_corr_);

      Eigen::Vector3d residual;
      cost_function->Evaluate(&vec[0], residual.data(), NULL);
      double res_norm = residual.norm();
      
      errors.emplace_back(res_norm);
    }

    return errors;
  }

  Eigen::aligned_vector<double> evaluateReprojectionErrors(Eigen::aligned_vector<Frame>& frames) 
  {
    Eigen::aligned_vector<double> errors;

    for(auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
      int64_t time_ns = static_cast<int64_t>(frame->timestamp * s_to_ns_);
      int64_t st_ns = (time_ns - start_t_ns_);

      if (st_ns < 0) {continue;}

      size_t n_reproj = frame->n_valid_reprojections;
      if (n_reproj == 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      double u = double(st_ns % dt_ns_) / double(dt_ns_);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > so3_knots_.size()) {continue;}

      using FunctorT = ReprojectionCostFunctor<N_>;

      // @ToDo: load noise. 1 px std atm
      FunctorT* functor = 
      new FunctorT(&(*frame), cameras_.data(), scale_, u, inv_dt_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for rotation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(4);
      }
      // parameter blocks for translation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      // parameter blocks for scale correction
      cost_function->AddParameterBlock(1);
      // parameter blocks for imu-cam rotation correction
      cost_function->AddParameterBlock(4);
      // parameter blocks for imu-cam translation correction
      cost_function->AddParameterBlock(3);

      // parameter block for 3d points
      for (size_t i = 0; i < n_reproj; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      cost_function->SetNumResiduals(2*n_reproj);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }
      vec.emplace_back(&scale_corr_);
      vec.emplace_back(R_b_c_corr_.data());
      vec.emplace_back(p_b_c_corr_.data());
      for(const auto& point2d: frame->points2D)
      {
        if(point2d.is_reproj_valid)
        {
          uint64_t point3D_idx = point2d.point3d_id;
          GVI_FUSION_ASSERT_STREAM(points3D_.find(point3D_idx) != points3D_.end(), 
          "Point 3D " <<  point3D_idx << " not found!");
          vec.emplace_back(points3D_.find(point3D_idx)->second.p_wl.data());
        }
      }

      Eigen::VectorXd residuals;
      residuals.setZero(n_reproj * 2);
      
      cost_function->Evaluate(&vec[0], residuals.data(), NULL);
      for (size_t i = 0; i < n_reproj; i++)
      {
        double res_norm = sqrt(residuals(2*i+0) * residuals(2*i+0) + residuals(2*i+1) * residuals(2*i+1));
        errors.emplace_back(res_norm);
      }
    }

    return errors;
  }

  Eigen::aligned_vector<double> evaluatePositionErrors(const Eigen::aligned_vector<Eigen::Vector3d>& measurements, 
  const Eigen::aligned_vector<double>& ts)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      Eigen::Vector3d meas = measurements[i];
      int64_t time_ns = static_cast<int64_t>(ts[i] * s_to_ns_);
      int64_t st_ns = (time_ns - start_t_ns_);

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      double u = double(st_ns % dt_ns_) / double(dt_ns_);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > so3_knots_.size()) {continue;}

      using FunctorT = PositionCostFunctor<N_>;

      // @ToDo: include noise
      FunctorT* functor = 
      new FunctorT(meas, u, inv_dt_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for translation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }

      Eigen::Vector3d residual;
      cost_function->Evaluate(&vec[0], residual.data(), NULL);
      double res_norm = residual.norm();
      
      errors.emplace_back(res_norm);
    }

    return errors;
  }

  Eigen::aligned_vector<double> evaluateOrientationErrors(const Eigen::aligned_vector<Sophus::SO3d>& measurements, 
  const Eigen::aligned_vector<double>& ts)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      Sophus::SO3d meas = measurements[i];
      int64_t time_ns = static_cast<int64_t>(ts[i] * s_to_ns_);
      int64_t st_ns = (time_ns - start_t_ns_);

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      double u = double(st_ns % dt_ns_) / double(dt_ns_);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > so3_knots_.size()) {continue;}

      using FunctorT = OrientationCostFunctor<N_>;

      FunctorT* functor = 
      new FunctorT(meas, u, inv_dt_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for translation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(4);
      }
      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }

      Eigen::Vector3d residual;
      cost_function->Evaluate(&vec[0], residual.data(), NULL);
      double res_norm = residual.norm();
      
      errors.emplace_back(res_norm);
    }

    return errors;
  }

  Eigen::aligned_vector<double> evaluateWorldColmapAlignmentErrors(
    const Eigen::aligned_vector<GlobalPosMeasurement>& measurements)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      GlobalPosMeasurement meas = measurements[i];
      Eigen::Vector3d gp_meas = meas.pos;
      
      int64_t time_ns = static_cast<int64_t>(meas.t * s_to_ns_);
      int64_t t_s_gp_init_ns = static_cast<int64_t>(t_s_gp_s_ * s_to_ns_);
      int64_t st_ns = (time_ns - start_t_ns_ - t_s_gp_init_ns);
      double st_s = st_ns * ns_to_s_;

      if (st_ns < 0) {continue;}

      int64_t s = st_ns / dt_ns_;
      
      // Helper variables to compute u corrected
      // with time offset correction in the cost functor.
      int k_int = st_ns / dt_ns_;
      // This is just to avoid integers in the cost functor
      // (Maybe, it's not needed.)
      double k = static_cast<double>(k_int);

      GVI_FUSION_ASSERT_STREAM(s >= 0, "s " << s);
      if (size_t(s + N_) > r3_knots_.size()) {continue;}

      using FunctorT = 
      GlobalPositionSensorWithAlignmentCorrectionCostFunctor<N_>;

      FunctorT* functor = 
      new FunctorT(gp_meas, T_w_g_, scale_, st_s, k, dt_s_, inv_dt_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for rotation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(4);
      }
      // parameter blocks for translation
      for (int i = 0; i < N_; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      // parameter block for time offset correction
      cost_function->AddParameterBlock(1);
      // parameter block for scale correction
      cost_function->AddParameterBlock(1);
      // parameter block for rotation correction
      cost_function->AddParameterBlock(4);
      // parameter block for translation correction
      cost_function->AddParameterBlock(3);
      // parameter block for p_bp
      cost_function->AddParameterBlock(3);

      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(so3_knots_[s + i].data());
      }
      for (int i = 0; i < N_; i++) 
      {
        vec.emplace_back(r3_knots_[s + i].data());
      }
      vec.emplace_back(&t_s_gp_corr_s_);
      vec.emplace_back(&scale_corr_);
      vec.emplace_back(R_w_g_corr_.data());
      vec.emplace_back(p_w_g_corr_.data());
      vec.emplace_back(p_bp_.data());

      Eigen::Vector3d residual;
      cost_function->Evaluate(&vec[0], residual.data(), NULL);
      double res_norm = residual.norm();
      
      errors.emplace_back(res_norm);
    }

    return errors;
  }

  ceres::Solver::Summary optimize(int max_num_iterations) 
  {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_num_iterations;
    options.num_threads = num_threads_;
    options.minimizer_progress_to_stdout = true;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem_, &summary);
    std::cout << summary.FullReport() << std::endl;

    return summary;
  }

  int64_t maxTimeNs() const 
  {
    return start_t_ns_ + (so3_knots_.size() - N_ + 1) * dt_ns_ - 1;
  }

  int64_t minTimeNs() const { return start_t_ns_; }

  size_t numKnots() const { return so3_knots_.size(); }

  void setGravity(Eigen::Vector3d& a) { g_ = a; }
  const Eigen::Vector3d& getGravity() { return g_; }
  void setOptimizeGravity(bool v) {optimize_gravity_ = v;}

  void setPbp(Eigen::Vector3d& a) { p_bp_ = a; }
  const Eigen::Vector3d& getPbp() { return p_bp_; }
  void setOptimizePbp(bool v) {optimize_p_bp_ = v;}

  double getSplineImuTimeOffset() const { return t_offset_cam_imu_s_; }
  double getSplineImuTimeOffsetCorr() const 
  {
    double t_offset_cam_imu_correction_s_ = t_offset_cam_imu_s_ - cameras_[0].t_offset_cam_imu;
    return t_offset_cam_imu_correction_s_;
  }
  void relinearizeSplineImuTimeOffset()
  {
    t_offset_cam_imu_s_ += t_offset_cam_imu_corr_s_;
    t_offset_cam_imu_corr_s_ = 0.0;
  }

  Sophus::SE3d getImuCamRelTransformation() const
  { 
    Sophus::SE3d T_b_c_corr_(R_b_c_corr_, p_b_c_corr_);
    return cameras_[0].T_b_c * T_b_c_corr_;
  }

  Sophus::SE3d getImuCamRelTransformationCorr() const
  { 
    Sophus::SE3d T_b_c_corr_(R_b_c_corr_, p_b_c_corr_);
    return T_b_c_corr_; 
  }

  void setWorldColmapTransformation(Sophus::SE3d& T) { T_w_g_ = T; }

  Sophus::SE3d getWorldColmapTransformation() const
  { 
    Eigen::Matrix3d R_w_g = T_w_g_.rotationMatrix() * R_w_g_corr_.matrix();
    Eigen::Vector3d p_w_g = T_w_g_.translation() + p_w_g_corr_;
    Sophus::SE3d T_w_g(R_w_g, p_w_g);
    
    return T_w_g;
  }

  Sophus::SE3d getWorldColmapTransformationCorr() const
  { 
    Sophus::SE3d T_w_g_corr_(R_w_g_corr_, p_w_g_corr_);
    return T_w_g_corr_; 
  }

  void setInitSplineGlobSensTimeOffset(double t_s) 
  { 
    t_s_gp_init_s_ = t_s;
    t_s_gp_s_ = t_s;
  }
  double getSplineGlobSensTimeOffset() { return t_s_gp_s_;}
  double getSplineGlobSensTimeOffsetCorr() 
  {
    double t_s_gp_total_correction_s_ = t_s_gp_s_ - t_s_gp_init_s_;
    return t_s_gp_total_correction_s_; 
  }
  void relinearizeSplineGlobSensTimeOffset()
  { 
    t_s_gp_s_ += t_s_gp_corr_s_;
    t_s_gp_corr_s_ = 0.0;
  }
  
  int64_t getStartTns() const { return start_t_ns_; }

  // Return the last (almost) time that can be used to sample from the spline.
  int64_t getEndTns() const 
  { 
    int a = static_cast<int>(numKnots()) - N_ + 1;
    int64_t t_end_ns = static_cast<int64_t>(a) * dt_ns_ + start_t_ns_;
    return t_end_ns - 1e6; 
  }

  // Return the last (almost) time that can be used to sample from the bias spline.
  int64_t getEndTnsBiasSpline() const 
  { 
    int a = static_cast<int>(accel_bias_.size()) - N_ + 1;
    int64_t t_end_ns = static_cast<int64_t>(a) * dt_ns_bias_ + start_t_ns_;
    return t_end_ns - 1e6; 
  }

  int64_t getDeltaTns() const { return dt_ns_; }
  int64_t getDeltaTs() const { return dt_s_; }

  void setScale(double s) { scale_ = s; }
  double getScale() { return scale_ + scale_corr_; }
  double getScaleCorr() { return scale_corr_; }
  void setOptimizeScale(bool v) {optimize_scale_ = v;}

  void setOptimizeCamImuExtrinsics(bool v) {optimize_camimu_extrinsics_ = v;}

  void setOptimizeTimeOffsetSplineGlobalSensor(bool v) {optimize_t_offset_s_gs_ = v;}

  // Only first element is used at the moment
  void setCameras(Eigen::aligned_vector<Camera>& cameras) 
  { 
    cameras_ = cameras;
    t_offset_cam_imu_s_ = cameras_[0].t_offset_cam_imu;
  }
  void setImu(const std::shared_ptr<Imu>& imu) { imu_ptr_ = imu; }

  void set3DPoints(const Eigen::aligned_unordered_map<uint64_t, Point3D>& points3D)
  { points3D_ = points3D; }

  Eigen::aligned_vector<Eigen::Vector3d> get3DPointsArray()
  {
    Eigen::aligned_vector<Eigen::Vector3d> points;
    for (auto& it: points3D_) 
    {
      Eigen::Vector3d pos = it.second.p_wl;
      points.push_back(pos);
    }
    return points;
  }

  void loadOptimizationParams(int n_max_outer_iters, int n_max_inner_iters, int num_threads)
  {
    n_max_outer_iters_ = n_max_outer_iters;
    n_max_inner_iters_ = n_max_inner_iters;
    num_threads_ = num_threads;

    std::cout << "\nLoaded optimization parameters: \n";
    std::cout << "max number of outer iterations = " <<  n_max_outer_iters_ << "\n";
    std::cout << "max number of inner iterations = " <<  n_max_inner_iters_ << "\n";
    std::cout << "number of threads = " <<  num_threads_ << "\n\n";
  }

  private:
  int64_t dt_ns_;
  int64_t start_t_ns_;
  double dt_s_;
  double inv_dt_;
  int64_t dt_ns_bias_;
  double inv_dt_bias_;

  // Not working when cell_size_px = 1
  int cell_size_px_;
  int n_grid_columns_;
  int n_grid_rows_;

  Eigen::aligned_vector<Sophus::SO3d> so3_knots_;
  Eigen::aligned_vector<Eigen::Vector3d> r3_knots_;
  Eigen::aligned_vector<Eigen::Vector3d> accel_bias_;
  Eigen::aligned_vector<Eigen::Vector3d> gyro_bias_;
  Eigen::Vector3d g_, p_bp_;

  bool optimize_p_bp_;
  bool optimize_gravity_;

  // Camera-Imu extrinsics corrections
  Sophus::SO3d R_b_c_corr_;
  Eigen::Vector3d p_b_c_corr_;

  // t_s (spline time) = t_c (camera time)
  double t_offset_cam_imu_s_;
  double t_offset_cam_imu_corr_s_;
  bool optimize_camimu_extrinsics_;

  // Time offset between spline (= camera) and global measuremements.
  double t_s_gp_init_s_; // Initial value
  double t_s_gp_s_;
  // Correction (this is optimized)
  double t_s_gp_corr_s_;
  bool optimize_t_offset_s_gs_;

  // Colmap - Global position scale 
  double scale_;
  // scale correction (optimized)
  double scale_corr_;
  bool optimize_scale_;

  // Spatial global - colmap frame alignment
  Sophus::SE3d T_w_g_;

  // Spatial global - colmap frame alignment corrections
  // Not optimized in full batch optimization
  Sophus::SO3d R_w_g_corr_;
  Eigen::Vector3d p_w_g_corr_;

  // Initial bias
  Eigen::Vector3d init_acc_bias_;
  Eigen::Vector3d init_gyro_bias_;

  Eigen::aligned_vector<Camera> cameras_;
  std::shared_ptr<Imu> imu_ptr_;

  Eigen::aligned_unordered_map<uint64_t, Point3D> points3D_;

  ceres::Problem problem_;

  // Ceres params
  int n_max_outer_iters_;
  int n_max_inner_iters_;
  int num_threads_;
};

