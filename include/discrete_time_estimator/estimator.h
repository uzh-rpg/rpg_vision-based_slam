/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
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

#include "discrete_time_estimator/state.h"
#include "optimization/lie_local_parametrization.h"
#include "optimization/discrete_time/reprojection_cost_functor.h"
#include "optimization/discrete_time/imu_cost_functor.h"
#include "optimization/discrete_time/global_position_cost_functor.h"
#include "optimization/discrete_time/pose_cost_functor.h"
#include "optimization/discrete_time/speed_cost_functor.h"
#include "util/eigen_utils.h"

using namespace gvi_fusion;


class DiscreteTimeEstimator {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DiscreteTimeEstimator()
  {
    // Initialization
    R_BC_corr_ = Sophus::SO3d(Eigen::Matrix3d::Identity());
    p_BC_corr_ << 0.0, 0.0, 0.0;
    t_offset_cam_imu_corr_s_ = 0.0;
    t_offset_globsens_imu_corr_s_ = 0.0;
    scale_corr_ = 0.0;

    // Overwritten by setOptimizePbp()
    optimize_p_BP_ = true;
    // Overwritten by setOptimizeScale()
    optimize_scale_ = false;
    // Overwritten by setOptimizeCamImuExtrinsics()
    optimize_camimu_extrinsics_ = true;
    // Overwritten by setOptimizeTimeOffsetGlobSensImu()
    optimize_t_offset_globsens_imu_s_ = true;
    // Overwritten by initializeGrid()
    cell_size_px_ = 0;
    n_grid_columns_ = 0;
    n_grid_rows_ = 0;
    // Overwritten by setGravity()
    g_W_.setZero();
    // Overwritten by setPbp()
    p_BP_.setZero();
    // Overwritten by setCamGlobSensTimeOffset()
    t_offset_globsens_imu_s_ = 0.0;
    // Overwritten by setScale()
    scale_ = 0.0;
    // Overwritten by loadOptimizationParams()
    n_max_iters_ = 20;
    n_threads_ = 4;
  };

  // Destructor.
  ~DiscreteTimeEstimator() {};

  void initializeStates(const Eigen::aligned_vector<Sophus::SE3d>& init_poses,
  const Eigen::aligned_vector<double>& init_cam_timestamps) 
  {
    size_t n = init_poses.size();
    size_t m = init_cam_timestamps.size();

    GVI_FUSION_ASSERT_STREAM(n == m, 
    "num. init_poses: " << n << "is different from num. cam timestamps: " << m);

    for (size_t i = 0; i < n; i++)
    {
      double ts_imu = init_cam_timestamps[i] + cameras_[0].t_offset_cam_imu;
      State state(init_poses[i], init_cam_timestamps[i], ts_imu);
      states_.push_back(state);
    }
  }

  void initializeLinearVelocity()
  {
    size_t n_states = states_.size();
    for (size_t i = 0; i < n_states-1; i++)
    {
      State state_0;
      state_0 = getState(i);
      double t0 = state_0.ts_imu_;
      Eigen::Vector3d p_WB_0 = state_0.T_WB_.translation();

      State state_1;
      state_1 = getState(i + 1);
      double t1 = state_1.ts_imu_;
      Eigen::Vector3d p_WB_1 = state_1.T_WB_.translation();

      double dt = t1 - t0;
      Eigen::Vector3d v_WB_0 = (p_WB_1 - p_WB_0) / dt;
      states_[i].setVel(v_WB_0);
    }
    states_[n_states-1].setVel(states_[n_states-2].speed_and_bias_.head<3>());
  }

  void initializeBiases(Eigen::Vector3d acc_bias, Eigen::Vector3d gyro_bias)
  {
    for (size_t i = 0; i < states_.size(); i++)
    {
      states_[i].setAccBias(acc_bias);
      states_[i].setGyroBias(gyro_bias);
    }
  }

  State& getState(size_t state_id)
  {
    return states_[state_id];
  }

  void getStates(Eigen::aligned_vector<State>& states)
  {
    for (size_t i = 0; i < states_.size(); i++)
    {
      states.push_back(states_[i]);
    }
  }

  void addLocalParametrization()
  {
    for (size_t i = 0; i < states_.size(); i++) 
    {
      ceres::LocalParameterization* local_parameterization = 
      new LieLocalParameterization<Sophus::SE3d>();

      problem_.AddParameterBlock(states_[i].T_WB_.data(), 
      Sophus::SE3d::num_parameters, local_parameterization);
    }

    // cam-imu rotation
    ceres::LocalParameterization* local_parameterization = 
    new LieLocalParameterization<Sophus::SO3d>();
    problem_.AddParameterBlock(R_BC_corr_.data(), 
    Sophus::SO3d::num_parameters, local_parameterization);
  }

  void initializeGrid(Frame* frame)
  {
    cell_size_px_ = frame->cell_size_px;
    n_grid_columns_ = frame->n_grid_columns;
    n_grid_rows_ = frame->n_grid_rows;
  }

  // Each frame is split in a grid.
  // Points included in the previous frame grid are tracked.
  // A new 2d point is added to free cell grid.
  // All the 2d points assigned to the grid will be used to compute
  // reprojection errors for the corresponding frame
  void computeValidReprojections(Eigen::aligned_vector<Frame>& frames)
  {
    size_t n_cells = n_grid_columns_ * n_grid_rows_;

    std::vector<std::vector<Point2D*>> grid_k_min_1;
    grid_k_min_1.resize(n_cells);

    int k = 0;
    for (auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
      GVI_FUSION_ASSERT_STREAM(abs(states_[k].ts_cam_ - frame->timestamp) < 0.0001, 
      "Frame and state do not correspond, states_[k].ts_cam_ = " << states_[k].ts_cam_ << 
      ", frame->timestamp = " << frame->timestamp);

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
      Sophus::SE3d T_WB = states_[k].T_WB_;
      
      Eigen::Matrix3d R_WC = T_WB.rotationMatrix() * cameras_[0].T_b_c.rotationMatrix();
      double s = getScale();
      Eigen::Vector3d p_WC = s * T_WB.rotationMatrix() * cameras_[0].T_b_c.translation() + 
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

  // Compute feature velocity as proposed in
  // T. Qin and S. Shen. "Online Temporal Calibration for Monocular Visual-Inertial Systems", IROS 2018.
  void computeFeatureVelocity(Eigen::aligned_vector<Frame>& frames)
  {
    for (auto frame = std::begin(frames); frame < std::end(frames) - 1; ++frame)
    {
      auto frame_next = frame;
      frame_next++;

      double dt = frame_next->timestamp - frame->timestamp;

      for (auto& point2D : frame->points2D)
      {
        if (!point2D.is_reproj_valid) continue;

        // use id of the corresponding 3d point to find
        // if point2D is tracked on the next frame
        uint64_t point3D_id = point2D.point3d_id;
        auto point3D_idx_it = 
        std::find(frame_next->points3D_ids.begin(), frame_next->points3D_ids.end(), point3D_id);
        if (point3D_idx_it != frame_next->points3D_ids.end())
        {
          size_t point3D_idx = point3D_idx_it - frame_next->points3D_ids.begin();
          Point2D* point2D_next;
          point2D_next = &frame_next->points2D[point3D_idx];
          
          // compute velocity
          Eigen::Vector2d v = (point2D_next->uv - point2D.uv) / dt;
          point2D.velocity = v;
        }
      }

      frame->computeAveragePixelVelocity();
    }
  }

  void addPosePrior(Sophus::SE3d& T_meas, size_t state_id)
  {
    using FunctorT = PoseCostFunctor;

    double noise = 1e-12;
    FunctorT* functor = new FunctorT(T_meas, 1.0 / noise);

    // @ToDo: this can become AutoDiffCostFunction, just needs to know the 
    // number of residuals
    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for pose
    cost_function->AddParameterBlock(7);

    cost_function->SetNumResiduals(6);

    std::vector<double*> vec;
    vec.emplace_back(states_[state_id].T_WB_.data());

    problem_.AddResidualBlock(cost_function, NULL, vec);
  }

  void addVelPrior(Eigen::Vector3d& v_meas, size_t state_id)
  {
    using FunctorT = SpeedCostFunctor;

    double noise = 1e-12;
    FunctorT* functor = new FunctorT(v_meas, 1.0 / noise);

    // @ToDo: this can become AutoDiffCostFunction, just needs to know the 
    // number of residuals
    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for pose
    cost_function->AddParameterBlock(9);

    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    vec.emplace_back(states_[state_id].speed_and_bias_.data());

    problem_.AddResidualBlock(cost_function, NULL, vec);
  }

  void addImagePointsMeasurement(Frame* frame, int id, double noise)
  {
    if (abs(states_[id].ts_cam_ - frame->timestamp) > 0.0001) return;

    size_t n_reproj = frame->n_valid_reprojections;
    if (n_reproj == 0) {return;}

    using FunctorT = ReprojectionCostFunctor;
    
    FunctorT* functor = 
    new FunctorT(frame, cameras_.data(), scale_, 1.0 / noise);

    // @ToDo: this can become AutoDiffCostFunction, just needs to know the 
    // number of residuals
    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for pose
    cost_function->AddParameterBlock(7);
    // parameter blocks for scale correction
    cost_function->AddParameterBlock(1);
    // parameter blocks for imu-cam rotatation correction
    cost_function->AddParameterBlock(4);
    // parameter blocks for imu-cam translation correction
    cost_function->AddParameterBlock(3);
    // parameter blocks for cam-imu time offset
    cost_function->AddParameterBlock(1);

    // parameter block for 3d points
    for (size_t i = 0; i < n_reproj; i++) 
    {
      cost_function->AddParameterBlock(3);
    }
    cost_function->SetNumResiduals(2*n_reproj);

    std::vector<double*> vec;
    vec.emplace_back(states_[id].T_WB_.data());
    vec.emplace_back(&scale_corr_);
    vec.emplace_back(R_BC_corr_.data());
    vec.emplace_back(p_BC_corr_.data());
    vec.emplace_back(&t_offset_cam_imu_corr_s_);
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

    // @ToDo: Load huber/cauchy loss param.
    //problem_.AddResidualBlock(cost_function, new ceres::HuberLoss(3.0), vec);
    problem_.AddResidualBlock(cost_function, new ceres::CauchyLoss(3.0), vec);

    // Set scale constant
    if (!optimize_scale_)
    {
      problem_.SetParameterBlockConstant(&scale_corr_);
    }

    if(!optimize_camimu_extrinsics_)
    {
      problem_.SetParameterBlockConstant(R_BC_corr_.data());
      problem_.SetParameterBlockConstant(p_BC_corr_.data());
      problem_.SetParameterBlockConstant(&t_offset_cam_imu_corr_s_);
    }
  }

  void addImuMeasurements(const Eigen::aligned_vector<ImuMeasurement>& meas, 
  size_t frame_0_id, size_t frame_1_id)
  {
    double t0 = states_[frame_0_id].ts_imu_;
    double t1 = states_[frame_1_id].ts_imu_;

    ImuCostFunctor* cost_function = 
    new ImuCostFunctor(meas, imu_ptr_->calib_ptr_, t0, t1);

    std::vector<double*> vec;
    vec.emplace_back(states_[frame_0_id].T_WB_.data());
    vec.emplace_back(states_[frame_0_id].speed_and_bias_.data());
    vec.emplace_back(states_[frame_1_id].T_WB_.data());
    vec.emplace_back(states_[frame_1_id].speed_and_bias_.data());

    problem_.AddResidualBlock(cost_function, NULL, vec);
  }

  void addGlobalSensMeasurement(const GlobalPosMeasurement& meas, double noise)
  {
    size_t frame_0_id = static_cast<size_t>(meas.closest_frame_id);
    size_t frame_1_id = frame_0_id + 1;
    GVI_FUSION_ASSERT_STREAM(frame_1_id < states_.size(), 
    "Trying to access frame id (" << frame_1_id 
    << ") larger than states size (" << states_.size() << ")");

    using FunctorT = GlobalPositionSensorCostFunctor;

    FunctorT* functor = 
    new FunctorT(
      meas, scale_, 
      states_[frame_0_id].ts_imu_, states_[frame_1_id].ts_imu_, 
      t_offset_globsens_imu_s_, 1.0 / noise);

    // @ToDo: this can become AutoDiffCostFunction, just needs to know the 
    // number of residuals
    ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
    new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

    // parameter blocks for pose 0
    cost_function->AddParameterBlock(7);
    // parameter blocks for pose 1
    cost_function->AddParameterBlock(7);
    // parameter blocks for scale correction
    cost_function->AddParameterBlock(1);
    // parameter blocks for glob sens - imu t offset correction
    cost_function->AddParameterBlock(1);
    // parameter blocks for imu - gps antenna correction
    cost_function->AddParameterBlock(3);

    cost_function->SetNumResiduals(3);

    std::vector<double*> vec;
    vec.emplace_back(states_[frame_0_id].T_WB_.data());
    vec.emplace_back(states_[frame_1_id].T_WB_.data());
    vec.emplace_back(&scale_corr_);
    vec.emplace_back(&t_offset_globsens_imu_corr_s_);
    vec.emplace_back(p_BP_.data());

    problem_.AddResidualBlock(cost_function, NULL, vec);

    // Set params constant
    if (!optimize_scale_)
    {
      problem_.SetParameterBlockConstant(&scale_corr_);
    }

    if(!optimize_p_BP_)
    {
      problem_.SetParameterBlockConstant(p_BP_.data());
    }

    if(!optimize_t_offset_globsens_imu_s_)
    {
      problem_.SetParameterBlockConstant(&t_offset_globsens_imu_corr_s_);
    }
  }

  Eigen::aligned_vector<double> evaluateReprojectionErrors(Eigen::aligned_vector<Frame>& frames)
  {
    Eigen::aligned_vector<double> errors;

    int id = 0;
    for(auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
      if (abs(states_[id].ts_cam_ - frame->timestamp) > 0.0001) continue;

      size_t n_reproj = frame->n_valid_reprojections;
      if (n_reproj == 0) {continue;}

      using FunctorT = ReprojectionCostFunctor;

      // @ToDo: load noise. Fixed to 1.0 px std atm.
      FunctorT* functor = 
      new FunctorT(&(*frame), cameras_.data(), scale_, 1.0);

      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for pose
      cost_function->AddParameterBlock(7);
      // parameter blocks for scale correction
      cost_function->AddParameterBlock(1);
      // parameter blocks for imu-cam rotatation correction
      cost_function->AddParameterBlock(4);
      // parameter blocks for imu-cam translation correction
      cost_function->AddParameterBlock(3);
      // parameter blocks for cam-imu time offset
      cost_function->AddParameterBlock(1);

      // parameter block for 3d points
      for (size_t i = 0; i < n_reproj; i++) 
      {
        cost_function->AddParameterBlock(3);
      }
      cost_function->SetNumResiduals(2*n_reproj);

      std::vector<double*> vec;
      vec.emplace_back(states_[id].T_WB_.data());
      vec.emplace_back(&scale_corr_);
      vec.emplace_back(R_BC_corr_.data());
      vec.emplace_back(p_BC_corr_.data());
      vec.emplace_back(&t_offset_cam_imu_corr_s_);
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

      id++;
    }

    return errors;
  }

  Eigen::aligned_vector<Eigen::Matrix<double, 15, 1>> evaluateImuErrors()
  {
    Eigen::aligned_vector<Eigen::Matrix<double, 15, 1>> errors;

    for(size_t i = 1; i < getNumOfStates(); i++)
    {
      size_t frame_0_id = i - 1; 
      size_t frame_1_id = i;

      double t0 = states_[frame_0_id].ts_imu_;
      double t1 = states_[frame_1_id].ts_imu_;
      Eigen::aligned_vector<ImuMeasurement> extracted_meas;
      if(!imu_ptr_->getMeasurementsContainingEdges(t0, t1, extracted_meas, false))
      continue;

      ImuCostFunctor* cost_function = 
      new ImuCostFunctor(extracted_meas, imu_ptr_->calib_ptr_, t0, t1);

      // ToDo: optimize gravity?
      std::vector<double*> vec;
      vec.emplace_back(states_[frame_0_id].T_WB_.data());
      vec.emplace_back(states_[frame_0_id].speed_and_bias_.data());
      vec.emplace_back(states_[frame_1_id].T_WB_.data());
      vec.emplace_back(states_[frame_1_id].speed_and_bias_.data());

      Eigen::Matrix<double, 15, 1> residuals;
      residuals.setZero();
      
      cost_function->Evaluate(&vec[0], residuals.data(), NULL);
      errors.push_back(residuals);
    }
    return errors;
  }

  Eigen::aligned_vector<double> evaluateGlobalPosErrors(Eigen::aligned_vector<GlobalPosMeasurement>& measurements)
  {
    Eigen::aligned_vector<double> errors;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      GlobalPosMeasurement meas = measurements[i];
      
      size_t frame_0_id = static_cast<size_t>(meas.closest_frame_id);
      size_t frame_1_id = frame_0_id + 1;
      GVI_FUSION_ASSERT_STREAM(frame_1_id < states_.size(), 
      "Trying to access frame id (" << frame_1_id 
      << ") larger than states size (" << states_.size() << ")");

      using FunctorT = GlobalPositionSensorCostFunctor;

      FunctorT* functor = 
      new FunctorT(
        meas, scale_, 
        states_[frame_0_id].ts_imu_, states_[frame_1_id].ts_imu_, 
        t_offset_globsens_imu_s_, 1.0);

      // @ToDo: this can become AutoDiffCostFunction, just needs to know the 
      // number of residuals
      ceres::DynamicAutoDiffCostFunction<FunctorT>* cost_function =
      new ceres::DynamicAutoDiffCostFunction<FunctorT>(functor);

      // parameter blocks for pose 0
      cost_function->AddParameterBlock(7);
      // parameter blocks for pose 1
      cost_function->AddParameterBlock(7);
      // parameter blocks for scale correction
      cost_function->AddParameterBlock(1);
      // parameter blocks for glob sens - imu t offset correction
      cost_function->AddParameterBlock(1);
      // parameter blocks for imu - gps antenna correction
      cost_function->AddParameterBlock(3);

      cost_function->SetNumResiduals(3);

      std::vector<double*> vec;
      vec.emplace_back(states_[frame_0_id].T_WB_.data());
      vec.emplace_back(states_[frame_1_id].T_WB_.data());
      vec.emplace_back(&scale_corr_);
      vec.emplace_back(&t_offset_globsens_imu_corr_s_);
      vec.emplace_back(p_BP_.data());

      Eigen::Vector3d residuals;
      residuals.setZero();
      cost_function->Evaluate(&vec[0], residuals.data(), NULL);
      errors.push_back(residuals.norm());
    }

    return errors;
  }

  void setStatePoseConstant(size_t id)
  {
    problem_.SetParameterBlockConstant(states_[id].T_WB_.data());
  }

  void setStateVelAndBiasConstant(size_t id)
  {
    problem_.SetParameterBlockConstant(states_[id].speed_and_bias_.data());
  }

  void setStateConstant(size_t id)
  {
    problem_.SetParameterBlockConstant(states_[id].T_WB_.data());
    problem_.SetParameterBlockConstant(states_[id].speed_and_bias_.data());
  }

  ceres::Solver::Summary optimize() 
  {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = n_max_iters_;
    options.num_threads = n_threads_;
    options.minimizer_progress_to_stdout = true;

    // Solve
    ceres::Solver::Summary summary;
    Solve(options, &problem_, &summary);
    std::cout << summary.FullReport() << std::endl;

    return summary;
  }

  void setImu(const std::shared_ptr<Imu>& imu) { imu_ptr_ = imu; }

  // Only first element is used at the moment
  void setCameras(Eigen::aligned_vector<Camera>& cameras) { cameras_ = cameras; }

  void set3DPoints(const Eigen::aligned_unordered_map<uint64_t, Point3D>& points3D)
  { points3D_ = points3D; }

  void setStateVelocity(Eigen::Vector3d v, size_t i)
  {
    states_[i].setVel(v);
  }

  void setGravity(Eigen::Vector3d& a) { g_W_ = a; }
  const Eigen::Vector3d& getGravity() { return g_W_; }

  void setPbp(Eigen::Vector3d& a) { p_BP_ = a; }
  const Eigen::Vector3d& getPbp() { return p_BP_; }
  void setOptimizePbp(bool v) {optimize_p_BP_ = v;}

  void setGlobSensImuTimeOffset(double t_s) { t_offset_globsens_imu_s_ = t_s; }
  const double getGlobSensImuTimeOffsetCorr() { return t_offset_globsens_imu_corr_s_; }
  const double getGlobSensImuTimeOffset() 
  { 
    double t = t_offset_globsens_imu_s_ + t_offset_globsens_imu_corr_s_;
    return t;
  }

  void setScale(double s) { scale_ = s; }
  double getScale() { return scale_ + scale_corr_; }
  double getScaleCorr() { return scale_corr_; }
  void setOptimizeScale(bool v) {optimize_scale_ = v;}

  void setOptimizeCamImuExtrinsics(bool v) {optimize_camimu_extrinsics_ = v;}

  void setOptimizeTimeOffsetGlobSensImu(bool v) {optimize_t_offset_globsens_imu_s_ = v;}

  size_t getNumOfStates() {return states_.size();}

  Sophus::SE3d getImuCamRelTransformation() const
  { 
    Sophus::SE3d T_BC_corr_(R_BC_corr_, p_BC_corr_);
    return cameras_[0].T_b_c * T_BC_corr_;
  }
  Sophus::SE3d getImuCamRelTransformationCorr() const
  { 
    Sophus::SE3d T_BC_corr_(R_BC_corr_, p_BC_corr_);
    return T_BC_corr_; 
  }

  const double getCamImuTimeOffset() 
  { 
    double t = cameras_[0].t_offset_cam_imu + t_offset_cam_imu_corr_s_;
    return t; 
  }
  const double getCamImuTimeOffsetCorr() { return t_offset_cam_imu_corr_s_; }

  void loadOptimizationParams(int n_max_iters, int num_threads)
  {
    n_max_iters_ = n_max_iters;
    n_threads_ = num_threads;

    std::cout << "\nLoaded optimization parameters: \n";
    std::cout << "max number of iterations = " <<  n_max_iters_ << "\n";
    std::cout << "number of threads = " <<  n_threads_ << "\n\n";
  }

  private:
  
  Eigen::aligned_vector<State> states_;

  // Not working when cell_size_px = 1
  int cell_size_px_;
  int n_grid_columns_;
  int n_grid_rows_;
  
  Eigen::Vector3d g_W_, p_BP_;

  bool optimize_p_BP_;

  // Camera-Imu extrinsics corrections
  Sophus::SO3d R_BC_corr_;
  Eigen::Vector3d p_BC_corr_;
  double t_offset_cam_imu_corr_s_;
  bool optimize_camimu_extrinsics_;

  // Time offset between camera and global measuremements.
  // Initial value
  double t_offset_globsens_imu_s_;
  // Correction (this is optimized)
  double t_offset_globsens_imu_corr_s_;
  bool optimize_t_offset_globsens_imu_s_;

  // Colmap - Global position scale 
  double scale_;
  // scale correction (optimized)
  double scale_corr_;
  bool optimize_scale_;

  Eigen::aligned_vector<Camera> cameras_;
  std::shared_ptr<Imu> imu_ptr_;

  Eigen::aligned_unordered_map<uint64_t, Point3D> points3D_;

  ceres::Problem problem_;

  // Ceres params
  int n_max_iters_;
  int n_threads_;
};

