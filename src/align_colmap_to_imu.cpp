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
I: (fixed) inertial frame. Imu trajectory is expressed in this frame (first pose is identity).
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

#include <fstream>
#include <iostream>
#include <filesystem>

#include "discrete_time_estimator/estimator.h"
#include "util/assert.h"
#include "util/colmap_utils.h"
#include "util/load.h"
#include "util/load_calibration.h"
#include "util/print.h"
#include "util/save.h"
#include "util/trajectory_utils.h"


void savePose(const std::string& trace_dir, const std::string& trace_fn, const Sophus::SE3d& pose)
{
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    tracePose(pose, trace);
}


Sophus::SE3d initializePoseFromImu(Eigen::aligned_vector<ImuMeasurement>& imu_measurements)
{
  // set translation to zero, unit rotation
  Sophus::SE3d T_WB(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

  const int n_measurements = imu_measurements.size();

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (int i = 0; i < n_measurements; ++i)
  {
    acc_B += imu_measurements[i].lin_acc;
  }
  acc_B /= static_cast<double>(n_measurements);
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> pose_increment;
  pose_increment.head<3>() = Eigen::Vector3d::Zero();
  //! this gives a bad result if ez_W.cross(e_acc) norm is
  //! close to zero, deal with it!
  pose_increment.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  pose_increment.tail<3>() *= angle;
  T_WB = Sophus::SE3d::exp(-pose_increment) * T_WB;
  //T_WB.unit_quaternion().normalize();

  return T_WB;
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    GVI_FUSION_ASSERT_STREAM(argc>1, "Please, pass the .yaml filename");
    std::string yaml_fn = argv[1];

    YAML::Node config = YAML::LoadFile(yaml_fn);

    const int cam_id = config["cam_id"].as<int>();
    const std::string camimu_calib_fn = config["camimu_calib_fn"].as<std::string>();
    const std::string imu_fn = config["imu_fn"].as<std::string>();
    const std::string colmap_dir = config["colmap_dir"].as<std::string>();
    
    std::string trace_dir = config["result_dir"].as<std::string>();
    std::filesystem::create_directory(trace_dir);

    const std::string dataset_fn = config["dataset"].as<std::string>();

    std::cout << "Dataset: " << dataset_fn << "\n";
    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "camimu calib fn: " << camimu_calib_fn << "\n";
    std::cout << "imu fn: " << imu_fn << "\n";
    std::cout << "colmap dir: " << colmap_dir << "\n";
    std::cout << "result dir: " << trace_dir << "\n\n";

    // load calibration
    SimpleCamImuCalib camimu_calib;
    bool calib_loaded = false;
    if (dataset_fn == "UZH_FPV")
    {
        const std::string imu_calib_fn = config["imu_calib_fn"].as<std::string>();
        std::cout << "imu calib fn: " << imu_calib_fn << "\n";
        calib_loaded = loadUZHFPVSimpleCamImuCalib(camimu_calib_fn, imu_calib_fn, cam_id, camimu_calib);
    }
    else if (dataset_fn == "EuRoC")
    {
        calib_loaded = loadEuRoCSimpleCamImuCalib(camimu_calib_fn, cam_id, camimu_calib);
    }
    else if (dataset_fn == "Custom")
    {
        std::cout << "Custom dataset. Using EuRoC caliration format." << "\n\n";
        calib_loaded = loadEuRoCSimpleCamImuCalib(camimu_calib_fn, cam_id, camimu_calib);
    }
    if(!calib_loaded)
    {
        GVI_FUSION_ASSERT_STREAM(false, "Camera-Imu calibration could not be loaded");
    }
    else
    {
        printSimpleCamImuCalib(camimu_calib);
    }

    auto imu_calib = std::make_shared<ImuCalibration>(
        camimu_calib.imu_rate_, 
        camimu_calib.acc_noise_density_, 
        camimu_calib.acc_random_walk_, 
        camimu_calib.gyro_noise_density_, 
        camimu_calib.gyro_random_walk_);
    auto imu = std::make_shared<Imu>(imu_calib);
    Eigen::aligned_vector<ImuMeasurement> full_measurements;
    if(loadImuMeasurementsFromFile(imu_fn, full_measurements))
    {
        std::cout << "Loaded " << full_measurements.size() << " imu measurements.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load imu");
    }

    // Read timestamp of first camera image
    const std::string colmap_cam_traj_fn = colmap_dir + "colmap_cam_estimates.txt";
    Eigen::aligned_vector<Sophus::SE3d> T_GC_s;
    Eigen::aligned_vector<double> cam_ts;
    if(!(loadSE3TrajectoryFromFile(colmap_cam_traj_fn, T_GC_s, cam_ts)))
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load COLMAP camera poses.");
    }

    double ts_first_imu = cam_ts[0]; // + camimu_calib.timeshift_cam_imu_;

    // sample imu meas
    for (size_t i = 0; i < full_measurements.size(); i++)
    {
        if (full_measurements[i].t >= ts_first_imu)
        {
            imu->measurements_.push_back(full_measurements[i]);
        }
    }

    // Initial orientation from accel. meas.
    Eigen::aligned_vector<ImuMeasurement> init_meas;
    double ti = imu->measurements_[0].t;
    double tf = imu->measurements_[10].t;
    imu->getMeasurementsContainingEdges(ti, tf, init_meas, false);

    Sophus::SE3d T_IB_0 = initializePoseFromImu(init_meas);

    // Initial Colmap-Imu reference frame
    Sophus::SE3d T_GC_0 = T_GC_s[0];
    Sophus::SE3d T_GB_0 = T_GC_0 * camimu_calib.T_cam_imu_;
    Sophus::SE3d T_IG = T_IB_0 * T_GB_0.inverse();

    // Save
    savePose(trace_dir, "T_IB.txt", T_IB_0);
    savePose(trace_dir, "T_IG.txt", T_IG);


    return 0;
}

