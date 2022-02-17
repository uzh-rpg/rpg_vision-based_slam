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

#include <chrono>
#include <ctime>
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


void saveTrajectoryVelocityBiases(const std::string& trace_dir, DiscreteTimeEstimator& dt_estimator, const bool optimized)
{
    std::string traj_trace_fn;
    std::string vel_trace_fn;
    std::string acc_bias_trace_fn;
    std::string gyro_bias_trace_fn;
    if (optimized)
    {
        traj_trace_fn = "trajectory.txt";
        vel_trace_fn = "velocity.txt";
        acc_bias_trace_fn = "acc_bias.txt";
        gyro_bias_trace_fn = "gyro_bias.txt";
    }
    else
    {
        traj_trace_fn = "initial_trajectory.txt";
        vel_trace_fn = "initial_velocity.txt";
        acc_bias_trace_fn = "initial_acc_bias.txt";
        gyro_bias_trace_fn = "initial_gyro_bias.txt";
    }

    std::ofstream traj_trace;
    createTrace(trace_dir, traj_trace_fn, &traj_trace);
    traj_trace.precision(15);

    std::ofstream vel_trace;
    createTrace(trace_dir, vel_trace_fn, &vel_trace);
    vel_trace.precision(15);
    
    std::ofstream acc_bias_trace;
    createTrace(trace_dir, acc_bias_trace_fn, &acc_bias_trace);
    acc_bias_trace.precision(15);

    std::ofstream gyro_bias_trace;
    createTrace(trace_dir, gyro_bias_trace_fn, &gyro_bias_trace);
    gyro_bias_trace.precision(15);

    Eigen::aligned_vector<double> ts;
    Eigen::aligned_vector<Eigen::Vector3d> pos;
    Eigen::aligned_vector<Sophus::SO3d> ori;
    Eigen::aligned_vector<Eigen::Vector3d> vel;
    Eigen::aligned_vector<Eigen::Vector3d> acc_bias;
    Eigen::aligned_vector<Eigen::Vector3d> gyro_bias;

    Eigen::aligned_vector<State> states;
    dt_estimator.getStates(states);

    for (auto& state : states)
    {
        ts.push_back(state.ts_imu_);
        pos.push_back(state.T_WB_.translation());
        ori.push_back(state.T_WB_.rotationMatrix());
        vel.push_back(state.speed_and_bias_.head<3>());
        acc_bias.push_back(state.speed_and_bias_.tail<3>());
        gyro_bias.push_back(state.speed_and_bias_.segment<3>(3));
    }
    
    traceTimeAndPose(ts, pos, ori, traj_trace);
    traceTimeAndPosition(ts, vel, vel_trace);
    traceTimeAndPosition(ts, acc_bias, acc_bias_trace);
    traceTimeAndPosition(ts, gyro_bias, gyro_bias_trace);
}


void saveErrors(const std::string& trace_dir, 
const Eigen::aligned_vector<double>& gp_errs, 
const bool optimized)
{
    std::string gp_errs_trace_fn;
    if (optimized)
    {
        gp_errs_trace_fn = "global_position_errors.txt";
    }
    else
    {
        gp_errs_trace_fn = "initial_global_position_errors.txt";
    }

    std::ofstream gp_errs_trace;
    createTrace(trace_dir, gp_errs_trace_fn, &gp_errs_trace);
    gp_errs_trace.precision(9);
    trace1DVector(gp_errs, gp_errs_trace);
}


void savePoses(const std::string& trace_dir, const std::string& trace_fn,
const Eigen::aligned_vector<double>& timestamps,
const Eigen::aligned_vector<Sophus::SE3d>& poses)
{
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    traceTrajectorySophusSE3(timestamps, poses, trace);
}


void savePose(const std::string& trace_dir, const std::string& trace_fn, 
const Sophus::SE3d& pose)
{
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    tracePose(pose, trace);
}


void savePosition(const std::string& trace_dir, const std::string& trace_fn, 
const Eigen::Vector3d& position)
{
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    tracePosition(position, trace);
}


void saveScalar(const std::string& trace_dir, const std::string& trace_fn, const double t)
{
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(16);
    traceScalar(t, trace);
}


int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    auto sys_start_time = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(sys_start_time);
    std::cout << "Starting at time: " << std::ctime(&start_time);

    GVI_FUSION_ASSERT_STREAM(argc>1, "Please, pass the .yaml filename");
    std::string yaml_fn = argv[1];

    YAML::Node config = YAML::LoadFile(yaml_fn);

    const std::string imu_fn = config["imu_fn"].as<std::string>();
    const std::string gp_fn = config["gp_fn"].as<std::string>();
    
    std::string trace_dir = config["full_batch_optimization_dir"].as<std::string>();
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/discrete_time";
    std::filesystem::create_directory(trace_dir);

    const std::string dataset_fn = config["dataset"].as<std::string>();

    std::cout << "Dataset: " << dataset_fn << "\n";
    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "imu fn: " << imu_fn << "\n";
    std::cout << "gp fn: " << gp_fn << "\n";
    std::cout << "result dir: " << trace_dir << "\n\n";

    // Initial and final trajectory time
    const double start_ts = config["start_t"].as<double>();
    const double end_ts = config["end_t"].as<double>();

    // Load imu noise from config file
    const double acc_noise_density = config["sigma_acc_c"].as<double>();
    const double acc_random_walk = config["sigma_acc_bias_c"].as<double>();
    const double gyro_noise_density = config["sigma_omega_c"].as<double>();
    const double gyro_random_walk = config["sigma_omega_bias_c"].as<double>();

    std::cout << "Using imu noise: \n";
    std::cout << "- acc_noise_density: " << acc_noise_density << "\n";
    std::cout << "- acc_random_walk: " << acc_random_walk << "\n";
    std::cout << "- gyro_noise_density: " << gyro_noise_density << "\n";
    std::cout << "- gyro_random_walk: " << gyro_random_walk << "\n\n";

    double imu_rate = 200.0;
    auto imu_calib = std::make_shared<ImuCalibration>(
        imu_rate, 
        acc_noise_density, 
        acc_random_walk, 
        gyro_noise_density, 
        gyro_random_walk);
    auto imu = std::make_shared<Imu>(imu_calib);
    if(loadImuMeasurementsFromFile(imu_fn, imu->measurements_))
    {
        std::cout << "Loaded " << imu->measurements_.size() << " imu measurements.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load imu");
    }

    Eigen::aligned_vector<GlobalPosMeasurement> gp_meas;
    if(loadGpMeasurements(gp_fn, gp_meas))
    {
        std::cout << "Loaded " << gp_meas.size() << " global measurements.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load global measurements.");
    }

    Eigen::aligned_vector<Sophus::SE3d> T_WB_s;
    Eigen::aligned_vector<double> times;
    for (size_t i = 0; i < gp_meas.size(); i++)
    {
        double t = gp_meas[i].t;
        if ( (t >= start_ts) && (t <= end_ts) )
        {
            Sophus::SO3d R_WB(Eigen::Matrix3d::Identity());
            Eigen::Vector3d p_WB = gp_meas[i].pos; // p_bp is initialized to 0.

            Sophus::SE3d T_WB_i(R_WB, p_WB);
            T_WB_s.push_back(T_WB_i);
            times.push_back(t);
        }
    }

    // Initialize estimator
    DiscreteTimeEstimator estimator;
    estimator.setImu(imu);
    
    // Adding the camera is just to avoid seg fault in initializeStates().
    Eigen::aligned_vector<Camera> cameras;
    Camera camera;
    cameras.push_back(camera);
    cameras[0].setTimeOffsetCamImu(0.0);
    estimator.setCameras(cameras);
    estimator.initializeStates(T_WB_s, times);

    // initialize frame velocity
    estimator.initializeLinearVelocity();
    
    Eigen::Vector3d g;
    g = imu->calib_ptr_->g_W_;
    estimator.setGravity(g);

    Eigen::Vector3d p_BP(config["p_BP"].as<std::vector<double>>().data());
    std::cout << "Loaded body-gp position offset (= p_BP): \n" << p_BP << "\n\n";
    estimator.setPbp(p_BP);
    const bool optimize_pbp = config["optimize_pBP"].as<bool>();
    estimator.setOptimizePbp(optimize_pbp);
    Eigen::Vector3d p_BP_init = p_BP;

    const bool optimize_t_offset_globsens_imu = config["optimize_t_offset_globsens_imu"].as<bool>();
    estimator.setOptimizeTimeOffsetGlobSensImu(optimize_t_offset_globsens_imu);

    estimator.addLocalParametrization();

    // Deal with gp measurements
    Eigen::aligned_vector<State> states;
    estimator.getStates(states);
    Eigen::aligned_vector<double> state_ts;
    for (auto& state : states)
    {
        state_ts.push_back(state.ts_imu_);
    }

    for (auto& m : gp_meas)
    {
        double t_gp_meas = m.t;
        int state_id = getLeftClosestStateId(state_ts, t_gp_meas);
        if (state_id >= 0)
        {
            m.setClosestFrameId(state_id);
            m.is_valid = true;
        }
    }

    // Add measurements
    // imu
    for(size_t i = 1; i < estimator.getNumOfStates(); i++)
    {
        size_t state_0_id = i - 1; 
        size_t state_1_id = i;

        State state_0;
        state_0 = estimator.getState(state_0_id);
        State state_1;
        state_1 = estimator.getState(state_1_id);

        double t0 = state_0.ts_imu_;
        double t1 = state_1.ts_imu_;

        Eigen::aligned_vector<ImuMeasurement> meas;
        if (imu->getMeasurementsContainingEdges(t0, t1, meas, false))
        {
            estimator.addImuMeasurements(meas, state_0_id, state_1_id);
        }
    }

    // gp
    double gp_meas_std = 1.0;
    if (dataset_fn == "UZH_FPV")
    {
        gp_meas_std = config["leica_std"].as<double>();
    }
    else if ((dataset_fn == "EuRoC") || (dataset_fn == "Custom"))
    {
        gp_meas_std = config["gp_std"].as<double>();
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Unknown dataset");
    }
    std::cout << "Sigma global measurements: " << gp_meas_std << " [m]\n\n";

    for (size_t i = 0; i < gp_meas.size(); i++)
    {
        if (gp_meas[i].is_valid)
        {
            estimator.addGlobalSensMeasurement(gp_meas[i], gp_meas_std);
        }
    }

    // Evaluate initial errors
    std::cout << "+-+-+-+-+- Initial Errors +-+-+-+-+-\n\n";
    // GP
    Eigen::aligned_vector<GlobalPosMeasurement> valid_gp_meas;
    for (size_t i = 0; i < gp_meas.size(); i++)
    {
        if (gp_meas[i].is_valid)
        {
            valid_gp_meas.push_back(gp_meas[i]);
        }
    }
    Eigen::aligned_vector<double> init_gp_errors = estimator.evaluateGlobalPosErrors(valid_gp_meas);
    double init_mean_gp_err = 0.0;
    for (double err : init_gp_errors) {init_mean_gp_err += err;}
    init_mean_gp_err /= static_cast<double>(init_gp_errors.size());
    std::cout<< "Number of global position errors: "<< init_gp_errors.size() <<"\n";
    std::cout<< "Initial global position error mean: "<< init_mean_gp_err <<"\n\n";

    // Save initial values
    std::cout<< "Saving initial values ...\n\n";
    saveTrajectoryVelocityBiases(trace_dir, estimator, false);
    saveErrors(trace_dir, init_gp_errors, false);
    savePosition(trace_dir, "initial_p_BP.txt", estimator.getPbp());
    savePosition(trace_dir, "initial_gravity.txt", estimator.getGravity());
    saveScalar(trace_dir, "initial_globalsensor_imu_time_offset.txt", estimator.getGlobSensImuTimeOffset());

    // Solve
    int n_max_iters = config["n_max_iters"].as<int>();
    int n_threads = config["n_threads"].as<int>();
    estimator.loadOptimizationParams(n_max_iters, n_threads);
    std::cout << "\n\nOptimizing ...\n\n";
    ceres::Solver::Summary summary = estimator.optimize();

    // Evaluate final errors
    std::cout << "+-+-+-+-+- Final Errors +-+-+-+-+-\n\n";
    // GP
    Eigen::aligned_vector<double> opti_gp_errors = estimator.evaluateGlobalPosErrors(valid_gp_meas);
    double opti_mean_gp_err = 0.0;
    for (double err : opti_gp_errors) {opti_mean_gp_err += err;}
    opti_mean_gp_err /= static_cast<double>(opti_gp_errors.size());
    std::cout<< "Number of global position errors: "<< opti_gp_errors.size() <<"\n";
    std::cout<< "Optimized global position error mean: "<< opti_mean_gp_err <<"\n\n";

    std::cout<< "Optimized global sensor - imu time offset: "<< estimator.getGlobSensImuTimeOffset() << " [s]\n";
    std::cout<< "Time offset correction was: "<< estimator.getGlobSensImuTimeOffsetCorr() << " [s]\n\n";
    Eigen::Vector3d p_BP_opt = estimator.getPbp();
    std::cout<< "Optimized p_bp:\n"<< p_BP_opt << "\n";
    std::cout<< "norm correction was: " << (p_BP_opt - p_BP_init).norm() << " [m]\n\n";
    std::cout<< "Gravity in world frame:\n"<< estimator.getGravity() << "\n\n";

    // Save final values
    std::cout<< "Saving optimized values ...\n\n";
    saveTrajectoryVelocityBiases(trace_dir, estimator, true);
    saveErrors(trace_dir, opti_gp_errors, true);
    savePosition(trace_dir, "p_BP.txt", estimator.getPbp());
    savePosition(trace_dir, "gravity.txt", estimator.getGravity());
    saveScalar(trace_dir, "globalsensor_imu_time_offset.txt", estimator.getGlobSensImuTimeOffset());

    auto sys_end_time = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(sys_end_time);
    std::cout << "Finishing at time: " << std::ctime(&end_time);

    return 0;
}

