/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
Reference frames (spatial and temporal):
W: (fixed) scale-aware colmap frame. Rescaled colmap trajectory is expressed in this coordinate frame.
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
const Eigen::aligned_vector<double>& reproj_errs, const bool optimized)
{
    std::string reproj_errs_trace_fn;
    if (optimized)
    {
        reproj_errs_trace_fn = "reprojection_errors.txt";
    }
    else
    {
        reproj_errs_trace_fn = "initial_reprojection_errors.txt";
    }

    std::ofstream reproj_errs_trace;
    createTrace(trace_dir, reproj_errs_trace_fn, &reproj_errs_trace);
    reproj_errs_trace.precision(9);
    trace1DVector(reproj_errs, reproj_errs_trace);
}


void save3DPoints(const std::string& trace_dir, 
const Eigen::aligned_unordered_map<uint64_t, Point3D>& points, const bool optimized)
{
    std::string trace_fn;
    if (optimized)
    {
        trace_fn = "points3D.txt";
    }
    else
    {
        trace_fn = "initial_points3D.txt";
    }
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    tracePoints3D(points, trace);
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

    const int cam_id = config["cam_id"].as<int>();
    const std::string camimu_calib_fn = config["camimu_calib_fn"].as<std::string>();
    const std::string imu_fn = config["imu_fn"].as<std::string>();
    const std::string colmap_dir = config["colmap_dir"].as<std::string>();
    const std::string colmap_fn = config["colmap_fn"].as<std::string>();
    
    std::string trace_dir = config["full_batch_optimization_dir"].as<std::string>();
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/discrete_time";
    std::filesystem::create_directory(trace_dir);

    const std::string dataset_fn = config["dataset"].as<std::string>();

    std::cout << "Dataset: " << dataset_fn << "\n";
    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "camimu calib fn: " << camimu_calib_fn << "\n";
    std::cout << "imu fn: " << imu_fn << "\n";
    std::cout << "colmap dir: " << colmap_dir << "\n";
    std::cout << "result dir: " << trace_dir << "\n\n";

    // Initial and final trajectory time
    const double start_ts = config["start_t"].as<double>();
    const double end_ts = config["end_t"].as<double>();

    // load calibration
    SimpleCamImuCalib camimu_calib;
    bool calib_loaded = false;
    if (dataset_fn == "UZH_FPV")
    {
        const std::string imu_calib_fn = config["imu_calib_fn"].as<std::string>();
        std::cout << "imu calib fn: " << imu_calib_fn << "\n";
        calib_loaded = loadUZHFPVSimpleCamImuCalib(camimu_calib_fn, imu_calib_fn, cam_id, camimu_calib);
    }
    else if ((dataset_fn == "EuRoC") || (dataset_fn == "Custom"))
    {
        std::cout << "Custom dataset. Using EuRoC caliration format." << "\n\n";
        calib_loaded = loadEuRoCSimpleCamImuCalib(camimu_calib_fn, cam_id, camimu_calib);
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Unknown dataset");
    }

    if(!calib_loaded)
    {
        GVI_FUSION_ASSERT_STREAM(false, "Camera-Imu calibration could not be loaded");
    }
    else
    {
        printSimpleCamImuCalib(camimu_calib);
    }

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

    auto imu_calib = std::make_shared<ImuCalibration>(
        camimu_calib.imu_rate_, 
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

    const std::string colmap_cam_fn = colmap_dir + "cameras.bin";
    Eigen::aligned_vector<Camera> cameras;
    readCamerasBinary(colmap_cam_fn, cameras);
    std::cout << "Loaded: " << cameras.size() << " cameras.\n\n";

    Sophus::SE3d T_CB = camimu_calib.T_cam_imu_;
    Sophus::SE3d T_BC = T_CB.inverse();
    cameras[0].setTbc(T_BC);
    cameras[0].setTimeOffsetCamImu(camimu_calib.timeshift_cam_imu_);

    const std::string colmap_img_fn = colmap_dir + "images.bin";
    Eigen::aligned_vector<Frame> frames_full_traj;
    const std::string colmap_points3d_fn = colmap_dir + "points3D.bin";
    Eigen::aligned_unordered_map<uint64_t, Point3D> points3d;
    const int grid_cell_size = config["grid_cell_size"].as<int>();
    readImagesAndPoints3DBinary(colmap_img_fn, colmap_points3d_fn, cameras, frames_full_traj, points3d, grid_cell_size);
    std::cout << "Full trajectory contains: " << frames_full_traj.size() << " images.\n\n";
    std::cout << "Full 3D map contains: " << points3d.size() << " 3D points.\n\n";

    Eigen::aligned_vector<Sophus::SE3d> T_GC_full_traj;
    Eigen::aligned_vector<double> cam_ts_full_traj;
    if(!(loadSE3TrajectoryFromFile(colmap_fn, T_GC_full_traj, cam_ts_full_traj)))
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load COLMAP camera poses.");
    }

    Eigen::aligned_vector<Sophus::SE3d> T_GC_s;
    Eigen::aligned_vector<double> cam_ts;
    for(size_t i = 0; i < cam_ts_full_traj.size(); i++)
    {
        if ( (cam_ts_full_traj[i] >= start_ts) && (cam_ts_full_traj[i] <= end_ts) )
        {
            cam_ts.push_back(cam_ts_full_traj[i]);
            T_GC_s.push_back(T_GC_full_traj[i]);
        }
    }

    // Transform camera to body
    Eigen::aligned_vector<Sophus::SE3d> T_GB_s;
    for (size_t i = 0; i < T_GC_s.size(); i++)
    {
        Sophus::SE3d T_GB_i = T_GC_s[i] * cameras[0].T_c_b;

        T_GB_s.push_back(T_GB_i);
    }

    // This is loaded as transpose
    Eigen::Matrix4d T_eigen(config["T_wg"].as<std::vector<double>>().data());
    Sophus::SE3d T_WG(T_eigen.transpose());
    std::cout << "Loaded T_wg:\n";
    printSophusPose(T_WG);

    double scale = config["initial_scale"].as<double>();
    std::cout << "Loaded scale: " << scale << "\n\n";
    const bool optimize_scale = config["optimize_scale"].as<bool>();
    
    // Re-scale colmap trajectory
    Eigen::aligned_vector<Sophus::SE3d> T_WB_s;
    for (size_t i = 0; i < T_GB_s.size(); i++)
    {
        Sophus::SO3d R_WB = T_WG.so3() * T_GB_s[i].so3();
        Eigen::Vector3d p_WB = scale * (T_WG.rotationMatrix() * T_GB_s[i].translation()) + T_WG.translation();

        Sophus::SE3d T_WB_i(R_WB, p_WB);
        T_WB_s.push_back(T_WB_i);
    }
    
    // There is no knowledge of world frame in vi fusion.
    // Thus, set position in world frame same as in colmap reference frame.
    // Above is not true atm
    for (auto& it: points3d) 
    {
        Eigen::Vector3d p_WL = scale * (T_WG.rotationMatrix() * it.second.p_gl) + T_WG.translation();
        it.second.setPositionInWorldFrame(p_WL);
    }

    // Select the relevant frames
    GVI_FUSION_ASSERT_STREAM(frames_full_traj.size() == cam_ts_full_traj.size(), 
    "frames_full_traj.size()= " << frames_full_traj.size() << 
    " cam_ts_full_traj.size()= " << cam_ts_full_traj.size());

    Eigen::aligned_vector<Frame> frames;
    for(size_t i = 0; i < cam_ts_full_traj.size(); i++)
    {
        if ( (cam_ts_full_traj[i] >= start_ts) && (cam_ts_full_traj[i] <= end_ts) )
        {
            frames.push_back(frames_full_traj[i]);
        }
    }

    // Set frame timestamps
    GVI_FUSION_ASSERT_STREAM(frames.size() == cam_ts.size(), 
    "frames.size()= " << frames.size() << " cam_ts.size()= " << cam_ts.size());
    for (size_t i = 0; i < frames.size(); i++)
    {
        frames[i].setTimestamp(cam_ts[i]);
    }

    // Initialize estimator
    DiscreteTimeEstimator estimator;
    
    estimator.setImu(imu);

    estimator.setCameras(cameras);
    
    estimator.initializeStates(T_WB_s, cam_ts);

    // initialize frame velocity
    estimator.initializeLinearVelocity();
    
    Eigen::Vector3d g;
    g = imu->calib_ptr_->g_W_;
    estimator.setGravity(g);

    estimator.setScale(scale);
    estimator.setOptimizeScale(optimize_scale);

    estimator.addLocalParametrization();

    estimator.set3DPoints(points3d);

    auto frame_it = frames.begin();
    estimator.initializeGrid(&(*frame_it));

    estimator.computeValidReprojections(frames);
    for (size_t i = 0; i < frames.size(); i++)
    {
        int n = 0;
        for (size_t j = 0; j < frames[i].points2D.size(); j++)
        {
            if (frames[i].points2D[j].is_reproj_valid) n++;
        }

        GVI_FUSION_ASSERT_STREAM(frames[i].n_valid_reprojections == n, 
        "frames[i].n_valid_reprojections != n, " << 
        frames[i].n_valid_reprojections << " != " << n);
    }

    estimator.computeFeatureVelocity(frames);

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

    // vision
    double sigma_reproj_px = config["sigma_reproj_px"].as<double>();
    std::cout << "Sigma reprojection errors: " << sigma_reproj_px << " [pixel]\n\n";
    int frame_id = 0;
    for(auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
        estimator.addImagePointsMeasurement(&(*frame), frame_id, sigma_reproj_px);
        frame_id++;
    }

    // Evaluate initial errors
    std::cout << "+-+-+-+-+- Initial Errors +-+-+-+-+-\n\n";

    // reprojection
    Eigen::aligned_vector<double> init_reproj_errors = 
    estimator.evaluateReprojectionErrors(frames);

    double init_mean_reproj_err = 0.0;
    int n_reproj_err = 0;
    for (double err : init_reproj_errors) 
    {
        if (err > 1e-6)
        {
            init_mean_reproj_err += err;
            n_reproj_err++;
        }
    }
    init_mean_reproj_err /= static_cast<double>(n_reproj_err);
    std::cout<< "Number of reprojection errors: "<< n_reproj_err <<"\n";
    std::cout<< "Initial reprojection error mean: "<< init_mean_reproj_err <<"\n\n";

    // Save initial values
    std::cout<< "Saving initial values ...\n\n";
    saveTrajectoryVelocityBiases(trace_dir, estimator, false);
    saveErrors(trace_dir, init_reproj_errors, false);
    saveScalar(trace_dir, "initial_scale.txt", estimator.getScale());
    savePose(trace_dir, "initial_T_BC.txt", estimator.getImuCamRelTransformation());
    saveScalar(trace_dir, "initial_cam_imu_time_offset.txt", estimator.getCamImuTimeOffset());
    savePosition(trace_dir, "initial_gravity.txt", estimator.getGravity());
    save3DPoints(trace_dir, points3d, false);

    // Solve
    int n_max_iters = config["n_max_iters"].as<int>();
    int n_threads = config["n_threads"].as<int>();
    estimator.loadOptimizationParams(n_max_iters, n_threads);
    std::cout << "\n\nOptimizing ...\n\n";
    ceres::Solver::Summary summary = estimator.optimize();

    // Evaluate final errors
    std::cout << "+-+-+-+-+- Final Errors +-+-+-+-+-\n\n";

    // Reprojection
    Eigen::aligned_vector<double> opti_reproj_errors = estimator.evaluateReprojectionErrors(frames);
    double opti_mean_reproj_err = 0.0;
    n_reproj_err = 0;
    for (double err : opti_reproj_errors) 
    {
        if (err > 1e-6)
        {
            opti_mean_reproj_err += err;
            n_reproj_err++;
        }
    }
    opti_mean_reproj_err /= static_cast<double>(n_reproj_err);
    std::cout<< "Number of reprojection errors: "<< n_reproj_err <<"\n";
    std::cout<< "Optimized reprojection error mean: "<< opti_mean_reproj_err <<"\n\n";

    // Cam-imu extrinsics
    std::cout<< "Optimized imu - camera transformation (= T_BC) :\n";
    printSophusPose(estimator.getImuCamRelTransformation());
    std::cout<< "Transformation correction was:\n"; 
    Sophus::SE3d T_BC_corr = estimator.getImuCamRelTransformationCorr();
    printSophusPose(T_BC_corr);
    double rot_angle_rad = acos((T_BC_corr.rotationMatrix().trace() - 1.0 - 1e-6) / 2.0);
    double rot_angle_deg = (rot_angle_rad * 180.0) / 3.14159;
    std::cout<< "rotation angle correction was: " << rot_angle_deg << " [deg]\n";
    std::cout<< "translation norm correction was: " << T_BC_corr.translation().norm() << " [m]\n\n";
    std::cout<< "Optimized cam - imu time offset: "<< estimator.getCamImuTimeOffset() << " [s]\n";
    std::cout<< "Time offset correction was: "<< estimator.getCamImuTimeOffsetCorr() << " [s]\n\n";
    std::cout<< "Optimized colmap - global reference scale: "<< estimator.getScale() << "\n";
    std::cout<< "Scale correction was: "<< estimator.getScaleCorr() << "\n\n";

    std::cout<< "Gravity in world frame:\n"<< estimator.getGravity() << "\n\n";

    // Save final values
    std::cout<< "Saving optimized values ...\n\n";
    saveTrajectoryVelocityBiases(trace_dir, estimator, true);
    saveErrors(trace_dir, opti_reproj_errors, true);
    saveScalar(trace_dir, "scale.txt", estimator.getScale());
    savePose(trace_dir, "T_BC.txt", estimator.getImuCamRelTransformation());
    saveScalar(trace_dir, "cam_imu_time_offset.txt", estimator.getCamImuTimeOffset());
    savePosition(trace_dir, "gravity.txt", estimator.getGravity());
    save3DPoints(trace_dir, points3d, false);

    auto sys_end_time = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(sys_end_time);
    std::cout << "Finishing at time: " << std::ctime(&end_time);

    return 0;
}

