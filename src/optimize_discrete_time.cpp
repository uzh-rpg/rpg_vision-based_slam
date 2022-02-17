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


void saveErrors(const std::string& trace_dir, const Eigen::aligned_vector<double>& gp_errs, 
const Eigen::aligned_vector<double>& reproj_errs, const bool optimized)
{
    std::string gp_errs_trace_fn;
    std::string reproj_errs_trace_fn;
    if (optimized)
    {
        gp_errs_trace_fn = "global_position_errors.txt";
        reproj_errs_trace_fn = "reprojection_errors.txt";
    }
    else
    {
        gp_errs_trace_fn = "initial_global_position_errors.txt";
        reproj_errs_trace_fn = "initial_reprojection_errors.txt";
    }

    std::ofstream gp_errs_trace;
    createTrace(trace_dir, gp_errs_trace_fn, &gp_errs_trace);
    gp_errs_trace.precision(9);
    trace1DVector(gp_errs, gp_errs_trace);

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
    const std::string gp_fn = config["gp_fn"].as<std::string>();
    const std::string colmap_dir = config["colmap_dir"].as<std::string>();
    const std::string colmap_fn = config["colmap_fn"].as<std::string>();
    
    const std::string alignment_dir = config["alignment_dir"].as<std::string>();
    
    std::string trace_dir = config["full_batch_optimization_dir"].as<std::string>();
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/discrete_time";
    std::filesystem::create_directory(trace_dir);

    const std::string dataset_fn = config["dataset"].as<std::string>();

    std::cout << "Dataset: " << dataset_fn << "\n";
    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "camimu calib fn: " << camimu_calib_fn << "\n";
    std::cout << "imu fn: " << imu_fn << "\n";
    std::cout << "gp fn: " << gp_fn << "\n";
    std::cout << "colmap dir: " << colmap_dir << "\n";
    std::cout << "alignment dir: " << alignment_dir << "\n";
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
    else if ((dataset_fn == "EuRoC"))
    {
        calib_loaded = loadEuRoCSimpleCamImuCalib(camimu_calib_fn, cam_id, camimu_calib);
    }
    else if (dataset_fn == "Custom")
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

    Eigen::aligned_vector<GlobalPosMeasurement> gp_meas;
    if(loadGpMeasurements(gp_fn, gp_meas))
    {
        std::cout << "Loaded " << gp_meas.size() << " global measurements.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load global measurements.");
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

    // It may happen that frames are not sorted
    std::vector<std::string> img_names;
    for(size_t i = 0; i < frames_full_traj.size(); i++)
    {
        img_names.push_back(frames_full_traj[i].name);
    }
    std::vector<size_t> sorting_idxs = argSort(img_names);
    Eigen::aligned_vector<Frame> frames_full_traj_sorted;
    for(size_t i = 0; i < sorting_idxs.size(); i++)
    {
        size_t idx = sorting_idxs[i];
        frames_full_traj_sorted.push_back(frames_full_traj[idx]);
    }
    frames_full_traj.clear();
    for(size_t i = 0; i < frames_full_traj_sorted.size(); i++)
    {
        frames_full_traj.push_back(frames_full_traj_sorted[i]);
    }

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

    // Transform poses from colmap world frame to leica/gps world frame
    Sophus::SE3d T_WG;
    if (loadPose(alignment_dir + "/T_wg.txt", T_WG))
    {
        std::cout << "Loaded initial T_WG:\n\n";
        printSophusPose(T_WG);
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not T_WG");
    }

    double scale;
    if (loadScalar<double>(alignment_dir + "/scale.txt", scale))
    {
        std::cout << "Loaded scale: " << scale << "\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not scale");
    }
    const bool optimize_scale = config["optimize_scale"].as<bool>();

    Eigen::aligned_vector<Sophus::SE3d> T_WB_s;
    for (size_t i = 0; i < T_GB_s.size(); i++)
    {
        Sophus::SO3d R_WB = T_WG.so3() * T_GB_s[i].so3();
        Eigen::Vector3d p_WB = scale * (T_WG.rotationMatrix() * T_GB_s[i].translation()) + T_WG.translation();

        Sophus::SE3d T_WB_i(R_WB, p_WB);
        T_WB_s.push_back(T_WB_i);
    }
    
    // Need to transform 3d points from G (colmap frame) to W (global world frame).
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

    double t_cam_gp_init = config["t_offset_cam_gp_init"].as<double>();
    std::cout << "Loaded camera - global sensor time offset: " 
    << std::setprecision(15) << t_cam_gp_init << "\n\n";
    double t_gp_imu_init;
    t_gp_imu_init = cameras[0].t_offset_cam_imu - t_cam_gp_init;
    estimator.setGlobSensImuTimeOffset(t_gp_imu_init);

    Eigen::Vector3d p_BP(config["p_BP"].as<std::vector<double>>().data());
    std::cout << "Loaded body-gp position offset (= p_BP): \n" << p_BP << "\n\n";
    estimator.setPbp(p_BP);
    const bool optimize_pbp = config["optimize_pBP"].as<bool>();
    estimator.setOptimizePbp(optimize_pbp);
    Eigen::Vector3d p_BP_init = p_BP;

    const bool optimize_t_offset_globsens_imu = config["optimize_t_offset_globsens_imu"].as<bool>();
    estimator.setOptimizeTimeOffsetGlobSensImu(optimize_t_offset_globsens_imu);

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
        double t_gp_meas = m.t + t_gp_imu_init;
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

    // vision
    double sigma_reproj_px = config["sigma_reproj_px"].as<double>();
    std::cout << "Sigma reprojection errors: " << sigma_reproj_px << " [pixel]\n\n";
    int frame_id = 0;
    for(auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
        estimator.addImagePointsMeasurement(&(*frame), frame_id, sigma_reproj_px);
        frame_id++;
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

    // IMU
    // Useless because errors are weighted
    /*Eigen::aligned_vector<Eigen::Matrix<double, 15, 1>> init_imu_errors = 
        estimator.evaluateImuErrors();
    double init_mean_imu_pos_err = 0.0;
    int n_imu_err = 0;
    for (Eigen::Matrix<double, 15, 1> err : init_imu_errors)
    {
        Eigen::Vector3d imu_pos_err_i;

        imu_pos_err_i << err[0], err[1], err[2];
        init_mean_imu_pos_err += imu_pos_err_i.norm();

        n_imu_err++;
    }
    init_mean_imu_pos_err /= static_cast<double>(n_imu_err);
    std::cout<< "Number of imu errors: "<< n_imu_err <<"\n";
    std::cout<< "Initial imu position error mean: "<< init_mean_imu_pos_err <<"\n\n";*/

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
    saveErrors(trace_dir, init_gp_errors, init_reproj_errors, false);
    saveScalar(trace_dir, "initial_scale.txt", estimator.getScale());
    savePosition(trace_dir, "initial_p_BP.txt", estimator.getPbp());
    savePose(trace_dir, "initial_T_BC.txt", estimator.getImuCamRelTransformation());
    saveScalar(trace_dir, "initial_cam_imu_time_offset.txt", estimator.getCamImuTimeOffset());
    savePosition(trace_dir, "initial_gravity.txt", estimator.getGravity());
    saveScalar(trace_dir, "initial_globalsensor_imu_time_offset.txt", estimator.getGlobSensImuTimeOffset());
    save3DPoints(trace_dir, points3d, false);

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

    // IMU
    // Useless because errors are weighted
    /*Eigen::aligned_vector<Eigen::Matrix<double, 15, 1>> opti_imu_errors = 
        estimator.evaluateImuErrors();
    double opti_mean_imu_pos_err = 0.0;
    n_imu_err = 0;
    for (Eigen::Matrix<double, 15, 1> err : opti_imu_errors)
    {
        Eigen::Vector3d imu_pos_err_i;
        imu_pos_err_i << err[0], err[1], err[2];
        opti_mean_imu_pos_err += imu_pos_err_i.norm();
        n_imu_err++;
    }
    opti_mean_imu_pos_err /= static_cast<double>(n_imu_err);
    std::cout<< "Number of imu errors: "<< n_imu_err <<"\n";
    std::cout<< "Optimized imu position error mean: "<< opti_mean_imu_pos_err <<"\n\n";*/

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
    std::cout<< "Optimized global sensor - imu time offset: "<< estimator.getGlobSensImuTimeOffset() << " [s]\n";
    std::cout<< "Time offset correction was: "<< estimator.getGlobSensImuTimeOffsetCorr() << " [s]\n\n";
    Eigen::Vector3d p_BP_opt = estimator.getPbp();
    std::cout<< "Optimized p_bp:\n"<< p_BP_opt << "\n";
    std::cout<< "norm correction was: " << (p_BP_opt - p_BP_init).norm() << " [m]\n\n";
    std::cout<< "Optimized colmap - global reference scale: "<< estimator.getScale() << "\n";
    std::cout<< "Scale correction was: "<< estimator.getScaleCorr() << "\n\n";

    std::cout<< "Gravity in world frame:\n"<< estimator.getGravity() << "\n\n";

    // Save final values
    std::cout<< "Saving optimized values ...\n\n";
    saveTrajectoryVelocityBiases(trace_dir, estimator, true);
    saveErrors(trace_dir, opti_gp_errors, opti_reproj_errors, true);
    saveScalar(trace_dir, "scale.txt", estimator.getScale());
    savePosition(trace_dir, "p_BP.txt", estimator.getPbp());
    savePose(trace_dir, "T_BC.txt", estimator.getImuCamRelTransformation());
    saveScalar(trace_dir, "cam_imu_time_offset.txt", estimator.getCamImuTimeOffset());
    savePosition(trace_dir, "gravity.txt", estimator.getGravity());
    saveScalar(trace_dir, "globalsensor_imu_time_offset.txt", estimator.getGlobSensImuTimeOffset());
    save3DPoints(trace_dir, points3d, false);

    auto sys_end_time = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(sys_end_time);
    std::cout << "Finishing at time: " << std::ctime(&end_time);

    return 0;
}

