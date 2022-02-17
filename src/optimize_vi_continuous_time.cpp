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
#include <iostream>
#include <filesystem>

#include "spline/bspline_so3r3.h"
#include "util/colmap_utils.h"
#include "util/load.h"
#include "util/load_calibration.h"
#include "util/print.h"
#include "util/save.h"


template <class Spline>
void saveSpline(const std::string& trace_dir, const Spline& spline, double sampling_dt_hz, const bool optimized)
{
    std::string spline_trace_fn;
    if (optimized)
    {
        spline_trace_fn = "spline.txt";
    }
    else
    {
        spline_trace_fn = "initial_spline.txt";
    }
    std::ofstream spline_trace;
    createTrace(trace_dir, spline_trace_fn, &spline_trace);
    spline_trace.precision(15);
    
    double spline_dt_hz = sampling_dt_hz;
    double spline_dt_s = 1.0 / spline_dt_hz;
    int64_t spline_dt_ns = static_cast<int64_t>(spline_dt_s * spline.s_to_ns_);
    traceSpline(spline, spline_dt_ns, spline_trace);
}


template <class Spline>
void saveBiasSplines(const std::string& trace_dir, const Spline& spline)
{
    // Acc bias
    std::string acc_bias_spline_trace_fn = "acc_bias_spline.txt";
    std::ofstream acc_bias_spline_trace;
    createTrace(trace_dir, acc_bias_spline_trace_fn, &acc_bias_spline_trace);
    acc_bias_spline_trace.precision(9);
    
    double spline_dt_hz = 10.0;
    double spline_dt_s = 1.0 / spline_dt_hz;
    int64_t spline_dt_ns = static_cast<int64_t>(spline_dt_s * spline.s_to_ns_);
    traceAccBiasSpline(spline, spline_dt_ns, acc_bias_spline_trace);

    // Gyro bias
    std::string gyro_bias_spline_trace_fn = "gyro_bias_spline.txt";
    std::ofstream gyro_bias_spline_trace;
    createTrace(trace_dir, gyro_bias_spline_trace_fn, &gyro_bias_spline_trace);
    gyro_bias_spline_trace.precision(9);

    traceGyroBiasSpline(spline, spline_dt_ns, gyro_bias_spline_trace);
}


template <class Spline>
void savePredictedImuMeasurements(const std::string& trace_dir, const Spline& spline, 
const std::shared_ptr<Imu>& imu, const bool optimized)
{
    std::string predicted_imu_trace_fn;
    if (optimized)
    {
        predicted_imu_trace_fn = "predicted_imu.txt";
    }
    else
    {
        predicted_imu_trace_fn = "initial_predicted_imu.txt";
    }

    Eigen::aligned_vector<ImuMeasurement> imu_sampled_from_spline;
    int64_t start_t_ns = spline.getStartTns();
    int64_t end_t_ns = spline.getEndTns();
    for (size_t i = 0; i < imu->measurements_.size(); i++)
    {
        ImuMeasurement meas = imu->measurements_[i];
        double t_offset_cam_imu_s = spline.getSplineImuTimeOffset();
        int64_t t_ns = static_cast<int64_t>((meas.t - t_offset_cam_imu_s) * spline.s_to_ns_);

        if (t_ns < start_t_ns) continue;
        if (t_ns > end_t_ns) break;

        Eigen::Vector3d acc;
        Eigen::Vector3d gyro;
        acc = spline.getAccel(t_ns);
        gyro = spline.getGyro(t_ns);

        ImuMeasurement m;
        m.t = meas.t;
        m.lin_acc = acc;
        m.ang_vel = gyro;
        imu_sampled_from_spline.emplace_back(m);
    }

    std::ofstream predicted_imu_trace;
    createTrace(trace_dir, predicted_imu_trace_fn, &predicted_imu_trace);
    predicted_imu_trace.precision(15);
    traceImuMeasurements(imu_sampled_from_spline, predicted_imu_trace);
}


void saveErrors(const std::string& trace_dir, 
const Eigen::aligned_vector<double>& acc_errs, 
const Eigen::aligned_vector<double>& gyro_errs,
const Eigen::aligned_vector<double>& reproj_errs, 
const bool optimized)
{
    std::string acc_errs_trace_fn;
    std::string gyro_errs_trace_fn;
    std::string reproj_errs_trace_fn;
    if (optimized)
    {
        acc_errs_trace_fn = "acc_errors.txt";
        gyro_errs_trace_fn = "gyro_errors.txt";
        reproj_errs_trace_fn = "reprojection_errors.txt";
    }
    else
    {
        acc_errs_trace_fn = "initial_acc_errors.txt";
        gyro_errs_trace_fn = "initial_gyro_errors.txt";
        reproj_errs_trace_fn = "initial_reprojection_errors.txt";
    }

    std::ofstream acc_errs_trace;
    createTrace(trace_dir, acc_errs_trace_fn, &acc_errs_trace);
    acc_errs_trace.precision(9);
    trace1DVector(acc_errs, acc_errs_trace);

    std::ofstream gyro_errs_trace;
    createTrace(trace_dir, gyro_errs_trace_fn, &gyro_errs_trace);
    gyro_errs_trace.precision(9);
    trace1DVector(gyro_errs, gyro_errs_trace);

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


void saveOptimized3DPoints(const std::string& trace_dir, const Eigen::aligned_vector<Eigen::Vector3d>& points)
{
    std::string trace_fn = "points3D.txt";
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    tracePoints3DArray(points, trace);
}


void savePose(const std::string& trace_dir, const std::string& trace_fn, const Sophus::SE3d& pose)
{
    std::ofstream trace;
    createTrace(trace_dir, trace_fn, &trace);
    trace.precision(9);
    tracePose(pose, trace);
}


void savePosition(const std::string& trace_dir, const std::string& trace_fn, const Eigen::Vector3d& position)
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

    // It has to be a const var.
    // Loading from yaml would require a big code refactoring.
    const int order = 6;
    std::cout << "Spline order: " << order << "\n\n";

    YAML::Node config = YAML::LoadFile(yaml_fn);

    const int cam_id = config["cam_id"].as<int>();
    const std::string camimu_calib_fn = config["camimu_calib_fn"].as<std::string>();
    const std::string imu_fn = config["imu_fn"].as<std::string>();
    const std::string colmap_dir = config["colmap_dir"].as<std::string>();

    const double spline_knots_dt_s = config["spline_control_nodes_dt_s"].as<double>();
    const double sampling_spline_rate_hz = config["sampling_spline_rate"].as<double>();
    const int spline_knots_dt_ms = spline_knots_dt_s * 1000;
    const std::string init_spline_fn = config["colmap_spline_dir"].as<std::string>() + 
    "/order_" + std::to_string(order) + "/dt_" + std::to_string(spline_knots_dt_ms) + "_ms/spline_knots.txt";
 
    std::string trace_dir = config["full_batch_optimization_dir"].as<std::string>();
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/continuous_time";
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/order_" + std::to_string(order);
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/dt_" + std::to_string(spline_knots_dt_ms) + "_ms";
    std::filesystem::create_directory(trace_dir);

    const std::string dataset_fn = config["dataset"].as<std::string>();

    std::cout << "Dataset: " << dataset_fn << "\n";
    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "camimu calib fn: " << camimu_calib_fn << "\n";
    std::cout << "colmap dir: " << colmap_dir << "\n";
    std::cout << "init spline fn: " << init_spline_fn << "\n";
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
        std::cout << "Custom dataset. Using EuRoC calibration format." << "\n\n";
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
        GVI_FUSION_ASSERT_STREAM(false, "Could not imu");
    }

    const std::string colmap_cam_fn = colmap_dir + "cameras.bin";
    Eigen::aligned_vector<Camera> cameras;
    readCamerasBinary(colmap_cam_fn, cameras);
    std::cout << "Loaded: " << cameras.size() << " cameras.\n\n";

    Sophus::SE3d T_cb = camimu_calib.T_cam_imu_;
    Sophus::SE3d T_bc = T_cb.inverse();
    cameras[0].setTbc(T_bc);
    cameras[0].setTimeOffsetCamImu(camimu_calib.timeshift_cam_imu_);

    const std::string colmap_img_fn = colmap_dir + "images.bin";
    Eigen::aligned_vector<Frame> frames;
    const std::string colmap_points3d_fn = colmap_dir + "points3D.bin";
    Eigen::aligned_unordered_map<uint64_t, Point3D> points3d;
    const int grid_cell_size = config["grid_cell_size"].as<int>();
    readImagesAndPoints3DBinary(colmap_img_fn, colmap_points3d_fn, cameras, frames, points3d, grid_cell_size);
    std::cout << "Loaded: " << frames.size() << " images.\n\n";
    std::cout << "Loaded: " << points3d.size() << " 3D points.\n\n";

    Eigen::aligned_vector<Sophus::SO3d> init_knots_q_gb;
    Eigen::aligned_vector<Eigen::Vector3d> init_knots_r_gb;
    std::vector<double> init_knots_ts;
    if(loadSplineFromFile(init_spline_fn, init_knots_q_gb, init_knots_r_gb, init_knots_ts))
    {
        std::cout << "Loaded " << init_knots_q_gb.size() << " initial spline knots.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load spline.");
    }

    // This is loaded as transpose
    Eigen::Matrix4d T_eigen(config["T_wg"].as<std::vector<double>>().data());
    Sophus::SE3d T_wg(T_eigen.transpose());

    std::cout << "Loaded T_wg:\n";
    printSophusPose(T_wg);

    double scale = config["initial_scale"].as<double>();
    std::cout << "Loaded scale: " << scale << "\n\n";
    const bool optimize_scale = config["optimize_scale"].as<bool>();
    
    // Re-scale colmap trajectory
    Eigen::aligned_vector<Sophus::SO3d> init_knots_q_wb; 
    Eigen::aligned_vector<Eigen::Vector3d> init_knots_r_wb;
    for (size_t i = 0; i < init_knots_q_gb.size(); i++)
    {
        Sophus::SO3d q_wb = T_wg.so3() * init_knots_q_gb[i];
        Eigen::Vector3d t_wb = scale * (T_wg.rotationMatrix() * init_knots_r_gb[i]) + T_wg.translation();

        init_knots_q_wb.push_back(q_wb);
        init_knots_r_wb.push_back(t_wb);
    }

    for (auto& it: points3d) 
    {
        Eigen::Vector3d p_wl = scale * (T_wg.rotationMatrix() * it.second.p_gl) + T_wg.translation();
        it.second.setPositionInWorldFrame(p_wl);
    }

    // Load frame timestamps from colmap
    const std::string colmap_cam_est_fn = colmap_dir + "colmap_cam_estimates.txt";
    Eigen::aligned_vector<double> frames_ts;
    if(!loadFrameTimes(colmap_cam_est_fn, frames_ts))
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load colmap frame timestamps.");
    }
    GVI_FUSION_ASSERT_STREAM(frames.size() == frames_ts.size(), 
    "frames.size()= " << frames.size() << " frames_ts.size()= " << frames_ts.size());
    for (size_t i = 0; i < frames.size(); i++)
    {
        frames[i].setTimestamp(frames_ts[i]);
    }

    // Initialize spline
    SplineParameters spline_params;
    spline_params.loadFromYaml(config);

    double dt_s = spline_params.control_nodes_dt_s;
    int64_t dt_ns = static_cast<int64_t>(dt_s * 1e9);

    const int64_t start_t_ns = init_knots_ts.front() * 1e9;

    BSplineSO3R3<order> spline_wb(dt_ns, start_t_ns);
    spline_wb.setCameras(cameras);
    spline_wb.setImu(imu);

    spline_wb.initializeFromNumTraj(init_knots_q_wb, init_knots_r_wb);

    Eigen::Vector3d g;
    g = imu->calib_ptr_->g_W_;
    spline_wb.setGravity(g);
    const bool optimize_gravity = config["optimize_gravity"].as<bool>();
    spline_wb.setOptimizeGravity(optimize_gravity);
    if (optimize_gravity)
    {
        spline_wb.addGravityNormError();
    }

    spline_wb.setScale(scale);
    spline_wb.setOptimizeScale(optimize_scale);
    spline_wb.addRotationLocalParametrization();
    spline_wb.set3DPoints(points3d);

    const bool optimize_camimu_extrinsics = config["optimize_camimu_extrinsics"].as<bool>();
    spline_wb.setOptimizeCamImuExtrinsics(optimize_camimu_extrinsics);
    
    double dt_bias_spline_s = config["bias_spline_control_nodes_dt_s"].as<double>();
    Eigen::Vector3d init_acc_bias(config["init_acc_bias"].as<std::vector<double>>().data());
    Eigen::Vector3d init_gyro_bias(config["init_gyro_bias"].as<std::vector<double>>().data());
    spline_wb.initializeBiases(dt_bias_spline_s, init_acc_bias, init_gyro_bias);

    auto frame_it = frames.begin();
    spline_wb.initializeGrid(&(*frame_it));
    spline_wb.computeValidReprojections(frames);
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

    // Add measurements
    // imu
    for (size_t i = 0; i < imu->measurements_.size(); i++)
    {
        ImuMeasurement meas = imu->measurements_[i];
        int64_t t_ns = static_cast<int64_t>(meas.t * 1e9);

        spline_wb.addAccelMeasurement(meas.lin_acc, t_ns);
        spline_wb.addGyroMeasurement(meas.ang_vel, t_ns);
    }

    // vision
    double sigma_reproj_px = config["sigma_reproj_px"].as<double>();
    std::cout << "Sigma reprojection errors: " << sigma_reproj_px << " [pixel]\n\n";
    for(auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
        int64_t t_ns = static_cast<int64_t>(frame->timestamp * 1e9);
        spline_wb.addImagePointsMeasurement(&(*frame), t_ns, sigma_reproj_px);
    }

    // add imu bias costs
    double bias_err_freq_hz = config["bias_err_freq_hz"].as<double>();
    double bias_err_dt_s = config["bias_err_dt_s"].as<double>();
    std::cout << "Using dt = " << dt_bias_spline_s << " s between knots of bias spline\n";
    std::cout << "Using bias error frequency: " << bias_err_freq_hz << " Hz\n";
    std::cout << "Using bias error dt: " << bias_err_dt_s << " s\n\n";
    GVI_FUSION_ASSERT_STREAM(bias_err_dt_s >= 4 * dt_bias_spline_s, 
    "Increase bias error dt such that bias_err_dt_s >= 4 * dt_bias_spline_s.");
    spline_wb.addImuBiasCost(bias_err_freq_hz, bias_err_dt_s);

    // Evaluate initial errors
    std::cout << "+-+-+-+-+- Initial Errors +-+-+-+-+-\n\n";
    // Imu
    Eigen::aligned_vector<double> init_acc_errors = spline_wb.evaluateAccelErrors(imu->measurements_);
    Eigen::aligned_vector<double> init_gyro_errors = spline_wb.evaluateGyroErrors(imu->measurements_);
    double init_mean_acc_err = 0.0;
    for (double err : init_acc_errors) {init_mean_acc_err += err;}
    init_mean_acc_err /= static_cast<int>(init_acc_errors.size());
    std::cout<< "Number of imu errors: "<< init_acc_errors.size() <<"\n";
    std::cout<< "Initial accelerometer error mean: "<< init_mean_acc_err <<"\n";
    double init_mean_gyro_err = 0.0;
    for (double err : init_gyro_errors) {init_mean_gyro_err += err;}
    init_mean_gyro_err /= static_cast<int>(init_gyro_errors.size());
    std::cout<< "Initial gyro error mean: "<< init_mean_gyro_err <<"\n\n";
    // Reprojection
    Eigen::aligned_vector<double> init_reproj_errors = spline_wb.evaluateReprojectionErrors(frames);
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
    saveSpline(trace_dir, spline_wb, sampling_spline_rate_hz, false);
    savePredictedImuMeasurements(trace_dir, spline_wb, imu, false);
    saveErrors(trace_dir, init_acc_errors, init_gyro_errors, init_reproj_errors, false);
    save3DPoints(trace_dir, points3d, false);
    saveScalar(trace_dir, "initial_scale.txt", spline_wb.getScale());
    savePose(trace_dir, "initial_T_bc.txt", spline_wb.getImuCamRelTransformation());
    saveScalar(trace_dir, "initial_cam_imu_time_offset.txt", spline_wb.getSplineImuTimeOffset());
    savePosition(trace_dir, "initial_gravity.txt", spline_wb.getGravity());

    // Solve
    int n_max_outer_iters = config["n_max_outer_iters"].as<int>();
    int n_max_inner_iters = config["n_max_inner_iters"].as<int>();
    int n_threads = config["n_threads"].as<int>();
    spline_wb.loadOptimizationParams(n_max_outer_iters, n_max_inner_iters, n_threads);
    std::cout << "\n\nOptimizing ...\n\n";
    ceres::Solver::Summary summary;
    // Optimize for just 1 iteration before relinearizing the time offset.
    // We expect the largest change in the time offset after the 1 iteration of the solver.
    summary = spline_wb.optimize(1);
    spline_wb.relinearizeSplineImuTimeOffset();

    for (int i = 0; i < n_max_outer_iters; i++)
    {
        summary = spline_wb.optimize(n_max_inner_iters);
        spline_wb.relinearizeSplineImuTimeOffset();

        if (summary.termination_type == 0)
        {
            std::cout << "\n============================\n";
            std::cout << "============================\n";
            std::cout << "===== Solver converged =====\n";
            std::cout << "============================\n";
            std::cout << "============================\n\n";
            
            std::cout << "number of outer iterations = " << i+1 << "\n";
            break;
        }

        if (i == (n_max_outer_iters-1))
        {
            std::cout << "\n\n===== Solver did not converge =====\n\n";
        }
    }
    // just update before saving
    spline_wb.relinearizeSplineImuTimeOffset();

    // Evaluate final errors
    std::cout << "+-+-+-+-+- Final Errors +-+-+-+-+-\n\n";
    // Imu
    Eigen::aligned_vector<double> opti_acc_errors = spline_wb.evaluateAccelErrors(imu->measurements_);
    Eigen::aligned_vector<double> opti_gyro_errors = spline_wb.evaluateGyroErrors(imu->measurements_);
    double opti_mean_acc_err = 0.0;
    for (double err : opti_acc_errors) {opti_mean_acc_err += err;}
    opti_mean_acc_err /= static_cast<int>(opti_acc_errors.size());
    std::cout<< "Number of imu errors: "<< opti_acc_errors.size() <<"\n";
    std::cout<< "Optimized accelerometer error mean: "<< opti_mean_acc_err <<"\n";
    double opti_mean_gyro_err = 0.0;
    for (double err : opti_gyro_errors) {opti_mean_gyro_err += err;}
    opti_mean_gyro_err /= static_cast<int>(opti_gyro_errors.size());
    std::cout<< "Optimized gyro error mean: "<< opti_mean_gyro_err <<"\n\n";
    // Reprojection
    Eigen::aligned_vector<double> opti_reproj_errors = spline_wb.evaluateReprojectionErrors(frames);
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

    std::cout<< "Optimized imu - camera transformation (= T_b_c) :\n";
    printSophusPose(spline_wb.getImuCamRelTransformation());
    std::cout<< "Transformation correction was:\n"; 
    Sophus::SE3d T_bc_corr = spline_wb.getImuCamRelTransformationCorr();
    printSophusPose(T_bc_corr);
    double rot_angle_rad = acos((T_bc_corr.rotationMatrix().trace() - 1.0 - 1e-6) / 2.0);
    double rot_angle_deg = (rot_angle_rad * 180.0) / 3.14159;
    std::cout<< "rotation angle correction was: " << rot_angle_deg << " [deg]\n";
    std::cout<< "translation norm correction was: " << T_bc_corr.translation().norm() << " [m]\n\n"; 

    std::cout<< "Optimized cam - imu time offset: "<< spline_wb.getSplineImuTimeOffset() << " [s]\n";
    std::cout<< "Time offset correction was: "<< spline_wb.getSplineImuTimeOffsetCorr() << " [s]\n\n";

    std::cout<< "Optimized colmap - global reference scale: "<< spline_wb.getScale() << "\n";
    std::cout<< "Scale correction was: "<< spline_wb.getScaleCorr() << "\n\n";

    std::cout<< "Gravity in world frame:\n"<< spline_wb.getGravity() << "\n\n";

    // Save final values
    std::cout<< "Saving optimized values ...\n\n";
    saveSpline(trace_dir, spline_wb, sampling_spline_rate_hz, true);
    savePredictedImuMeasurements(trace_dir, spline_wb, imu, true);
    saveBiasSplines(trace_dir, spline_wb);
    saveErrors(trace_dir, opti_acc_errors, opti_gyro_errors, opti_reproj_errors, true);
    Eigen::aligned_vector<Eigen::Vector3d> opti_points3d = spline_wb.get3DPointsArray();
    saveOptimized3DPoints(trace_dir, opti_points3d);
    saveScalar(trace_dir, "scale.txt", spline_wb.getScale());
    savePose(trace_dir, "T_bc.txt", spline_wb.getImuCamRelTransformation());
    saveScalar(trace_dir, "cam_imu_time_offset.txt", spline_wb.getSplineImuTimeOffset());
    savePosition(trace_dir, "gravity.txt", spline_wb.getGravity());

    auto sys_end_time = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(sys_end_time);
    std::cout << "Finishing at time: " << std::ctime(&end_time);

    return 0;
}

