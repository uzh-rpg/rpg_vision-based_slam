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
const Eigen::aligned_vector<double>& gp_errs, 
const bool optimized)
{
    std::string acc_errs_trace_fn;
    std::string gyro_errs_trace_fn;
    std::string gp_errs_trace_fn;
    if (optimized)
    {
        acc_errs_trace_fn = "acc_errors.txt";
        gyro_errs_trace_fn = "gyro_errors.txt";
        gp_errs_trace_fn = "global_position_errors.txt";
    }
    else
    {
        acc_errs_trace_fn = "initial_acc_errors.txt";
        gyro_errs_trace_fn = "initial_gyro_errors.txt";
        gp_errs_trace_fn = "initial_global_position_errors.txt";
    }

    std::ofstream acc_errs_trace;
    createTrace(trace_dir, acc_errs_trace_fn, &acc_errs_trace);
    acc_errs_trace.precision(9);
    trace1DVector(acc_errs, acc_errs_trace);

    std::ofstream gyro_errs_trace;
    createTrace(trace_dir, gyro_errs_trace_fn, &gyro_errs_trace);
    gyro_errs_trace.precision(9);
    trace1DVector(gyro_errs, gyro_errs_trace);

    std::ofstream gp_errs_trace;
    createTrace(trace_dir, gp_errs_trace_fn, &gp_errs_trace);
    gp_errs_trace.precision(9);
    trace1DVector(gp_errs, gp_errs_trace);
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

    YAML::Node config = YAML::LoadFile(yaml_fn);

    const std::string imu_fn = config["imu_fn"].as<std::string>();
    const std::string gp_fn = config["gp_fn"].as<std::string>();

    const double spline_knots_dt_s = config["spline_control_nodes_dt_s"].as<double>();
    const double sampling_spline_rate_hz = config["sampling_spline_rate"].as<double>();
    const int spline_knots_dt_ms = spline_knots_dt_s * 1000;
    const std::string init_spline_fn = config["spline_dir"].as<std::string>() + 
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
    std::cout << "gp fn: " << gp_fn << "\n";
    std::cout << "init spline fn: " << init_spline_fn << "\n";
    std::cout << "result dir: " << trace_dir << "\n\n";

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
        GVI_FUSION_ASSERT_STREAM(false, "Could not imu");
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

    Eigen::aligned_vector<Sophus::SO3d> init_knots_q_wb;
    Eigen::aligned_vector<Eigen::Vector3d> init_knots_r_wb;
    std::vector<double> init_knots_ts;
    if(loadSplineFromFile(init_spline_fn, init_knots_q_wb, init_knots_r_wb, init_knots_ts))
    {
        std::cout << "Loaded " << init_knots_q_wb.size() << " initial spline knots.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load spline.");
    }

    // Initialize spline
    SplineParameters spline_params;
    spline_params.loadFromYaml(config);

    double dt_s = spline_params.control_nodes_dt_s;
    int64_t dt_ns = static_cast<int64_t>(dt_s * 1e9);

    const int64_t start_t_ns = init_knots_ts.front() * 1e9;

    BSplineSO3R3<order> spline_wb(dt_ns, start_t_ns);
    spline_wb.setImu(imu);

    spline_wb.initializeFromNumTraj(init_knots_q_wb, init_knots_r_wb);

    Eigen::Vector3d g;
    g = imu->calib_ptr_->g_W_;
    spline_wb.setGravity(g);
    const bool optimize_gravity = config["optimize_gravity"].as<bool>();
    spline_wb.setOptimizeGravity(optimize_gravity);
    // p_bp is initialize to zero
    const bool optimize_pbp = config["optimize_pBP"].as<bool>();
    spline_wb.setOptimizePbp(optimize_pbp);
    Eigen::Vector3d p_bp_init = Eigen::Vector3d::Zero();

    const bool optimize_spline_globsens_time_offset = 
    config["optimize_spline_globsens_time_offset"].as<bool>();
    spline_wb.setOptimizeTimeOffsetSplineGlobalSensor(optimize_spline_globsens_time_offset);

    spline_wb.addRotationLocalParametrization();
    
    double dt_bias_spline_s = config["bias_spline_control_nodes_dt_s"].as<double>();
    Eigen::Vector3d init_acc_bias(config["init_acc_bias"].as<std::vector<double>>().data());
    Eigen::Vector3d init_gyro_bias(config["init_gyro_bias"].as<std::vector<double>>().data());
    spline_wb.initializeBiases(dt_bias_spline_s, init_acc_bias, init_gyro_bias);

    // Add measurements
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

    if (dataset_fn == "EuRoC")
    {
        for (size_t i = 0; i < gp_meas.size(); i++)
        {
            GlobalPosMeasurement meas = gp_meas[i];
            int64_t t_ns = static_cast<int64_t>(meas.t * 1e9);
            spline_wb.addGlobalPosMeasurement(meas.pos, t_ns, gp_meas_std);
        }
    }
    else
    {
        /* We need to do:
        1 - use addGlobalPosMeasurement() instead of addPositionMeasurement()
        2 - optimize p_BP (done with point 1)
        3 - point 1 would optimize t offset between spline and gp sensor.
        This is wrong since spline has gp sensor times.
        */
        GVI_FUSION_ASSERT_STREAM(false, "GI fusion not implemented on this dataset!");
    }
    
    // imu
    for (size_t i = 0; i < imu->measurements_.size(); i++)
    {
        ImuMeasurement meas = imu->measurements_[i];
        int64_t t_ns = static_cast<int64_t>(meas.t * 1e9);

        spline_wb.addAccelMeasurement(meas.lin_acc, t_ns);
        spline_wb.addGyroMeasurement(meas.ang_vel, t_ns);
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
    // GP
    Eigen::aligned_vector<double> init_gp_errors = spline_wb.evaluateGlobalPosErrors(gp_meas);
    double init_mean_gp_err = 0.0;
    for (double err : init_gp_errors) {init_mean_gp_err += err;}
    init_mean_gp_err /= static_cast<double>(init_gp_errors.size());
    std::cout<< "Number of global position errors: "<< init_gp_errors.size() <<"\n";
    std::cout<< "Initial global position error mean: "<< init_mean_gp_err <<"\n\n";

    // Save initial values
    std::cout<< "Saving initial values ...\n\n";    
    saveSpline(trace_dir, spline_wb, sampling_spline_rate_hz, false);
    savePredictedImuMeasurements(trace_dir, spline_wb, imu, false);
    saveErrors(trace_dir, init_acc_errors, init_gyro_errors, init_gp_errors, false);
    saveScalar(trace_dir, "initial_spline_globalsensor_time_offset.txt", spline_wb.getSplineGlobSensTimeOffset());
    savePosition(trace_dir, "initial_p_bp.txt", spline_wb.getPbp());
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
    // GP
    Eigen::aligned_vector<double> opti_gp_errors = spline_wb.evaluateGlobalPosErrors(gp_meas);
    double opti_mean_gp_err = 0.0;
    for (double err : opti_gp_errors) {opti_mean_gp_err += err;}
    opti_mean_gp_err /= static_cast<double>(opti_gp_errors.size());
    std::cout<< "Number of global position errors: "<< opti_gp_errors.size() <<"\n";
    std::cout<< "Optimized global position error mean: "<< opti_mean_gp_err <<"\n\n";

    std::cout<< "Optimized global sensor - imu time offset: "<< spline_wb.getSplineImuTimeOffset() << " [s]\n";

    Eigen::Vector3d p_bp_fin = spline_wb.getPbp();
    std::cout<< "Optimized p_bp:\n"<< p_bp_fin << "\n";
    std::cout<< "norm correction was: " << (p_bp_fin - p_bp_init).norm() << " [m]\n\n";

    std::cout<< "Gravity in world frame:\n"<< spline_wb.getGravity() << "\n\n";

    // Save final values
    std::cout<< "Saving optimized values ...\n\n";
    saveSpline(trace_dir, spline_wb, sampling_spline_rate_hz, true);
    savePredictedImuMeasurements(trace_dir, spline_wb, imu, true);
    saveBiasSplines(trace_dir, spline_wb);
    saveErrors(trace_dir, opti_acc_errors, opti_gyro_errors, opti_gp_errors, true);
    savePosition(trace_dir, "p_bp.txt", spline_wb.getPbp());
    saveScalar(trace_dir, "globalsensor_imu_time_offset.txt", spline_wb.getSplineImuTimeOffset());
    savePosition(trace_dir, "gravity.txt", spline_wb.getGravity());

    auto sys_end_time = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(sys_end_time);
    std::cout << "Finishing at time: " << std::ctime(&end_time);

    return 0;
}

