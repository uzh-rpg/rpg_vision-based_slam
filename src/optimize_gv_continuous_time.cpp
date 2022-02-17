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


void saveErrors(const std::string& trace_dir, 
const Eigen::aligned_vector<double>& gp_errs, 
const Eigen::aligned_vector<double>& reproj_errs, 
const bool optimized)
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
    const std::string gp_fn = config["gp_fn"].as<std::string>();
    const std::string colmap_dir = config["colmap_dir"].as<std::string>();

    const double spline_knots_dt_s = config["spline_control_nodes_dt_s"].as<double>();
    const double sampling_spline_rate_hz = config["sampling_spline_rate"].as<double>();
    const int spline_knots_dt_ms = spline_knots_dt_s * 1000;
    const std::string init_spline_fn = config["colmap_spline_dir"].as<std::string>() + 
    "/order_" + std::to_string(order) + "/dt_" + std::to_string(spline_knots_dt_ms) + "_ms/spline_knots.txt";
    
    const std::string alignment_dir = config["spline_global_ref_alignment_dir"].as<std::string>() + 
    "/order_" + std::to_string(order) + "/dt_" + std::to_string(spline_knots_dt_ms) + "_ms";
    
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
    std::cout << "gp fn: " << gp_fn << "\n";
    std::cout << "colmap dir: " << colmap_dir << "\n";
    std::cout << "init spline fn: " << init_spline_fn << "\n";
    std::cout << "alignment dir: " << alignment_dir << "\n";
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
    else if ((dataset_fn == "EuRoC") || (dataset_fn == "Custom"))
    {
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
    // To be consistent in the comparison with other approaches,
    // we still estimate body (= imu) poses.
    std::vector<double> init_knots_ts;
    if(loadSplineFromFile(init_spline_fn, init_knots_q_gb, init_knots_r_gb, init_knots_ts))
    {
        std::cout << "Loaded " << init_knots_q_gb.size() << " initial spline knots.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load spline.");
    }

    // Need to transform knots from colmap world frame to world frame
    Sophus::SE3d T_wg;
    // Here, it is not needed to load the initialization of T_wg and scale
    if (loadPose(alignment_dir + "/T_wg_init.txt", T_wg))
    {
        std::cout << "Loaded initial T_wg:\n\n";
        printSophusPose(T_wg);
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not T_wg");
    }

    double scale;
    if (loadScalar<double>(alignment_dir + "/scale_init.txt", scale))
    {
        std::cout << "Loaded scale: " << scale << "\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not scale");
    }
    const bool optimize_scale = config["optimize_scale"].as<bool>();

    Eigen::aligned_vector<Sophus::SO3d> init_knots_q_wb; 
    Eigen::aligned_vector<Eigen::Vector3d> init_knots_r_wb;
    for (size_t i = 0; i < init_knots_q_gb.size(); i++)
    {
        Sophus::SO3d q_wb = T_wg.so3() * init_knots_q_gb[i];
        Eigen::Vector3d t_wb = scale * (T_wg.rotationMatrix() * init_knots_r_gb[i]) + T_wg.translation();

        init_knots_q_wb.push_back(q_wb);
        init_knots_r_wb.push_back(t_wb);
    }

    // Need to transform 3d points from G (colmap frame) to W (global world frame).
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

    spline_wb.initializeFromNumTraj(init_knots_q_wb, init_knots_r_wb);

    double t_cam_gp_init = 0.0;
    spline_wb.setInitSplineGlobSensTimeOffset(t_cam_gp_init);
    Eigen::Vector3d p_bp;
    if (loadVector3d(alignment_dir + "/p_bp.txt", p_bp))
    {
        std::cout << "Loaded p_bp: \n" << p_bp << "\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load p_bp");
    }
    const bool optimize_pbp = config["optimize_pBP"].as<bool>();
    
    spline_wb.setPbp(p_bp);
    spline_wb.setOptimizePbp(optimize_pbp);
    Eigen::Vector3d p_bp_init = p_bp;

    const bool optimize_spline_globsens_time_offset = 
    config["optimize_spline_globsens_time_offset"].as<bool>();
    spline_wb.setOptimizeTimeOffsetSplineGlobalSensor(optimize_spline_globsens_time_offset);

    spline_wb.setScale(scale);
    spline_wb.setOptimizeScale(optimize_scale);
    spline_wb.addRotationLocalParametrization();
    spline_wb.set3DPoints(points3d);

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
        GlobalPosMeasurement meas = gp_meas[i];
        int64_t t_ns = static_cast<int64_t>(meas.t * 1e9);
        spline_wb.addGlobalPosMeasurement(meas.pos, t_ns, gp_meas_std);
    }

    // vision
    double sigma_reproj_px = config["sigma_reproj_px"].as<double>();
    std::cout << "Sigma reprojection errors: " << sigma_reproj_px << " [pixel]\n\n";
    for(auto frame = std::begin(frames); frame != std::end(frames); ++frame)
    {
        int64_t t_ns = static_cast<int64_t>(frame->timestamp * 1e9);
        spline_wb.addImagePointsMeasurement(&(*frame), t_ns, sigma_reproj_px);
    }


    // Evaluate initial errors
    std::cout << "+-+-+-+-+- Initial Errors +-+-+-+-+-\n\n";
    // GP
    Eigen::aligned_vector<double> init_gp_errors = spline_wb.evaluateGlobalPosErrors(gp_meas);
    double init_mean_gp_err = 0.0;
    for (double err : init_gp_errors) {init_mean_gp_err += err;}
    init_mean_gp_err /= static_cast<double>(init_gp_errors.size());
    std::cout<< "Number of global position errors: "<< init_gp_errors.size() <<"\n";
    std::cout<< "Initial global position error mean: "<< init_mean_gp_err <<"\n\n";
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
    saveErrors(trace_dir, init_gp_errors, init_reproj_errors, false);
    save3DPoints(trace_dir, points3d, false);
    saveScalar(trace_dir, "initial_spline_globalsensor_time_offset.txt", spline_wb.getSplineGlobSensTimeOffset());
    saveScalar(trace_dir, "initial_scale.txt", spline_wb.getScale());
    savePosition(trace_dir, "initial_p_bp.txt", spline_wb.getPbp());

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
    spline_wb.relinearizeSplineGlobSensTimeOffset();

    for (int i = 0; i < n_max_outer_iters; i++)
    {
        summary = spline_wb.optimize(n_max_inner_iters);
        spline_wb.relinearizeSplineGlobSensTimeOffset();

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
    spline_wb.relinearizeSplineGlobSensTimeOffset();

    // Evaluate final errors
    std::cout << "+-+-+-+-+- Final Errors +-+-+-+-+-\n\n";
    // GP
    Eigen::aligned_vector<double> opti_gp_errors = spline_wb.evaluateGlobalPosErrors(gp_meas);
    double opti_mean_gp_err = 0.0;
    for (double err : opti_gp_errors) {opti_mean_gp_err += err;}
    opti_mean_gp_err /= static_cast<double>(opti_gp_errors.size());
    std::cout<< "Number of global position errors: "<< opti_gp_errors.size() <<"\n";
    std::cout<< "Optimized global position error mean: "<< opti_mean_gp_err <<"\n\n";
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

    std::cout<< "Optimized spline - global sensor time offset: "<< spline_wb.getSplineGlobSensTimeOffset() << " [s]\n";
    std::cout<< "Time offset correction was: "<< spline_wb.getSplineGlobSensTimeOffsetCorr() << " [s]\n\n";

    Eigen::Vector3d p_bp_fin = spline_wb.getPbp();
    std::cout<< "Optimized p_bp:\n"<< p_bp_fin << "\n";
    std::cout<< "norm correction was: " << (p_bp_fin - p_bp_init).norm() << " [m]\n\n";

    std::cout<< "Optimized colmap - global reference scale: "<< spline_wb.getScale() << "\n";
    std::cout<< "Scale correction was: "<< spline_wb.getScaleCorr() << "\n\n";

    // Save final values
    std::cout<< "Saving optimized values ...\n\n";
    saveSpline(trace_dir, spline_wb, sampling_spline_rate_hz, true);
    saveErrors(trace_dir, opti_gp_errors, opti_reproj_errors, false);
    Eigen::aligned_vector<Eigen::Vector3d> opti_points3d = spline_wb.get3DPointsArray();
    saveOptimized3DPoints(trace_dir, opti_points3d);
    saveScalar(trace_dir, "spline_globalsensor_time_offset.txt", spline_wb.getSplineGlobSensTimeOffset());
    saveScalar(trace_dir, "scale.txt", spline_wb.getScale());
    savePosition(trace_dir, "p_bp.txt", spline_wb.getPbp());
    savePosition(trace_dir, "gravity.txt", spline_wb.getGravity());

    auto sys_end_time = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(sys_end_time);
    std::cout << "Finishing at time: " << std::ctime(&end_time);

    return 0;
}

