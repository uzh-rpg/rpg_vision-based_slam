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

#include <iostream>
#include <math.h>
#include <filesystem>

#include <yaml-cpp/yaml.h>

#include "spline/bspline_so3r3.h"
#include "util/colmap_utils.h"
#include "util/load.h"
#include "util/print.h"
#include "util/save.h"


template <class Spline>
void saveSpline(const std::string& trace_dir, const Spline& spline)
{
    std::string spline_trace_fn = "spline.txt";
    std::ofstream spline_trace;
    createTrace(trace_dir, spline_trace_fn, &spline_trace);
    spline_trace.precision(9);
    
    double spline_dt_hz = 100.0;
    double spline_dt_s = 1.0 / spline_dt_hz;
    int64_t spline_dt_ns = static_cast<int64_t>(spline_dt_s * spline.s_to_ns_);
    traceSpline(spline, spline_dt_ns, spline_trace);
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

    GVI_FUSION_ASSERT_STREAM(argc>1, "Please, pass the .yaml filename");
    std::string yaml_fn = argv[1];

    // It has to be a const var.
    // Loading from yaml would require a big code refactoring.
    const int order = 6;
    std::cout << "Spline order: " << order << "\n\n";

    YAML::Node config = YAML::LoadFile(yaml_fn);

    // This is just to make sure to load the correct initialization.
    int config_order = config["spline_order"].as<int>();
    GVI_FUSION_ASSERT_STREAM(config_order == order, 
    "Spline order in src and yaml file do not match!");

    const std::string gp_fn = config["gp_fn"].as<std::string>();
    const double spline_knots_dt_s = config["spline_control_nodes_dt_s"].as<double>();
    const int spline_knots_dt_ms = spline_knots_dt_s * 1000;
    const std::string spline_fn = config["colmap_spline_dir"].as<std::string>() + 
    "/order_" + std::to_string(order) + "/dt_" + std::to_string(spline_knots_dt_ms) + "_ms/spline_knots.txt";
    std::string trace_dir = config["spline_global_ref_alignment_dir"].as<std::string>();
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/order_" + std::to_string(order);
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/dt_" + std::to_string(spline_knots_dt_ms) + "_ms";
    std::filesystem::create_directory(trace_dir);

    SplineParameters spline_params;
    spline_params.loadFromYaml(config);

    const std::string dataset_fn = config["dataset"].as<std::string>();

    std::cout << "Dataset: " << dataset_fn << "\n";
    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "global measurements fn: " << gp_fn << "\n";
    std::cout << "spline fn: " << spline_fn << "\n";
    std::cout << "output dir: " << trace_dir << "\n";

    // Load global measurements
    Eigen::aligned_vector<GlobalPosMeasurement> gp_meas;
    if(loadGpMeasurements(gp_fn, gp_meas))
    {
        std::cout << "Loaded " << gp_meas.size() << " leica measurements.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load global measurements.");
    }

    // Load spline knots
    Eigen::aligned_vector<Sophus::SO3d> knots_r_gb;
    Eigen::aligned_vector<Eigen::Vector3d> knots_p_gb;
    std::vector<double> knots_ts;
    if(loadSplineFromFile(spline_fn, knots_r_gb, knots_p_gb, knots_ts))
    {
        std::cout << "Loaded " << knots_ts.size() << " spline knots.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load spline");
    }
    
    double dt_s = spline_params.control_nodes_dt_s;
    int64_t dt_ns = static_cast<int64_t>(dt_s * 1e9);
    const int64_t start_t_ns = knots_ts.front() * 1e9;

    BSplineSO3R3<order> spline_gb(dt_ns, start_t_ns);
    spline_gb.initializeFromNumTraj(knots_r_gb, knots_p_gb);

    // Load initial T_wg and scale
    Sophus::SE3d T_wg_init;
    if (loadPose(trace_dir + "/T_wg_init.txt", T_wg_init))
    {
        std::cout << "Loaded initial T_wg:\n\n";
        printSophusPose(T_wg_init);
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not initial T_wg");
    }

    double scale_init;
    if (loadScalar<double>(trace_dir + "/scale_init.txt", scale_init))
    {
        std::cout << "Loaded initial scale: " << scale_init << "\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not initial scale");
    }
    
    spline_gb.setWorldColmapTransformation(T_wg_init);
    spline_gb.setScale(scale_init);

    double t_cam_gp_init = config["init_t_offset_cam_gp"].as<double>();
    spline_gb.setInitSplineGlobSensTimeOffset(t_cam_gp_init);

    const bool optimize_spline_globsens_time_offset = 
    config["optimize_spline_globsens_time_offset"].as<bool>();
    spline_gb.setOptimizeTimeOffsetSplineGlobalSensor(optimize_spline_globsens_time_offset);

    Eigen::Vector3d init_p_bp;
    init_p_bp.setZero();
    spline_gb.setPbp(init_p_bp);
    const bool optimize_pbp = config["optimize_pBP"].as<bool>();
    spline_gb.setOptimizePbp(optimize_pbp);

    spline_gb.addRotationLocalParametrization();

    // add measurements
    for (size_t i = 0; i < gp_meas.size(); i++)
    {
        GlobalPosMeasurement meas = gp_meas[i];
        int64_t t_ns = static_cast<int64_t>(meas.t * 1e9);

        spline_gb.addWorldColmapAlignmentMeasurement(meas.pos, t_ns, 1.0);
    }

    // Evaluate initial errors
    Eigen::aligned_vector<double> init_errors = spline_gb.evaluateWorldColmapAlignmentErrors(gp_meas);
    double init_mean_err = 0.0;
    for (double err : init_errors) {init_mean_err += err;}
    init_mean_err /= static_cast<double>(init_errors.size());
    std::cout<< "Number of errors: "<< init_errors.size() <<"\n";
    std::cout<< "Initial error mean: "<< init_mean_err <<"\n\n";

    // Solve
    int n_max_iters = config["n_max_iters"].as<int>();
    int n_threads = config["n_threads"].as<int>();
    spline_gb.loadOptimizationParams(n_max_iters, 0, n_threads);
    std::cout << "\n\nOptimizing ...\n\n";
    ceres::Solver::Summary summary = spline_gb.optimize(n_max_iters);
    spline_gb.relinearizeSplineGlobSensTimeOffset();

    // Evaluate final errors
    Eigen::aligned_vector<double> errors = spline_gb.evaluateWorldColmapAlignmentErrors(gp_meas);
    double mean_err = 0.0;
    for (double err : errors) {mean_err += err;}
    mean_err /= static_cast<double>(errors.size());
    std::cout<< "Number of errors: "<< errors.size() <<"\n";
    std::cout<< "Final error mean: "<< mean_err <<"\n\n";

    std::cout<< "Optimized colmap - global reference scale: "<< spline_gb.getScale() << "\n";
    std::cout<< "Scale correction was: "<< spline_gb.getScaleCorr() << "\n\n";

    std::cout<< "Optimized world - colmap transformation (= T_w_g) :\n";
    printSophusPose(spline_gb.getWorldColmapTransformation());
    std::cout<< "Transformation correction was:\n"; 
    Sophus::SE3d T_wg_corr = spline_gb.getWorldColmapTransformationCorr();
    printSophusPose(T_wg_corr);
    double rot_angle = acos((T_wg_corr.rotationMatrix().trace() - 1.0 - 1e-6) / 2.0);
    std::cout<< "rotation angle correction was: " << (rot_angle*180.0)/M_PI << " [deg]\n";
    std::cout<< "translation norm correction was: " << T_wg_corr.translation().norm() << " [m]\n\n"; 

    std::cout<< "Optimized spline - global sensor time offset: " << std::setprecision(16) << spline_gb.getSplineGlobSensTimeOffset() << " [s]\n";
    std::cout<< "Time offset correction was: "<< spline_gb.getSplineGlobSensTimeOffsetCorr() << " [s]\n\n";
    
    std::cout<< "Optimized p_bp:\n" << std::setprecision(6) << spline_gb.getPbp() << "\n\n";

    // Save final values
    saveScalar(trace_dir, "scale.txt", spline_gb.getScale());
    Sophus::SE3d T_wg = spline_gb.getWorldColmapTransformation();
    //Eigen::Matrix3d R_wg;
    //fixRotationMatrix(T_wg.rotationMatrix(), R_wg);
    //Sophus::SE3d T_wg_fixed = Sophus::SE3d(R_wg, T_wg.translation());
    savePose(trace_dir, "T_wg.txt", T_wg);
    saveScalar(trace_dir, "spline_globalsensor_time_offset.txt", spline_gb.getSplineGlobSensTimeOffset());
    savePosition(trace_dir, "p_bp.txt", spline_gb.getPbp());
    saveSpline(trace_dir, spline_gb);

    return 0;
}

