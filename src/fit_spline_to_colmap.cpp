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
#include <filesystem>

#include <yaml-cpp/yaml.h>

#include "spline/bspline_so3r3.h"
#include "util/colmap_utils.h"
#include "util/print.h"
#include "util/save.h"
#include "util/trajectory_utils.h"


template <class Spline>
void saveSplineKnots(const std::string& trace_dir, const Spline& spline, const Eigen::aligned_vector<double>& ts)
{
    std::string spline_trace_fn = "spline_knots.txt";
    std::ofstream spline_trace;
    createTrace(trace_dir, spline_trace_fn, &spline_trace);
    spline_trace.precision(15);
    
    traceSplineKnots(spline, ts, spline_trace);
}


template <class Spline>
void saveSpline(const std::string& trace_dir, const Spline& spline)
{
    std::string spline_trace_fn = "spline.txt";
    std::ofstream spline_trace;
    createTrace(trace_dir, spline_trace_fn, &spline_trace);
    spline_trace.precision(15);
    
    double spline_dt_hz = 100.0;
    double spline_dt_s = 1.0 / spline_dt_hz;
    int64_t spline_dt_ns = static_cast<int64_t>(spline_dt_s * spline.s_to_ns_);
    traceSpline(spline, spline_dt_ns, spline_trace);
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

    const std::string colmap_fn = config["colmap_fn"].as<std::string>();
    std::string colmap_spline_dir = config["colmap_spline_dir"].as<std::string>();
    std::filesystem::create_directory(colmap_spline_dir);
    colmap_spline_dir = colmap_spline_dir + "/colmap_fitted_spline";
    std::filesystem::create_directory(colmap_spline_dir);

    const double colmap_start_t = config["start_t"].as<double>();
    const double colmap_end_t = config["end_t"].as<double>();

    SplineParameters spline_params;
    spline_params.loadFromYaml(config);
    int dt_ms = spline_params.control_nodes_dt_s * 1e3;

    std::string trace_dir = colmap_spline_dir;
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/order_" + std::to_string(order);
    std::filesystem::create_directory(trace_dir);
    trace_dir = trace_dir + "/dt_" + std::to_string(dt_ms) + "_ms";
    std::filesystem::create_directory(trace_dir);

    std::cout << "+-+-+-+-+- Files +-+-+-+-+-\n";
    std::cout << "colmap_fn: " << colmap_fn << "\n";
    std::cout << "output dir: " << trace_dir << "\n";

    // Load colmap estimates
    Eigen::aligned_vector<Sophus::SO3d> colmap_r_gb;
    Eigen::aligned_vector<Eigen::Vector3d> colmap_p_gb;
    std::vector<double> colmap_ts;
    if(loadTrajectoryFromFile(colmap_fn, colmap_r_gb, colmap_p_gb, colmap_ts))
    {
        std::cout << "Loaded " << colmap_p_gb.size() << " colmap estimates.\n\n";
    }
    else
    {
        GVI_FUSION_ASSERT_STREAM(false, "Could not load colmap trajectory.");
    }

    // sample the relevant colmap measurements
    Eigen::aligned_vector<Sophus::SO3d> sampled_colmap_r_gb;
    Eigen::aligned_vector<Eigen::Vector3d> sampled_colmap_p_gb;
    Eigen::aligned_vector<double> sampled_colmap_ts;
    for (size_t i = 0; i < colmap_ts.size(); i++)
    {
        if ( (colmap_ts[i] >= colmap_start_t) && (colmap_ts[i] <= colmap_end_t) )
        {
            sampled_colmap_p_gb.push_back(colmap_p_gb[i]);
            sampled_colmap_r_gb.push_back(colmap_r_gb[i]);
            sampled_colmap_ts.push_back(colmap_ts[i]);
        }
    }    
    double start_knot_t_s = sampled_colmap_ts.front();
    double end_knot_t_s = sampled_colmap_ts.back();

    // Initialize spline
    double dt_s = spline_params.control_nodes_dt_s;
    int64_t dt_ns = static_cast<int64_t>(dt_s * 1e9);
    const int64_t start_t_ns = start_knot_t_s * 1e9;

    BSplineSO3R3<order> spline_gb(dt_ns, start_t_ns);

    Eigen::aligned_vector<double> knots_ts;
    double knot_t = start_knot_t_s;
    while (knot_t <= end_knot_t_s)
    {
        knots_ts.push_back(knot_t);
        knot_t += dt_s;
    }
    
    Eigen::aligned_vector<Sophus::SO3d> init_knots_r;
    Eigen::aligned_vector<Eigen::Vector3d> init_knots_p;

    linearInterpolation(sampled_colmap_p_gb, sampled_colmap_r_gb, sampled_colmap_ts, 
    knots_ts, init_knots_p, init_knots_r);
    
    spline_gb.initializeFromNumTraj(init_knots_r, init_knots_p);
    spline_gb.addRotationLocalParametrization();

    // add measurements
    double n_errs = config["num_errors"].as<double>();
    double dt_meas = (sampled_colmap_ts.back() - sampled_colmap_ts.front()) / n_errs;
    Eigen::aligned_vector<double> meas_ts;
    double meas_t = sampled_colmap_ts.front();
    while (meas_t <= sampled_colmap_ts.back())
    {
        meas_ts.push_back(meas_t);
        meas_t += dt_meas;
    }

    Eigen::aligned_vector<Sophus::SO3d> meas_r;
    Eigen::aligned_vector<Eigen::Vector3d> meas_p;
    linearInterpolation(sampled_colmap_p_gb, sampled_colmap_r_gb, sampled_colmap_ts, 
    meas_ts, meas_p, meas_r);

    for (size_t i = 0; i < meas_ts.size(); i++)
    {
        int64_t t_ns = static_cast<int64_t>(meas_ts[i] * 1e9);
        spline_gb.addPositionMeasurement(meas_p[i], t_ns, 1.0);
    }

    for (size_t i = 0; i < meas_ts.size(); i++)
    {
        int64_t t_ns = static_cast<int64_t>(meas_ts[i] * 1e9);
        spline_gb.addOrientationMeasurement(meas_r[i], t_ns, 1.0);
    }

    // Evaluate initial errors
    // Position
    Eigen::aligned_vector<double> init_pos_errors = spline_gb.evaluatePositionErrors(meas_p, meas_ts);
    double init_mean_pos_err = 0.0;
    for (double err : init_pos_errors) {init_mean_pos_err += err;}
    init_mean_pos_err /= static_cast<double>(init_pos_errors.size());
    std::cout<< "Number of position errors: "<< init_pos_errors.size() <<"\n";
    std::cout<< "Initial position error mean: "<< init_mean_pos_err << " [m]\n\n";

    // Orientation
    Eigen::aligned_vector<double> init_ori_errors = spline_gb.evaluateOrientationErrors(meas_r, meas_ts);
    double init_mean_ori_err = 0.0;
    for (double err : init_ori_errors) {init_mean_ori_err += err;}
    init_mean_ori_err /= static_cast<double>(init_ori_errors.size());
    std::cout<< "Number of orientation errors: "<< init_ori_errors.size() <<"\n";
    std::cout<< "Initial orientation error mean: "<< init_mean_ori_err <<" [rad]\n\n";

    // Solve
    int n_max_iters = config["n_max_iters"].as<int>();
    int n_threads = config["n_threads"].as<int>();
    spline_gb.loadOptimizationParams(n_max_iters, 0, n_threads);
    std::cout << "\n\nOptimizing ...\n\n";
    ceres::Solver::Summary summary = spline_gb.optimize(n_max_iters);

    // Evaluate final errors
    // Position
    Eigen::aligned_vector<double> opti_pos_errors = spline_gb.evaluatePositionErrors(meas_p, meas_ts);
    double opti_mean_pos_err = 0.0;
    for (double err : opti_pos_errors) {opti_mean_pos_err += err;}
    opti_mean_pos_err /= static_cast<double>(opti_pos_errors.size());
    std::cout<< "Number of position errors: "<< opti_pos_errors.size() <<"\n";
    std::cout<< "Position error mean: "<< opti_mean_pos_err <<" [m]\n\n";

    // Orientation
    Eigen::aligned_vector<double> opti_ori_errors = spline_gb.evaluateOrientationErrors(meas_r, meas_ts);
    double opti_mean_ori_err = 0.0;
    for (double err : opti_ori_errors) {opti_mean_ori_err += err;}
    opti_mean_ori_err /= static_cast<double>(opti_ori_errors.size());
    std::cout<< "Number of orientation errors: "<< opti_ori_errors.size() <<"\n";
    std::cout<< "Orientation error mean: "<< opti_mean_ori_err <<" [rad]\n\n";

    // Save final values
    std::cout<< "Saving spline ...\n\n";
    saveSplineKnots(trace_dir, spline_gb, knots_ts);
    saveSpline(trace_dir, spline_gb);

    return 0;
}

