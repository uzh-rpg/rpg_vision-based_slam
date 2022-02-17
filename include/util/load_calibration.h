/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <fstream>
#include <iostream>
#include <string>

#include <sophus/se3.hpp>
#include <yaml-cpp/yaml.h>


// Contains a simplified version of the cam-imu calibration quantities 
// obtained from Kalibr
struct SimpleKalibrFormatCamImuCalib
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::string camera_model_;
    std::string distortion_model_;
    // @ToDO: this only supports 4 dist. coeffs. atm.
    Eigen::Vector4d distortion_coeffs_;
    Eigen::Vector2d resolution_; // [w, h]
    Eigen::Vector4d intrinsics_; // [fx, fy, cx, cy]
    
    double timeshift_cam_imu_;
    Sophus::SE3d T_cam_imu_;

    SimpleKalibrFormatCamImuCalib(){};
    ~SimpleKalibrFormatCamImuCalib(){};
};


// Contains a simplified version of the imu .yaml config file 
// required to run Kalibr.
struct SimpleKalibrFormatImuCalib
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double rate_; //Hz
    double acc_noise_density_;
    double acc_random_walk_;
    double gyro_noise_density_;
    double gyro_random_walk_;

    SimpleKalibrFormatImuCalib(){};
    ~SimpleKalibrFormatImuCalib(){};
};


struct SimpleCamImuCalib
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // imu related stuff
    double imu_rate_; //Hz
    double acc_noise_density_;
    double acc_random_walk_;
    double gyro_noise_density_;
    double gyro_random_walk_;

    // camera related stuff
    int idx_;
    std::string camera_model_;
    std::string distortion_model_;
    Eigen::VectorXd distortion_coeffs_;
    Eigen::Vector2d resolution_; // [w, h]
    Eigen::Vector4d intrinsics_; // [fx, fy, cx, cy]

    // cam-imu related stuff
    double timeshift_cam_imu_;
    Sophus::SE3d T_cam_imu_;

    SimpleCamImuCalib(){};
    ~SimpleCamImuCalib(){};
};


bool loadSimpleKalibrFormatCamImuCalib(const std::string& calib_fn, const int cam_id, SimpleKalibrFormatCamImuCalib& calib)
{
    YAML::Node config = YAML::LoadFile(calib_fn);

    YAML::Node config_cam;
    if (cam_id == 0)
    {
        config_cam = config["cam0"];
    }
    else if (cam_id == 1)
    {
        config_cam = config["cam1"];
    }
    else
    {
        return false;
    }

    calib.camera_model_ = config_cam["camera_model"].as<std::string>();
    calib.distortion_model_ = config_cam["distortion_model"].as<std::string>();
    Eigen::Vector4d distortion_coeffs(config_cam["distortion_coeffs"].as<std::vector<double>>().data());
    calib.distortion_coeffs_ = distortion_coeffs;
    Eigen::Vector2d resolution(config_cam["resolution"].as<std::vector<double>>().data());
    calib.resolution_ = resolution;
    Eigen::Vector4d intrinsics(config_cam["intrinsics"].as<std::vector<double>>().data());
    calib.intrinsics_ = intrinsics;
    calib.timeshift_cam_imu_ = config_cam["timeshift_cam_imu"].as<double>();
    Eigen::Matrix4d T_cam_imu_transp_eigen(config_cam["T_cam_imu"].as<std::vector<double>>().data());
    Sophus::SE3d T_cam_imu(T_cam_imu_transp_eigen.transpose());
    calib.T_cam_imu_ = T_cam_imu;

    return true;
}


bool loadSimpleKalibrFormatImuCalib(const std::string& calib_fn, SimpleKalibrFormatImuCalib& calib)
{
    YAML::Node config = YAML::LoadFile(calib_fn);

    calib.rate_ = config["update_rate"].as<double>();
    calib.acc_noise_density_ = config["accelerometer_noise_density"].as<double>();
    calib.acc_random_walk_ = config["accelerometer_random_walk"].as<double>();
    calib.gyro_noise_density_ = config["gyroscope_noise_density"].as<double>();
    calib.gyro_random_walk_ = config["gyroscope_random_walk"].as<double>();

    return true;
}


bool loadUZHFPVSimpleCamImuCalib(
    const std::string& camimu_calib_fn, 
    const std::string& imu_calib_fn, 
    const int cam_id, 
    SimpleCamImuCalib& calib)
{
    SimpleKalibrFormatCamImuCalib camimu_calib;
    SimpleKalibrFormatImuCalib imu_calib;
    if ( loadSimpleKalibrFormatCamImuCalib(camimu_calib_fn, cam_id, camimu_calib) 
    && loadSimpleKalibrFormatImuCalib(imu_calib_fn, imu_calib) )
    {
        calib.idx_ = cam_id;
        calib.camera_model_ = camimu_calib.camera_model_;
        calib.distortion_model_ = camimu_calib.distortion_model_;
        calib.distortion_coeffs_ = camimu_calib.distortion_coeffs_;
        calib.resolution_ = camimu_calib.resolution_;
        calib.intrinsics_ = camimu_calib.intrinsics_;
        calib.timeshift_cam_imu_ = camimu_calib.timeshift_cam_imu_;
        calib.T_cam_imu_ = camimu_calib.T_cam_imu_;
        calib.imu_rate_ = imu_calib.rate_;
        calib.acc_noise_density_ = imu_calib.acc_noise_density_;
        calib.acc_random_walk_ = imu_calib.acc_random_walk_;
        calib.gyro_noise_density_ = imu_calib.gyro_noise_density_;
        calib.gyro_random_walk_ = imu_calib.gyro_random_walk_;

        return true;
    }
    else
    {
        return false;
    }
}


bool loadEuRoCSimpleCamImuCalib(
    const std::string& calib_fn, const int cam_id, SimpleCamImuCalib& calib)
{
    YAML::Node config = YAML::LoadFile(calib_fn);

    YAML::Node cameras_node = config["cameras"];
    YAML::Node cam_calib_node = (cameras_node[cam_id])["camera"];
    calib.idx_ = cam_id;
    calib.camera_model_ = cam_calib_node["type"].as<std::string>();
    double height = cam_calib_node["image_height"].as<double>();
    double width = cam_calib_node["image_width"].as<double>();
    Eigen::Vector2d resolution;
    resolution << width, height;
    calib.resolution_ = resolution;
    YAML::Node dist_node = cam_calib_node["distortion"];
    calib.distortion_model_ = dist_node["type"].as<std::string>();
    YAML::Node dist_params_node = dist_node["parameters"];
    Eigen::Vector4d distortion_coeffs(dist_params_node["data"].as<std::vector<double>>().data());
    calib.distortion_coeffs_ = distortion_coeffs;
    YAML::Node intrinsics_node = cam_calib_node["intrinsics"];
    Eigen::Vector4d intrinsics(intrinsics_node["data"].as<std::vector<double>>().data());
    calib.intrinsics_ = intrinsics;
    YAML::Node extrinsics_node = (cameras_node[cam_id])["T_B_C"];
    Eigen::Matrix4d T_imu_cam_transp_eigen(extrinsics_node["data"].as<std::vector<double>>().data());
    Sophus::SE3d T_imu_cam(T_imu_cam_transp_eigen.transpose());
    calib.T_cam_imu_ = T_imu_cam.inverse();
    YAML::Node imu_node = config["imu_params"];
    calib.timeshift_cam_imu_ = -1.0 * imu_node["delay_imu_cam"].as<double>();
    calib.imu_rate_ = imu_node["imu_rate"].as<double>();
    calib.acc_noise_density_ = imu_node["sigma_acc_c"].as<double>();
    calib.acc_random_walk_ = imu_node["sigma_acc_bias_c"].as<double>();
    calib.gyro_noise_density_ = imu_node["sigma_omega_c"].as<double>();
    calib.gyro_random_walk_ = imu_node["sigma_omega_bias_c"].as<double>();
    
    return true;
}


void printSimpleCamImuCalib(SimpleCamImuCalib& calib)
{
    std::cout << "\n===== Cam-Imu calibration =====\n";
    std::cout << "Loaded camera\n";
    std::cout << "- idx: " << calib.idx_ << "\n";
    std::cout << "- model: " << calib.camera_model_ << "\n";
    std::cout << "- distortion model: " << calib.distortion_model_ << "\n";
    std::cout << "- distortion coeff: \n" << calib.distortion_coeffs_ << "\n";
    std::cout << "- resolution: \n" << calib.resolution_ << "\n";
    std::cout << "- intrinsics: \n" << calib.intrinsics_ << "\n";
    std::cout << "- timeshift_cam_imu: " << calib.timeshift_cam_imu_ << "\n";
    std::cout << "- T_cam_imu: \n" << calib.T_cam_imu_.matrix3x4() << "\n";
    std::cout << "Loaded imu\n";
    std::cout << "- rate: " << calib.imu_rate_ << "\n";
    std::cout << "- acc_noise_density: " << calib.acc_noise_density_ << "\n";
    std::cout << "- acc_random_walk: " << calib.acc_random_walk_ << "\n";
    std::cout << "- gyro_noise_density: " << calib.gyro_noise_density_ << "\n";
    std::cout << "- gyro_random_walk: " << calib.gyro_random_walk_;
    std::cout << "\n==============================\n\n";
}

