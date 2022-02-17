/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <iostream>
#include <memory>


class ImuCalibration {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    double rate_; //Hz
    double acc_noise_density_;
    double acc_random_walk_;
    double gyro_noise_density_;
    double gyro_random_walk_;
    double a_max_ = 150;  // Accelerometer saturation. [m/s^2]
    double g_max_ = 7.8;  // Gyroscope saturation. [rad/s]
    Eigen::Vector3d g_W_;

    // Default constructor
    ImuCalibration() {};

    // Constructor
    ImuCalibration(double rate, double acc_noise_density, double acc_random_walk, 
    double gyro_noise_density, double gyro_random_walk) :
    rate_(rate), acc_noise_density_(acc_noise_density), acc_random_walk_(acc_random_walk), 
    gyro_noise_density_(gyro_noise_density), gyro_random_walk_(gyro_random_walk)
    {
        g_W_ << 0.0, 0.0, g_;
    };

    // Destructor
    ~ImuCalibration() {};

    private:
    const double g_ = 9.80665;
};

