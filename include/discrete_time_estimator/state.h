/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#pragma once


class State
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sophus::SE3d T_WB_;
    double ts_cam_;
    double ts_imu_;
    Eigen::Matrix<double, 9, 1> speed_and_bias_;

    State(){};
    State(Sophus::SE3d T_WB, double ts_cam, double ts_imu): 
    T_WB_(T_WB), ts_cam_(ts_cam), ts_imu_(ts_imu)
    {
        speed_and_bias_.setZero();
    };

    ~State(){};

    void setPose(Sophus::SE3d T){T_WB_ = T;}
    void setCamTs(double t){ts_cam_ = t;}
    void setImuTs(double t){ts_imu_ = t;}
    void setVel(Eigen::Vector3d v){speed_and_bias_.head<3>() = v;}
    void setAccBias(Eigen::Vector3d v){speed_and_bias_.tail<3>() = v;}
    void setGyroBias(Eigen::Vector3d v){speed_and_bias_.segment<3>(3) = v;}

};

