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

#include <util/utils.h>


bool loadPose(const std::string& filename, Sophus::SE3d& pose)
{
    std::ifstream fs(filename.c_str());
    if(!fs.is_open())
    {
        LOG(ERROR) << "Could not open file: " << filename;
        return false;
    }

    // Read measurements
    double r11, r12, r13, r21, r22, r23, r31, r32, r33;
    double t1, t2, t3;

    if(fs.peek() == '#') // skip comments
    {
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    fs >> std::setprecision(9) >> r11 >> r12 >> r13 >> t1 >> r21 >> r22 >> r23 >> t2 >> r31 >> r32 >> r33 >> t3;
    
    Eigen::Vector3d p(t1, t2, t3);
    Eigen::Matrix3d R;
    R << r11, r12, r13, r21, r22, r23, r31, r32, r33;

    Eigen::Matrix3d R_new;
    fixRotationMatrix(R, R_new);
    
    Sophus::SE3d T(R_new, p);

    pose = T;
    return true;
}


bool loadVector3d(const std::string& filename, Eigen::Vector3d& v)
{
    std::ifstream fs(filename.c_str());
    if(!fs.is_open())
    {
        LOG(ERROR) << "Could not open file: " << filename;
        return false;
    }

    // Read measurements
    double t1, t2, t3;

    if(fs.peek() == '#') // skip comments
    {
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    fs >> t1 >> t2 >> t3;
    
    Eigen::Vector3d t(t1, t2, t3);
    v = t;

    return true;
}


template <typename T>
bool loadScalar(const std::string& filename, T& scalar)
{
    std::ifstream fs(filename.c_str());
    if(!fs.is_open())
    {
        LOG(ERROR) << "Could not open file: " << filename;
        return false;
    }

    // Read measurements
    T s;

    if(fs.peek() == '#') // skip comments
    {
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    fs >> s;

    scalar = s;

    return true;
}

