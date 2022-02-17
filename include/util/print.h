/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <iostream>

#include <sophus/se3.hpp>

#include <util/eigen_utils.h>

namespace gvi_fusion {

template <class Spline>
void printSplineKnots(const Spline& spline)
{
    size_t n_knots = spline.numKnots();
    for (size_t i = 0; i < n_knots; i++)
    {
        Sophus::SE3d knot;
        knot = spline.getKnot(i);
        std::cout << "-- Knot " << i << "\n";
        std::cout << "rotation:\n" << knot.rotationMatrix() << "\n";
        std::cout << "translation:\n" << knot.translation() << "\n";
    }
}


void printEigenQuaternion(const Eigen::Quaterniond& q)
{
    std::cout << "q.x: " << q.x() << ", q.y: " << q.y() 
    << ", q.z: " << q.z() << ", q.w: " << q.w() << "\n";
}


void printSophusPose(const Sophus::SE3d& T)
{
    std::cout << "rotation:\n" << T.rotationMatrix() << "\n";
    std::cout << "translation:\n" << T.translation() << "\n";
}


template <typename T>
void printVector(const Eigen::aligned_vector<T>& vec, size_t di = 1)
{
    for (size_t i = 0; i < vec.size(); i+=di)
    {
        std::cout << "[" << i << "]: " << vec[i] << "\n"; 
    }
}


void printVectorSophusSE3(const Eigen::aligned_vector<Sophus::SE3d>& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        std::cout << "[" << i << "]:\n"; 
        std::cout << "rotation\n" << vec[i].rotationMatrix() << "\n";
        std::cout << "translation\n" << vec[i].translation() << "\n";
    }
}

}  // namespace gvi_fusion

