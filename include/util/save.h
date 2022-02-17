/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This code has been adapted from:
https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo_ros/src/benchmark_node.cpp
*/

#include <fstream>
#include <iostream>
#include <string>

#include <sophus/se3.hpp>

# include "util/eigen_utils.h"

namespace gvi_fusion {


void createTrace(const std::string& trace_dir, const std::string& trace_fn, 
std::ofstream* fs)
{
  const std::string full_fn = trace_dir + "/" + trace_fn;
  fs->open(full_fn.c_str());
  if (fs->fail())
  {
    std::cout << "Fail to create trace: " << full_fn << "\n";
    std::abort();
  }
  else
  {
    std::cout << "Created trace: " << full_fn << "\n";
  }
}


void traceTrajectorySophusSE3(const Eigen::aligned_vector<double>& ts, 
const Eigen::aligned_vector<Sophus::SE3d>& traj, std::ofstream& s)
{
    s << "# timestamp tx ty tz qx qy qz qw" << std::endl;

    for (size_t i = 0; i < ts.size(); i++)
    {
      double t = ts[i];
      Sophus::SE3d T = traj[i];

      const Eigen::Quaterniond q = T.unit_quaternion();
      const Eigen::Vector3d p = T.translation();

      s << std::setprecision(15) << t << " " << p.x() << " " << p.y() << " " << p.z() << " " 
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
}


void traceImuMeasurements(const Eigen::aligned_vector<ImuMeasurement> measurements, 
std::ofstream& s)
{
    s << "# timestamp wx wy wz ax ay az" << std::endl;

    for (size_t i = 0; i < measurements.size(); i++)
    {
      ImuMeasurement meas = measurements[i];

      double t = meas.t;
      const Eigen::Vector3d a = meas.lin_acc;
      const Eigen::Vector3d w = meas.ang_vel;

      s << std::setprecision(15) << t << " " << w[0] << " " << w[1] << " " << w[2] << " " 
      << a[0] << " " << a[1] << " " << a[2] << std::endl;
    }
}


template <class Spline>
void traceSplineKnots(const Spline& spline, const Eigen::aligned_vector<double>& ts, std::ofstream& s)
{
    s << "# timestamp tx ty tz qx qy qz qw" << std::endl;

    size_t n_knots = spline.numKnots();
    for (size_t i = 0; i < n_knots; i++)
    {
        Sophus::SE3d knot;
        knot = spline.getKnot(i);
        const Eigen::Quaterniond q = knot.unit_quaternion();
        const Eigen::Vector3d p = knot.translation();

      s << std::setprecision(15) << ts[i] << " " << p.x() << " " << p.y() << " " << p.z() << " " 
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
}


template <class Spline>
void traceSpline(const Spline& spline, int64_t dt_ns, std::ofstream& s)
{
  int64_t t_start_ns = spline.getStartTns();
  int64_t t_end_ns = spline.getEndTns();

  s << "# timestamp tx ty tz qx qy qz qw" << std::endl;
  for (int64_t t_ns = t_start_ns; t_ns < t_end_ns; t_ns+=dt_ns)
  {
      Sophus::SE3d pose;
      pose = spline.getPose(t_ns);
      const Eigen::Quaterniond q = pose.unit_quaternion();
      const Eigen::Vector3d p = pose.translation();
      double t_s = t_ns * spline.ns_to_s_;

      s << std::setprecision(15) << t_s << " " << p.x() << " " << p.y() << " " << p.z() << " " 
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
}


template <class Spline>
void traceAccBiasSpline(const Spline& spline, int64_t dt_ns, std::ofstream& s)
{
  int64_t t_start_ns = spline.getStartTns();
  int64_t t_end_ns = spline.getEndTnsBiasSpline();

  s << "# timestamp x y z" << std::endl;
  for (int64_t t_ns = t_start_ns; t_ns < t_end_ns; t_ns+=dt_ns)
  {
      Eigen::Vector3d bias;
      bias = spline.getAccelBias(t_ns);
      double t_s = t_ns * spline.ns_to_s_;

      s << std::setprecision(15) << t_s << " " << bias.x() << " " << bias.y() << " " << bias.z() << std::endl;
  }
}


template <class Spline>
void traceGyroBiasSpline(const Spline& spline, int64_t dt_ns, std::ofstream& s)
{
  int64_t t_start_ns = spline.getStartTns();
  int64_t t_end_ns = spline.getEndTnsBiasSpline();

  s << "# timestamp x y z" << std::endl;
  for (int64_t t_ns = t_start_ns; t_ns < t_end_ns; t_ns+=dt_ns)
  {
      Eigen::Vector3d bias;
      bias = spline.getGyroBias(t_ns);
      double t_s = t_ns * spline.ns_to_s_;

      s << std::setprecision(15) << t_s << " " << bias.x() << " " << bias.y() << " " << bias.z() << std::endl;
  }
}


void traceScalar(const double time, std::ofstream& s)
{
  s << time << std::endl;
}


void tracePose(const Sophus::SE3d& pose, std::ofstream& s)
{
  Eigen::Matrix3d R = pose.rotationMatrix();
  Eigen::Vector3d p = pose.translation();

  s << R(0,0) << " " << R(0,1) << " " << R(0,2) << " " << p.x() << "\n"
  << R(1,0) << " " << R(1,1) << " " << R(1,2) << " " << p.y() << "\n"
  << R(2,0) << " " << R(2,1) << " " << R(2,2) << " " << p.z() << "\n"
  << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << std::endl;
}


void tracePosition(const Eigen::Vector3d& pos, std::ofstream& s)
{
  s << "# x y z" << std::endl;
  s << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;
}

void traceOrientation(const Sophus::SO3d& ori, std::ofstream& s)
{
  Eigen::Matrix3d R = ori.matrix();

  s << R(0,0) << " " << R(0,1) << " " << R(0,2) << "\n"
  << R(1,0) << " " << R(1,1) << " " << R(1,2) << "\n"
  << R(2,0) << " " << R(2,1) << " " << R(2,2) << std::endl;
}


template <typename T>
void trace1DVector(const Eigen::aligned_vector<T>& vec, std::ofstream& s)
{
  for (size_t i = 0; i < vec.size(); i++)
  {
    s << vec[i] << std::endl;
  }
}


void tracePoints3D(const Eigen::aligned_unordered_map<uint64_t, Point3D>& points, std::ofstream& s)
{
  s << "# x y z" << std::endl;
  for (auto& it: points) 
  {
    Eigen::Vector3d pos = it.second.p_wl;
    s << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;
  }
}


void tracePoints3DArray(const Eigen::aligned_vector<Eigen::Vector3d>& points, std::ofstream& s)
{
  s << "# x y z" << std::endl;
  for (size_t i = 0; i < points.size(); i++)
  {
    s << points[i].x() << " " << points[i].y() << " " << points[i].z() << std::endl;
  }
}

void traceTimeAndPosition(const Eigen::aligned_vector<double>& ts, 
const Eigen::aligned_vector<Eigen::Vector3d>& pos, std::ofstream& s)
{
  s << "# ts x y z" << std::endl;
  for (size_t i = 0; i < pos.size(); i++)
  {
    s << ts[i] << " " << pos[i].x() << " " << pos[i].y() << " " << pos[i].z() << std::endl;
  }
}

void traceTimeAndPose(const Eigen::aligned_vector<double>& ts, 
const Eigen::aligned_vector<Eigen::Vector3d>& pos, 
Eigen::aligned_vector<Sophus::SO3d> ori, 
std::ofstream& s)
{
  s << "# ts x y z qx qy qz qw" << std::endl;
  for (size_t i = 0; i < pos.size(); i++)
  {
    Eigen::Quaterniond q = ori[i].unit_quaternion();
    s << ts[i] << " " << pos[i].x() << " " << pos[i].y() << " " << pos[i].z() << " " 
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
}

}  // namespace gvi_fusion

