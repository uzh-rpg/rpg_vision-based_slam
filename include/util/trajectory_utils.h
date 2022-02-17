/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <iostream>

#include <sophus/se3.hpp>
#include <sophus/interpolate.hpp>

#include "util/eigen_utils.h"


bool loadTrajectoryFromFile(const std::string& filename, 
Eigen::aligned_vector<Sophus::SO3d>& ori, 
Eigen::aligned_vector<Eigen::Vector3d>& pos,
std::vector<double>& ts)
{
    std::ifstream fs(filename.c_str());
    if(!fs.is_open())
    {
        LOG(WARNING) << "Could not open file: " << filename;
        return false;
    }

    // Read measurements
    size_t n = 0;
    while(fs.good() && !fs.eof())
    {
        if(fs.peek() == '#') // skip comments
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        
        double stamp, tx, ty, tz, qw, qx, qy, qz;
        fs >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        const Eigen::Vector3d p(tx, ty, tz);
        const Eigen::Quaterniond q(qw, qx, qy, qz);
        const Sophus::SO3d q_sophus(q);
        ts.emplace_back(stamp);
        ori.emplace_back(q_sophus);
        pos.emplace_back(p);

        n++;
    }
    // ToDo: why does the for loop above read twice the last line?
    ts.pop_back();
    ori.pop_back();
    pos.pop_back();

    LOG(INFO) << "Loaded " << n << " samples.";
    return true;
}


bool loadSE3TrajectoryFromFile(const std::string& filename, 
Eigen::aligned_vector<Sophus::SE3d>& poses,
Eigen::aligned_vector<double>& ts)
{
    std::ifstream fs(filename.c_str());
    if(!fs.is_open())
    {
        LOG(WARNING) << "Could not open file: " << filename;
        return false;
    }

    // Read measurements
    size_t n = 0;
    while(fs.good() && !fs.eof())
    {
        if(fs.peek() == '#') // skip comments
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        
        double stamp, tx, ty, tz, qw, qx, qy, qz;
        fs >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        const Eigen::Vector3d p(tx, ty, tz);
        const Eigen::Quaterniond q(qw, qx, qy, qz);
        const Sophus::SE3d pose(q, p);
        ts.emplace_back(stamp);
        poses.emplace_back(pose);

        n++;
    }
    // ToDo: why does the for loop above read twice the last line?
    ts.pop_back();
    poses.pop_back();

    LOG(INFO) << "Loaded " << n << " samples.";
    return true;
 }


void linearInterpolation(const Eigen::aligned_vector<Eigen::Vector3d>& pos,
const Eigen::aligned_vector<Sophus::SO3d>& ori,
const Eigen::aligned_vector<double>& ts, 
const Eigen::aligned_vector<double>& interpolated_ts,
Eigen::aligned_vector<Eigen::Vector3d>& interpolated_pos, 
Eigen::aligned_vector<Sophus::SO3d>& interpolated_ori)
{
    size_t it_i = 0;
    size_t t_i = 0;
    size_t n = interpolated_ts.size();
    size_t m = ts.size() - 1;

    while (true)
    {
        if( (interpolated_ts[it_i] >= ts[t_i]) && (interpolated_ts[it_i] < ts[t_i + 1]) )
        {
            Eigen::Vector3d p1 = pos[t_i];
            Sophus::SO3d r1 = ori[t_i];

            Eigen::Vector3d p2 = pos[t_i + 1];
            Sophus::SO3d r2 = ori[t_i + 1];

            double dt = (interpolated_ts[it_i] - ts[t_i]) / (ts[t_i + 1] - ts[t_i]);

            Eigen::Vector3d p = p1 + dt * (p2 - p1);
            Sophus::SO3d r;
            r = Sophus::interpolate(r1, r2, dt);

            interpolated_pos.push_back(p);
            interpolated_ori.push_back(r);

            it_i += 1;

            if (it_i == n) { break; };
        }
        else
        {
            t_i += 1;
            if (t_i == m) { break; }
        }
        
    }
    GVI_FUSION_ASSERT_STREAM(interpolated_pos.size() == interpolated_ori.size(), 
    "interpolated_pos.size() != interpolated_ori.size()");
    GVI_FUSION_ASSERT_STREAM(interpolated_pos.size() == n, 
    "interpolated_pos.size() (= " << interpolated_pos.size() << ")" 
    << " != interpolated_ts.size() (= " << n << ")");
}

