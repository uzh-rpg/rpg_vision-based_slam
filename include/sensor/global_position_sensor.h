/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <glog/logging.h>

#include "util/eigen_utils.h"


struct GlobalPosMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    double t;
    Eigen::Vector3d pos;
    double noise;
    // Id of the closest frame (= k) in time such that 
    // t_k < t_gp and t_k+1 > t_gp
    int closest_frame_id;
    // used only in discrete time estimator
    bool is_valid = false;

    GlobalPosMeasurement() {}
    GlobalPosMeasurement(const double timestamp, const Eigen::Vector3d& xyz) : 
    t(timestamp), pos(xyz) 
    {
        noise = 1e-2;
        closest_frame_id = -1;
    }

    void setClosestFrameId(int id) { closest_frame_id = id; }
};


bool loadGpMeasurements(const std::string& filename, 
Eigen::aligned_vector<GlobalPosMeasurement>& measurements)
{
    std::ifstream fs(filename.c_str());
    if(!fs.is_open())
    {
        LOG(WARNING) << "Could not open imu file: " << filename;
        return false;
    }

    // Read measurements
    size_t n = 0;
    while(fs.good() && !fs.eof())
    {
        if(fs.peek() == '#') // skip comments
        fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        
        double stamp, x, y, z;
        fs >> stamp >> x >> y >> z;

        Eigen::Vector3d xyz(x, y, z);
        const GlobalPosMeasurement meas(stamp, xyz);
        measurements.emplace_back(meas);

        n++;
    }
    // @TODO: why does this while-loop read twice the last line?
    measurements.pop_back();
    n--;
    LOG(INFO) << "Leica parser: Loaded " << n << " measurements.";
    return true;
}


// Return the id of the closest frame (= k) in time such that 
// t_k < t_gp and t_k+1 > t_gp
// [in] state_ts: timestamps of states
// [in] glob. pos. meas. timestamp
// [out] states_id to be found
int getLeftClosestStateId(Eigen::aligned_vector<double>& state_ts, double t_gp)
{
    int state_id = -1;
    for (size_t i = 0; i < state_ts.size(); i++)
    {
        if (state_ts[i] > t_gp)
        {
            state_id = i - 1;
            break;
        }
    }

    return state_id;
}


void printVectorLeicaMeasurements(const Eigen::aligned_vector<GlobalPosMeasurement>& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        std::cout << "[" << i << "]:\n"; 
        std::cout << "timestamp\n" << vec[i].t << "\n";
        std::cout << "pos\n" << vec[i].pos << "\n";
    }
}


void printLeicaMeasurement(const GlobalPosMeasurement& meas)
{
    std::cout << "timestamp\n" << std::setprecision(19) << meas.t << "\n";
    std::cout << "pos\n" << meas.pos << "\n";
}

