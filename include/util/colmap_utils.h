/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <iostream>

#include <glog/logging.h>


bool loadFrameTimes(const std::string& filename, 
Eigen::aligned_vector<double>& timestamps)
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
        
        double stamp, tx, ty, tz, qx, qy, qz, qw;
        fs >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        timestamps.push_back(stamp);

        n++;
    }
    // @TODO: why does this while-loop read twice the last line?
    timestamps.pop_back();
    n--;
    LOG(INFO) << "Loaded " << n << " colmap frame time stamps.";
    return true;
}

