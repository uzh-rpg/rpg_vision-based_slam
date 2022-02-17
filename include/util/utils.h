/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <algorithm>
#include <iostream>
#include <numeric>


// From: https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
std::vector<size_t> argSort(const std::vector<T>& v) 
{

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(), 
  [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


void fixRotationMatrix(const Eigen::Matrix3d& R, Eigen::Matrix3d& R_new)
{
    Eigen::BDCSVD<Eigen::Matrix3d> bdcsvd(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
    bdcsvd.computeU();
    bdcsvd.computeV();
    Eigen::Matrix3d U = bdcsvd.matrixU();
    Eigen::Matrix3d V = bdcsvd.matrixV();
    Eigen::Matrix3d Vt = V.transpose();

    Eigen::Matrix3d R_fix = U * Vt;
    if (R_fix.determinant() < 0)
    {
        R_fix = -1.0 * R_fix;
    }

    R_new = R_fix;
}

