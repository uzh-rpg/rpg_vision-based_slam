/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <iostream>
#include <math.h> 
#include <memory>

#include "util/assert.h"


// This code is based on: https://github.com/uzh-rpg/uzh_fpv/blob/full-batch-optimization/python/uzh_fpv/util_colmap.py#L100
size_t numberParameters(const int model_id)
{
    switch (model_id) 
    {
        // simple pinhole
        case 0: return 3; break;
        //pinhole
        case 1: return 4; break;
        //simple radial
        case 2: return 4; break;
        //radial
        case 3: return 5; break;
        //opencv (= pinhole radial tangential)
        case 4: return 8; break;
        //opencv fisheye
        case 5: return 4; break;
        // full opencv
        case 6: return 12; break;
        // fov
        case 7: return 5; break;
        // simple radial fisheye
        case 8: return 4; break;
        // radial fisheye
        case 9: return 5; break;
        // thin prism fisheye
        case 10: return 12; break;
        default:
        GVI_FUSION_ASSERT_STREAM(false, "Unknown Camera model!");
        return 0;
    }
}


// This code is based on:
// https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cameras/include/aslam/cameras/implementation/RadialTangentialDistortion.hpp#L5
template <typename T>
void distortRadTan(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p)
{
    T mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
    
    mx2_u = p[0] * p[0];
    my2_u = p[1] * p[1];
    mxy_u = p[0] * p[1];
    rho2_u = mx2_u + my2_u;
    rad_dist_u = dist_coeff[0] * rho2_u + dist_coeff[1] * rho2_u * rho2_u;

    p[0] += p[0] * rad_dist_u + 2.0 * dist_coeff[2] * mxy_u + dist_coeff[3] * (rho2_u + 2.0 * mx2_u);
    p[1] += p[1] * rad_dist_u + 2.0 * dist_coeff[3] * mxy_u + dist_coeff[2] * (rho2_u + 2.0 * my2_u);
}


// This code is based on:
// https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cameras/include/aslam/cameras/implementation/RadialTangentialDistortion.hpp#L28
template <typename T>
void distortRadTanWithJac(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p, Eigen::Matrix<T, 2, 2>& J)
{
    T mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
    J.setZero();
    
    mx2_u = p[0] * p[0];
    my2_u = p[1] * p[1];
    mxy_u = p[0] * p[1];
    rho2_u = mx2_u + my2_u;
    rad_dist_u = dist_coeff[0] * rho2_u + dist_coeff[1] * rho2_u * rho2_u;

    J(0, 0) = 1 + rad_dist_u + dist_coeff[0] * 2.0 * mx2_u + 
    dist_coeff[1] * rho2_u * 4 * mx2_u + 2.0 * dist_coeff[2] * p[1] + 6 * dist_coeff[3] * p[0];
    J(1, 0) = dist_coeff[0] * 2.0 * p[0] * p[1] + dist_coeff[1] * 4 * rho2_u * p[0] * p[1] + 
    dist_coeff[2] * 2.0 * p[0] + 2.0 * dist_coeff[3] * p[1];
    J(0, 1) = J(1, 0);
    J(1, 1) = 1 + rad_dist_u + dist_coeff[0] * 2.0 * my2_u + dist_coeff[1] * rho2_u * 4 * my2_u + 
    6 * dist_coeff[2] * p[1] + 2.0 * dist_coeff[3] * p[0];

    p[0] += p[0] * rad_dist_u + 2.0 * dist_coeff[2] * mxy_u + dist_coeff[3] * (rho2_u + 2.0 * mx2_u);
    p[1] += p[1] * rad_dist_u + 2.0 * dist_coeff[3] * mxy_u + dist_coeff[2] * (rho2_u + 2.0 * my2_u);
}


// This code is based on:
// https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cameras/include/aslam/cameras/implementation/RadialTangentialDistortion.hpp#L68
template <typename T>
void undistortRadTan(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p)
{
    using Vector2 = Eigen::Matrix<T, 2, 1>;
    using Matrix2 = Eigen::Matrix<T, 2, 2>;

    Vector2 pbar = p;
    const int n = 5;
    Matrix2 F;

    Vector2 p_tmp;

    for (int i = 0; i < n; i++) 
    {
        p_tmp = pbar;

        distortRadTanWithJac(dist_coeff, p_tmp, F);

        Vector2 e(p - p_tmp);
        Vector2 du = (F.transpose() * F).inverse() * F.transpose() * e;

        pbar += du;

        if (e.dot(e) < 1e-15)
        break;
    }

    p[0] = pbar[0];
    p[1] = pbar[1];
}


// This code is based on:
// https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cameras/include/aslam/cameras/implementation/EquidistantDistortion.hpp#L5
template <typename T>
void distortEquidistant(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p)
{
    T r, theta, theta2, theta4, theta6, theta8, thetad, scaling;

    r = sqrt(p[0] * p[0] + p[1] * p[1]);
    theta = atan(r);
    theta2 = theta * theta;
    theta4 = theta2 * theta2;
    theta6 = theta4 * theta2;
    theta8 = theta4 * theta4;
    thetad = theta * 
    (1.0 +  dist_coeff[0] * theta2 + dist_coeff[1] * theta4 + dist_coeff[2] * theta6 + dist_coeff[3] * theta8);

    //scaling = (r > 1e-8) ? thetad / r : 1.0;
    scaling = thetad / r;
    p[0] *= scaling;
    p[1] *= scaling;
}


// This code is based on:
//https://github.com/ethz-asl/kalibr/blob/master/aslam_cv/aslam_cameras/include/aslam/cameras/implementation/EquidistantDistortion.hpp#L186
template <typename T>
void undistortEquidistant(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p) 
{
    T theta, theta2, theta4, theta6, theta8, thetad, scaling;

    thetad = sqrt(p[0] * p[0] + p[1] * p[1]);
    theta = thetad;  // initial guess
    for (int i = 20; i > 0; i--) 
    {
        theta2 = theta * theta;
        theta4 = theta2 * theta2;
        theta6 = theta4 * theta2;
        theta8 = theta4 * theta4;
        theta = thetad
            / (1.0 + dist_coeff[0] * theta2 + dist_coeff[1] * theta4 + dist_coeff[2] * theta6 + dist_coeff[3] * theta8);
    }
    scaling = tan(theta) / thetad;

    p[0] *= scaling;
    p[1] *= scaling;
}


template <typename T>
void distortFov(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p)
{
    T s = T(dist_coeff[0]);
    T tan_s_half_x2 = 2.0 * tan(s / 2.0);
    T rad = sqrt(p[0] * p[0] + p[1] * p[1]);
    T factor = (rad < 0.001) ? T(1.0) : atan(rad * tan_s_half_x2) / (s * rad);
    p[0] *= factor;
    p[1] *= factor;
}


template <typename T>
void undistortFov(const std::vector<double>& dist_coeff, Eigen::Matrix<T, 2, 1>& p)
{
    T s = T(dist_coeff[0]);
    T tan_s_half_x2 = 2.0 * tan(s / 2.0);
    T rad = sqrt(p[0] * p[0] + p[1] * p[1]);
    T factor = (rad < 0.001) ? T(1.0) : (tan(rad * s) / tan_s_half_x2) / rad;
    p[0] *= factor;
    p[1] *= factor;
}


class DistortionModel
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int dist_model_ = -1;

    // Constructor
    DistortionModel() {};

    // Destructor
    ~DistortionModel() {};

    void setDistortionModel(const int dist_model) {dist_model_ = dist_model;}

    template <typename T>
    void distort(const std::vector<double>& dist_coeffs, Eigen::Matrix<T, 2, 1>& p) 
    {
        switch (dist_model_)
        {
            case -1:
            GVI_FUSION_ASSERT_STREAM(false, "Distortion model not set!"); 
            break;
            
            // opencv full camera model (= pinhole, rad-tan)
            case 4: 
            GVI_FUSION_ASSERT_STREAM(dist_coeffs.size() > 0, "Distortion coeffs not set!");
            distortRadTan(dist_coeffs, p); 
            break;

            //opencv fisheye
            case 5: 
            GVI_FUSION_ASSERT_STREAM(dist_coeffs.size() > 0, "Distortion coeffs not set!");
            distortEquidistant(dist_coeffs, p); 
            break;

            //fov
            case 7: 
            GVI_FUSION_ASSERT_STREAM(dist_coeffs.size() > 0, "Distortion coeffs not set!");
            distortFov(dist_coeffs, p); 
            break;
            
            default:
            GVI_FUSION_ASSERT_STREAM(false, "Camera model id " << dist_model_ << " not supported!");
        }
    };

    template <typename T>
    void undistort(const std::vector<double>& dist_coeffs, Eigen::Matrix<T, 2, 1>& p) 
    {
        switch (dist_model_)
        {
            // opencv full camera model (= pinhole, rad-tan)
            case 4: 
            GVI_FUSION_ASSERT_STREAM(dist_coeffs.size() > 0, "Distortion coeffs not set!");
            undistortRadTan(dist_coeffs, p); 
            break;

            //opencv fisheye
            case 5: 
            GVI_FUSION_ASSERT_STREAM(dist_coeffs.size() > 0, "Distortion coeffs not set!");
            undistortEquidistant(dist_coeffs, p); 
            break;

            //fov
            case 7: 
            GVI_FUSION_ASSERT_STREAM(dist_coeffs.size() > 0, "Distortion coeffs not set!");
            undistortFov(dist_coeffs, p); 
            break;
            
            default:
            GVI_FUSION_ASSERT_STREAM(false, "Camera model id " << dist_model_ << " not supported!");
        }
    };
};

