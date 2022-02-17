/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <glog/logging.h>
#include <sophus/se3.hpp>

#include "sensor/imu_calib.h"
#include "util/eigen_utils.h"


// From: https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo_ceres_backend/include/svo/ceres_backend/imu_error.hpp#L65
// To make things a bit faster than using angle-axis conversion
__inline__ double sinc(double x)
{
  if (fabs(x) > 1e-6)
  {
   return sin(x) / x;
  }
  else
  {
    static const double c_2 = 1.0 / 6.0;
    static const double c_4 = 1.0 / 120.0;
    static const double c_6 = 1.0 / 5040.0;
    const double x_2 = x * x;
    const double x_4 = x_2 * x_2;
    const double x_6 = x_2 * x_2 * x_2;
    return 1.0 - c_2 * x_2 + c_4 * x_4 - c_6 * x_6;
  }
}


struct ImuMeasurement {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    double t;
    Eigen::Vector3d lin_acc;
    Eigen::Vector3d ang_vel;

    ImuMeasurement() {}
    ImuMeasurement(
      const double timestamp,
      const Eigen::Vector3d& linear_acceleration,
      const Eigen::Vector3d& angular_velocity)
      : t(timestamp), lin_acc(linear_acceleration), ang_vel(angular_velocity) {}
};


class Imu {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::shared_ptr<ImuCalibration> calib_ptr_;
    Eigen::aligned_vector<ImuMeasurement> measurements_;

    // Default constructor
    Imu() {};

    // Constructor
    Imu(std::shared_ptr<ImuCalibration> calib_ptr) : calib_ptr_(calib_ptr) {};

    // Destructor
    ~Imu() {};

    // Reverse measurements for easier sampling in disc. time fusion.
    void ReverseMeasurements()
    {
        std::reverse(measurements_.begin(), measurements_.end());
    }

    // Adapted from: https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo/src/imu_handler.cpp
    /// Get IMU measurements up to for the exact borders (using linear interpolation in imu error).
    // t_k: imu timestamp of frame k.
    // t_k_p_1: imu timestamp of frame k + 1.
    // if remove_measurements is true, it will remove the extracted measurements.
    bool getMeasurementsContainingEdges(const double t_k, const double t_k_p_1, 
    Eigen::aligned_vector<ImuMeasurement>& extracted_measurements,
    const bool remove_measurements)
    {
        if(measurements_.empty()) return false;
        if (measurements_.front().t > t_k)
        {
            LOG(WARNING) << "\n\nFirst imu meas newer than first frame. t_imu = " << measurements_.front().t << ", t_frame = " << t_k << "\n\n";
            return false;
        }
        if(measurements_.back().t < t_k_p_1)
        {
            LOG(WARNING) << "\n\nLast imu meas newer than last frame. t_imu = " << measurements_.back().t << ", t_frame = " << t_k_p_1 << "\n\n";
            return false;
        }

        // Find the first measurement older than t_k.
        Eigen::aligned_vector<ImuMeasurement>::iterator it_k = measurements_.begin();
        for(; it_k != measurements_.end(); ++it_k)
        {
            if(it_k->t > t_k)
            {
                //decrement iterator to point to 
                // first measurements older than t_k
                --it_k;
                break;
            }
        }

        // Find the first measurement newer than t_k_p_1.
        Eigen::aligned_vector<ImuMeasurement>::iterator it_k_p_1 = it_k;
        for(; it_k_p_1 != measurements_.end(); ++it_k_p_1)
        {
            if(it_k_p_1->t > t_k_p_1)
            {
                break;
            }
        }

        // extract measurements
        extracted_measurements.insert(extracted_measurements.begin(), it_k, it_k_p_1+1);

        // remove
        if(remove_measurements)
        {
            measurements_.erase(measurements_.begin(), it_k_p_1-1);
        }
        
        return true;
    }
};


class RandomMeasurements {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //  Default constructor
    RandomMeasurements() {};
    
    // Constructor
    RandomMeasurements(std::shared_ptr<Imu> imu_ptr, bool use_bias, bool straight_line = false)
    {
        imu_ptr_ = imu_ptr;
        use_bias_ = use_bias;
        acc_bias_.setZero();
        gyro_bias_.setZero();

        if(straight_line)
        {
            // straight line parameters
            w_omega_B_x_ = 5;
            w_omega_B_y_ = 5;
            w_omega_B_z_ = 5;
            // phase
            p_omega_B_x_ = M_PI / 2.;
            p_omega_B_y_ = M_PI / 2.;
            p_omega_B_z_ = M_PI / 2.;
            // magnitude
            m_omega_B_x_ = 0.;
            m_omega_B_y_ = 0.;
            m_omega_B_z_ = 0.;

            // Acc
            // circular frequency
            w_a_W_x_ = 0.;
            w_a_W_y_ = 0.;
            w_a_W_z_ = 0.;
            // phase
            p_a_W_x_ = M_PI / 2.;
            p_a_W_y_ = M_PI / 2.;
            p_a_W_z_ = M_PI / 2.;
            // magnitude
            m_a_W_x_ = 2.;
            m_a_W_y_ = 2.;
            m_a_W_z_ = 2.;
        }
        else
        {
            // random motion parameters
            // Gyro
            // circular frequency
            w_omega_B_x_ = Eigen::internal::random(0.1,10.0);
            w_omega_B_y_ = Eigen::internal::random(0.1,10.0);
            w_omega_B_z_ = Eigen::internal::random(0.1,10.0);
            // phase
            p_omega_B_x_ = Eigen::internal::random(0.0,M_PI);
            p_omega_B_y_ = Eigen::internal::random(0.0,M_PI);
            p_omega_B_z_ = Eigen::internal::random(0.0,M_PI);
            // magnitude
            m_omega_B_x_ = Eigen::internal::random(0.1,1.0);
            m_omega_B_y_ = Eigen::internal::random(0.1,1.0);
            m_omega_B_z_ = Eigen::internal::random(0.1,1.0);

            // Acc
            // circular frequency
            w_a_W_x_ = Eigen::internal::random(0.1,10.0);
            w_a_W_y_ = Eigen::internal::random(0.1,10.0);
            w_a_W_z_ = Eigen::internal::random(0.1,10.0);
            // phase
            p_a_W_x_ = Eigen::internal::random(0.1,M_PI);
            p_a_W_y_ = Eigen::internal::random(0.1,M_PI);
            p_a_W_z_ = Eigen::internal::random(0.1,M_PI);
            // magnitude
            m_a_W_x_ = Eigen::internal::random(0.1,1.0);
            m_a_W_y_ = Eigen::internal::random(0.1,1.0);
            m_a_W_z_ = Eigen::internal::random(0.1,1.0);
        }
    }

    // Destructor.
    ~RandomMeasurements() {};

    void generateRandomMeasurementsAndTrajectory(const double duration)
    {
        const double rate = imu_ptr_->calib_ptr_->rate_;
        const size_t num_measurements = static_cast<size_t>(duration * rate);
        const double dt = 1.0 / rate; // time discretization
        double t = 0.0; 

        // Initialize at identity and zero vel
        Sophus::SE3d T_wb;
        Eigen::Vector3d v_wb;
        v_wb << 0.0, 0.0, 0.0;

        poses_.emplace_back(T_wb);
        v_wb_i_.emplace_back(v_wb);
        ts_.emplace_back(t);

        // states
        Eigen::Quaterniond q = T_wb.unit_quaternion();
        Eigen::Vector3d r = T_wb.translation();

        for(size_t i = num_measurements-1; i<num_measurements; --i)
        {
            double time_s = double(num_measurements-i) / rate;

            // Angular velocity
            Eigen::Vector3d omega_B(m_omega_B_x_*sin(w_omega_B_x_*time_s+p_omega_B_x_),
                m_omega_B_y_*sin(w_omega_B_y_*time_s+p_omega_B_y_),
                m_omega_B_z_*sin(w_omega_B_z_*time_s+p_omega_B_z_));
            // Linear acceleration
            Eigen::Vector3d a_W(m_a_W_x_*sin(w_a_W_x_*time_s+p_a_W_x_),
                m_a_W_y_*sin(w_a_W_y_*time_s+p_a_W_y_),
                m_a_W_z_*sin(w_a_W_z_*time_s+p_a_W_z_));

            if(use_bias_)
            {
                omega_B += gyro_bias_;
            }

            Eigen::Quaterniond dq;
            // propagate orientation
            const double theta_half = omega_B.norm()*dt*0.5;
            const double sinc_theta_half = sinc(theta_half);
            const double cos_theta_half = cos(theta_half);
            dq.vec()=sinc_theta_half*0.5*dt*omega_B;
            dq.w()=cos_theta_half;
            q = q * dq;

            if(use_bias_)
            { 
                a_W += q.toRotationMatrix() * acc_bias_;
            }

            // propagate linear velocity
            v_wb += dt*a_W;
            v_wb_i_.emplace_back(v_wb);

            // propagate position
            r += dt*v_wb;
            
            // store propagated pose
            T_wb = Sophus::SE3d(q, r);
            poses_.emplace_back(T_wb);
            ts_.emplace_back(time_s);

            // generate acc meas
            Eigen::Vector3d a_B = T_wb.inverse().rotationMatrix() * 
            (a_W + imu_ptr_->calib_ptr_->g_W_);

            // store imu meas
            ImuMeasurement meas;
            meas.t = time_s;
            meas.lin_acc = a_B;
            meas.ang_vel = omega_B;
            imu_ptr_->measurements_.emplace_back(meas);

        }
    }

    std::shared_ptr<Imu> getImu() const { return imu_ptr_; }

    Eigen::aligned_vector<Sophus::SE3d> getPoses() const { return poses_; }

    Eigen::aligned_vector<Eigen::Vector3d> getVelocities() const { return v_wb_i_; }

    Eigen::aligned_vector<double> getTimes() const { return ts_; }

    void setAccBias(Eigen::Vector3d b) {acc_bias_ = b;}
    void setGyroBias(Eigen::Vector3d b) {gyro_bias_ = b;}

    private:
    // random motion parameters
    // Gyro
    // circular frequency
    double w_omega_B_x_;
    double w_omega_B_y_;
    double w_omega_B_z_;
    // phase
    double p_omega_B_x_;
    double p_omega_B_y_;
    double p_omega_B_z_;
    // magnitude
    double m_omega_B_x_;
    double m_omega_B_y_;
    double m_omega_B_z_;
    // Acc
    // circular frequency
    double w_a_W_x_;
    double w_a_W_y_;
    double w_a_W_z_;
    // phase
    double p_a_W_x_;
    double p_a_W_y_;
    double p_a_W_z_;
    // magnitude
    double m_a_W_x_;
    double m_a_W_y_;
    double m_a_W_z_;

    bool use_bias_;

    Eigen::Vector3d acc_bias_;
    Eigen::Vector3d gyro_bias_;

    // imu sensor
    std::shared_ptr<Imu> imu_ptr_;

    Eigen::aligned_vector<Sophus::SE3d> poses_;
    Eigen::aligned_vector<Eigen::Vector3d> v_wb_i_;
    Eigen::aligned_vector<double> ts_; // timestamps
};


/*
This code has been adapted from: 
https://github.com/uzh-rpg/rpg_svo_pro/blob/feature/global_map/svo/src/imu_handler.cpp
*/
bool loadImuMeasurementsFromFile(const std::string& filename, 
Eigen::aligned_vector<ImuMeasurement>& measurements)
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
        
        size_t imu_id;
        double stamp, wx, wy, wz, ax, ay, az;
        fs >> imu_id >> stamp >> wx >> wy >> wz >> ax >> ay >> az;

        const Eigen::Vector3d angvel(wx, wy, wz);
        const Eigen::Vector3d linacc(ax, ay, az);
        const ImuMeasurement m(stamp, linacc, angvel);
        measurements.emplace_back(m);

        n++;
    }
    // @TODO: why does this while-loop read twice the last line?
    measurements.pop_back();
    n--;
    LOG(INFO) << "Imu parser: Loaded " << n << " measurements.";
    return true;
}


void printVectorImuMeasurements(const Eigen::aligned_vector<ImuMeasurement>& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        std::cout << "[" << i << "]:\n"; 
        std::cout << "timestamp\n" << vec[i].t << "\n";
        std::cout << "lin acc\n" << vec[i].lin_acc << "\n";
        std::cout << "ang vel\n" << vec[i].ang_vel << "\n";
    }
}


void printImuMeasurement(const ImuMeasurement& meas)
{
    std::cout << "timestamp\n" << std::setprecision(9) << meas.t << "\n";
    std::cout << "lin acc\n" << meas.lin_acc << "\n";
    std::cout << "ang vel\n" << meas.ang_vel << "\n";
}

