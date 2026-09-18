#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct CustomPointType {
    float x = 0.f, y = 0.f, z = 0.f;
    float intensity = 0.f;
    double timestamp = 0.0;  // timestamp denotes (point_time - pointcloud_begin_time)

    Eigen::Vector3f getVector3fMap() const { return Eigen::Vector3f(x, y, z); }
    Eigen::Vector4f getVector4fMap() const { return Eigen::Vector4f(x, y, z, 1.0f); }
};

namespace se3_lio {

using PointCloudType = std::vector<CustomPointType>;
struct Header {
    uint32_t seq;
    double timestamp;
    std::string frame_id;
};

struct IMU {
    Header header;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
};

struct LiDAR {
    Header header;
    PointCloudType points;
    std::vector<Eigen::Matrix3d> noises;
};

struct GPS {
    Header header;
    Eigen::Matrix4d pose;
    Eigen::Matrix<double, 6, 6> covariance;
};

enum StateIdx {
    POSE = 0,
    POS = 0,
    ROT = 3,
    VEL = 6,
    B_G = 9,
    B_A = 12,
    GRAV = 15,
};

// Noise Cov Matrix Index
enum NoiseIdx {
    N_GYRO = 0,
    N_ACC = 3,
    N_B_GYRO = 6,
    N_B_ACC = 9,
};

#define DOF_DIM 17
#define ERR_DIM 17
#define INPUT_DIM 12
using ErrVector = Eigen::Matrix<double, ERR_DIM, 1>;
using CovMatrix = Eigen::Matrix<double, ERR_DIM, ERR_DIM>;
using MeasCovMatrix = Eigen::Matrix<double, ERR_DIM, INPUT_DIM>;

struct Measurement {
    bool is_synced;

    std::vector<IMU> imu;
    LiDAR lidar;
    std::vector<LiDAR> lidars;  // per-sensor scans (sensor frame); merged into `lidar` by the pipeline
    double end_time = 0.0;      // absolute epoch end (e.g. image stamp); 0 -> last point time
    LiDAR raw_lidar;     // raw lidar points
    LiDAR merged_lidar;  // merged lidar points on IMU
    GPS gps;
};

struct State {
    double stamp;
    int num_inliers;
    double residual;
    Eigen::Matrix4d pose;
    Eigen::Vector3d vel;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Eigen::Vector3d grav;

    CovMatrix covariance;

    State() {
        stamp = 0.0;
        pose = Eigen::Matrix4d::Identity();
        vel = Eigen::Vector3d::Zero();
        bg = Eigen::Vector3d::Zero();
        ba = Eigen::Vector3d::Zero();
        grav = Eigen::Vector3d::Zero();
        covariance = CovMatrix::Zero();
    }

    inline Eigen::Matrix3d rot() const { return pose.block<3, 3>(0, 0); }
    inline Eigen::Vector3d pos() const { return pose.block<3, 1>(0, 3); }
    inline Eigen::Matrix<double, 6, 6> pose_cov() const { return covariance.block<6, 6>(0, 0); }
};

using MeasurementPtr = std::shared_ptr<Measurement>;
using StatePtr = std::shared_ptr<State>;

}  // namespace se3_lio

#endif  // DATA_TYPE_H