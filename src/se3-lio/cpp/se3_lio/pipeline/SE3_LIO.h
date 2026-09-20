#pragma once

// C++ standard libraries
#include <Eigen/Core>

#include <memory>

// Common
#include "common/data_type.h"
#include "common/utils.h"

// Core
#include "core/map_management.h"
#include "core/state_predict.h"
#include "core/state_update.h"

// Visual
#include <opencv2/core.hpp>

#include "visual/camera.h"
#include "visual/visual_update.h"

namespace se3_lio::pipeline {

struct CameraConfig {
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> dist;   // radtan k1 k2 p1 p2 [k3], or equidistant k1..k4 when fisheye
    bool fisheye = false;
    Eigen::Matrix4d T_cam_imu = Eigen::Matrix4d::Identity();  // camera <- body(IMU)
};

struct SE3_LIO_Config {
    double acc_noise = 0.1;
    double gyr_noise = 0.1;
    double bg_noise = 0.0001;
    double ba_noise = 0.0001;

    double lidar_range_noise = 0.02;
    double lidar_angle_noise = 0.15;
    // Per-LiDAR noise, indexed by point.intensity (multi-LiDAR merged scans).
    // Empty -> the scalars above apply to every point.
    std::vector<double> lidar_range_noises;
    std::vector<double> lidar_angle_noises;
    // Sensor -> body extrinsic per LiDAR, used only for Measurement::lidars (multi-LiDAR merge).
    std::vector<Eigen::Matrix4d> lidar_extrinsics;
    // Per-LiDAR near cut [m] in the sensor frame, applied in mergeLiDARs (empty -> no cut).
    std::vector<double> lidar_min_ranges;

    double downsample_resolution = 0.5;
    bool downsample_centroid = false;   // voxel mean (pcl::VoxelGrid) instead of the real point nearest the center
    // Adaptive downsample: steer the leaf between downsample_resolution and downsample_max_resolution so the
    // LiDAR update keeps about this many inliers (0 = off, fixed downsample_resolution).
    int downsample_target_inliers = 0;
    double downsample_max_resolution = 0.5;
    double downsample_start_resolution = 0.0;   // leaf of the first 5 s (0 = downsample_resolution); a first map built at 0.1 m broke SNOW-3

    int max_iter = 4;

    double voxel_map_resolution = 1.0;
    int voxel_map_max_layer = 2;
    std::vector<int> voxel_map_layer_size = {5, 5, 5, 5, 5};
    int voxel_map_max_point_size = 1000;
    float voxel_map_plane_thres = 0.01;

    bool voxel_map_sliding_en = false;
    double voxel_map_sliding_thresh = 8.0;
    int voxel_map_half_size = 50;

    bool verbose = false;

    // A/B switches against FAST-LIVO2 behaviour (defaults keep SE3-LIO as published).
    double residual_floor = 0.0;   // added to every point-to-plane residual variance (FL2: 0.001)
    bool deskew_cov_en = true;     // add predicted relative-pose covariance to point noise (UAMC)
    bool init_fl2 = false;         // bg = 0 and P0 = 0.01*I at init (FL2) instead of gyro mean / small P0
    bool const_r = false;          // drop the per-point term from the residual variance (plane covariance kept)

    // Cameras (FAST-LIVO2-style sparse-direct photometric update after the LiDAR update).
    // Off by default; when on, estimatePose(meas, grays) runs one sequential update per camera
    // whose image is non-empty (same trigger for all cameras -> one epoch). Each camera keeps
    // its own visual map (patches are camera-specific).
    bool visual_en = false;
    VisualUpdateConfig visual;
    std::vector<CameraConfig> cameras;
};

class SE3_LIO {
public:
    explicit SE3_LIO(const SE3_LIO_Config &_config);
    SE3_LIO() : config_(){};

    // grays[i]: undistorted CV_8UC1 image of camera i at the epoch end (Measurement::end_time);
    // empty Mat / empty list -> no photometric update for that camera / at all.
    void estimatePose(MeasurementPtr &_measurement_ptr, const std::vector<cv::Mat> &_grays = {});

    // Merge `lidars` into `lidar`: apply each extrinsic, tag intensity with the
    // LiDAR index, re-base point offsets to the earliest scan start, sort by time.
    void mergeLiDARs(Measurement &_measurement) const;
    void estimatePoseWithGPS_v2(MeasurementPtr &_measurement_ptr);

    State getState() const { return state_; }

    size_t getMapSize() const { return map_manager_ ? map_manager_->mapSize() : 0; }
    int numTrackedPoints() const {
        int n = 0;
        for (const auto &vu : visual_updaters_) n += vu->numTracked();
        return n;
    }

private:
    SE3_LIO_Config config_;

    MeasurementPtr meas_;
    State state_;

    double adaptiveLeaf(double _stamp);
    double leaf_ = 0.0, first_stamp_ = -1.0;
    int lidar_inliers_ = 0;  // inliers of the previous LiDAR update

    se3_lio::StatePredict state_predictor_;
    se3_lio::StateUpdate state_updater_;
    std::shared_ptr<se3_lio::ManageMap> map_manager_;
    std::vector<std::unique_ptr<se3_lio::VisualUpdate>> visual_updaters_;  // empty unless visual_en
};

}  // namespace se3_lio::pipeline