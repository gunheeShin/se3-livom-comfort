#include "SE3_LIO.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstdio>
#include <cstdlib>

namespace se3_lio::pipeline {

SE3_LIO::SE3_LIO(const SE3_LIO_Config &_config) : config_(_config) {
    state_predictor_.setIMUNoise(config_.gyr_noise, config_.acc_noise, config_.bg_noise,
                                 config_.ba_noise);
    std::vector<double> range_noises = config_.lidar_range_noises;
    std::vector<double> angle_noises = config_.lidar_angle_noises;
    if (range_noises.empty()) range_noises = {config_.lidar_range_noise};
    if (angle_noises.empty()) angle_noises = {config_.lidar_angle_noise};
    state_predictor_.setLiDARNoise(range_noises, angle_noises);
    state_predictor_.setVerbose(config_.verbose);
    state_predictor_.setDeskewCovEnabled(config_.deskew_cov_en);
    state_predictor_.setInitFL2(config_.init_fl2);

    se3_lio::ManageMapConfig manage_map_config;
    manage_map_config.resolution = config_.voxel_map_resolution;
    manage_map_config.max_layer = config_.voxel_map_max_layer;
    manage_map_config.layer_size = config_.voxel_map_layer_size;
    manage_map_config.max_point_size = config_.voxel_map_max_point_size;
    manage_map_config.plane_thres = config_.voxel_map_plane_thres;
    manage_map_config.map_sliding_en = config_.voxel_map_sliding_en;
    manage_map_config.sliding_thresh = config_.voxel_map_sliding_thresh;
    manage_map_config.half_map_size = config_.voxel_map_half_size;
    manage_map_config.verbose = config_.verbose;

    map_manager_ = std::make_shared<se3_lio::ManageMap>(manage_map_config);

    state_updater_ = se3_lio::StateUpdate(config_.max_iter);
    state_updater_.setResidualFloor(config_.residual_floor);
    state_updater_.setConstR(config_.const_r);
    state_updater_.setManageMapConfig(manage_map_config);
    state_updater_.setMap(map_manager_);

    if (config_.visual_en) {
        for (const auto &c : config_.cameras) {
            PinholeCamera cam(c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.dist, c.fisheye);
            visual_updaters_.push_back(std::make_unique<se3_lio::VisualUpdate>(
                config_.visual, cam, toSE3(c.T_cam_imu)));
        }
    }
}

void SE3_LIO::mergeLiDARs(Measurement &_measurement) const {
    const auto &lidars = _measurement.lidars;
    double t0 = lidars.front().header.timestamp;
    size_t n = 0;
    for (const auto &l : lidars) {
        if (l.points.empty()) continue;
        t0 = std::min(t0, l.header.timestamp);
        n += l.points.size();
    }
    LiDAR merged;
    merged.header = lidars.front().header;
    merged.header.timestamp = t0;
    merged.points.reserve(n);
    for (size_t i = 0; i < lidars.size(); i++) {
        if (lidars[i].points.empty()) continue;
        PointCloudType pts = transformPointCloud(lidars[i].points, config_.lidar_extrinsics[i]);
        double offset = lidars[i].header.timestamp - t0;
        float min_r2 = 0.f;
        if (i < config_.lidar_min_ranges.size())
            min_r2 = static_cast<float>(config_.lidar_min_ranges[i] * config_.lidar_min_ranges[i]);
        for (size_t k = 0; k < pts.size(); k++) {
            const auto &raw = lidars[i].points[k];
            if (raw.x * raw.x + raw.y * raw.y + raw.z * raw.z <= min_r2) continue;
            auto p = pts[k];
            p.intensity = static_cast<float>(i);
            p.timestamp += offset;
            merged.points.push_back(p);
        }
    }
    std::stable_sort(merged.points.begin(), merged.points.end(),
                     [](const CustomPointType &a, const CustomPointType &b) {
                         return a.timestamp < b.timestamp;
                     });
    _measurement.lidar = merged;
}

namespace {
// Diagnostic: SE3LIO_DUMP=1 prints the full state after each update stage to stderr.
void dumpState(const char *tag, const se3_lio::State &s) {
    static const bool on = std::getenv("SE3LIO_DUMP") != nullptr;
    if (!on) return;
    const Eigen::AngleAxisd aa(Eigen::Matrix3d(s.rot()));
    const Eigen::Vector3d rv = aa.angle() * aa.axis();
    fprintf(stderr, "DUMP %s %.6f pos %.4f %.4f %.4f rot %.5f %.5f %.5f vel %.4f %.4f %.4f bg %.5f %.5f %.5f ba %.4f %.4f %.4f grav %.4f %.4f %.4f sig", tag, s.stamp,
            s.pos().x(), s.pos().y(), s.pos().z(), rv.x(), rv.y(), rv.z(), s.vel.x(), s.vel.y(), s.vel.z(),
            s.bg.x(), s.bg.y(), s.bg.z(), s.ba.x(), s.ba.y(), s.ba.z(), s.grav.x(), s.grav.y(), s.grav.z());
    for (int i = 0; i < 9; i++) fprintf(stderr, " %.2e", std::sqrt(std::max(0.0, s.covariance(i, i))));
    fprintf(stderr, " inl %d res %.4f\n", s.num_inliers, s.residual);
}

// Diagnostic: SE3LIO_TIMING=1 prints per-frame wall-clock (ms) of predict / update / map to stderr.
struct StageTimer {
    const bool on = std::getenv("SE3LIO_TIMING") != nullptr;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    double lap() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
        return ms;
    }
};
}  // namespace

double SE3_LIO::adaptiveLeaf(double _stamp) {
    const double lo = config_.downsample_resolution;
    if (config_.downsample_target_inliers <= 0) return lo;
    if (first_stamp_ < 0) first_stamp_ = _stamp;
    if (leaf_ == 0.0 || _stamp - first_stamp_ < 5.0) {
        leaf_ = std::max(lo, config_.downsample_start_resolution);  // leaf of the first map
    } else if (lidar_inliers_ > 0) {
        // inliers ~ leaf^-1.5 (measured); grow by at most 5 % per scan, shrink by up to 20 %
        const double ratio = std::pow(double(lidar_inliers_) / config_.downsample_target_inliers, 1.0 / 1.5);
        leaf_ = std::clamp(leaf_ * std::clamp(ratio, 0.8, 1.05), lo, std::max(lo, config_.downsample_max_resolution));
    }
    return leaf_;
}

void SE3_LIO::estimatePose(MeasurementPtr &_measurement_ptr, const std::vector<cv::Mat> &_grays) {
    StageTimer timer;
    if (!_measurement_ptr->lidars.empty()) mergeLiDARs(*_measurement_ptr);

    state_predictor_.setState(state_);
    state_predictor_.setMeasurement(_measurement_ptr);
    state_ = state_predictor_.predictState();
    state_predictor_.undistortCloud();

    if (!state_predictor_.isValid()) return;

    _measurement_ptr->raw_lidar = _measurement_ptr->lidar;
    const double leaf = adaptiveLeaf(state_.stamp);
    if (config_.voxel_map_plane_thres_start > 0.0f) {
        if (first_stamp_ < 0) first_stamp_ = state_.stamp;
        map_manager_->setPlaneThres(state_.stamp - first_stamp_ < 5.0 ? config_.voxel_map_plane_thres_start : config_.voxel_map_plane_thres);
    }
    downsampleCloud(_measurement_ptr->lidar, leaf, config_.downsample_centroid);

    state_predictor_.calculateUndistCloudCov(_measurement_ptr->lidar);

    const double t_predict = timer.lap();
    state_updater_.setState(state_);
    state_updater_.setMeasurement(_measurement_ptr);
    state_ = state_updater_.updateState();
    lidar_inliers_ = state_updater_.isValid() ? state_.num_inliers : 0;
    double t_update = timer.lap();
    dumpState("L", state_);

    std::vector<size_t> active;  // cameras with an image this epoch
    for (size_t i = 0; i < visual_updaters_.size() && i < _grays.size(); i++)
        if (!_grays[i].empty()) active.push_back(i);

    if (active.empty()) {
        map_manager_->setState(state_);
        map_manager_->setMeasurement(_measurement_ptr->lidar);
        map_manager_->updateMap();
        if (timer.on)
            fprintf(stderr, "TIMING %.6f predict %.2f update %.2f map %.2f leaf %.3f inl %d\n", state_.stamp, t_predict, t_update, timer.lap(), leaf, lidar_inliers_);
        return;
    }

    // Photometric updates take the LiDAR posterior as prior (FAST-LIVO2 order), one camera
    // after another (independent measurements -> sequential Kalman updates).
    const auto &lidar = _measurement_ptr->lidar;
    std::vector<Eigen::Vector3d> scan_world;
    scan_world.reserve(lidar.points.size());
    const Eigen::Matrix3d rot = state_.rot();
    const Eigen::Vector3d pos = state_.pos();
    for (const auto &pt : lidar.points)
        scan_world.push_back(rot * pt.getVector3fMap().cast<double>() + pos);

    bool fast_turn = false;
    if (config_.visual.max_gyro_deg > 0.0) {
        double w = 0.0;
        for (const auto &imu : _measurement_ptr->imu)
            w = std::max(w, (imu.angular_velocity - state_.bg).norm());
        fast_turn = w > config_.visual.max_gyro_deg * M_PI / 180.0;
    }
    if (!fast_turn)
        for (size_t i : active) state_ = visual_updaters_[i]->update(state_, _grays[i], scan_world);
    t_update += timer.lap();  // photometric update on top of the LiDAR one
    dumpState("V", state_);

    map_manager_->setState(state_);
    map_manager_->setMeasurement(lidar);
    map_manager_->updateMap();

    for (size_t i : active) {
        visual_updaters_[i]->updateMap(state_, _grays[i], scan_world, map_manager_->voxel_map_,
                                       config_.voxel_map_resolution);
        if (config_.voxel_map_sliding_en)
            visual_updaters_[i]->map().maybeSlide(
                state_.pos(), config_.voxel_map_sliding_thresh,
                config_.voxel_map_half_size * config_.voxel_map_resolution);
    }
    if (timer.on)
        fprintf(stderr, "TIMING %.6f predict %.2f update %.2f map %.2f leaf %.3f inl %d\n", state_.stamp, t_predict, t_update, timer.lap(), leaf, lidar_inliers_);
}

void SE3_LIO::estimatePoseWithGPS_v2(MeasurementPtr &_measurement_ptr) {
    state_predictor_.setState(state_);
    state_predictor_.setMeasurement(_measurement_ptr);
    state_ = state_predictor_.predictState();
    state_predictor_.undistortCloud();

    if (!state_predictor_.isValid()) return;

    downsampleCloud(_measurement_ptr->lidar, config_.downsample_resolution, config_.downsample_centroid);

    state_predictor_.calculateUndistCloudCov(_measurement_ptr->lidar);

    state_updater_.setState(state_);
    state_updater_.setMeasurement(_measurement_ptr);
    state_ = state_updater_.updateState_v2();

    map_manager_->setState(state_);
    map_manager_->setMeasurement(_measurement_ptr->lidar);
    map_manager_->updateMap();
}

}  // namespace se3_lio::pipeline