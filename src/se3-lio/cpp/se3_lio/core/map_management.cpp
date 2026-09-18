#include "map_management.h"

#include <algorithm>
#include <iostream>

namespace se3_lio {

inline std::vector<pointWithCov> transformGlobalPointsWithNoise(
    const LiDAR &_cloud,
    const Eigen::Matrix4d &_pose,
    const Eigen::Matrix<double, 6, 6> &_pose_cov) {
    Eigen::Matrix3d rot = _pose.block<3, 3>(0, 0);
    Eigen::Vector3d tls = _pose.block<3, 1>(0, 3);

    std::vector<pointWithCov> pv_list(_cloud.points.size());
    for (size_t i = 0; i < _cloud.points.size(); i++) {
        CustomPointType point = _cloud.points[i];

        Eigen::Vector3d pt = point.getVector3fMap().cast<double>();
        Eigen::Matrix3d noise = _cloud.noises[i];

        Eigen::Vector3d transformed_pt = rot * pt + tls;

        Eigen::Matrix<double, 3, 6> J_pt;
        J_pt << rot, -rot * skew_sym(pt);
        Eigen::Matrix3d transformed_noise =
            J_pt * _pose_cov * J_pt.transpose() + rot * noise * rot.transpose();

        pv_list[i].point << transformed_pt;
        pv_list[i].cov = transformed_noise;
    }

    return pv_list;
}

ManageMap::ManageMap(ManageMapConfig _config) : config_(_config) {
    is_initialized_ = false;

    if (config_.verbose) {
        std::cout << " ManageMap parameters check: " << std::endl;
        std::cout << "   resolution: " << config_.resolution << std::endl;
        std::cout << "   max_layer: " << config_.max_layer << std::endl;
        std::cout << "   layer_size: ";
        for (const auto &size : config_.layer_size) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        std::cout << "   plane_thres: " << config_.plane_thres << std::endl;
    }
}

void ManageMap::setMeasurement(LiDAR _measurement) { undistorted_cloud_ = _measurement; }

void ManageMap::setState(State _state) { curr_state_ = _state; };

void ManageMap::initMap() {
    pv_list_ = transformGlobalPointsWithNoise(undistorted_cloud_, curr_pose_, curr_pose_cov_);

    buildVoxelMap(pv_list_, config_.resolution, config_.max_layer, config_.layer_size,
                  config_.max_point_size, config_.max_point_size, config_.plane_thres, voxel_map_);

    is_initialized_ = true;

    return;
}

std::vector<Plane> ManageMap::getPlanes() {
    std::vector<Plane> planes;
    for (const auto &voxel : voxel_map_) {
        VOXEL_LOC key = voxel.first;
        Eigen::Vector3d voxel_center(key.x * config_.resolution, key.y * config_.resolution,
                                     key.z * config_.resolution);

        // if (voxel_center.norm() > config_.pub_voxelmap_max_range) continue;

        GetUpdatePlane(voxel.second, config_.max_layer, planes);
    }

    return planes;
}

void ManageMap::updateMap() {
    curr_pose_ = curr_state_.pose;
    curr_pose_cov_ = curr_state_.pose_cov();

    if (!is_initialized_) {
        initMap();
        return;
    }

    pv_list_ = transformGlobalPointsWithNoise(undistorted_cloud_, curr_pose_, curr_pose_cov_);

    std::sort(pv_list_.begin(), pv_list_.end(), [](const pointWithCov &x, const pointWithCov &y) {
        return x.cov.diagonal().norm() < y.cov.diagonal().norm();
    });

    updateVoxelMapOMP(pv_list_, config_.resolution, config_.max_layer, config_.layer_size,
                      config_.max_point_size, config_.max_point_size, config_.plane_thres,
                      voxel_map_);

    if (config_.map_sliding_en) {
        mapSliding(curr_pose_.block<3, 1>(0, 3));
    }

    if (config_.verbose) {
        std::cout << "Map is updated. current map size: " << voxel_map_.size() << std::endl;
    }
}

void ManageMap::mapSliding(const Eigen::Vector3d &position) {
    if (has_slid_ && (position - last_slide_position_).norm() < config_.sliding_thresh) {
        return;
    }
    has_slid_ = true;
    last_slide_position_ = position;

    // Body voxel index, matching the keying convention in voxel_map_util.cpp.
    int64_t loc[3];
    for (int j = 0; j < 3; j++) {
        double v = position[j] / config_.resolution;
        if (v < 0) v -= 1.0;
        loc[j] = static_cast<int64_t>(v);
    }
    int64_t h = config_.half_map_size;
    clearMemOutOfMap(loc[0] + h, loc[0] - h, loc[1] + h, loc[1] - h, loc[2] + h, loc[2] - h);
}

void ManageMap::clearMemOutOfMap(int64_t x_max, int64_t x_min, int64_t y_max, int64_t y_min,
                                 int64_t z_max, int64_t z_min) {
    int deleted = 0;
    for (auto it = voxel_map_.begin(); it != voxel_map_.end();) {
        const VOXEL_LOC &loc = it->first;
        bool out = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min ||
                   loc.z > z_max || loc.z < z_min;
        if (out) {
            delete it->second;
            it = voxel_map_.erase(it);
            deleted++;
        } else {
            ++it;
        }
    }

    if (config_.verbose) {
        std::cout << "[map sliding] deleted " << deleted << " voxels, map size "
                  << voxel_map_.size() << std::endl;
    }
}

}  // namespace se3_lio
