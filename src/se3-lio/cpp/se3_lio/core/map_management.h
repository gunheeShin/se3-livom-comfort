
#ifndef MANAGE_MAP_H
#define MANAGE_MAP_H

// C++ standard libraries
// #include <boost/filesystem.hpp>
#include <filesystem>
#include <mutex>
#include <queue>

// Eigen library
#include <Eigen/Dense>

// Common
#include "common/data_type.h"
#include "common/utils.h"

// core
#include "core/lie.h"
#include "core/temporary/voxel_map_util.h"

namespace se3_lio {

struct ManageMapConfig {
    double resolution = 1.0;
    int max_layer = 2;
    std::vector<int> layer_size = {5, 5, 5, 5, 5};
    // int max_point_size = 1000;
    int max_point_size;
    float plane_thres = 0.01f;
    bool verbose = false;

    // Local-map sliding window: evict root voxels outside a box of
    // [+/- half_map_size] voxels around the body once it has moved
    // sliding_thresh meters since the last slide. Bounds RAM on long runs.
    bool map_sliding_en = false;
    double sliding_thresh = 8.0;
    int half_map_size = 50;
};

class ManageMap {
public:
    explicit ManageMap(ManageMapConfig _config);
    ManageMap() : config_() { is_initialized_ = false; }

    ~ManageMap(){};

    /**
     * @brief Set the State object
     *
     * @param _state
     */
    void setState(State _state);

    /**
     * @brief Set the Synced Measurement object
     *
     * @param _measurement
     */
    void setMeasurement(LiDAR _measurement);

    /**
     * @brief Initialize the map
     *
     */
    void initMap();

    /**
     * @brief Update the map
     *
     */
    void updateMap();

    /**
     * @brief Check whether the map is initialized
     *
     * @return true or false
     */
    bool isInitialized() { return is_initialized_; }

    /**
     * @brief Get the planes
     *
     */
    std::vector<Plane> getPlanes();

    /**
     * @brief Number of root voxels currently held by the map.
     */
    size_t mapSize() const { return voxel_map_.size(); }

    std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map_;

private:
    /**
     * @brief Slide the local map: when the body has moved more than
     * sliding_thresh since the last slide, drop voxels outside the box around
     * its current voxel index.
     */
    void mapSliding(const Eigen::Vector3d &position);
    void clearMemOutOfMap(int64_t x_max, int64_t x_min, int64_t y_max, int64_t y_min,
                          int64_t z_max, int64_t z_min);

    ManageMapConfig config_;

    State curr_state_;
    LiDAR undistorted_cloud_;

    Eigen::Matrix4d curr_pose_ = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 6, 6> curr_pose_cov_;

    std::vector<pointWithCov> pv_list_;

    Eigen::Vector3d last_slide_position_ = Eigen::Vector3d::Zero();
    bool has_slid_ = false;

    bool is_initialized_;
};

}  // namespace se3_lio

#endif  // MANAGE_MAP_H