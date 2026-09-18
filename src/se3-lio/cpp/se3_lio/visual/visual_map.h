#ifndef VISUAL_MAP_H
#define VISUAL_MAP_H

#include <deque>
#include <memory>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/temporary/voxel_map_util.h"
#include "visual/camera.h"

namespace se3_lio {

// One patch observation of a visual map point.
struct Feature {
    cv::Mat img;         // grayscale frame at capture (shared, refcounted)
    Eigen::Vector2d px;  // observed pixel (level 0)
    Sophus::SE3d T_c_w;  // world-to-camera at capture
    // Colour sampled at px (r, g, b); (-1,-1,-1) when the input is mono/IR.
    // Carried for map colouring only — the photometric update uses img (gray).
    Eigen::Vector3i rgb{-1, -1, -1};
};

struct VisualPoint {
    Eigen::Vector3d pos;     // world
    Eigen::Vector3d normal;  // world, oriented toward the first observing camera
    std::deque<std::shared_ptr<Feature>> obs;

    // Observation with the smallest view angle w.r.t. the current camera
    // position; nullptr if the best one exceeds max_angle_rad.
    const Feature *getCloseViewObs(const Eigen::Vector3d &cam_pos_w, double max_angle_rad) const;
};

// Sparse map of visual points bucketed in fixed-size world voxels.
class VisualMap {
public:
    VisualMap(double voxel_size, int max_obs) : voxel_size_(voxel_size), max_obs_(max_obs) {}

    void insert(const std::shared_ptr<VisualPoint> &pt);

    // All points in the voxels touched by the current scan.
    std::vector<VisualPoint *> collectCandidates(
        const std::vector<Eigen::Vector3d> &scan_world) const;

    // Spawn new points from scan points lying on lidar-map planes: keep the
    // highest Shi-Tomasi-score point per image grid cell, skipping cells
    // already covered by tracked points (occupied_px). Returns #new points.
    // color (optional, CV_8UC3 BGR): sampled at the kept pixel into Feature::rgb.
    int generatePoints(const cv::Mat &img, const PinholeCamera &cam, const Sophus::SE3d &T_c_w,
                       const std::vector<Eigen::Vector3d> &scan_world,
                       const std::unordered_map<VOXEL_LOC, OctoTree *> &lidar_map,
                       double lidar_voxel_size, int grid_size,
                       const std::vector<Eigen::Vector2d> &occupied_px,
                       const cv::Mat &color = cv::Mat());

    // Append a new observation if the viewpoint moved enough since the last
    // one; evicts the oldest when the cap is reached.
    void addObservation(VisualPoint *pt, const cv::Mat &img, const PinholeCamera &cam,
                        const Sophus::SE3d &T_c_w, const cv::Mat &color = cv::Mat());

    size_t size() const { return num_points_; }

    // Sliding window synced with the LiDAR voxel map (same policy): once the
    // body has moved thresh meters since the last slide, drop every voxel
    // outside [+/- half_meters] around position. Returns #points evicted.
    // Evicted points are the future keyframe-archive hook (loop closing).
    int maybeSlide(const Eigen::Vector3d &position, double thresh, double half_meters);

    // Dump every visual map point as "x y z nx ny nz r g b" (one per line);
    // colour from the most recent observation (Feature::rgb), falling back to a
    // gray triplet sampled from the patch, or -1 -1 -1 if unavailable.
    void dumpTo(const std::string &path) const;

private:
    double voxel_size_;
    int max_obs_;
    size_t num_points_ = 0;
    bool has_slid_ = false;
    Eigen::Vector3d last_slide_position_ = Eigen::Vector3d::Zero();
    std::unordered_map<VOXEL_LOC, std::vector<std::shared_ptr<VisualPoint>>> map_;
};

// Shi-Tomasi corner score of the 8x8 box around (u, v); 0 near the border.
double shiTomasiScore(const cv::Mat &img, int u, int v);

// Descend the lidar voxel map to the plane containing p_w, nullptr if none.
const Plane *findPlaneForPoint(const std::unordered_map<VOXEL_LOC, OctoTree *> &lidar_map,
                               const Eigen::Vector3d &p_w, double voxel_size);

inline VOXEL_LOC toVoxelLoc(const Eigen::Vector3d &p, double voxel_size) {
    double loc[3];
    for (int j = 0; j < 3; j++) {
        loc[j] = p[j] / voxel_size;
        if (loc[j] < 0) {
            loc[j] -= 1.0;
        }
    }
    return VOXEL_LOC((int64_t)loc[0], (int64_t)loc[1], (int64_t)loc[2]);
}

}  // namespace se3_lio

#endif  // VISUAL_MAP_H
