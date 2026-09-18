#ifndef VISUAL_UPDATE_H
#define VISUAL_UPDATE_H

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <vector>

#include "common/data_type.h"
#include "core/lie.h"
#include "visual/camera.h"
#include "visual/visual_map.h"

namespace se3_lio {

// Quaternion round-trip keeps Sophus' orthonormality assertion happy.
inline Sophus::SE3d toSE3(const Eigen::Matrix4d &T) {
    return Sophus::SE3d(Eigen::Quaterniond(Eigen::Matrix3d(T.block<3, 3>(0, 0))).normalized(),
                        T.block<3, 1>(0, 3));
}

struct VisualUpdateConfig {
    int patch_size = 8;        // patch side length (px)
    int pyramid_levels = 3;    // dilated-sampling pyramid levels
    int grid_size = 40;        // image grid cell for point selection (px)
    int max_iter = 5;          // iterations per pyramid level
    double img_point_cov = 100.0;       // photometric measurement variance
    double outlier_threshold = 1000.0;  // per-pixel squared-intensity gate
    double depth_continuous_thresh = 0.5;  // m, depth-discontinuity rejection
    double voxel_size = 0.5;               // visual map voxel size (m)
    int max_obs = 30;                      // observations per point
    double max_view_angle_deg = 60.0;      // reference patch view-angle gate
    double max_gyro_deg = 0.0;             // diagnostic: skip the update when |gyro| exceeds this (0 = off)
    bool probe = false;                    // diagnostic: log the camera-only pose offset, apply nothing
    bool pose_only = false;                // diagnostic: correct only the pose block (velocity/bias/gravity untouched)
    bool keep_cov = false;                 // diagnostic: apply the state correction but leave the covariance untouched
    bool verbose = false;
};

// FAST-LIVO2-style sparse-direct photometric update on the SE(3) error state.
// Sequential: consumes the LiDAR posterior as prior (no exposure estimation,
// no UAMC — brightness constancy assumed).
class VisualUpdate {
public:
    VisualUpdate(const VisualUpdateConfig &config, const PinholeCamera &cam,
                 const Sophus::SE3d &T_c_i);

    // Iterated photometric update. scan_world: current undistorted scan in the
    // world frame (candidate voxels + occlusion depth). Returns the posterior
    // (prior unchanged when too few points track).
    State update(const State &prior, const cv::Mat &gray,
                 const std::vector<Eigen::Vector3d> &scan_world);

    // Map maintenance with the updated state: new observations for tracked
    // points, new points from lidar-map planes in uncovered grid cells.
    // color (optional, CV_8UC3 BGR): threaded to the map for point colouring.
    void updateMap(const State &state, const cv::Mat &gray,
                   const std::vector<Eigen::Vector3d> &scan_world,
                   const std::unordered_map<VOXEL_LOC, OctoTree *> &lidar_map,
                   double lidar_map_resolution, const cv::Mat &color = cv::Mat());

    VisualMap &map() { return map_; }
    int numTracked() const { return (int)submap_.size(); }

    // T_c_w = T_c_i * T_i_w
    Sophus::SE3d cameraPose(const State &s) const { return T_c_i_ * toSE3(s.pose).inverse(); }

private:
    struct SubmapPoint {
        VisualPoint *pt;
        std::vector<float> ref_patch;  // warped, pyramid_levels * patch_size^2
        int search_level;
    };

    void retrieveSubmap(const State &prior, const cv::Mat &gray,
                        const std::vector<Eigen::Vector3d> &scan_world);
    // Stacks h_, h_x_ at the given pyramid level; returns false if nothing
    // usable. error: accumulated squared photometric residual.
    bool buildResidual(const State &state, const cv::Mat &gray, int level, double &error);

    VisualUpdateConfig cfg_;
    PinholeCamera cam_;
    Sophus::SE3d T_c_i_;
    VisualMap map_;

    std::vector<SubmapPoint> submap_;
    std::vector<Eigen::Vector2d> tracked_px_;  // submap pixels at the prior pose

    Eigen::Matrix<double, Eigen::Dynamic, 1> h_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x_;
};

// ---- helpers shared with the tests ----

// Bilinear intensity at px and central-difference gradient at stride `scale`
// (level-0 pixel units). Supports CV_8UC1 and CV_32FC1.
bool sampleWithGradient(const cv::Mat &img, const Eigen::Vector2d &px, int scale, double &val,
                        Eigen::Vector2d &grad);
double sampleBilinear(const cv::Mat &img, const Eigen::Vector2d &px);

// d(p_c)/d(delta) for a right SE(3) perturbation of T_w_i: R_c_i * [-I | skew(p_i)]
Eigen::Matrix<double, 3, 6> pointJacobian(const Eigen::Matrix3d &R_c_i,
                                          const Eigen::Vector3d &p_i);

// First-order affine warp d(px_cur)/d(px_ref) of the plane patch around p_ref
// (all quantities in the reference camera frame; normal_ref must not be
// perpendicular to the viewing ray).
Eigen::Matrix2d computeWarpAffine(const PinholeCamera &cam, const Sophus::SE3d &T_cur_ref,
                                  const Eigen::Vector2d &ref_px, const Eigen::Vector3d &p_ref,
                                  const Eigen::Vector3d &normal_ref);

// Coarsest pyramid level at which the warped patch is roughly pixel-scale.
int bestSearchLevel(const Eigen::Matrix2d &A_cur_ref, int max_level);

}  // namespace se3_lio

#endif  // VISUAL_UPDATE_H
