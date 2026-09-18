#include "visual/visual_map.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <unordered_set>

namespace se3_lio {

namespace {
// New-observation gates (FAST-LIVO2 defaults).
constexpr double kMinObsTrans = 0.5;    // m
constexpr double kMinObsRot = 0.3;      // rad
constexpr double kMinObsPxDist = 40.0;  // px

// Colour (r, g, b) at px in a BGR image; (-1,-1,-1) when unavailable.
Eigen::Vector3i sampleRGB(const cv::Mat &color, const Eigen::Vector2d &px) {
    if (color.empty() || color.type() != CV_8UC3) {
        return {-1, -1, -1};
    }
    const int u = static_cast<int>(std::lround(px.x()));
    const int v = static_cast<int>(std::lround(px.y()));
    if (u < 0 || v < 0 || u >= color.cols || v >= color.rows) {
        return {-1, -1, -1};
    }
    const cv::Vec3b &bgr = color.at<cv::Vec3b>(v, u);
    return {bgr[2], bgr[1], bgr[0]};
}
}  // namespace

const Feature *VisualPoint::getCloseViewObs(const Eigen::Vector3d &cam_pos_w,
                                            double max_angle_rad) const {
    const Eigen::Vector3d dir_cur = (cam_pos_w - pos).normalized();
    const Feature *best = nullptr;
    double best_cos = std::cos(max_angle_rad);
    for (const auto &ftr : obs) {
        const Eigen::Vector3d cam_pos_ref = ftr->T_c_w.inverse().translation();
        const double cos_angle = (cam_pos_ref - pos).normalized().dot(dir_cur);
        if (cos_angle > best_cos) {
            best_cos = cos_angle;
            best = ftr.get();
        }
    }
    return best;
}

void VisualMap::insert(const std::shared_ptr<VisualPoint> &pt) {
    map_[toVoxelLoc(pt->pos, voxel_size_)].push_back(pt);
    num_points_++;
}

int VisualMap::maybeSlide(const Eigen::Vector3d &position, double thresh, double half_meters) {
    if (has_slid_ && (position - last_slide_position_).norm() < thresh) {
        return 0;
    }
    has_slid_ = true;
    last_slide_position_ = position;

    const VOXEL_LOC c = toVoxelLoc(position, voxel_size_);
    const int64_t h = static_cast<int64_t>(half_meters / voxel_size_);
    int evicted = 0;
    for (auto it = map_.begin(); it != map_.end();) {
        const VOXEL_LOC &loc = it->first;
        const bool out = loc.x > c.x + h || loc.x < c.x - h || loc.y > c.y + h ||
                         loc.y < c.y - h || loc.z > c.z + h || loc.z < c.z - h;
        if (out) {
            // Keyframe-archive hook: evicted points (it->second) leave here.
            evicted += it->second.size();
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
    num_points_ -= evicted;
    return evicted;
}

void VisualMap::dumpTo(const std::string &path) const {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(4);
    for (const auto &kv : map_) {
        for (const auto &pt : kv.second) {
            int r = -1, g = -1, b = -1;
            if (!pt->obs.empty()) {
                const auto &ftr = pt->obs.back();
                if (ftr->rgb.x() >= 0) {
                    r = ftr->rgb.x();
                    g = ftr->rgb.y();
                    b = ftr->rgb.z();
                } else {
                    // Mono/IR input: fall back to a gray triplet from the patch.
                    const int u = static_cast<int>(std::lround(ftr->px.x()));
                    const int v = static_cast<int>(std::lround(ftr->px.y()));
                    if (!ftr->img.empty() && u >= 0 && v >= 0 && u < ftr->img.cols &&
                        v < ftr->img.rows) {
                        r = g = b = ftr->img.at<uchar>(v, u);
                    }
                }
            }
            out << pt->pos.x() << ' ' << pt->pos.y() << ' ' << pt->pos.z() << ' '
                << pt->normal.x() << ' ' << pt->normal.y() << ' ' << pt->normal.z()
                << ' ' << r << ' ' << g << ' ' << b << '\n';
        }
    }
}

std::vector<VisualPoint *> VisualMap::collectCandidates(
    const std::vector<Eigen::Vector3d> &scan_world) const {
    std::unordered_set<VOXEL_LOC> voxels;
    for (const auto &p : scan_world) {
        voxels.insert(toVoxelLoc(p, voxel_size_));
    }
    std::vector<VisualPoint *> out;
    for (const auto &loc : voxels) {
        auto it = map_.find(loc);
        if (it == map_.end()) {
            continue;
        }
        for (const auto &pt : it->second) {
            if (!pt->obs.empty()) {
                out.push_back(pt.get());
            }
        }
    }
    return out;
}

int VisualMap::generatePoints(const cv::Mat &img, const PinholeCamera &cam,
                              const Sophus::SE3d &T_c_w,
                              const std::vector<Eigen::Vector3d> &scan_world,
                              const std::unordered_map<VOXEL_LOC, OctoTree *> &lidar_map,
                              double lidar_voxel_size, int grid_size,
                              const std::vector<Eigen::Vector2d> &occupied_px,
                              const cv::Mat &color) {
    const int n_cols = (cam.width() + grid_size - 1) / grid_size;
    const int n_rows = (cam.height() + grid_size - 1) / grid_size;
    const int n_cells = n_cols * n_rows;

    auto cellIndex = [&](const Eigen::Vector2d &px) {
        return (int)(px.y() / grid_size) * n_cols + (int)(px.x() / grid_size);
    };

    std::vector<bool> occupied(n_cells, false);
    for (const auto &px : occupied_px) {
        occupied[cellIndex(px)] = true;
    }

    std::vector<double> best_score(n_cells, 0.0);
    std::vector<Eigen::Vector3d> best_point(n_cells);
    std::vector<Eigen::Vector2d> best_px(n_cells);

    const Eigen::Vector3d cam_pos_w = T_c_w.inverse().translation();
    for (const auto &p_w : scan_world) {
        const Eigen::Vector3d p_c = T_c_w * p_w;
        if (p_c.z() <= 0.1) {
            continue;
        }
        const Eigen::Vector2d px = cam.project(p_c);
        if (!cam.isInFrame(px, 20)) {
            continue;
        }
        const int cell = cellIndex(px);
        if (occupied[cell]) {
            continue;
        }
        const double score = shiTomasiScore(img, (int)px.x(), (int)px.y());
        if (score > best_score[cell]) {
            best_score[cell] = score;
            best_point[cell] = p_w;
            best_px[cell] = px;
        }
    }

    int num_added = 0;
    for (int cell = 0; cell < n_cells; cell++) {
        if (best_score[cell] <= 1e-3) {
            continue;
        }
        const Plane *plane = findPlaneForPoint(lidar_map, best_point[cell], lidar_voxel_size);
        if (plane == nullptr) {
            continue;
        }

        auto pt = std::make_shared<VisualPoint>();
        pt->pos = best_point[cell];
        pt->normal = plane->normal;
        if (pt->normal.dot(cam_pos_w - pt->pos) < 0) {
            pt->normal = -pt->normal;
        }

        auto ftr = std::make_shared<Feature>();
        ftr->img = img;
        ftr->px = best_px[cell];
        ftr->T_c_w = T_c_w;
        ftr->rgb = sampleRGB(color, best_px[cell]);
        pt->obs.push_back(ftr);

        insert(pt);
        num_added++;
    }
    return num_added;
}

void VisualMap::addObservation(VisualPoint *pt, const cv::Mat &img, const PinholeCamera &cam,
                               const Sophus::SE3d &T_c_w, const cv::Mat &color) {
    const Eigen::Vector3d p_c = T_c_w * pt->pos;
    if (p_c.z() <= 0.1) {
        return;
    }
    const Eigen::Vector2d px = cam.project(p_c);
    if (!cam.isInFrame(px, 20)) {
        return;
    }

    const Feature &last = *pt->obs.back();
    const Sophus::SE3d T_cur_last = T_c_w * last.T_c_w.inverse();
    const bool moved = T_cur_last.translation().norm() > kMinObsTrans ||
                       T_cur_last.so3().log().norm() > kMinObsRot ||
                       (px - last.px).norm() > kMinObsPxDist;
    if (!moved) {
        return;
    }

    auto ftr = std::make_shared<Feature>();
    ftr->img = img;
    ftr->px = px;
    ftr->T_c_w = T_c_w;
    ftr->rgb = sampleRGB(color, px);
    pt->obs.push_back(ftr);
    if ((int)pt->obs.size() > max_obs_) {
        pt->obs.pop_front();
    }
}

double shiTomasiScore(const cv::Mat &img, int u, int v) {
    constexpr int kHalfBox = 4;
    if (u < kHalfBox + 1 || v < kHalfBox + 1 || u >= img.cols - kHalfBox - 1 ||
        v >= img.rows - kHalfBox - 1) {
        return 0.0;
    }

    double dxx = 0.0, dyy = 0.0, dxy = 0.0;
    for (int y = v - kHalfBox; y < v + kHalfBox; y++) {
        for (int x = u - kHalfBox; x < u + kHalfBox; x++) {
            double dx, dy;
            if (img.type() == CV_32FC1) {
                dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
                dy = img.at<float>(y + 1, x) - img.at<float>(y - 1, x);
            } else {
                dx = (double)img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1);
                dy = (double)img.at<uchar>(y + 1, x) - img.at<uchar>(y - 1, x);
            }
            dxx += dx * dx;
            dyy += dy * dy;
            dxy += dx * dy;
        }
    }
    const double n_pix = (2 * kHalfBox) * (2 * kHalfBox);
    dxx /= 2.0 * n_pix;
    dyy /= 2.0 * n_pix;
    dxy /= 2.0 * n_pix;
    return 0.5 * (dxx + dyy - std::sqrt((dxx - dyy) * (dxx - dyy) + 4.0 * dxy * dxy));
}

const Plane *findPlaneForPoint(const std::unordered_map<VOXEL_LOC, OctoTree *> &lidar_map,
                               const Eigen::Vector3d &p_w, double voxel_size) {
    auto it = lidar_map.find(toVoxelLoc(p_w, voxel_size));
    if (it == lidar_map.end()) {
        return nullptr;
    }
    const OctoTree *node = it->second;
    while (node != nullptr) {
        if (node->plane_ptr_ != nullptr && node->plane_ptr_->is_plane) {
            return node->plane_ptr_;
        }
        if (node->octo_state_ != 1) {
            return nullptr;
        }
        const int leafnum = 4 * (p_w[0] > node->voxel_center_[0]) +
                            2 * (p_w[1] > node->voxel_center_[1]) +
                            (p_w[2] > node->voxel_center_[2]);
        node = node->leaves_[leafnum];
    }
    return nullptr;
}

}  // namespace se3_lio
