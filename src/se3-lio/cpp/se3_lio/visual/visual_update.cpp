#include "visual/visual_update.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <cstdio>

namespace se3_lio {

namespace {
constexpr double kMinDepth = 0.1;         // m, points behind/too close to the camera
constexpr double kWarpHalfPatch = 4.0;    // px, finite-difference step for the affine warp
constexpr double kMinGrazingCos = 0.087;  // cos(85 deg), degenerate warp gate
constexpr int kMinTrackedPoints = 5;
}  // namespace

double sampleBilinear(const cv::Mat &img, const Eigen::Vector2d &px) {
    const int x = (int)std::floor(px.x());
    const int y = (int)std::floor(px.y());
    const double subx = px.x() - x;
    const double suby = px.y() - y;
    const double w00 = (1.0 - subx) * (1.0 - suby);
    const double w01 = subx * (1.0 - suby);
    const double w10 = (1.0 - subx) * suby;
    const double w11 = subx * suby;
    if (img.type() == CV_32FC1) {
        return w00 * img.at<float>(y, x) + w01 * img.at<float>(y, x + 1) +
               w10 * img.at<float>(y + 1, x) + w11 * img.at<float>(y + 1, x + 1);
    }
    return w00 * img.at<uchar>(y, x) + w01 * img.at<uchar>(y, x + 1) +
           w10 * img.at<uchar>(y + 1, x) + w11 * img.at<uchar>(y + 1, x + 1);
}

bool sampleWithGradient(const cv::Mat &img, const Eigen::Vector2d &px, int scale, double &val,
                        Eigen::Vector2d &grad) {
    if (px.x() < scale + 1 || px.x() >= img.cols - scale - 2 || px.y() < scale + 1 ||
        px.y() >= img.rows - scale - 2) {
        return false;
    }
    val = sampleBilinear(img, px);
    const Eigen::Vector2d dx(scale, 0.0), dy(0.0, scale);
    grad.x() = (sampleBilinear(img, px + dx) - sampleBilinear(img, px - dx)) / (2.0 * scale);
    grad.y() = (sampleBilinear(img, px + dy) - sampleBilinear(img, px - dy)) / (2.0 * scale);
    return true;
}

Eigen::Matrix<double, 3, 6> pointJacobian(const Eigen::Matrix3d &R_c_i,
                                          const Eigen::Vector3d &p_i) {
    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0) = -R_c_i;
    J.block<3, 3>(0, 3) = R_c_i * skew_sym(p_i);
    return J;
}

Eigen::Matrix2d computeWarpAffine(const PinholeCamera &cam, const Sophus::SE3d &T_cur_ref,
                                  const Eigen::Vector2d &ref_px, const Eigen::Vector3d &p_ref,
                                  const Eigen::Vector3d &normal_ref) {
    const Eigen::Vector2d px_cur = cam.project(T_cur_ref * p_ref);
    const double d = normal_ref.dot(p_ref);
    Eigen::Matrix2d A;
    for (int i = 0; i < 2; i++) {
        Eigen::Vector2d px_ref_i = ref_px;
        px_ref_i[i] += kWarpHalfPatch;
        const Eigen::Vector3d ray = cam.unproject(px_ref_i);
        const Eigen::Vector3d p_i = ray * (d / normal_ref.dot(ray));
        A.col(i) = (cam.project(T_cur_ref * p_i) - px_cur) / kWarpHalfPatch;
    }
    return A;
}

int bestSearchLevel(const Eigen::Matrix2d &A_cur_ref, int max_level) {
    int level = 0;
    double D = A_cur_ref.determinant();
    while (D > 3.0 && level < max_level) {
        level++;
        D *= 0.25;
    }
    return level;
}

VisualUpdate::VisualUpdate(const VisualUpdateConfig &config, const PinholeCamera &cam,
                           const Sophus::SE3d &T_c_i)
    : cfg_(config), cam_(cam), T_c_i_(T_c_i), map_(config.voxel_size, config.max_obs) {}

void VisualUpdate::retrieveSubmap(const State &prior, const cv::Mat &gray,
                                  const std::vector<Eigen::Vector3d> &scan_world) {
    submap_.clear();
    tracked_px_.clear();

    const Sophus::SE3d T_c_w = cameraPose(prior);
    const Eigen::Vector3d cam_pos_w = T_c_w.inverse().translation();
    const int half = cfg_.patch_size / 2;
    const int area = cfg_.patch_size * cfg_.patch_size;
    const double max_angle = cfg_.max_view_angle_deg * M_PI / 180.0;

    // Sparse depth buffer from the current scan (for occlusion rejection).
    cv::Mat depth = cv::Mat::zeros(cam_.height(), cam_.width(), CV_32FC1);
    for (const auto &p_w : scan_world) {
        const Eigen::Vector3d p_c = T_c_w * p_w;
        if (p_c.z() <= kMinDepth) {
            continue;
        }
        const Eigen::Vector2d px = cam_.project(p_c);
        if (!cam_.isInFrame(px, 1)) {
            continue;
        }
        depth.at<float>((int)std::round(px.y()), (int)std::round(px.x())) = (float)p_c.z();
    }

    // Grid dedup: keep the closest candidate per image grid cell.
    const int n_cols = (cam_.width() + cfg_.grid_size - 1) / cfg_.grid_size;
    const int n_rows = (cam_.height() + cfg_.grid_size - 1) / cfg_.grid_size;
    struct Candidate {
        VisualPoint *pt = nullptr;
        Eigen::Vector2d px;
        double depth = std::numeric_limits<double>::max();
    };
    std::vector<Candidate> cells(n_cols * n_rows);

    for (VisualPoint *pt : map_.collectCandidates(scan_world)) {
        const Eigen::Vector3d p_c = T_c_w * pt->pos;
        if (p_c.z() <= kMinDepth) {
            continue;
        }
        const Eigen::Vector2d px = cam_.project(p_c);
        if (!cam_.isInFrame(px, (half + 2))) {
            continue;
        }
        Candidate &cell =
            cells[(int)(px.y() / cfg_.grid_size) * n_cols + (int)(px.x() / cfg_.grid_size)];
        if (p_c.z() < cell.depth) {
            cell = {pt, px, p_c.z()};
        }
    }

    for (const Candidate &cand : cells) {
        if (cand.pt == nullptr) {
            continue;
        }

        // Depth continuity: reject patches over occlusions / depth edges.
        bool depth_continuous = true;
        for (int dy = -half; dy <= half && depth_continuous; dy++) {
            for (int dx = -half; dx <= half; dx++) {
                const float z = depth.at<float>((int)cand.px.y() + dy, (int)cand.px.x() + dx);
                if (z > 0.0f && std::fabs(z - cand.depth) > cfg_.depth_continuous_thresh) {
                    depth_continuous = false;
                    break;
                }
            }
        }
        if (!depth_continuous) {
            continue;
        }

        const Feature *ref = cand.pt->getCloseViewObs(cam_pos_w, max_angle);
        if (ref == nullptr) {
            continue;
        }

        const Eigen::Vector3d p_ref = ref->T_c_w * cand.pt->pos;
        const Eigen::Vector3d n_ref = ref->T_c_w.so3() * cand.pt->normal;
        if (std::fabs(n_ref.dot(p_ref.normalized())) < kMinGrazingCos) {
            continue;
        }

        const Sophus::SE3d T_cur_ref = T_c_w * ref->T_c_w.inverse();
        const Eigen::Matrix2d A_cur_ref =
            computeWarpAffine(cam_, T_cur_ref, ref->px, p_ref, n_ref);
        if (std::fabs(A_cur_ref.determinant()) < 1e-3) {
            continue;
        }
        const int search_level = bestSearchLevel(A_cur_ref, cfg_.pyramid_levels - 1);
        const Eigen::Matrix2d A_ref_cur = A_cur_ref.inverse();

        // Warp the reference patch for every pyramid level (dilated sampling).
        SubmapPoint sp;
        sp.pt = cand.pt;
        sp.search_level = search_level;
        sp.ref_patch.resize(cfg_.pyramid_levels * area);
        bool valid = true;
        for (int level = 0; level < cfg_.pyramid_levels && valid; level++) {
            const int scale = 1 << (level + search_level);
            for (int y = 0; y < cfg_.patch_size && valid; y++) {
                for (int x = 0; x < cfg_.patch_size; x++) {
                    const Eigen::Vector2d offset((x - half) * scale, (y - half) * scale);
                    const Eigen::Vector2d px_ref = ref->px + A_ref_cur * offset;
                    if (!ref->img.empty() &&
                        (px_ref.x() < 1 || px_ref.x() >= ref->img.cols - 2 || px_ref.y() < 1 ||
                         px_ref.y() >= ref->img.rows - 2)) {
                        valid = false;
                        break;
                    }
                    sp.ref_patch[level * area + y * cfg_.patch_size + x] =
                        (float)sampleBilinear(ref->img, px_ref);
                }
            }
        }
        if (!valid) {
            continue;
        }

        // Photometric outlier gate at level 0.
        double patch_error = 0.0;
        {
            const int scale = 1 << search_level;
            bool sampled = true;
            for (int y = 0; y < cfg_.patch_size && sampled; y++) {
                for (int x = 0; x < cfg_.patch_size; x++) {
                    const Eigen::Vector2d offset((x - half) * scale, (y - half) * scale);
                    const Eigen::Vector2d px = cand.px + offset;
                    if (px.x() < 1 || px.x() >= gray.cols - 2 || px.y() < 1 ||
                        px.y() >= gray.rows - 2) {
                        sampled = false;
                        break;
                    }
                    const double res =
                        sampleBilinear(gray, px) - sp.ref_patch[y * cfg_.patch_size + x];
                    patch_error += res * res;
                }
            }
            if (!sampled || patch_error > cfg_.outlier_threshold * area) {
                continue;
            }
        }

        tracked_px_.push_back(cand.px);
        submap_.push_back(std::move(sp));
    }
}

bool VisualUpdate::buildResidual(const State &state, const cv::Mat &gray, int level,
                                 double &error) {
    const Sophus::SE3d T_i_w = toSE3(state.pose).inverse();
    const Eigen::Matrix3d R_c_i = T_c_i_.rotationMatrix();
    const int half = cfg_.patch_size / 2;
    const int area = cfg_.patch_size * cfg_.patch_size;

    h_.resize(submap_.size() * area, 1);
    h_x_.resize(submap_.size() * area, 6);

    int row = 0;
    error = 0.0;
    for (const SubmapPoint &sp : submap_) {
        const Eigen::Vector3d p_i = T_i_w * sp.pt->pos;
        const Eigen::Vector3d p_c = T_c_i_ * p_i;
        if (p_c.z() <= kMinDepth) {
            continue;
        }
        const Eigen::Vector2d px = cam_.project(p_c);
        const int scale = 1 << (level + sp.search_level);
        if (!cam_.isInFrame(px, (half + 2) * scale)) {
            continue;
        }

        const Eigen::Matrix<double, 2, 3> J_pi = cam_.projectJacobian(p_c);
        const Eigen::Matrix<double, 2, 6> J_px = J_pi * pointJacobian(R_c_i, p_i);
        const float *ref = sp.ref_patch.data() + level * area;

        for (int y = 0; y < cfg_.patch_size; y++) {
            for (int x = 0; x < cfg_.patch_size; x++) {
                const Eigen::Vector2d offset((x - half) * scale, (y - half) * scale);
                double val;
                Eigen::Vector2d grad;
                if (!sampleWithGradient(gray, px + offset, scale, val, grad)) {
                    continue;
                }
                const double res = val - ref[y * cfg_.patch_size + x];
                h_(row, 0) = -res;
                h_x_.row(row) = grad.transpose() * J_px;
                error += res * res;
                row++;
            }
        }
    }

    if (row < area) {
        return false;
    }
    h_.conservativeResize(row, 1);
    h_x_.conservativeResize(row, 6);
    return true;
}

State VisualUpdate::update(const State &_prior, const cv::Mat &gray,
                           const std::vector<Eigen::Vector3d> &scan_world) {
    // probe: loosen the pose prior so the iteration lands on the camera's own optimum,
    // report it, and hand back the untouched prior (pipeline behaves as "off").
    State prior = _prior;
    if (cfg_.probe) prior.covariance.block<6, 6>(0, 0) *= 1e4;
    retrieveSubmap(prior, gray, scan_world);
    if ((int)submap_.size() < kMinTrackedPoints) {
        if (cfg_.verbose) {
            std::cout << "Visual update skipped: " << submap_.size() << " tracked points\n";
        }
        return prior;
    }

    State updated = prior;
    bool have_gain = false;
    CovMatrix K_x_final;
    CovMatrix new_prior_P_final;
    State lin_state_final;
    double mean_err_final = 0.0;

    for (int level = cfg_.pyramid_levels - 1; level >= 0; level--) {
        double last_mean_err = std::numeric_limits<double>::max();
        for (int iter = 0; iter < cfg_.max_iter; iter++) {
            double error;
            if (!buildResidual(updated, gray, level, error)) {
                break;
            }
            const double mean_err = error / h_.rows();
            if (mean_err > last_mean_err) {
                break;  // diverging at this level: keep the previous iterate
            }
            last_mean_err = mean_err;

            const auto &[dx_prior, J_prior] = resetErrorState(prior, updated);
            const CovMatrix new_prior_P =
                J_prior.inverse() * prior.covariance * J_prior.inverse().transpose();

            const Eigen::Matrix<double, 6, Eigen::Dynamic> HT_Rinv =
                h_x_.transpose() / cfg_.img_point_cov;
            CovMatrix P_temp = new_prior_P.inverse();
            P_temp.template block<6, 6>(0, 0) += HT_Rinv * h_x_;
            Eigen::Matrix<double, ERR_DIM, Eigen::Dynamic> K =
                (P_temp.inverse()).template block<ERR_DIM, 6>(0, 0) * HT_Rinv;
            if (cfg_.pose_only) K.bottomRows(ERR_DIM - 6).setZero();

            const ErrVector K_h = K * h_;
            CovMatrix K_x = CovMatrix::Zero();
            K_x.template block<ERR_DIM, 6>(0, 0) = K * h_x_;

            const ErrVector dx_updated =
                K_h + (K_x - CovMatrix::Identity()) * J_prior.inverse() * dx_prior;

            State new_state = boxplus(updated, dx_updated);
            new_state.stamp = prior.stamp;
            new_state.covariance = updated.covariance;

            have_gain = true;
            K_x_final = K_x;
            new_prior_P_final = new_prior_P;
            lin_state_final = updated;
            mean_err_final = mean_err;

            updated = new_state;

            if (dx_updated.cwiseAbs().maxCoeff() < 0.001) {
                break;
            }
        }
    }

    if (!have_gain) {
        return prior;
    }

    ErrVector dx_f;
    CovMatrix J_updated;
    std::tie(dx_f, J_updated) = resetErrorState(lin_state_final, updated);
    CovMatrix new_P = (CovMatrix::Identity() - K_x_final) * new_prior_P_final;
    new_P = J_updated.inverse() * new_P * J_updated.inverse().transpose();
    updated.covariance = cfg_.keep_cov ? prior.covariance : new_P;
    updated.num_inliers = (int)submap_.size();
    updated.residual = mean_err_final;
    if (cfg_.probe) {
        const ErrVector dx = boxminus(updated, _prior);
        fprintf(stderr, "PROBE %.6f n %d err %.2f dx %.5f %.5f %.5f %.6f %.6f %.6f\n", _prior.stamp,
                (int)submap_.size(), mean_err_final, dx(0), dx(1), dx(2), dx(3), dx(4), dx(5));
        return _prior;
    }

    if (cfg_.verbose) {
        std::cout << "Visual update: " << submap_.size() << " points, mean err " << mean_err_final
                  << "\n";
    }
    return updated;
}

void VisualUpdate::updateMap(const State &state, const cv::Mat &gray,
                             const std::vector<Eigen::Vector3d> &scan_world,
                             const std::unordered_map<VOXEL_LOC, OctoTree *> &lidar_map,
                             double lidar_map_resolution, const cv::Mat &color) {
    const Sophus::SE3d T_c_w = cameraPose(state);
    for (const SubmapPoint &sp : submap_) {
        map_.addObservation(sp.pt, gray, cam_, T_c_w, color);
    }
    map_.generatePoints(gray, cam_, T_c_w, scan_world, lidar_map, lidar_map_resolution,
                        cfg_.grid_size, tracked_px_, color);
}

}  // namespace se3_lio
