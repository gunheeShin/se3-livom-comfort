#include "state_update.h"

#include <iostream>
#include <memory>

namespace se3_lio {

inline std::vector<pointWithCov> transformLocalPointsWithNoise(
    const LiDAR &_cloud,
    const Eigen::Matrix4d &_pose,
    const Eigen::Matrix<double, 6, 6> &_pose_cov) {
    Eigen::Matrix3d rot = _pose.block<3, 3>(0, 0);
    Eigen::Vector3d tls = _pose.block<3, 1>(0, 3);

    std::vector<pointWithCov> pv_list;
    pv_list.reserve(_cloud.points.size());
    for (size_t i = 0; i < _cloud.points.size(); i++) {
        Eigen::Vector3d pt = _cloud.points[i].getVector3fMap().cast<double>();
        Eigen::Matrix3d noise = _cloud.noises[i];

        Eigen::Vector3d transformed_pt = rot * pt + tls;

        Eigen::Matrix<double, 3, 6> J_pt;
        J_pt << rot, -rot * skew_sym(pt);
        Eigen::Matrix3d transformed_noise =
            J_pt * _pose_cov * J_pt.transpose() + rot * noise * rot.transpose();

        // pv.point << pt;
        // pv.cov_lidar = noise;
        // pv.point_world << transformed_pt;
        // pv.cov = transformed_noise;

        pv_list.emplace_back(pt, transformed_pt, noise, transformed_noise);
    }

    return pv_list;
}

}  // namespace se3_lio

namespace se3_lio {

StateUpdate::StateUpdate(int max_iter = 4) : max_iter_(max_iter) {}

void StateUpdate::setManageMapConfig(ManageMapConfig _config) { map_config_ = _config; };

void StateUpdate::setState(State _state) { curr_state_ = _state; };

void StateUpdate::setMeasurement(MeasurementPtr _measurement) { measurement_ = _measurement; };

void StateUpdate::setMap(std::shared_ptr<ManageMap> _map) { map_manager_ = _map; };

bool StateUpdate::isValid() { return num_inliers_ > 5; }

bool StateUpdate::measurementModel(State _state) {
    Eigen::Matrix4d curr_pose = _state.pose;
    Eigen::Matrix<double, 6, 6> curr_pose_cov = _state.pose_cov();

    std::vector<pointWithCov> pv_list =
        transformLocalPointsWithNoise(measurement_->lidar, curr_pose, curr_pose_cov);

    std::vector<ptpl> ptpl_list;
    std::vector<Eigen::Vector3d> non_match_list;
    BuildResidualListOMP(map_manager_->voxel_map_, map_config_.resolution, 3.0,
                         map_config_.max_layer, pv_list, ptpl_list, non_match_list);

    double num_inliers = ptpl_list.size();

    if (num_inliers < 5) {
        if (map_config_.verbose) {
            std::cout << " : No Effective Points! \n";
        }
        return false;
    }

    h_x_.resize(num_inliers, 12);
    h_.resize(num_inliers, 1);
    R_.resize(num_inliers, 1);

    for (int i = 0; i < num_inliers; i++) {
        Eigen::Vector3d pt_world = ptpl_list[i].point_world;
        Eigen::Vector3d norm_vec = ptpl_list[i].normal;
        Eigen::Matrix<double, 1, 3> nT_dot_R = norm_vec.transpose() * _state.rot();

        h_x_.block<1, 3>(i, 0) = nT_dot_R;
        h_x_.block<1, 3>(i, 3) = -nT_dot_R * skew_sym(ptpl_list[i].point);
        double residual = norm_vec.transpose() * pt_world + ptpl_list[i].d;
        h_(i, 0) = -residual;

        Eigen::Matrix<double, 1, 3> J_pt = nT_dot_R;
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = (pt_world - ptpl_list[i].center).transpose();
        J_nq.block<1, 3>(0, 3) = -norm_vec.transpose();

        double residual_cov = 0.0;
        if (!const_r_) residual_cov = J_pt * ptpl_list[i].cov_lidar * J_pt.transpose();
        residual_cov += J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        R_(i, 0) = 1.0 / (residual_floor_ + residual_cov);
    }

    return true;
};

State StateUpdate::updateState() {
    if (!map_manager_->isInitialized()) {
        if (map_config_.verbose) {
            std::cout << "Map is not initialized, so skip the update" << std::endl;
        }
        return curr_state_;
    }

    updated_state_ = curr_state_;
    valid_iter_ = 0;
    num_inliers_ = 0;
    converged_ = true;
    for (int iter_num = -1; iter_num < max_iter_; iter_num++) {
        if (!measurementModel(updated_state_)) {
            continue;
        };

        const auto &[dx_prior, J_prior] = resetErrorState(curr_state_, updated_state_);

        CovMatrix new_prior_P =
            J_prior.inverse() * curr_state_.covariance * J_prior.inverse().transpose();

        auto R_inv = R_.asDiagonal();

        Eigen::Matrix<double, 6, Eigen::Dynamic> HT_Rinv = h_x_.transpose() * R_inv;

        CovMatrix P_temp = new_prior_P.inverse();
        P_temp.template block<6, 6>(0, 0) += HT_Rinv * h_x_;
        Eigen::Matrix<double, ERR_DIM, Eigen::Dynamic> K =
            (P_temp.inverse()).template block<ERR_DIM, 6>(0, 0) * HT_Rinv;

        ErrVector K_h = ErrVector::Zero();
        CovMatrix K_x = CovMatrix::Zero();
        K_h = K * h_;
        K_x.template block<ERR_DIM, 6>(0, 0) = K * h_x_;

        ErrVector dx_updated = K_h + (K_x - CovMatrix::Identity()) * J_prior.inverse() * dx_prior;

        State new_state = boxplus(updated_state_, dx_updated);
        new_state.covariance = updated_state_.covariance;
        double mean_residual = 0.0;
        for (int i = 0; i < R_.rows(); i++) {
            mean_residual += fabs(1.0 / R_(i, 0));
        }
        if (R_.rows() > 0) {
            mean_residual /= R_.rows();
        }
        new_state.residual = mean_residual;
        new_state.num_inliers = R_.rows();

        bool converged_ = true;
        for (int i = 0; i < ERR_DIM; i++) {
            if (fabs(dx_updated(i)) > 0.001) {
                converged_ = false;
                break;
            }
        }

        if (converged_) {
            valid_iter_++;
        }

        if (!valid_iter_ && iter_num == max_iter_ - 2) {
            converged_ = true;
        }

        if (valid_iter_ > 1 || iter_num == max_iter_ - 1) {
            ErrVector dx_updated;
            CovMatrix J_updated;
            std::tie(dx_updated, J_updated) = resetErrorState(updated_state_, new_state);

            CovMatrix new_P = (CovMatrix::Identity() - K_x) * new_prior_P;
            new_P = J_updated.inverse() * new_P * J_updated.inverse().transpose();

            new_state.covariance = new_P;

            num_inliers_ = h_.rows();

            if (map_config_.verbose) {
                std::cout << "State is updated with " << valid_iter_ << " iterations with "
                          << h_.rows() << " inliers" << std::endl;
            }

            return new_state;
        }

        updated_state_ = new_state;
    }

    return updated_state_;
};

void StateUpdate::appendGPSResidual(const State &state) {
    if (!measurement_ || measurement_->gps.header.timestamp <= 0.0 || !has_prev_gps_) {
        return;
    }

    const Eigen::Vector3d gps_pos = measurement_->gps.pose.block<3, 1>(0, 3);
    Eigen::Matrix2d gps_cov = measurement_->gps.covariance.block<2, 2>(0, 0);

    if (gps_cov.isZero(0)) {
        if (map_config_.verbose) {
            std::cout << "GPS covariance is zero, set to default small value." << std::endl;
        }
        gps_cov = Eigen::Matrix2d::Identity() * 0.001;
    }

    // Relative displacement between consecutive GPS measurements vs LIO displacement
    Eigen::Vector2d gps_rel = gps_pos.head<2>() - last_gps_pos_.head<2>();
    Eigen::Vector2d lio_rel = state.pos().head<2>() - last_state_pos_.head<2>();
    Eigen::Vector2d residual_xy = gps_rel - lio_rel;

    int prev_rows = static_cast<int>(h_.rows());
    int new_rows = prev_rows + 2;

    h_.conservativeResize(new_rows, 1);
    h_x_.conservativeResize(new_rows, ERR_DIM);
    R_.conservativeResize(new_rows, 1);

    // X component
    h_(prev_rows, 0) = residual_xy.x();
    h_x_.row(prev_rows).setZero();
    h_x_.block<1, 1>(prev_rows, POS) = Eigen::Matrix<double, 1, 1>::Identity();
    double var_x = gps_cov(0, 0);
    if (var_x <= 0.0) var_x = 1e-3;
    R_(prev_rows, 0) = 1.0 / var_x;

    // Y component
    h_(prev_rows + 1, 0) = residual_xy.y();
    h_x_.row(prev_rows + 1).setZero();
    h_x_.block<1, 1>(prev_rows + 1, POS + 1) = Eigen::Matrix<double, 1, 1>::Identity();
    double var_y = gps_cov(1, 1);
    if (var_y <= 0.0) var_y = 1e-3;
    R_(prev_rows + 1, 0) = 1.0 / var_y;

    // Add global GPS position as well (optional)
    Eigen::Vector3d pos_residual = gps_pos - state.pos();

    prev_rows = static_cast<int>(h_.rows());
    new_rows = prev_rows + 2;

    h_.conservativeResize(new_rows, 1);
    h_x_.conservativeResize(new_rows, ERR_DIM);
    R_.conservativeResize(new_rows, 1);

    // X component
    h_(prev_rows, 0) = pos_residual.x();
    h_x_.row(prev_rows).setZero();
    h_x_.block<1, 1>(prev_rows, POS) = Eigen::Matrix<double, 1, 1>::Identity();
    R_(prev_rows, 0) = 1.0 / (var_x * 10.0);  // downweight global position
    // Y component
    h_(prev_rows + 1, 0) = pos_residual.y();
    h_x_.row(prev_rows + 1).setZero();
    h_x_.block<1, 1>(prev_rows + 1, POS + 1) = Eigen::Matrix<double, 1, 1>::Identity();
    R_(prev_rows + 1, 0) = 1.0 / (var_y * 10.0);  // downweight global position
}

State StateUpdate::updateState_v2() {
    if (!map_manager_->isInitialized()) {
        if (map_config_.verbose) {
            std::cout << "Map is not initialized, so skip the update" << std::endl;
        }
        return curr_state_;
    }

    const bool gps_available = measurement_ && measurement_->gps.header.timestamp > 0.0 &&
                               measurement_->gps.covariance(0, 0) < 10.0;

    updated_state_ = curr_state_;
    valid_iter_ = 0;
    num_inliers_ = 0;
    converged_ = true;
    for (int iter_num = -1; iter_num < max_iter_; iter_num++) {
        if (!measurementModel(updated_state_)) {
            continue;
        }

        if (gps_available) {
            if (map_config_.verbose) {
                std::cout << "Appending GPS residuals to the measurement model." << std::endl;
            }
            appendGPSResidual(updated_state_);
        }

        const auto &[dx_prior, J_prior] = resetErrorState(curr_state_, updated_state_);

        CovMatrix new_prior_P =
            J_prior.inverse() * curr_state_.covariance * J_prior.inverse().transpose();

        auto R_inv = R_.asDiagonal();

        Eigen::Matrix<double, 6, Eigen::Dynamic> HT_Rinv = h_x_.transpose() * R_inv;

        CovMatrix P_temp = new_prior_P.inverse();
        P_temp.template block<6, 6>(0, 0) += HT_Rinv * h_x_;
        Eigen::Matrix<double, ERR_DIM, Eigen::Dynamic> K =
            (P_temp.inverse()).template block<ERR_DIM, 6>(0, 0) * HT_Rinv;

        ErrVector K_h = ErrVector::Zero();
        CovMatrix K_x = CovMatrix::Zero();
        K_h = K * h_;
        K_x.template block<ERR_DIM, 6>(0, 0) = K * h_x_;

        ErrVector dx_updated = K_h + (K_x - CovMatrix::Identity()) * J_prior.inverse() * dx_prior;

        State new_state = boxplus(updated_state_, dx_updated);
        new_state.covariance = updated_state_.covariance;
        double mean_residual = 0.0;
        for (int i = 0; i < R_.rows(); i++) {
            mean_residual += fabs(1.0 / R_(i, 0));
        }
        if (R_.rows() > 0) {
            mean_residual /= R_.rows();
        }
        new_state.residual = mean_residual;
        new_state.num_inliers = h_.rows();

        bool converged_ = true;
        for (int i = 0; i < ERR_DIM; i++) {
            if (fabs(dx_updated(i)) > 0.001) {
                converged_ = false;
                break;
            }
        }

        if (converged_) {
            valid_iter_++;
        }

        if (!valid_iter_ && iter_num == max_iter_ - 2) {
            converged_ = true;
        }

        if (valid_iter_ > 1 || iter_num == max_iter_ - 1) {
            ErrVector dx_updated;
            CovMatrix J_updated;
            std::tie(dx_updated, J_updated) = resetErrorState(updated_state_, new_state);

            CovMatrix new_P = (CovMatrix::Identity() - K_x) * new_prior_P;
            new_P = J_updated.inverse() * new_P * J_updated.inverse().transpose();

            new_state.covariance = new_P;

            num_inliers_ = h_.rows();

            if (map_config_.verbose) {
                std::cout << "State is updated (LiDAR + GPS) with " << valid_iter_
                          << " iterations and " << h_.rows() << " residuals" << std::endl;
            }

            if (gps_available) {
                last_gps_pos_ = measurement_->gps.pose.block<3, 1>(0, 3);
                last_state_pos_ = new_state.pos();
                has_prev_gps_ = true;
            }

            return new_state;
        }

        updated_state_ = new_state;
    }

    if (gps_available) {
        last_gps_pos_ = measurement_->gps.pose.block<3, 1>(0, 3);
        last_state_pos_ = updated_state_.pos();
        has_prev_gps_ = true;
    }

    return updated_state_;
};

}  // namespace se3_lio

// ----------------------------------------------------------------- //
