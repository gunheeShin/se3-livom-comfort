#include "state_predict.h"

#include <algorithm>
#include <iostream>

namespace se3_lio {

StatePredict::StatePredict() {
    measurement_.reset(new Measurement());
    init_cov_ = CovMatrix::Identity() * 0.0001;
    // init_cov_ = CovMatrix::Zero();
    init_cov_(9, 9) = init_cov_(10, 10) = init_cov_(11, 11) = 0.0001;   // bg
    init_cov_(12, 12) = init_cov_(13, 13) = init_cov_(14, 14) = 0.001;  // ba
    init_cov_(15, 15) = init_cov_(16, 16) = 0.00001;                    // gravity

    is_initialized_ = false;
    is_valid_ = false;
    is_init_imu_ = false;
};

void StatePredict::setState(State _state) { init_state_ = _state; };

void StatePredict::setMeasurement(MeasurementPtr _measurement) { measurement_ = _measurement; };

void StatePredict::setIMUNoise(double _gyr_noise,
                               double _acc_noise,
                               double _bg_noise,
                               double _ba_noise) {
    Q_.setZero();
    Q_.block<3, 3>(0, 0).diagonal() << Eigen::Vector3d::Constant(_gyr_noise);
    Q_.block<3, 3>(3, 3).diagonal() << Eigen::Vector3d::Constant(_acc_noise);
    Q_.block<3, 3>(6, 6).diagonal() << Eigen::Vector3d::Constant(_bg_noise);
    Q_.block<3, 3>(9, 9).diagonal() << Eigen::Vector3d::Constant(_ba_noise);
}

void StatePredict::setLiDARNoise(const std::vector<double> &_range_noises,
                                 const std::vector<double> &_angle_noises) {
    range_noises_ = _range_noises;
    angle_noises_ = _angle_noises;
}

bool StatePredict::isValid() { return is_valid_; }

Eigen::Matrix3d StatePredict::getInitGrav() {
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = init_mean_acc_.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = std::atan2(R0(1, 0), R0(0, 0));
    Eigen::AngleAxisd yaw_correction(-yaw, Eigen::Vector3d::UnitZ());
    R0 = yaw_correction.toRotationMatrix() * R0;

    return R0;
};

void StatePredict::initState() {
    if (measurement_->imu.size() == 0) {
        if (verbose_) {
            std::cout << "No IMU data, skip state initialization" << std::endl;
        }
        return;
    }

    if (!is_init_imu_) {
        for (const auto &imu : measurement_->imu) {
            const auto &imu_acc = imu.linear_acceleration;
            const auto &imu_gyr = imu.angular_velocity;

            init_mean_acc_ += (imu_acc - init_mean_acc_) / (imu_count_ + 1);
            init_mean_gyr_ += (imu_gyr - init_mean_gyr_) / (imu_count_ + 1);

            imu_count_++;
        }
    }

    init_state_.bg = init_fl2_ ? Eigen::Vector3d::Zero() : init_mean_gyr_;
    init_state_.grav = -init_mean_acc_ / init_mean_acc_.norm() * 9.81;
    init_state_.covariance = init_fl2_ ? CovMatrix(CovMatrix::Identity() * 0.01) : init_cov_;

    init_state_.stamp = measurement_->imu.back().header.timestamp;

    last_imu_ = measurement_->imu.back();

    InputType input;
    input.gyro = last_imu_.angular_velocity;
    input.acc = last_imu_.linear_acceleration / last_imu_.linear_acceleration.norm() * 9.81;
    inputs_.push_back(input);

    if (imu_count_ > 20) {
        is_init_imu_ = true;
        is_initialized_ = true;
        if (verbose_) {
            std::cout << "StatePredict() initialized" << std::endl;
        }
    }
};

State StatePredict::predictState() {
    if (!is_initialized_) {
        if (verbose_) {
            std::cout << "StatePredict() is not initialized, so initialize the state first"
                      << std::endl;
        }
        initState();
        return init_state_;
    }

    init_state_time_ = init_state_.stamp;
    cloud_beg_time_ = measurement_->lidar.header.timestamp;
    cloud_end_time_ = measurement_->end_time > 0.0
                          ? measurement_->end_time
                          : cloud_beg_time_ + measurement_->lidar.points.back().timestamp;

    State last_state = init_state_;

    pred_states_ = {last_state};
    pred_states_forward_ = {last_state};
    pred_states_backward_ = {last_state};

    inputs_forward_.clear();
    inputs_backward_.clear();

    Eigen::Vector3d angvel_avr, acc_avr;
    double dt, head_time, tail_time;
    Eigen::Matrix<double, 6, 1> twist;

    std::vector<IMU> imu_vec = last_imu_vec_;
    imu_vec.insert(imu_vec.end(), measurement_->imu.begin(), measurement_->imu.end());
    std::sort(imu_vec.begin(), imu_vec.end(),
              [](const IMU &x, const IMU &y) { return (x.header.timestamp < y.header.timestamp); });

    int imu_init_idx;
    for (imu_init_idx = 0; imu_init_idx < imu_vec.size(); imu_init_idx++) {
        if (imu_vec[imu_init_idx].header.timestamp >= init_state_time_) break;
    }

    if (imu_init_idx >= imu_vec.size()) {
        imu_vec.push_back(last_imu_);
    }

    IMU imu_init = imu_vec[imu_init_idx];
    imu_init.header.timestamp = init_state_time_;
    std::vector<IMU> imu_forward_vec = {imu_init};
    std::vector<IMU> imu_backward_vec = {imu_init};  // stamps are reversed

    // Set backward IMUs
    if (cloud_beg_time_ < init_state_time_) {
        int imu_backward_idx = imu_init_idx - 1;

        while (imu_backward_idx >= 0 &&
               imu_vec[imu_backward_idx].header.timestamp > cloud_beg_time_) {
            imu_backward_vec.push_back(imu_vec[imu_backward_idx]);
            imu_backward_idx--;
        }

        if (imu_backward_idx >= 0) {
            IMU imu_temp = imu_vec[imu_backward_idx];
            imu_temp.header.timestamp = cloud_beg_time_;
            imu_backward_vec.push_back(imu_temp);
        } else {
            IMU imu_temp = imu_vec.front();
            imu_temp.header.timestamp = cloud_beg_time_;
            imu_backward_vec.push_back(imu_temp);
        }

        last_state = init_state_;
        F_x_backward_vec_.clear();
        F_w_backward_vec_.clear();

        for (int imu_idx = 0; imu_idx < imu_backward_vec.size() - 1; imu_idx++) {
            IMU imu_head = imu_backward_vec[imu_idx];
            IMU imu_tail = imu_backward_vec[imu_idx + 1];

            head_time = imu_head.header.timestamp;
            tail_time = imu_tail.header.timestamp;

            angvel_avr = 0.5 * (imu_head.angular_velocity + imu_tail.angular_velocity);
            acc_avr = 0.5 * (imu_head.linear_acceleration + imu_tail.linear_acceleration);
            acc_avr = acc_avr * 9.81 / init_mean_acc_.norm();

            InputType input;
            input.gyro = angvel_avr;
            input.acc = acc_avr;

            inputs_backward_.push_back(input);
            inputs_.insert(inputs_.begin(), input);

            State pred_state;
            std::tuple<CovMatrix, MeasCovMatrix> cov_tuple;

            std::tie(pred_state, cov_tuple) =
                predictStateWithCov(last_state, input, tail_time - head_time);
            pred_state.stamp = tail_time;

            F_x_backward_vec_.push_back(std::get<0>(cov_tuple));
            F_w_backward_vec_.push_back(std::get<1>(cov_tuple));

            pred_states_backward_.push_back(pred_state);
            pred_states_.insert(pred_states_.begin(), pred_state);

            last_state = pred_state;
        }
    }

    // Set forward IMUs
    int imu_forward_idx = imu_init_idx;
    while (imu_forward_idx < imu_vec.size() &&
           imu_vec[imu_forward_idx].header.timestamp < cloud_end_time_) {
        imu_forward_vec.push_back(imu_vec[imu_forward_idx]);
        imu_forward_idx++;
    }

    if (imu_forward_idx < imu_vec.size()) {
        IMU imu_temp = imu_vec[imu_forward_idx];
        imu_temp.header.timestamp = cloud_end_time_;
        imu_forward_vec.push_back(imu_temp);
    } else {
        IMU imu_temp = imu_vec.back();
        imu_temp.header.timestamp = cloud_end_time_;
        imu_forward_vec.push_back(imu_temp);
    }

    last_state = init_state_;
    F_x_forward_vec_.clear();
    F_w_forward_vec_.clear();
    for (int imu_idx = 0; imu_idx < imu_forward_vec.size() - 1; imu_idx++) {
        IMU imu_head = imu_forward_vec[imu_idx];
        IMU imu_tail = imu_forward_vec[imu_idx + 1];

        head_time = imu_head.header.timestamp;
        tail_time = imu_tail.header.timestamp;

        angvel_avr = 0.5 * (imu_head.angular_velocity + imu_tail.angular_velocity);
        acc_avr = 0.5 * (imu_head.linear_acceleration + imu_tail.linear_acceleration);
        acc_avr = acc_avr * 9.81 / init_mean_acc_.norm();

        InputType input;
        input.gyro = angvel_avr;
        input.acc = acc_avr;

        inputs_forward_.push_back(input);
        inputs_.push_back(input);

        State pred_state;
        CovTuple cov_tuple;

        std::tie(pred_state, cov_tuple) =
            predictStateWithCov(last_state, input, tail_time - head_time);
        pred_state.stamp = tail_time;

        F_x_forward_vec_.push_back(std::get<0>(cov_tuple));
        F_w_forward_vec_.push_back(std::get<1>(cov_tuple));

        pred_states_forward_.push_back(pred_state);
        pred_states_.push_back(pred_state);

        last_state = pred_state;
    }

    if (verbose_) {
        std::cout << "StatePredict: " << pred_states_.size()
                  << " states predicted, forward: " << pred_states_forward_.size()
                  << ", backward: " << pred_states_backward_.size() << std::endl;
    }

    last_imu_vec_ = measurement_->imu;
    if (last_imu_vec_.size() != 0) last_imu_ = last_imu_vec_.back();
    is_valid_ = true;

    return last_state;
}

std::tuple<State, CovTuple> StatePredict::predictStateWithCov(State last_state,
                                                              InputType _input,
                                                              double dt) {
    Eigen::Vector3d angvel_hat = _input.gyro - last_state.bg;
    Eigen::Vector3d acc_hat =
        _input.acc - last_state.ba + last_state.rot().transpose() * last_state.grav;

    State pred_state = last_state;

    Eigen::Matrix<double, 6, 1> twist;
    twist.block<3, 1>(0, 0) = last_state.vel;
    twist.block<3, 1>(3, 0) = angvel_hat;

    pred_state.pose = last_state.pose * Sophus::SE3d::exp(twist * dt).matrix();
    pred_state.vel = Sophus::SO3d::exp(-angvel_hat * dt).matrix() * (last_state.vel) +
                     acc_hat * dt;  // Should be updated

    // Covariance
    CovMatrix F_x = CovMatrix::Identity();

    F_x.block<6, 6>(POSE, POSE) = Sophus::SE3d::exp(twist * dt).Adj().matrix().inverse();
    Eigen::Matrix<double, 6, 6> rightJacobian = Sophus::SE3d::leftJacobian(-twist * dt);
    F_x.block<6, 3>(POSE, VEL) = rightJacobian.block<6, 3>(0, 0) * dt;
    F_x.block<6, 3>(POSE, B_G) = -rightJacobian.block<6, 3>(0, 3) * dt;

    F_x.block<3, 3>(VEL, ROT) = skew_sym(last_state.rot().transpose() * last_state.grav) * dt;
    F_x.block<3, 3>(VEL, VEL) = Sophus::SO3d::exp(-angvel_hat * dt).matrix();
    F_x.block<3, 3>(VEL, B_G) = -Sophus::SO3d::exp(-angvel_hat * dt).matrix() *
                                skew_sym(last_state.vel) *
                                Sophus::SO3d::leftJacobian(angvel_hat * dt) * dt;
    F_x.block<3, 3>(VEL, B_A) = -Eigen::Matrix3d::Identity() * dt;
    F_x.block<3, 2>(VEL, GRAV) = last_state.rot().transpose() *
                                 S2_Mx(last_state.grav, Eigen::Vector2d::Zero()) * dt;

    MeasCovMatrix F_w = MeasCovMatrix::Zero();
    F_w.block<6, 3>(POSE, N_GYRO) = -rightJacobian.block<6, 3>(0, 3) * dt;
    F_w.block<3, 3>(VEL, N_GYRO) = -Sophus::SO3d::exp(-angvel_hat * dt).matrix() *
                                   skew_sym(last_state.vel) *
                                   Sophus::SO3d::leftJacobian(angvel_hat * dt) * dt;
    F_w.block<3, 3>(VEL, N_ACC) = -Eigen::Matrix3d::Identity() * dt;
    F_w.block<3, 3>(B_G, N_B_GYRO) = Eigen::Matrix3d::Identity() * dt;
    F_w.block<3, 3>(B_A, N_B_ACC) = Eigen::Matrix3d::Identity() * dt;

    pred_state.covariance =
        F_x * last_state.covariance * F_x.transpose() + F_w * Q_ * F_w.transpose();

    return std::make_tuple(pred_state, std::make_tuple(F_x, F_w));
}

void StatePredict::undistortCloud() {
    if (!is_valid_) return;

    State last_state = pred_states_.back();
    Eigen::Matrix4d last_pose = last_state.pose;

    LiDAR undist_cloud;
    undist_cloud.header = measurement_->lidar.header;
    undist_cloud.points.reserve(measurement_->lidar.points.size());

    auto it_pcl = measurement_->lidar.points.end() - 1;
    for (int state_idx = pred_states_.size() - 1; state_idx > 0; state_idx--) {
        State head_state = pred_states_[state_idx - 1];
        State tail_state = pred_states_[state_idx];
        InputType input = inputs_[state_idx];

        Eigen::Matrix4d head_pose = head_state.pose;
        Eigen::Vector3d head_vel = head_state.vel;
        Eigen::Vector3d tail_gyro = input.gyro - tail_state.bg;
        Eigen::Vector3d tail_acc =
            input.acc - tail_state.ba +
            tail_state.rot().transpose() * tail_state.grav;  // Should be updated

        double head_offset = head_state.stamp - cloud_beg_time_;

        for (; it_pcl->timestamp >= head_offset; it_pcl--) {
            double dt = it_pcl->timestamp - head_offset;

            Eigen::Matrix<double, 6, 1> twist;
            twist.block<3, 1>(0, 0) = head_vel + 0.5 * tail_acc * dt;
            twist.block<3, 1>(3, 0) = tail_gyro;
            Eigen::Matrix4d lidar_pose = head_pose * Sophus::SE3d::exp(twist * dt).matrix();
            Eigen::Matrix4f rel_pose = (last_pose.inverse() * lidar_pose).cast<float>();

            // int lid_idx = static_cast<int>(it_pcl->intensity);
            Eigen::Vector4f distorted_pt = it_pcl->getVector4fMap();
            distorted_pt[3] = 1.0f;
            Eigen::Vector4f undistorted_pt = rel_pose * distorted_pt;

            it_pcl->x = undistorted_pt[0];
            it_pcl->y = undistorted_pt[1];
            it_pcl->z = undistorted_pt[2];

            undist_cloud.points.push_back(*it_pcl);

            if (it_pcl == measurement_->lidar.points.begin()) {
                break;
            }
        }
    }

    // flip the undistorted cloud
    std::reverse(undist_cloud.points.begin(), undist_cloud.points.end());
    measurement_->lidar.points = undist_cloud.points;

    measurement_->lidar.header.timestamp = last_state.stamp;
    return;
}

std::vector<std::tuple<Eigen::Matrix4d, Eigen::Matrix<double, 6, 6>>>
StatePredict::calculateRelPoseWithCov() {
    // Should be set to parameters
    constexpr int state_dim = ERR_DIM;
    constexpr int noise_dim = 12;
    constexpr int pose_dim = 6;

    State last_state = pred_states_.back();
    Eigen::Matrix4d last_pose = last_state.pose;

    CovMatrix init_cov = init_state_.covariance;

    std::vector<std::tuple<Eigen::Matrix4d, Eigen::Matrix<double, 6, 6>>> rel_poses_with_cov;

    // Backward preparation
    int backward_state_num = pred_states_backward_.size();  // k+1
    int backward_input_num = inputs_backward_.size();       // k
    if (backward_state_num != backward_input_num + 1) {
        if (verbose_) {
            std::cout << "StatePredict: backward state num and input num mismatch, "
                      << backward_state_num << " vs " << backward_input_num << std::endl;
        }
        return {};
    }

    Eigen::MatrixXd F_backward = Eigen::MatrixXd::Zero(pose_dim * backward_state_num,
                                                       state_dim + noise_dim * backward_input_num);
    CovMatrix F_x_from_init = CovMatrix::Identity();
    F_backward.block<pose_dim, state_dim>(0, 0) = F_x_from_init.block<pose_dim, state_dim>(0, 0);

    for (int input_idx = 0; input_idx < backward_input_num; input_idx++) {
        F_x_from_init = F_x_backward_vec_[input_idx] * F_x_from_init;
        F_backward.block<pose_dim, state_dim>(pose_dim * (input_idx + 1), 0) =
            F_x_from_init.block<pose_dim, state_dim>(0, 0);
    }

    for (int input_idx = 0; input_idx < backward_input_num; input_idx++) {  // goes column direction
        MeasCovMatrix F_w_from_init = F_w_backward_vec_[input_idx];

        F_backward.block<pose_dim, noise_dim>(pose_dim * (input_idx + 1),
                                              state_dim + input_idx * noise_dim) =
            F_w_from_init.block<pose_dim, noise_dim>(0, 0);

        for (int input_idx_tmp = input_idx + 1; input_idx_tmp < backward_input_num;
             input_idx_tmp++) {  // goes row direction
            F_w_from_init = F_x_backward_vec_[input_idx_tmp] * F_w_from_init;
            F_backward.block<pose_dim, noise_dim>(pose_dim * (input_idx_tmp + 1),
                                                  state_dim + input_idx * noise_dim) =
                F_w_from_init.block<pose_dim, noise_dim>(0, 0);
        }
    }

    // Forward preparation
    int forward_state_num = pred_states_forward_.size();  // k+1
    int forward_input_num = inputs_forward_.size();       // k
    if (forward_state_num != forward_input_num + 1) {
        if (verbose_) {
            std::cout << "StatePredict: forward state num and input num mismatch, "
                      << forward_state_num << " vs " << forward_input_num << std::endl;
        }
        return {};
    }

    Eigen::MatrixXd F_forward = Eigen::MatrixXd::Zero(pose_dim * forward_state_num,
                                                      state_dim + noise_dim * forward_input_num);

    F_x_from_init = CovMatrix::Identity();
    F_forward.block<pose_dim, state_dim>(0, 0) = F_x_from_init.block<pose_dim, state_dim>(0, 0);

    for (int input_idx = 0; input_idx < forward_input_num; input_idx++) {
        F_x_from_init = F_x_forward_vec_[input_idx] * F_x_from_init;
        F_forward.block<pose_dim, state_dim>(pose_dim * (input_idx + 1), 0) =
            F_x_from_init.block<pose_dim, state_dim>(0, 0);
    }

    for (int input_idx = 0; input_idx < forward_input_num; input_idx++) {  // goes column direction
        MeasCovMatrix F_w_from_init = F_w_forward_vec_[input_idx];

        F_forward.block<pose_dim, noise_dim>(pose_dim * (input_idx + 1),
                                             state_dim + input_idx * noise_dim) =
            F_w_from_init.block<pose_dim, noise_dim>(0, 0);

        for (int input_idx_tmp = input_idx + 1; input_idx_tmp < forward_input_num;
             input_idx_tmp++) {  // goes row direction
            F_w_from_init = F_x_forward_vec_[input_idx_tmp] * F_w_from_init;
            F_forward.block<pose_dim, noise_dim>(pose_dim * (input_idx_tmp + 1),
                                                 state_dim + input_idx * noise_dim) =
                F_w_from_init.block<pose_dim, noise_dim>(0, 0);
        }
    }

    // Calculate relative poses with covariance, backward first
    for (int state_idx = 1; state_idx < backward_state_num; state_idx++) {
        Eigen::Matrix4d rel_pose = last_pose.inverse() * pred_states_backward_[state_idx].pose;
        Eigen::Matrix<double, 6, 6> rel_inv_adj = Sophus::SE3d(rel_pose.inverse()).Adj().matrix();

        Eigen::Matrix<double, pose_dim, state_dim> F_x =
            F_backward.block<pose_dim, state_dim>(pose_dim * state_idx, 0) -
            rel_inv_adj *
                F_forward.block<pose_dim, state_dim>(pose_dim * (forward_state_num - 1), 0);

        Eigen::Matrix<double, 6, 6> rel_pose_cov = F_x * init_cov * F_x.transpose();
        for (int input_idx = 0; input_idx < backward_input_num; input_idx++) {
            Eigen::Matrix<double, pose_dim, noise_dim> F_w = F_backward.block<pose_dim, noise_dim>(
                pose_dim * state_idx, state_dim + input_idx * noise_dim);

            rel_pose_cov += F_w * Q_ * F_w.transpose();
        }

        for (int input_idx = 0; input_idx < forward_input_num;
             input_idx++) {  // goes column direction
            Eigen::Matrix<double, pose_dim, noise_dim> F_w = F_forward.block<pose_dim, noise_dim>(
                pose_dim * (forward_state_num - 1), state_dim + input_idx * noise_dim);

            rel_pose_cov += rel_inv_adj * F_w * Q_ * F_w.transpose() * rel_inv_adj.transpose();
        }

        rel_poses_with_cov.insert(rel_poses_with_cov.begin(),
                                  std::make_tuple(rel_pose, rel_pose_cov));
    }

    // Then forward
    for (int state_idx = 0; state_idx < forward_state_num - 1; state_idx++) {
        Eigen::Matrix4d rel_pose = last_pose.inverse() * pred_states_forward_[state_idx].pose;
        Eigen::Matrix<double, 6, 6> rel_inv_adj = Sophus::SE3d(rel_pose.inverse()).Adj().matrix();

        Eigen::Matrix<double, pose_dim, state_dim> F_x =
            F_forward.block<pose_dim, state_dim>(pose_dim * state_idx, 0) -
            rel_inv_adj *
                F_forward.block<pose_dim, state_dim>(pose_dim * (forward_state_num - 1), 0);

        Eigen::Matrix<double, 6, 6> rel_pose_cov = F_x * init_cov * F_x.transpose();

        for (int input_idx = 0; input_idx < forward_input_num; input_idx++) {
            Eigen::Matrix<double, pose_dim, noise_dim> F_w =
                F_forward.block<pose_dim, noise_dim>(pose_dim * state_idx,
                                                     state_dim + input_idx * noise_dim) -
                rel_inv_adj *
                    F_forward.block<pose_dim, noise_dim>(pose_dim * (forward_state_num - 1),
                                                         state_dim + input_idx * noise_dim);
            rel_pose_cov += F_w * Q_ * F_w.transpose();
        }

        rel_poses_with_cov.push_back(std::make_tuple(rel_pose, rel_pose_cov));
    }

    rel_poses_with_cov.push_back(
        std::make_tuple(Eigen::Matrix4d::Identity(), Eigen::Matrix<double, 6, 6>::Zero()));

    return rel_poses_with_cov;
}

void StatePredict::calculateUndistCloudCov(LiDAR &_measurement) {
    State last_state = pred_states_.back();
    Eigen::Matrix4d last_pose = last_state.pose;

    std::vector<std::tuple<Eigen::Matrix4d, Eigen::Matrix<double, 6, 6>>> rel_poses_with_cov =
        calculateRelPoseWithCov();

    std::sort(_measurement.points.begin(), _measurement.points.end(),
              [](const CustomPointType &x, const CustomPointType &y) {
                  return (x.timestamp < y.timestamp);
              });

    std::vector<Eigen::Matrix3d> pt_noises;
    pt_noises.resize(_measurement.points.size(), Eigen::Matrix3d::Zero());
    int pt_idx = 0;

    for (int state_idx = 0; state_idx < pred_states_.size() - 1; state_idx++) {
        State head_state = pred_states_[state_idx];
        State tail_state = pred_states_[state_idx + 1];

        Eigen::Matrix3d head_rot = head_state.rot();
        InputType input = inputs_[state_idx + 1];
        Eigen::Vector3d angvel_hat = input.gyro - tail_state.bg;

        Eigen::Matrix4d rel_pose = std::get<0>(rel_poses_with_cov[state_idx]);
        Eigen::Matrix<double, 6, 6> rel_pose_cov = std::get<1>(rel_poses_with_cov[state_idx]);

        double head_offset = head_state.stamp - cloud_beg_time_;

        double tail_offset = tail_state.stamp - cloud_beg_time_;

        for (; pt_idx < _measurement.points.size(); pt_idx++) {
            auto it_pcl = _measurement.points.begin() + pt_idx;
            if (it_pcl->timestamp > tail_offset) {
                break;
            }

            Eigen::Vector3d pt = it_pcl->getVector3fMap().cast<double>();
            // intensity carries the LiDAR index for merged scans; clamped so a
            // single-LiDAR cloud with real intensities still maps to entry 0.
            int lidar_idx = std::max(0, static_cast<int>(it_pcl->intensity));
            Eigen::Matrix3d pt_noise = calculateMeasurementNoise(
                pt, range_noises_[std::min<size_t>(lidar_idx, range_noises_.size() - 1)],
                angle_noises_[std::min<size_t>(lidar_idx, angle_noises_.size() - 1)]);

            double dt = it_pcl->timestamp - head_offset;

            Eigen::Matrix3d pt_rel_rot =
                rel_pose.block<3, 3>(0, 0) * Sophus::SO3d::exp(angvel_hat * dt).matrix();

            Eigen::Matrix<double, 3, 6> pose_jaco = Eigen::Matrix<double, 3, 6>::Zero();
            pose_jaco.block<3, 3>(0, 0) = pt_rel_rot;
            pose_jaco.block<3, 3>(0, 3) = -skew_sym(pt) * Sophus::SO3d(pt_rel_rot).Adj().matrix();
            if (deskew_cov_en_) pt_noise += pose_jaco * rel_pose_cov * pose_jaco.transpose();

            pt_noises[pt_idx] = pt_noise;

            if (it_pcl == _measurement.points.end() - 1) {
                break;
            }
        }
    }

    _measurement.noises = pt_noises;
}

void StatePredict::reset() {
    init_state_ = State();
    measurement_.reset(new Measurement());
    pred_states_.clear();
    inputs_.clear();
    F_x_vec_.clear();
    F_w_vec_.clear();

    is_valid_ = false;
    is_initialized_ = false;

    // init_cov_acc_.setZero();
    // init_cov_gyr_.setZero();
    // init_cov_ = CovMatrix::Zero();
    init_cov_ = CovMatrix::Identity() * 0.0001;
    init_cov_(9, 9) = init_cov_(10, 10) = init_cov_(11, 11) = 0.0001;   // bg
    init_cov_(12, 12) = init_cov_(13, 13) = init_cov_(14, 14) = 0.001;  // ba
    init_cov_(15, 15) = init_cov_(16, 16) = 0.00001;                    // gravity
};

}  // namespace se3_lio