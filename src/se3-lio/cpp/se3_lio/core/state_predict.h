#ifndef STATE_PREDICT_H
#define STATE_PREDICT_H

// Eigen library
#include <Eigen/Dense>

// Common
#include "common/data_type.h"

// Core classes
#include "core/lie.h"

namespace se3_lio {

struct InputType {
    Eigen::Vector3d gyro;
    Eigen::Vector3d acc;
};

using CovTuple = std::tuple<CovMatrix, MeasCovMatrix>;

class StatePredict {
public:
    StatePredict();
    ~StatePredict(){};

    /**
     * @brief  Reset the StatePredict
     *
     */
    void reset();

    void setIMUNoise(double _gyr_noise, double _acc_noise, double _bg_noise, double _ba_noise);

    void setLiDARNoise(const std::vector<double> &_range_noises,
                       const std::vector<double> &_angle_noises);

    void setVerbose(bool _verbose) { verbose_ = _verbose; }
    void setDeskewCovEnabled(bool _en) { deskew_cov_en_ = _en; }
    void setInitFL2(bool _en) { init_fl2_ = _en; }

    /**
     * @brief Set the State
     *
     * @param _state input state
     */
    void setState(State _state);

    /**
     * @brief Set the Measurement
     *
     *  @param _measurement input measurement
     */
    void setMeasurement(MeasurementPtr _measurement);

    /**
     * @brief Check if the State is initialized
     *
     * @return bool
     */
    bool isValid();

    /**
     * @brief Calculate the initial gravity from the mean values of the
     * accelerometer
     *
     * @return Gravity rotation matrix
     */
    Eigen::Matrix3d getInitGrav();

    /**
     * @brief Predict the final State and Deskew the LiDAR point cloud
     *
     * @return predicted State and undistorted point cloud
     */
    State predictState();

    std::tuple<State, CovTuple> predictStateWithCov(State last_state, InputType _input, double dt);

    /**
     * @brief Initialize the State
     *
     */
    void initState();

    /**
     * @brief Undisotort the LiDAR point cloud (backward propagation & motion
     * compensation)
     *
     * @return undistorted point cloud
     */
    void undistortCloud();

    // ----------------------------------------------------------------- //
    /**
     * @brief Calculate the relative pose between last predicted pose and the
     * ealier predicted poses
     *
     * @return relative pose and covariance
     */
    std::vector<std::tuple<Eigen::Matrix4d, Eigen::Matrix<double, 6, 6>>> calculateRelPoseWithCov();

    /**
     * @brief Calculate the point covariance of the undistorted point cloud
     *
     * @param _measuremnt input measurement
     */
    void calculateUndistCloudCov(LiDAR &_measuremnt);

private:
    MeasurementPtr measurement_;

    State init_state_;

    // LiDAR undistorted_cloud_;

    std::vector<State> pred_states_;
    std::vector<State> pred_states_forward_;
    std::vector<State> pred_states_backward_;

    std::vector<InputType> inputs_;
    std::vector<InputType> inputs_forward_;
    std::vector<InputType> inputs_backward_;

    std::vector<CovMatrix> F_x_vec_;
    std::vector<MeasCovMatrix> F_w_vec_;

    std::vector<CovMatrix> F_x_forward_vec_;
    std::vector<MeasCovMatrix> F_w_forward_vec_;
    std::vector<CovMatrix> F_x_backward_vec_;
    std::vector<MeasCovMatrix> F_w_backward_vec_;

    bool verbose_ = false;
    bool deskew_cov_en_ = true;
    bool init_fl2_ = false;

    int imu_count_ = 0;
    bool is_initialized_ = false;
    bool is_valid_ = false;
    bool is_init_imu_ = false;
    IMU last_imu_;
    std::vector<IMU> last_imu_vec_ = {};

    double init_state_time_ = 0;
    double cloud_beg_time_ = 0;
    double cloud_end_time_ = 0;

    Eigen::Matrix<double, 12, 12> Q_;

    std::vector<double> range_noises_;
    std::vector<double> angle_noises_;

    Eigen::Vector3d init_mean_acc_ = Eigen::Vector3d::Zero();  // running means over the init window; the first sample is
    Eigen::Vector3d init_mean_gyr_ = Eigen::Vector3d::Zero();  // added as x += (a - x), so an uninitialized x leaks into the result
    Eigen::Vector3d init_cov_acc_;
    Eigen::Vector3d init_cov_gyr_;

    CovMatrix init_cov_;
};

}  // namespace se3_lio

#endif  // STATE_PREDICT_H