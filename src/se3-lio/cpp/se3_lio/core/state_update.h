#ifndef STATE_UPDATE_H
#define STATE_UPDATE_H

// Eigen library
#include <Eigen/Dense>

#include <memory>

// common
#include "common/data_type.h"

// core
#include "core/lie.h"
#include "core/map_management.h"

namespace se3_lio {

class StateUpdate {
public:
    explicit StateUpdate(int max_iter);
    StateUpdate() : map_config_() {}
    ~StateUpdate(){};

    void setManageMapConfig(ManageMapConfig _config);
    void setResidualFloor(double _floor) { residual_floor_ = _floor; }
    void setConstR(bool _en) { const_r_ = _en; }

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
     * @brief Set the Map
     *
     * @param _map input map
     */
    void setMap(std::shared_ptr<ManageMap> _map);

    /**
     * @brief Get the measurement distribution
     *
     * @param _state current updated state
     * @return true if the measurement model is valid
     */
    bool measurementModel(State _state);

    /**
     * @brief Opitimize the prior distribution with the measurement distribution
     * to update the state
     *
     * @return optimal state
     */
    State updateState();

    /**
     * @brief LiDAR update followed by GPS correction in one call
     *
     * @return optimal state after LiDAR + GPS
     */
    State updateState_v2();

    bool isValid();

    // --------------------------------------------------------------- //

private:
    ManageMapConfig map_config_;

    void appendGPSResidual(const State &state);

    State curr_state_;
    State updated_state_;

    MeasurementPtr measurement_;
    // LiDAR undistorted_cloud_;

    std::shared_ptr<ManageMap> map_manager_;
    std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map_;

    bool converged_ = false;
    int valid_iter_ = 0;
    int number_of_iterations_ = 0;

    int num_inliers_ = 0;

    double residual_floor_ = 0.0;
    bool const_r_ = false;
    int max_iter_ = 4;

    Eigen::Matrix<double, Eigen::Dynamic, 1> h_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R_;

    // For relative GPS residuals
    bool has_prev_gps_ = false;
    Eigen::Vector3d last_gps_pos_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_state_pos_ = Eigen::Vector3d::Zero();
};

}  // namespace se3_lio

#endif  // STATE_UPDATE_H
