#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <iomanip>
#include <mutex>
#include <string>
#include <thread>

#include "common/data_type.h"

namespace se3_lio::synchronizer {

class MeasurementSynchronizer {
public:
    MeasurementSynchronizer();
    ~MeasurementSynchronizer();

    void addIMU(const IMU &_imu);
    void addLiDAR(const LiDAR &_lidar);

    MeasurementPtr getSyncedMeasurement();

    void pause() { pause_ = true; }
    void resume() { pause_ = false; }

private:
    std::mutex mutex_;
    std::thread sync_thread_;
    std::atomic<bool> running_;

    bool pause_ = false;

    std::deque<IMU> imu_queue_;
    std::deque<LiDAR> lidar_queue_;

    std::deque<Measurement> synced_queue_;

    int sync_status_;
    bool lidar_pushed_ = false;

    double last_imu_time_ = 0.0;

    void synchronizeIMULiDAR();
};

}  // namespace se3_lio::synchronizer
