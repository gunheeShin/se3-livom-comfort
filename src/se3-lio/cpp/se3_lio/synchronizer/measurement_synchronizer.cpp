#include "measurement_synchronizer.h"

namespace se3_lio::synchronizer {

MeasurementSynchronizer::MeasurementSynchronizer() : running_(true), sync_status_(false) {
    sync_thread_ = std::thread(&MeasurementSynchronizer::synchronizeIMULiDAR, this);
};

MeasurementSynchronizer::~MeasurementSynchronizer() {
    running_ = false;
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }
};

void MeasurementSynchronizer::addIMU(const IMU &_imu) {
    std::lock_guard<std::mutex> lock(mutex_);
    imu_queue_.push_back(_imu);
    last_imu_time_ = _imu.header.timestamp;
};

void MeasurementSynchronizer::addLiDAR(const LiDAR &_lidar) {
    std::lock_guard<std::mutex> lock(mutex_);
    lidar_queue_.push_back(_lidar);
};

MeasurementPtr MeasurementSynchronizer::getSyncedMeasurement() {
    MeasurementPtr measurement_ptr(new Measurement());

    // Guard the whole access: the sync thread push_back()s under the same mutex.
    std::lock_guard<std::mutex> lock(mutex_);

    if (synced_queue_.empty()) {
        measurement_ptr->is_synced = false;
        return measurement_ptr;
    }

    *measurement_ptr = synced_queue_.front();
    measurement_ptr->is_synced = true;
    synced_queue_.pop_front();

    return measurement_ptr;
};

void MeasurementSynchronizer::synchronizeIMULiDAR() {
    Measurement measurement;
    double lidar_end_time = 0;

    while (running_) {
        // Sleep before locking so producers (addIMU/addLiDAR) can acquire the
        // mutex; this also avoids busy-spinning on the early `continue` paths.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Guard concurrent access to imu_queue_/lidar_queue_ shared with the
        // sensor callbacks; released at the end of each loop iteration.
        std::lock_guard<std::mutex> lock(mutex_);

        if (imu_queue_.empty() || lidar_queue_.empty()) {
            continue;
        }

        if (!lidar_pushed_) {
            LiDAR lidar_in = lidar_queue_.front();
            lidar_end_time = lidar_in.header.timestamp + lidar_in.points.back().timestamp;
            measurement.lidar = lidar_in;

            lidar_pushed_ = true;
        }

        if (last_imu_time_ < lidar_end_time) {
            continue;
        }

        if (imu_queue_.front().header.timestamp >= lidar_end_time) {
            // No IMU data before the end of LiDAR scan
            lidar_queue_.pop_front();
            lidar_pushed_ = false;
            continue;
        }

        std::vector<IMU> imu_in;
        while (!imu_queue_.empty() && imu_queue_.front().header.timestamp < lidar_end_time) {
            imu_in.push_back(imu_queue_.front());
            imu_queue_.pop_front();
        }

        if (imu_in.empty()) {
            continue;
        }

        measurement.imu = imu_in;

        lidar_queue_.pop_front();
        lidar_pushed_ = false;

        synced_queue_.push_back(measurement);
    }
}

}  // namespace se3_lio::synchronizer
