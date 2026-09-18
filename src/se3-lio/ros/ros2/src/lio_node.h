#pragma once

// C++ standard library
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// ROS
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "tf2_ros/transform_broadcaster.h"

// SE3-LIO::ROS1 Node
#include "ros2_conversion.h"

// SE3-LIO
#include "common/utils.h"
#include "pipeline/SE3_LIO.h"
#include "synchronizer/measurement_synchronizer.h"

namespace se3_lio::ros2_node {

using IMU_MSG_TYPE = sensor_msgs::msg::Imu;

// LiDAR input type is selected at build time via CMake (-DLIDAR_TYPE=ouster|livox).
#if defined(LIDAR_LIVOX)
using LIDAR_MSG_TYPE = livox_ros_driver2::msg::CustomMsg;
#else  // LIDAR_OUSTER
using LIDAR_MSG_TYPE = sensor_msgs::msg::PointCloud2;
#endif

class LioNode : public rclcpp::Node {
public:
    LioNode();
    ~LioNode();

    void run();

private:
    std::mutex mutex_;
    std::thread thread_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_;

    rclcpp::Subscription<IMU_MSG_TYPE>::SharedPtr sub_imu_;
    rclcpp::Subscription<LIDAR_MSG_TYPE>::SharedPtr sub_lidar_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::queue<IMU_MSG_TYPE::SharedPtr> imu_queue_;
    std::queue<LIDAR_MSG_TYPE::SharedPtr> lidar_queue_;

    se3_lio::synchronizer::MeasurementSynchronizer synchronizer_;

    se3_lio::pipeline::SE3_LIO se3_lio_pipeline_;
    se3_lio::pipeline::SE3_LIO_Config se3_lio_config_;

    Eigen::Matrix4d lidar_extrinsic_ = Eigen::Matrix4d::Identity();

    nav_msgs::msg::Path path_;

    double lidar_min_range_ = 0.1;
    int lidar_point_filter_num_ = 1;

    bool keep_process_ = false;

    void imuCallback(const IMU_MSG_TYPE::SharedPtr _msg);
    void lidarCallback(const LIDAR_MSG_TYPE::SharedPtr _msg);

    void pubOdometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom,
                     const se3_lio::State &_state,
                     std::string frame_id,
                     std::string child_frame_id);
    void pubPath(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path,
                 const se3_lio::State &_state,
                 std::string frame_id);
    void pubCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud,
                  const se3_lio::LiDAR &_lidar,
                  std::string frame_id);

    void broadcastTF(const se3_lio::State &_state,
                     std::string frame_id,
                     std::string child_frame_id);

    void pushAllROSMessages();

    void process();
};

}  // namespace se3_lio::ros2_node
