#pragma once

// C++ standard library
#include <iostream>
#include <mutex>
#include <queue>

// ROS
#include <livox_ros_driver2/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

// SE3-LIO::ROS1 Node
#include "ros1_conversion.h"

// SE3-LIO
#include "common/utils.h"
#include "pipeline/SE3_LIO.h"
#include "synchronizer/measurement_synchronizer.h"

namespace se3_lio::ros1_node {

using IMU_MSG_TYPE = sensor_msgs::Imu::ConstPtr;

// LiDAR input type is selected at build time via CMake (-DLIDAR_TYPE=ouster|hesai|livox|velodyne).
#if defined(LIDAR_LIVOX)
using LIDAR_MSG_TYPE = livox_ros_driver2::CustomMsg::ConstPtr;
#else  // LIDAR_OUSTER / LIDAR_HESAI / LIDAR_VELODYNE (all sensor_msgs::PointCloud2)
using LIDAR_MSG_TYPE = sensor_msgs::PointCloud2::ConstPtr;
#endif

class LioNode {
public:
    LioNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
    ~LioNode();

    void run();

private:
    std::mutex mutex_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    ros::Publisher pub_cloud_body_;

    ros::Subscriber sub_imu_;
    ros::Subscriber sub_lidar_;

    std::string imu_topic_;
    std::string lidar_topic_;

    std::queue<IMU_MSG_TYPE> imu_queue_;
    std::queue<LIDAR_MSG_TYPE> lidar_queue_;

    se3_lio::synchronizer::MeasurementSynchronizer synchronizer_;

    se3_lio::pipeline::SE3_LIO se3_lio_pipeline_;
    se3_lio::pipeline::SE3_LIO_Config se3_lio_config_;

    Eigen::Matrix4d lidar_extrinsic_ = Eigen::Matrix4d::Identity();

    nav_msgs::Path path_;

    double lidar_min_range_ = 0.1;
    int lidar_point_filter_num_ = 1;

    void imuCallback(const IMU_MSG_TYPE &_msg);
    void lidarCallback(const LIDAR_MSG_TYPE &_msg);

    void pubOdometry(ros::Publisher pub_odom,
                     const se3_lio::State &_state,
                     std::string frame_id,
                     std::string child_frame_id);
    void pubPath(ros::Publisher pub_path, const se3_lio::State &_state, std::string frame_id);
    void pubCloud(ros::Publisher pub_cloud, const se3_lio::LiDAR &_lidar, std::string frame_id);

    void broadcastTF(const se3_lio::State &_state,
                     std::string frame_id,
                     std::string child_frame_id);

    void pushAllROSMessages();
};

}  // namespace se3_lio::ros1_node
