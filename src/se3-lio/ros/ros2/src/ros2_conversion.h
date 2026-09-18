#pragma once

// C++ standard libraries
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// ROS2
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// SE3-LIO
#include "common/data_type.h"

struct OusterPointType {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    OusterPointType,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t))

namespace se3_lio::ros2_node {

inline IMU convertIMUMessage(const sensor_msgs::msg::Imu::SharedPtr &_imu_msg) {
    IMU imu;
    rclcpp::Time stamp(_imu_msg->header.stamp);
    imu.header.timestamp = stamp.seconds();
    imu.header.frame_id = _imu_msg->header.frame_id;

    imu.linear_acceleration.x() = _imu_msg->linear_acceleration.x;
    imu.linear_acceleration.y() = _imu_msg->linear_acceleration.y;
    imu.linear_acceleration.z() = _imu_msg->linear_acceleration.z;

    imu.angular_velocity.x() = _imu_msg->angular_velocity.x;
    imu.angular_velocity.y() = _imu_msg->angular_velocity.y;
    imu.angular_velocity.z() = _imu_msg->angular_velocity.z;

    return imu;
};

inline LiDAR convertLivoxMessage(livox_ros_driver2::msg::CustomMsg::SharedPtr _lidar_msg,
                                 double _min_range, int _point_filter_num = 1) {
    LiDAR lidar;
    rclcpp::Time stamp(_lidar_msg->header.stamp);
    lidar.header.timestamp = stamp.seconds();
    lidar.header.frame_id = _lidar_msg->header.frame_id;

    int num_points = _lidar_msg->point_num;
    lidar.points.reserve(num_points);

    CustomPointType last_point;
    last_point.x = last_point.y = last_point.z = 0.0;  // origin: range filter drops it anyway

    for (int i = 0; i < num_points; i += _point_filter_num) {
        if ((_lidar_msg->points[i].tag & 0x30) == 0x10 ||
            (_lidar_msg->points[i].tag & 0x30) == 0x00) {
            // url::PointType point;
            CustomPointType point;
            point.x = _lidar_msg->points[i].x;
            point.y = _lidar_msg->points[i].y;
            point.z = _lidar_msg->points[i].z;

            if (((abs(point.x - last_point.x) > 1e-7) || (abs(point.y - last_point.y) > 1e-7) ||
                 (abs(point.z - last_point.z) > 1e-7)) &&
                (point.x * point.x + point.y * point.y + point.z * point.z >
                 (_min_range * _min_range))) {
                point.intensity = _lidar_msg->points[i].reflectivity;
                point.timestamp = static_cast<double>(_lidar_msg->points[i].offset_time) * 1e-9;

                if (point.timestamp > 0.1) continue;
                lidar.points.push_back(point);
            }
            last_point = point;
        }
    }

    return lidar;
};

inline LiDAR convertOusterMessage(const sensor_msgs::msg::PointCloud2::SharedPtr &_lidar_msg,
                                  double _min_range, int _point_filter_num = 1) {
    LiDAR lidar;
    rclcpp::Time stamp(_lidar_msg->header.stamp);
    lidar.header.timestamp = stamp.seconds();
    lidar.header.frame_id = _lidar_msg->header.frame_id;

    pcl::PointCloud<OusterPointType>::Ptr points_ouster(new pcl::PointCloud<OusterPointType>);
    pcl::fromROSMsg(*_lidar_msg, *points_ouster);

    lidar.points.reserve(points_ouster->size());
    for (size_t i = 0; i < points_ouster->size(); i += _point_filter_num) {
        CustomPointType point;
        point.x = points_ouster->points[i].x;
        point.y = points_ouster->points[i].y;
        point.z = points_ouster->points[i].z;

        if (point.x * point.x + point.y * point.y + point.z * point.z <= (_min_range * _min_range))
            continue;

        point.intensity = points_ouster->points[i].intensity;
        point.timestamp = static_cast<double>(points_ouster->points[i].t) * 1e-9;
        lidar.points.push_back(point);
    }

    return lidar;
};

inline void convertStateToROSOdomMsg(const se3_lio::State &_state,
                                     nav_msgs::msg::Odometry &_odom_msg) {
    Eigen::Quaterniond q(_state.rot());
    _odom_msg.pose.pose.position.x = _state.pos()(0);
    _odom_msg.pose.pose.position.y = _state.pos()(1);
    _odom_msg.pose.pose.position.z = _state.pos()(2);
    _odom_msg.pose.pose.orientation.x = q.x();
    _odom_msg.pose.pose.orientation.y = q.y();
    _odom_msg.pose.pose.orientation.z = q.z();
    _odom_msg.pose.pose.orientation.w = q.w();
}

inline void convertStateToROSPoseStampedMsg(const se3_lio::State &_state,
                                            geometry_msgs::msg::PoseStamped &_pose_msg) {
    Eigen::Quaterniond q(_state.rot());
    _pose_msg.pose.position.x = _state.pos()(0);
    _pose_msg.pose.position.y = _state.pos()(1);
    _pose_msg.pose.position.z = _state.pos()(2);
    _pose_msg.pose.orientation.x = q.x();
    _pose_msg.pose.orientation.y = q.y();
    _pose_msg.pose.orientation.z = q.z();
    _pose_msg.pose.orientation.w = q.w();
};

inline void convertStateToROSTFMsg(const se3_lio::State &_state,
                                   geometry_msgs::msg::TransformStamped &_tf_msg) {
    Eigen::Quaterniond q(_state.rot());
    _tf_msg.transform.translation.x = _state.pos()(0);
    _tf_msg.transform.translation.y = _state.pos()(1);
    _tf_msg.transform.translation.z = _state.pos()(2);
    _tf_msg.transform.rotation.x = q.x();
    _tf_msg.transform.rotation.y = q.y();
    _tf_msg.transform.rotation.z = q.z();
    _tf_msg.transform.rotation.w = q.w();
};

}  // namespace se3_lio::ros2_node
