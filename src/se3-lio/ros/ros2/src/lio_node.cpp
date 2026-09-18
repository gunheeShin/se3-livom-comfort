#include "lio_node.h"

namespace se3_lio::ros2_node {

LioNode::LioNode() : Node("se3_lio_node") {
    std::string imu_topic, lidar_topic;
    this->declare_parameter<std::string>("imu_topic", "/imu/data");
    this->declare_parameter<std::string>("lidar_topic", "/lidar/points");

    this->get_parameter("imu_topic", imu_topic);
    this->get_parameter("lidar_topic", lidar_topic);

    // Sensor streams (IMU/LiDAR) are typically published best_effort — and so is
    // `ros2 bag play` for such topics. A reliable subscriber would not match a
    // best_effort publisher (no data arrives), so subscribe best_effort here.
    // Keep a deep IMU queue for the high-rate stream.
    auto qos_imu = rclcpp::QoS(rclcpp::KeepLast(10000)).best_effort();
    auto qos_best_effort = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    auto qos_reliable = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();  // node outputs (odom/path/cloud)

    sub_imu_ = this->create_subscription<IMU_MSG_TYPE>(
        imu_topic, qos_imu, std::bind(&LioNode::imuCallback, this, std::placeholders::_1));

    sub_lidar_ = this->create_subscription<LIDAR_MSG_TYPE>(
        lidar_topic, qos_best_effort, std::bind(&LioNode::lidarCallback, this, std::placeholders::_1));

    this->declare_parameter<double>("sensors.imu.acc_cov", 0.1);
    this->declare_parameter<double>("sensors.imu.gyr_cov", 0.1);
    this->declare_parameter<double>("sensors.imu.b_acc_cov", 0.0001);
    this->declare_parameter<double>("sensors.imu.b_gyr_cov", 0.0001);

    this->get_parameter("sensors.imu.acc_cov", se3_lio_config_.acc_noise);
    this->get_parameter("sensors.imu.gyr_cov", se3_lio_config_.gyr_noise);
    this->get_parameter("sensors.imu.b_acc_cov", se3_lio_config_.bg_noise);
    this->get_parameter("sensors.imu.b_gyr_cov", se3_lio_config_.ba_noise);

    this->declare_parameter<double>("sensors.lidar.min_range", 0.1);
    this->declare_parameter<double>("sensors.lidar.range_cov", 0.001);
    this->declare_parameter<double>("sensors.lidar.angle_cov", 0.01);
    this->declare_parameter<int>("sensors.lidar.point_filter_num", 1);

    this->get_parameter("sensors.lidar.min_range", lidar_min_range_);
    this->get_parameter("sensors.lidar.range_cov", se3_lio_config_.lidar_range_noise);
    this->get_parameter("sensors.lidar.angle_cov", se3_lio_config_.lidar_angle_noise);
    this->get_parameter("sensors.lidar.point_filter_num", lidar_point_filter_num_);
    if (lidar_point_filter_num_ < 1) lidar_point_filter_num_ = 1;

    std::vector<double> lidar_extrinsic_t, lidar_extrinsic_q;
    this->declare_parameter<std::vector<double>>("sensors.t_exts", {0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>("sensors.q_exts", {1.0, 0.0, 0.0, 0.0});

    this->get_parameter("sensors.t_exts", lidar_extrinsic_t);
    this->get_parameter("sensors.q_exts", lidar_extrinsic_q);

    this->declare_parameter<double>("downsample.resolution", 0.5);
    this->declare_parameter<int>("max_iter", 4);

    this->get_parameter("downsample.resolution", se3_lio_config_.downsample_resolution);
    this->get_parameter("max_iter", se3_lio_config_.max_iter);

    this->declare_parameter<bool>("verbose", false);
    this->get_parameter("verbose", se3_lio_config_.verbose);

    this->declare_parameter<double>("voxel_map.resolution", 1.0);
    this->declare_parameter<int>("voxel_map.max_layer", 2);
    this->declare_parameter<std::vector<int>>("voxel_map.layer_size", {5, 5, 5, 5, 5});
    this->declare_parameter<int>("voxel_map.max_point_size", 1000);
    this->declare_parameter<float>("voxel_map.plane_threshold", 0.01f);
    this->declare_parameter<bool>("voxel_map.map_sliding_en", false);
    this->declare_parameter<double>("voxel_map.sliding_thresh", 8.0);
    this->declare_parameter<int>("voxel_map.half_map_size", 50);

    std::vector<int64_t> temp_layer_size;
    this->get_parameter("voxel_map.resolution", se3_lio_config_.voxel_map_resolution);
    this->get_parameter("voxel_map.max_layer", se3_lio_config_.voxel_map_max_layer);
    this->get_parameter("voxel_map.layer_size", temp_layer_size);
    this->get_parameter("voxel_map.max_point_size", se3_lio_config_.voxel_map_max_point_size);
    this->get_parameter("voxel_map.plane_threshold", se3_lio_config_.voxel_map_plane_thres);
    this->get_parameter("voxel_map.map_sliding_en", se3_lio_config_.voxel_map_sliding_en);
    this->get_parameter("voxel_map.sliding_thresh", se3_lio_config_.voxel_map_sliding_thresh);
    this->get_parameter("voxel_map.half_map_size", se3_lio_config_.voxel_map_half_size);

    for (const auto &size : temp_layer_size) {
        se3_lio_config_.voxel_map_layer_size.push_back(static_cast<int>(size));
    }

    lidar_extrinsic_.block<3, 1>(0, 3) =
        Eigen::Vector3d(lidar_extrinsic_t[0], lidar_extrinsic_t[1], lidar_extrinsic_t[2]);
    Eigen::Quaterniond q_lidar_extrinsic(lidar_extrinsic_q[0], lidar_extrinsic_q[1],
                                         lidar_extrinsic_q[2], lidar_extrinsic_q[3]);
    lidar_extrinsic_.block<3, 3>(0, 0) = q_lidar_extrinsic.toRotationMatrix();
    std::cout << "LiDAR extrinsic: \n" << lidar_extrinsic_ << std::endl;

    se3_lio_pipeline_ = se3_lio::pipeline::SE3_LIO(se3_lio_config_);

    pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/local/odometry", qos_reliable);
    pub_path_ = this->create_publisher<nav_msgs::msg::Path>("/local/path", qos_reliable);
    pub_cloud_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/local/cloud_registered_body", qos_reliable);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
}

LioNode::~LioNode() {
    keep_process_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void LioNode::imuCallback(const IMU_MSG_TYPE::SharedPtr _msg) {
    mutex_.lock();
    imu_queue_.push(_msg);
    mutex_.unlock();
}

void LioNode::lidarCallback(const LIDAR_MSG_TYPE::SharedPtr _msg) {
    mutex_.lock();
    lidar_queue_.push(_msg);
    mutex_.unlock();
}

void LioNode::pubOdometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom,
                          const se3_lio::State &_state,
                          std::string frame_id,
                          std::string child_frame_id) {
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(_state.stamp * 1e9));
    odom_msg.header.frame_id = frame_id;
    odom_msg.child_frame_id = child_frame_id;

    convertStateToROSOdomMsg(_state, odom_msg);

    pub_odom->publish(odom_msg);
}

void LioNode::pubPath(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path,
                      const se3_lio::State &_state,
                      std::string frame_id) {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(_state.stamp * 1e9));
    pose_msg.header.frame_id = frame_id;

    convertStateToROSPoseStampedMsg(_state, pose_msg);

    path_.header = pose_msg.header;
    path_.poses.push_back(pose_msg);

    pub_path->publish(path_);
}

void LioNode::pubCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud,
                       const se3_lio::LiDAR &_lidar,
                       std::string frame_id) {
    sensor_msgs::msg::PointCloud2 cloud_msg;

    pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
    pcl_cloud.reserve(_lidar.points.size());
    for (const auto &p : _lidar.points) {
        pcl::PointXYZI q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.intensity = p.intensity;
        pcl_cloud.push_back(q);
    }
    pcl::toROSMsg(pcl_cloud, cloud_msg);

    cloud_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(_lidar.header.timestamp * 1e9));
    cloud_msg.header.frame_id = frame_id;

    pub_cloud->publish(cloud_msg);
}

void LioNode::broadcastTF(const se3_lio::State &_state,
                          std::string frame_id,
                          std::string child_frame_id) {
    geometry_msgs::msg::TransformStamped transformStamped;
    transformStamped.header.stamp = rclcpp::Time(static_cast<int64_t>(_state.stamp * 1e9));
    transformStamped.header.frame_id = frame_id;
    transformStamped.child_frame_id = child_frame_id;

    convertStateToROSTFMsg(_state, transformStamped);

    tf_broadcaster_->sendTransform(transformStamped);
}

void LioNode::pushAllROSMessages() {
    mutex_.lock();
    synchronizer_.pause();
    while (!imu_queue_.empty()) {
        IMU imu = convertIMUMessage(imu_queue_.front());
        synchronizer_.addIMU(imu);
        imu_queue_.pop();
    }

    while (!lidar_queue_.empty()) {
#if defined(LIDAR_LIVOX)
        LiDAR lidar = convertLivoxMessage(lidar_queue_.front(), lidar_min_range_, lidar_point_filter_num_);
#else  // LIDAR_OUSTER
        LiDAR lidar = convertOusterMessage(lidar_queue_.front(), lidar_min_range_, lidar_point_filter_num_);
#endif
        synchronizer_.addLiDAR(lidar);
        lidar_queue_.pop();
    }
    synchronizer_.resume();
    mutex_.unlock();
}

void LioNode::run() {
    keep_process_ = true;
    thread_ = std::thread(&LioNode::process, this);
}

void LioNode::process() {
    rclcpp::Rate rate(2000);

    MeasurementPtr synced_measurement;

    while (rclcpp::ok() && keep_process_) {
        pushAllROSMessages();

        synced_measurement = synchronizer_.getSyncedMeasurement();
        if (!synced_measurement->is_synced) continue;

        if (se3_lio_config_.verbose) {
            std::cout << std::fixed << std::setprecision(18);
            std::cout << "[MeasurementSynchronizer] Synced IMU size: "
                      << synced_measurement->imu.size()
                      << ", LiDAR points size: " << synced_measurement->lidar.points.size()
                      << std::endl;
            std::cout << "IMU first time : " << synced_measurement->imu.front().header.timestamp
                      << ", last time: " << synced_measurement->imu.back().header.timestamp
                      << std::endl;
            std::cout << "LiDAR start time : " << synced_measurement->lidar.header.timestamp
                      << ", end time: "
                      << synced_measurement->lidar.header.timestamp +
                             synced_measurement->lidar.points.back().timestamp
                      << std::endl;
        }

        synced_measurement->lidar.points =
            transformPointCloud(synced_measurement->lidar.points, lidar_extrinsic_);

        std::sort(synced_measurement->lidar.points.begin(), synced_measurement->lidar.points.end(),
                  [](const CustomPointType &a, const CustomPointType &b) {
                      return a.timestamp < b.timestamp;
                  });

        se3_lio_pipeline_.estimatePose(synced_measurement);

        se3_lio::State state = se3_lio_pipeline_.getState();

        pubOdometry(pub_odom_, state, "map", "base_link");
        pubPath(pub_path_, state, "map");
        pubCloud(pub_cloud_body_, synced_measurement->lidar, "base_link");
        broadcastTF(state, "map", "base_link");

        rate.sleep();
    }
}
}  // namespace se3_lio::ros2_node

int main(int argc, char **argv) {
    // ros::init(argc, argv, "se3_lio");
    rclcpp::init(argc, argv);

    auto node = std::make_shared<se3_lio::ros2_node::LioNode>();

    node->run();

    rclcpp::spin(node);

    rclcpp::shutdown();

    // ros::NodeHandle nh;
    // ros::NodeHandle nh_private("~");

    // se3_lio::ros1_node::LioNode lio_node(nh, nh_private);
    // lio_node.run();

    return 0;
}