#include "lio_node.h"

namespace se3_lio::ros1_node {
LioNode::LioNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private) : nh_(nh), pnh_(nh_private) {
    std::string imu_topic, lidar_topic;
    pnh_.param<std::string>("/imu_topic", imu_topic, "/imu/data");
    pnh_.param<std::string>("/lidar_topic", lidar_topic, "/lidar/points");

    sub_imu_ = nh_.subscribe(imu_topic, 1000, &LioNode::imuCallback, this);
    sub_lidar_ = nh_.subscribe(lidar_topic, 1000, &LioNode::lidarCallback, this);

    pnh_.param<double>("/sensors/imu/acc_cov", se3_lio_config_.acc_noise, 0.1);
    pnh_.param<double>("/sensors/imu/gyr_cov", se3_lio_config_.gyr_noise, 0.1);
    pnh_.param<double>("/sensors/imu/b_acc_cov", se3_lio_config_.bg_noise, 0.0001);
    pnh_.param<double>("/sensors/imu/b_gyr_cov", se3_lio_config_.ba_noise, 0.0001);

    pnh_.param<double>("/sensors/lidar/min_range", lidar_min_range_, 0.1);
    pnh_.param<int>("/sensors/lidar/point_filter_num", lidar_point_filter_num_, 1);
    if (lidar_point_filter_num_ < 1) lidar_point_filter_num_ = 1;
    pnh_.param<double>("/sensors/lidar/range_cov", se3_lio_config_.lidar_range_noise,
                       0.001);  // 벡터로 수정해야함
    pnh_.param<double>("/sensors/lidar/angle_cov", se3_lio_config_.lidar_angle_noise, 0.01);

    std::vector<double> lidar_extrinsic_t, lidar_extrinsic_q;
    pnh_.param<std::vector<double>>("/sensors/t_exts", lidar_extrinsic_t, {0.0, 0.0, 0.0});
    pnh_.param<std::vector<double>>("/sensors/q_exts", lidar_extrinsic_q, {1.0, 0.0, 0.0, 0.0});

    pnh_.param<double>("/downsample/resolution", se3_lio_config_.downsample_resolution, 0.5);
    pnh_.param<int>("/max_iter", se3_lio_config_.max_iter, 4);

    pnh_.param<double>("/voxel_map/resolution", se3_lio_config_.voxel_map_resolution, 1.0);
    pnh_.param<int>("/voxel_map/max_layer", se3_lio_config_.voxel_map_max_layer, 2);
    pnh_.param<std::vector<int>>("/voxel_map/layer_size", se3_lio_config_.voxel_map_layer_size,
                                 {5, 5, 5, 5, 5});
    pnh_.param<int>("/voxel_map/max_point_size", se3_lio_config_.voxel_map_max_point_size, 1000);
    pnh_.param<float>("/voxel_map/plane_threshold", se3_lio_config_.voxel_map_plane_thres, 0.01f);
    pnh_.param<bool>("/voxel_map/map_sliding_en", se3_lio_config_.voxel_map_sliding_en, false);
    pnh_.param<double>("/voxel_map/sliding_thresh", se3_lio_config_.voxel_map_sliding_thresh, 8.0);
    pnh_.param<int>("/voxel_map/half_map_size", se3_lio_config_.voxel_map_half_size, 50);

    lidar_extrinsic_.block<3, 1>(0, 3) =
        Eigen::Vector3d(lidar_extrinsic_t[0], lidar_extrinsic_t[1], lidar_extrinsic_t[2]);
    Eigen::Quaterniond q_lidar_extrinsic(lidar_extrinsic_q[0], lidar_extrinsic_q[1],
                                         lidar_extrinsic_q[2], lidar_extrinsic_q[3]);
    lidar_extrinsic_.block<3, 3>(0, 0) = q_lidar_extrinsic.toRotationMatrix();
    std::cout << "LiDAR extrinsic: \n" << lidar_extrinsic_ << std::endl;

    se3_lio_pipeline_ = se3_lio::pipeline::SE3_LIO(se3_lio_config_);

    pub_odom_ = pnh_.advertise<nav_msgs::Odometry>("/local/odometry", 1000);
    pub_path_ = pnh_.advertise<nav_msgs::Path>("/local/path", 1000);
    pub_cloud_body_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("/local/cloud_registered_body", 1000);
}

LioNode::~LioNode() {}

void LioNode::imuCallback(const IMU_MSG_TYPE &_msg) {
    mutex_.lock();
    imu_queue_.push(_msg);
    mutex_.unlock();
}

void LioNode::lidarCallback(const LIDAR_MSG_TYPE &_msg) {
    mutex_.lock();
    lidar_queue_.push(_msg);
    mutex_.unlock();
}

void LioNode::pubOdometry(ros::Publisher pub_odom,
                          const se3_lio::State &_state,
                          std::string frame_id,
                          std::string child_frame_id) {
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time(_state.stamp);
    odom_msg.header.frame_id = frame_id;
    odom_msg.child_frame_id = child_frame_id;

    convertStateToROSOdomMsg(_state, odom_msg);

    pub_odom.publish(odom_msg);
}

void LioNode::pubPath(ros::Publisher pub_path, const se3_lio::State &_state, std::string frame_id) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time(_state.stamp);
    pose_msg.header.frame_id = frame_id;

    convertStateToROSPoseStampedMsg(_state, pose_msg);

    path_.header = pose_msg.header;
    path_.poses.push_back(pose_msg);

    pub_path.publish(path_);
}

void LioNode::pubCloud(ros::Publisher pub_cloud,
                       const se3_lio::LiDAR &_lidar,
                       std::string frame_id) {
    sensor_msgs::PointCloud2 cloud_msg;

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

    cloud_msg.header.stamp = ros::Time(_lidar.header.timestamp);
    cloud_msg.header.frame_id = frame_id;

    pub_cloud.publish(cloud_msg);
}

void LioNode::broadcastTF(const se3_lio::State &_state,
                          std::string frame_id,
                          std::string child_frame_id) {
    static tf2_ros::TransformBroadcaster br;

    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time(_state.stamp);
    transformStamped.header.frame_id = frame_id;
    transformStamped.child_frame_id = child_frame_id;

    convertStateToROSTFMsg(_state, transformStamped);

    br.sendTransform(transformStamped);
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
#elif defined(LIDAR_HESAI)
        LiDAR lidar = convertHesaiMessage(lidar_queue_.front(), lidar_min_range_, lidar_point_filter_num_);
#elif defined(LIDAR_VELODYNE)
        LiDAR lidar = convertVelodyneMessage(lidar_queue_.front(), lidar_min_range_, lidar_point_filter_num_);
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
    ros::Rate rate(2000);

    MeasurementPtr synced_measurement;

    while (ros::ok()) {
        ros::spinOnce();

        pushAllROSMessages();

        synced_measurement = synchronizer_.getSyncedMeasurement();
        if (!synced_measurement->is_synced) continue;

        std::cout << std::fixed << std::setprecision(18);
        std::cout << "[MeasurementSynchronizer] Synced IMU size: " << synced_measurement->imu.size()
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
}  // namespace se3_lio::ros1_node

int main(int argc, char **argv) {
    ros::init(argc, argv, "se3_lio");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    se3_lio::ros1_node::LioNode lio_node(nh, nh_private);
    lio_node.run();

    return 0;
}