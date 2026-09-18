#!/usr/bin/env python3
"""SE(3)-LIO ROS2 node (Python) — thin rclpy wrapper over the pybind core.

Same data path as the C++ node (ros2/src/lio_node.cpp): subscribe IMU + LiDAR,
convert each message, synchronize, run one odometry step, publish
odom/path/cloud/TF. Converters, synchronizer and odometry are the shared
``se3_lio`` core — this file is only the ROS2 I/O glue.

Parameters (from the launch's config yaml, read via ``load_node_params`` — the
same mapping the C++ node and the offline pipeline use):
  config_file : path to the dataset yaml (extrinsic + topics + algorithm keys)
  imu_topic / lidar_topic : optional overrides
  lidar_type  : "livox" (CustomMsg) or "pointcloud2" (Ouster/Hesai/Velodyne)
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from se3_lio import SE3LIO, load_node_params
from se3_lio.online_sync import OnlineSynchronizer
from se3_lio.pipeline import _rot_to_quat_xyzw
from se3_lio.datasets.rosbag import _convert_livox
from se3_lio.datasets.ros1bag import _convert_lidar


def _stamp_msg(t):
    ns = int(round(t * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def _stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _cloud_msg(header, pts):
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate(("x", "y", "z"))
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


class SE3LioNode(Node):
    def __init__(self):
        super().__init__("se3_lio_node")
        self.declare_parameter("config_file", "")
        self.declare_parameter("imu_topic", "")
        self.declare_parameter("lidar_topic", "")
        self.declare_parameter("lidar_type", "livox")

        config_file = self.get_parameter("config_file").value
        if not config_file:
            raise RuntimeError("se3_lio_node: 'config_file' parameter is required")
        params = load_node_params(config_file)
        imu_topic = self.get_parameter("imu_topic").value or params["imu_topic"]
        lidar_topic = self.get_parameter("lidar_topic").value or params["lidar_topic"]
        self._lidar_type = self.get_parameter("lidar_type").value
        self._min_range = params["min_range"]
        extrinsic = np.asarray(params["extrinsic"], dtype=float)

        self._sync = OnlineSynchronizer()
        self._odom = SE3LIO(params["config"], extrinsic)
        self._path = Path()
        self._lock = threading.Lock()

        # QoS mirrors the C++ node: best_effort sensors (matches `ros2 bag play`),
        # reliable outputs; a deep IMU queue for the high-rate stream.
        qos_imu = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=10000)
        qos_lidar = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                               history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        qos_out = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(Imu, imu_topic, self._imu_cb, qos_imu)
        if self._lidar_type == "livox":
            from livox_ros_driver2.msg import CustomMsg
            self.create_subscription(CustomMsg, lidar_topic, self._lidar_cb, qos_lidar)
        else:
            self.create_subscription(PointCloud2, lidar_topic, self._lidar_cb, qos_lidar)

        self._pub_odom = self.create_publisher(Odometry, "/local/odometry", qos_out)
        self._pub_path = self.create_publisher(Path, "/local/path", qos_out)
        self._pub_cloud = self.create_publisher(PointCloud2, "/local/cloud_registered_body", qos_out)
        self._tf = TransformBroadcaster(self)
        self.get_logger().info(
            f"se3_lio_node up | imu={imu_topic} lidar={lidar_topic} ({self._lidar_type})"
        )

    def _imu_cb(self, msg):
        imu_row = [_stamp_sec(msg.header),
                   msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                   msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        with self._lock:
            self._sync.add_imu(imu_row)
            self._drain_and_publish()

    def _lidar_cb(self, msg):
        if self._lidar_type == "livox":
            pts, offsets = _convert_livox(msg, self._min_range)
            header_ts = _stamp_sec(msg.header)
        else:
            pts, offsets, header_shift = _convert_lidar(msg, self._min_range)
            header_ts = _stamp_sec(msg.header) + header_shift
        with self._lock:
            self._sync.add_scan(header_ts, pts, offsets)
            self._drain_and_publish()

    def _drain_and_publish(self):
        for scan, imu_block in self._sync.drain():
            state, cloud = self._odom.register_frame(
                scan["pts"], scan["offsets"], imu_block, scan["header_ts"]
            )
            self._publish(state, cloud)

    def _publish(self, state, cloud):
        stamp = _stamp_msg(state.stamp)
        pose = np.asarray(state.pose, dtype=float)
        pos, quat = pose[:3, 3], _rot_to_quat_xyzw(pose[:3, :3])

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = pos
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = quat
        self._pub_odom.publish(odom)

        pose_st = PoseStamped()
        pose_st.header = odom.header
        pose_st.pose = odom.pose.pose
        self._path.header = odom.header
        self._path.poses.append(pose_st)
        self._pub_path.publish(self._path)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_link"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pos
        (tf.transform.rotation.x, tf.transform.rotation.y,
         tf.transform.rotation.z, tf.transform.rotation.w) = quat
        self._tf.sendTransform(tf)

        # deskewed cloud in body frame (from register_frame), like the C++ node.
        # Fresh header — reusing odom.header would mutate the shared path/tf header.
        self._pub_cloud.publish(_cloud_msg(Header(stamp=stamp, frame_id="base_link"), cloud))


def main(args=None):
    rclpy.init(args=args)
    node = SE3LioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
