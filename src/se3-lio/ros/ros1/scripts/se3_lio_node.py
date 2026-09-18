#!/usr/bin/env python3
"""SE(3)-LIO ROS1 node (Python) — thin rospy wrapper over the pybind core.

Same data path as the C++ node (ros1/src/lio_node.cpp): subscribe IMU + LiDAR
(PointCloud2: Ouster/Hesai/Velodyne), convert, synchronize, run one odometry
step, publish odom/path/cloud/TF. Converters, synchronizer and odometry are the
shared ``se3_lio`` core; this file is only the ROS1 I/O glue.

Parameters (loaded from the launch's config yaml via ``load_node_params`` — the
same mapping the C++ node and the offline pipeline use):
  config_file : path to the dataset yaml (extrinsic + topics + algorithm keys)
  imu_topic / lidar_topic : optional overrides
"""

import threading

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from se3_lio import SE3LIO, load_node_params
from se3_lio.online_sync import OnlineSynchronizer
from se3_lio.pipeline import _rot_to_quat_xyzw
from se3_lio.datasets.ros1bag import _convert_lidar


def _stamp_sec(header):
    return header.stamp.secs + header.stamp.nsecs * 1e-9


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


class SE3LioNode:
    def __init__(self):
        config_file = rospy.get_param("~config_file", "")
        if not config_file:
            raise RuntimeError("se3_lio_node: '~config_file' parameter is required")
        params = load_node_params(config_file)
        imu_topic = rospy.get_param("~imu_topic", "") or params["imu_topic"]
        lidar_topic = rospy.get_param("~lidar_topic", "") or params["lidar_topic"]
        self._min_range = params["min_range"]
        extrinsic = np.asarray(params["extrinsic"], dtype=float)

        self._sync = OnlineSynchronizer()
        self._odom = SE3LIO(params["config"], extrinsic)
        self._path = Path()
        self._lock = threading.Lock()  # rospy callbacks run on separate threads

        self._pub_odom = rospy.Publisher("/local/odometry", Odometry, queue_size=10)
        self._pub_path = rospy.Publisher("/local/path", Path, queue_size=10)
        self._pub_cloud = rospy.Publisher("/local/cloud_registered_body", PointCloud2, queue_size=10)
        self._tf = TransformBroadcaster()

        rospy.Subscriber(imu_topic, Imu, self._imu_cb, queue_size=10000)
        rospy.Subscriber(lidar_topic, PointCloud2, self._lidar_cb, queue_size=10)
        rospy.loginfo(f"se3_lio_node up | imu={imu_topic} lidar={lidar_topic}")

    def _imu_cb(self, msg):
        imu_row = [_stamp_sec(msg.header),
                   msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                   msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        with self._lock:
            self._sync.add_imu(imu_row)
            self._drain_and_publish()

    def _lidar_cb(self, msg):
        pts, offsets, header_shift = _convert_lidar(msg, self._min_range)
        with self._lock:
            self._sync.add_scan(_stamp_sec(msg.header) + header_shift, pts, offsets)
            self._drain_and_publish()

    def _drain_and_publish(self):
        for scan, imu_block in self._sync.drain():
            state, cloud = self._odom.register_frame(
                scan["pts"], scan["offsets"], imu_block, scan["header_ts"]
            )
            self._publish(state, cloud)

    def _publish(self, state, cloud):
        stamp = rospy.Time.from_sec(float(state.stamp))
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

        # deskewed cloud in body frame. Fresh header — reusing odom.header would
        # mutate the shared path/tf header.
        self._pub_cloud.publish(_cloud_msg(Header(stamp=stamp, frame_id="base_link"), cloud))


def main():
    rospy.init_node("se3_lio_node")
    SE3LioNode()
    rospy.spin()


if __name__ == "__main__":
    main()
