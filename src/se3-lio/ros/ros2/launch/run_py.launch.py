"""Launch the Python (rclpy) SE(3)-LIO node — mirrors run.launch.py for the C++ node.

The node reads its dataset yaml via `config_file` (default: the package's
params.yaml, same file the C++ node loads), so both nodes share one config.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory("se3_lio")
    default_config = os.path.join(share, "config", "params.yaml")
    rviz_file = os.path.join(share, "rviz", "rviz.rviz")

    config = LaunchConfiguration("config", default=default_config)
    lidar_type = LaunchConfiguration("lidar_type", default="livox")
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    rviz = LaunchConfiguration("rviz", default="false")

    node = Node(
        package="se3_lio",
        executable="se3_lio_py_node",
        name="se3_lio_node",
        output="screen",
        parameters=[{
            "config_file": config,
            "lidar_type": lidar_type,
            "use_sim_time": use_sim_time,
        }],
    )
    rviz_node = Node(
        package="rviz2", executable="rviz2", arguments=["-d", rviz_file],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config,
                              description="dataset yaml (extrinsic + topics + algorithm keys)"),
        DeclareLaunchArgument("lidar_type", default_value="livox",
                              description="livox (CustomMsg) or pointcloud2 (Ouster/Hesai/Velodyne)"),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="Use /clock (set true when replaying a rosbag with --clock)"),
        node,
        rviz_node,
    ])
