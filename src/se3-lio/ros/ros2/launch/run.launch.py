from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # set path to config file
    config_file_path = os.path.join(
        get_package_share_directory("se3_lio"),
        "config",
        "params.yaml",
    )

    # set path to rviz config file
    rviz_file_path = os.path.join(
        get_package_share_directory("se3_lio"),
        "rviz",
        "rviz.rviz",
    )

    rviz = LaunchConfiguration("rviz", default="false")
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_file_path],
        condition=IfCondition(rviz),
    )

    # Create Node
    node = Node(
        package="se3_lio",
        executable="lio_node",
        name="lio_node",
        parameters=[config_file_path, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use /clock time (set true when replaying a rosbag with --clock)",
            ),
            node,
            rviz_node,
        ]
    )
