import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    
    config_path = os.path.join(
        get_package_share_directory("hdl_localization"),
        "config", "params.yaml"
    )
    
    ld = LaunchDescription()
    
    ld.add_action(DeclareLaunchArgument("point_cloud_topic", "velodyne_points"))
    point_cloud_topic = LaunchConfiguration("point_cloud_topic")
    ld.add_action(DeclareLaunchArgument("imu_topic", "/adis/imu/data"))
    imu_topic = LaunchConfiguration("imu_topic")
    ld.add_action(DeclareLaunchArgument(name="plot_estimation_errors", default_value="False", choices=["False", "True", "false", "true"]))
    plot_estimation_errors = LaunchConfiguration("plot_estimation_errors")
    
    hdl_global_localization_node = Node(
        package="hdl_global_localization",
        executable="hdl_global_localization_node",
        name="hdl_global_localization_node",
        parameters=[{
            "config_path" : os.path.join(
                get_package_share_directory("hdl_global_localization"),
                "config", "config.json"
            )
        }],
        output="screen"
    )
    ld.add_action(hdl_global_localization_node)    
    
    globalmap_server_node = Node(
        package="hdl_localization",
        executable="globalmap_server_node",
        name="globalmap_server_node",
        parameters=[config_path],
        output="screen"
    )
    ld.add_action(globalmap_server_node)
    
    hdl_localization_node = Node(
        package="hdl_localization",
        executable="hdl_localization_node",
        name="hdl_localization_node",
        parameters=[config_path],
        remappings=[
            ("velodyne_points", point_cloud_topic),
            ("imu_data", imu_topic)
        ],
        output="screen"
    )
    ld.add_action(hdl_localization_node)
    
    plot_status = ExecuteProcess(
        cmd=[
            FindExecutable("python3"),
            os.path.join(
                get_package_share_directory("hdl_localization"),
                "scripts", "plot_status.py"
            )
        ],
        condition=IfCondition(plot_estimation_errors)
    )
    ld.add_action(plot_status)
    
    return ld
