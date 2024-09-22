import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch.conditions import IfCondition
from launch_ros.actions import Node
from whill_navi2.modules.ros2_launch_utils import DataPath, get_param_dict

def generate_launch_description():
    
    config_path = os.path.join(
        get_package_share_directory("hdl_localization"),
        "config", "params.yaml"
    )
    data_path = DataPath()
    globalmap_server_config_dict = get_param_dict(config_path, "globalmap_server_node")
    globalmap_server_config_dict["globalmap_pcd"] = os.path.join(data_path.pcd_map_dir_path, "glim_pcd.pcd")
    
    ld = LaunchDescription()
    
    ld.add_action(DeclareLaunchArgument(name="point_cloud_topic", default_value="velodyne_points"))
    point_cloud_topic = LaunchConfiguration("point_cloud_topic")
    ld.add_action(DeclareLaunchArgument(name="imu_topic", default_value="adis/imu/data"))
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
                "config"
            )
        }],
        output="screen"
    )
    ld.add_action(hdl_global_localization_node)    
    
    globalmap_server_node = Node(
        package="hdl_localization",
        executable="globalmap_server_node",
        name="globalmap_server_node",
        parameters=[globalmap_server_config_dict],
        output="screen"
    )
    ld.add_action(globalmap_server_node)
    
    hdl_localization_node = Node(
        package="hdl_localization",
        executable="hdl_localization_node",
        name="hdl_localization_node",
        parameters=[config_path],
        remappings=[
            # Subscribe
            ("velodyne_points", point_cloud_topic),
            ("imu_data", imu_topic),
            # Publish
            ("odom", "hdl_localization/odom")
        ],
        output="screen"
    )
    ld.add_action(hdl_localization_node)
    
    plot_status = ExecuteProcess(
        cmd=[
            FindExecutable(name="python3"),
            os.path.join(
                get_package_share_directory("hdl_localization"),
                "scripts", "plot_status.py"
            )
        ],
        condition=IfCondition(plot_estimation_errors)
    )
    ld.add_action(plot_status)
    
    return ld
