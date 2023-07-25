from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from hamster_launch.robot_name import get_robot_name


def generate_launch_description():
    ld = LaunchDescription()

    # Get robot name
    robot_name = get_robot_name()

    # Declare arguments
    ld.add_action(
        DeclareLaunchArgument('enable_lidar', default_value='false'))
    ld.add_action(
        DeclareLaunchArgument('enable_camera', default_value='false'))
    ld.add_action(
        DeclareLaunchArgument('launch_controller', default_value='true'))

    # Add serial driver node
    ld.add_action(Node(
        package='hamster_driver',
        executable='hamster_serial',
        name='hamster_serial',
        namespace=robot_name,
        parameters=[{'speed_limit': 0.6}]
    ))

    # Include lidar launch file
    lidar_launch_desc_src = PythonLaunchDescriptionSource(
        PathJoinSubstitution([ThisLaunchFileDir(), 'lidar.launch.py']))
    ld.add_action(IncludeLaunchDescription(
        launch_description_source=lidar_launch_desc_src,
        condition=IfCondition(LaunchConfiguration('enable_lidar'))
    ))

    # Include camera launch file
    camera_launch_desc_src = PythonLaunchDescriptionSource(
        PathJoinSubstitution([ThisLaunchFileDir(), 'camera.launch.py']))
    ld.add_action(IncludeLaunchDescription(
        launch_description_source=camera_launch_desc_src,
        condition=IfCondition(LaunchConfiguration('enable_camera'))
    ))

    # Include localization launchfile
    localization_launch_desc_src = PythonLaunchDescriptionSource(
        PathJoinSubstitution([ThisLaunchFileDir(), 'localization.launch.py']))
    ld.add_action(IncludeLaunchDescription(
        launch_description_source=localization_launch_desc_src
    ))

    # Include controller launchfile
    controller_launch_desc_src = PythonLaunchDescriptionSource(
        PathJoinSubstitution([ThisLaunchFileDir(), 'controller.launch.py']))
    ld.add_action(IncludeLaunchDescription(
        launch_description_source=controller_launch_desc_src,
        condition=IfCondition(LaunchConfiguration('launch_controller'))
    ))

    return ld
