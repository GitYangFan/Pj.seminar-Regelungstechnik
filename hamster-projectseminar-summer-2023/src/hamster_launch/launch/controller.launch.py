from launch import LaunchDescription
from launch_ros.actions import Node

from hamster_launch.robot_name import get_robot_name


def generate_launch_description():
    ld = LaunchDescription()

    # Get robot name
    robot_name = get_robot_name()

    # Simple tracking controller
    ld.add_action(Node(
        package='tracking_controller',
        executable='simple_tracking_controller',
        name='simple_tracking_controller',
        namespace=robot_name
    ))

    return ld
