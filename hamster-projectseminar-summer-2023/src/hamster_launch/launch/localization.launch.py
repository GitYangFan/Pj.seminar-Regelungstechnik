from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution

from hamster_launch.robot_name import get_robot_name


def generate_launch_description():
    ld = LaunchDescription()

    # Get robot name
    robot_name = get_robot_name()

    ld.add_action(Node(
        package='hamster_localization',
        executable='only_optitrack_localization',
        namespace=robot_name,
        name='optitrack_localization'
    ))

    return ld
