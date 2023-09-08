from launch import LaunchDescription
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node, RosTimer

from hamster_launch.robot_name import get_robot_name


def generate_launch_description():
    ld = LaunchDescription()

    # Get robot name
    robot_name = get_robot_name()

    lidar_frame_id = [robot_name, TextSubstitution(text='/laser')]
    robot_frame_id = [robot_name, TextSubstitution(text='/base_link')]

    # Static transform from base_link to lidar
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        namespace=robot_name,
        name='tf_base_lidar',
        arguments=['--x', '-0.05', '--z', '0.16',
                   '--frame-id', robot_frame_id,
                   '--child-frame-id', lidar_frame_id],
        ros_arguments=['--log-level', 'warn']
    ))

    # Add serial driver node
    ld.add_action(RosTimer(period=5.0, actions=[Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        namespace=robot_name,
        name='rplidar',
        output='screen',
        parameters=[{'serial_port': '/dev/ttyUSB0'},
                    {'serial_baudrate': 115200},
                    {'frame_id': lidar_frame_id},
                    {'inverted': False},
                    {'angle_compensate': True}]
    )]))

    return ld
