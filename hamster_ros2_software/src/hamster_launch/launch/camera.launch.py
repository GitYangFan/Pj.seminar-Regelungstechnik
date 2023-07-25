from launch import LaunchDescription
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node

from hamster_launch.robot_name import get_robot_name


def generate_launch_description():
    ld = LaunchDescription()

    # Get robot name
    robot_name = get_robot_name()

    # Static transform from base_link to camera_link
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_camera',
        namespace=robot_name,
        arguments=['--x', '0.13', '--z', '0.07',
                   '--frame-id', [robot_name,
                                  TextSubstitution(text='/base_link')],
                   '--child-frame-id', [robot_name, TextSubstitution(text='/camera_link')]],
        ros_arguments=['--log-level', 'warn']
    ))

    # Realsense camera node
    realsense_configurable_parameters = {
        # Camera name (important, since it is the prefix of all camera tf frames)
        'camera_name': f'{robot_name}/camera',

        # Logging and output
        'log_level': 'info',
        'output': 'screen',

        # Enable/disable camera stream
        'enable_depth': True,
        'enable_color': True,
        'enable_infra1': False,
        'enable_fisheye1': False,
        'enable_fisheye2': False,
        'enable_infra2': False,
        'enable_confidence': False,
        'enable_gyro': True,
        'enable_accel': True,
        'enable_pose': False,

        # Camera profiles (width,height,fps)
        'depth_module.profile': '640,480,6',
        'rgb_camera.profile': '640,480,6',

        # IMU settings
        'unite_imu_method': 1,
        'gyro_fps': 0,
        'accel_fps': 0,
        'linear_accel_cov': 0.01,
        'angular_velocity_cov': 0.01,

        # Post-processing
        'align_depth.enable': False,
        'colorizer.enable': False,
        'pointcloud.enable': False,
        'pointcloud.stream_filter': 2,
        'pointcloud.stream_index_filter': 0,
        'pointcloud.ordered_pc': False,
        'pointcloud.allow_no_texture_points': False,
        'hdr_merge.enable': False,
        'depth_module.exposure.1': 7500,
        'depth_module.gain.1': 16,
        'depth_module.exposure.2': 1,
        'depth_module.gain.2': 16,
        'decimation_filter.enable': False,
        'clip_distance': -2.,

        # Other settings
        'enable_sync': False,
        'wait_for_device_timeout': 10.,
        'reconnect_timeout': 3.,
        'initial_reset': True,
        'tf_publish_rate': 0.0,
        'diagnostics_period': 0.0
    }
    ld.add_action(Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace=robot_name,
        output=realsense_configurable_parameters['output'],
        ros_arguments=['--log-level',
                       realsense_configurable_parameters['log_level']],
        emulate_tty=True,
        parameters=[realsense_configurable_parameters]))

    return ld
