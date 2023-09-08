import re
import socket


def get_robot_name() -> str:

    # Get hostname to use it as namespace for ROS
    # to give inidividual topic/node names to each hamster robot
    HOSTNAME = socket.gethostname()

    # Replace dash by underscore
    HOSTNAME = HOSTNAME.replace('-', '_')

    # If hostname is invalid ROS name, replace it
    re_pattern = '^[a-zA-Z][a-zA-Z0-9_]+$'
    if not re.match(re_pattern, HOSTNAME):
        HOSTNAME = 'robot_with_invalid_hostname'
        print('WARNING: Hostname contains invalid chars.')

    return HOSTNAME