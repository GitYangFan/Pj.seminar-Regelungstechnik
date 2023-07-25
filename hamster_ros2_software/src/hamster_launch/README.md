## hamster_launch

This package contains all the launch files to startup all or parts of the hamster ros2 nodes.

**Important note:**
All nodes should always be launched within a namespace.
This is neccessary to prevent overlapping topic and node names between multiple robots in the same network.
E.g. launch a node that publishes the topic ```velocity``` in the namespace ```hamster2``` will automatically adjust the topic name to ```/hamster2/velocity```.
The launch files handle this by selecting the systems hostname as the namespace.
(Note that this does not happen if the node sets the topic to ```/velocity```. So **never** publish a topic starting with a slash, since it will be regarded as absolute and no namespace can be added.)


### **common.launch.py**
This is the main launch file. It starts up the hamster driver and, depending on the settings, multiple other nodes.
Depending on the settings it includes the launch files below.

### camera.launch.py
This launches the ```realsense2_camera_node``` from the official realsense2 package provided by Intel.
The launch file containes all settings that can be adjusted regarding the camera.
Note that this includes the **IMU**, since it is part of the Intel Realsense camera.

Additionally, this launch file includes a static transform publisher for the transformation from the camera_link to the robot's base_link.

### lidar.launch.py
This launch file starts up the lidar driver, as well as a static transform publisher for the transformation from the lidar's coordinate system to the robot's base_link.

### controller.launch.py

### localization.launch.py



