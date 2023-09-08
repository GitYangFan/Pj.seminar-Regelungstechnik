## hamster_driver

This package contains the hamster_serial node that provides the interface to the Hamster's microcontroller board via a serial connection.


### Subscriptions
This node subscribes to the following topics:

| Topic name | Message type | Description |
|-|-|-|
| command | ackermann_msgs::msg::AckermannDriveStamped | Set target velocity (in m/s) and steering angle (in degree). <br> The hamster will stop if no command (either ackermann or twist) is received for a certain amount of time. |
| twist_command | geometry_msgs::msg::Twist | Set target velocity (in m/s) and yaw rate (in rad/s). <br> The target yaw rate will internally be converted to target steering angle using ```yaw_rate = velocity/WHEELBASE * tan(steering_angle)```. |
| interlock | std_msgs::msg::Bool | Needs to be received continuously, otherwise the hamster will stop. This topic should be publisher from another PC on the network, so that the hamster stops if you loose connection. |

### Publishers
This node provides to the following topics:

| Topic name | Message type | Description |
|-|-|-|
| voltage | std_msgs::msg::Float32 | The hamsters battery voltage (in V). |
| pid | geometry_msgs::msg::Vector3 | (undocumented) |
| velocity | std_msgs::msg::Float32 | The hamsters velocity (in m/s). |

### Parameters
The node requires the following parameters:

| Parameter name | Type | Description |
|-|-|-|
| speed_limit | double | The maximum speed in (m/s). |
