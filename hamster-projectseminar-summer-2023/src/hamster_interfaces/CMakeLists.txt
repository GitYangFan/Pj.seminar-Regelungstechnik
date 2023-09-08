cmake_minimum_required(VERSION 3.8)
project(hamster_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackingControllerState.msg"
  "msg/VelocityWithHeader.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_package()
