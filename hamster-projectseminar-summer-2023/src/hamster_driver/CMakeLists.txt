cmake_minimum_required(VERSION 3.8)
project(hamster_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(hamster_interfaces REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Create hamster_serial executable
add_executable(hamster_serial
  src/hamster_serial.cpp
  src/serialib.cpp)

target_include_directories(hamster_serial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(hamster_serial PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(hamster_serial
  rclcpp
  rclpy
  ackermann_msgs
  std_msgs
  geometry_msgs
  hamster_interfaces)

install(TARGETS hamster_serial DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/test_node.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/EKF_node.py DESTINATION lib/${PROJECT_NAME})
# Testing stuff
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
