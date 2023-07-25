cmake_minimum_required(VERSION 3.8)
project(tracking_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(hamster_interfaces REQUIRED)

# Add tracking controller executable
add_executable(simple_tracking_controller
  src/simple_tracking_controller.cpp
  src/trajectory.cpp)

target_include_directories(simple_tracking_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(simple_tracking_controller PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(simple_tracking_controller
  rclcpp
  ackermann_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  hamster_interfaces)

install(TARGETS simple_tracking_controller
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  data
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
