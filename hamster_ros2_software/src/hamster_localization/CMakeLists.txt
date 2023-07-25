cmake_minimum_required(VERSION 3.8)
project(hamster_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


# Add tracking controller executable
add_executable(only_optitrack_localization
  src/only_optitrack_localization.cpp)

target_include_directories(only_optitrack_localization PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(only_optitrack_localization PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(only_optitrack_localization
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs)

install(TARGETS only_optitrack_localization
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
