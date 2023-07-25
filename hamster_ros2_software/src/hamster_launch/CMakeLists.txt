cmake_minimum_required(VERSION 3.8)
project(hamster_launch)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install launch files.
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install the hamster_launch python package
ament_python_install_package(hamster_launch)

ament_package()
