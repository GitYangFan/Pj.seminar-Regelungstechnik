#include "hamster_localization/only_optitrack_localization.hpp"


OnlyOptitrackLocalization::OnlyOptitrackLocalization()
: rclcpp::Node("optitrack_localization")
{
  // Frame ids
  robot_frame_id_ = this->getRobotName() + "/base_link";
  odom_frame_id_ = this->getRobotName() + "/odom";
  map_frame_id_ = this->getRobotName() + "/map";


  // Initialize tf stuff
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

  // Wait until the frames exist (== their id is not zero)
  auto error_throttle_clock = rclcpp::Clock(RCL_STEADY_TIME);
  while (rclcpp::ok() && !tf_buffer_->_lookupFrameNumber(OPTITRACK_FRAME_ID_)) {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(
      this->get_logger(), error_throttle_clock, 5000,
      "Optitrack frame id \"%s\" still does not exist. Waiting... ", OPTITRACK_FRAME_ID_.c_str());
  }
  while (rclcpp::ok() && !tf_buffer_->_lookupFrameNumber(robot_frame_id_)) {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(
      this->get_logger(), error_throttle_clock, 5000,
      "Robot frame id \"%s\" still does not exist. Waiting... ", robot_frame_id_.c_str());
  }

  // Wait until transform is available
  while (rclcpp::ok() &&
    !(tf_buffer_->canTransform(OPTITRACK_FRAME_ID_, robot_frame_id_, rclcpp::Time(0))))
  {}

  // Get initial tf from optitrack frame to robot
  geometry_msgs::msg::TransformStamped optitrack_robot_tf_initial = tf_buffer_->lookupTransform(
    OPTITRACK_FRAME_ID_, robot_frame_id_, rclcpp::Time(0));

  // Broadcast static tf from optitrack-frame to robot-map-frame
  // Map origin will be at the starting location of the robot
  geometry_msgs::msg::TransformStamped optitrack_map_tf;
  optitrack_map_tf.header.stamp = this->get_clock()->now();
  optitrack_map_tf.header.frame_id = OPTITRACK_FRAME_ID_;
  optitrack_map_tf.child_frame_id = map_frame_id_;
  optitrack_map_tf.transform = optitrack_robot_tf_initial.transform;
  static_tf_broadcaster_->sendTransform(optitrack_map_tf);

  // Broadcast identity transform from map to odom
  geometry_msgs::msg::TransformStamped map_odom_tf;
  map_odom_tf.header.stamp = this->get_clock()->now();
  map_odom_tf.header.frame_id = map_frame_id_;
  map_odom_tf.child_frame_id = odom_frame_id_;
  map_odom_tf.transform.translation.x = 0;
  map_odom_tf.transform.translation.y = 0;
  map_odom_tf.transform.translation.z = 0;
  map_odom_tf.transform.rotation.x = 0;
  map_odom_tf.transform.rotation.y = 0;
  map_odom_tf.transform.rotation.z = 0;
  map_odom_tf.transform.rotation.w = 1;
  static_tf_broadcaster_->sendTransform(map_odom_tf);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OnlyOptitrackLocalization>());
  rclcpp::shutdown();
  return 0;
}
