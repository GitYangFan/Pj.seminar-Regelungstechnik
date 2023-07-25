#include "tracking_controller/simple_tracking_controller.hpp"


SimpleTrackingController::SimpleTrackingController()
: Node("tracking_controller")
{
  // Frame ids
  robot_frame_id_ = this->getRobotName() + "/base_link";
  odom_frame_id_ = this->getRobotName() + "/odom";

  // Publishers
  command_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("command", 1);
  tracking_state_pub_ = this->create_publisher<hamster_interfaces::msg::TrackingControllerState>(
    "tracking_controller_state", 1);
  traj_visualization_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
    "trajectory_visualization", 1);

  // Update timers
  controller_update_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(20),
    std::bind(&SimpleTrackingController::updateController, this));
  trajectory_tf_timer_ =
    this->create_wall_timer(
    std::chrono::milliseconds(200),
    std::bind(&SimpleTrackingController::retransformTrajectory, this));

  // tf stuff
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

  // Load trajectory
  auto traj_file_name = ament_index_cpp::get_package_share_directory("tracking_controller") +
    "/data/test_traj.csv";
  TrajectoryUtils::Trajectory::loadFromCsv(
    traj_file_name, current_trajectory_, this->getRobotName() + "/map");

  // Wait until frames exist   and transforms are available
  auto error_throttle_clock = rclcpp::Clock(RCL_STEADY_TIME);
  bool frames_missing = true;
  while (rclcpp::ok() && (frames_missing != 0)) {
    frames_missing = false;
    // Check robot frame
    if (!tf_buffer_->_lookupFrameNumber(robot_frame_id_)) {
      RCLCPP_WARN_SKIPFIRST_THROTTLE(
        this->get_logger(), error_throttle_clock, 5000,
        "Frame with id \"%s\" still does not exist. Waiting... ", robot_frame_id_.c_str());
      frames_missing = true;
    }
    // Check odom frame
    if (!tf_buffer_->_lookupFrameNumber(odom_frame_id_)) {
      RCLCPP_WARN_SKIPFIRST_THROTTLE(
        this->get_logger(), error_throttle_clock, 5000,
        "Frame with id \"%s\" still does not exist. Waiting... ", odom_frame_id_.c_str());
      frames_missing = true;
    }
  }
}


void SimpleTrackingController::updateController()
{
  auto pose = this->lookupVehiclePose();
  auto tracking_state = current_trajectory_.getTrackingState(pose);
  double steering_angle = this->controlLaw(tracking_state);

  // Construct and publish command message
  ackermann_msgs::msg::AckermannDriveStamped msg;
  msg.header.stamp = this->get_clock()->now();
  msg.drive.speed = tracking_state.velocity;
  msg.drive.steering_angle = steering_angle * 180.0 / 3.141592;
  command_pub_->publish(msg);

  // Publish for debug purposes
  tracking_state_pub_->publish(
    tracking_state.toMsg(
      this->get_clock()->now(),
      current_trajectory_.getFrameID()));
}


void SimpleTrackingController::retransformTrajectory()
{
  current_trajectory_ = current_trajectory_.transformTo(tf_buffer_.get(), odom_frame_id_);
  traj_visualization_pub_->publish(current_trajectory_.toMarker(this->get_clock()->now()));
}


double SimpleTrackingController::controlLaw(TrajectoryUtils::TrackingState state) const
{
  const double & v = state.velocity;
  if (std::abs(v) < 0.1) {
    return 0;
  }

  double delta = 0;
  delta += (a0_ * WHEELBASE_ / std::pow(v, 2)) * state.lat_dev;
  delta += (a1_ * WHEELBASE_ / v) * state.head_dev;
  delta += WHEELBASE_ * state.curvature;
  return delta;
}


TrajectoryUtils::VehiclePose SimpleTrackingController::lookupVehiclePose() const
{
  static TrajectoryUtils::VehiclePose latest_pose = {{0, 0}, 0};

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(odom_frame_id_, robot_frame_id_, rclcpp::Time(0));
  } catch (tf2::TransformException & ex) {
    auto error_throttle_clock = rclcpp::Clock(RCL_STEADY_TIME);
    RCLCPP_ERROR_THROTTLE(
      this->get_logger(), error_throttle_clock, 5000,
      "Could not get transform from %s to %s.", robot_frame_id_.c_str(), odom_frame_id_.c_str());
  }

  latest_pose.pos = {transform.transform.translation.x, transform.transform.translation.y};
  tf2::Quaternion quaternion; tf2::fromMsg(transform.transform.rotation, quaternion);
  tf2::Matrix3x3(quaternion).getRPY(*(new double), *(new double), latest_pose.yaw);

  return latest_pose;
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleTrackingController>());
  rclcpp::shutdown();
  return 0;
}
