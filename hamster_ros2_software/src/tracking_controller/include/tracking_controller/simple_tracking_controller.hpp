#pragma once

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tracking_controller/point2d.hpp"
#include "tracking_controller/trajectory.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"


class SimpleTrackingController : public rclcpp::Node
{
private:
  double a0_ = -10;
  double a1_ = -10;
  const double WHEELBASE_ = 0.16;


  /* Frame IDs */
  std::string odom_frame_id_;
  std::string robot_frame_id_;

  /** This holds the currently tracked trajectory */
  TrajectoryUtils::Trajectory current_trajectory_;

  /** Publisher which outputs the steering angle and velocity command */
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr command_pub_;

  /** Publisher for tracking state (debug purposes) */
  rclcpp::Publisher<hamster_interfaces::msg::TrackingControllerState>::SharedPtr tracking_state_pub_;

  /** Publisher for trajectory visualization (debug purposes) */
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr traj_visualization_pub_;

  /** Timer for controller updates */
  rclcpp::TimerBase::SharedPtr controller_update_timer_;

  /** Timer to re-transform the trajectory from map into the odom frame from time to time */
  rclcpp::TimerBase::SharedPtr trajectory_tf_timer_;

  /** Timer for controller updates */
  rclcpp::TimerBase::SharedPtr update_timer_;

  /* Buffer and listener for received transforms */
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;

  /** The update function called by timer */
  void updateController();

  /** Usually there are three relevant frames:
   *  - "base_link" (vehicle-fixed), "odom" and "map" (world-fixed)
   *
   *  "odom" is theoretically world-fixed, but no discontinuities of the vehicle position in this frame are allowed.
   *  The odom frame may drift away from the map frame from time to time.
   *  Usually the transform base_link->odom is provided by a state estimation module and should no jump,
   *  while the transform map-->odom is provided through a SLAM module or e.g. the global camera tracking system.
   *
   * The controller needs to works with the odom-->base_link relation as the robot position, since it maked use of the property that the position does not jump within the odom frame.
   * But the trajectory is usually world-fixed in the map frame.
   * Hence it needs to be transformed to the odom frame first, as well as again periodically to correct for potential drift between map and odom.
   * */
  void retransformTrajectory();

  /** Computes the steering angle based on current state */
  double controlLaw(TrajectoryUtils::TrackingState state) const;

  /** Performs tf tree lookup to get current vehicle pose */
  TrajectoryUtils::VehiclePose lookupVehiclePose() const;

  std::string getRobotName() const {return std::string(this->get_namespace()).erase(0, 1);}

public:
  SimpleTrackingController();
};
