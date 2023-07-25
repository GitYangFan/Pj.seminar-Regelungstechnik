#pragma once

#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class OnlyOptitrackLocalization : public rclcpp::Node
{
private:
  /* Frame IDs */
  std::string odom_frame_id_;
  std::string robot_frame_id_;
  std::string map_frame_id_;
  const std::string OPTITRACK_FRAME_ID_ = "world";

  /* Buffer and listener for received transforms */
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  /** Get robot name, which is the namespace without the prepending slash */
  std::string getRobotName() const {return std::string(this->get_namespace()).erase(0, 1);}

public:
  OnlyOptitrackLocalization();
};
