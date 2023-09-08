#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "tracking_controller/point2d.hpp"
#include "hamster_interfaces/msg/tracking_controller_state.hpp"


namespace TrajectoryUtils
{
struct TrackingState
{
  double lat_dev;
  double head_dev;
  double velocity;
  double curvature;

  hamster_interfaces::msg::TrackingControllerState toMsg(
    rclcpp::Time now,
    std::string frame_id) const
  {
    auto msg = hamster_interfaces::msg::TrackingControllerState();
    msg.header.stamp = now;
    msg.header.frame_id = frame_id;
    msg.lat_dev = lat_dev;
    msg.head_dev = head_dev;
    msg.velocity = velocity;
    msg.curvature = curvature;
    return msg;
  }
};


class Trajectory
{
public:
  /** Calculates the lateral and heading eviation of a pose to the closest point on this trajectory. */
  TrackingState getTrackingState(VehiclePose vehicle_pose) const;

  /** Return a shorter segment starting close to the specified position */
  std::tuple<unsigned int, Trajectory> getSegment(
    unsigned int idx_current_position,
    double target_length) const;

  /** This method reads a trajectory from filename into trajectory object
       *  and returns true if it was successful. */
  static bool loadFromCsv(
    std::string filename, Trajectory & trajectory,
    std::string trajectory_frame_id);

  /** Returns a copy of the trajectory, transformed to the given frame */
  Trajectory transformTo(tf2_ros::Buffer * tf_buffer_ptr, std::string target_frame_id) const;

  /** Frame id */
  std::string getFrameID() const {return frame_id_;}

  /** Convert to marker for visualization id */
  visualization_msgs::msg::Marker toMarker(rclcpp::Time stamp) const;

private:
  std::string frame_id_;
  std::vector<double> distances_;
  std::vector<Point2d> coordinates_;
  std::vector<double> headings_;
  std::vector<double> curvatures_;
  std::vector<double> velocities_;


  /** This method finds the point on the (linearly interpolated) trajectory, that is closest to the given point.
       * It also returns the index of the next point on the line, after the found closest point. */
  std::tuple<Point2d, int> calcClosestPoint(Point2d point) const;
};

}  // namespace TrajectoryUtils
