#include "tracking_controller/trajectory.hpp"


std::tuple<TrajectoryUtils::Point2d, int> TrajectoryUtils::Trajectory::calcClosestPoint(
  TrajectoryUtils::Point2d point) const
{
  if (this->coordinates_.size() < 2) {
    RCLCPP_ERROR(
      rclcpp::get_logger(
        "TrajUtils"), "Cannot calculate closest point to empty trajectory.");
    throw std::out_of_range("Cannot calculate closest point to empty trajectory.");
  }

  int idx_next_point_after_closest_point = 1;
  auto closest_point_so_far = this->coordinates_[0];
  double closest_distance_so_far = 10000;

  // Only loop to the 2nd last point, so that next point is always available and not out of bounds
  for (size_t idx_reference_point = 0; idx_reference_point < this->coordinates_.size() - 1;
    idx_reference_point++)
  {
    // A is current point, B is next point
    auto A = this->coordinates_[idx_reference_point];
    auto B = this->coordinates_[idx_reference_point + 1];

    // This code finds the point on a line segment AB, which is closest to a point P by projecting P onto
    // the line and restricting it to the points that are actually on the line, not its extension to both sides.
    auto AB = B - A;
    auto AP = point - A;
    double t = (AP * AB) / AB.length2();
    t = std::max(0.0, std::min(1.0, t));
    auto projected_point = A + t * AB;

    // Check new distance and save if its lower
    double new_distance = Point2d::distance(projected_point, point);
    if (new_distance <= closest_distance_so_far) {
      closest_distance_so_far = new_distance;
      closest_point_so_far = projected_point;
      idx_next_point_after_closest_point = idx_reference_point + 1;
    }
  }

  return {closest_point_so_far, idx_next_point_after_closest_point};
}


TrajectoryUtils::TrackingState TrajectoryUtils::Trajectory::getTrackingState(
  TrajectoryUtils::VehiclePose vehicle_pose) const
{
  TrackingState state;

  auto [closest_point, idx_next_point] = this->calcClosestPoint(vehicle_pose.pos);
  double trajectory_heading = (coordinates_[idx_next_point] - closest_point).angle();

  state.head_dev = vehicle_pose.yaw - trajectory_heading;
  if (state.head_dev > M_PI) {
    state.head_dev -= 2 * M_PI;
  } else if (state.head_dev < -M_PI) {
    state.head_dev += 2 * M_PI;
  }

  Point2d normal_vec_on_trajectory = Point2d::rotate({1, 0}, trajectory_heading + M_PI / 2);
  state.lat_dev = (vehicle_pose.pos - closest_point) * normal_vec_on_trajectory;

  state.curvature = curvatures_[idx_next_point];
  state.velocity = velocities_[idx_next_point];

  return state;
}


std::tuple<unsigned int, TrajectoryUtils::Trajectory> TrajectoryUtils::Trajectory::getSegment(
  unsigned int idx_current_position, double target_length) const
{
  // The provided idx_current_position is the index of the car position on the full trajectory.
  // The method returns the segmented trajectory, as well as the index of that first point
  // of the extracted segment in the full trajectory.
  unsigned int & index = idx_current_position;

  // Move index back to start behind the car segment behind the car
  double length_to_append_backwards = 2;
  target_length += length_to_append_backwards;
  while ((length_to_append_backwards > 0) && (index > 0)) {
    auto & current_point = this->coordinates_.at(index--);
    auto & previous_point = this->coordinates_.at(index);
    length_to_append_backwards -=
      Point2d::distance(Point2d(current_point), Point2d(previous_point));
  }

  // This will be the returned index of the first point of the extracted segment in the full refline
  unsigned int firstPointOfSegmentIndex = index;

  // Generate segment
  Trajectory segment;
  while ((target_length > 0) && (index + 1 < this->coordinates_.size())) {
    segment.coordinates_.push_back(this->coordinates_.at(index));
    segment.distances_.push_back(this->distances_.at(index));
    segment.headings_.push_back(this->headings_.at(index));
    segment.curvatures_.push_back(this->curvatures_.at(index));
    segment.velocities_.push_back(this->velocities_.at(index));

    auto next_point = this->coordinates_.at(++index);
    target_length -= Point2d::distance(next_point, segment.coordinates_.back());
  }

  segment.frame_id_ = this->frame_id_;
  return {firstPointOfSegmentIndex, segment};
}


bool TrajectoryUtils::Trajectory::loadFromCsv(
  std::string filename, Trajectory & trajectory,
  std::string trajectory_frame_id)
{
  auto logger = rclcpp::get_logger("TrajUtils");
  RCLCPP_INFO(logger, "Trying to load trajectory from file %s.", filename.c_str());

  std::vector<std::string> row;
  std::string line, word;
  std::fstream file(filename, std::ios::in);

  if (!file.is_open()) {
    RCLCPP_ERROR(logger, "Error opening file: %s", filename.c_str());
    return false;
  }

  try {
    while (std::getline(file, line)) {
      std::stringstream str(line);
      row.clear();
      while (std::getline(str, word, ';')) {
        row.push_back(word);
      }

      // Skip empty lines and first line with titles
      if (row.size() == 0 || row.at(0) == "" || row.at(0)[0] == '#') {
        continue;
      }

      if (row.size() != 7) {
        throw std::runtime_error("Invalid number of columns. Line was: " + line);
      }

      trajectory.distances_.push_back(std::stod(row.at(0)));
      trajectory.coordinates_.push_back({std::stod(row.at(1)), std::stod(row.at(2))});
      trajectory.headings_.push_back(std::stod(row.at(3)) + M_PI_2);
      trajectory.curvatures_.push_back(std::stod(row.at(4)));
      trajectory.velocities_.push_back(std::stod(row.at(5)));
    }
    file.close();

    trajectory.frame_id_ = trajectory_frame_id;

    RCLCPP_INFO(
      logger, "Loaded trajectory with a length of %zu points.",
      trajectory.coordinates_.size());
    return true;
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger, "Error parsing file %s: %s", filename.c_str(), e.what());
    file.close();
    return false;
  }
}


TrajectoryUtils::Trajectory TrajectoryUtils::Trajectory::transformTo(
  tf2_ros::Buffer * tf_buffer_ptr,
  std::string target_frame_id) const
{
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_ptr->lookupTransform(target_frame_id, this->frame_id_, rclcpp::Time(0));
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TrajUtils"),
      "Could not transform trajectory from %s to %s.",
      this->frame_id_.c_str(), target_frame_id.c_str());
  }

  Point2d translation = {transform.transform.translation.x, transform.transform.translation.y};
  double roll, pitch, yaw;

  tf2::Quaternion quaternion; tf2::fromMsg(transform.transform.rotation, quaternion);
  tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);


  Trajectory transformed_trajectory = Trajectory(*this);
  transformed_trajectory.frame_id_ = target_frame_id;

  for (size_t idx = 0; idx < this->coordinates_.size(); ++idx) {
    // Transform point coordinate
    transformed_trajectory.coordinates_[idx] =
      Point2d::rotate(this->coordinates_[idx], yaw) + translation;
    transformed_trajectory.headings_[idx] = this->headings_[idx] - yaw;
  }

  return transformed_trajectory;
}


visualization_msgs::msg::Marker TrajectoryUtils::Trajectory::toMarker(rclcpp::Time stamp) const
{
  auto m = visualization_msgs::msg::Marker();
  m.header.frame_id = frame_id_;
  m.header.stamp = stamp;

  m.ns = "trajectory";
  m.type = visualization_msgs::msg::Marker::LINE_STRIP;
  m.action = visualization_msgs::msg::Marker::MODIFY;

  std_msgs::msg::ColorRGBA color;
  color.r = 1.0; color.g = 0.4; color.b = 0; color.a = 1.0;
  m.color = color;
  m.scale.x = 0.05;
  m.scale.y = 0.05;
  m.scale.z = 0.05;
  m.lifetime = rclcpp::Duration(0, 0);
  m.frame_locked = true;

  for (auto & c : coordinates_) {
    geometry_msgs::msg::Point p;
    p.x = c.x;
    p.y = c.y;
    m.points.push_back(p);
  }

  return m;
}
