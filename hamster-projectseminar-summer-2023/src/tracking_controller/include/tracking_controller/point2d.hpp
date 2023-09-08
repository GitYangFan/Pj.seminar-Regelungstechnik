#pragma once

#include <cmath>
#include <cstring>
#include "geometry_msgs/msg/point.hpp"


namespace TrajectoryUtils
{

struct Point2d
{
  double x;
  double y;

  Point2d()
  {
    this->x = 0;
    this->y = 0;
  }

  Point2d(const Point2d & p)
  {
    this->x = p.x;
    this->y = p.y;
  }

  Point2d & operator=(const Point2d & p)
  {
    this->x = p.x;
    this->y = p.y;
    return *this;
  }

  Point2d(double x, double y)
  {
    this->x = x;
    this->y = y;
  }

  explicit Point2d(const geometry_msgs::msg::Point & p)
  {
    this->x = p.x;
    this->y = p.y;
  }

  friend inline bool operator==(const Point2d & a, const Point2d & b)
  {
    return (a.x == b.x) && (a.y == b.y);
  }

  friend Point2d operator+(const Point2d & a, const Point2d & b)
  {
    return {a.x + b.x, a.y + b.y};
  }

  friend Point2d operator-(const Point2d & a, const Point2d & b)
  {
    return {a.x - b.x, a.y - b.y};
  }

  friend Point2d operator-(const Point2d & p)
  {
    return {-p.x, -p.y};
  }

  friend Point2d operator*(double a, const Point2d & b)
  {
    return {a * b.x, a * b.y};
  }

  friend Point2d operator*(const Point2d & a, double b)
  {
    return b * a;
  }

  friend double operator*(const Point2d & a, const Point2d & b)
  {
    return a.x * b.x + a.y * b.y;
  }

  static double crossProduct(const Point2d & a, const Point2d & b)
  {
    return a.x * b.y - a.y * b.x;
  }

  double length() const
  {
    return std::sqrt(x * x + y * y);
  }

  double length2() const
  {
    return x * x + y * y;
  }

  double angle() const
  {
    return std::atan2(y, x);
  }

  static Point2d normalized(const Point2d & p)
  {
    double length = p.length();
    return {p.x / length, p.y / length};
  }

  static double distance(const Point2d & a, const Point2d & b)
  {
    Point2d diff = b - a;
    return diff.length();
  }

  static Point2d rotate(const Point2d & p, double angle)
  {
    auto c = std::cos(angle);
    auto s = std::sin(angle);
    return {c * p.x - s * p.y, s * p.x + c * p.y};
  }

  geometry_msgs::msg::Point toGeometryMsg() const
  {
    geometry_msgs::msg::Point p;
    p.x = x;
    p.y = y;
    return p;
  }

  bool isnan() const
  {
    return std::isnan(this->x) || std::isnan(this->y);
  }
};

/** Struct representing position and oriantation in 2D space */
struct VehiclePose
{
  Point2d pos;
  double yaw;
};

}  // namespace TrajectoryUtils
