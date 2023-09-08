#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include "serialib/serialib.h"

#include "rclcpp/rclcpp.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "hamster_interfaces/msg/velocity_with_header.hpp"

class HamsterSerial : public rclcpp::Node
{
private:
  // Serial settings
  const std::string SERIAL_PORT_ = "/dev/serial0";
  const unsigned int SERIAL_BAUD_RATE_ = 115200;

  // Serial interface library
  serialib serial_;

  // Read thread
  std::thread read_thread_;

  // Command subscription
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr command_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr interlock_sub_;

  ackermann_msgs::msg::AckermannDriveStamped latest_command_msg_;
  std_msgs::msg::Bool latest_interlock_msg_;

  rclcpp::Time latest_controller_received_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  rclcpp::Time latest_interlock_received_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

  float speed_limit_;

  // Publishers
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr voltage_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pid_pub_;
  rclcpp::Publisher<hamster_interfaces::msg::VelocityWithHeader>::SharedPtr velocity_pub_;

  // Timer for read and write tasks
  rclcpp::TimerBase::SharedPtr read_timer_;
  rclcpp::TimerBase::SharedPtr write_timer_;

public:
  HamsterSerial();


  void ReadData();
  void WriteCommand();

  void CommandCallback(const ackermann_msgs::msg::AckermannDriveStamped & msg);
  void TwistCallback(const geometry_msgs::msg::Twist & msg);
  void InterlockCallback(const std_msgs::msg::Bool & msg);

  void ParseData(const std::string & data);

  void ParseVoltage(const std::string & data) const;
  void ParseOdom(const std::string & data);
  void ParsePid(const std::string & data) const;
};
