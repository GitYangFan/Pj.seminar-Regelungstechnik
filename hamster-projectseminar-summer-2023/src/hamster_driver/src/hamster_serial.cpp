#include "hamster_driver/hamster_serial.hpp"


HamsterSerial::HamsterSerial()
: Node("hamster_serial")
{
  // Declare parameters
  speed_limit_ = std::abs(this->declare_parameter<double>("speed_limit"));

  // Open serial port
  int8_t serial_opening_error = serial_.openDevice(SERIAL_PORT_.c_str(), SERIAL_BAUD_RATE_);
  if (serial_opening_error != 1) {
    RCLCPP_FATAL(
      this->get_logger(), "HamsterSerial: Opening serial device %s failed with code %d",
      SERIAL_PORT_.c_str(), serial_opening_error);
    rclcpp::shutdown();
    exit(1);
  }

  // Create command subscription
  command_sub_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
    "command", 1, std::bind(&HamsterSerial::CommandCallback, this, std::placeholders::_1));
  twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "twist_command", 1, std::bind(&HamsterSerial::TwistCallback, this, std::placeholders::_1));
  interlock_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    "interlock", 1, std::bind(&HamsterSerial::InterlockCallback, this, std::placeholders::_1));

  // Create publishers
  voltage_pub_ = this->create_publisher<std_msgs::msg::Float32>("voltage", 1);
  pid_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("pid", 1);
  velocity_pub_ = this->create_publisher<hamster_interfaces::msg::VelocityWithHeader>("velocity", 1);

  // Create timers
  read_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(1),
    std::bind(&HamsterSerial::ReadData, this));
  write_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&HamsterSerial::WriteCommand, this));
}


void HamsterSerial::ReadData()
{
  // Return if no data received yet
  if (serial_.available() == 0) {
    return;
  }

  const uint16_t MAXLEN_BYTES = 1000;
  const uint16_t TIMEOUT_MS = 1000;

  char buffer[MAXLEN_BYTES + 2];
  int read_success = serial_.readString(buffer, '\n', MAXLEN_BYTES, TIMEOUT_MS);
  if (read_success <= 0) {
    RCLCPP_ERROR(
      this->get_logger(),
      "HamsterSerial: Read serial port failed with code %d", read_success);
    return;
  }

  this->ParseData(std::string(buffer));
}


void HamsterSerial::WriteCommand()
{
  float velocity = latest_command_msg_.drive.speed;
  int16_t steering_angle_deg = latest_command_msg_.drive.steering_angle;

  // Limit velocity
  velocity = std::clamp(velocity, -speed_limit_, speed_limit_);

  // Set command to zero if no message received from controller or interlock for a while
  auto & last_receive_time = std::min(
    latest_controller_received_time_,
    latest_interlock_received_time_);
  auto time_since_last_receive = (this->get_clock()->now() - last_receive_time);
  if (time_since_last_receive > std::chrono::milliseconds(400)) {
    velocity = 0;
    steering_angle_deg = 0;
  }

  // Set to zero commands if latest interlock message was not true
  if (latest_interlock_msg_.data != true) {
    velocity = 0;
    steering_angle_deg = 0;
  }

  // Format commands into string
  static char stringCommand[12];
  snprintf(stringCommand, sizeof(stringCommand), "%05.2f:%03i;\n", velocity, steering_angle_deg);

  // Send command string via serial
  int write_success = serial_.writeString(stringCommand);
  if (write_success != 1) {
    RCLCPP_ERROR(this->get_logger(), "HamsterSerial: Write serial port failed.");
  }
}


void HamsterSerial::CommandCallback(const ackermann_msgs::msg::AckermannDriveStamped & msg)
{
  latest_command_msg_ = msg;
  latest_controller_received_time_ = this->get_clock()->now();

  RCLCPP_WARN_EXPRESSION(
    this->get_logger(), msg.drive.steering_angle_velocity != 0,
    "HamsterSerial: Ignoring 'steering_angle_velocity' in AckermannDrive message.");
  RCLCPP_WARN_EXPRESSION(
    this->get_logger(), msg.drive.acceleration != 0,
    "HamsterSerial: Ignoring 'acceleration' in AckermannDrive message.");
  RCLCPP_WARN_EXPRESSION(
    this->get_logger(), msg.drive.jerk != 0,
    "HamsterSerial: Ignoring 'jerk' in AckermannDrive message.");
}


void HamsterSerial::TwistCallback(const geometry_msgs::msg::Twist & msg)
{
  // Abbreviations
  auto & v = msg.linear.x;
  auto & yawrate = msg.angular.z;
  const double WHEELBASE = 0.16;

  latest_controller_received_time_ = this->get_clock()->now();

  if (std::abs(v) < 0.1) {
    latest_command_msg_.drive.speed = 0;
    latest_command_msg_.drive.steering_angle = 0;
  } else {
    latest_command_msg_.drive.speed = v;
    latest_command_msg_.drive.steering_angle = 180 / 3.14 * std::atan(WHEELBASE * yawrate / v);
  }
}


void HamsterSerial::InterlockCallback(const std_msgs::msg::Bool & msg)
{
  latest_interlock_msg_ = msg;
  latest_interlock_received_time_ = this->get_clock()->now();
}


void HamsterSerial::ParseData(const std::string & data)
{
  const std::string VOLTAGE_PREFIX = "#VOLTAGE:";
  const std::string ODOMETRY_PREFIX = "#ODOM:";
  const std::string IMU_PREFIX = "#IMU:";
  const std::string PID_PREFIX = "#PID:";

  if (data.rfind(VOLTAGE_PREFIX, 0) == 0) {
    ParseVoltage(data.substr(VOLTAGE_PREFIX.length()));
  } else if (data.rfind(ODOMETRY_PREFIX, 0) == 0) {
    ParseOdom(data.substr(ODOMETRY_PREFIX.length()));
  } else if (data.rfind(PID_PREFIX, 0) == 0) {
    ParsePid(data.substr(PID_PREFIX.length()));
  } else {
    RCLCPP_WARN(this->get_logger(), "HamsterSerial: Invalid packet received. Canno parse data.");
  }
}


void HamsterSerial::ParseVoltage(const std::string & data) const
{
  float voltage = std::stof(data);
  auto msg = std_msgs::msg::Float32().set__data(voltage);
  voltage_pub_->publish(msg);
}

// Project seminar
// Velocity type was changed to add time stamp to the message
// Find message under hamster_interfaces/msg

void HamsterSerial::ParseOdom(const std::string & data)
{
  float velocity = -0.001 * std::stof(data);
 // auto msg = std_msgs::msg::Float32().set__data(velocity);
 // velocity_pub_->publish(msg);
  auto msg = hamster_interfaces::msg::VelocityWithHeader();
  msg.header.stamp = this->get_clock()->now(); // Add timestamp using the current node's time
  msg.velocity = velocity;
  velocity_pub_->publish(msg);
}


void HamsterSerial::ParsePid(const std::string & data) const
{
  auto comma_idx = data.find(',');
  std::string x_string = data.substr(0, comma_idx);
  std::string y_string = data.substr(comma_idx + 1);
  float x = std::stof(x_string);
  float y = std::stof(y_string);
  auto msg = geometry_msgs::msg::Vector3().set__x(x).set__y(y);
  pid_pub_->publish(msg);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HamsterSerial>());
  rclcpp::shutdown();
  return 0;
}
