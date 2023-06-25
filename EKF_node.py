import rclpy
from rclpy.node import Node
from simulation import *

from EKF.msg import EKF_estimate        # the message type of EKF
from IMUInfo.msg import IMUInfo         # the message type of imu
from geometry_msgs.msg import TransformStamped      # the message type of tf


data = np.empty((0, 6), float)

def simulation_kf_realtime(data):
    # ==========================================Simulation================================================
    # calculate the yaw angle of tf
    wxyz = data[:, 32:36]  # wxyz = [w, x, y, z]
    print(wxyz.shape[0])
    yaw_z = np.zeros((1, wxyz.shape[0]))  # shape: (1, n)
    for i in range(wxyz.shape[0]):
        _, _, yaw_z[0][i] = euler_from_quaternion(wxyz[i][1], wxyz[i][2], wxyz[i][3], wxyz[i][0])

    u_k = np.append(data[:, 13].reshape(1, -1), data[:, 2].reshape(1, -1), axis=0)  # a_z, w_y, with nan, shape: (2, n)
    y_k_vel = - data[:, 30] / 1000  # unit of data is mm/s and data is inverse
    y_k_tf = np.concatenate((data[:, 36].reshape(1, -1), data[:, 37].reshape(1, -1), yaw_z),
                            axis=0)  # [x, y, yaw_angle].T , shape: (3, n)
    Ts = np.diff(data[:, 0])  # get time steps
    # Switch, if the data are NaN, True. To determine prediction or correction in KF
    s_imu = np.isnan(data[:, 29])
    s_vel = np.isnan(data[:, 30])
    s_tf = np.isnan(data[:, 31])

    # Initial
    x0 = [2.669, -3.347, 0, 0]
    Q = np.diag([1, 1, 1, 1])
    R_v = np.array([[0.1]])
    R_t = np.diag([0.1, 0.1, 0.1])
    P0 = np.diag([10, 10, 10, 10])

    x_all = sim_all(x0, u_k, Ts, y_k_vel, y_k_tf, s_imu, s_vel, s_tf, Q, R_v, R_t, P0)
    return x_all


class EKF_node(Node):

    def __init__(self):
        super().__init__('EKF_publisher')
        self.publisher_ = self.create_publisher(EKF_estimate, 'estimationResults', 10)
        self.imu_subscription_ = self.create_subscription(
            IMUInfo,
            '/hamster2/imu',
            self.imu_callback,
            10
        )
        self.tf_subscription_ = self.create_subscription(
            TransformStamped,
            'tf_topic',
            self.tf_callback,
            10
        )
        self.imu_subscription_
        self.tf_subscription_
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.ekf_callback)
        self.get_logger().info('EKF node initialized')

    def imu_callback(self, msg_imu):
        global data
        imu_data = np.array([msg_imu.linear_acceleration.x, msg_imu.linear_acceleration.y, msg_imu.linear_acceleration.z,
                  msg_imu.angular_velocity.x, msg_imu.angular_velocity.y, msg_imu.angular_velocity.z])
        data = np.vstack((data, imu_data))      # Stack the latest imu data after global variable data
        result = simulation_kf_realtime(data)   # get the simulation results
        ekf_msg = EKF_estimate()
        ekf_msg.data = result[:, -1].tolist()    # only publish the simulation results for current position
        self.publisher_.publish(ekf_msg)
        self.get_logger().info('Published EKF result: "%s"' % ekf_msg.data)

    def tf_callback(self, msg_tf):
        global data
        tf_data = np.array(
            [msg_tf.transform.translation.x, msg_tf.transform.translation.y, msg_tf.transform.translation.z,
             msg_tf.transform.rotation.x, msg_tf.transform.rotation.y, msg_tf.transform.rotation.z])
        data = np.vstack((data, tf_data))       # Stack the latest tf data after global variable data
        result = simulation_kf_realtime(data)   # get the simulation results
        ekf_msg = EKF_estimate()
        ekf_msg.data = result[-1, :].tolist()
        self.publisher_.publish(ekf_msg)
        self.get_logger().info('Published KF results: %s' % ekf_msg.data)


def main(args=None):
    rclpy.init(args=args)
    EKF_publisher = EKF_node()
    rclpy.spin(EKF_publisher)
    # EKF_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
