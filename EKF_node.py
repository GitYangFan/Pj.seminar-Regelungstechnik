import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
from hamster_interfaces.msg import VelocityWithHeader
from KFRealTime import KFRealTime

from euler_from_quaternion import euler_from_quaternion

# initialization of global data
data = np.zeros((1, 7))  # [time_stamp, imu_a_z, imu_w_y, velocity, tf_x, tf_y, tf_yaw]
# initialization of previous ekf results
ekf_pre = np.zeros((1, 5))  # [time_stamp, position_x, position_y, yaw_angle, velocity]
# initialization of previous imu data
imu_pre = [0, 0]    # [a_z, w_y]
# clk = None  # initialize the global clock
length = 20  # the length of data cache
NaN = np.nan

# initialization of EKF
Q = np.diag([1, 1, 1, 1])  # covariance matrix for state noise
R_vel = np.array([[0.1]])  # covariance matrix for velocity measurement noise
R_tf = np.diag([0.1, 0.1, 0.1])  # covariance matrix for TF measurement noise
P = np.diag([1, 1, 1, 1])  # initial covariance matrix for current state

sim = KFRealTime(Q, R_vel, R_tf, P)
sim.initialization()


def ode_x(x, u):
    """
    ODE equations

    :param x: [x, y, theta, v].T  , all system states
    :param u: [a_z, w_y].T  , 'inputs' of the system
    :param p: covariance matrix of states
    :param q: covariance matrix of state noise
    :return: dot_x
    """
    # print('x:',x)
    A = np.array([[0, 0, 0, np.cos(x[2])],
                  [0, 0, 0, np.sin(x[2])],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, -1],
                  [1, 0]])
    dx = A @ x + B @ u

    return dx


def f(x, u, ts):  # RK-4 integral for x
    """
    Integrate the states and matrix P,
    according to their ODE
    :param x:
    :param u:
    :param ts:
    :param p:
    :param q:
    :return: x_next
    """
    k1 = ode_x(x, u)
    k2 = ode_x(x + ts / 2 * k1, u)
    k3 = ode_x(x + ts / 2 * k2, u)
    k4 = ode_x(x + ts * k3, u)

    x_next = x + 1 / 6 * ts * (k1 + 2 * k2 + 2 * k3 + k4)

    x_next[2] = (x_next[2] + np.pi) % (2 * np.pi) - np.pi

    return x_next


class EKF_node(Node):

    def __init__(self):
        super().__init__('EKF_node')
        self.imu_subscription = self.create_subscription(
            Imu,
            '/hamster2/imu',
            self.imu_callback,
            1)
        self.tf_subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            1)
        self.vel_subscription = self.create_subscription(
            VelocityWithHeader,
            '/hamster2/velocity',
            self.vel_callback,
            1)
        # self.clock_subscription = self.create_subscription(
        #     Clock,
        #     '/clock',
        #     self.clock_callback,
        #     1)
        self.publisher_ = self.create_publisher(Float64MultiArray, 'EKF', 1)
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.ekf_callback)
        self.i = 0
        self.imu_subscription  # prevent unused variable warning
        self.tf_subscription
        self.vel_subscription
        # self.clock_subscription

    def imu_callback(self, msg):
        global data
        # self.get_logger().info('imu info: "%s"' % msg.angular_velocity.x)
        imu_data = np.array([(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec), msg.linear_acceleration.z,
                             msg.angular_velocity.y, NaN, NaN, NaN, NaN])
        data = np.concatenate((data, imu_data[np.newaxis, :]), axis=0)
        if data.shape[0] > length:
            data = np.delete(data, 0, axis=0)
            # print('---------------------------------------------')
            # print('data:', data)

    def tf_callback(self, msg):
        global data
        # self.get_logger().info('tf info: "%s"' % msg.transforms[0].transform.translation.x)
        xyzw = [msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w]
        _, _, yaw_z = euler_from_quaternion(xyzw[0], xyzw[1], xyzw[2], xyzw[3])
        tf_data = np.array([(msg.transforms[0].header.stamp.sec + 1e-9 * msg.transforms[0].header.stamp.nanosec), NaN,
                            NaN, NaN, msg.transforms[0].transform.translation.x,
                            msg.transforms[0].transform.translation.y,
                            yaw_z])
        data = np.concatenate((data, tf_data[np.newaxis, :]), axis=0)
        if data.shape[0] > length:
            data = np.delete(data, 0, axis=0)
            # print('---------------------------------------------')
            # print('data:', data)

    def vel_callback(self, msg):
        global data
        vel = msg.velocity
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        # print('---------------------------------------------')
        # print('vel: ', vel)
        # print('---------------------------------------------')
        # print('stamp: ', stamp)
        vel_data = np.array([stamp, NaN, NaN, vel, NaN, NaN, NaN])

        data = np.concatenate((data, vel_data[np.newaxis, :]), axis=0)
        if data.shape[0] > length:
            data = np.delete(data, 0, axis=0)
            # print('---------------------------------------------')
            # print('data:', data)

    # def clock_callback(self, msg):
    #     global clk
    #     print('clock msg:', msg)
    #     # clk = msg
    #     print('---------------------------------------------')
    #     print('clk: ', clk)

    def ekf_callback(self):
        global data
        global ekf_pre
        global imu_pre

        msg = Float64MultiArray()

        if data[-1, 0] != 0:
            # the switch of data type
            s_imu = np.isnan(data[-1, 1])
            s_vel = np.isnan(data[-1, 3])
            s_tf = np.isnan(data[-1, 4])

            if not s_imu:
                data_typ = 'imu'
                t_stamp = data[-1, 0]  # header time stamp
                value = data[-1, 1:3].reshape(2, 1)
                imu_pre = [data[-1, 1], data[-1, 2]]    # [a_z, w_y]
                # print('value:', value)
            elif not s_vel:
                data_typ = 'vel'
                t_stamp = data[-1, 0]  # header time stamp
                value = data[-1, 3].reshape(1, 1)
                # print('value:', value)
            elif not s_tf:
                data_typ = 'tf'
                t_stamp = data[-1, 0]  # header time stamp
                value = data[-1, 4:7].reshape(3, 1)
                # print('value:', value)
            else:
                data_typ = 'imu'
                t_stamp = data[-2, 0]  # header time stamp
                value = data[-2, 1:3].reshape(2, 1)

            sim(t_stamp, data_typ, value)

            # msg.data = [0.1, 0.2 + self.i]
            # print('---------------------------------------------')
            # result = sim.df.iloc[-1, 9:11].values.tolist()
            # print('ekf results:', result)
            # msg.data = result

            # print(sim.dataset)
            # print('length: ', len(sim.dataset['time']))
            # if len(sim.dataset['time']) != 0:
            #     msg.data = sim.dataset['time'][-1]

            """ It is plan A. We use integral to get the current status x """

            if len(sim.dataset['X_rt']) != 0:
                # print('X_rt: ', [sim.dataset['X_rt'][-1][0][0], sim.dataset['X_rt'][-1][1][0],sim.dataset['X_rt'][-1][2][0],sim.dataset['X_rt'][-1][3][0]])
                # print(sim.dataset['X_rt'][-1])
                if abs(sim.dataset['X_rt'][-1][0][0])>5 or abs(sim.dataset['X_rt'][-1][1][0])>5:
                    print('---------------------------------------------')
                    print ("WARNING!!!!!!!!")
                    print("x: ", sim.dataset['X_rt'][-1][0][0])
                    print("y: ", sim.dataset['X_rt'][-1][1][0])
                ekf_result = [sim.dataset['X_rt'][-1][0][0], sim.dataset['X_rt'][-1][1][0],sim.dataset['X_rt'][-1][2][0],sim.dataset['X_rt'][-1][3][0]] #sim.dataset['X_rt'][0]

                # calculate the time difference between latest sensor and current time
                current_time = 1e-9 * self.get_clock().now().nanoseconds
                # print('t_stamp:', t_stamp)
                ts = abs(current_time - t_stamp)
                if ts < 0.0001 or np.isnan(ekf_result[0]):  # if the time difference is smaller than 0.0001s, the error can be ignored.
                    clk = t_stamp
                    x_current = ekf_result
                    msg.data = x_current
                else:
                    clk = current_time
                    print('---------------------------------------------')
                    print("current clk:", clk)
                    print("last sensor clk:", t_stamp)
                    x = np.array(ekf_result).T
                    u = np.array(imu_pre).T
                    x_current = f(x, u, ts)
                    # print('x_current:', x_current.tolist())
                    msg.data = x_current.tolist()
            self.publisher_.publish(msg)
            self.i += 1


            """ It is plan B. We use the interpolation instead of integral to get the current status x. 
            
            if len(sim.dataset['X_rt']) != 0:
                # print('X_rt: ', [sim.dataset['X_rt'][-1][0][0], sim.dataset['X_rt'][-1][1][0],sim.dataset['X_rt'][-1][2][0],sim.dataset['X_rt'][-1][3][0]])
                # print(sim.dataset['X_rt'][-1])
                if abs(sim.dataset['X_rt'][-1][0][0]) > 5 or abs(sim.dataset['X_rt'][-1][1][0]) > 5:
                    print('---------------------------------------------')
                    print("WARNING!!!!!!!!")
                    print("x: ", sim.dataset['X_rt'][-1][0][0])
                    print("y: ", sim.dataset['X_rt'][-1][1][0])
                ekf_new = np.array(
                    [clk, sim.dataset['X_rt'][-1][0][0], sim.dataset['X_rt'][-1][1][0], sim.dataset['X_rt'][-1][2][0],
                     sim.dataset['X_rt'][-1][3][0]])  # sim.dataset['X_rt'][0]
                ekf_pre = np.vstack((ekf_pre, ekf_new))
                ekf_interpolate = ekf_new  # initial value of ekf result at current time
                print('ekf_result:', ekf_new)
                coefficients = np.zeros((length, 2))
                if ekf_pre.shape[0] > length:
                    ekf_pre = ekf_pre[1:]
                    clk = ekf_new[0]
                    for i in range(4):
                        coefficients[i] = np.polyfit(ekf_pre[:, 0], ekf_pre[:, i + 1], 1)
                        ekf_interpolate[i + 1] = np.polyval(coefficients[i], clk)
                msg.data = ekf_interpolate.tolist()
                print('ekf_interpolate:', ekf_interpolate)
            self.publisher_.publish(msg)
            # self.get_logger().info('Publishing ekf results: "%s"' % msg.data)
            self.i += 1
            
            """


def main(args=None):
    rclpy.init(args=args)

    # -------------test the ekf node-----------
    ekf_node = EKF_node()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
