#!/usr/bin/env python3 


import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
from hamster_interfaces.msg import VelocityWithHeader
from KFRealTime import KFRealTime
import pandas as pd

from euler_from_quaternion import euler_from_quaternion

# initialization of global data
data = np.zeros((1, 7))  # [time_stamp, imu_a_z, imu_w_y, velocity, tf_x, tf_y, tf_yaw]
clk = None   # initialize the global clock
length = 10     # the length of data cache
NaN = np.nan

# initialization of EKF
Q = np.diag([1, 1, 1, 1])         # covariance matrix for state noise
R_vel = np.array([[0.1]])        # covariance matrix for velocity measurement noise
R_tf = np.diag([0.1, 0.1, 0.1])    # covariance matrix for TF measurement noise
P = np.diag([1, 1, 1, 1])         # initial covariance matrix for current state

sim = KFRealTime(Q, R_vel, R_tf, P)
sim.initialization()


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
        self.publisher_ = self.create_publisher(Float64MultiArray, '/hamster2/EKF', 1)
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.ekf_callback)
        self.i = 0
        self.imu_subscription  # prevent unused variable warning
        self.tf_subscription
        self.vel_subscription
        # self.clock_subscription

    def imu_callback(self, msg):
        global data
        o_R_e = np.genfromtxt("/home/baho/hamster_ros2_software/src/hamster_driver/scripts/o_R_e.csv", delimiter=",")

        # Correction of IMU Data with a rotation matrix
        # self.get_logger().info('imu info: "%s"' % msg.angular_velocity.x)
        acc_vector = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        # print("Acceleration before correction:" , acc_vector)
        acc_vector = acc_vector @ o_R_e
        # print("Acceleration after correction:" , acc_vector)
        vel_vector = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        vel_vector = vel_vector @ o_R_e
        # print("--------------------------------------------")
        imu_data = np.array([(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec), acc_vector[2],
                    vel_vector[1], NaN, NaN, NaN, NaN])
        data = np.concatenate((data, imu_data[np.newaxis, :]), axis=0)
        if data.shape[0] > length:
            data = np.delete(data, 0, axis=0)
            # print('---------------------------------------------')
            # print('data:', data)

    def tf_callback(self, msg):
        global data
        print(self.get_clock().now().nanoseconds) #1690636269990265038 1690636316122178390 1690636644763948980
        # self.get_logger().info('tf info: "%s"' % msg.transforms[0].transform.translation.x)
        xyzw = [msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w]
        _, _, yaw_z = euler_from_quaternion(xyzw[0], xyzw[1], xyzw[2], xyzw[3])
        tf_data = np.array([(msg.transforms[0].header.stamp.sec + 1e-9 * msg.transforms[0].header.stamp.nanosec), NaN,
                   NaN, NaN, msg.transforms[0].transform.translation.x, msg.transforms[0].transform.translation.y,
                   yaw_z])
        data = np.concatenate((data, tf_data[np.newaxis, :]), axis=0)
        if data.shape[0] > length:
            data = np.delete(data, 0, axis=0)
            # print('---------------------------------------------')
            # print('data:', data)

    def vel_callback(self, msg):    # no time stamp ??
        global data
        vel = msg.velocity
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        print('---------------------------------------------')
        print('vel: ', vel)
        print('---------------------------------------------')
        print('stamp: ', stamp)
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
                # print('value:', value)
            elif not s_vel:
                data_typ = 'vel'
                t_stamp = data[-1, 0]   # header time stamp
                value = data[-1, 3].reshape(1, 1)
                # print('value:', value)
            elif not s_tf:
                data_typ = 'tf'
                t_stamp = data[-1, 0]   # header time stamp
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

            if len(sim.dataset['X_rt']) != 0:
                # print('X_rt: ', [sim.dataset['X_rt'][-1][0][0], sim.dataset['X_rt'][-1][1][0],sim.dataset['X_rt'][-1][2][0],sim.dataset['X_rt'][-1][3][0]])
                # print(sim.dataset['X_rt'][-1])
                if abs(sim.dataset['X_rt'][-1][0][0])>5 or abs(sim.dataset['X_rt'][-1][1][0])>5:
                    print('---------------------------------------------')
                    print ("WARNING!!!!!!!!")
                    print("x: ", sim.dataset['X_rt'][-1][0][0])
                    print("y: ", sim.dataset['X_rt'][-1][1][0])
                msg.data = [sim.dataset['X_rt'][-1][0][0], sim.dataset['X_rt'][-1][1][0],sim.dataset['X_rt'][-1][2][0],sim.dataset['X_rt'][-1][3][0]] #sim.dataset['X_rt'][0]
            self.publisher_.publish(msg)
            # self.get_logger().info('Publishing ekf results: "%s"' % msg.data)
            self.i += 1


def main(args=None):
    rclpy.init(args=args)

    # imu_subscriber = ImuSubscriber()
    # tf_subcriber = tfSubscriber()
    # ekf_subcriber = EKF_Subscriber()

    # rclpy.spin(imu_subscriber)
    # rclpy.spin(tf_subcriber)
    # rclpy.spin(ekf_subcriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # imu_subscriber.destroy_node()
    # tf_subcriber.destroy_node()
    # ekf_subcriber.destroy_node()

    # -------------test the ekf publisher-----------
    # ekf_publisher = EKF_Publisher()
    # rclpy.spin(ekf_publisher)
    # ekf_publisher.destroy_node()

    # -------------test the ekf node-----------
    ekf_node = EKF_node()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()



"""
class ImuSubscriber(Node):

    def __init__(self):
        super().__init__('Imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/hamster2/imu',
            self.imu_callback,
            10)
        self.subscription  # prevent unused variable warning

    def imu_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.angular_velocity.x)


class tfSubscriber(Node):

    def __init__(self):
        super().__init__('tf_subscriber')
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)
        self.subscription  # prevent unused variable warning

    def tf_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.transforms[0].transform.translation)


class EKF_Publisher(Node):

    def __init__(self):
        super().__init__('ekf_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'EKF', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.ekf_callback)
        self.i = 0

    def ekf_callback(self):
        msg = Float64MultiArray()
        msg.data = [0.1, 0.2 + self.i]
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


class EKF_Subscriber(Node):

    def __init__(self):
        super().__init__('EKF_subscriber')
        self.imu_subscription = self.create_subscription(
            Imu,
            '/hamster2/imu',
            self.imu_callback,
            10)
        self.tf_subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)
        self.imu_subscription  # prevent unused variable warning
        self.tf_subscription

    def imu_callback(self, msg):
        self.get_logger().info('imu info: "%s"' % msg.angular_velocity.x)

    def tf_callback(self, msg):
        self.get_logger().info('tf info: "%s"' % msg.transforms[0].transform.translation.x)
"""
