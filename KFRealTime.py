"""
KF version 2
use dictionary and list to store data
faster than using pandas package
"""
import time

import numpy as np
import pandas as pd


def ode_x_and_p(x, u, p, q):
    """
    ODE equations

    :param x: [x, y, theta, v].T  , all system states
    :param u: [a_z, w_y].T  , 'inputs' of the system
    :param p: covariance matrix of states
    :param q: covariance matrix of state noise
    :return: dot_x
    """

    A = np.array([[0, 0, 0, np.cos(x[2][0])],
                  [0, 0, 0, np.sin(x[2][0])],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [0, -1],
                  [1, 0]])
    dx = A @ x + B @ u
    dp = A @ p + p @ A.T + q

    return dx, dp


def f(x, u, ts, p, q):  # RK-4 integral
    """
    Integrate the states and matrix P,
    according to their ODE
    :param x:
    :param u:
    :param ts:
    :param p:
    :param q:
    :return: x_next, P_next
    """
    k1, kp1 = ode_x_and_p(x, u, p, q)
    k2, kp2 = ode_x_and_p(x + ts / 2 * k1, u, p + ts / 2 * kp1, q)
    k3, kp3 = ode_x_and_p(x + ts / 2 * k2, u, p + ts / 2 * kp2, q)
    k4, kp4 = ode_x_and_p(x + ts * k3, u, p + ts / 2 * kp3, q)

    x_next = x + 1 / 6 * ts * (k1 + 2 * k2 + 2 * k3 + k4)
    p_next = p + 1 / 6 * ts * (kp1 + 2 * kp2 + 2 * kp3 + kp4)

    # # 0 <= theta < 2*pi
    # x_next[2][0] = x_next[2][0] % (2 * np.pi)

    # -pi <= theta < pi
    x_next[2][0] = (x_next[2][0] + np.pi) % (2 * np.pi) - np.pi

    return x_next, p_next


class KFRealTime:
    def __init__(self, q_matrix: np.ndarray, r_vel: np.ndarray, r_tf: np.ndarray, p0_matrix: np.ndarray):
        if q_matrix.shape != (4, 4): raise ValueError("'q_matrix' should be a 4*4 numpy array!")
        if r_vel.shape != (1, 1): raise ValueError("'r_vel' should be a 1*1 numpy array!")
        if r_tf.shape != (3, 3): raise ValueError("'r_tf' should be a 3*3 numpy array!")
        if p0_matrix.shape != (4, 4): raise ValueError("'p_matrix' should be a 4*4 numpy array!")

        self.q_matrix = q_matrix                    # covariance matrix of state/system noise
        self.r_vel = r_vel                          # covariance matrix of velocity measurement noise
        self.r_tf = r_tf                            # covariance matrix of TF measurement noise
        self.p0_matrix = p0_matrix                  # initial covariance matrix of state error

        self.C_vel = np.array([[0., 0., 0., 1.]])   # C_vel output matrix for velocity data
        self.C_tf = np.array([[1., 0., 0., 0.],     # C_tf output matrix for tf  data
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.]])
        # # Info holder
        self.initialized = False
        self.i = None      # id number of the last item in list
        self.dataset = {'seq':         [],     # 0  | int           sorted seq according to time
                        'time':        [],     # 1  | float
                        'sensor':      [],     # 2  | str 'imu' 'vel' 'tf'
                        'data':        [],     # 3  | [(2,1), (1,1), (3,1)] array, respectively data from imu, vel, tf
                        'X':           [],     # 4  | (4,1) array   states updated continually
                        'X_rt':        [],     # 5  | (4,1) array   states in real time  (sent to controller)
                        'P':           [],     # 6  | (4,4) array   covariance matrix in KF
                        't_com':       []}     # 7  | float         computational time

    def __call__(self, t_stamp: float, data_typ: str, data: np.ndarray):
        """

        :param t_stamp:
        :param data_typ: 'imu', 'vel' or 'tf'
        :param data:
        :return:
        """
        self.update_data(t_stamp, data_typ, data)

    def initialization(self):
        pass

    def update_data(self, t_stamp: float, data_typ: str, data):
        t0 = time.perf_counter()
        if self.initialized:
            self.i += 1
            id_1 = self.i
            while t_stamp < self.dataset['time'][id_1 - 1]:
                id_1 -= 1
                if id_1 == 0:
                    break
            self.dataset['seq'].insert(id_1, self.i)            # sorted according to time stamp
            self.dataset['time'].insert(id_1, t_stamp)
            self.dataset['sensor'].insert(id_1, data_typ)
            self.dataset['data'].insert(id_1, data)
            self.dataset['X'].insert(id_1, None)
            self.dataset['X_rt'].insert(id_1, np.array([[np.nan]]*4))
            self.dataset['P'].insert(id_1, None)

            self.update_states(id_1)        # Do Simulation, update 'X' and 'P', store 'X_rt'

            t1 = time.perf_counter()
            self.dataset['t_com'].append(t1 - t0)

        else:   # not yet receive data from TF
            if data_typ == 'tf':
                self.i = 0
                self.dataset['seq'].append(self.i)
                self.dataset['time'].append(t_stamp)
                self.dataset['sensor'].append(data_typ)
                self.dataset['data'].append(data)
                self.dataset['X'].append(np.append(data, [[0.]], axis=0))   # append v=0 at the end of tf data
                self.dataset['X_rt'].append(np.array([[np.nan]] * 4))       # X_rt set to nan
                self.dataset['P'].append(self.p0_matrix)
                self.initialized = True     # once receive data from TF, initialize X

                t1 = time.perf_counter()
                self.dataset['t_com'].append(t1 - t0)

    def update_states(self, id_1):
        """
        :param id_1: # id of the last seq number (id of the newest data 0~inf) and the first point to update
        :return:
        """

        id_0 = id_1 - 1  # sim start from id_n-1, whose state is initial state of this loop
        id_u0 = id_0  # id of U start from id_last - 1 , cuz the last u is useless

        # such id of recent u of newest data until id_u0 = -1, if there is no imu data
        while id_u0 != -1:
            if self.dataset['sensor'][id_u0] == 'imu':
                break
            id_u0 -= 1
        # get u0
        if id_u0 != -1:                                             # There are imu data before
            u = self.dataset['data'][id_u0]
        else:                                                       # There is no imu data yet, use zero u0
            u = np.array([[0.], [0.]])

        # get x0, P0 matrix and ts list
        if id_0 != -1:                                              # There are data before the newest data
            # list_seq = self.dataset['seq'][id_1:]
            list_time = self.dataset['time'][id_0:]
            list_sensor = self.dataset['sensor'][id_1:]
            list_data = self.dataset['data'][id_1:]
            list_ts = np.diff(list_time)

            x = self.dataset['X'][id_0]
            p = self.dataset['P'][id_0]

        else:   # id_0==-1                                          # There is no data before the newest data
            list_seq = self.dataset['seq']
            list_time = self.dataset['time']
            list_sensor = self.dataset['sensor']
            list_data = self.dataset['data']
            list_ts = np.append(0., np.diff(list_time))             # add 0 to the beginning of Ts list

            x = self.dataset['X'][list_seq.index(0)]
            p = self.p0_matrix

        # Simulation
        for (i, ts) in enumerate(list_ts):
            # < Prediction Step >
            x, p = f(x, u, ts, p, self.q_matrix)

            if list_sensor[i] == 'imu':
                u = list_data[i]  # if imu, update u

            # < Correction Step >
            elif list_sensor[i] == 'vel':
                y = list_data[i]
                k = p @ self.C_vel.T @ np.linalg.inv(self.C_vel @ p @ self.C_vel.T + self.r_vel)
                x = x + k @ (y - self.C_vel @ x)            # correction
                p = (np.eye(4) - k @ self.C_vel) @ p

            elif list_sensor[i] == 'tf':
                y = list_data[i]
                k = p @ self.C_tf.T @ np.linalg.inv(self.C_tf @ p @ self.C_tf.T + self.r_tf)
                y_delta = y - self.C_tf @ x
                # -pi <= delta_theta < pi
                y_delta[2][0] = (y_delta[2][0] + np.pi) % (2 * np.pi) - np.pi
                x = x + k @ y_delta                         # correction
                p = (np.eye(4) - k @ self.C_tf) @ p
            # update or store all x and p in the loop
            self.dataset['X'][id_1 + i] = x
            self.dataset['P'][id_1 + i] = p
        # store the real time x with the largest time stamp
        self.dataset['X_rt'][id_1 + i] = x

    @property
    def df(self):
        n = self.i + 1
        columns = ['rt_seq',        # 0
                   'time',          # 1
                   'sensor.type',   # 2
                   'U.a_z',         # 3
                   'U.w_y',         # 4
                   'velocity',      # 5
                   'tf.x',          # 6
                   'tf.y',          # 7
                   'tf.yaw_z',      # 8
                   'x',             # 9
                   'y',             # 10
                   'theta',         # 11
                   'v',             # 12
                   'rt_x',          # 13
                   'rt_y',          # 14
                   'rt_theta',      # 15
                   'rt_v',          # 16
                   'P11',           # 17
                   'P12',           # 18
                   'P13',           # 19
                   'P14',           # 20
                   'P21',           # 21
                   'P22',           # 22
                   'P23',           # 23
                   'P24',           # 24
                   'P31',           # 25
                   'P32',           # 26
                   'P33',           # 27
                   'P34',           # 28
                   'P41',           # 29
                   'P42',           # 30
                   'P43',           # 31
                   'P44',           # 32
                   't_com']         # 33
        df = pd.DataFrame(data=[[np.nan] * 34] * n, columns=columns)

        df.iloc[:, 0] = self.dataset['seq']
        df.iloc[:, 1] = self.dataset['time']
        df.iloc[:, 2] = self.dataset['sensor']
        for i in range(n):
            sensor = self.dataset['sensor'][i]
            if sensor == 'imu':
                df.iloc[i, 3:5] = self.dataset['data'][i].reshape(2)
            elif sensor == 'vel':
                df.iloc[i, 5] = self.dataset['data'][i].reshape(1)
            elif sensor == 'tf':
                df.iloc[i, 6:9] = self.dataset['data'][i].reshape(3)
            df.iloc[i, 9:13] = self.dataset['X'][i].T
            df.iloc[i, 13:17] = self.dataset['X_rt'][i].T
            df.iloc[i, 17:33] = self.dataset['P'][i].reshape(16)

        df.iloc[:, 33] = self.dataset['t_com']
        return df

